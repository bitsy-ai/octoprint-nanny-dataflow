from typing import Tuple, Iterable, Optional, Any, NamedTuple
import logging
from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import apache_beam as beam

from print_nanny_dataflow.metrics import time_distribution

from apache_beam.io.gcp import gcsio

from print_nanny_client.protobuf.monitoring_pb2 import (
    MonitoringImage,
    AnnotatedMonitoringImage,
    BoxAnnotations,
    DeviceCalibration,
)
from print_nanny_dataflow.coders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    Metadata,
    NestedWindowedHealthTrend,
    CATEGORY_INDEX,
)
from print_nanny_dataflow.metrics.area_of_interest import filter_area_of_interest


logger = logging.getLogger(__name__)

# def health_score_trend_polynomial_v1(
#     df: pd.DataFrame, degree=1
# ) -> Tuple[pd.DataFrame, np.polynomial.polynomial.Polynomial]:
#     """
#     Takes a pandas DataFrame of WindowedHealthRecords and returns a polynommial fit to degree
#     """
#     xy = (
#         df[df["health_weight"] > 0]
#         .groupby(["ts"])["health_score"]
#         .max()
#         .add(
#             df[df["health_weight"] < 0].groupby(["ts"])["health_score"].min(),
#             fill_value=0,
#         )
#         .cumsum()
#     )
#     trend = np.polynomial.polynomial.Polynomial.fit(xy.index, xy, degree)
#     return xy, trend


@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(MonitoringImage)
class ParseMonitoringImage(beam.DoFn):
    def process(self, element: bytes) -> Iterable[MonitoringImage]:
        parsed = MonitoringImage().ParseFromString(element)
        yield beam.window.TimestampedValue(parsed, parsed.metadata.ts)


@beam.typehints.with_input_types(MonitoringImage)
@beam.typehints.with_output_types(AnnotatedMonitoringImage)
class PredictBoundingBoxes(beam.DoFn):
    def __init__(self, gcs_model_path: str):

        self.gcs_model_path = gcs_model_path

    @time_distribution("print_health", "predict_bounding_boxes_elapsed")
    def process_timed(self, element: MonitoringImage) -> AnnotatedMonitoringImage:
        gcs = gcsio.GcsIO()
        with gcs.open(self.gcs_model_path) as f:
            tflite_interpreter = tf.lite.Interpreter(model_content=f.read())

        tflite_interpreter.allocate_tensors()
        output_details = tflite_interpreter.get_output_details()

        tflite_interpreter.invoke()

        box_data = tflite_interpreter.get_tensor(output_details[0]["index"])

        class_data = tflite_interpreter.get_tensor(output_details[1]["index"])
        score_data = tflite_interpreter.get_tensor(output_details[2]["index"])
        num_detections = tflite_interpreter.get_tensor(output_details[3]["index"])

        class_data = np.squeeze(class_data, axis=0).astype(np.int64) + 1
        box_data = np.squeeze(box_data, axis=0)
        score_data = np.squeeze(score_data, axis=0)
        num_detections = np.squeeze(num_detections, axis=0)

        ymin, xmin, ymax, xmax = box_data.T

        params = dict(
            detection_scores=score_data,
            num_detections=int(num_detections),
            detection_classes=class_data,
            boxes_ymin=ymin,
            boxes_xmin=xmin,
            boxes_ymax=ymax,
            boxes_xmax=xmax,
        )
        defaults = element.to_dict()
        defaults.update(params)
        detection_boxes = [BoxAnnotations(*b) for b in box_data]
        health_weights = [CATEGORY_INDEX[i]["health_weight"] for i in class_data]
        annotations = BoxAnnotations(
            num_detections=num_detections,
            detection_scores=score_data,
            detection_boxes=detection_boxes,
            detection_classes=class_data,
            health_weights=health_weights,
        )
        return AnnotatedMonitoringImage(
            monitoring_image=element, annotations_all=annotations
        )

    def process(self, element: MonitoringImage) -> Iterable[AnnotatedMonitoringImage]:
        yield self.process_timed(element)
