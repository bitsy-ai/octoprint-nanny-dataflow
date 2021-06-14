from typing import Tuple, Iterable, Optional, Any, NamedTuple, NewType
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
    Box,
)
from print_nanny_dataflow.coders.types import (
    MonitoringImageT,
    AnnotatedMonitoringImageT,
    BoxAnnotationsT,
    DeviceCalibrationT,
    BoxT,
)
from print_nanny_dataflow.coders.types import get_health_weight
from print_nanny_dataflow.metrics.area_of_interest import merge_filtered_annotations

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
@beam.typehints.with_output_types(MonitoringImageT)
class ParseMonitoringImage(beam.DoFn):
    def process(self, element: bytes) -> Iterable[MonitoringImageT]:
        parsed = MonitoringImage()
        parsed.ParseFromString(element)
        yield parsed


@beam.typehints.with_input_types(MonitoringImageT)
@beam.typehints.with_output_types(AnnotatedMonitoringImageT)
class PredictBoundingBoxes(beam.DoFn):
    def __init__(self, gcs_model_path: str):

        self.gcs_model_path = gcs_model_path

    # @time_distribution("print_health", "tflite_predict_bounding_boxes_duration")
    def process(self, element: MonitoringImageT) -> Iterable[AnnotatedMonitoringImageT]:
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
        num_detections = int(np.squeeze(num_detections, axis=0))

        detection_boxes = [Box(xy=b) for b in box_data]
        health_weights = map(get_health_weight, class_data)

        annotations = BoxAnnotations(
            num_detections=num_detections,
            detection_scores=score_data,
            detection_boxes=detection_boxes,
            detection_classes=class_data,
            health_weights=health_weights,
        )
        yield AnnotatedMonitoringImage(
            monitoring_image=element, annotations_all=annotations
        )

    # def process(
    #     self, element: MonitoringImageT
    # ) -> Iterable[AnnotatedMonitoringImageT]:
    #     yield self.process_timed(element)


@beam.typehints.with_input_types(Tuple[str, Iterable[AnnotatedMonitoringImage]])
@beam.typehints.with_output_types(AnnotatedMonitoringImage)
class FilterBoxAnnotations(beam.DoFn):
    def __init__(
        self,
        calibration_base_path: str,
        calibration_filename: str = "calibration.json",
    ):
        self.calibration_base_path = calibration_base_path
        self.calibration_filename = calibration_filename

    def load_calibration(
        self, element: AnnotatedMonitoringImageT
    ) -> Optional[DeviceCalibrationT]:
        gcs_client = beam.io.gcp.gcsio.GcsIO()
        device_id = element.monitoring_image.metadata.octoprint_device_id
        device_calibration_path = os.path.join(
            self.calibration_base_path, str(device_id), self.calibration_filename
        )
        if gcs_client.exists(device_calibration_path):
            with gcs_client.open(device_calibration_path, "r") as f:
                logger.info(
                    f"Loading device calibration from {device_calibration_path}"
                )
                calibration_json = json.load(f)

            return DeviceCalibration(**calibration_json)
        return None

    def process(
        self,
        keyed_elements=Tuple[str, Iterable[AnnotatedMonitoringImage]],
    ) -> Iterable[AnnotatedMonitoringImage]:
        session, elements = keyed_elements
        calibration = self.load_calibration(elements[0])
        logger.info("Loaded calibration")
        return elements | beam.Map(
            lambda x: merge_filtered_annotations(x, calibration=calibration)
        )
