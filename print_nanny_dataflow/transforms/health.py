from typing import Tuple, Iterable, Optional, Any, NamedTuple
import logging

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import apache_beam as beam
from apache_beam.dataframe.transforms import DataframeTransform

from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    WindowedHealthDataFrames,
    DeviceCalibration,
    PendingAlert,
    Metadata,
)

logger = logging.getLogger(__name__)

# @todo load dynamically from active experiment
CATEGORY_INDEX = {
    0: {"name": "background", "id": 0, "health_weight": 0},
    1: {"name": "nozzle", "id": 1, "health_weight": 0},
    2: {"name": "adhesion", "id": 2, "health_weight": -0.5},
    3: {"name": "spaghetti", "id": 3, "health_weight": -0.5},
    4: {"name": "print", "id": 4, "health_weight": 1},
    5: {"name": "raftt", "id": 5, "health_weight": 1},
}


def health_score_trend_polynomial_v1(
    df: pd.DataFrame, degree=1
) -> Tuple[pd.DataFrame, np.polynomial.polynomial.Polynomial]:
    """
    Takes a pandas DataFrame of WindowedHealthRecords and returns a polynormial fit to degree
    """
    xy = (
        df[df["health_weight"] > 0]
        .groupby(["ts"])["health_score"]
        .max()
        .add(
            df[df["health_weight"] < 0].groupby(["ts"])["health_score"].min(),
            fill_value=0,
        )
    )

    logger.info(f"Calculating polyfit with degree={degree} on df: \n {xy}")
    trend = np.polynomial.polynomial.Polynomial.fit(xy.index, xy, degree)
    return xy, trend


def predict_bounding_boxes(element: NestedTelemetryEvent, model_path: str):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
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
    return NestedTelemetryEvent(**defaults)


class ExplodeWindowedHealthRecord(beam.DoFn):
    def process(
        self,
        element: NestedTelemetryEvent,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[WindowedHealthRecord]:
        window_start = int(window.start)
        window_end = int(window.end)

        metadata = Metadata(
            session=element.session,
            client_version=element.client_version,
            user_id=element.user_id,
            device_id=element.device_id,
            device_cloudiot_id=element.device_cloudiot_id,
            window_start=window_start,
            window_end=window_end,
        )
        return [
            WindowedHealthRecord(
                metadata=metadata,
                ts=element.ts,
                session=element.session,
                detection_score=element.detection_scores[i],
                detection_class=element.detection_classes[i],
                health_weight=CATEGORY_INDEX[element.detection_classes[i]][
                    "health_weight"
                ],
                health_score=CATEGORY_INDEX[element.detection_classes[i]][
                    "health_weight"
                ]
                * element.detection_scores[i],
            )
            for i in range(0, element.num_detections)
        ]


class FilterAreaOfInterest(beam.DoFn):
    def __init__(
        self,
        calibration_base_path: str,
        score_threshold: float = 0.5,
        calibration_filename: str = "calibration.json",
    ):
        self.calibration_base_path = calibration_base_path
        self.score_threshold = score_threshold
        self.calibration_filename = calibration_filename

    def load_calibration(
        self, element: NestedTelemetryEvent
    ) -> Optional[DeviceCalibration]:
        gcs_client = beam.io.gcp.gcsio.GcsIO()
        device_id = element.device_id
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

    def process(
        self,
        keyed_elements=beam.DoFn.ElementParam,
        key=beam.DoFn.KeyParam,
    ) -> Iterable[NestedTelemetryEvent]:
        session, elements = keyed_elements

        calibration = self.load_calibration(elements[0])
        if calibration:
            return elements | beam.Map(lambda event: calibration.filter_event(event))
        else:
            return elements


class SortWindowedHealthDataframe(beam.DoFn):
    """
    Optional Dataframe checkpoint/write for debugging and analysis
    @todo this uses pandas.DataFrame as a first pass but beam provides a Dataframe API
    https://beam.apache.org/documentation/dsls/dataframes/overview/
    """

    def __init__(self, polyfit_degree=1):
        self.polyfit_degree = 1

    def process(
        self,
        keyed_elements: Tuple[
            Any, Iterable[WindowedHealthRecord]
        ] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Tuple[Any, WindowedHealthDataFrames]:
        key, windowed_health_records = keyed_elements

        window_start = int(window.start)
        window_end = int(window.end)
        df = (
            pd.DataFrame(data=windowed_health_records)
            .sort_values("ts")
            .set_index(["ts"])
        )

        # yield
        # n_frames = len(df.index.unique())
        # window_start = int(window.start)
        # window_end = int(window.end)
        # if n_frames <= self.warmup:
        #     logger.warning(
        #         f"Ignoring CalcHealthScoreTrend called with n_frames={n_frames} warmup={self.warmup} session={session} window=({window_start}_{window_end})"
        #     )
        #     return

        cumsum_df, trend = health_score_trend_polynomial_v1(
            df, degree=self.polyfit_degree
        )
        metadata = df.iloc[0]["metadata"]
        record_df = df.drop(columns=["metadata"], axis=1)
        yield key, WindowedHealthDataFrames(
            session=key,
            trend=trend,
            record_df=record_df,
            cumsum_df=cumsum_df,
            metadata=metadata
        )
        # yield
        # should_alert = self.should_alert(session, trend)
        # logger.info(f"should_alert={should_alert} for trend={trend}")
        # if should_alert:
        #     file_pattern = os.path.join(self.parquet_sink, session, "*")
        #     sample_event = windowed_health_records[0]

        #     pending_alert = PendingAlert(
        #         session=session,
        #         client_version=sample_element.client_version,
        #         user_id=sample_element.user_id,
        #         device_id=sample_element.device_id,
        #         device_cloudiot_id=sample_element.device_cloudiot_id,
        #         window_start=sample_element.window_start,
        #         window_end=sample_element.window_end,
        #         file_pattern=file_pattern,
        #     )
        #     yield pending_alert.to_bytes()


class MonitorHealthStateful(beam.DoFn):
    def __init__(self, pubsub_topic, failure_threshold=2, quiet=True):
        self.failure_threshold = failure_threshold
        self.pubsub_topic = pubsub_topic
        self.quiet = quiet

    FAILURES = beam.transforms.userstate.CombiningValueStateSpec("failures", sum)

    def process(
        self,
        element: WindowedHealthDataFrames,
        pane_info=beam.DoFn.PaneInfoParam,
        failures=beam.DoFn.StateParam(FAILURES),
    ) -> Iterable[WindowedHealthDataFrames]:
        key, value = element

        slope, intercept = element.trend
        if slope < 0:
            failures.add(1)

        current_failures = failures.read()
        element = element.with_failure_count(current_failures)
        # @TODO analyze production distribution and write alert behavior

        # @TODO write pyarrow schema instead of inferring it here
        # table = pa.Table.from_pydict(element.to_dict())

        yield key, element

        # if this is the last window pane in session, begin video rendering
        if pane_info.is_last:
            pending_alert = PendingAlert(
                metadata=element.metadata,
                session=key,
            )
            pending_alert | beam.WriteToPubSub(self.pubsub_topic)
