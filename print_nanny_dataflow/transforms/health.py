from typing import Tuple, Iterable, Optional, Any, NamedTuple
import logging

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import apache_beam as beam

from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    PendingAlert,
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
) -> np.polynomial.polynomial.Polynomial:
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
    return trend


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
        keyed_element: Tuple[Any, NestedTelemetryEvent],
        window=beam.DoFn.WindowParam,
    ) -> Iterable[WindowedHealthRecord]:
        key, event = keyed_element
        window_start = int(window.start)
        window_end = int(window.end)

        return [
            WindowedHealthRecord(
                ts=event.ts,
                session=event.session,
                client_version=event.client_version,
                user_id=event.user_id,
                device_id=event.device_id,
                device_cloudiot_id=event.device_cloudiot_id,
                detection_score=event.detection_scores[i],
                detection_class=event.detection_classes[i],
                window_start=window_start,
                window_end=window_end,
                health_weight=CATEGORY_INDEX[event.detection_classes[i]][
                    "health_weight"
                ],
                health_score=CATEGORY_INDEX[event.detection_classes[i]]["health_weight"]
                * event.detection_scores[i],
            )
            for i in range(0, event.num_detections)
        ]


class SortedHealthCumsum(beam.DoFn):
    def process(
        self,
        keyed_elements: Tuple[
            Any, Iterable[WindowedHealthRecord]
        ] = beam.DoFn.ElementParam,
    ):
        key, elements = keyed_elements
        yield elements | DataframeTransform(
            lambda df: df.sort_values("ts").groupby(["session", "ts"]).cumsum()
        )


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
        self, event: NestedTelemetryEvent
    ) -> Optional[DeviceCalibration]:
        gcs_client = beam.io.gcp.gcsio.GcsIO()
        device_id = event.device_id
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
        keyed_elements: Tuple[
            Any, Iterable[NestedTelemetryEvent]
        ] = beam.DoFn.ElementParam,
        key=beam.DoFn.KeyParam,
    ) -> Iterable[NestedTelemetryEvent]:
        session, elements = keyed_elements

        calibration = self.load_calibration(elements[0])

        if calibration:
            yield beam.Map(lambda event: calibration.filter_event(event))
        else:
            yield elements


# class WindowedHealthDataframe(beam.DoFn):
#     """
#         Optional Dataframe checkpoint/write for debugging and analysis
#         @todo this uses pandas.DataFrame as a first pass but beam provides a Dataframe API
#         https://beam.apache.org/documentation/dsls/dataframes/overview/
#     """
#     def __init__(
#         self,
#         base_path,
#         warmup: int=20,
#     ):
#         self.base_path = base_path
#         self.warmup = warmup

#     def process(
#         self,
#         keyed_elements: Tuple[Any, Iterable[WindowedHealthRecord]] = beam.DoFn.ElementParam,
#         window=beam.DoFn.WindowParam,
#     ) -> Iterable[bytes]:
#         session, windowed_health_records = keyed_elements

#         window_start = int(window.start)
#         window_end = int(window.end)
#         output_path = os.path.join(
#             self.checkpoint_sink, session, f"{window_start}_{window_end}.parquet"
#         )
#         df = (
#             pd.DataFrame(data=windowed_health_records)
#             .sort_values("ts")
#             .set_index(["ts"])
#         )
#         df.to_parquet(output_path, engine="pyarrow")

#         yield
#         n_frames = len(df.index.unique())
#         window_start = int(window.start)
#         window_end = int(window.end)
#         if n_frames <= self.warmup:
#             logger.warning(
#                 f"Ignoring CalcHealthScoreTrend called with n_frames={n_frames} warmup={self.warmup} session={session} window=({window_start}_{window_end})"
#             )
#             return

#         trend = health_score_trend_polynomial_v1(df, degree=self.polyfit_degree)

#         should_alert = self.should_alert(session, trend)
#         logger.info(f"should_alert={should_alert} for trend={trend}")
#         if should_alert:
#             file_pattern = os.path.join(self.parquet_sink, session, "*")
#             sample_event = windowed_health_records[0]

#             pending_alert = PendingAlert(
#                 session=session,
#                 client_version=sample_event.client_version,
#                 user_id=sample_event.user_id,
#                 device_id=sample_event.device_id,
#                 device_cloudiot_id=sample_event.device_cloudiot_id,
#                 window_start=sample_event.window_start,
#                 window_end=sample_event.window_end,
#                 file_pattern=file_pattern,
#             )
#             yield pending_alert.to_bytes()


class WindowedHealthScore(beam.DoFn):
    def __init__(
        self,
        checkpoint_sink,
        parquet_sink,
        render_video_topic,
        window_size,
        window_period,
        output_path=None,
        health_threshold=3,
        polyfit_degree=1,
        warmup=20,
    ):
        self.output_path
        self.render_video_topic = render_video_topic
        self.window_size = window_size
        self.window_period = window_period
        self.warmup = warmup
        self.api_url = api_url
        self.api_token = api_token

    async def should_alert_async(self, session: str) -> bool:
        rest_client = RestAPIClient(api_token=self.api_token, api_url=self.api_url)

        print_session = await rest_client.get_print_session(
            print_session=session,
        )
        return print_session.supress_alerts is False

    def should_alert(
        self, session: str, trend: np.polynomial.polynomial.Polynomial
    ) -> bool:
        slope, intercept = tuple(trend)
        if slope < 0:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.should_alert_async(session))
        return False

    def process(
        self,
        keyed_elements: Tuple[Any, Iterable[NamedTuple]] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[bytes]:
        session, windowed_health_records = keyed_elements

        window_start = int(window.start)
        window_end = int(window.end)
        output_path = os.path.join(
            self.checkpoint_sink, session, f"{window_start}_{window_end}.parquet"
        )
        df = (
            pd.DataFrame(data=windowed_health_records)
            .sort_values("ts")
            .set_index(["ts"])
        )
        df.to_parquet(output_path, engine="pyarrow")
        n_frames = len(df.index.unique())
        window_start = int(window.start)
        window_end = int(window.end)
        if n_frames <= self.warmup:
            logger.warning(
                f"Ignoring CalcHealthScoreTrend called with n_frames={n_frames} warmup={self.warmup} session={session} window=({window_start}_{window_end})"
            )
            return

        trend = health_score_trend_polynomial_v1(df, degree=self.polyfit_degree)

        should_alert = self.should_alert(session, trend)
        logger.info(f"should_alert={should_alert} for trend={trend}")
        if should_alert:
            file_pattern = os.path.join(self.parquet_sink, session, "*")
            sample_event = windowed_health_records[0]

            pending_alert = PendingAlert(
                session=session,
                client_version=sample_event.client_version,
                user_id=sample_event.user_id,
                device_id=sample_event.device_id,
                device_cloudiot_id=sample_event.device_cloudiot_id,
                window_start=sample_event.window_start,
                window_end=sample_event.window_end,
                file_pattern=file_pattern,
            )
            yield pending_alert.to_bytes()
