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
    try:
        trend = np.polynomial.polynomial.Polynomial.fit(xy.index, xy, degree)
    except ValueError as e:
        logger.error(e)
        logger.info(df)
        return
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

    def __init__(self, warmup=3, polyfit_degree=1):
        self.polyfit_degree = 1
        self.warmup = warmup

    def process(
        self,
        keyed_elements: Tuple[
            Any, Iterable[WindowedHealthRecord]
        ] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Tuple[str, WindowedHealthDataFrames]]:
        key, windowed_health_records = keyed_elements

        window_start = int(window.start)
        window_end = int(window.end)
        df = (
            pd.DataFrame(data=windowed_health_records)
            .sort_values("ts")
            .set_index(["ts"])
        )
        if len(df.index) < self.warmup:
            return
        cumsum, trend = health_score_trend_polynomial_v1(df, degree=self.polyfit_degree)

        # import pdb; pdb.set_trace()
        metadata = df.iloc[0]["metadata"]
        record_df = df.drop(columns=["metadata"], axis=1)
        yield (
            key,
            WindowedHealthDataFrames(
                session=key,
                trend=trend,
                record_df=record_df,
                cumsum=cumsum,
                metadata=metadata,
            ),
        )


class MonitorHealthStateful(beam.DoFn):
    def __init__(self, pubsub_topic, failure_threshold=2, quiet=True):
        self.failure_threshold = failure_threshold
        self.pubsub_topic = pubsub_topic
        self.quiet = quiet

    FAILURES = beam.transforms.userstate.CombiningValueStateSpec("failures", sum)

    def process(
        self,
        element: Tuple[Any, WindowedHealthDataFrames],
        pane_info=beam.DoFn.PaneInfoParam,
        failures=beam.DoFn.StateParam(FAILURES),
    ) -> Iterable[Tuple[str, WindowedHealthDataFrames]]:
        key, value = element

        slope, intercept = value.trend
        if slope < 0:
            failures.add(1)

        current_failures = failures.read()
        value = value.with_failure_count(current_failures)
        # @TODO analyze production distribution and write alert behavior

        # @TODO write pyarrow schema instead of inferring it here
        # table = pa.Table.from_pydict(element.to_dict())

        logger.info(f"Current WindowedHealthDataFrames {value}")
        yield key, value

        # if this is the last window pane in session, begin video rendering
        if current_failures > self.failure_threshold:
            logger.warning(
                f"FAILURE DETECTED session={value.session} current_failurest={current_failures}"
            )
        if pane_info.is_last:
            logger.info(f"Last pane fired in pane_info={pane_info}")

            # Exception: PubSub I/O is only available in streaming mode (use the --streaming flag). [while running 'Stateful health score threshold monitor']
            # pending_alert = PendingAlert(
            #     metadata=value.metadata,
            #     session=key,
            # )
            # [pending_alert] | beam.io.WriteToPubSub(self.pubsub_topic)
