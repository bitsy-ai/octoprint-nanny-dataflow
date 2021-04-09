from typing import Tuple, Iterable, Optional, Any, NamedTuple
import logging
from datetime import datetime
import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import apache_beam as beam
from apache_beam.dataframe.transforms import DataframeTransform

from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    NestedWindowedHealthTrend,
    DeviceCalibration,
    RenderVideoMessage,
    Metadata,
    NestedWindowedHealthTrend,
    CATEGORY_INDEX,
    AlertMessageType,
)

logger = logging.getLogger(__name__)


def health_score_trend_polynomial_v1(
    df: pd.DataFrame, degree=1
) -> Tuple[pd.DataFrame, np.polynomial.polynomial.Polynomial]:
    """
    Takes a pandas DataFrame of WindowedHealthRecords and returns a polynommial fit to degree
    """
    xy = (
        df[df["health_weight"] > 0]
        .groupby(["ts"])["health_score"]
        .max()
        .add(
            df[df["health_weight"] < 0].groupby(["ts"])["health_score"].min(),
            fill_value=0,
        )
        .cumsum()
    )
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
            client_version=element.client_version,
            session=element.session,
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
        keyed_elements=Tuple[str, Iterable[NestedTelemetryEvent]],
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
    ) -> Iterable[Tuple[str, NestedWindowedHealthTrend]]:
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
        try:
            cumsum, trend = health_score_trend_polynomial_v1(
                df, degree=self.polyfit_degree
            )
        except ValueError as e:
            logger.error(
                {
                    "error": e,
                    "data": df,
                    "msg": "Fatal error in SortWindowedHealthDataframe transform",
                }
            )
            return
        metadata = df.iloc[0]["metadata"]
        logger.info(
            f"SortWindowedHealthDataframe emitting NestedWindowedHealthTrend window_start={window_start} window_end={window_end}"
        )
        yield (
            key,
            NestedWindowedHealthTrend(
                session=key,
                poly_coef=np.array(trend.coef),
                poly_domain=np.array(trend.domain),
                poly_roots=np.array(trend.roots()),
                poly_degree=trend.degree(),
                cumsum=cumsum.to_numpy(),
                metadata=metadata,
                health_score=df["health_score"].to_numpy(),
                health_weight=df["health_weight"].to_numpy(),
                detection_class=df["detection_class"].to_numpy(),
                detection_score=df["detection_score"].to_numpy(),
            ),
        )


class CreateVideoRenderMessage(beam.DoFn):
    def __init__(
        self, in_base_path, out_base_path, cdn_base_path, cdn_upload_path, bucket
    ):
        self.in_base_path = in_base_path
        self.out_base_path = out_base_path
        self.cdn_base_path = cdn_base_path
        self.cdn_upload_path = cdn_upload_path
        self.bucket = bucket

    def process(
        self,
        element=Tuple[Any, Iterable[NestedWindowedHealthTrend]],
        window=beam.DoFn.WindowParam,
        pane_info=beam.DoFn.PaneInfoParam,
    ) -> Iterable[bytes]:
        key, values = element
        datestamp = datetime.now().strftime("%Y/%m/%d")

        gcs_prefix_in = os.path.join(self.in_base_path, key)
        gcs_prefix_out = os.path.join(self.out_base_path, key, "annotated_video.mp4")

        suffix = os.path.join(
            key,
            datestamp,
            "annotated_video.mp4",
        )
        cdn_prefix_out = os.path.join(self.cdn_base_path, self.cdn_upload_path, suffix)

        cdn_suffix = os.path.join(self.cdn_upload_path, suffix)

        # publish video rendering message
        if pane_info.is_last:
            msg = RenderVideoMessage(
                session=key,
                metadata=values[0].metadata,
                alert_type=AlertMessageType.SESSION_DONE,
                gcs_prefix_in=gcs_prefix_in,
                gcs_prefix_out=gcs_prefix_out,
                cdn_prefix_out=cdn_prefix_out,
                cdn_suffix=cdn_suffix,
                bucket=self.bucket,
            ).to_bytes()
            yield msg
        # @TODO analyze production distribution and write alert behavior for session panes
        else:
            msg = RenderVideoMessage(
                session=key,
                metadata=values[0].metadata,
                alert_type=AlertMessageType.FAILURE,
                gcs_prefix_in=gcs_prefix_in,
                gcs_prefix_out=gcs_prefix_out,
                cdn_prefix_out=cdn_prefix_out,
                cdn_suffix=cdn_suffix,
                bucket=self.bucket,
            ).to_bytes()
            yield msg


class MonitorHealthStateful(beam.DoFn):
    def __init__(self, pubsub_topic, failure_threshold=2, quiet=True):
        self.failure_threshold = failure_threshold
        self.pubsub_topic = pubsub_topic
        self.quiet = quiet

    FAILURES = beam.transforms.userstate.CombiningValueStateSpec("failures", sum)

    def process(
        self,
        element: Tuple[Any, Iterable[NestedWindowedHealthTrend]],
        pane_info=beam.DoFn.PaneInfoParam,
        window=beam.DoFn.WindowParam,
        failures=beam.DoFn.StateParam(FAILURES),
    ) -> Iterable[Tuple[str, Iterable[NestedWindowedHealthTrend]]]:

        key, values = element
        current_failures = failures.read()
        failures.add(1)

        window_start = int(window.start)
        window_end = int(window.end)
        logger.info(
            f"MonitorHealthStateful received n={len(element)} elements window_start={window_start} window_end={window_end}"
        )
        logger.info(
            f"Pane fired with pane_info={pane_info} window={window} failures={current_failures}"
        )
        # import pdb; pdb.set_trace()
        yield key, values
        # key, value = element

        # slope, intercept = value.trend.coef
        # if slope < 0:
        #     failures.add(1)

        # current_failures = failures.read()
        # value = value.with_failure_count(current_failures)
        # # @TODO analyze production distribution and write alert behavior

        # # @TODO write pyarrow schema instead of inferring it here
        # # table = pa.Table.from_pydict(element.to_dict())

        # yield key, value

        # logger.info(f"Pane fired with pane_info={pane_info} window={window} failures={current_failures}")

        # if current_failures > self.failure_threshold:
        #     logger.warning(
        #         f"FAILURE DETECTED session={value.session} current_failurest={current_failures}"
        #     )
        # # if this is the last window pane in session, begin video rendering
        # if pane_info.is_last:
        #     logger.info(f"Last pane fired in pane_info={pane_info} window={window} failures={current_failures}")

        #     # Exception: PubSub I/O is only available in streaming mode (use the --streaming flag). [while running 'Stateful health score threshold monitor']
        # pending_alert = RenderVideoMessage(
        #     metadata=value.metadata,
        #     session=key,
        # )
        # [pending_alert] | beam.io.WriteToPubSub(self.pubsub_topic)
