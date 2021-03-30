#!/usr/bin/env python

from __future__ import absolute_import

import pandas as pd
from datetime import datetime
import aiohttp
import argparse
import logging
import io
import os
import json
import logging
import asyncio
import tarfile
import tempfile

import numpy as np
import tensorflow as tf
import apache_beam as beam
from typing import List, Tuple, Any, Iterable, Generator, Coroutine
from apache_beam import window
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import PIL
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_serving.apis import predict_pb2
from apache_beam.dataframe.convert import to_dataframe, to_pcollection

from apache_beam.transforms import trigger

from encoders.tfrecord_example import ExampleProtoEncoder
from encoders.types import NestedTelemetryEvent, WindowedHealthRecord, DeviceCalibration
from models.health_score import health_score_trend_polynormial_v1
from utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from clients.rest import RestAPIClient
import pyarrow as pa
import PIL

logger = logging.getLogger(__name__)

# @todo load from labels dict
CATEGORY_INDEX = {
    0: {"name": "background", "id": 0},
    1: {"name": "nozzle", "id": 1},
    2: {"name": "adhesion", "id": 2},
    3: {"name": "spaghetti", "id": 3},
    4: {"name": "print", "id": 4},
    5: {"name": "raftt", "id": 5},
}

HEALTH_WEIGHTS = {1: 0, 2: -0.5, 3: -0.5, 4: 1, 5: 0}


class WriteBatchedTFRecords(beam.DoFn):
    """write one file per window/key"""

    def __init__(self, outdir, schema):
        self.outdir = outdir
        self.schema = schema

    def process(self, batched_elements):
        key, elements = batched_elements
        coder = ExampleProtoEncoder(self.schema)
        ts = int(datetime.now().timestamp())
        output = os.path.join(self.outdir, key, str(ts))
        logger.info(f"Writing {output} with coder {coder}")
        yield (
            elements
            | beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=output,
                num_shards=1,
                shard_name_template="",
                file_name_suffix=".tfrecords.gz",
                coder=coder,
            )
        )


class WriteBatchedParquet(beam.DoFn):
    def __init__(self, parquet_base_path: str, schema: pa.Schema, batch_size: int):
        self.parquet_base_path = parquet_base_path
        self.batch_size = batch_size
        self.schema = schema

    def process(self, batched_elements):

        session, elements = batched_elements
        output_path = os.path.join(
            self.parquet_base_path, session, str(int(datetime.now().timestamp()))
        )

        yield (
            e.to_dict()
            for e in elements
            | beam.io.parquetio.WriteToParquet(output_path, self.schema)
        )


def write_batched_parquet(batched_elements, parquet_base_path, schema):
    session, elements = batched_elements
    output_path = os.path.join(
        parquet_base_path, session, str(int(datetime.now().timestamp()))
    )

    elements = beam.Create([e.to_dict() for e in elements])
    return elements | beam.io.parquetio.WriteToParquet(output_path, schema)


async def download_active_experiment_model(tmp_dir=".tmp/", model_artifact_id=1):

    tmp_artifacts_tarball = os.path.join(tmp_dir, "artifacts.tar.gz")
    rest_client = RestAPIClient(api_token=args.api_token, api_url=args.api_url)

    model_artifacts = await rest_client.get_model_artifact(model_artifact_id)

    async with aiohttp.ClientSession() as session:
        logger.info(f"Downloading model artfiact tarball")
        async with session.get(model_artifacts.artifacts) as res:
            artifacts_gzipped = await res.read()
            with open(tmp_artifacts_tarball, "wb+") as f:
                f.write(artifacts_gzipped)
            logger.info(f"Finished writing {tmp_artifacts_tarball}")
    with tarfile.open(tmp_artifacts_tarball, "r:gz") as tar:
        tar.extractall(tmp_dir)
    logger.info(f"Finished extracting {tmp_artifacts_tarball}")


class WriteHealthCheckpoint(beam.DoFn):
    """
    Writes a SlidingWindow checkpoint to gcs
    Emits all record paths within window size
    """

    def __init__(self, checkpoint_sink, window_size, window_period):
        self.checkpoint_sink = checkpoint_sink
        self.window_size = window_size
        self.window_period = window_period
        self.window_lookback = window_size // window_period

    def process(
        self,
        keyed_elements: Tuple[Any, Iterable[WindowedHealthRecord]],
        window=beam.DoFn.WindowParam,
    ):

        session, health_records = keyed_elements

        base_path = os.path.join(self.checkpoint_sink, session)

        window_start = int(window.start)
        window_end = int(window.end)
        output_path = os.path.join(
            base_path,
            f"{window_start}_{window_end}.parquet",
        )
        df = pd.concat([e.to_dataframe() for e in health_records]).sort_values("ts")
        df.to_parquet(output_path, engine="pyarrow")
        logger.info(f"Wrote {output_path}")

        gcs_client = beam.io.gcp.gcsio.GcsIO()

        past_checkpoints = list(gcs_client.list_prefix(base_path).keys())
        if len(past_checkpoints) > self.window_lookback:
            # get most recent windows from selection
            past_checkpoints = past_checkpoints[-self.window_lookback :]

        return past_checkpoints


class RenderVideoTriggerAlert(beam.DoFn):
    def __init__(
        self,
        parquet_sink,
        video_upload_path,
        api_url,
        api_token,
        max_batches=3,
        category_index=CATEGORY_INDEX,
        score_threshold=0.5,
        max_boxes_to_draw=10,
    ):
        self.parquet_sink = parquet_sink
        self.max_batches = max_batches
        self.api_url = api_url
        self.api_token = api_token
        self.category_index = category_index
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.video_upload_path = video_upload_path

    async def trigger_alert_async(self, session: str):
        rest_client = RestAPIClient(api_token=self.api_token, api_url=self.api_url)

        res = await rest_client.create_defect_alert(
            print_session=session,
        )

        logger.info(f"create_defect_alert res={res}")

    def trigger_alert(self, session: str):
        logger.warning(f"Sending alert for session={session}")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self.trigger_alert_async(session))

    def annotate_image(self, event: NestedTelemetryEvent):
        image_np = np.array(PIL.Image.open(io.BytesIO(event.image_data)))
        if event.calibration is None:
            return visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes,
                event.detection_classes,
                event.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
            )
        else:
            detection_boundary_mask = self.calibration["mask"]
            ignored_mask = np.invert(detection_boundary_mask)
            return visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes,
                event.detection_classes,
                event.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
                detection_boundary_mask=detection_boundary_mask,
                detection_box_ignored=ignored_mask,
            )

    def write_video(self, df: pd.DataFrame, session: str) -> str:
        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        output_path = os.path.join(
            self.video_upload_path, session, f"{min_ts}_{max_ts}.mp4"
        )
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        with gcs_client.open(output_path, "wb+") as f:
            writer = imageio.get_writer(f, mode="I")
            for i, value in df["annotated_image"].iteritems():
                writer.append_data(value)
            writer.close()

    def process(self, session: str):

        base_path = os.path.join(self.parquet_sink, session)
        gcs_client = beam.io.gcp.gcsio.GcsIO()
        batched_records = list(gcs_client.list_prefix(base_path).keys())
        if len(batched_records) > self.max_batches:
            # get most recent windows from selection
            batched_records = batched_records[-self.max_batches :]

        # convert pcollection to Beam DataFrame API/DSL: https://beam.apache.org/blog/dataframe-api-preview-available/
        df = to_dataframe(
            beam.Create(batched_records)
            | f"Read last {self.max_batches} batched records"
            >> beam.io.parquetio.ReadAllFromParquet().with_output_types(
                NestedTelemetryEvent
            )
            | f"Filter area of interest and detections above threshold"
            >> beam.ParDo(
                FilterDetections(
                    args.calibration_base_path,
                    score_threshold=self.score_threshold,
                    output_calibration=True,
                )
            )
        )
        df.sort_values("ts", inplace=True)
        df["annotated_image"] = df.apply(self.annotate_image)
        yield df.agg(lambda x: self.write_video(x, session))


class AsWindowedHealthRecord(beam.DoFn):
    def process(self, event: NestedTelemetryEvent, window=beam.DoFn.WindowParam):
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
                health_multiplier=HEALTH_WEIGHTS[event.detection_classes[i]],
                health_score=HEALTH_WEIGHTS[event.detection_classes[i]]
                * event.detection_scores[i],
            )
            for i in range(0, event.num_detections)
        ]


def absolute_log(df, log_fn=np.log2):
    """
    Get the log of absolute value and multiply by original sign
    """
    return log_fn(np.abs(df))


class CalcHealthScoreTrend(beam.DoFn):
    def __init__(
        self,
        checkpoint_sink,
        window_size,
        window_period,
        api_url,
        api_token,
        health_threshold=3,
        polyfit_degree=1,
        warmup=3,
    ):
        self.health_threshold = health_threshold
        self.checkpoint_sink = checkpoint_sink
        self.polyfit_degree = polyfit_degree
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
            # @todo investigate event loop policy used by beam python SDK (if any) https://docs.python.org/3/library/asyncio-policy.html
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.should_alert_async(session))

    def process(
        self,
        keyed_elements: Tuple[str, Iterable[WindowedHealthRecord]],
        window=beam.DoFn.WindowParam,
    ) -> Iterable[str]:
        session, windowed_health_records = keyed_elements

        df = (
            pd.DataFrame(data=windowed_health_records)
            .sort_values("ts")
            .set_index(["ts", "detection_class"])
        )
        n_windows = len(df["window_start"].unique())
        window_start = int(window.start)
        window_end = int(window.end)
        if n_windows <= self.warmup:
            logger.warning(
                f"Ignoring CalcHealthScoreTrend called with n_windows={n_windows} warmup={self.warmup} session={session} window=({window_start}_{window_end})"
            )
            return

        trend = health_score_trend_polynormial_v1(df, degree=self.polyfit_degree)

        should_alert = self.should_alert(session, trend)
        logger.info(f"should_alert={should_alert} for trend={trend}")
        if should_alert:
            yield session


def predict_bounding_boxes(element, model_path):
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


class FilterDetections(beam.DoFn):
    def __init__(
        self,
        calibration_base_path: str,
        score_threshold: float = 0.5,
        calibration_filename: str = "calibration.json",
        output_calibration=False,
    ):
        self.score_threshold = 0.5
        self.calibration_base_path = calibration_base_path
        self.score_threshold = score_threshold
        self.calibration_filename = calibration_filename
        self.output_calibration = output_calibration

    def process(self, event: NestedTelemetryEvent) -> Iterable[NestedTelemetryEvent]:
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        device_id = event.device_id
        device_calibration_path = os.path.join(
            self.calibration_base_path, str(device_id), self.calibration_filename
        )

        event = event.min_score_filter(score_threshold=self.score_threshold)

        calibration_json = None
        if gcs_client.exists(device_calibration_path):
            with gcs_client.open(device_calibration_path, "r") as f:
                logger.info(
                    f"Loading device calibration from {device_calibration_path}"
                )
                calibration_json = json.load(f)

            coordinates = calibration_json["coordinates"]
            event = NestedTelemetryEvent.calibration_filter(event, coordinates)
        if event.num_detections >= 1:
            if self.output_calibration is True:
                if calibration_json:
                    calibration = DeviceCalibration(**calibration_json)
                else:
                    calibration = None
                yield NestedTelemetryEvent(calibration=calibration ** event.to_dict())
            else:
                yield event


def run_pipeline(args, pipeline_args):
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    topic_path = os.path.join("projects", args.project, "topics", args.topic)

    # download model tarball
    if args.runner == "DataflowRunner":
        asyncio.get_event_loop.run_until_complete(download_active_experiment_model())

    # load input shape from model metadata
    model_path = os.path.join(args.tmp_dir, args.model_version, "model.tflite")
    model_metadata_path = os.path.join(
        args.tmp_dir, args.model_version, "tflite_metadata.json"
    )
    model_metadata = json.load(open(model_metadata_path, "r"))
    input_shape = model_metadata["inputShape"]
    # any batch size
    input_shape[0] = None

    tmp_sink = os.path.join(args.tfrecord_sink, "tmp")

    with beam.Pipeline(options=beam_options) as p:
        with beam_impl.Context(tmp_sink):
            parsed_dataset = (
                p
                | "Read TelemetryEvent"
                >> beam.io.ReadFromPubSub(
                    topic=topic_path,
                )
                | "Deserialize Flatbuffer"
                >> beam.Map(NestedTelemetryEvent.from_flatbuffer).with_output_types(
                    NestedTelemetryEvent
                )
                | "With timestamps"
                >> beam.Map(lambda x: beam.window.TimestampedValue(x, x.ts))
                | "Add Bounding Box Annotations"
                >> beam.Map(lambda x: predict_bounding_boxes(x, model_path))
            )

            parsed_dataset_by_session = (
                parsed_dataset
                | "Key NestedTelemetryEvent by session id"
                >> beam.Map(lambda x: (x.session, x))
            )
            tf_feature_spec = NestedTelemetryEvent.tf_feature_spec(args.num_detections)
            metadata = NestedTelemetryEvent.tfrecord_metadata(tf_feature_spec)

            batched_records = (
                parsed_dataset_by_session
                | f"Create batches size: {args.batch_size}"
                >> beam.GroupIntoBatches(args.batch_size)
            )

            tfrecord_sink_pipeline = batched_records | "Write TFRecords" >> beam.ParDo(
                WriteBatchedTFRecords(args.tfrecord_sink, metadata.schema)
            )

            parquet_sink_pipeline = batched_records | "Write Parquet" >> beam.Map(
                lambda x: write_batched_parquet(
                    x,
                    args.parquet_sink,
                    NestedTelemetryEvent.pyarrow_schema(args.num_detections),
                )
            )

            should_alert_per_session_windowed = (
                parsed_dataset
                | "Drop detections below confidence threshold & outside of calibration area of interest"
                >> beam.ParDo(
                    FilterDetections(args.calibration_base_path, score_threshold=0.5)
                )
                | "Add sliding window"
                >> beam.WindowInto(
                    beam.transforms.window.SlidingWindows(
                        args.health_window_size, args.health_window_period
                    ),
                )
                | "Flatten remaining observations in NestedTelemetryEvent"
                >> beam.ParDo(AsWindowedHealthRecord()).with_output_types(
                    WindowedHealthRecord
                )
                # | DataframeTransform(lambda df: df.groupby(['session', 'window_start', 'window_end']).sum())
                | "Key WindowedHealthRecord by session"
                >> beam.Map(lambda e: (e.session, e))
                | "Group by key" >> beam.GroupByKey()
                | "Write health checkpoint"
                >> beam.ParDo(
                    WriteHealthCheckpoint(
                        args.health_checkpoint_sink,
                        args.health_window_size,
                        args.health_window_period,
                    )
                )
                | f"Load last {args.health_window_size}s parquet checkpoints"
                >> beam.io.parquetio.ReadAllFromParquet()
                | "Rekey by session id" >> beam.Map(lambda x: (x["session"], x))
                | "Group by session id" >> beam.GroupByKey()
                | "Calculate health score trend"
                >> beam.ParDo(
                    CalcHealthScoreTrend(
                        args.health_checkpoint_sink,
                        args.health_window_size,
                        args.health_window_period,
                        args.api_url,
                        args.api_token,
                        health_threshold=3,
                    )
                )
            )
            render_video = (
                should_alert_per_session_windowed
                | "Render video and trigger alert"
                >> beam.ParDo(
                    RenderVideoTriggerAlert(
                        args.parquet_sink,
                        args.video_upload_path,
                        api_url=args.api_url,
                        api_token=args.api_token,
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--project", default="print-nanny-sandbox")

    parser.add_argument(
        "--topic",
        default="monitoring-frame-raw",
        help="PubSub topic",
    )

    parser.add_argument(
        "--quiet",
        default=False,
        help="Enable quiet mode to only log results and supress alert sending",
    )

    parser.add_argument(
        "--bucket",
        default="print-nanny-sandbox",
        help="PubSub topic",
    )

    parser.add_argument(
        "--tfrecord-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/tfrecords/batched",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--parquet-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/parquet/batched",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--video-upload-path",
        default="gs://print-nanny-sandbox/public/uploads/defect_alert",
    )

    parser.add_argument(
        "--health-checkpoint-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/windowed_health",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--calibration-base-path",
        default="gs://print-nanny-sandbox/uploads/device_calibration",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--health-window-size",
        default=60 * 10,
        help="Size of sliding event window (in seconds)",
    )

    parser.add_argument(
        "--health-window-period",
        default=30,
        help="Size of sliding event window slices (in seconds)",
    )

    parser.add_argument(
        "--num-detections",
        default=40,
        help="Max number of bounding boxes output by nms operation",
    )

    parser.add_argument("--api-token", help="Print Nanny API token")

    parser.add_argument(
        "--api-url", default="https://print-nanny.com/api", help="Print Nanny API url"
    )

    parser.add_argument(
        "--model-version",
        default="tflite-print3d_20201101015829-2021-02-24T05:16:05.082500Z",
    )

    parser.add_argument("--tmp-dir", default=".tmp/", help="Filesystem tmp directory")

    parser.add_argument(
        "--batch-size",
        default=3,
    )

    parser.add_argument("--postgres-host", default="localhost", help="Postgres host")

    parser.add_argument("--postgres-port", default=5432, help="Postgres port")

    parser.add_argument("--postgres-user", default="debug")

    parser.add_argument("--postgres-pass", default="debug")

    parser.add_argument("--postgres-db", default="print_nanny")

    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    run_pipeline(args, pipeline_args)
