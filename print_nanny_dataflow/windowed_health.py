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
import numpy as np
import tensorflow as tf
import apache_beam as beam
from typing import List, Tuple, Any, Iterable, Generator
from apache_beam import window
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import PIL
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_serving.apis import predict_pb2

from encoders.tfrecord_example import ExampleProtoEncoder
from encoders.types import NestedTelemetryEvent, FlatTelemetryEvent
from clients.rest import RestAPIClient
import pyarrow as pa

logger = logging.getLogger(__name__)

DETECTION_LABELS = {
    1: "nozzle",
    2: "adhesion",
    3: "spaghetti",
    4: "print",
    5: "raft",
}

NEUTRAL_LABELS = {1: "nozzle", 5: "raft"}

NEGATIVE_LABELS = {
    2: "adhesion",
    3: "spaghetti",
}

POSITIVE_LABELS = {
    4: "print",
}


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
            e.asdict()
            for e in elements
            | beam.io.parquetio.WriteToParquet(output_path, self.schema)
        )


def write_batched_parquet(batched_elements, parquet_base_path, schema):
    session, elements = batched_elements
    output_path = os.path.join(
        parquet_base_path, session, str(int(datetime.now().timestamp()))
    )

    elements = beam.Create([e.asdict() for e in elements])
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
        keyed_elements: Tuple[Any, Iterable[NestedTelemetryEvent]],
        window=beam.DoFn.WindowParam,
    ):

        session, telemetry_events = keyed_elements

        base_path = os.path.join(self.checkpoint_sink, session)
        pq_glob_path = os.path.join(
            base_path,
        )

        window_start = int(window.start)
        window_end = int(window.end)
        output_path = os.path.join(
            base_path,
            f"{window_start}_{window_end}.parquet",
        )
        df = pd.concat(
            [
                e.to_health_dataframe(window_start=window_start, window_end=window_end)
                for e in telemetry_events
            ]
        ).sort_values("ts")
        df.to_parquet(output_path, engine="pyarrow")
        logger.info(f"Wrote {output_path}")

        gcs_client = beam.io.gcp.gcsio.GcsIO()

        past_checkpoints = list(gcs_client.list_prefix(pq_glob_path).keys())
        if len(past_checkpoints) > self.window_lookback:
            # get most recent windows from selection
            past_checkpoints = past_checkpoints[-self.window_lookback :]

        return past_checkpoints


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

    # def trailing_window_trends(
    #     self, session: str, window: beam.DoFn.WindowParam
    # ) -> bool:
    #     gcs_client = beam.io.gcp.gcsio.GcsIO()

    #     # if health trending download, look back at past n observations and alert if health_threshold is exceeded
    #     past_trend_dfs = []
    #     for n in range(1, self.health_threshold):

    #         window_start = window.start - self.window_size
    #         window_end = window.end - (n * self.window_size)
    #         read_csv_path = os.path.join(
    #             self.checkpoint_sink,
    #             session,
    #             f"{window_start}_{window_end}",
    #             "trend.csv",
    #         )

    #         if gcs_client.exists(read_csv_path):
    #             past_trend_df = pd.read_csv(past_trend_df)
    #             past_trend_dfs.append(past_trend_df)
    #     if len(past_trend_df) > 0:
    #         return pd.concat(past_trend_dfs)

    def should_alert(
        self, trailing_trend_df: pd.DataFrame, current_trend_df: pd.DataFrame
    ) -> bool:
        df = trailing_trend_df.concat(current_trend_df)
        return len(df["slope"] < 0) >= self.health_threshold

    async def send_alert_async(
        self, session: str, telemetry_event: NestedTelemetryEvent
    ):
        rest_client = RestAPIClient(api_token=self.api_token, api_url=self.api_url)

        res = await rest_client.create_defect_alert(
            telemetry_event.device_id,
            user=telemetry_event.user_id,
            print_session=session,
        )

        logger.info(f"create_defect_alert res={res}")

    def process(
        self,
        keyed_elements,
        window=beam.DoFn.WindowParam,
    ):
        session, telemetry_events = keyed_elements

        df = (
            pd.DataFrame.from_records(telemetry_events)
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

        df = pd.concat(
            {
                "unhealthy": df[
                    df.index.get_level_values("detection_class").isin(
                        NEGATIVE_LABELS.keys()
                    )
                ],
                "healthy": df[
                    df.index.get_level_values("detection_class").isin(
                        POSITIVE_LABELS.keys()
                    )
                ],
            }
        ).reset_index()

        mask = df.level_0 == "unhealthy"
        healthy_cumsum = (
            df[~mask].groupby("detection_class")["detection_score"].sum().cumsum()
        )

        unhealthy_cumsum = (
            df[mask].groupby("detection_class")["detection_score"].sum().cumsum()
        )

        xy = healthy_cumsum.subtract(unhealthy_cumsum, fill_value=0)

        import pdb

        pdb.set_trace()

        if len(xy) <= 1:
            logger.info(f"Length of cumulative sum <= 1 {xy}")
            return
        trend = np.polynomial.polynomial.Polynomial.fit(
            xy.index, xy, self.polyfit_degree
        )

        slope, intercept = tuple(trend)

        current_trend_df = pd.DataFrame(
            [[slope, intercept, window.start, window.end]],
            columns=["slope", "intercept", "window_start", "window_end"],
        )
        current_trend_df.to_csv(trend_output_path)

        # if health trending download, look back at past n observations and alert if health_threshold is exceeded
        if slope < 0:
            trailing_trend_df = self.trailing_window_trends(session, window)
            if self.should_alert(trailing_trend_df, current_trend_df):
                res = asyncio.get_event_loop().run_until_complete(send_alert_async())

        res = asyncio.get_event_loop().run_until_complete(send_alert_async())


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
    defaults = element.asdict()
    defaults.update(params)
    return NestedTelemetryEvent(**defaults)


class FilterDetections(beam.DoFn):
    def __init__(
        self,
        calibration_base_path: str,
        score_threshold: float = 0.5,
        calibration_filename: str = "calibration.json",
    ):
        self.score_threshold = 0.5
        self.calibration_base_path = calibration_base_path
        self.score_threshold = score_threshold
        self.calibration_filename = calibration_filename

    def process(
        self, keyed_row: Tuple[Any, NestedTelemetryEvent]
    ) -> Tuple[Any, NestedTelemetryEvent]:
        session, event = keyed_row
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        device_id = event.device_id
        device_calibration_path = os.path.join(
            self.calibration_base_path, str(device_id), self.calibration_filename
        )

        event = event.min_score_filter(score_threshold=self.score_threshold)

        if gcs_client.exists(device_calibration_path):
            with gcs_client.open(device_calibration_path, "r") as f:
                logger.info(
                    f"Loading device calibration from {device_calibration_path}"
                )
                calibration_json = json.load(f)

            coordinates = calibration_json["coordinates"]
            event = NestedTelemetryEvent.calibration_filter(event, coordinates)

        if event.num_detections >= 1:
            yield (session, event)


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
                | "Key by session id" >> beam.Map(lambda x: (x.session, x))
            )

            tf_feature_spec = NestedTelemetryEvent.tf_feature_spec(args.num_detections)
            metadata = NestedTelemetryEvent.tfrecord_metadata(tf_feature_spec)

            # box_annotations = (
            #     parsed_dataset
            #     | "Add Bounding Box Annotations"
            #     >> beam.ParDo(PredictBoundingBoxes(model_path))
            # )

            # batched_records = (
            #     parsed_dataset
            #     | f"Create batches size: {args.batch_size}"
            #     >> beam.GroupIntoBatches(args.batch_size)
            # )

            # tfrecord_sink_pipeline = (
            #     batched_records
            #     | "Write TFRecords"
            #     >> beam.ParDo(
            #         WriteBatchedTFRecords(args.tfrecord_sink, metadata.schema)
            #     )
            #     | "Print TFRecord paths" >> beam.Map(print)
            # )

            # parquet_sink_pipeline = (
            #     batched_records
            #     | "Write Parquet" >> beam.Map(lambda x:
            #     write_batched_parquet(
            #         x,
            #         args.parquet_sink,
            #         NestedTelemetryEvent.pyarrow_schema(args.num_detections),
            #     ))
            #     | "Print Parquet paths" >> beam.Map(print)
            # )

            # @ todo sink annotated frames to GCS for video reconstruction
            # annotated_image_sink_pipeline = ()

            # @todo implement BoundedSession
            # probably easier to port everything to Java before attempting this
            # https://www.oreilly.com/library/view/streaming-systems/9781491983867/ch04.html
            health_per_session = (
                parsed_dataset
                | "Drop image bytes/Tensor"
                >> beam.Map(lambda x: (x[0], x[1].drop_image_data()))
                | "Drop detections below confidence threshold & outside of calibration area of interest"
                >> beam.ParDo(
                    FilterDetections(args.calibration_base_path, score_threshold=0.5)
                )
                # @todo implement area of interest filter
                | "Add sliding window"
                >> beam.WindowInto(
                    beam.transforms.window.SlidingWindows(
                        args.health_window_size, args.health_window_period
                    )
                )
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


# class WindowedHealthOptions(PipelineOptions):

#   @classmethod
#   def _add_argparse_args(cls, parser):


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
        "--health-checkpoint-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/health",
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
