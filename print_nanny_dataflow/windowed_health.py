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
from typing import List
from apache_beam import window
from typing import Tuple
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

    def process(self, element):
        import pdb

        pdb.set_trace()

        key, elements = element
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


class CalcHealthScoreTrend(beam.DoFn):

    STALE_TIMER = beam.transforms.userstate.TimerSpec(
        "stale", beam.TimeDomain.REAL_TIME
    )

    # HEALTH_TREND = beam.transforms.userstate.CombiningValueStateSpec('health_score_acc', beam.coders.coders.Coder(), combine_fn=HealthScoreCombineFn)
    UNHEALTHY_COUNT_ACC = beam.transforms.userstate.CombiningValueStateSpec(
        "unhealthy_count_acc", combine_fn=sum
    )

    def __init__(
        self,
        score_threshold=0.5,
        health_threshold=3,
    ):
        self.score_threshold = score_threshold
        self.health_threshold = health_threshold

    def process(
        self, elements, unhealthy_count_acc=beam.DoFn.StateParam(UNHEALTHY_COUNT_ACC)
    ):
        session, telemetry_events = elements
        logger.info(session)
        unhealthy_count_acc.add(1)
        import pdb

        pdb.set_trace()
        yield session, telemetry_events


class PredictBoundingBoxes(beam.DoFn):
    def __init__(self, model_path):
        self.model_path = model_path

    def process(self, element):

        tflite_interpreter = tf.lite.Interpreter(model_path=self.model_path)
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

        ymin = [b[0] for b in box_data]
        xmin = [b[1] for b in box_data]
        ymax = [b[2] for b in box_data]
        xmax = [b[3] for b in box_data]

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
        return [NestedTelemetryEvent(**defaults)]


class FilterDetections(beam.DoFn):
    def __init__(
        self,
        calibration_base_path: str,
        score_threshold: float = 0.5,
        calibration_filename: str = "calibration.json",
    ):
        self.score_threshold = 0.5
        self.calibration_filename = calibration_filename
        self.calibration_base_path = calibration_base_path

    def process(self, row: Tuple[bytes, NestedTelemetryEvent]):
        session, event = row
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        device_id = event.device_id
        device_calibration_path = os.path.join(
            self.calibration_base_path, str(device_id), self.calibration_filename
        )
        device_calibration_path = f"gcs://{device_calibration_path}"

        event = event.min_score_filter(score_threshold=self.score_threshold)

        if gcs_client.exists(device_calibration_path):
            with gcs_client.open(device_calibration_path, "r") as f:
                logger.info(
                    f"Loading device calibration from {device_calibration_path}"
                )
                calibration_json = json.load(f)

            coordinates = calibration_json["coordinates"]
            event = NestedTelemetryEvent.calibration_filter(event, coordinates)

        else:
            logger.info("Area of interest calibration not set")

        return session, event


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

        yield (elements | beam.io.parquetio.WriteToParquet(output_path, self.schema))


def run_pipeline(args, pipeline_args):
    topic_path = os.path.join("projects", args.project, "topics", args.topic)
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    # download model tarball
    if args.runner == "DataflowRunner":
        asyncio.run(download_active_experiment_model())

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
            )

            tf_feature_spec = NestedTelemetryEvent.tf_feature_spec(args.num_detections)
            metadata = NestedTelemetryEvent.tfrecord_metadata(tf_feature_spec)

            box_annotations = (
                parsed_dataset
                | "Add Bounding Box Annotations"
                >> beam.ParDo(PredictBoundingBoxes(model_path))
                | "Key by session id" >> beam.Map(lambda x: (x.session, x))
            )

            batched_records = (
                box_annotations
                | f"Create batches size: {args.batch_size}"
                >> beam.GroupIntoBatches(args.batch_size)
            )

            tfrecord_sink_pipeline = (
                batched_records
                | "Write TFRecords"
                >> beam.ParDo(
                    WriteBatchedTFRecords(args.tfrecord_sink, metadata.schema)
                )
                | "Print TFRecord paths" >> beam.Map(print)
            )

            parquet_sink_pipeline = batched_records | "Write Parquet" >> beam.ParDo(
                WriteBatchedParquet(
                    args.parquet_sink,
                    schema=NestedTelemetryEvent.pyarrow_schema(args.num_detections),
                    batch_size=args.batch_size,
                )
            )

            # @ todo sink annotated frames to GCS for video reconstruction
            # annotated_image_sink_pipeline = ()

            # @todo implement BoundedSession
            # probably easier to port everything to Java before attempting this
            # https://www.oreilly.com/library/view/streaming-systems/9781491983867/ch04.html
            health_models_by_device_id = (
                box_annotations
                | "Drop image bytes/Tensor"
                >> beam.Map(lambda x: (x[0], x[1].drop_image_data()))
                | "Drop detections outside of calibration area of interest"
                >> beam.ParDo(
                    FilterDetections(args.calibration_base_path, score_threshold=0.5)
                )
                # @todo implement area of interest filter
                | "Add Session Window"
                >> beam.WindowInto(
                    beam.transforms.window.SlidingWindows(
                        args.sliding_window_size, args.sliding_window_period
                    )
                )
                | "Group by key" >> beam.GroupByKey()
                | "Calculate health score trend"
                >> beam.ParDo(CalcHealthScoreTrend(health_threshold=3))
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

    parser.add_argument("--sliding-window-size", default=600)  # 10 minutes

    parser.add_argument("--sliding-window-period", default=5)  # 30 seconds

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
        default="gs://print-nanny-sandbox/telemetry_event/parquet/session",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--health-score-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/health",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--calibration-base-path",
        default="gs://print-nanny-sandbox/uploads/device_calibration",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--health-window-duration",
        default=60 * 20,
        help="Size of sliding event window (in seconds)",
    )

    parser.add_argument(
        "--health-window-interval",
        default=30,
        help="Size of sliding event window slices (in seconds)",
    )

    parser.add_argument(
        "--tfrecord-fixed-window",
        default=60,
        help="Size of fixed streaming event window (in seconds)",
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
