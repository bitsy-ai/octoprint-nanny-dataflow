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
from typing import List, Tuple, Any, Iterable, Generator, Coroutine, Optional, Union
from apache_beam import window
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import PIL
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_serving.apis import predict_pb2
from apache_beam.dataframe.convert import to_dataframe, to_pcollection
from apache_beam.dataframe.transforms import DataframeTransform
from apache_beam.transforms.trigger import (
    OrFinally,
    AfterCount,
    Repeatedly,
    AfterWatermark,
)
from apache_beam.transforms import trigger
from tensorflow_transform.tf_metadata import dataset_metadata

from print_nanny_dataflow.transforms.io import (
    WriteWindowedTFRecord,
    WriteWindowedParquet,
)
from print_nanny_dataflow.transforms.health import (
    ExplodeWindowedHealthRecord,
    predict_bounding_boxes,
    health_score_trend_polynomial_v1,
    FilterAreaOfInterest,
    SortWindowedHealthDataframe,
    MonitorHealthStateful,
)

from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    WindowedHealthDataFrames,
)

from print_nanny_dataflow.utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from print_nanny_dataflow.clients.rest import RestAPIClient
import pyarrow as pa

logger = logging.getLogger(__name__)


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


def run_pipeline(args, pipeline_args):
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    input_topic_path = os.path.join("projects", args.project, "topics", args.topic)
    output_topic_path = os.path.join(
        "projects", args.project, "topics", args.render_video_topic
    )

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

    with beam.Pipeline(options=beam_options) as p:
        # parse events from PubSub topic, add timestamp used in windowing functions, annotate with bounding boxes
        parsed_dataset = (
            p
            | "Read TelemetryEvent"
            >> beam.io.ReadFromPubSub(
                topic=input_topic_path,
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

        # key by session id
        parsed_dataset_by_session = (
            parsed_dataset
            | "Key NestedTelemetryEvent by session id"
            >> beam.Map(lambda x: (x.session, x))
        )

        fixed_window_view = (
            parsed_dataset_by_session
            | f"Add fixed window"
            >> beam.WindowInto(
                beam.transforms.window.FixedWindows(args.health_window_period)
            )
            | "Group FixedWindow NestedTelemetryEvent by key" >> beam.GroupByKey()
        )

        _ = fixed_window_view | "Write FixedWindow TFRecords" >> beam.ParDo(
            WriteWindowedTFRecord(
                args.fixed_window_tfrecord_sink,
                NestedTelemetryEvent.tfrecord_schema(args.num_detections),
            )
        )

        _ = fixed_window_view | "Write FixedWindow Parquet" >> beam.ParDo(
            WriteWindowedParquet(
                args.fixed_window_parquet_sink,
                NestedTelemetryEvent.pyarrow_schema(args.num_detections),
            )
        )

        sliding_window_view = parsed_dataset | "Add sliding window" >> beam.WindowInto(
            beam.transforms.window.SlidingWindows(
                args.health_window_size, args.health_window_period
            ),
            accumulation_mode=beam.transforms.trigger.AccumulationMode.ACCUMULATING,
        )

        _ = (
            sliding_window_view
            | "Write SlidingWindow ExplodeWindowedHealthRecord Parquet (unfilterd)"
            >> beam.ParDo(ExplodeWindowedHealthRecord())
            | "Group unfiltered health records by key" >> beam.GroupBy("session")
            | "Write SlidingWindow Parquet"
            >> beam.ParDo(
                WriteWindowedParquet(
                    args.sliding_window_health_raw_sink,
                    WindowedHealthRecord.pyarrow_schema(),
                )
            )
        )

        # @TODO enrich pcol with calibration as side input to avoid rdisk reload -> group -> transform -> regroup?
        # This is implemented as a Singleton instance in Java, which can be shared across threads and used as a cache
        # The Python implementation probably uses multiprocessing.shared_memory, which might actually be higher latency than hitting disk for most workloads?

        windowed_health_dataframe = (
            sliding_window_view
            | "Drop image data" >> beam.Map(lambda v: v.drop_image_data())
            | "Group alert pipeline by session" >> beam.GroupBy("session")
            | "Filter detections below threshold & outside area of interest"
            >> beam.ParDo(
                FilterAreaOfInterest(args.calibration_base_path, score_threshold=0.5)
            )
            | beam.ParDo(ExplodeWindowedHealthRecord())
            | beam.GroupBy("session")
            | "Windowed health DataFrame" >> beam.ParDo(SortWindowedHealthDataframe())
        )

        _ = (
            windowed_health_dataframe
            | "Write health trend parquet" >> beam.GroupByKey()
            | beam.ParDo(
                WriteWindowedParquet(
                    args.sliding_window_health_trend_sink,
                    WindowedHealthDataFrames.pyarrow_schema(args.num_detections),
                )
            )
        )

        alert_pipeline_trigger = OrFinally(Repeatedly(AfterCount(1)), AfterWatermark())
        session_gap = args.health_window_period * 3

        # accumulates failure count
        stateful_health_dataframe = (
            windowed_health_dataframe
            | beam.WindowInto(
                beam.transforms.window.Sessions(session_gap),
                trigger=alert_pipeline_trigger,
                accumulation_mode=beam.transforms.trigger.AccumulationMode.DISCARDING,
            )
            | "Stateful health score threshold monitor"
            >> beam.ParDo(MonitorHealthStateful(output_topic_path))
        )

        _ = (
            stateful_health_dataframe
            | "Write session windows to Parquet" >> beam.GroupByKey()
            | beam.ParDo(
                WriteWindowedParquet(
                    args.session_window_health_trend_sink,
                    WindowedHealthDataFrames.pyarrow_schema(),
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
        "--session-window-gap-size",
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
        help="GCS Bucket",
    )

    parser.add_argument(
        "--render-video-topic",
        default="monitoring-video-render",
        help="Video rendering and alert push jobs will be published to this PubSub topic",
    )

    parser.add_argument(
        "--fixed-window-tfrecord-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/tfrecords",
        help="Unfiltered NestedTelemetryEvent emitted from FixedWindow (single point in time)",
    )

    parser.add_argument(
        "--fixed-window-parquet-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/parquet",
        help="Unfiltered NestedTelemetryEvent emitted from FixedWindow (single point in time)",
    )

    parser.add_argument(
        "--sliding-window-health-raw-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/sliding_window/WindowedHealthRecord/parquet",
        help="Unfiltered WindowedHealthRecord emitted from SlidingWindow",
    )

    parser.add_argument(
        "--session-window-health-trend-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/session_window/WindowedHealthDataFrames",
        help="Post-filtered WindowedHelathDataframe emitted from session window",
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
        default=60,
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
        default=256,
    )

    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    run_pipeline(args, pipeline_args)
