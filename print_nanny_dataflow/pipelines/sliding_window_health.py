#!/usr/bin/env python

from __future__ import absolute_import

import aiohttp
import argparse
import logging
import io
import os
import json
import logging
import tarfile

import apache_beam as beam
from typing import List, Tuple, Any, Iterable, Generator, Coroutine, Optional, Union
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions

from apache_beam.transforms import trigger
from tensorflow_transform.tf_metadata import dataset_metadata

from print_nanny_dataflow.transforms.io import (
    WriteWindowedTFRecord,
    WriteWindowedParquet,
)
from print_nanny_dataflow.transforms.health import (
    ExplodeWindowedHealthRecord,
    PredictBoundingBoxes,
    FilterAreaOfInterest,
    SortWindowedHealthDataframe,
    CreateVideoRenderMessage,
)

from print_nanny_dataflow.transforms.video import WriteAnnotatedImage

from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    NestedWindowedHealthTrend,
)

import print_nanny_dataflow
from print_nanny_dataflow.clients.rest import RestAPIClient

logger = logging.getLogger(__name__)


async def download_active_experiment_model(model_dir=".tmp/", model_artifact_id=1):

    tmp_artifacts_tarball = os.path.join(model_dir, "artifacts.tar.gz")
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
        tar.extractall(model_dir)
    logger.info(f"Finished extracting {tmp_artifacts_tarball}")


def add_timestamp(element):
    import apache_beam as beam

    return beam.window.TimestampedValue(element, element.ts)


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
        "--cdn-base-path",
        default="media",
    )

    parser.add_argument(
        "--cdn-upload-path",
        default="uploads/PrintSessionAlert",
    )

    parser.add_argument(
        "--render-video-topic",
        default="monitoring-video-render",
        help="Video rendering and alert push jobs will be published to this PubSub topic",
    )

    parser.add_argument(
        "--fixed-window-tfrecord-sink",
        default="dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/tfrecords",
        help="Unfiltered NestedTelemetryEvent emitted from FixedWindow (single point in time)",
    )

    parser.add_argument(
        "--fixed-window-parquet-sink",
        default="dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/parquet",
        help="Unfiltered NestedTelemetryEvent emitted from FixedWindow (single point in time)",
    )

    parser.add_argument(
        "--fixed-window-jpg-sink",
        default="dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/jpg",
        help="Bounding-box annotated images (single point in time)",
    )

    parser.add_argument(
        "--fixed-window-mp4-sink",
        default="dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/mp4",
        help="Bounding-box annotated video (single point in time)",
    )

    parser.add_argument(
        "--sliding-window-health-raw-sink",
        default="dataflow/telemetry_event/sliding_window/WindowedHealthRecord/parquet",
        help="Unfiltered WindowedHealthRecord emitted from SlidingWindow",
    )

    parser.add_argument(
        "--sliding-window-health-filtered-sink",
        default="dataflow/telemetry_event/sliding_window/WindowedHealthRecord/filtered/parquet",
        help="Unfiltered WindowedHealthRecord emitted from SlidingWindow",
    )

    parser.add_argument(
        "--session-window-health-trend-sink",
        default="dataflow/telemetry_event/session_window/NestedWindowedHealthTrend/parquet",
        help="Post-filtered WindowedHelathDataframe emitted from session window",
    )

    parser.add_argument(
        "--calibration-base-path",
        default="uploads/device_calibration",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--health-window-size",
        default=60 * 5,
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
        "--batch-size",
        default=256,
    )

    parser.add_argument("--model-path", default="dataflow/models")
    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))

    fixed_window_tfrecord_sink = os.path.join(
        "gs://", args.bucket, args.fixed_window_tfrecord_sink
    )
    fixed_window_parquet_sink = os.path.join(
        "gs://", args.bucket, args.fixed_window_parquet_sink
    )
    fixed_window_jpg_sink = os.path.join(
        "gs://", args.bucket, args.fixed_window_jpg_sink
    )
    fixed_window_mp4_sink = os.path.join(
        "gs://", args.bucket, args.fixed_window_mp4_sink
    )
    sliding_window_health_raw_sink = os.path.join(
        "gs://", args.bucket, args.sliding_window_health_raw_sink
    )
    sliding_window_health_filtered_sink = os.path.join(
        "gs://", args.bucket, args.sliding_window_health_filtered_sink
    )
    calibration_base_path = os.path.join(
        "gs://", args.bucket, args.calibration_base_path
    )

    beam_options = PipelineOptions(
        pipeline_args, streaming=True, runner=args.runner, project=args.project
    )

    input_topic_path = os.path.join("projects", args.project, "topics", args.topic)
    output_topic_path = os.path.join(
        "projects", args.project, "topics", args.render_video_topic
    )

    # load input shape from model metadata
    model_path = os.path.join("gs://", args.bucket, args.model_path)
    # model_metadata_path = os.path.join(
    #     "gs://", args.bucket, args.model_path, "tflite_metadata.json"
    # )
    # any batch size

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
            | "With timestamps" >> beam.Map(add_timestamp)
            | "Add Bounding Box Annotations"
            >> beam.ParDo(PredictBoundingBoxes(model_path))
        )

        # key by session id
        parsed_dataset_by_session = (
            parsed_dataset
            | "Key NestedTelemetryEvent by session id"
            >> beam.Map(lambda x: (x.print_session, x))
        )

        fixed_window_view = (
            parsed_dataset_by_session
            | f"Add fixed window"
            >> beam.WindowInto(
                beam.transforms.window.FixedWindows(args.health_window_period)
            )
        )

        fixed_window_view_by_key = (
            fixed_window_view
            | "Group FixedWindow NestedTelemetryEvent by key" >> beam.GroupByKey()
        )

        _ = (
            fixed_window_view_by_key
            | "Filter area of interest and detections above threshold"
            >> beam.ParDo(
                FilterAreaOfInterest(
                    calibration_base_path,
                )
            )
            | "Write annotated jpgs"
            >> beam.ParDo(WriteAnnotatedImage(args.fixed_window_jpg_sink))
        )

        _ = fixed_window_view_by_key | "Write FixedWindow TFRecords" >> beam.ParDo(
            WriteWindowedTFRecord(
                fixed_window_tfrecord_sink,
                NestedTelemetryEvent.tfrecord_schema(args.num_detections),
            )
        )

        _ = fixed_window_view_by_key | "Write FixedWindow Parquet" >> beam.ParDo(
            WriteWindowedParquet(
                fixed_window_parquet_sink,
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
            | "Group unfiltered health records by key" >> beam.GroupBy("print_session")
            | "Write SlidingWindow Parquet"
            >> beam.ParDo(
                WriteWindowedParquet(
                    args.sliding_window_health_filtered_sink,
                    WindowedHealthRecord.pyarrow_schema(),
                )
            )
        )

        # @TODO enrich pcol with calibration as side input to avoid rdisk reload -> group -> transform -> regroup?
        # This is implemented as a Singleton instance in Java, which can be shared across threads and used as a cache
        # The Python implementation probably uses multiprocessing.shared_memory, which might actually be higher latency than hitting disk for most workloads?

        filtered_health_dataframe = (
            sliding_window_view
            | "Drop image data" >> beam.Map(lambda v: v.drop_image_data())
            | "Group alert pipeline by session" >> beam.GroupBy("print_session")
            | "Filter detections below threshold & outside area of interest"
            >> beam.ParDo(
                FilterAreaOfInterest(calibration_base_path, score_threshold=0.5)
            )
            | beam.ParDo(ExplodeWindowedHealthRecord())
            | "Windowed health DataFrame" >> beam.GroupBy("print_session")
            | beam.ParDo(SortWindowedHealthDataframe())
            | beam.GroupByKey()
        )

        _ = (
            filtered_health_dataframe
            | "Write SlidingWindow (calibration & threshold filtered) Parquet"
            >> beam.ParDo(
                WriteWindowedParquet(
                    args.sliding_window_health_raw_sink,
                    NestedWindowedHealthTrend.pyarrow_schema(),
                )
            )
        )

        # TODO re-enable Afterwatermark triggers with MonitorHealthStateful
        # alert_pipeline_trigger = AfterWatermark(
        #     early=AfterProcessingTime(args.health_window_period), late=AfterCount(1)
        # )
        session_gap = args.health_window_period * 1.5
        logging.info(f"Accumulating events with session gap={session_gap}")

        # accumulates failure count
        session_accumulating_dataframe = (
            # windowed_health_dataframe
            parsed_dataset_by_session
            # filtered_health_dataframe
            | beam.WindowInto(
                beam.transforms.window.Sessions(session_gap),
                # TODO re-enable with MonitorHealthStateful
                # trigger=alert_pipeline_trigger,
                accumulation_mode=beam.transforms.trigger.AccumulationMode.DISCARDING,
            )
            # | "Stateful health score threshold monitor"
            # >> beam.ParDo(MonitorHealthStateful(output_topic_path))FW
        )

        on_session_end = (
            session_accumulating_dataframe
            | "Should alert for session?" >> beam.GroupByKey()
            | beam.ParDo(
                CreateVideoRenderMessage(
                    fixed_window_jpg_sink,
                    fixed_window_mp4_sink,
                    args.cdn_base_path,
                    args.cdn_upload_path,
                    args.bucket,
                )
            )
            | "Write to PubSub" >> beam.io.WriteToPubSub(output_topic_path)
        )
