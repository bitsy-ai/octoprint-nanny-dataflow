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
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.metrics import Metrics

from print_nanny_dataflow.transforms.io import (
    WriteWindowedTFRecord,
    WriteWindowedParquet,
)
from print_nanny_dataflow.transforms.video import EncodeVideoRenderRequest
from print_nanny_dataflow.transforms.health import (
    ParseMonitoringImage,
    PredictBoundingBoxes,
    FilterBoxAnnotations,
)

from print_nanny_client.protobuf.monitoring_pb2 import (
    MonitoringImage,
    AnnotatedMonitoringImage,
)

from print_nanny_dataflow.transforms.io import TypedPathMixin
from print_nanny_dataflow.transforms.video import WriteAnnotatedImage
from print_nanny_client.protobuf.alert_pb2 import VideoRenderRequest

logger = logging.getLogger(__name__)


def add_timestamp(element: MonitoringImage):
    import apache_beam as beam

    return beam.window.TimestampedValue(element, element.metadata.ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--project", default="print-nanny-sandbox")

    parser.add_argument(
        "--subscription",
        default="sliding-window-health",
        help="PubSub subscription",
    )

    parser.add_argument(
        "--input-topic",
        default="MonitoringImage",
        help="PubSub subscription",
    )

    parser.add_argument(
        "--output-topic",
        default="VideoRenderRequest",
        help="PubSub subscription",
    )

    parser.add_argument(
        "--bucket",
        default="print-nanny-sandbox",
        help="GCS Bucket",
    )

    parser.add_argument(
        "--base-gcs-path",
        default="dataflow/sliding_window_health/",
        help="Base path for telemetry & monitoring event sinks",
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

    parser.add_argument(
        "--buffer-limit",
        default=20,
        help="Max number of bounding boxes output by nms operation",
    )

    parser.add_argument("--min-score-threshold", default=0.66)

    parser.add_argument("--max-boxes-to-draw", default=5)

    parser.add_argument("--model-path", default="dataflow/models")
    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))

    calibration_base_path = os.path.join(
        "gs://", args.bucket, args.calibration_base_path
    )

    pipeline_options = PipelineOptions(
        pipeline_args,
        streaming=True,
        runner=args.runner,
        project=args.project,
    )

    # use named subscription in dataflow runner
    if args.runner == "DataflowRunner":
        input_kwargs = dict(
            subscription=os.path.join(
                "projects", args.project, "subscriptions", args.subscription
            )
        )
    # otherwise create a new subscription for topic
    else:
        input_kwargs = dict(
            topic=os.path.join("projects", args.project, "topics", args.input_topic)
        )

    output_topic_path = os.path.join(
        "projects", args.project, "topics", args.output_topic
    )
    model_path = os.path.join("gs://", args.bucket, args.model_path, "model.tflite")

    p = beam.Pipeline(options=pipeline_options)

    # parse events from PubSub topic, add timestamp used in windowing functions, annotate with bounding boxes

    parsed_dataset_by_session = (
        p
        | "Read TelemetryEvent" >> beam.io.ReadFromPubSub(**input_kwargs)
        | "Deserialize Protobuf" >> beam.ParDo(ParseMonitoringImage())
        | beam.Map(add_timestamp)
        | "Add Bounding Box Annotations" >> beam.ParDo(PredictBoundingBoxes(model_path))
        | "Key by session"
        >> beam.Map(
            lambda x: (x.monitoring_image.metadata.print_session.session, x)
        ).with_output_types(Tuple[str, AnnotatedMonitoringImage])
    )

    fixed_window_view_by_key = (
        parsed_dataset_by_session
        | f"Add fixed window"
        >> beam.WindowInto(
            beam.transforms.window.FixedWindows(args.health_window_period)
        )
        | "Group FixedWindow by key" >> beam.GroupByKey()
    )

    annotated_images = (
        fixed_window_view_by_key
        | beam.ParDo(
            FilterBoxAnnotations(
                calibration_base_path,
            )
        )
        | "Write annotated jpgs"
        >> beam.ParDo(
            WriteAnnotatedImage(
                base_path=args.base_gcs_path,
                bucket=args.bucket,
                pipeline_options=pipeline_options,
                score_threshold=args.min_score_threshold,
                max_boxes_to_draw=args.max_boxes_to_draw,
            )
        )
    )

    tfrecord_sink = fixed_window_view_by_key | beam.ParDo(
        WriteWindowedTFRecord(
            base_path=args.base_gcs_path,
            bucket=args.bucket,
            module=f"{AnnotatedMonitoringImage.__module__}.{AnnotatedMonitoringImage.__name__}",
        )
    )

    # render video after session is finished
    session_gap = args.health_window_period * 2
    sessions_latest_by_key = (
        parsed_dataset_by_session
        | beam.WindowInto(
            beam.transforms.window.Sessions(session_gap),
            # trigger=trigger_fn,
        )
        | beam.combiners.Latest.PerKey()
    )

    render_video_request = (
        sessions_latest_by_key
        | beam.ParDo(EncodeVideoRenderRequest())
        | beam.io.WriteToPubSub(output_topic_path)
    )

    result = p.run()
    if args.runner == "DirectRunner":
        result.wait_until_finish()
