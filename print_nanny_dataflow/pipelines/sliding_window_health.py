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

from print_nanny_dataflow.transforms.io import (
    WriteWindowedTFRecord,
    WriteWindowedParquet,
)
from print_nanny_dataflow.transforms.health import (
    ParseMonitoringImage,
    PredictBoundingBoxes,
    FilterBoxAnnotations,
)

from print_nanny_client.protobuf.monitoring_pb2 import (
    MonitoringImage,
    AnnotatedMonitoringImage,
)

from print_nanny_dataflow.transforms.video import WriteAnnotatedImage
from print_nanny_dataflow.metrics import FixedWindowMetricStart, FixedWindowMetricEnd


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

    input_subscription_path = os.path.join(
        "projects", args.project, "subscriptions", args.subscription
    )

    model_path = os.path.join("gs://", args.bucket, args.model_path, "model.tflite")

    p = beam.Pipeline(options=pipeline_options)

    # parse events from PubSub topic, add timestamp used in windowing functions, annotate with bounding boxes

    parsed_dataset_by_session = (
        p
        | "Read TelemetryEvent"
        >> beam.io.ReadFromPubSub(subscription=input_subscription_path)
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

    fixed_window_view_by_key | beam.Map(
        lambda key: print(f"Processed {len(key[1])} for session {key[0]}")
    )

    _ = (
        fixed_window_view_by_key
        | "Calculate metrics over fixed window intervals"
        >> beam.ParDo(FixedWindowMetricStart(args.health_window_period, "print_health"))
    )

    _ = (
        fixed_window_view_by_key
        | "Filter area of interest and detections above threshold"
        >> beam.ParDo(
            FilterBoxAnnotations(
                calibration_base_path,
            )
        )
        | "Write annotated jpgs"
        >> beam.ParDo(
            WriteAnnotatedImage(
                base_path=args.base_gcs_path,
                bucket=args.bucket,
                score_threshold=args.min_score_threshold,
                max_boxes_to_draw=args.max_boxes_to_draw,
                window_type=beam.transforms.window.FixedWindows.__name__,
            )
        )
    )

    _ = fixed_window_view_by_key | "Write FixedWindow TFRecords" >> beam.ParDo(
        WriteWindowedTFRecord(
            base_path=args.base_gcs_path,
            bucket=args.bucket,
            module=f"{AnnotatedMonitoringImage.__module__}.{AnnotatedMonitoringImage.__name__}",
            window_type=beam.transforms.window.FixedWindows.__name__,
        )
    )

    # _ = fixed_window_view_by_key | "Write FixedWindow Parquet" >> beam.ParDo(
    #     WriteWindowedParquet(
    #         args.base_gcs_path,
    #         bucket=args.bucket,
    #         module=f"{AnnotatedMonitoringImage.__module__}.{AnnotatedMonitoringImage.__name__}"
    #     )
    # )

    # sliding_window_view = parsed_dataset | "Add sliding window" >> beam.WindowInto(
    #     beam.transforms.window.SlidingWindows(
    #         args.health_window_size, args.health_window_period
    #     ),
    #     accumulation_mode=beam.transforms.trigger.AccumulationMode.ACCUMULATING,
    # )

    # _ = (
    #     sliding_window_view
    #     | "Write SlidingWindow ExplodeWindowedHealthRecord Parquet (unfilterd)"
    #     >> beam.ParDo(ExplodeWindowedHealthRecord())
    #     | "Group unfiltered health records by key" >> beam.GroupBy("print_session")
    #     | "Write SlidingWindow Parquet"
    #     >> beam.ParDo(
    #         WriteWindowedParquet(
    #             args.base_gcs_path,
    #             WindowedHealthRecord.pyarrow_schema(),
    #             record_type="WindowedHealthRecord/parquet",
    #         )
    #     )
    # )

    # filtered_health_dataframe = (
    #     sliding_window_view
    #     | "Drop image data" >> beam.Map(lambda v: v.drop_image_data())
    #     | "Group alert pipeline by session" >> beam.GroupBy("print_session")
    #     | "Filter detections below threshold & outside area of interest"
    #     >> beam.ParDo(FilterAreaOfInterest(calibration_base_path))
    #     | beam.ParDo(ExplodeWindowedHealthRecord())
    #     | "Windowed health DataFrame" >> beam.GroupBy("print_session")
    #     | beam.ParDo(SortWindowedHealthDataframe())
    #     | beam.GroupByKey()
    # )

    # _ = (
    #     filtered_health_dataframe
    #     | "Write SlidingWindow (calibration & threshold filtered) Parquet"
    #     >> beam.ParDo(
    #         WriteWindowedParquet(
    #             args.base_gcs_path,
    #             NestedWindowedHealthTrend.pyarrow_schema(),
    #             record_type="NestedWindowedHealthTrend/parquet",
    #         )
    #     )
    # )

    result = p.run()
    if args.runner == "DirectRunner":
        result.wait_until_finish()
