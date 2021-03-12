#!/usr/bin/env python

from __future__ import absolute_import

import argparse
import logging
import io
import os
import logging

import apache_beam as beam
from apache_beam import window
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions

from tensorflow_transform.beam import impl as beam_impl

from encoders.tfrecord_example import ExampleProtoEncoder
from encoders.types import FlatTelemetryEvent

logger = logging.getLogger(__name__)

class AddWindowingInfoFn(beam.DoFn):
    """output tuple of window(key) + element(value)"""

    def process(self, element, window=beam.DoFn.WindowParam):
        yield (window, element)


class WriteWindowedTFRecords(beam.DoFn):
    """write one file per window/key"""

    def __init__(self, outdir, schema):
        self.outdir = outdir
        self.schema = schema

    def process(self, element):
        (window, elements) = element
        window_start = str(window.start.to_rfc3339())
        window_end = str(window.end.to_rfc3339())
        output = os.path.join(self.outdir, f"{window_start}-{window_end}")
        coder = ExampleProtoEncoder(self.schema)
        logger.info(f"Writing {output} with coder {coder}")
        yield (
            elements
            | beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=output,
                num_shards=0,
                shard_name_template="",
                file_name_suffix=".tfrecords.gz",
                coder=coder,
            )
        )

class PredictBoundingBoxes(beam.DoFn):
    pass

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
        "--sink",
        default="gs://print-nanny-sandbox/dataflow/tfrecords/windowed",
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
        default=30,
        help="Size of fixed streaming event window (in seconds)",
    )

    parser.add_argument(
        "--num-detections",
        default=40,
        help="Max number of bounding boxes output by nms operation",
    )

    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    topic_path = os.path.join("projects", args.project, "topics", args.topic)
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    tmp_sink = os.path.join(args.sink, "tmp")

    with beam.Pipeline(options=beam_options) as p:
        with beam_impl.Context(tmp_sink):
            parsed_dataset = (
                p
                | "Read TelemetryEvent"
                >> beam.io.ReadFromPubSub(
                    topic=topic_path,
                )
                | "Deserialize Flatbuffer"
                >> beam.Map(FlatTelemetryEvent.from_flatbuffer).with_output_types(
                    FlatTelemetryEvent
                )
                | "With timestamps"
                >> beam.Map(lambda x: beam.window.TimestampedValue(x, x.ts))
            )

            # health_pipeline = (
            #     parsed_dataset
            #     | "Add Sliding Window"
            #     >> beam.WindowInto(
            #         window.SlidingWindows(
            #             args.health_window_duration, args.health_window_interval
            #         )
            #     )
            #     | "Add Sliding Window Info" >> beam.ParDo(AddWindowingInfoFn())
            #     | "Group By Sliding Window" >> beam.GroupByKey()
            #     # | "Calculate health score"
            #     # | "Send alerts"
            # )

            feature_spec = FlatTelemetryEvent.feature_spec(args.num_detections)
            metadata = FlatTelemetryEvent.tfrecord_metadata(feature_spec)
            tfrecord_pipeline = (
                parsed_dataset
                | "Add Fixed Window"
                >> beam.WindowInto(window.FixedWindows(args.tfrecord_fixed_window))
                | "Add Fixed Window Info" >> beam.ParDo(AddWindowingInfoFn())
                | "Group By Fixed Window" >> beam.GroupByKey()
                | "Write Windowed TFRecords"
                >> beam.ParDo(WriteWindowedTFRecords(args.sink, metadata.schema))
            )
