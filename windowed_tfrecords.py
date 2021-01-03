#!/usr/bin/env python

from __future__ import absolute_import

import argparse
import json
import gzip
import logging
import sys
import base64
import io
import os
import logging

import apache_beam as beam
from apache_beam import window

from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.beam import impl as beam_impl


def preprocess_features(input_bytes):
    fileobj = io.BytesIO(input_bytes)
    with gzip.GzipFile(fileobj=fileobj, mode="r") as f:
        output_features = f.read()

    output_features = json.loads(output_features)
    output_features["original_image"] = base64.b64decode(
        output_features["original_image"]
    )
    output_features["detection_boxes_x0"] = [
        x[0] for x in output_features["detection_boxes"]
    ]
    output_features["detection_boxes_y0"] = [
        x[1] for x in output_features["detection_boxes"]
    ]
    output_features["detection_boxes_x1"] = [
        x[2] for x in output_features["detection_boxes"]
    ]
    output_features["detection_boxes_y1"] = [
        x[3] for x in output_features["detection_boxes"]
    ]

    output_features["calibration_x0"] = output_features["calibration"][0]
    output_features["calibration_y0"] = output_features["calibration"][1]
    output_features["calibration_x1"] = output_features["calibration"][2]
    output_features["calibration_y1"] = output_features["calibration"][3]

    del output_features["calibration"]
    del output_features["detection_boxes"]
    return output_features


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
        yield (
            elements
            | beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=os.path.join(
                    self.outdir, f"{window_start}-{window_end}"
                ),
                num_shards=1,
                shard_name_template="",
                file_name_suffix=".tfrecords.gz",
                coder=example_proto_coder.ExampleProtoCoder(self.schema),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--loglevel",
        default="INFO"
    )

    parser.add_argument(
        "--topic",
        default="projects/print-nanny/topics/bounding-boxes-dev,
        help="PubSub topic to subscribe for bounding box predictions",
    )

    parser.add_argument(
        "--sink",
        default="gs://print-nanny-dev/dataflow/bounding-box-events/windowed",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument("--project", default="print-nany", help="GCP Project ID")

    parser.add_argument(
        "--window", default=30, help="Size of fixed streaming event window (in seconds)"
    )

    parser.add_argument(
        "--max-detections",
        default=40,
        help="Max number of bounding boxes output by nms operation",
    )

    parser.add_argument(
        "--runner",
        default="DataflowRunner"
    )

    args, pipeline_args = parser.parse_known_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args,
        save_main_session=True,
        streaming=True,
        runner=args.runner
    )

    tmp_sink = os.path.join(args.sink, "tmp")
    input_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(
            {
                "ts": tf.io.FixedLenFeature([], tf.float32),
                "device_id": tf.io.FixedLenFeature([], tf.int64),
                "device_cloudiot_name": tf.io.FixedLenFeature([], tf.string),
                "user_id": tf.io.FixedLenFeature([], tf.int64),
                "calibration_x0": tf.io.FixedLenFeature([], tf.float32),
                "calibration_y0": tf.io.FixedLenFeature([], tf.float32),
                "calibration_x1": tf.io.FixedLenFeature([], tf.float32),
                "calibration_y1": tf.io.FixedLenFeature([], tf.float32),
                "num_detections": tf.io.FixedLenFeature([], tf.float32),
                "detection_classes": tf.io.FixedLenFeature(
                    [args.max_detections], tf.int64
                ),
                "detection_scores": tf.io.FixedLenFeature(
                    [args.max_detections], tf.float32
                ),
                "original_image": tf.io.FixedLenFeature([], tf.string),
                "detection_boxes_x0": tf.io.FixedLenFeature(
                    [args.max_detections], tf.float32
                ),
                "detection_boxes_y0": tf.io.FixedLenFeature(
                    [args.max_detections], tf.float32
                ),
                "detection_boxes_x1": tf.io.FixedLenFeature(
                    [args.max_detections], tf.float32
                ),
                "detection_boxes_y1": tf.io.FixedLenFeature(
                    [args.max_detections], tf.float32
                ),
            }
        )
    )

    with beam.Pipeline(options=beam_options) as p:
        with beam_impl.Context(tmp_sink):
            windowed_dataset = (
                (
                    p
                    | "Read ObjectDetectEvents"
                    >> beam.io.ReadFromPubSub(topic=args.topic)
                    | "Decompress and JSON Serialize" >> beam.Map(preprocess_features)
                    | "Add timestamps"
                    >> beam.Map(lambda x: beam.window.TimestampedValue(x, x["ts"]))
                )
                | "Add Window" >> beam.WindowInto(window.FixedWindows(args.window))
                | "Add Window Info" >> beam.ParDo(AddWindowingInfoFn())
                | "Group By Window" >> beam.GroupByKey()
                | "Write Windowed TFRecords"
                >> beam.ParDo(WriteWindowedTFRecords(args.sink, input_metadata.schema))
            )
