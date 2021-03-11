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
import numpy as np
import nptyping as npt

import apache_beam as beam
from apache_beam import window
import typing
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.beam import impl as beam_impl

from print_nanny_client.telemetry_event import TelemetryEvent

logger = logging.getLogger(__name__)
# @todo Flatbuffer -> NamedTuple codegen?
class FlatTelemetryEvent(typing.NamedTuple):
    """
    flattened data structures for
    tensorflow_transform.tf_metadata.schema_utils.schema_from_feature_spec
    """

    ts: int
    version: str
    event_type: int
    event_data_type: int

    # Image
    image_data: bytes
    image_width: npt.Float32
    image_height: npt.Float32

    # Metadata
    user_id: npt.Float32
    device_id: npt.Float32
    device_cloudiot_id: npt.Float32

    # BoundingBoxes
    scores: npt.NDArray[npt.Float32] = None
    classes: npt.NDArray[npt.Int32] = None
    num_detections: npt.NDArray[npt.Int32] = None
    boxes_ymin: npt.NDArray[npt.Float32] = None
    boxes_xmin: npt.NDArray[npt.Float32] = None
    boxes_ymax: npt.NDArray[npt.Float32] = None
    boxes_xmax: npt.NDArray[npt.Float32] = None

    @staticmethod
    def feature_spec(num_detections):
        return schema_utils.schema_from_feature_spec(
            {
                "ts": tf.io.FixedLenFeature([], tf.float32),
                "version": tf.io.FixedLenFeature([], tf.string),
                "event_type": tf.io.FixedLenFeature([], tf.int64),
                "event_data_type": tf.io.FixedLenFeature([], tf.int64),
                "image_data": tf.io.FixedLenFeature([], tf.string),
                "image_height": tf.io.FixedLenFeature([], tf.int64),
                "image_width": tf.io.FixedLenFeature([], tf.int64),
                "user_id": tf.io.FixedLenFeature([], tf.int64),
                "device_id": tf.io.FixedLenFeature([], tf.int64),
                "device_cloudiot_id": tf.io.FixedLenFeature([], tf.int64),
                "num_detections": tf.io.FixedLenFeature([], tf.float32),
                "detection_classes": tf.io.FixedLenFeature([num_detections], tf.int64),
                "detection_scores": tf.io.FixedLenFeature([num_detections], tf.float32),
                "original_image": tf.io.FixedLenFeature([], tf.string),
                "boxes_ymin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_xmin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymax": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_xmax": tf.io.FixedLenFeature([num_detections], tf.float32),
            }
        )

    @staticmethod
    def tfrecord_metadata(feature_spec):
        return dataset_metadata.DatasetMetadata(feature_spec)

    @classmethod
    def from_flatbuffer(cls, input_bytes):

        msg = TelemetryEvent.TelemetryEvent.GetRootAsTelemetryEvent(input_bytes, 0)
        obj = TelemetryEvent.TelemetryEventT.InitFromObj(msg)

        scores = None
        num_detections = None
        classes = None
        boxes_ymin = None
        boxes_xmin = None
        boxes_ymax = None
        boxes_xmax = None

        if obj.eventData.boundingBoxes is not None:
            scores = obj.eventData.boundingBoxes.scores
            classes = obj.eventData.boundingBoxes.classes
            num_detections = obj.eventData.boundingBoxes.numDetections
            boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = [
                np.array([b.ymin, b.xmin, x.ymax, b.xmax])
                for b in obj.eventData.boundingBoxes
            ]
        return cls(
            ts=obj.metadata.ts,
            version=obj.version,
            event_type=obj.eventType,
            event_data_type=obj.eventDataType,
            image_height=obj.eventData.image.height,
            image_width=obj.eventData.image.width,
            image_data=obj.eventData.image.data,
            user_id=obj.metadata.userId,
            device_id=obj.metadata.deviceId,
            device_cloudiot_id=obj.metadata.deviceCloudiotId,
            scores=scores,
            classes=classes,
            num_detections=num_detections,
            boxes_ymin=boxes_ymin,
            boxes_xmin=boxes_xmin,
            boxes_ymax=boxes_ymax,
            boxes_xmax=boxes_xmax,
        )


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
        logger.info(f"Writing {output}")
        yield (
            elements
            | beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=output,
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
