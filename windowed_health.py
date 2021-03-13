#!/usr/bin/env python

from __future__ import absolute_import

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
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import PIL
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_serving.apis import predict_pb2

from encoders.tfrecord_example import ExampleProtoEncoder
from encoders.types import FlatTelemetryEvent
from clients.rest import RestAPIClient

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
                num_shards=1,
                shard_name_template="",
                file_name_suffix=".tfrecords.gz",
                coder=coder,
            )
        )

async def download_active_experiment_model(tmp_dir='.tmp/', model_artifact_id=1):

    tmp_artifacts_tarball = os.path.join(tmp_dir, 'artifacts.tar.gz')
    rest_client = RestAPIClient(api_token=args.api_token, api_url=args.api_url)

    model_artifacts = await rest_client.get_model_artifact(model_artifact_id)

    async with aiohttp.ClientSession() as session:
        logger.info(f"Downloading model artfiact tarball")
        async with session.get(model_artifacts.artifacts) as res:
            artifacts_gzipped = await res.read()
            with open(tmp_artifacts_tarball, 'wb+') as f:
                f.write(artifacts_gzipped)
            logger.info(
                f"Finished writing {tmp_artifacts_tarball}"
            )
    with tarfile.open(tmp_artifacts_tarball, "r:gz") as tar:
        tar.extractall(tmp_dir)
    logger.info(f"Finished extracting {tmp_artifacts_tarball}")

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
        num_detections = tflite_interpreter.get_tensor(
            output_details[3]["index"]
        )

        class_data = np.squeeze(class_data, axis=0).astype(np.int64) + 1
        box_data = np.squeeze(box_data, axis=0)
        score_data = np.squeeze(score_data, axis=0)
        num_detections = np.squeeze(num_detections, axis=0)

        ymin = [ b[0] for b in box_data]
        xmin = [ b[1] for b in box_data ]
        ymax = [ b[2] for b in box_data ]
        xmax = [ b[3] for b in box_data ]

        params = dict(
            scores=score_data,
            num_detections=num_detections,
            classes=class_data,
            boxes_ymin = ymin,
            boxes_xmin = xmin,
            boxes_ymax = ymax,
            boxes_xmax = xmax
        )

        defaults = element._asdict()
        defaults.update(params)
        return [FlatTelemetryEvent(
            **defaults
        )]

def is_print_healthy(elements):
    import pdb; pdb.set_trace()
    return elements

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
        default=60,
        help="Size of fixed streaming event window (in seconds)",
    )

    parser.add_argument(
        "--num-detections",
        default=40,
        help="Max number of bounding boxes output by nms operation",
    )

    parser.add_argument(
        "--api-token",
        help="Print Nanny API token"
    )

    parser.add_argument(
        "--api-url",
        default="https://print-nanny.com/api",
        help="Print Nanny API url"
    )

    parser.add_argument(
        "--model-version",
        default="tflite-print3d_20201101015829-2021-02-24T05:16:05.082500Z"
    )

    parser.add_argument(
        "--tmp-dir",
        default=".tmp/",
        help="Filesystem tmp directory"
    )

    parser.add_argument(
        "--batch-size",
        default=24,
    )


    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    topic_path = os.path.join("projects", args.project, "topics", args.topic)
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    # download model tarball
    asyncio.run(download_active_experiment_model())
    # load input shape from model metadata
    model_path = os.path.join(args.tmp_dir, args.model_version, 'model.tflite')
    model_metadata_path = os.path.join(args.tmp_dir, args.model_version, 'tflite_metadata.json')
    model_metadata = json.load(open(model_metadata_path, 'r'))
    input_shape = model_metadata['inputShape']
    # any batch size
    input_shape[0] = None

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

            feature_spec = FlatTelemetryEvent.feature_spec(args.num_detections)
            metadata = FlatTelemetryEvent.tfrecord_metadata(feature_spec)

            prediction_pipeline = (
                parsed_dataset
                | "Add Bounding Box Prediction" >>  beam.ParDo(PredictBoundingBoxes(model_path))
            )

            tfrecord_sink_pipeline = (
                prediction_pipeline
                | "Add Fixed Window" >> beam.WindowInto(window.FixedWindows(args.tfrecord_fixed_window))
                | "Add Fixed Window Info" >> beam.ParDo(AddWindowingInfoFn())
                | "Group by Window" >> beam.GroupByKey()
                | "Write TFRecords" >> beam.ParDo(WriteWindowedTFRecords(args.sink, metadata.schema))
                | "Print TFRecord paths" >> beam.Map(print)
            )

            health_pipeline = (
                prediction_pipeline
                | "Group by device_id" >> beam.GroupBy('device_id')
                | "Add Sliding Windows" >> beam.WindowInto(window.SlidingWindows(600, 10))
                | "Print health groups" >> beam.Map(is_print_healthy)
            )

            # feature_spec = FlatTelemetryEvent.feature_spec(args.num_detections)
            # metadata = FlatTelemetryEvent.tfrecord_metadata(feature_spec)

            # tfrecord_pipeline = (
            #     windowed_pipeline
            #     >> beam.ParDo(WriteWindowedTFRecords(args.sink, metadata.schema))
            # )



            # predict_pipeline = (
            #     windowed_pipeline
            #     | "Predict Bounding Boxes" >> beam.ParDo(PredictBoundingBoxes(model_path))

            # )
