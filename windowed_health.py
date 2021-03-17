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
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
import PIL
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_serving.apis import predict_pb2

from encoders.tfrecord_example import ExampleProtoEncoder
from encoders.types import FlatTelemetryEvent
from clients.rest import RestAPIClient
from beam_nuggets.io import relational_db

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

    def process(self, elements):
        coder = ExampleProtoEncoder(self.schema)
        ts = datetime.now().timestamp()
        output =   os.path.join(self.outdir, str(ts))
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

class TelemetryEventStatefulFn(beam.DoFn):

    # PREVIOUS_STATE= beam.transforms.userstate.BagStateSpec('previous_state', beam.coders.coders.Coder())
    
    UNHEALTHY_STATE = beam.transforms.userstate.CombiningValueStateSpec('unhealthy_state', beam.coders.coders.Coder(), combine_fn=sum)
    MODEL_STATE = beam.transforms.userstate.BagStateSpec('model_state', beam.coders.coders.Coder())

    def process(self, 
        telemetry_event: FlatTelemetryEvent, 
        model_state=beam.DoFn.StateParam(MODEL_STATE)
    ):
        device_id, event = telemetry_event

        model = model_state.read()
        if model is None:
            model = PolyfitHealthModel()
        
        model.add(telemetry_event)


        new_prediction = model.predict(event)
        model_state.add(event)

        previous_pred_state.clear()
        previous_pred_state.add(new_prediction)
        yield (user, new_prediction)


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
        defaults = element.asdict()
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

    parser.add_argument(
        "--postgres-host",
        default="localhost",
        help="Postgres host"
    )

    parser.add_argument(
        "--postgres-port",
        default=5432,
        help="Postgres port"
    )

    parser.add_argument(
        "--postgres-user",
        default="debug"
    )

    parser.add_argument(
        "--postgres-pass",
        default="debug"
    )

    parser.add_argument(
        "--postgres-db",
        default="print_nanny"
    )


    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    topic_path = os.path.join("projects", args.project, "topics", args.topic)
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    # download model tarball
    if args.runner == "DataflowRunner":
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

            box_annotations = (
                parsed_dataset
                | "Add Bounding Box Annotations" >>  beam.ParDo(PredictBoundingBoxes(model_path))
            )

            tfrecord_sink_pipeline = (
                box_annotations
                | "Batch TFRecords" >> beam.transforms.util.BatchElements(min_batch_size=args.batch_size, max_batch_size=args.batch_size)
                | "Write TFRecords" >> beam.ParDo(WriteWindowedTFRecords(args.sink, metadata.schema))
                | "Print TFRecord paths" >> beam.Map(print)
            )

            health_models_by_device_id = (
                box_annotations
                | "Add Fixed Window" >> beam.WindowInto(window.FixedWindows(30))
                | "Group by session" >> beam.GroupBy("session")
                | "Explode classes/scores with FlatMap" >> beam.FlatMap(lambda b: [Box(
                    detection_score=b.detection_scores[i],
                    detection_class=b.detection_classes[i],
                    ymin=b[1].ymin[i],
                    xmin=b[1].xmin[i],
                    ymax=b[1].ymax[i],
                    xmax=b[1].xmax[i]
                ) for i in range(0, b[1].num_detections) ])

                # | "Combine into health model" >> beam.core.CombinePerKey(TelemetryEventStatefulFn()))
            )

            # @todo join device calibration
            # source_config = relational_db.SourceConfiguration(
            #     drivername='postgresql+psycopg2',
            #     host=args.postgres_host=,
            #     port=args.postgres_port,
            #     username=args.postgres_user,
            #     password=args.postgres_pass,
            #     database=args.postgres_db,
            #     create_if_missing=False,
            # )
            # table_config = relational_db.TableConfiguration(
            #     name='client_events_monitoringframeevent',
            #     create_if_missing=False
            # )

            # image_sink_pipeline = (
            #     prediction_pipeline
            #     | "Write image bytes to gcs" >> 
            # )

            # postgres_sink_pipeline = (
            #     prediction_pipeline
            #     | "Reserialize as dict" >> beam.Map(lambda e: print_nanny_client.MonitoringFrameEvent(
            #         device=e.device_id,
            #         user=e.user_id,
            #         ts=e.ts,
            #         session=e.session,

            #     )

            #     )
            #     | "Save to Postgres" >> relational_db.Write(
            #         source_config=source_config,
            #         table_config=table_config
            #     )
            # )