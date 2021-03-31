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

from apache_beam.transforms import trigger
from tensorflow_transform.tf_metadata import dataset_metadata

from print_nanny_dataflow.transforms.io import (
    WriteWindowedTFRecord,
    WriteWindowedParquet,
)
from print_nanny_dataflow.statistics.health import (
    health_score_trend_polynomial_v1,
    CATEGORY_INDEX,
)

from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    PendingAlert,
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


class ExplodeWindowedHealthRecord(beam.DoFn):
    def process(
        self,
        keyed_element: Tuple[str, Iterable[NestedTelemetryEvent]],
        window=beam.DoFn.WindowParam,
    ) -> Tuple[str, Iterable[WindowedHealthRecord]]:
        session, elements = keyed_element
        window_start = int(window.start)
        window_end = int(window.end)

        # save frame for later rendering
        yield session, (
            elements
            | beam.FlatMap(
                lambda event: [
                    WindowedHealthRecord(
                        ts=event.ts,
                        session=event.session,
                        client_version=event.client_version,
                        user_id=event.user_id,
                        device_id=event.device_id,
                        device_cloudiot_id=event.device_cloudiot_id,
                        detection_score=event.detection_scores[i],
                        detection_class=event.detection_classes[i],
                        window_start=window_start,
                        window_end=window_end,
                        health_multiplier=CATEGORY_INDEX[event.detection_classes[i]][
                            "health_weight"
                        ],
                        health_score=CATEGORY_INDEX[event.detection_classes[i]][
                            "health_weight"
                        ]
                        * event.detection_scores[i],
                    )
                    for i in range(0, event.num_detections)
                ]
            )
        )


class CheckpointHealthScoreTrend(beam.DoFn):
    def __init__(
        self,
        checkpoint_sink,
        parquet_sink,
        render_video_topic,
        window_size,
        window_period,
        api_url,
        api_token,
        health_threshold=3,
        polyfit_degree=1,
        warmup=20,
    ):
        self.health_threshold = health_threshold
        self.checkpoint_sink = checkpoint_sink
        self.parquet_sink = parquet_sink
        self.render_video_topic = render_video_topic
        self.polyfit_degree = polyfit_degree
        self.window_size = window_size
        self.window_period = window_period
        self.warmup = warmup
        self.api_url = api_url
        self.api_token = api_token

    async def should_alert_async(self, session: str) -> bool:
        rest_client = RestAPIClient(api_token=self.api_token, api_url=self.api_url)

        print_session = await rest_client.get_print_session(
            print_session=session,
        )
        return print_session.supress_alerts is False

    def should_alert(
        self, session: str, trend: np.polynomial.polynomial.Polynomial
    ) -> bool:
        slope, intercept = tuple(trend)
        if slope < 0:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.should_alert_async(session))
        return False

    def process(
        self,
        keyed_elements: Tuple[str, Iterable[WindowedHealthRecord]],
        window=beam.DoFn.WindowParam,
    ) -> Iterable[bytes]:
        session, windowed_health_records = keyed_elements

        window_start = int(window.start)
        window_end = int(window.end)
        output_path = os.path.join(
            self.checkpoint_sink, session, f"{window_start}_{window_end}.parquet"
        )
        df = (
            pd.DataFrame(data=windowed_health_records)
            .sort_values("ts")
            .set_index(["ts"])
        )
        df.to_parquet(output_path, engine="pyarrow")
        n_frames = len(df.index.unique())
        window_start = int(window.start)
        window_end = int(window.end)
        if n_frames <= self.warmup:
            logger.warning(
                f"Ignoring CalcHealthScoreTrend called with n_frames={n_frames} warmup={self.warmup} session={session} window=({window_start}_{window_end})"
            )
            return

        trend = health_score_trend_polynomial_v1(df, degree=self.polyfit_degree)

        should_alert = self.should_alert(session, trend)
        logger.info(f"should_alert={should_alert} for trend={trend}")
        if should_alert:
            file_pattern = os.path.join(self.parquet_sink, session, "*")
            sample_event = windowed_health_records[0]

            pending_alert = PendingAlert(
                session=session,
                client_version=sample_event.client_version,
                user_id=sample_event.user_id,
                device_id=sample_event.device_id,
                device_cloudiot_id=sample_event.device_cloudiot_id,
                window_start=sample_event.window_start,
                window_end=sample_event.window_end,
                file_pattern=file_pattern,
            )
            yield pending_alert.to_bytes()


def predict_bounding_boxes(element, model_path):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
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

    ymin, xmin, ymax, xmax = box_data.T

    params = dict(
        detection_scores=score_data,
        num_detections=int(num_detections),
        detection_classes=class_data,
        boxes_ymin=ymin,
        boxes_xmin=xmin,
        boxes_ymax=ymax,
        boxes_xmax=xmax,
    )
    defaults = element.to_dict()
    defaults.update(params)
    return NestedTelemetryEvent(**defaults)


class FilterAreaOfInterest(beam.DoFn):
    def __init__(
        self,
        calibration_base_path: str,
        score_threshold: float = 0.5,
        calibration_filename: str = "calibration.json",
    ):
        self.calibration_base_path = calibration_base_path
        self.score_threshold = score_threshold
        self.calibration_filename = calibration_filename

    def load_calibration(
        self, event: NestedTelemetryEvent
    ) -> Optional[DeviceCalibration]:
        gcs_client = beam.io.gcp.gcsio.GcsIO()
        device_id = event.device_id
        device_calibration_path = os.path.join(
            self.calibration_base_path, str(device_id), self.calibration_filename
        )
        if gcs_client.exists(device_calibration_path):
            with gcs_client.open(device_calibration_path, "r") as f:
                logger.info(
                    f"Loading device calibration from {device_calibration_path}"
                )
                calibration_json = json.load(f)

            return DeviceCalibration(**calibration_json)

    def process(
        self,
        elements: Iterable[NestedTelemetryEvent] = beam.DoFn.ElementParam,
        key=beam.DoFn.KeyParam,
    ) -> Tuple[str, Iterable[NestedTelemetryEvent]]:
        # session, elements = keyed_element

        calibration = self.load_calibration(elements[0])

        if calibration:
            yield key, elements | beam.Map(
                lambda event: calibration.filter_event(event)
            )
        else:
            yield key, elements


def run_pipeline(args, pipeline_args):
    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    input_topic_path = os.path.join("projects", args.project, "topics", args.topic)

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
            # .with_output_types(
            #     Tuple[str, NestedTelemetryEvent]
            # )
        )

        tf_record_schema = NestedTelemetryEvent.tfrecord_schema(args.num_detections)
        pa_schema = NestedTelemetryEvent.pyarrow_schema(args.num_detections)

        fixed_window_view = (
            parsed_dataset_by_session
            | f"Add fixed window"
            >> beam.WindowInto(
                beam.transforms.window.FixedWindows(args.health_window_period)
            )
            | "Group FixedWindow NestedTelemetryEvent by key" >> beam.GroupByKey()
        )

        tfrecord_sink_pipeline = fixed_window_view | "Write TFRecords" >> beam.ParDo(
            WriteWindowedTFRecord(args.tfrecord_sink, tf_record_schema)
        )

        parquet_sink_pipeline = fixed_window_view | "Write Parquet" >> beam.ParDo(
            WriteWindowedParquet(args.parquet_sink, pa_schema)
            # | "Convert annotated images to MP4"
        )

        # sliding_window_view = (
        #     parsed_dataset_by_session
        #     | "Add sliding window"
        #     >> beam.WindowInto(
        #         beam.transforms.window.SlidingWindows(
        #             args.health_window_size, args.health_window_period
        #         ),
        #         accumulation_mode=beam.transforms.trigger.AccumulationMode.ACCUMULATING,
        #     )
        #     | "Group windowed NestedTelemeryEvent by session" >> beam.GroupByKey()
        # )

        # output_topic = os.path.join(
        #     "projects", args.project, "topics", args.render_video_topic
        # )
        # should_alert_per_session_windowed = (
        #     sliding_window_view
        #     | "Drop detections below confidence threshold & outside of calibration area of interest"
        #     >> beam.ParDo(
        #         FilterAreaOfInterest(args.calibration_base_path, score_threshold=0.5)
        #     ).with_output_types(Tuple[str, Iterable[NestedTelemetryEvent]])
        #     | "Flatten remaining observations in NestedTelemetryEvent"
        #     >> beam.ParDo(ExplodeWindowedHealthRecord()).with_output_types(
        #         Tuple[str, Iterable[WindowedHealthRecord]]
        #     )
        #     | "Calculate health score trend and publish alerts"
        #     >> beam.ParDo(
        #         CheckpointHealthScoreTrend(
        #             args.health_checkpoint_sink,
        #             args.parquet_sink,
        #             args.render_video_topic,
        #             args.health_window_size,
        #             args.health_window_period,
        #             args.api_url,
        #             args.api_token,
        #         )
        #     )
        #     | "Publish PendingAlert messages to PubSub"
        #     >> beam.io.WriteToPubSub(output_topic)
        # )


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
        "--tfrecord-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/tfrecords",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--parquet-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/parquet",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument(
        "--health-checkpoint-sink",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/sliding_window/health",
        help="Files will be output to this gcs bucket",
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
