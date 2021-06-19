import logging
import argparse
import os
from typing import (
    Any,
    Iterable,
)
import subprocess

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from print_nanny_client.protobuf.alert_pb2 import VideoRenderRequest
from print_nanny_client.protobuf.monitoring_pb2 import AnnotatedMonitoringImage
from print_nanny_dataflow.transforms.io import TypedPathMixin
from print_nanny_dataflow.transforms.video import DecodeVideoRenderRequest
import print_nanny_dataflow

logger = logging.getLogger(__name__)


class RenderVideo(TypedPathMixin, beam.DoFn):
    def __init__(self, bucket: str, base_path: str):
        self.bucket = bucket
        self.base_path = base_path

    def process(self, msg: VideoRenderRequest) -> Iterable[bytes]:
        path = os.path.dirname(print_nanny_dataflow.__file__)
        script = os.path.join(path, "scripts", "render_video.sh")
        module = (
            f"{AnnotatedMonitoringImage.__module__}.{AnnotatedMonitoringImage.__name__}"
        )
        key = msg.metadata.print_session.session
        datesegment = msg.metadata.print_session.datesegment

        filename = "annotated_video.mp4"
        input_path = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=key,
            module=module,
            ext="jpg",
            window_type="FixedWindows",
            datesegment=datesegment,
        )
        output_path = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=key,
            module=module,
            ext="mp4",
            filename=filename,
            window_type="FixedWindows",
            datesegment=datesegment,
        )
        cdn_output_path = os.path.join("gs://", self.bucket, msg.cdn_output_path)

        val = subprocess.check_call(
            [
                script,
                "-i",
                input_path,
                "-o",
                output_path,
                "-c",
                cdn_output_path,
            ]
        )
        logger.info(val)
        yield msg.SerializeToString()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--loglevel", default="INFO")

    parser.add_argument(
        "--base-path",
        default="dataflow/sliding_window_health",
    )

    parser.add_argument(
        "--bucket",
        default="print-nanny-sandbox",
        help="GCS Bucket",
    )

    parser.add_argument(
        "--subscription",
        default="video-render",
        help="Video rendering job subscription",
    )

    parser.add_argument(
        "--topic",
        default="VideoRenderRequest",
        help="Video rendering job topic",
    )

    parser.add_argument(
        "--output-topic",
        default="alerts",
        help="Alert push jobs published to this PubSub topic",
    )

    parser.add_argument("--project", default="print-nanny-sandbox")

    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))

    beam_options = PipelineOptions(
        pipeline_args, streaming=True, runner=args.runner, project=args.project
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
            topic=os.path.join("projects", args.project, "topics", args.topic)
        )

    output_topic_path = os.path.join(
        "projects", args.project, "topics", args.output_topic
    )

    p = beam.Pipeline(options=beam_options)
    # TODO adjust window triggers
    tmp_file_spec_by_session = (
        p
        | f"Read from {input_kwargs}" >> beam.io.ReadFromPubSub(**input_kwargs)
        | beam.ParDo(DecodeVideoRenderRequest())
        | "Run render_video.sh"
        >> beam.ParDo(RenderVideo(base_path=args.base_path, bucket=args.bucket))
        | "Write to PubSub" >> beam.io.WriteToPubSub(output_topic_path)
    )
    result = p.run()
    if args.runner == "DirectRunner":
        result.wait_until_finish()
