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
from print_nanny_client.protobuf.monitoring_pb2 import MonitoringImage
from print_nanny_dataflow.transforms.io import TypedPathMixin
import print_nanny_dataflow

logger = logging.getLogger(__name__)


class RenderVideo(TypedPathMixin, beam.DoFn):
    def __init__(self, bucket: str, base_path: str):
        self.bucket = bucket
        self.base_path = base_path

    def process(self, msg: VideoRenderRequest) -> Iterable[bytes]:
        path = os.path.dirname(print_nanny_dataflow.__file__)
        script = os.path.join(path, "scripts", "render_video.sh")
        module = f"{MonitoringImage.__module__}.{MonitoringImage.__name__}"

        filename = "annotated_video.mp4"
        input_path = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=msg.print_session.session,
            datesegment=msg.print_session.datesegment,
            module=module,
            ext="jpg",
            window_type="fixed",
        )
        output_path = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=msg.print_session.session,
            datesegment=msg.print_session.datesegment,
            module=module,
            ext="mp4",
            filename=filename,
            window_type="fixed",
        )
        cdn_output_path = os.path.join("gs://", self.bucket, msg.cdn_output_path)

        val = subprocess.check_call(
            [
                script,
                "-i",
                input_path,
                "-s",
                msg.print_session.session,
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
        default="dataflow/monitoring",
    )

    parser.add_argument(
        "--bucket",
        default="print-nanny-sandbox",
        help="GCS Bucket",
    )

    parser.add_argument(
        "--input-topic",
        default="monitoring-video-render",
        help="Video rendering jobs published to this PubSub topic",
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

    input_topic_path = os.path.join(
        "projects", args.project, "topics", args.input_topic
    )

    output_topic_path = os.path.join(
        "projects", args.project, "topics", args.output_topic
    )

    p = beam.Pipeline(options=beam_options)
    # TODO adjust window triggers
    tmp_file_spec_by_session = (
        p
        | f"Read from {input_topic_path}"
        >> beam.io.ReadFromPubSub(topic=input_topic_path)
        | beam.Map(lambda b: VideoRenderRequest().ParseFromString(b)).with_output_types(
            VideoRenderRequest
        )
        | "Run render_video.sh"
        >> beam.ParDo(RenderVideo(base_path=args.base_pth, bucket=args.bucket))
        | "Write to PubSub" >> beam.io.WriteToPubSub(output_topic_path)
    )
    result = p.run()
    if args.runner == "DirectRunner":
        result.wait_until_finish()
