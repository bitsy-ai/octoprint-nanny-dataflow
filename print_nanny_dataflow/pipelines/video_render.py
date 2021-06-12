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

from print_nanny_client.protobuf.monitoring_pb2 import VideoRenderRequest
import print_nanny_dataflow

logger = logging.getLogger(__name__)


class RenderVideo(beam.DoFn):
    def __init__(self, input_path: str, output_path: str, bucket: str):
        self.input_path = os.path.join("gs://", bucket, input_path)
        self.output_path = os.path.join("gs://", bucket, output_path)
        self.bucket = bucket

    def process(self, msg_bytes: bytes) -> Iterable[bytes]:
        msg = VideoRenderRequest()
        msg.ParseFromString(msg_bytes)
        path = os.path.dirname(print_nanny_dataflow.__file__)
        script = os.path.join(path, "scripts", "render_video.sh")
        output_path = self.output_path.format(print_session=msg.print_session)
        input_path = self.input_path.format(print_session=msg.print_session)
        cdn_output_path = os.path.join("gs://", self.bucket, msg.cdn_output_path)

        val = subprocess.check_call(
            [
                script,
                "-i",
                input_path,
                "-s",
                msg.print_session,
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
        "--input-path",
        default="dataflow/telemetry_event/{print_session}/NestedTelemetryEvent/jpg",
    )

    parser.add_argument(
        "--output-path",
        default="dataflow/telemetry_event/{print_session}/NestedTelemetryEvent/mp4",
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
        # | beam.Map(
        #     lambda b: VideoRenderRequest().ParseFromString(b)
        # ).with_output_types(VideoRenderRequest)
        | "Run render_video.sh"
        >> beam.ParDo(RenderVideo(args.input_path, args.output_path, args.bucket))
        | "Write to PubSub" >> beam.io.WriteToPubSub(output_topic_path)
    )
    result = p.run()
    if args.runner == "DirectRunner":
        result.wait_until_finish()
