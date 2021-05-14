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
from print_nanny_dataflow.encoders.types import (
    RenderVideoMessage,
)
from apache_beam.transforms.trigger import AfterCount, AfterWatermark, AfterAny
import print_nanny_dataflow

logger = logging.getLogger(__name__)


class RenderVideo(beam.DoFn):
    def process(self, msg: RenderVideoMessage) -> Iterable[bytes]:
        path = os.path.dirname(print_nanny_dataflow.__file__)
        script = os.path.join(path, "scripts", "render_video.sh")
        val = subprocess.check_call(
            [
                script,
                "-i",
                msg.gcs_input,
                "-s",
                msg.print_session,
                "-o",
                msg.gcs_output,
                "-c",
                msg.full_cdn_path(),
            ]
        )
        logger.info(val)
        yield bytes(msg.to_flatbuffer())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--loglevel", default="INFO")
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

    parser.add_argument("--api-token", help="Print Nanny API token")

    parser.add_argument(
        "--api-url", default="https://print-nanny.com/api", help="Print Nanny API url"
    )

    parser.add_argument(
        "--session-gap",
        default=300,
    )

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

    with beam.Pipeline(options=beam_options) as p:
        # TODO adjust window triggers
        alert_pipeline_trigger = AfterAny(
            AfterCount(1), AfterWatermark(late=AfterCount(1))
        )
        tmp_file_spec_by_session = (
            p
            | f"Read from {input_topic_path}"
            >> beam.io.ReadFromPubSub(topic=input_topic_path)
            | beam.WindowInto(
                beam.transforms.window.Sessions(args.session_gap),
                trigger=alert_pipeline_trigger,
                accumulation_mode=beam.transforms.trigger.AccumulationMode.DISCARDING,
            )
            | "Decode bytes" >> beam.Map(lambda b: RenderVideoMessage.from_bytes(b))
            | "Run render_video.sh" >> beam.ParDo(RenderVideo())
            | "Write to PubSub" >> beam.io.WriteToPubSub(output_topic_path)
        )
