import logging
import argparse
import os
from typing import (
    Tuple,
    Any,
    Iterable,
    NamedTuple,
)
import subprocess

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from print_nanny_dataflow.encoders.types import (
    CreateVideoMessage,
)
from apache_beam.transforms.trigger import AfterCount, AfterWatermark, AfterAny


logger = logging.getLogger(__name__)


class FileSpec(NamedTuple):
    session: str
    gcs_prefix: str
    gcs_outfile: str


class RenderVideoTriggerAlert(beam.DoFn):
    def __init__(
        self,
        video_upload_path,
        api_url,
        api_token,
    ):
        self.api_url = api_url
        self.api_token = api_token
        self.video_upload_path = video_upload_path

    async def trigger_alert_async(self, session: str):
        rest_client = RestAPIClient(api_token=self.api_token, api_url=self.api_url)

        res = await rest_client.create_defect_alert(
            print_session=session,
        )

        logger.info(f"create_defect_alert res={res}")

    def trigger_alert(self, session: str):
        logger.warning(f"Sending alert for session={session}")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self.trigger_alert_async(session))

    def should_alert(self, session):
        logging.info(f"should_alert=True for session={session}")
        return True


class RenderVideo(beam.DoFn):
    def process(self, msg: CreateVideoMessage):
        path = os.path.dirname(__file__)
        script = os.path.join(path, "render_video.sh")
        val = subprocess.check_call(
            [
                script,
                "-i",
                msg.gcs_prefix_in,
                "-s",
                msg.session,
                "-o",
                msg.gcs_prefix_out,
            ]
        )
        yield msg.session, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument(
        "--render-video-topic",
        default="monitoring-video-render",
        help="Video rendering jobs will be output to this PubSub topic",
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
    beam_options = PipelineOptions(pipeline_args, streaming=True, runner=args.runner)

    input_topic_path = os.path.join(
        "projects", args.project, "topics", args.render_video_topic
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
            | "Decode bytes" >> beam.Map(lambda b: CreateVideoMessage.from_bytes(b))
            | "Run render_video.sh" >> beam.ParDo(RenderVideo())
            | beam.Map(print)
        )
