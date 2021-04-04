import logging
import argparse
import os
import io
import imageio
from typing import (
    List,
    Tuple,
    Any,
    Iterable,
    Generator,
    Coroutine,
    Optional,
    NamedTuple,
)
import tempfile
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import PIL
import ffmpeg
import pandas as pd
import numpy as np
from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    CreateVideoMessage,
    AnnotatedImage,
    CATEGORY_INDEX,
    NestedWindowedHealthTrend,
    CreateVideoMessage,
    Metadata,
)
from print_nanny_dataflow.utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from print_nanny_dataflow.transforms.health import FilterAreaOfInterest

logger = logging.getLogger(__name__)


class RenderVideoTriggerAlert(beam.DoFn):
    def __init__(
        self,
        fixed_window_sink,
        video_upload_path,
        calibration_base_path,
        api_url,
        api_token,
        max_batches=3,
        category_index=CATEGORY_INDEX,
        score_threshold=0.5,
        max_boxes_to_draw=10,
    ):
        self.fixed_window_sink = fixed_window_sink
        self.max_batches = max_batches
        self.api_url = api_url
        self.api_token = api_token
        self.category_index = category_index
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.video_upload_path = video_upload_path
        self.calibration_base_path = calibration_base_path

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


class TmpFileSpec(NamedTuple):
    session: str
    gcs_prefix: str
    gcs_file_list: str
    local_filepattern: str
    local_tmp_dir: str
    gcs_outfile: str


class CreateTmpFileSpec(beam.DoFn):
    def __init__(self, video_upload_path, pipeline_options):
        self.video_upload_path = video_upload_path
        self.pipeline_options = pipeline_options

    def process(self, element: CreateVideoMessage):

        gcs_client = beam.io.gcp.gcsio.GcsIO()
        gcsfs = beam.io.gcp.gcsfilesystem.GCSFileSystem(self.pipeline_options)

        file_list = list(gcs_client.list_prefix(element.gcs_prefix).keys())
        file_list.sort()

        start_ts = gcsfs.split(file_list[0])[1].split(".")[0]
        end_ts = gcsfs.split(file_list[-1])[1].split(".")[0]

        gcs_outfile = os.path.join(
            self.video_upload_path, element.session, f"{start_ts}_{end_ts}.mp4"
        )

        local_tmp_dir = tempfile.mkdtemp(suffix=f"_{element.session}")
        local_filepattern = os.path.join(local_tmp_dir, "*.jpg")
        tmp_spec = TmpFileSpec(
            session=element.session,
            gcs_prefix=element.gcs_prefix,
            gcs_outfile=gcs_outfile,
            gcs_file_list=file_list,
            local_filepattern=local_filepattern,
            local_tmp_dir=local_tmp_dir,
        )
        logger.info(f"Created tmp file spec {tmp_spec}")
        yield element.session, tmp_spec
        # | beam.Map(lambda filename: beam.io.gcp.gcsfilesystem.GCSFileSystem)
        # with gcs_client.open(outpath, "wb") as f:
        #     f.write(img)

        # stream = ffmpeg.input(element.filepattern).output(outpath).run()

        # yield element.session, outpath


class DownloadTmpFileSpec(beam.DoFn):
    def __init__(self, pipeline_options):
        self.pipeline_options = pipeline_options

    def write(filename: str, data: bytes, tmp_dir: str):
        localfs = beam.io.localfilesystem.LocalFileSystem(self.pipeline_options)
        outfile = os.path.join(tmp_dir, filename)
        with localfs.create(outfile) as f:
            f.write(data)
        yield outfile

    def read(self, infile: str):
        gcsfs = beam.io.gcp.gcsfilesystem.GCSFileSystem(self.pipeline_options)
        filename = gcsfs.split(infile)
        with gcsfs.open(infile) as f:
            data = f.read()
        yield filename, data

    def process(self, element: Tuple[str, Iterable[TmpFileSpec]]):
        key, value = elements

        local_tmp_dir = value[0].local_tmp_dir

        yield key, (
            elements
            | beam.FlatMap(lambda x: x.gcs_file_list)
            | beam.Map(self.read)
            | beam.Map(
                lambda filename, data: self.write_to_tmp(filename, data, local_tmp_dir)
            )
        )


def download_tmp_filespec(key: str, spec: TmpFileSpec, pipeline_options):
    localfs = beam.io.localfilesystem.LocalFileSystem(self.pipeline_options)
    outfile = os.path.join(tmp_dir, filename)
    with localfs.create(outfile) as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--loglevel", default="INFO")

    parser.add_argument("--project", default="print-nanny-sandbox")

    parser.add_argument(
        "--render-video-topic",
        default="monitoring-video-render",
        help="Video rendering jobs will be output to this PubSub topic",
    )

    parser.add_argument(
        "--parquet-input",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/parquet",
        help="Read monitoring frames from GCS bucket",
    )

    parser.add_argument(
        "--calibration-base-path",
        default="gs://print-nanny-sandbox/uploads/device_calibration",
        help="Files will be output to this gcs bucket",
    )

    parser.add_argument("--api-token", help="Print Nanny API token")

    parser.add_argument(
        "--api-url", default="https://print-nanny.com/api", help="Print Nanny API url"
    )
    parser.add_argument(
        "--defect-video-upload-path",
        default="gs://print-nanny-sandbox/public/uploads/DefectAlert",
    )
    parser.add_argument(
        "--video-upload-path",
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/NestedTelemetryEvent/mp4",
    )

    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    input_topic_path = os.path.join(
        "projects", args.project, "topics", args.render_video_topic
    )

    with beam.Pipeline(options=beam_options) as p:
        tmp_local_files_by_session = (
            p
            | f"Read from {input_topic_path}"
            >> beam.io.ReadFromPubSub(topic=input_topic_path)
            | "Decode bytes" >> beam.Map(lambda b: CreateVideoMessage.from_bytes(b))
            | "Create a spec of remote/local file patterns"
            >> beam.ParDo(CreateTmpFileSpec(args.video_upload_path, beam_options))
            | beam.GroupByKey()
            | "Download images" >> beam.ParDo(DownloadTmpFileSpec(beam_options))
            | beam.Map(print)
        )

        # annotated_images = (
        #     frames_by_session
        #     | "Annotate images"
        #     >> beam.ParDo(
        #         AnnotateImage(
        #             args.calibration_base_path,
        #         )
        #     ) | beam.MapTuple(lambda session,x: print(session))
        # | "Write tmp jpg files" >> beam.MapTuple(lambda key, value: (key, values | beam.io.fileio.WriteToFiles(
        #     path="/tmp/annotated_frames",
        #     destination=lambda x: x.session,
        #     file_naming=beam.io.destination_prefix_naming(suffix=".jpg"),
        # )))
        # | "Render video from tmp files" >> beam.ParDo(RenderVideo(args.session_video_upload_path))
