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
from timeit import timeit
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
from apache_beam.transforms.trigger import (
    AfterCount,
    AfterWatermark,
    AfterAny
)
from print_nanny_dataflow.utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from print_nanny_dataflow.transforms.health import FilterAreaOfInterest

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
    def process(self, elements: Iterable[Tuple[str, TmpFileSpec]]):
        key, values = elements

        local_filepattern = values[0]

class CreateTmpFileSpec(beam.DoFn):
    def __init__(self, video_upload_path, pipeline_options):
        self.video_upload_path = video_upload_path
        self.pipeline_options = pipeline_options

    def process(self, element: CreateVideoMessage) -> Iterable[Tuple[str, TmpFileSpec]]:

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
        local_infilepattern = os.path.join(local_tmp_dir, "*.jpg")
        local_outfile = os.path.join(local_tmp_dir, f"{element.session}.mp4")
        tmp_spec = TmpFileSpec(
            session=element.session,
            gcs_prefix=element.gcs_prefix,
            gcs_outfile=gcs_outfile,
            gcs_file_list=file_list,
            local_infilepattern=local_infilepattern,
            local_tmp_dir=local_tmp_dir,
            local_outfile=local_outfile
        )
        yield element.session, tmp_spec
        # | beam.Map(lambda filename: beam.io.gcp.gcsfilesystem.GCSFileSystem)
        # with gcs_client.open(outpath, "wb") as f:
        #     f.write(img)

        # stream = ffmpeg.input(element.filepattern).output(outpath).run()

        # yield element.session, outpath


class DownloadTmpFileSpec(beam.DoFn):
    def __init__(self, pipeline_options):
        self.pipeline_options = pipeline_options

    # def write(filename: str, data: bytes, tmp_dir: str) -> Iterable[Tuple[str, TmpFileSpec]]:
    #     localfs = beam.io.localfilesystem.LocalFileSystem(self.pipeline_options)
    #     outfile = os.path.join(tmp_dir, filename)
    #     with localfs.create(outfile) as f:
    #         f.write(data)
    #     return (spec.session, outfile)

    def local_copy(self, key: str, infile: str, tmp_dir: str) -> Iterable[Tuple[str, bytes]]:
        gcsfs = beam.io.gcp.gcsfilesystem.GCSFileSystem(self.pipeline_options)
        localfs = beam.io.localfilesystem.LocalFileSystem(self.pipeline_options)

        head, filename = gcsfs.split(infile)
        outfile = os.path.join(tmp_dir, filename)
        with gcsfs.open(infile) as f:
            data = f.read()
        with localfs.create(outfile) as f:
            f.write(data)
        return (key, outfile)
    
    def create_video(self, key: str, infilepattern:str, outfile: str):
        logging.info(f"Begin rendering video for session={key} outfile={outfile}")
        timing = timeit(lambda: ffmpeg.input(infilepattern, pattern_type="glob").output(outfile).run())
        logging.info(f"Finished rendering video session={key} outfile={outfile} in seconds={timing}")
        return key, outfile
    
    def upload(self, infile: str, outfile: str):
        gcsfs = beam.io.gcp.gcsfilesystem.GCSFileSystem(self.pipeline_options)
        localfs = beam.io.localfilesystem.LocalFileSystem(self.pipeline_options)

        with localfs.open(infile) as local_f:
            data = local_f.read()
        with gcsfs.create(outfile) as remote_f:
            remote_f.write(data)
        logger.info(f"Finished uploading {outfile}")
        return outfile

    def process(self, element: Tuple[str, Iterable[TmpFileSpec]]):
        key, values = element

        local_tmp_dir = values[0].local_tmp_dir
        local_infilepattern = values[0].local_infilepattern
        local_outfile = values[0].local_outfile
        gcs_outfile = values[0].gcs_outfile
        logger.info(f"Writing session={key} tmp files to {local_tmp_dir}")

        yield key, (
            values
            | beam.FlatMap(lambda x: x.gcs_file_list)
            | beam.Map(
                lambda filename: self.local_copy(key, filename, local_tmp_dir)
            )
            | beam.GroupByKey() | beam.MapTuple(lambda key, outfiles: self.create_video(key, local_infilepattern, local_outfile))
            | beam.MapTuple(lambda key, _: self.upload(local_outfile, gcs_outfile))
        )

class RenderVideo(beam.ParDo):
    def process(self, spec: TmpFileSpec):


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

    parser.add_argument(
        "--session-gap",
        default=300,
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
        # TODO adjust window 
        alert_pipeline_trigger = AfterAny(
            AfterCount(1),
            AfterWatermark(
            late=AfterCount(1)
        ))
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
            # | "Create a spec of remote/local file patterns"
            # >> beam.ParDo(CreateTmpFileSpec(args.video_upload_path, beam_options))
            # | beam.GroupByKey().with_output_types(Tuple[str, TmpFileSpec])
            # | "Download images" >> beam.ParDo(DownloadTmpFileSpec(beam_options))
        )

        # video_render_pipeline = (
        #     tmp_file_spec_by_session
        #     | beam.FlatMapTuple(lambda key, spec: spec.gcs_file_list | beam.Map(lambda f: local_copy(key, spec, f, beam_options))) | beam.GroupByKey() | beam.MapTuple(lambda key, specs: (key, specs[0]))
        #     | beam.MapTuple(create_video)
        #     | beam.MapTuple(lambda key, spec: upload(key, spec, beam_options))
        # )


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
