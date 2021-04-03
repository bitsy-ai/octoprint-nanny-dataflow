import logging
import argparse
import os
from typing import List, Tuple, Any, Iterable, Generator, Coroutine, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import pandas as pd
import numpy as np
from encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    PendingAlert,
    AnnotatedImage,
)

from statistics.types import CATEGORY_INDEX


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

    def annotate_image(self, event: NestedTelemetryEvent):
        image_np = np.array(PIL.Image.open(io.BytesIO(event.image_data)))
        if event.calibration is None:
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes,
                event.detection_classes,
                event.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
            )
        else:
            detection_boundary_mask = self.calibration["mask"]
            ignored_mask = np.invert(detection_boundary_mask)
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes,
                event.detection_classes,
                event.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
                detection_boundary_mask=detection_boundary_mask,
                detection_box_ignored=ignored_mask,
            )
        return event.session, NestedTelemetryEvent(
            annotated_image_data=annotated_image_data, **event.to_dict()
        )

    def write_video(
        self,
        keyed_elements: Tuple[str, Iterable[NestedTelemetryEvent]],
        window_start: int,
        window_end: int,
    ) -> str:

        output_path = os.path.join(
            self.video_upload_path, session, f"{window_start}_{window_end}.mp4"
        )
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        with gcs_client.open(output_path, "wb+") as f:
            writer = imageio.get_writer(f, mode="I")
            for i, value in df["annotated_image"].iteritems():
                writer.append_data(value)
            writer.close()
        yield output_path

    def process(
        self,
        keyed_elements: Tuple[str, Iterable[NestedTelemetryEvent]],
        window=beam.DoFn.WindowParam,
    ):
        session, elements = keyed_elements
        window_start = int(window.start)
        window_end = int(window.end)
        yield (
            elements
            | "Filter area of interest and detections above threshold"
            | "Group by session"
            >> beam.GroupBy("session")
            >> beam.ParDo(
                FilterAreaOfInterest(
                    self.calibration_base_path,
                    score_threshold=self.score_threshold,
                    output_calibration=True,
                )
            )
            | "Transform to DataFrame"
            >> DataframeTransform(lambda df: df.sort_values("ts", inplace=True))
            | "Add annotated image keyed by session"
            >> beam.Map(self.annotate_image).with_output_types(NestedTelemetryEvent)
            | "Render and write video" >> beam.CombineValues(self.write_video)
        )


class RenderVideo(beam.DoFn):
    def __init__(self, video_upload_path):
        self.video_upload_path = video_upload_path

    def process(self, keyed_elements: Iterable[str, Iterable[AnnotatedImage]]):

        key, values = keyed_elements

        values = values.sort(key=lambda x: x.ts)


class AnnotateImage(beam.DoFn):
    def __init__(
        self,
        calibration_base_path,
        category_index=CATEGORY_INDEX,
        score_threshold=0.5,
        max_boxes_to_draw=10,
    ):
        self.category_index = category_index
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.video_upload_path = video_upload_path
        self.calibration_base_path = calibration_base_path

    def annotate_image(self, event: NestedTelemetryEvent) -> AnnotatedImage:
        image_np = np.array(PIL.Image.open(io.BytesIO(event.image_data)))
        if event.calibration is None:
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes,
                event.detection_classes,
                event.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
            )
        else:
            detection_boundary_mask = self.calibration["mask"]
            ignored_mask = np.invert(detection_boundary_mask)
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes,
                event.detection_classes,
                event.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
                detection_boundary_mask=detection_boundary_mask,
                detection_box_ignored=ignored_mask,
            )
        return event.session, AnnotatedImage(
            metadata=event.metadata,
            ts=event.ts,
            session=event.session,
            annotated_image_data=annotated_image_data,
        )

    # def write_video(
    #     self,
    #     keyed_elements: Tuple[str, Iterable[NestedTelemetryEvent]],
    #     window_start: int,
    #     window_end: int,
    # ) -> str:

    #     output_path = os.path.join(
    #         self.video_upload_path, session, f"{window_start}_{window_end}.mp4"
    #     )
    #     gcs_client = beam.io.gcp.gcsio.GcsIO()

    #     with gcs_client.open(output_path, "wb+") as f:
    #         writer = imageio.get_writer(f, mode="I")
    #         for i, value in df["annotated_image"].iteritems():
    #             writer.append_data(value)
    #         writer.close()
    #     yield output_path

    def process(
        self,
        keyed_elements: Tuple[str, Iterable[NestedTelemetryEvent]],
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Tuple[str, Iterable[AnnotatedImage]]]:
        key, elements = keyed_elements
        window_start = int(window.start)
        window_end = int(window.end)
        yield key, (
            elements
            | "Filter area of interest and detections above threshold"
            | "Group by session"
            >> beam.GroupBy("session")
            >> beam.ParDo(
                FilterAreaOfInterest(
                    self.calibration_base_path,
                    score_threshold=self.score_threshold,
                )
            )
            | "Add annotated image keyed by session"
            >> beam.Map(self.annotate_image).with_output_types(NestedTelemetryEvent)
        )


class WriteTmpJpgs(beam.DoFn):
    def process(
        self, keyed_elements: Tuple[str, Iterable[AnnotatedImage]]
    ) -> Iterable[Tuple[str, Iterable[str]]]:
        key, values = keyed_elements
        yield key, beam.io.fileio.WriteToFiles(
            path="/tmp/annotated_frames",
            destination=lambda x: x.session,
            file_naming=beam.io.destination_prefix_naming(suffix=".jpg"),
        )


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
        default="gs://print-nanny-sandbox/dataflow/telemetry_event/fixed_window/parquet",
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
        "--video-upload-path",
        default="gs://print-nanny-sandbox/public/uploads/defect_alert",
    )

    parser.add_argument("--runner", default="DataflowRunner")

    args, pipeline_args = parser.parse_known_args()

    run_pipeline(args, pipeline_args)

    logging.basicConfig(level=getattr(logging, args.loglevel))
    beam_options = PipelineOptions(
        pipeline_args, save_main_session=True, streaming=True, runner=args.runner
    )

    input_topic_path = os.path.join(
        "projects", args.project, "topics", args.render_video_topic
    )

    with beam.Pipeline(options=beam_options) as p:
        frames_by_session = (
            p
            | f"Read from {input_topic_path}"
            >> beam.io.ReadFromPubSub(topic=input_topic_path)
            | "Decode bytes"
            >> beam.Map(lambda b: NestedWindowedHealthTrend.from_bytes(b))
            | beam.Map(lambda x: x.filepattern)
            | beam.io.ReadAllFromParquet().with_output_types(NestedTelemetryEvent)
            | beam.GroupBy(lambda x: x["session"])
        )

        annotated_images = (
            frames_by_session
            | "Annotate images"
            >> beam.ParDo(
                AnnotateImage(
                    args.calibration_base_path,
                )
            )
            | beam.GroupByKey()
            | "Write jpgs to disk" >> beam.ParDo(WriteTmpJpgs)
            | beam.GroupByKey()
            | "Create mp4" >> beam.ParDo(RenderVideo(args.video_upload_path))
        )

        rendered_videos = annotated_images | beam.Map(lambda x: x.annotated_image_data)
