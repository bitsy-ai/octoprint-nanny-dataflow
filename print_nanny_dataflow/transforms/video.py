from typing import Tuple, Iterable
import logging
import io
import os
import apache_beam as beam
from print_nanny_dataflow.utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from apache_beam.options.pipeline_options import PipelineOptions
from print_nanny_dataflow.coders.types import (
    CATEGORY_INDEX,
)
from print_nanny_dataflow.transforms.io import TypedPathMixin
from print_nanny_client.protobuf.monitoring_pb2 import AnnotatedMonitoringImage
from print_nanny_client.protobuf.alert_pb2 import VideoRenderRequest
import PIL
import numpy as np


logger = logging.getLogger(__name__)


@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(VideoRenderRequest)
class DecodeVideoRenderRequest(beam.DoFn):
    def process(self, element: bytes) -> Iterable[VideoRenderRequest]:
        parsed = VideoRenderRequest()
        parsed.ParseFromString(element)
        yield parsed


@beam.typehints.with_input_types(AnnotatedMonitoringImage)
@beam.typehints.with_output_types(bytes)
class EncodeVideoRenderRequest(beam.DoFn):
    def process(self, element: AnnotatedMonitoringImage) -> Iterable[bytes]:
        datesegment = element.monitoring_image.metadata.print_session.datesegment
        key = element.monitoring_image.metadata.print_session.session
        cdn_output_path = os.path.join(
            "media/uploads/PrintSessionAlert", datesegment, key
        )
        msg = VideoRenderRequest(
            cdn_output_path=cdn_output_path, metadata=element.monitoring_image.metadata
        )
        yield msg.SerializeToString()


@beam.typehints.with_input_types(AnnotatedMonitoringImage)
@beam.typehints.with_output_types(Tuple[str, str])
class WriteAnnotatedImage(TypedPathMixin, beam.DoFn):
    def __init__(
        self,
        base_path: str,
        bucket: str,
        pipeline_options: PipelineOptions,
        category_index=CATEGORY_INDEX,
        score_threshold=0.5,
        max_boxes_to_draw=10,
        ext="jpg",
        module=(
            f"{AnnotatedMonitoringImage.__module__}.{AnnotatedMonitoringImage.__name__}"
        ),
    ):
        self.pipeline_options = pipeline_options
        self.category_index = category_index
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.base_path = base_path
        self.bucket = bucket
        self.ext = ext
        self.module = module

    def annotate_image(self, el: AnnotatedMonitoringImage) -> bytes:
        if el.monitoring_image.data:
            image_np = np.array(PIL.Image.open(io.BytesIO(el.monitoring_image.data)))
        else:
            raise ValueError(
                f"Expecented NestedTelemetryEvent().image_data to be bytes, received None"
            )
        if el.annotations_filtered.num_detections > 0:
            # TODO re-enable boundary mask
            # detection_boundary_mask = self.calibration["mask"]
            ignored_mask = np.invert(detection_boundary_mask)  # type: ignore
            detection_boxes = np.array(
                b.xy for b in el.annotations_filtered.detection_boxes
            )  # type: ignore
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                detection_boxes,
                el.annotations_filtered.detection_classes,
                el.annotations_filtered.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
                # TODO re-enable boundary mask
                # detection_boundary_mask=detection_boundary_mask,
                detection_box_ignored=ignored_mask,
            )
        else:
            detection_boxes = np.array(
                [b.xy for b in el.annotations_all.detection_boxes]
            )
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                detection_boxes,
                el.annotations_all.detection_classes,
                el.annotations_all.detection_scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=self.score_threshold,
                max_boxes_to_draw=self.max_boxes_to_draw,
            )

        image = PIL.Image.fromarray(annotated_image_data, "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def process(
        self,
        element: AnnotatedMonitoringImage,
    ) -> Iterable[Tuple[str, str]]:

        key = element.monitoring_image.metadata.print_session.session
        datesegment = element.monitoring_image.metadata.print_session.datesegment
        filename = f"{element.monitoring_image.metadata.ts}.{self.ext}"
        outpath = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=key,
            datesegment=datesegment,
            ext=self.ext,
            filename=filename,
            module=self.module,
        )
        img = self.annotate_image(element)
        # gcs_client = beam.io.gcp.gcsio.GcsIO()

        # fs = beam.io.gcp.gcsfilesystem.GCSFileSystem(self.pipeline_options)
        # with fs.create(outpath) as f:
        # with gcs_client.open(outpath, "wb") as f:
        # f.write(img)
        logger.info(f"Writing {outpath} to gcs")
        writer = beam.io.filesystems.FileSystems.create(outpath)
        writer.write(img)
        writer.close()
        yield key, outpath
