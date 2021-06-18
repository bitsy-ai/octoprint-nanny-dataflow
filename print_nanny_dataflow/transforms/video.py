from sys import modules
from typing import Tuple, Iterable, NewType
import logging
import io
import os
import apache_beam as beam
from print_nanny_dataflow.utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from print_nanny_dataflow.coders.types import (
    CATEGORY_INDEX,
)
from print_nanny_dataflow.transforms.io import TypedPathMixin
from print_nanny_client.protobuf.monitoring_pb2 import AnnotatedMonitoringImage
import PIL
import numpy as np


logger = logging.getLogger(__name__)


@beam.typehints.with_input_types(AnnotatedMonitoringImage)
@beam.typehints.with_output_types(Tuple[str, str])
class WriteAnnotatedImage(TypedPathMixin, beam.DoFn):
    def __init__(
        self,
        base_path: str,
        bucket: str,
        category_index=CATEGORY_INDEX,
        score_threshold=0.5,
        max_boxes_to_draw=10,
        window_type="fixed",
        ext="jpg",
        module=(
            f"{AnnotatedMonitoringImage.__module__}.{AnnotatedMonitoringImage.__name__}"
        ),
    ):
        self.category_index = category_index
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.base_path = base_path
        self.bucket = bucket
        self.ext = ext
        self.window_type = window_type
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
        outpath = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=key,
            datesegment=element.monitoring_image.metadata.print_session.datesegment,
            ext=self.ext,
            filename=f"{element.monitoring_image.metadata.ts}.{self.ext}",
            module=self.module,
            window_type=self.window_type,
        )
        img = self.annotate_image(element)
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        with gcs_client.open(outpath, "wb") as f:
            logger.debug(f"Writing to {outpath}")
            f.write(img)

        yield key, outpath
