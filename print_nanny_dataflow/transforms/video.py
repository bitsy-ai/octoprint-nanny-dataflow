from typing import Tuple, Iterable
import logging
import io
import os
import apache_beam as beam
from print_nanny_dataflow.utils.visualization import (
    visualize_boxes_and_labels_on_image_array,
)
from print_nanny_dataflow.transforms.health import FilterAreaOfInterest
from print_nanny_dataflow.encoders.types import (
    NestedTelemetryEvent,
    WindowedHealthRecord,
    DeviceCalibration,
    RenderVideoMessage,
    AnnotatedImage,
    CATEGORY_INDEX,
    NestedWindowedHealthTrend,
    RenderVideoMessage,
    Metadata,
)
import PIL
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class WriteAnnotatedImage(beam.DoFn):
    def __init__(
        self,
        base_path,
        category_index=CATEGORY_INDEX,
        score_threshold=0.5,
        max_boxes_to_draw=10,
    ):
        self.category_index = category_index
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.base_path = base_path

    def annotate_image(self, event: NestedTelemetryEvent) -> bytes:
        image_np = np.array(PIL.Image.open(io.BytesIO(event.image_data)))

        if event.calibration is None:
            annotated_image_data = visualize_boxes_and_labels_on_image_array(
                image_np,
                event.detection_boxes(),
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
                event.detection_boxes(),
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
        image = PIL.Image.fromarray(annotated_image_data, "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def process(
        self,
        element: NestedTelemetryEvent,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Tuple[str, str]]:
        outpath = os.path.join(self.base_path, element.session, f"{element.ts}.jpg")
        img = self.annotate_image(element)
        gcs_client = beam.io.gcp.gcsio.GcsIO()

        with gcs_client.open(outpath, "wb") as f:
            f.write(img)

        yield element.session, outpath
