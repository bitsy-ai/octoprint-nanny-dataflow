from __future__ import annotations
from typing import Iterable, Tuple, Optional, Collection, List
import numpy as np
import logging

from google.protobuf.internal.containers import RepeatedScalarFieldContainer
from print_nanny_client.protobuf.monitoring_pb2 import (
    AnnotatedMonitoringImage,
    DeviceCalibration,
    BoxAnnotations,
    Box,
)
from print_nanny_dataflow.coders.types import (
    BoxAnnotationsT,
    get_health_weight,
    AnnotatedMonitoringImageT,
    DeviceCalibrationT,
)

logger = logging.getLogger(__name__)


def calc_percent_intersection(
    detection_boxes: Collection[Box],
    aoi_coords: RepeatedScalarFieldContainer,
) -> Iterable:
    """
    Returns intersection-over-union area, normalized between 0 and 1
    """

    # initialize array of zeroes
    aou = np.zeros(len(detection_boxes))

    # for each bounding box, calculate the intersection-over-area
    for i, box in enumerate(detection_boxes):
        # determine the coordinates of the intersection rectangle

        x_left = max(aoi_coords[0], box.xy[0])
        y_top = max(aoi_coords[1], box.xy[1])
        x_right = min(aoi_coords[2], box.xy[2])
        y_bottom = min(aoi_coords[3], box.xy[3])

        # boxes do not intersect, area is 0
        if x_right < x_left or y_bottom < y_top:
            aou[i] = 0.0
            continue

        # calculate
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of detection box
        box_area = (box.xy[2] - box.xy[0]) * (box.xy[3] - box.xy[1])

        if (intersection_area / box_area) == 1.0:
            aou[i] = 1.0
            continue

        aou[i] = intersection_area / box_area

    return aou


def filter_area_of_interest(
    element: BoxAnnotationsT,
    calibration: DeviceCalibrationT,
    min_calibration_area_overlap=0.75,
) -> BoxAnnotationsT:

    percent_intersection = calc_percent_intersection(
        element.detection_boxes, calibration.coordinates
    )
    ignored_mask = percent_intersection > min_calibration_area_overlap

    filtered_detection_boxes = [
        b for i, b in enumerate(element.detection_boxes) if ignored_mask[i]
    ]
    filtered_detection_scores = np.array(list(element.detection_scores))[ignored_mask]
    filtered_detection_classes = np.array(list(element.detection_classes))[ignored_mask]

    num_detections = np.count_nonzero(ignored_mask)  # type: ignore
    health_weights = map(get_health_weight, filtered_detection_classes)

    annotations = BoxAnnotations(
        num_detections=num_detections,
        detection_scores=filtered_detection_scores,
        detection_boxes=filtered_detection_boxes,
        detection_classes=filtered_detection_classes,
        health_weights=health_weights,
    )
    return annotations


def merge_filtered_annotations(
    element: AnnotatedMonitoringImageT, calibration: Optional[DeviceCalibrationT] = None
) -> AnnotatedMonitoringImageT:
    if calibration:
        annotations_filtered = filter_area_of_interest(
            element.annotations_all, calibration
        )
        msg = AnnotatedMonitoringImage(annotations_filtered=annotations_filtered)
        logger.info("Merging annotations_filtered")
        return element.MergeFrom(msg)
    logger.info("No calibration detected, returning original")
    return element
