from typing import Iterable, Tuple, Optional, Collection
import numpy as np

from google.protobuf.internal.containers import RepeatedScalarFieldContainer
from print_nanny_client.protobuf.monitoring_pb2 import (
    DeviceCalibration,
    BoxAnnotations,
    Box,
)
from print_nanny_dataflow.coders.types import CATEGORY_INDEX


def calc_percent_intersection(
    detection_boxes: Collection[Box],
    aoi_coords: RepeatedScalarFieldContainer[float],
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
    element: BoxAnnotations,
    calibration: DeviceCalibration,
    min_calibration_area_overlap=0.75,
) -> BoxAnnotations:

    percent_intersection = calc_percent_intersection(
        element.detection_boxes, calibration.coordinates
    )
    ignored_mask = percent_intersection > min_calibration_area_overlap

    filtered_detection_boxes = [
        Box(xy=b) for b in element.detection_boxes[ignored_mask]
    ]

    filtered_detection_scores = element.detection_scores[ignored_mask]
    filtered_detection_classes = element.detection_classes[ignored_mask]

    num_detections = np.count_nonzero(ignored_mask)  # type: ignore
    health_weights = [
        CATEGORY_INDEX[i]["health_weight"] for i in filtered_detection_classes
    ]
    annotations = BoxAnnotations(
        num_detections=num_detections,
        detection_scores=filtered_detection_scores,
        detection_boxes=filtered_detection_boxes,
        detection_classes=filtered_detection_classes,
        health_weights=health_weights,
    )
    return annotations
