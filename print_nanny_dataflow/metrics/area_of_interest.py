from typing import Iterable, Tuple
import numpy as np

from print_nanny_client.protobuf.monitoring_pb2 import (
    AnnotatedMonitoringImage,
    DeviceCalibration,
)


def percent_intersection(
    detection_boxes: Iterable, aoi_coords: Tuple[float, float, float, float]
) -> Iterable:
    """
    Returns intersection-over-union area, normalized between 0 and 1
    """
    # initialize array of zeroes
    aou = np.zeros(len(detection_boxes))

    # for each bounding box, calculate the intersection-over-area
    for i, box in enumerate(detection_boxes):
        # determine the coordinates of the intersection rectangle
        x_left = max(aoi_coords[0], box[0])
        y_top = max(aoi_coords[1], box[1])
        x_right = min(aoi_coords[2], box[2])
        y_bottom = min(aoi_coords[3], box[3])

        # boxes do not intersect, area is 0
        if x_right < x_left or y_bottom < y_top:
            aou[i] = 0.0
            continue

        # calculate
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of detection box
        box_area = (box[2] - box[0]) * (box[3] - box[1])

        if (intersection_area / box_area) == 1.0:
            aou[i] = 1.0
            continue

        aou[i] = intersection_area / box_area

    return aou


def filter_area_of_interest(
    element: AnnotatedMonitoringImage, calibration: DeviceCalibration
) -> AnnotatedMonitoringImage:
    pass
    # percent_intersection = event.percent_intersection(self.coordinates)
    #     ignored_mask = percent_intersection <= min_overlap_area

    #     detection_boxes = event.detection_boxes
    #     included_mask = np.invert(ignored_mask)
    #     detection_boxes = np.squeeze(detection_boxes[included_mask])
    #     detection_scores = np.squeeze(event.detection_scores[included_mask])
    #     detection_classes = np.squeeze(event.detection_classes[included_mask])

    #     num_detections = int(np.count_nonzero(included_mask))

    #     filter_fields = [
    #         "detection_scores",
    #         "detection_classes",
    #         "boxes_ymin",
    #         "boxes_xmin",
    #         "boxes_ymax",
    #         "boxes_xmax",
    #         "num_detections",
    #     ]
    #     default_fieldset = {
    #         k: v for k, v in event.to_dict().items() if k not in filter_fields
    #     }
    #     boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T

    #     _kwargs = event.to_dict()
    #     _kwargs.update(
    #         dict(
    #             detection_scores=detection_scores,
    #             detection_classes=detection_classes,
    #             num_detections=num_detections,
    #             boxes_ymin=boxes_ymin,
    #             boxes_xmin=boxes_xmin,
    #             boxes_ymax=boxes_ymax,
    #    )        boxes_xmax=boxes_xmax,
    #             calibration=self,
    #         )
