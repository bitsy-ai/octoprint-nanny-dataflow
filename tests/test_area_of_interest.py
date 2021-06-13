import numpy as np


from print_nanny_client.protobuf.monitoring_pb2 import (
    DeviceCalibration,
    BoxAnnotations,
    Box,
)
from print_nanny_dataflow.metrics.area_of_interest import (
    calc_percent_intersection,
    filter_box_annotations,
)


def test_object_in_aoi():
    detection_boxes = np.array([[0.3, 0.3, 0.9, 0.9]])
    calibration_box = [0.2, 0.2, 0.8, 0.8]
    percent_area = calc_percent_intersection(detection_boxes, calibration_box)
    expected = (0.5 ** 2) / (0.6 ** 2)
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_object_not_in_aoi_1():
    detection_boxes = np.array([[0.3, 0.3, 0.9, 0.9]])
    calibration_box = [0.1, 0.1, 0.2, 0.2]
    percent_area = calc_percent_intersection(detection_boxes, calibration_box)
    expected = 0.0
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_object_not_in_aoi_2():
    detection_boxes = np.array([[0.5, 0.2, 0.9, 0.4]])
    calibration_box = [0.1, 0.7, 0.39, 0.8]
    percent_area = calc_percent_intersection(detection_boxes, calibration_box)
    expected = 0.0
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_object_in_aoi_full():
    detection_boxes = np.array([[0.2, 0.2, 0.8, 0.8]])
    calibration_box = [0.1, 0.1, 0.9, 0.9]
    percent_area = calc_percent_intersection(detection_boxes, calibration_box)
    expected = 1.0
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_filter_all_detections_by_confidence_0():
    """
    all detections are below confidence threshold
    """

    num_detections = 10
    threshold = 0.5
    detection_boxes = np.array([[0.2, 0.2, 0.8, 0.8] for _ in range(0, num_detections)])
    detection_scores = np.linspace(0, threshold, num_detections)
    detection_classes = np.ones(num_detections, dtype=np.int32)

    _detection_boxes = [
        Box(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3]) for b in detection_boxes
    ]
    annotations = BoxAnnotations(
        detection_boxes=_detection_boxes,
        detection_scores=detection_scores,
        detection_classes=detection_classes,
    )

    filtered_annotations = filter_box_annotations(
        annotations, min_score_threshold=threshold
    )

    assert filtered_annotations.num_detections == 0
    assert len(filtered_annotations.detection_scores) == 0
    assert len(filtered_annotations.detection_classes) == 0
    assert len(filtered_annotations.detection_boxes) == 0


def test_filter_all_detections_by_confidence_1():
    """
    all detections are above confidence threshold
    """

    num_detections = 10
    threshold = 0.5
    detection_boxes = np.array([[0.2, 0.2, 0.8, 0.8] for _ in range(0, num_detections)])
    detection_scores = np.linspace(threshold + 0.1, 1, num_detections)
    detection_classes = np.ones(num_detections, dtype=np.int32)

    _detection_boxes = [
        Box(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3]) for b in detection_boxes
    ]
    annotations = BoxAnnotations(
        detection_boxes=_detection_boxes,
        detection_scores=detection_scores,
        detection_classes=detection_classes,
    )

    filtered_annotations = filter_box_annotations(
        annotations, min_score_threshold=threshold
    )

    assert filtered_annotations.num_detections == num_detections
    assert len(filtered_annotations.detection_scores) == num_detections
    assert len(filtered_annotations.detection_classes) == num_detections
    assert len(filtered_annotations.detection_boxes) == num_detections


def test_filter_some_detections_by_confidence_1():
    """
    some detections above, some below threshold
    """

    num_detections = 10
    threshold = 0.5
    detection_boxes = np.array([[0.2, 0.2, 0.8, 0.8] for _ in range(0, num_detections)])
    detection_scores = np.append(
        np.linspace(0, threshold, num_detections // 2),
        np.linspace(threshold, 1, num_detections // 2),
    )
    detection_classes = np.ones(num_detections, dtype=np.int32)

    _detection_boxes = [
        Box(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3]) for b in detection_boxes
    ]
    annotations = BoxAnnotations(
        detection_boxes=_detection_boxes,
        detection_scores=detection_scores,
        detection_classes=detection_classes,
    )

    filtered_annotations = filter_box_annotations(
        annotations, min_score_threshold=threshold
    )

    expected_num = (num_detections // 2) - 1  # exclusive threshold
    assert filtered_annotations.num_detections == expected_num
    assert len(filtered_annotations.detection_scores) == expected_num
    assert len(filtered_annotations.detection_classes) == expected_num
    assert len(filtered_annotations.detection_boxes) == expected_num
