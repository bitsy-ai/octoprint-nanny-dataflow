import pytest
import numpy as np
import print_nanny_dataflow
from print_nanny_dataflow.encoders.types import NestedTelemetryEvent


def test_min_score_filter(partial_nested_telemetry_event_kwargs):
    box = np.array([0.3, 0.3, 0.9, 0.9])
    detection_boxes = np.stack((box for i in range(3)))
    detection_scores = np.array([0.7, 0.1, 0.1])
    detection_classes = np.array([1, 1, 1])

    boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T

    event = NestedTelemetryEvent(
        boxes_ymin=boxes_ymin,
        boxes_xmin=boxes_xmin,
        boxes_ymax=boxes_ymax,
        boxes_xmax=boxes_xmax,
        detection_classes=detection_classes,
        detection_scores=detection_scores,
        **partial_nested_telemetry_event_kwargs,
    )

    filtered_event = event.min_score_filter()

    expected_detection_boxes = np.array([box]).T
    (
        expected_boxes_ymin,
        expected_boxes_xmin,
        expected_boxes_ymax,
        expected_boxes_xmax,
    ) = detection_boxes.T
    expected_scores = np.array([0.7])
    expected_classes = np.array([1])
    assert filtered_event.num_detections == 1
    assert all(filtered_event.boxes_ymin == expected_boxes_ymin)
    assert all(filtered_event.boxes_ymax == expected_boxes_ymax)
    assert all(filtered_event.boxes_xmin == expected_boxes_xmin)
    assert all(filtered_event.boxes_xmax == expected_boxes_xmax)
    assert all(filtered_event.detection_scores == expected_scores)
    assert all(filtered_event.detection_classes == expected_classes)


def test_area_of_intersection_overlap(partial_nested_telemetry_event_kwargs):

    detection_boxes = np.array([[0.3, 0.3, 0.9, 0.9]])
    boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T

    calibration_box = [0.2, 0.2, 0.8, 0.8]

    detection_scores = np.array([1])

    detection_classes = np.array([4])

    event = NestedTelemetryEvent(
        boxes_ymin=boxes_ymin,
        boxes_xmin=boxes_xmin,
        boxes_ymax=boxes_ymax,
        boxes_xmax=boxes_xmax,
        detection_classes=detection_classes,
        detection_scores=detection_scores,
        **partial_nested_telemetry_event_kwargs,
    )

    percent_area = event.percent_intersection(calibration_box)
    expected = (0.5 ** 2) / (0.6 ** 2)
    np.testing.assert_almost_equal(percent_area[0], expected)

    filtered_event = event.calibration_filter(calibration_box, min_overlap_area=0.66)
    assert event == filtered_event


def test_area_of_intersection_no_overlap_0(partial_nested_telemetry_event_kwargs):

    detection_boxes = np.array([[0.3, 0.3, 0.9, 0.9]])
    boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T

    calibration_box = [0.1, 0.1, 0.2, 0.2]

    detection_scores = np.array([1])

    detection_classes = np.array([4])

    event = NestedTelemetryEvent(
        boxes_ymin=boxes_ymin,
        boxes_xmin=boxes_xmin,
        boxes_ymax=boxes_ymax,
        boxes_xmax=boxes_xmax,
        detection_classes=detection_classes,
        detection_scores=detection_scores,
        **partial_nested_telemetry_event_kwargs,
    )

    percent_area = event.percent_intersection(calibration_box)
    expected = 0.0
    np.testing.assert_almost_equal(percent_area[0], expected)
    filtered_event = event.calibration_filter(calibration_box)

    assert filtered_event.num_detections == 0
    assert len(filtered_event.detection_scores) == 0
    assert len(filtered_event.detection_classes) == 0
    assert len(filtered_event.boxes_ymin) == 0
    assert len(filtered_event.boxes_xmin) == 0
    assert len(filtered_event.boxes_ymax) == 0
    assert len(filtered_event.boxes_xmax) == 0


def test_area_of_intersection_no_overlap_1(partial_nested_telemetry_event_kwargs):
    detection_boxes = np.array([[0.5, 0.2, 0.9, 0.4]])
    boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T

    calibration_box = [0.1, 0.7, 0.39, 0.8]

    detection_scores = np.array([1])

    detection_classes = np.array([4])

    event = NestedTelemetryEvent(
        boxes_ymin=boxes_ymin,
        boxes_xmin=boxes_xmin,
        boxes_ymax=boxes_ymax,
        boxes_xmax=boxes_xmax,
        detection_classes=detection_classes,
        detection_scores=detection_scores,
        **partial_nested_telemetry_event_kwargs,
    )

    percent_area = event.percent_intersection(calibration_box)
    expected = 0.0
    np.testing.assert_almost_equal(percent_area[0], expected)
    filtered_event = event.calibration_filter(calibration_box)

    assert filtered_event.num_detections == 0
    assert len(filtered_event.detection_scores) == 0
    assert len(filtered_event.detection_classes) == 0
    assert len(filtered_event.boxes_ymin) == 0
    assert len(filtered_event.boxes_xmin) == 0
    assert len(filtered_event.boxes_ymax) == 0
    assert len(filtered_event.boxes_xmax) == 0


def test_area_of_intersection_prediction_contained_0(
    partial_nested_telemetry_event_kwargs,
):

    detection_boxes = np.array([[0.2, 0.2, 0.8, 0.8]])
    boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T

    calibration_box = [0.1, 0.1, 0.9, 0.9]

    detection_scores = np.array([1])

    detection_classes = np.array([4])

    event = NestedTelemetryEvent(
        boxes_ymin=boxes_ymin,
        boxes_xmin=boxes_xmin,
        boxes_ymax=boxes_ymax,
        boxes_xmax=boxes_xmax,
        detection_classes=detection_classes,
        detection_scores=detection_scores,
        **partial_nested_telemetry_event_kwargs,
    )

    percent_area = event.percent_intersection(calibration_box)
    expected = 1.0
    np.testing.assert_almost_equal(percent_area[0], expected)

    filtered_event = event.calibration_filter(calibration_box)

    assert event == filtered_event


# def test_print_health_trend_increasing(partial_nested_telemetry_event_kwargs):
#     num_detections = 40
#     prediction1 = octoprint_nanny.types.BoundingBoxPrediction(
#         detection_classes=np.repeat(4, num_detections),
#         detection_scores=np.linspace(0, 0.6, num=num_detections),
#         num_detections=num_detections,
#         detection_boxes=np.repeat([0.1, 0.1, 0.8, 0.8], num_detections),
#     )
#     prediction2 = octoprint_nanny.types.BoundingBoxPrediction(
#         detection_classes=np.repeat(4, num_detections),
#         detection_scores=np.linspace(0.7, 1, num=num_detections),
#         num_detections=num_detections,
#         detection_boxes=np.repeat([0.1, 0.1, 0.8, 0.8], num_detections),
#     )
#     df = explode_prediction_df(1234, prediction1)
#     df = df.append(explode_prediction_df(2345, prediction2))

#     assert print_is_healthy(df) == True


# def test_print_health_trend_decreasing():
#     num_detections = 40
#     classes = np.concatenate(
#         [
#             np.repeat(4, num_detections // 2),  # print
#             np.repeat(3, num_detections // 2),  # spaghetti
#         ]
#     )

#     scores1 = np.concatenate(
#         [
#             np.linspace(0, 0.6, num=num_detections // 2),  # print
#             np.linspace(0.4, 0.8, num=num_detections // 2),  # spaghetti
#         ]
#     )

#     scores2 = np.concatenate(
#         [
#             np.linspace(0, 0.7, num=num_detections // 2),  # print
#             np.linspace(0.6, 0.9, num=num_detections // 2),  # spaghetti
#         ]
#     )

#     prediction1 = octoprint_nanny.types.BoundingBoxPrediction(
#         detection_classes=classes,
#         detection_scores=scores1,
#         num_detections=num_detections,
#         detection_boxes=np.repeat([0.1, 0.1, 0.8, 0.8], num_detections),
#     )
#     prediction2 = octoprint_nanny.types.BoundingBoxPrediction(
#         detection_classes=classes,
#         detection_scores=scores2,
#         num_detections=num_detections,
#         detection_boxes=np.repeat([0.1, 0.1, 0.8, 0.8], num_detections),
#     )
#     df = explode_prediction_df(1234, prediction1)
#     df = df.append(explode_prediction_df(2345, prediction2))

#     assert print_is_healthy(df) == False


# def test_print_health_trend_initial():
#     num_detections = 40
#     prediction = octoprint_nanny.types.BoundingBoxPrediction(
#         detection_classes=np.repeat(4, num_detections),
#         detection_scores=np.linspace(0, 1, num=num_detections),
#         num_detections=num_detections,
#         detection_boxes=np.repeat([0.1, 0.1, 0.8, 0.8], num_detections),
#     )
#     df = explode_prediction_df(1234, prediction)

#     assert print_is_healthy(df) == True
