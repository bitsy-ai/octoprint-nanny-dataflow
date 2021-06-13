import pytest
import numpy as np

from print_nanny_dataflow.metrics.area_of_interest import percent_intersection


def test_object_in_aoi():
    detection_boxes = np.array([[0.3, 0.3, 0.9, 0.9]])
    calibration_box = [0.2, 0.2, 0.8, 0.8]
    percent_area = percent_intersection(detection_boxes, calibration_box)
    expected = (0.5 ** 2) / (0.6 ** 2)
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_object_not_in_aoi_1():
    detection_boxes = np.array([[0.3, 0.3, 0.9, 0.9]])
    calibration_box = [0.1, 0.1, 0.2, 0.2]
    percent_area = percent_intersection(detection_boxes, calibration_box)
    expected = 0.0
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_object_not_in_aoi_2():
    detection_boxes = np.array([[0.5, 0.2, 0.9, 0.4]])
    calibration_box = [0.1, 0.7, 0.39, 0.8]
    percent_area = percent_intersection(detection_boxes, calibration_box)
    expected = 0.0
    np.testing.assert_almost_equal(percent_area[0], expected)


def test_object_in_aoi_full():
    detection_boxes = np.array([[0.2, 0.2, 0.8, 0.8]])
    calibration_box = [0.1, 0.1, 0.9, 0.9]
    percent_area = percent_intersection(detection_boxes, calibration_box)
    expected = 1.0
    np.testing.assert_almost_equal(percent_area[0], expected)
