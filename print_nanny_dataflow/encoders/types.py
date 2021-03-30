from typing import Tuple, Dict, Any, NamedTuple
import pandas as pd
import numpy as np
import nptyping as npt
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata

import pyarrow as pa
from print_nanny_client.telemetry_event import TelemetryEvent

from dataclasses import dataclass, asdict

DETECTION_LABELS = {
    1: "nozzle",
    2: "adhesion",
    3: "spaghetti",
    4: "print",
    5: "raft",
}

HEALTH_MULTIPLER = {1: 0, 2: -0.5, 3: -0.5, 4: 1, 5: 0}

NEUTRAL_LABELS = {1: "nozzle", 5: "raft"}

NEGATIVE_LABELS = {
    2: "adhesion",
    3: "spaghetti",
}

POSITIVE_LABELS = {
    4: "print",
}


class Image(NamedTuple):
    height: int
    width: int
    data: bytes
    # ndarray: np.ndarray


class Box(NamedTuple):
    detection_score: npt.Float32
    detection_class: npt.Int32
    ymin: npt.Float32
    xmin: npt.Float32
    ymax: npt.Float32
    xmax: npt.Float32


class BoundingBoxAnnotation(NamedTuple):
    num_detections: int
    detection_scores: np.ndarray
    detection_boxes: np.ndarray
    detection_classes: np.ndarray


class MonitoringFrame(NamedTuple):
    ts: int
    image: Image
    bounding_boxes: BoundingBoxAnnotation = None


class WindowedHealthRecord(NamedTuple):
    """
    Many FlatTelemetryEvent : 1 Monitoring Frame
    """

    ts: int
    client_version: str
    session: str

    # Metadata
    user_id: npt.Float32
    device_id: npt.Float32
    device_cloudiot_id: npt.Float32

    health_score: npt.Float32
    health_multiplier: npt.Float32
    detection_score: npt.Float32
    detection_class: npt.Int32

    window_start: int
    window_end: int

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(), index=["ts", "detection_class"])

    @staticmethod
    def records_to_health_dataframe(records) -> pd.DataFrame:
        data = {
            "ts": self.ts,
            "detection_class": self.detection_classes,
            "detection_score": self.detection_scores,
            "window_start": window_start,
            "window_end": window_end,
            "session": self.session,
            "user_id": self.user_id,
            "device_id": self.device_id,
        }
        df = (
            pd.DataFrame.from_records([data])
            .set_index("ts")
            .apply(pd.Series.explode)
            .reset_index()
        )

        return df.set_index(["ts", "detection_class"])


class DeviceCalibration(NamedTuple):
    coordinates: npt.NDArray[npt.Float32]
    mask = npt.NDArray[npt.Bool]
    fpm = int


class NestedTelemetryEvent(NamedTuple):
    """
    1 NestedTelemetryEvent : 1 Monitoring Frame
    """

    ts: int
    client_version: str
    event_type: int
    event_data_type: int
    session: str

    # Metadata
    user_id: npt.Float32
    device_id: npt.Float32
    device_cloudiot_id: npt.Float32

    # BoundingBoxes
    detection_scores: npt.NDArray[npt.Float32]
    detection_classes: npt.NDArray[npt.Int32]
    num_detections: int
    boxes_ymin: npt.NDArray[npt.Float32]
    boxes_xmin: npt.NDArray[npt.Float32]
    boxes_ymax: npt.NDArray[npt.Float32]
    boxes_xmax: npt.NDArray[npt.Float32]

    # Image
    image_width: npt.Float32
    image_height: npt.Float32
    image_data: bytes = None
    image_tensor: tf.Tensor = None
    calibration: DeviceCalibration = None
    annotated_image_data: bytes = None

    @staticmethod
    def pyarrow_schema(num_detections):
        return pa.schema(
            [
                pa.field(
                    "boxes_xmax", pa.list_(pa.float32(), list_size=num_detections)
                ),
                pa.field(
                    "boxes_xmin", pa.list_(pa.float32(), list_size=num_detections)
                ),
                pa.field(
                    "boxes_ymax", pa.list_(pa.float32(), list_size=num_detections)
                ),
                pa.field(
                    "boxes_ymin", pa.list_(pa.float32(), list_size=num_detections)
                ),
                pa.field("client_version", pa.string()),
                pa.field(
                    "detection_classes", pa.list_(pa.int32(), list_size=num_detections)
                ),
                pa.field(
                    "detection_scores", pa.list_(pa.float32(), list_size=num_detections)
                ),
                pa.field("device_cloudiot_id", pa.int32()),
                pa.field("device_id", pa.int32()),
                pa.field("event_data_type", pa.string()),
                pa.field("event_type", pa.string()),
                pa.field("image_data", pa.binary()),
                pa.field("image_height", pa.int32()),
                pa.field("image_width", pa.int32()),
                pa.field("num_detections", pa.int32()),
                pa.field("ts", pa.int32()),
                pa.field("user_id", pa.int32()),
            ]
        )

    @staticmethod
    def tf_feature_spec(num_detections):
        return schema_utils.schema_from_feature_spec(
            {
                "boxes_xmax": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_xmin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymax": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "client_version": tf.io.FixedLenFeature([], tf.string),
                "detection_classes": tf.io.FixedLenFeature([num_detections], tf.int64),
                "detection_scores": tf.io.FixedLenFeature([num_detections], tf.float32),
                "device_cloudiot_id": tf.io.FixedLenFeature([], tf.int64),
                "device_id": tf.io.FixedLenFeature([], tf.int64),
                "event_data_type": tf.io.FixedLenFeature([], tf.int64),
                "event_type": tf.io.FixedLenFeature([], tf.int64),
                "image_data": tf.io.FixedLenFeature([], tf.string),
                "image_height": tf.io.FixedLenFeature([], tf.int64),
                "image_width": tf.io.FixedLenFeature([], tf.int64),
                "num_detections": tf.io.FixedLenFeature([], tf.float32),
                "session": tf.io.FixedLenFeature([], tf.string),
                "ts": tf.io.FixedLenFeature([], tf.float32),
                "user_id": tf.io.FixedLenFeature([], tf.int64),
            }
        )

    @staticmethod
    def tfrecord_metadata(tf_feature_spec):
        return dataset_metadata.DatasetMetadata(tf_feature_spec)

    @classmethod
    def from_flatbuffer(cls, input_bytes):

        msg = TelemetryEvent.TelemetryEvent.GetRootAsTelemetryEvent(input_bytes, 0)
        obj = TelemetryEvent.TelemetryEventT.InitFromObj(msg)

        scores = []
        num_detections = []
        classes = []
        boxes_ymin = []
        boxes_xmin = []
        boxes_ymax = []
        boxes_xmax = []

        if obj.eventData.boundingBoxes is not None:
            scores = obj.eventData.boundingBoxes.detectionScores
            classes = obj.eventData.boundingBoxes.detectionClasses
            num_detections = obj.eventData.boundingBoxes.numDetections
            boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = [
                np.array([b.ymin, b.xmin, x.ymax, b.xmax])
                for b in obj.eventData.boundingBoxes
            ]

        image_data = obj.eventData.image.data.tobytes()
        return cls(
            ts=obj.metadata.ts,
            session=obj.metadata.session.decode("utf-8"),
            client_version=obj.metadata.clientVersion.decode("utf-8"),
            event_type=obj.eventType,
            event_data_type=obj.eventDataType,
            image_height=obj.eventData.image.height,
            image_width=obj.eventData.image.width,
            image_tensor=tf.expand_dims(tf.io.decode_jpeg(image_data), axis=0),
            image_data=image_data,
            user_id=obj.metadata.userId,
            device_id=obj.metadata.deviceId,
            device_cloudiot_id=obj.metadata.deviceCloudiotId,
            detection_scores=scores,
            detection_classes=classes,
            num_detections=num_detections,
            boxes_ymin=boxes_ymin,
            boxes_xmin=boxes_xmin,
            boxes_ymax=boxes_ymax,
            boxes_xmax=boxes_xmax,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())

    def flatten(self):
        array_fields = [
            "detection_scores",
            "detection_classes",
            "boxes_ymin",
            "boxes_xmin",
            "boxes_ymax",
            "boxes_xmax",
        ]
        default_fieldset = {
            k: v for k, v in self.to_dict().items() if k not in array_fields
        }
        return (
            FlatTelemetryEvent(
                **default_fieldset,
                detection_class=self.detection_classes[i],
                detection_score=self.detection_scores[i],
                box_xmin=self.boxes_xmin[i],
                box_ymin=self.boxes_ymin[i],
                box_ymax=self.boxes_ymax[i],
                box_xmax=self.boxes_xmax[i]
            )
            for i in range(0, self.num_detections)
        )

    def drop_image_data(self):
        exclude = ["image_data", "image_tensor"]
        fieldset = self.to_dict()
        return self.__class__(**{k: v for k, v in fieldset.items() if k not in exclude})

    def min_score_filter(self, score_threshold=0.5):
        masked_fields = [
            "detection_scores",
            "detection_classes",
            "boxes_ymin",
            "boxes_xmin",
            "boxes_ymax",
            "boxes_xmax",
        ]
        ignored_fields = ["num_detections"]
        fieldset = self.to_dict()
        mask = self.detection_scores >= score_threshold

        default_fieldset = {
            k: v
            for k, v in fieldset.items()
            if k not in masked_fields and k not in ignored_fields
        }
        if np.count_nonzero(mask) == 0:
            masked_fields = {
                k: np.array([])
                for k, v in fieldset.items()
                if k in masked_fields and k not in ignored_fields
            }
        else:
            masked_fields = {
                k: v[mask]
                for k, v in fieldset.items()
                if k in masked_fields and k not in ignored_fields
            }
        return self.__class__(
            **default_fieldset, **masked_fields, num_detections=np.count_nonzero(mask)
        )

    def percent_intersection(self, aoi_coords: Tuple[float]):
        """
        Returns intersection-over-union area, normalized between 0 and 1
        """
        detection_boxes = self.detection_boxes()

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

    def detection_boxes(self):
        return np.array(
            [self.boxes_ymin, self.boxes_xmin, self.boxes_ymax, self.boxes_xmax]
        ).T

    def calibration_filter(self, aoi_coords, min_overlap_area: float = 0.75):

        percent_intersection = self.percent_intersection(aoi_coords)
        ignored_mask = percent_intersection <= min_overlap_area

        detection_boxes = self.detection_boxes()
        included_mask = np.invert(ignored_mask)
        detection_boxes = np.squeeze(detection_boxes[included_mask])
        detection_scores = np.squeeze(self.detection_scores[included_mask])
        detection_classes = np.squeeze(self.detection_classes[included_mask])

        num_detections = int(np.count_nonzero(included_mask))

        filter_fields = [
            "detection_scores",
            "detection_classes",
            "boxes_ymin",
            "boxes_xmin",
            "boxes_ymax",
            "boxes_xmax",
            "num_detections",
        ]
        default_fieldset = {
            k: v for k, v in self.to_dict().items() if k not in filter_fields
        }
        boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = detection_boxes.T
        return self.__class__(
            detection_scores=detection_scores,
            detection_classes=detection_classes,
            num_detections=num_detections,
            boxes_ymin=boxes_ymin,
            boxes_xmin=boxes_xmin,
            boxes_ymax=boxes_ymax,
            boxes_xmax=boxes_xmax,
            **default_fieldset
        )
