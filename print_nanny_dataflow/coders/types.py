from __future__ import annotations
import json
from enum import Enum
from typing import Tuple, Dict, Any, NamedTuple, TypeVar, Generic, Optional
import pandas as pd
import numpy as np
import nptyping as npt
import tensorflow as tf
import os
import flatbuffers
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_metadata.proto.v0 import schema_pb2

import pyarrow as pa

from print_nanny_client.flatbuffers.alert import (
    Alert,
    AnnotatedVideo,
)
from print_nanny_client.flatbuffers.alert.AlertEventTypeEnum import AlertEventTypeEnum
from print_nanny_client.flatbuffers.alert import Metadata as MetadataFB
from print_nanny_client.flatbuffers.monitoring import MonitoringEvent


CATEGORY_INDEX = {
    0: {"name": "background", "id": 0, "health_weight": 0},
    1: {"name": "nozzle", "id": 1, "health_weight": 0},
    2: {"name": "adhesion", "id": 2, "health_weight": -0.5},
    3: {"name": "spaghetti", "id": 3, "health_weight": -0.5},
    4: {"name": "print", "id": 4, "health_weight": 1},
    5: {"name": "raftt", "id": 5, "health_weight": 1},
}


def get_health_weight(label: int) -> float:
    return CATEGORY_INDEX[label].get("health_weight")  # type: ignore


class WindowType(Enum):
    Fixed = 1
    Sliding = 2
    Sessions = 3


class Metadata(NamedTuple):
    client_version: str
    print_session: str
    user_id: int
    octoprint_device_id: int
    cloudiot_device_id: int
    window_start: Optional[int] = None
    window_end: Optional[int] = None

    def to_dict(self):
        return self._asdict()

    @staticmethod
    def pyarrow_fields():
        return [
            ("client_version", pa.string()),
            ("session", pa.string()),
            ("user_id", pa.int32()),
            ("device_id", pa.int32()),
            ("cloudiot_device_id", pa.int64()),
            ("window_start", pa.int64()),
            ("window_end", pa.int64()),
        ]

    @classmethod
    def pyarrow_struct(cls):
        return pa.struct(cls.pyarrow_fields())

    @classmethod
    def pyarrow_schema(cls):
        return pa.schema(cls.pyarrow_fields())


class HealthTrend(NamedTuple):
    coef: npt.NDArray[npt.Float32]
    domain: npt.NDArray[npt.Float32]
    window: npt.NDArray[npt.Float32]
    roots: npt.NDArray[npt.Float32]
    degree: int

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()

    @staticmethod
    def pyarrow_fields():
        return [
            ("coef", pa.list_(pa.float32())),
            ("domain", pa.list_(pa.float32())),
            ("window", pa.list_(pa.float32())),
            ("roots", pa.list_(pa.float32())),
            ("degree", pa.int32()),
        ]

    @classmethod
    def pyarrow_struct(cls):
        return pa.struct(cls.pyarrow_fields())

    @classmethod
    def pyarrow_schema(cls):
        return pa.schema(cls.pyarrow_fields())


class WindowedHealthDataFrameRow(NamedTuple):
    ts: int
    print_session: str
    health_score: npt.Float32
    health_weight: npt.Float32
    detection_class: npt.Int32
    detection_score: npt.Float32

    def to_dict(self):
        return self._asdict()

    @staticmethod
    def pyarrow_fields():
        return [
            ("ts", pa.int64()),
            ("print_session", pa.string()),
            ("health_score", pa.float32()),
            ("health_weight", pa.float32()),
            ("detection_class", pa.int32()),
            ("detection_score", pa.float32()),
        ]

    @classmethod
    def pyarrow_struct(cls):
        return pa.struct(cls.pyarrow_fields())

    @classmethod
    def pyarrow_schema(cls):
        return pa.schema(cls.pyarrow_fields())


class NestedWindowedHealthTrend(NamedTuple):
    print_session: str
    metadata: Metadata
    health_score: npt.NDArray[npt.Float32]
    health_weight: npt.NDArray[npt.Float32]
    detection_class: npt.NDArray[npt.Int32]
    detection_score: npt.NDArray[npt.Float32]
    cumsum: npt.NDArray[npt.Float32]
    poly_coef: npt.NDArray[npt.Float32]
    poly_domain: npt.NDArray[npt.Float32]
    poly_roots: npt.NDArray[npt.Float32]
    poly_degree: int

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()

    def to_bytes(self):
        return pa.serialize(self.to_dict()).to_buffer().to_pybytes()

    @classmethod
    def from_bytes(cls, pyarrow_bytes):
        return cls(**pa.deserialize(pyarrow_bytes))

    def to_parquetio_serializable(self):
        """
        apache_beam.io.parquetio expects a dicts of native python types as input
        TODO Investigate next steps...
        Write custom parquetio transform to handle my Flatbuffer schemas?
        Extend apache_beam.coders.coders Base?

        Beam monkey-patches dill
        https://github.com/apache/beam/blob/master/sdks/python/apache_beam/internal/pickler.py
        Apache arrow falls back to standard lib pickle
        https://arrow.apache.org/docs/python/ipc.html#serializing-custom-data-types
        """

        pass

    @staticmethod
    def pyarrow_fields():
        return [
            ("print_session", pa.string()),
            ("metadata", Metadata.pyarrow_struct()),
            ("health_score", pa.list_(pa.float32())),
            ("health_weight", pa.list_(pa.float32())),
            ("detection_class", pa.list_(pa.float32())),
            ("detection_score", pa.list_(pa.float32())),
            ("cumsum", pa.list_(pa.float32())),
            ("poly_coef", pa.list_(pa.float32())),
            ("poly_domain", pa.list_(pa.float32())),
            ("poly_roots", pa.list_(pa.float32())),
            ("poly_degree", pa.int32()),
        ]

    @classmethod
    def pyarrow_struct(cls):
        return pa.struct(cls.pyarrow_fields())

    @classmethod
    def pyarrow_schema(cls):
        return pa.schema(cls.pyarrow_fields())


class WindowedHealthRecord(NamedTuple):

    ts: int
    print_session: str
    metadata: Metadata
    health_score: npt.Float32
    health_weight: npt.Float32
    detection_score: npt.Float32
    detection_class: npt.Int32

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(), index=["ts", "detection_class"])

    @staticmethod
    def pyarrow_fields():
        return [
            ("ts", pa.int64()),
            ("print_session", pa.string()),
            ("metadata", Metadata.pyarrow_struct()),
            ("health_score", pa.float32()),
            ("health_weight", pa.float32()),
            ("detection_class", pa.int32()),
            ("detection_score", pa.float32()),
        ]

    @classmethod
    def pyarrow_struct(cls):
        return pa.struct(cls.pyarrow_fields())

    @classmethod
    def pyarrow_schema(cls):
        return pa.schema(cls.pyarrow_fields())


class AnnotatedImage(NamedTuple):
    ts: int
    print_session: str
    metadata: Metadata
    annotated_image_data: bytes


class NestedTelemetryEvent(NamedTuple):
    """
    1 NestedTelemetryEvent : 1 Monitoring Frame
    """

    ts: int
    client_version: str
    print_session: str

    user_id: int
    octoprint_device_id: int
    cloudiot_device_id: int
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
    image_data: Optional[bytes] = None
    image_tensor: Optional[tf.Tensor] = None
    annotated_image_data: Optional[bytes] = None

    @staticmethod
    def pyarrow_schema(num_detections):
        return pa.schema(
            [
                ("ts", pa.int64()),
                ("client_version", pa.string()),
                ("print_session", pa.string()),
                ("user_id", pa.int32()),
                ("boxes_xmax", pa.list_(pa.float32(), list_size=num_detections)),
                ("boxes_xmin", pa.list_(pa.float32(), list_size=num_detections)),
                ("boxes_ymax", pa.list_(pa.float32(), list_size=num_detections)),
                ("boxes_ymin", pa.list_(pa.float32(), list_size=num_detections)),
                ("detection_classes", pa.list_(pa.int32(), list_size=num_detections)),
                ("detection_scores", pa.list_(pa.float32(), list_size=num_detections)),
                ("cloudiot_device_id", pa.int64()),
                ("octoprint_device_id", pa.int32()),
                ("image_data", pa.binary()),
                ("annotated_image_data", pa.binary()),
                ("image_height", pa.int32()),
                ("image_width", pa.int32()),
                ("num_detections", pa.int32()),
            ]
        )

    @staticmethod
    def tfrecord_schema(num_detections: int) -> schema_pb2.Schema:
        return schema_utils.schema_from_feature_spec(
            {
                "boxes_xmax": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_xmin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymax": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "client_version": tf.io.FixedLenFeature([], tf.string),
                "detection_classes": tf.io.FixedLenFeature([num_detections], tf.int64),
                "detection_scores": tf.io.FixedLenFeature([num_detections], tf.float32),
                "cloudiot_device_id": tf.io.FixedLenFeature([], tf.int64),
                "octoprint_device_id": tf.io.FixedLenFeature([], tf.int64),
                "image_data": tf.io.FixedLenFeature([], tf.string),
                "image_height": tf.io.FixedLenFeature([], tf.int64),
                "image_width": tf.io.FixedLenFeature([], tf.int64),
                "num_detections": tf.io.FixedLenFeature([], tf.float32),
                "print_session": tf.io.FixedLenFeature([], tf.string),
                "ts": tf.io.FixedLenFeature([], tf.int64),
                "user_id": tf.io.FixedLenFeature([], tf.int64),
            }
        )

    @classmethod
    def tfrecord_metadata(cls, num_detections: int) -> dataset_metadata.DatasetMetadata:
        schema = cls.tfrecord_schema(num_detections)
        return dataset_metadata.DatasetMetadata(schema)

    @classmethod
    def from_flatbuffer(cls, input_bytes):

        obj = MonitoringEvent.MonitoringEvent.GetRootAsMonitoringEvent(input_bytes, 0)

        if obj.BoundingBoxes() is not None:
            scores = obj.BoundingBoxes().DetectionScores()
            classes = obj.BoundingBoxes().DetectionClasses()
            num_detections = obj.BoundingBoxes().NumDetections()
            boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = [
                np.array([b.Ymin(), b.Xmin(), b.Ymax(), b.Xmax()])
                for b in obj.BoundingBoxes()
            ]
        else:
            num_detections = 0
            classes = np.array([], dtype=np.int32)
            scores = np.array([], dtype=np.float32)
            boxes_ymin = np.array([], dtype=np.float32)
            boxes_xmin = np.array([], dtype=np.float32)
            boxes_ymax = np.array([], dtype=np.float32)
            boxes_xmax = np.array([], dtype=np.float32)

        image_data = obj.Image().DataAsNumpy().tobytes()
        session = obj.Metadata().PrintSession().decode("utf-8")
        return cls(
            ts=obj.Metadata().Ts(),
            print_session=session,
            image_height=obj.Image().Height(),
            image_width=obj.Image().Width(),
            image_tensor=tf.expand_dims(tf.io.decode_jpeg(image_data), axis=0),
            image_data=image_data,
            detection_scores=scores,
            detection_classes=classes,
            num_detections=num_detections,
            boxes_ymin=boxes_ymin,
            boxes_xmin=boxes_xmin,
            boxes_ymax=boxes_ymax,
            boxes_xmax=boxes_xmax,
            user_id=obj.Metadata().UserId(),
            octoprint_device_id=obj.Metadata().OctoprintDeviceId(),
            cloudiot_device_id=obj.Metadata().CloudiotDeviceId(),
            client_version=obj.Metadata().ClientVersion().decode("utf-8"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self._asdict()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())

    def drop_image_data(self):
        exclude = ["image_data", "image_tensor"]
        fieldset = self.to_dict()
        return self.__class__(**{k: v for k, v in fieldset.items() if k not in exclude})

    def percent_intersection(self, aoi_coords: Tuple[float, float, float, float]):
        """
        Returns intersection-over-union area, normalized between 0 and 1
        """
        detection_boxes = self.detection_boxes

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

    @property
    def detection_boxes(self):
        return np.array(
            [self.boxes_ymin, self.boxes_xmin, self.boxes_ymax, self.boxes_xmax]
        ).T
