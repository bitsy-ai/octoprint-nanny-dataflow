import typing
import numpy as np
import nptyping as npt
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata

from print_nanny_client.telemetry_event import TelemetryEvent

from dataclasses import dataclass, asdict


@dataclass
class Image:
    height: int
    width: int
    data: bytes
    # ndarray: np.ndarray

@dataclass
class Box:
    detection_score: npt.Float32
    detection_class: npt.Int32
    ymin: npt.Float32
    xmin: npt.Float32
    ymax: npt.Float32
    xmax: npt.Float32
    
@dataclass
class BoundingBoxAnnotation:
    num_detections: int
    detection_scores: np.ndarray
    detection_boxes: np.ndarray
    detection_classes: np.ndarray

@dataclass
class MonitoringFrame:
    ts: int
    image: Image
    bounding_boxes: BoundingBoxAnnotation = None

@dataclass
class FlatTelemetryEvent:
    """
    flattened data structures for
    tensorflow_transform.tf_metadata.schema_utils.schema_from_feature_spec
    """

    ts: int
    client_version: str
    event_type: int
    event_data_type: int
    session: str

    # Image
    image_data: tf.Tensor
    image_width: npt.Float32
    image_height: npt.Float32

    # Metadata
    user_id: npt.Float32
    device_id: npt.Float32
    device_cloudiot_id: npt.Float32

    # BoundingBoxes
    detection_score: npt.Float32
    detection_class: npt.Int32
    num_detections: npt.Int32
    box_ymin: npt.Float32
    box_xmin: npt.Float32
    box_ymax: npt.Float32
    box_xmax: npt.Float32
    image_tensor: tf.Tensor

@dataclass
class NestedTelemetryEvent:
    """
    flattened data structures for
    tensorflow_transform.tf_metadata.schema_utils.schema_from_feature_spec
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
    image_data: tf.Tensor = None
    image_tensor: tf.Tensor = None

    @staticmethod
    def feature_spec(num_detections):
        return schema_utils.schema_from_feature_spec(
            {
                "ts": tf.io.FixedLenFeature([], tf.float32),
                "session": tf.io.FixedLenFeature([], tf.string),
                "client_version": tf.io.FixedLenFeature([], tf.string),
                "event_type": tf.io.FixedLenFeature([], tf.int64),
                "event_data_type": tf.io.FixedLenFeature([], tf.int64),
                "image_data": tf.io.FixedLenFeature([], tf.string),
                "image_height": tf.io.FixedLenFeature([], tf.int64),
                "image_width": tf.io.FixedLenFeature([], tf.int64),
                "user_id": tf.io.FixedLenFeature([], tf.int64),
                "device_id": tf.io.FixedLenFeature([], tf.int64),
                "device_cloudiot_id": tf.io.FixedLenFeature([], tf.int64),
                "num_detections": tf.io.FixedLenFeature([], tf.float32),
                "detection_classes": tf.io.FixedLenFeature([num_detections], tf.int64),
                "detection_scores": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_xmin": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_ymax": tf.io.FixedLenFeature([num_detections], tf.float32),
                "boxes_xmax": tf.io.FixedLenFeature([num_detections], tf.float32),
            }
        )

    @staticmethod
    def tfrecord_metadata(feature_spec):
        return dataset_metadata.DatasetMetadata(feature_spec)

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
            session=obj.metadata.session,
            client_version=obj.metadata.clientVersion,
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

    def asdict(self):
        return asdict(self)
    
    def flatten(self):
        unwind_fields = ["detection_scores", "detection_classes", "boxes_ymin", "boxes_xmin", "boxes_ymax", "boxes_xmax"]
        default_fieldset = {k:v for k, v in self.asdict().items() if k not in unwind_fields }
        return ( FlatTelemetryEvent(
            **default_fieldset,
            detection_class=self.detection_classes[i],
            detection_score=self.detection_scores[i],
            box_xmin=self.boxes_xmin[i],
            box_ymin=self.boxes_ymin[i],
            box_ymax=self.boxes_ymax[i],
            box_xmax=self.boxes_xmax[i]
            ) for i in range(0, self.num_detections)
        )
    
    @classmethod
    def minimal(cls, instance):
        exclude = ["image_data", "image_tensor"]
        fieldset = instance.asdict()
        return cls(**{k:v for k,v in fieldset.items() if k not in exclude})