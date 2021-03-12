import typing
import numpy as np
import nptyping as npt
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata

from print_nanny_client.telemetry_event import TelemetryEvent

class FlatTelemetryEvent(typing.NamedTuple):
    """
    flattened data structures for
    tensorflow_transform.tf_metadata.schema_utils.schema_from_feature_spec
    """

    ts: int
    version: str
    event_type: int
    event_data_type: int

    # Image
    image_data: bytes
    image_width: npt.Float32
    image_height: npt.Float32

    # Metadata
    user_id: npt.Float32
    device_id: npt.Float32
    device_cloudiot_id: npt.Float32

    # BoundingBoxes
    scores: npt.NDArray[npt.Float32] = []
    classes: npt.NDArray[npt.Int32] = []
    num_detections: npt.NDArray[npt.Int32] = []
    boxes_ymin: npt.NDArray[npt.Float32] = []
    boxes_xmin: npt.NDArray[npt.Float32] = []
    boxes_ymax: npt.NDArray[npt.Float32] = []
    boxes_xmax: npt.NDArray[npt.Float32] = []

    @staticmethod
    def feature_spec(num_detections):
        return schema_utils.schema_from_feature_spec(
            {
                "ts": tf.io.FixedLenFeature([], tf.float32),
                "version": tf.io.FixedLenFeature([], tf.string),
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
                "original_image": tf.io.FixedLenFeature([], tf.string),
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
            scores = obj.eventData.boundingBoxes.scores
            classes = obj.eventData.boundingBoxes.classes
            num_detections = obj.eventData.boundingBoxes.numDetections
            boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax = [
                np.array([b.ymin, b.xmin, x.ymax, b.xmax])
                for b in obj.eventData.boundingBoxes
            ]
        return cls(
            ts=obj.metadata.ts,
            version=obj.version,
            event_type=obj.eventType,
            event_data_type=obj.eventDataType,
            image_height=obj.eventData.image.height,
            image_width=obj.eventData.image.width,
            image_data=obj.eventData.image.data,
            user_id=obj.metadata.userId,
            device_id=obj.metadata.deviceId,
            device_cloudiot_id=obj.metadata.deviceCloudiotId,
            scores=scores,
            classes=classes,
            num_detections=num_detections,
            boxes_ymin=boxes_ymin,
            boxes_xmin=boxes_xmin,
            boxes_ymax=boxes_ymax,
            boxes_xmax=boxes_xmax,
        )
