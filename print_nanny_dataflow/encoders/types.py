import typing
import numpy as np
import nptyping as npt
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.tf_metadata import dataset_metadata

import pyarrow as pa
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
    tensorflow_transform.tf_metadata.schema_utils.schema_from_tf_feature_spec
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
    tensorflow_transform.tf_metadata.schema_utils.schema_from_tf_feature_spec
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
    def pyarrow_schema(num_detections):
       return pa.schema([
            pa.field("boxes_xmax", pa.list_(pa.float32(), list_size=num_detections)),
            pa.field("boxes_xmin", pa.list_(pa.float32(), list_size=num_detections)),
            pa.field("boxes_ymax", pa.list_(pa.float32(), list_size=num_detections)),
            pa.field("boxes_ymin", pa.list_(pa.float32(), list_size=num_detections)),
            pa.field("client_version", pa.string()),
            pa.field("detection_classes", pa.list_(pa.int32(), list_size=num_detections)),
            pa.field("detection_scores", pa.list_(pa.float32(), list_size=num_detections)),
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

       ])

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
        array_fields = ["detection_scores", "detection_classes", "boxes_ymin", "boxes_xmin", "boxes_ymax", "boxes_xmax"]
        default_fieldset = {k:v for k, v in self.asdict().items() if k not in array_fields }
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
    
    def drop_image_data(self):
        exclude = ["image_data", "image_tensor"]
        fieldset = self.asdict()
        return self.__init__(**{k:v for k,v in fieldset.items() if k not in exclude})
    
    def min_score_filter(self, score_threshold=0.5):
        masked_fields = ["detection_scores", "detection_classes", "boxes_ymin", "boxes_xmin", "boxes_ymax", "boxes_xmax"]
        fieldset = instance.asdict()
        mask = event.detection_scores[event.detection_scores >= self.score_threshold]

        default_fieldset = {k:v for k,v in fieldset.items() if k not in masked_fields}
        masked_fields = {k: v[mask] for k,v in fieldset.items() if k in masked_fields}
        return cls(
            **default_fieldset,
            **masked_fields
        )
    
    def percent_intersection(self, aoi_coords: typing.Tuple[float]):
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
        return np.array([self.boxes_ymin, self.boxes_xmin, self.boxes_ymax, self.boxes_xmax ]).T

    def calibration_filter(self, aoi_coords, min_overlap_area:float=0.75):

        percent_intersection = self.percent_intersection(aoi_coords)
        ignored_mask = percent_intersection <= min_overlap_area

        detection_boxes = self.detection_boxes()
        included_mask = np.invert(ignored_mask)
        detection_boxes = np.squeeze(detection_boxes[included_mask])
        detection_scores = np.squeeze(self.detection_scores[included_mask])
        detection_classes = np.squeeze(self.detection_classes[included_mask])

        num_detections = int(np.count_nonzero(included_mask))

        filter_fields = ["detection_scores", "detection_classes", "boxes_ymin", "boxes_xmin", "boxes_ymax", "boxes_xmax", "num_detections"]
        default_fieldset = {k:v for k,v in self.asdict().items() if k not in filter_fields}
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