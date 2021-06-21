from tensorflow_transform.coders import example_proto_coder
import tensorflow as tf
from .types import NestedTelemetryEvent


class ExampleProtoEncoder(example_proto_coder.ExampleProtoCoder):
    """
    Extends example_proto_coder.ExampleProtoCoder to accept NamedTuple / Dataclass in addition to untyped dict
    """

    def encode_dict(self, instance):
        """Encode a tf.transform encoded dict as tf.Example."""
        # The feature handles encode using the self._encode_example_cache.
        for feature_handler in self._feature_handlers:
            value = instance[feature_handler.name]
            try:
                feature_handler.encode_value(value)
            except TypeError as e:
                raise TypeError(
                    '%s while encoding feature "%s"' % (e, feature_handler.name)
                )

        if self._serialized:
            return self._encode_example_cache.SerializeToString()

        result = tf.train.Example()
        result.CopyFrom(self._encode_example_cache)
        return result

    def encode_named_tuple(self, instance):
        """Encode a NamedTuple as tf.Example."""
        # The feature handles encode using the self._encode_example_cache.
        for feature_handler in self._feature_handlers:
            value = getattr(instance, feature_handler.name)
            try:
                feature_handler.encode_value(value)
            except TypeError as e:
                raise TypeError(
                    '%s while encoding feature "%s"' % (e, feature_handler.name)
                )

        if self._serialized:
            return self._encode_example_cache.SerializeToString()

        result = tf.train.Example()
        result.CopyFrom(self._encode_example_cache)
        return result

    def encode(self, instance):
        if isinstance(instance, dict):
            return self.encode_dict(instance)
        elif isinstance(instance, NestedTelemetryEvent):
            return self.encode_named_tuple(instance)
        else:
            raise NotImplementedError(
                f"unsupported source data type {type(instance)} passed to ExampleProtoEncoder. Please use a <dict> or <types.NestedTelemetryEvent>"
            )
