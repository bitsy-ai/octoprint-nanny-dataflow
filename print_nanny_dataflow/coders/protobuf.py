from typing import Generator
import google.protobuf.message


# https://building.enlyze.com/posts/apache-beam-python-protobuf-decorater/
def proto_coder(proto_cls: google.protobuf.message.Message):
    assert issubclass(proto_cls, google.protobuf.message.Message)

    def serialize_proto(func: callable) -> callable:

        # as we will use this decorator for class methods,
        # transform is the reference to e.g. the DoFn instance
        def wrap(transform, element, *args, **kwargs) -> Generator[bytes, None, None]:

            # deserialize the original element
            msg = proto_cls()
            msg.ParseFromString(element)

            # this part is equal to `serialize_protos`
            for result in func(transform, element, *args, **kwargs):
                if not issubclass(type(result), google.protobuf.message.Message):
                    raise Exception(f"== {func.__name__} returned no protobuf message")

                yield result.SerializeToString()

        return wrap

    return serialize_proto
