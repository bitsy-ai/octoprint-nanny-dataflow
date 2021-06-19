import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Any, Iterable, NamedTuple
import apache_beam as beam

from print_nanny_client.protobuf.monitoring_pb2 import AnnotatedMonitoringImage

logger = logging.getLogger(__name__)


class TypedPathMixin:
    """
    Mixin support for an outpath conforming to:

    gs://<bucket>/<base_path>/<module>.<class>/datesegment/key/<ext>/<window_start>_<window_end>

    The key (typically a session uuid) is prepended to avoid sequential bottleneck when writing to GS backend
    https://cloud.google.com/blog/products/gcp/optimizing-your-cloud-storage-performance-google-cloud-performance-atlas

    Datesegment is calculated from session start time
    """

    def path(
        self,
        bucket: str,
        base_path: str,
        key: str,
        datesegment: str,
        module: str,
        ext: str,
        window_type: str,
        filename: str = "",
        protocol: str = "gs://",
    ) -> str:
        """
        Constructs output path from parts:

        base_path: gs://bucket-name/dataflow/base/path/to/sinks/
        key: session
        classname: NestedTelemetryEvent
        suffix: tfrecords

        Results in:
        gs://bucket-name/dataflow/base/path/to/sinks/<session>/<classname>/tfrecords/
        """
        return os.path.join(
            protocol,
            bucket,
            base_path,
            module,
            window_type,
            datesegment,
            key,
            ext,
            filename,
        )


class WriteWindowedTFRecord(TypedPathMixin, beam.DoFn):
    """Output one TFRecord file per window per key"""

    def __init__(
        self,
        base_path: str,
        bucket: str,
        module,
        window_type: str,
        ext: str = "tfrecord",
    ):
        self.base_path = base_path
        self.bucket = bucket
        self.module = module
        self.ext = ext
        self.window_type = window_type

    def process(
        self,
        keyed_elements: Tuple[
            Any, Iterable[AnnotatedMonitoringImage]
        ] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Iterable[str]]:

        key, elements = keyed_elements
        element = elements[0]  # type: ignore

        window_start = int(window.start)
        window_end = int(window.end)
        filename = f"{window_start}_{window_end}"

        outpath = self.path(
            bucket=self.bucket,
            base_path=self.base_path,
            key=key,
            datesegment=element.monitoring_image.metadata.print_session.datesegment,
            ext=self.ext,
            filename=filename,
            module=self.module,
            window_type=self.window_type,
        )

        coder = beam.coders.coders.ProtoCoder(element.__class__)
        yield (
            elements
            | beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=outpath,
                num_shards=1,
                shard_name_template="",
                coder=coder,
                file_name_suffix=".tfrecords.gz",
            )
        )


# TODO beam parquetio pyarrow serializer only handles basic Python types, hence list() on itertuples zip generator below
# pyarrow.lib.ArrowTypeError: Could not convert <zip object at 0x7f578011ef80> with type zip: was not a sequence or recognized null for conversion to list type [while running 'WriteToParquet/Write/WriteImpl/WriteBundles']
SERIALIZE_FNS = {
    pd.DataFrame: lambda x: list(tuple(n) for n in x.itertuples(name=None)),
    pd.Series: lambda x: list(x),
    np.polynomial.polynomial.Polynomial: lambda x: list(x),  # type: ignore
}


class WriteWindowedParquet(beam.DoFn):
    def __init__(self, base_path: str, schema, record_type="parquet"):
        self.base_path = base_path
        self.schema = schema
        self.record_type = record_type

    def outpath(self, key: str, window_start: int, window_end: int):
        return os.path.join(
            self.base_path,
            key,
            self.record_type,
            f"{window_start}_{window_end}.parquet",
        )

    def process(
        self,
        keyed_elements: Tuple[Any, Iterable[NamedTuple]] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Iterable[str]]:
        key, elements = keyed_elements
        window_start = int(window.start)
        window_end = int(window.end)

        output_path = self.outpath(
            key, window_start=window_start, window_end=window_end
        )
        # @todo apache-beam == 2.28
        # Transforms in beam.io.parquetio only operate on dict representations of data
        yield elements | beam.Map(
            lambda e: e.to_dict()
        ) | beam.io.parquetio.WriteToParquet(
            output_path, self.schema, num_shards=1, shard_name_template=""
        )
