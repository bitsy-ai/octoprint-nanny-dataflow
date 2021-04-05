import os
import logging
import pandas as pd
import numpy as np
from collections import Sequence
from collections import OrderedDict
from typing import Tuple, Any, Iterable, NamedTuple, List
import apache_beam as beam

from tensorflow_metadata.proto.v0 import schema_pb2
from apache_beam.pvalue import PCollection
from print_nanny_dataflow.encoders.tfrecord_example import ExampleProtoEncoder

logger = logging.getLogger(__name__)


class WriteWindowedTFRecord(beam.DoFn):
    """Output one TFRecord file per window per key"""

    def __init__(self, base_path: str, schema: schema_pb2.Schema):
        self.base_path = base_path
        self.schema = schema

    def process(
        self,
        keyed_elements: Tuple[Any, Iterable[NamedTuple]] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Iterable[str]]:
        key, elements = keyed_elements
        window_start = int(window.start)
        window_end = int(window.end)

        coder = ExampleProtoEncoder(self.schema)
        output = os.path.join(self.base_path, key, f"{window_start}_{window_end}")
        logger.info(f"Writing {output} with coder {coder}")

        yield (
            elements
            | beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=output,
                num_shards=1,
                shard_name_template="",
                file_name_suffix=".tfrecords.gz",
                coder=coder,
            )
        )


# TODO beam parquetio pyarrow serializer only handles basic Python types, hence list() on itertuples zip generator below
# pyarrow.lib.ArrowTypeError: Could not convert <zip object at 0x7f578011ef80> with type zip: was not a sequence or recognized null for conversion to list type [while running 'WriteToParquet/Write/WriteImpl/WriteBundles']
SERIALIZE_FNS = {
    pd.DataFrame: lambda x: list(tuple(n) for n in x.itertuples(name=None)),
    pd.Series: lambda x: list(x),
    np.polynomial.polynomial.Polynomial: lambda x: list(x),
}


class WriteWindowedParquet(beam.DoFn):
    def __init__(self, base_path: str, schema):
        self.base_path = base_path
        self.schema = schema

    def process(
        self,
        keyed_elements: Tuple[Any, Iterable[NamedTuple]] = beam.DoFn.ElementParam,
        window=beam.DoFn.WindowParam,
    ) -> Iterable[Iterable[str]]:
        key, elements = keyed_elements
        window_start = int(window.start)
        window_end = int(window.end)

        output_path = os.path.join(
            self.base_path, key, f"{window_start}_{window_end}.parquet"
        )
        # @todo apache-beam == 2.28
        # Transforms in beam.io.parquetio only operate on dict representations of data
        yield elements | beam.Map(
            lambda e: e.to_dict()
        ) | beam.io.parquetio.WriteToParquet(
            output_path, self.schema, num_shards=1, shard_name_template=""
        )
