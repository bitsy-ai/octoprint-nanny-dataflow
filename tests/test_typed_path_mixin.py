from print_nanny_dataflow.transforms.io import TypedPathMixin


def test_path_no_window():
    bucket = "test-bucket"
    base_path = "dataflow/base/path"
    key = "abcd1234"
    datesegment = "2021/01/01"
    module = "module_pb"
    struct = "StructName"
    ext = "jpg"
    actual = TypedPathMixin().path(
        bucket=bucket,
        base_path=base_path,
        key=key,
        datesegment=datesegment,
        module=module,
        struct=struct,
        ext=ext,
    )
    expected = f"gs://{bucket}/{base_path}/{module}.{struct}/{datesegment}/{key}/{ext}"
    assert actual == expected


def test_path_with_window():
    bucket = "test-bucket"
    base_path = "dataflow/base/path"
    key = "abcd1234"
    datesegment = "2021/01/01"
    module = "module_pb"
    struct = "StructName"
    ext = "jpg"
    window = (1234, 1239)
    actual = TypedPathMixin().path(
        bucket=bucket,
        base_path=base_path,
        key=key,
        datesegment=datesegment,
        module=module,
        struct=struct,
        ext=ext,
        window=window,
    )
    expected = f"gs://{bucket}/{base_path}/{module}.{struct}/{datesegment}/{key}/{ext}/{window[0]}_{window[1]}"
    assert actual == expected


def test_path_nondefault_protocol_window():
    bucket = "test-bucket"
    base_path = "dataflow/base/path"
    key = "abcd1234"
    datesegment = "2021/01/01"
    module = "module_pb"
    struct = "StructName"
    ext = "jpg"
    window = (1234, 1239)
    protocol = "s3://"
    actual = TypedPathMixin().path(
        bucket=bucket,
        base_path=base_path,
        key=key,
        datesegment=datesegment,
        module=module,
        struct=struct,
        ext=ext,
        window=window,
        protocol=protocol,
    )
    expected = f"{protocol}{bucket}/{base_path}/{module}.{struct}/{datesegment}/{key}/{ext}/{window[0]}_{window[1]}"
    assert actual == expected


def test_path_nondefault_protocol():
    bucket = "test-bucket"
    base_path = "dataflow/base/path"
    key = "abcd1234"
    datesegment = "2021/01/01"
    module = "module_pb"
    struct = "StructName"
    ext = "jpg"
    protocol = "s3://"
    actual = TypedPathMixin().path(
        bucket=bucket,
        base_path=base_path,
        key=key,
        datesegment=datesegment,
        module=module,
        struct=struct,
        ext=ext,
        protocol=protocol,
    )
    expected = (
        f"{protocol}{bucket}/{base_path}/{module}.{struct}/{datesegment}/{key}/{ext}"
    )
    assert actual == expected
