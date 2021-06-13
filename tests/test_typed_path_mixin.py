from print_nanny_dataflow.transforms.io import TypedPathMixin


def test_path_no_window():
    bucket = "test-bucket"
    base_path = "dataflow/base/path"
    key = "abcd1234"
    datesegment = "2021/01/01"
    module = "module_pb.StructName"
    ext = "jpg"
    filename = f"1234_1239.{ext}"
    window_type = "fixed"

    actual = TypedPathMixin().path(
        bucket=bucket,
        base_path=base_path,
        key=key,
        datesegment=datesegment,
        module=module,
        ext=ext,
        filename=filename,
        window_type=window_type,
    )
    expected = f"gs://{bucket}/{base_path}/{module}/{window_type}/{datesegment}/{key}/{ext}/{filename}"
    assert actual == expected


def test_path_nondefaults():
    bucket = "test-bucket"
    base_path = "dataflow/base/path"
    key = "abcd1234"
    datesegment = "2021/01/01"
    module = "module_pb.StructName"
    ext = "jpg"
    protocol = "s3://"
    window_type = "fixed"
    filename = f"1234_1239.{ext}"
    actual = TypedPathMixin().path(
        bucket=bucket,
        base_path=base_path,
        key=key,
        datesegment=datesegment,
        module=module,
        ext=ext,
        filename=filename,
        protocol=protocol,
        window_type=window_type,
    )
    expected = f"{protocol}{bucket}/{base_path}/{module}/{window_type}/{datesegment}/{key}/{ext}/{filename}"
    assert actual == expected
