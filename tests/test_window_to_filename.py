import pytest
from datetime import datetime

from print_nanny_dataflow.utils.gcs import windows_by_interval


def test_window_to_filename_defaults():

    window_size = 10 * 60  # 10 minutes
    window_period = 30  # 30 seconds
    window_cursor = 1616379416

    output = windows_by_interval(
        window_size=window_size,
        window_period=window_period,
        window_cursor=window_cursor,
    )

    assert tuple(output) == (
        1616379386,
        1616379356,
        1616379326,
        1616379296,
        1616379266,
        1616379236,
        1616379206,
        1616379176,
        1616379146,
        1616379116,
        1616379086,
        1616379056,
        1616379026,
        1616378996,
        1616378966,
        1616378936,
        1616378906,
        1616378876,
        1616378846,
    )
