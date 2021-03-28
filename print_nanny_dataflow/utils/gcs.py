from typing import Iterable


def windows_by_interval(
    window_size: int, window_period: int, window_cursor: int
) -> Iterable[str]:
    """
    Outputs filenames for each window, looking back window_size from window_cursor
    """
    n_windows = window_size // window_period
    return (window_cursor - (window_period * n) for n in range(1, n_windows))
