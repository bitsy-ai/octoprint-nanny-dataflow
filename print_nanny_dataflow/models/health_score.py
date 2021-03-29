import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def health_score_trend_polynormial_v1(
    df: pd.DataFrame, degree=1
) -> np.polynomial.polynomial.Polynomial:
    """
    Takes a pandas DataFrame of WindowedHealthRecords and returns a polynormial fit to degree
    """
    xy = (
        df[df["health_multiplier"] > 0]
        .groupby(["window_start"])["health_score"]
        .max()
        .add(
            df[df["health_multiplier"] < 0]
            .groupby(["window_start"])["health_score"]
            .min(),
            fill_value=0,
        )
    )

    logger.info(f"Calculating polyfit with degree={degree} on df: \n {xy}")
    trend = np.polynomial.polynomial.Polynomial.fit(xy.index, xy, degree)
    return trend
