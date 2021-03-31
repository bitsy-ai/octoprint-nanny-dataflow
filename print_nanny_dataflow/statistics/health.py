import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# @todo load dynamically from active experiment
CATEGORY_INDEX = {
    0: {"name": "background", "id": 0, "health_weight": 0},
    1: {"name": "nozzle", "id": 1, "health_weight": 0},
    2: {"name": "adhesion", "id": 2, "health_weight": -0.5},
    3: {"name": "spaghetti", "id": 3, "health_weight": -0.5},
    4: {"name": "print", "id": 4, "health_weight": 1},
    5: {"name": "raftt", "id": 5, "health_weight": 1},
}


def health_score_trend_polynomial_v1(
    df: pd.DataFrame, degree=1
) -> np.polynomial.polynomial.Polynomial:
    """
    Takes a pandas DataFrame of WindowedHealthRecords and returns a polynormial fit to degree
    """
    xy = (
        df[df["health_multiplier"] > 0]
        .groupby(["ts"])["health_score"]
        .max()
        .add(
            df[df["health_multiplier"] < 0].groupby(["ts"])["health_score"].min(),
            fill_value=0,
        )
    )

    logger.info(f"Calculating polyfit with degree={degree} on df: \n {xy}")
    trend = np.polynomial.polynomial.Polynomial.fit(xy.index, xy, degree)
    return trend
