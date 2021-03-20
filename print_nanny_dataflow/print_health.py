import pandas as pd
import numpy as np
import typing

from types import BoundingBoxAnnotation, FlatTelemetryEvent

class PolyfitHealthModel:
    LABELS = {
        1: "nozzle",
        2: "adhesion",
        3: "spaghetti",
        4: "print",
        5: "raft",
    }

    NEUTRAL_LABELS = {1: "nozzle", 5: "raft"}

    NEGATIVE_LABELS = {
        2: "adhesion",
        3: "spaghetti",
    }

    POSITIVE_LABELS = {
        4: "print",
    }

    def __init__(self, df=typing[pd.DataFrame] = None, unhealthy_threshold=3):
        if df is None:
            df = pd.DataFrame()

        self.df = df
        self.trend_observations = []
        self.unhealthy_count = 0
    
    def add(self, telemetry_event: FlatTelemetryEvent) -> pd.DataFrame:
        df = self.explode_nested_annotation(telemetry_event)
        self.df = self.df.append(df)
        return self.df

    def explode_nested_annotation(
            self, ts: int, telemetry_event: FlatTelemetryEvent
        ) -> pd.DataFrame:

        data = {
            "frame_id": ts, 
            "scores": telemetry_event.scores, 
            "num_detections": telemetry_event.num_detections,
            "classes": telemetry_event.classes
        }
        df = pd.DataFrame.from_records([data])

        df = df[["frame_id", "classes", "scores"]]
        df = df.reset_index()

        NUM_FRAMES = len(df)

        # explode nested scores and classes series
        df = df.set_index(["frame_id"]).apply(pd.Series.explode).reset_index()
        assert len(df) == NUM_FRAMES * annotation.num_detections

        # add string labels
        df["label"] = df["classes"].map(self.LABELS)

        # create a hierarchal index from exploded data, append to dataframe state
        return df.set_index(["frame_id", "label"])


    def trend(degree: int = 1) -> float:
        if df.empty:
            return 0.0
        df = pd.concat(
            {
                "unhealthy": self.df[self.df["classes"].isin(self.NEGATIVE_LABELS)],
                "healthy": self.df[self.df["classes"].isin(self.POSITIVE_LABELS)],
            }
        ).reset_index()

        mask = df.level_0 == "unhealthy"
        healthy_cumsum = np.log10(
            df[~mask].groupby("frame_id")["scores"].sum().cumsum()
        )

        unhealthy_cumsum = np.log10(
            df[mask].groupby("frame_id")["scores"].sum().cumsum()
        )

        xy = healthy_cumsum.subtract(unhealthy_cumsum, fill_value=0)

        if len(xy) == 1:
            return 0.0

        return np.polynomial.polynomial.Polynomial.fit(xy.index, xy, degree)

    def health_check(self):
        return self.unhealthy_count >= self.unhealthy_threshold
