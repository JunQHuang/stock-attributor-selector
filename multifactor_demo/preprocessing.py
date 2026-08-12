"""Cross-sectional preprocessing that never uses future observations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def winsorize_by_date(
    frame: pd.DataFrame,
    columns: Iterable[str],
    mad_limit: float = 5.0,
) -> pd.DataFrame:
    """Clip each date's factor values with a median absolute-deviation rule."""
    if mad_limit <= 0:
        raise ValueError("mad_limit must be positive")

    result = frame.copy()
    for column in columns:
        if column not in result:
            raise ValueError(f"missing factor column: {column}")

        def clip_group(values: pd.Series) -> pd.Series:
            median = values.median()
            mad = (values - median).abs().median()
            if pd.isna(mad) or mad <= 0:
                return values
            return values.clip(median - mad_limit * mad, median + mad_limit * mad)

        result[column] = result.groupby("date", sort=False)[column].transform(clip_group)
    return result


def zscore_by_date(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Standardize factors independently inside each date's cross-section."""
    result = frame.copy()
    for column in columns:
        if column not in result:
            raise ValueError(f"missing factor column: {column}")
        grouped = result.groupby("date", sort=False)[column]
        mean = grouped.transform("mean")
        standard_deviation = grouped.transform(lambda values: values.std(ddof=0))
        result[column] = (result[column] - mean) / standard_deviation.replace(0, np.nan)
        result[column] = result[column].fillna(0.0)
    return result


def preprocess_factors(
    frame: pd.DataFrame,
    columns: Iterable[str],
    mad_limit: float = 5.0,
) -> pd.DataFrame:
    """Apply robust cross-sectional clipping followed by z-scoring."""
    names = tuple(columns)
    return zscore_by_date(winsorize_by_date(frame, names, mad_limit), names)
