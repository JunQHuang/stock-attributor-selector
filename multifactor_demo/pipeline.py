"""Chronological training, cross-sectional selection, and a simple backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .factors import DEMO_FACTORS, calculate_factor_frame


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def make_supervised_dataset(
    panel: pd.DataFrame,
    forward_periods: int = 5,
) -> pd.DataFrame:
    """Append public demo factors and a future-return regression label."""
    if forward_periods < 1:
        raise ValueError("forward_periods must be positive")

    data = calculate_factor_frame(panel)
    data["target_return"] = data.groupby("symbol", sort=False)["close"].transform(
        lambda close: close.shift(-forward_periods) / close - 1.0
    )
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=[*DEMO_FACTORS, "target_return"]).reset_index(drop=True)


def chronological_split(
    dataset: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> TimeSplit:
    """Split complete dates in chronological order to reduce look-ahead risk."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    dates = np.array(sorted(pd.to_datetime(dataset["date"]).unique()))
    if len(dates) < 5:
        raise ValueError("at least five distinct dates are required")

    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(dates) - 1)

    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])

    return TimeSplit(
        train=dataset[dataset["date"].isin(train_dates)].copy(),
        validation=dataset[dataset["date"].isin(validation_dates)].copy(),
        test=dataset[dataset["date"].isin(test_dates)].copy(),
    )


def train_and_predict(split: TimeSplit, random_state: int = 42) -> tuple[HistGradientBoostingRegressor, pd.DataFrame]:
    """Fit a deterministic baseline model and score the untouched test period."""
    if split.train.empty or split.test.empty:
        raise ValueError("train and test sets must not be empty")

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        max_iter=120,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(split.train[list(DEMO_FACTORS)], split.train["target_return"])

    scored = split.test.copy()
    scored["score"] = model.predict(scored[list(DEMO_FACTORS)])
    return model, scored


def _softmax(values: pd.Series) -> pd.Series:
    centered = values - values.max()
    exponent = np.exp(centered)
    return exponent / exponent.sum()


def backtest_top_n(scored: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select each date's Top-N scores and summarize forward-period returns.

    This is intentionally a compact research illustration. It does not model
    order execution, overlapping holdings, transaction costs, or market limits.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    required = {"date", "symbol", "score", "target_return"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing scored columns: {sorted(missing)}")

    selections: list[pd.DataFrame] = []
    for _, day in scored.groupby("date", sort=True):
        selected = day.nlargest(min(top_n, len(day)), "score").copy()
        selected["weight"] = _softmax(selected["score"])
        selections.append(selected)

    if not selections:
        raise ValueError("no rows are available for backtesting")

    holdings = pd.concat(selections, ignore_index=True)
    period_returns = (
        holdings.assign(weighted_return=holdings["weight"] * holdings["target_return"])
        .groupby("date", sort=True)["weighted_return"]
        .sum()
        .rename("strategy_return")
        .reset_index()
    )
    period_returns["cumulative_return"] = (1.0 + period_returns["strategy_return"]).cumprod() - 1.0

    summary = {
        "periods": float(len(period_returns)),
        "mean_period_return": float(period_returns["strategy_return"].mean()),
        "positive_period_ratio": float((period_returns["strategy_return"] > 0).mean()),
        "cumulative_return": float(period_returns["cumulative_return"].iloc[-1]),
    }
    return period_returns, summary
