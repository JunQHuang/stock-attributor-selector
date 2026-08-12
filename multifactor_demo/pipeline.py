"""Leakage-aware modelling, signal construction, and cost-aware backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .factors import DEMO_FACTORS, calculate_factor_frame
from .preprocessing import preprocess_factors


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def make_supervised_dataset(
    panel: pd.DataFrame,
    forward_periods: int = 5,
    normalize: bool = True,
) -> pd.DataFrame:
    """Append factors, target return, and the date on which the target is known."""
    if forward_periods < 1:
        raise ValueError("forward_periods must be positive")

    data = calculate_factor_frame(panel)
    grouped = data.groupby("symbol", sort=False)
    data["target_return"] = grouped["close"].transform(
        lambda close: close.shift(-forward_periods) / close - 1.0
    )
    data["target_date"] = grouped["date"].shift(-forward_periods)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[*DEMO_FACTORS, "target_return", "target_date"]).reset_index(drop=True)
    if normalize:
        data = preprocess_factors(data, DEMO_FACTORS)
    return data


def chronological_split(
    dataset: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> TimeSplit:
    """Split complete dates and purge labels that cross either boundary."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if "target_date" not in dataset:
        raise ValueError("dataset must contain target_date for leakage purging")

    dates = np.array(sorted(pd.to_datetime(dataset["date"]).unique()))
    if len(dates) < 5:
        raise ValueError("at least five distinct dates are required")

    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(dates) - 1)
    validation_start = pd.Timestamp(dates[train_end])
    test_start = pd.Timestamp(dates[validation_end])

    train = dataset[
        (dataset["date"] < validation_start) & (dataset["target_date"] < validation_start)
    ].copy()
    validation = dataset[
        (dataset["date"] >= validation_start)
        & (dataset["date"] < test_start)
        & (dataset["target_date"] < test_start)
    ].copy()
    test = dataset[dataset["date"] >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("purged time split produced an empty partition")
    return TimeSplit(train=train, validation=validation, test=test)


def make_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Create the intentionally conservative public baseline model."""
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        max_iter=120,
        l2_regularization=1.0,
        random_state=random_state,
    )


def train_and_predict(
    split: TimeSplit,
    random_state: int = 42,
) -> tuple[HistGradientBoostingRegressor, pd.DataFrame]:
    """Fit the baseline model and score the untouched test partition."""
    if split.train.empty or split.test.empty:
        raise ValueError("train and test sets must not be empty")

    model = make_model(random_state)
    model.fit(split.train[list(DEMO_FACTORS)], split.train["target_return"])
    scored = split.test.copy()
    scored["score"] = model.predict(scored[list(DEMO_FACTORS)])
    return model, scored


def walk_forward_predict(
    dataset: pd.DataFrame,
    min_train_dates: int = 120,
    test_block_dates: int = 20,
    max_train_dates: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run expanding- or rolling-window out-of-sample prediction.

    A row can enter a training fold only if its ``target_date`` is strictly
    earlier than the first test date, which purges forward-label leakage.
    """
    if min_train_dates < 20:
        raise ValueError("min_train_dates must be at least 20")
    if test_block_dates < 1:
        raise ValueError("test_block_dates must be positive")
    if max_train_dates is not None and max_train_dates < min_train_dates:
        raise ValueError("max_train_dates cannot be below min_train_dates")
    required = {"date", "target_date", "target_return", *DEMO_FACTORS}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"missing modelling columns: {sorted(missing)}")

    dates = np.array(sorted(pd.to_datetime(dataset["date"]).unique()))
    if len(dates) <= min_train_dates:
        raise ValueError("dataset does not contain enough dates for walk-forward prediction")

    folds: list[pd.DataFrame] = []
    for fold_number, start in enumerate(range(min_train_dates, len(dates), test_block_dates)):
        test_dates = dates[start : start + test_block_dates]
        test_start = pd.Timestamp(test_dates[0])
        train = dataset[
            (dataset["date"] < test_start) & (dataset["target_date"] < test_start)
        ].copy()
        if max_train_dates is not None:
            available = np.array(sorted(train["date"].unique()))
            retained = set(available[-max_train_dates:])
            train = train[train["date"].isin(retained)]

        test = dataset[dataset["date"].isin(set(test_dates))].copy()
        if train.empty or test.empty:
            continue
        model = make_model(random_state + fold_number)
        model.fit(train[list(DEMO_FACTORS)], train["target_return"])
        test["score"] = model.predict(test[list(DEMO_FACTORS)])
        test["fold"] = fold_number
        test["training_cutoff"] = train["target_date"].max()
        folds.append(test)

    if not folds:
        raise ValueError("walk-forward prediction produced no test folds")
    return pd.concat(folds, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _softmax(values: pd.Series, temperature: float = 1.0) -> pd.Series:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = (values - values.max()) / temperature
    exponent = np.exp(scaled)
    return exponent / exponent.sum()


def latest_signal(
    scored: pd.DataFrame,
    top_n: int = 5,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Build a deterministic Top-N signal for the most recent scored date."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    required = {"date", "symbol", "score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing scored columns: {sorted(missing)}")
    if scored.empty:
        raise ValueError("scored data must not be empty")

    latest_date = scored["date"].max()
    selected = scored[scored["date"] == latest_date].nlargest(top_n, "score").copy()
    selected["weight"] = _softmax(selected["score"], temperature)
    return selected[["date", "symbol", "score", "weight"]].reset_index(drop=True)


def backtest_top_n(
    scored: pd.DataFrame,
    top_n: int = 5,
    rebalance_every: int = 5,
    transaction_cost_bps: float = 10.0,
    temperature: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Backtest Top-N target weights with turnover-based transaction costs.

    The calculation uses forward-period labels and is a research diagnostic,
    not an execution simulator. Set ``rebalance_every`` equal to the label
    horizon to avoid overlapping holding periods in the example.
    """
    if top_n < 1 or rebalance_every < 1:
        raise ValueError("top_n and rebalance_every must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    required = {"date", "symbol", "score", "target_return"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing scored columns: {sorted(missing)}")

    dates = np.array(sorted(pd.to_datetime(scored["date"]).unique()))
    rebalance_dates = dates[::rebalance_every]
    holdings_rows: list[pd.DataFrame] = []
    return_rows: list[dict[str, object]] = []
    previous = pd.Series(dtype=float)

    for date in rebalance_dates:
        day = scored[scored["date"] == date]
        if day.empty:
            continue
        selected = day.nlargest(min(top_n, len(day)), "score").copy()
        selected["weight"] = _softmax(selected["score"], temperature)
        current = selected.set_index("symbol")["weight"]
        union = previous.index.union(current.index)
        turnover = float(
            (current.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0)).abs().sum()
        )
        gross_return = float((selected["weight"] * selected["target_return"]).sum())
        cost = turnover * transaction_cost_bps / 10_000.0
        net_return = gross_return - cost

        holdings_rows.append(selected.assign(turnover=turnover))
        return_rows.append(
            {
                "date": pd.Timestamp(date),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "net_return": net_return,
            }
        )
        previous = current

    if not return_rows:
        raise ValueError("no rows are available for backtesting")

    holdings = pd.concat(holdings_rows, ignore_index=True)
    returns = pd.DataFrame(return_rows)
    returns["equity"] = (1.0 + returns["net_return"]).cumprod()
    returns["cumulative_return"] = returns["equity"] - 1.0
    running_peak = returns["equity"].cummax()
    returns["drawdown"] = returns["equity"] / running_peak - 1.0

    summary = {
        "periods": float(len(returns)),
        "mean_gross_return": float(returns["gross_return"].mean()),
        "mean_net_return": float(returns["net_return"].mean()),
        "positive_period_ratio": float((returns["net_return"] > 0).mean()),
        "average_turnover": float(returns["turnover"].mean()),
        "cumulative_return": float(returns["cumulative_return"].iloc[-1]),
        "max_drawdown": float(returns["drawdown"].min()),
    }
    return holdings, returns, summary
