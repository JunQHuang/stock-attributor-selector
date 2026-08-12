from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from examples.run_demo import make_synthetic_panel
from multifactor_demo import (
    DEMO_FACTORS,
    backtest_top_n,
    calculate_factor_frame,
    chronological_split,
    factor_report,
    latest_signal,
    make_supervised_dataset,
    train_and_predict,
    walk_forward_predict,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    panel = make_synthetic_panel(symbols=12, periods=220)
    return make_supervised_dataset(panel, forward_periods=5)


def test_public_factor_set_is_substantial_and_symbol_isolated() -> None:
    panel = make_synthetic_panel(symbols=2, periods=100)
    featured = calculate_factor_frame(panel)

    assert len(DEMO_FACTORS) == 24
    assert set(DEMO_FACTORS).issubset(featured.columns)
    for _, symbol_data in featured.groupby("symbol"):
        first_row = symbol_data.sort_values("date").iloc[0]
        assert pd.isna(first_row["alpha_007"])
        assert pd.isna(first_row["alpha_022"])


def test_invalid_panel_is_rejected() -> None:
    panel = make_synthetic_panel(symbols=1, periods=80)
    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        calculate_factor_frame(duplicated)

    invalid_ohlc = panel.copy()
    invalid_ohlc.loc[0, "high"] = invalid_ohlc.loc[0, "low"]
    with pytest.raises(ValueError, match="high"):
        calculate_factor_frame(invalid_ohlc)


def test_cross_sectional_factors_are_standardized(dataset: pd.DataFrame) -> None:
    means = dataset.groupby("date")[list(DEMO_FACTORS)].mean().abs()
    assert float(means.to_numpy().max()) < 1e-10


def test_factor_diagnostics_cover_every_public_factor(dataset: pd.DataFrame) -> None:
    report = factor_report(dataset, min_assets=8)
    assert set(report["factor"]) == set(DEMO_FACTORS)
    assert (report["dates"] > 0).all()
    assert report["rank_ic_mean"].notna().all()


def test_purged_split_and_fixed_model(dataset: pd.DataFrame) -> None:
    split = chronological_split(dataset)
    _, scored = train_and_predict(split)

    assert split.train["target_date"].max() < split.validation["date"].min()
    assert split.validation["target_date"].max() < split.test["date"].min()
    assert scored["score"].notna().all()


def test_walk_forward_backtest_and_signal(dataset: pd.DataFrame) -> None:
    scored = walk_forward_predict(dataset, min_train_dates=70, test_block_dates=20)
    assert scored["score"].notna().all()
    assert (scored["training_cutoff"] < scored["date"]).all()

    holdings, returns, summary = backtest_top_n(
        scored,
        top_n=3,
        rebalance_every=5,
        transaction_cost_bps=10.0,
    )
    assert not holdings.empty
    assert np.allclose(returns["net_return"], returns["gross_return"] - returns["transaction_cost"])
    assert (returns["transaction_cost"] >= 0).all()
    assert summary["periods"] == float(len(returns))

    signal = latest_signal(scored, top_n=3)
    assert len(signal) == 3
    assert signal["date"].nunique() == 1
    assert signal["weight"].sum() == pytest.approx(1.0)
