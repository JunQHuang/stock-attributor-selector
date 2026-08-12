from __future__ import annotations

import pandas as pd
import pytest

from examples.run_demo import make_synthetic_panel
from multifactor_demo import (
    DEMO_FACTORS,
    backtest_top_n,
    calculate_factor_frame,
    chronological_split,
    make_supervised_dataset,
    train_and_predict,
)


def test_factor_calculation_stays_inside_each_symbol() -> None:
    panel = make_synthetic_panel(symbols=2, periods=80)
    featured = calculate_factor_frame(panel)

    assert set(DEMO_FACTORS).issubset(featured.columns)
    for _, symbol_data in featured.groupby("symbol"):
        first_row = symbol_data.sort_values("date").iloc[0]
        assert pd.isna(first_row["alpha_007"])
        assert pd.isna(first_row["alpha_022"])


def test_duplicate_symbol_date_is_rejected() -> None:
    panel = make_synthetic_panel(symbols=1, periods=40)
    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        calculate_factor_frame(duplicated)


def test_end_to_end_pipeline() -> None:
    panel = make_synthetic_panel(symbols=12, periods=180)
    dataset = make_supervised_dataset(panel, forward_periods=5)
    split = chronological_split(dataset)
    _, scored = train_and_predict(split)
    returns, summary = backtest_top_n(scored, top_n=3)

    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
    assert scored["score"].notna().all()
    assert len(returns) == scored["date"].nunique()
    assert summary["periods"] == float(len(returns))
