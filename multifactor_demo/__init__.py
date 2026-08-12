"""Privacy-safe multi-factor research toolkit."""

from .evaluation import daily_information_coefficient, factor_report, quantile_return_table
from .factors import DEMO_FACTORS, AlphaFactorEngine, calculate_factor_frame
from .pipeline import (
    TimeSplit,
    backtest_top_n,
    chronological_split,
    latest_signal,
    make_model,
    make_supervised_dataset,
    train_and_predict,
    walk_forward_predict,
)
from .preprocessing import preprocess_factors, winsorize_by_date, zscore_by_date

__all__ = [
    "DEMO_FACTORS",
    "AlphaFactorEngine",
    "TimeSplit",
    "calculate_factor_frame",
    "winsorize_by_date",
    "zscore_by_date",
    "preprocess_factors",
    "daily_information_coefficient",
    "factor_report",
    "quantile_return_table",
    "make_supervised_dataset",
    "chronological_split",
    "make_model",
    "train_and_predict",
    "walk_forward_predict",
    "latest_signal",
    "backtest_top_n",
]
