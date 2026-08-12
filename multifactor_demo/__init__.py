"""Minimal, privacy-safe multi-factor research example."""

from .factors import DEMO_FACTORS, AlphaFactorEngine, calculate_factor_frame
from .pipeline import (
    backtest_top_n,
    chronological_split,
    make_supervised_dataset,
    train_and_predict,
)

__all__ = [
    "DEMO_FACTORS",
    "AlphaFactorEngine",
    "calculate_factor_frame",
    "make_supervised_dataset",
    "chronological_split",
    "train_and_predict",
    "backtest_top_n",
]
