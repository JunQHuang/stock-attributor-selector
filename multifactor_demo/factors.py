"""A deliberately small subset of reusable price-volume factors.

The formulas mirror the shape of a larger research factor engine while keeping
the public example compact. Every factor is calculated inside one symbol's
time series; callers should use :func:`calculate_factor_frame` for panel data.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}
DEMO_FACTORS = (
    "alpha_002",
    "alpha_004",
    "alpha_005",
    "alpha_007",
    "alpha_015",
    "alpha_022",
)


class AlphaFactorEngine:
    """Calculate a small factor set for one symbol sorted by date."""

    def __init__(self, data: pd.DataFrame) -> None:
        missing = {"open", "high", "low", "close", "volume"} - set(data.columns)
        if missing:
            raise ValueError(f"missing OHLCV columns: {sorted(missing)}")

        self.open = data["open"].astype(float)
        self.high = data["high"].astype(float)
        self.low = data["low"].astype(float)
        self.close = data["close"].astype(float)
        self.volume = data["volume"].astype(float)
        self.returns = self.close.pct_change()

    @staticmethod
    def ts_mean(values: pd.Series, window: int) -> pd.Series:
        return values.rolling(window).mean()

    @staticmethod
    def ts_std(values: pd.Series, window: int) -> pd.Series:
        return values.rolling(window).std()

    @staticmethod
    def ts_sum(values: pd.Series, window: int) -> pd.Series:
        return values.rolling(window).sum()

    def calculate(self, name: str) -> pd.Series:
        method = getattr(self, name, None)
        if method is None or name.startswith("_"):
            raise ValueError(f"unknown factor: {name}")
        return method()

    def alpha_002(self) -> pd.Series:
        """Negative short/long volume moving-average spread."""
        volume_ma_5 = self.ts_mean(self.volume, 5)
        volume_ma_20 = self.ts_mean(self.volume, 20)
        return -(volume_ma_5 - volume_ma_20) / (volume_ma_20 + 1e-8)

    def alpha_004(self) -> pd.Series:
        """Negative 20-period deviation from a close-volume proxy VWAP."""
        value = self.ts_sum(self.close * self.volume, 20)
        volume = self.ts_sum(self.volume, 20)
        vwap_proxy = value / (volume + 1e-8)
        return -(self.close - vwap_proxy) / (vwap_proxy + 1e-8)

    def alpha_005(self) -> pd.Series:
        """Moving-average deviation scaled by return volatility."""
        close_ma_20 = self.ts_mean(self.close, 20)
        volatility_20 = self.ts_std(self.returns, 20)
        return -((self.close - close_ma_20) / (close_ma_20 + 1e-8)) * volatility_20

    def alpha_007(self) -> pd.Series:
        """Fifteen-period short-term reversal."""
        return -self.close.pct_change(15)

    def alpha_015(self) -> pd.Series:
        """Negative average true range relative to close."""
        previous_close = self.close.shift(1)
        true_range = np.maximum(
            self.high - self.low,
            np.maximum((self.high - previous_close).abs(), (self.low - previous_close).abs()),
        )
        atr_14 = self.ts_mean(pd.Series(true_range, index=self.close.index), 14)
        return -atr_14 / (self.close + 1e-8)

    def alpha_022(self) -> pd.Series:
        """Negative average absolute overnight-gap proxy."""
        previous_close = self.close.shift(1)
        gap = (self.open - previous_close) / (previous_close + 1e-8)
        return -self.ts_mean(gap.abs(), 20)


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"missing panel columns: {sorted(missing)}")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    duplicates = data.duplicated(["symbol", "date"])
    if duplicates.any():
        raise ValueError("each symbol/date pair must be unique")
    return data.sort_values(["symbol", "date"]).reset_index(drop=True)


def calculate_factor_frame(
    panel: pd.DataFrame,
    factor_names: Iterable[str] = DEMO_FACTORS,
) -> pd.DataFrame:
    """Return a sorted panel with the requested factors appended."""
    data = _validate_panel(panel)
    names = tuple(factor_names)
    unknown = set(names) - set(DEMO_FACTORS)
    if unknown:
        raise ValueError(f"factors are outside the public demo set: {sorted(unknown)}")

    groups: list[pd.DataFrame] = []
    for _, symbol_data in data.groupby("symbol", sort=False):
        featured = symbol_data.copy()
        engine = AlphaFactorEngine(featured)
        for name in names:
            featured[name] = engine.calculate(name)
        groups.append(featured)

    if not groups:
        return data.assign(**{name: pd.Series(dtype=float) for name in names})
    return pd.concat(groups, ignore_index=True)
