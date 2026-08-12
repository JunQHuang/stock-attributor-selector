"""A compact but useful library of price-volume factors.

The public factor set is intentionally generic. It demonstrates the structure
of a larger research engine without exposing any production factor selection,
admission thresholds, fitted parameters, or performance claims.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}
DEMO_FACTORS = (
    "alpha_001",
    "alpha_002",
    "alpha_003",
    "alpha_004",
    "alpha_005",
    "alpha_006",
    "alpha_007",
    "alpha_008",
    "alpha_009",
    "alpha_011",
    "alpha_012",
    "alpha_013",
    "alpha_014",
    "alpha_015",
    "alpha_016",
    "alpha_017",
    "alpha_018",
    "alpha_019",
    "alpha_020",
    "alpha_021",
    "alpha_022",
    "alpha_023",
    "alpha_024",
    "alpha_025",
)


class AlphaFactorEngine:
    """Calculate the public factor set for one symbol sorted by date."""

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

    @staticmethod
    def ts_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        return left.rolling(window).corr(right)

    @staticmethod
    def ts_rank(values: pd.Series, window: int) -> pd.Series:
        def percentile_of_last(sample: np.ndarray) -> float:
            less = (sample < sample[-1]).sum()
            less_or_equal = (sample <= sample[-1]).sum()
            return float((less + less_or_equal + 1) / (2.0 * len(sample)))

        return values.rolling(window).apply(percentile_of_last, raw=True)

    @staticmethod
    def decay_linear(values: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        return values.rolling(window).apply(lambda sample: np.dot(sample, weights), raw=True)

    def calculate(self, name: str) -> pd.Series:
        if name not in DEMO_FACTORS:
            raise ValueError(f"factor is outside the public demo set: {name}")
        return getattr(self, name)()

    def alpha_001(self) -> pd.Series:
        """Negative smoothed positive directional movement."""
        upward = self.high - self.high.shift(1)
        downward = self.low.shift(1) - self.low
        positive_dm = np.where((upward > downward) & (upward > 0), upward, 0.0)
        return -self.ts_mean(pd.Series(positive_dm, index=self.close.index), 14)

    def alpha_002(self) -> pd.Series:
        """Negative short/long volume moving-average spread."""
        volume_ma_5 = self.ts_mean(self.volume, 5)
        volume_ma_20 = self.ts_mean(self.volume, 20)
        return -(volume_ma_5 - volume_ma_20) / (volume_ma_20 + 1e-8)

    def alpha_003(self) -> pd.Series:
        """Deviation of linearly decayed volume from its moving average."""
        decayed = self.decay_linear(self.volume, 20)
        average = self.ts_mean(self.volume, 20)
        return -(decayed - average) / (average + 1e-8)

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

    def alpha_006(self) -> pd.Series:
        """Negative lower-shadow size relative to close."""
        lower_shadow = np.minimum(self.open, self.close) - self.low
        return -lower_shadow / (self.close + 1e-8)

    def alpha_007(self) -> pd.Series:
        """Fifteen-period short-term reversal."""
        return -self.close.pct_change(15)

    def alpha_008(self) -> pd.Series:
        """Negative linearly decayed return."""
        return -self.decay_linear(self.returns, 20)

    def alpha_009(self) -> pd.Series:
        """Average time-series rank of reversal and low-volatility signals."""
        reversal = -self.close.pct_change(20)
        low_volatility = -self.ts_std(self.returns, 20)
        return (self.ts_rank(reversal, 60) + self.ts_rank(low_volatility, 60)) / 2.0

    def alpha_011(self) -> pd.Series:
        """Negative smoothed force-index proxy."""
        force = self.returns * self.volume
        return -self.ts_mean(force, 13) / (self.ts_mean(self.volume, 13) + 1e-8)

    def alpha_012(self) -> pd.Series:
        """Negative dispersion around the 20-period moving average."""
        average = self.ts_mean(self.close, 20)
        dispersion = self.ts_std(self.close - average, 20)
        return -dispersion / (average + 1e-8)

    def alpha_013(self) -> pd.Series:
        """Negative volume z-score."""
        average = self.ts_mean(self.volume, 20)
        standard_deviation = self.ts_std(self.volume, 20)
        return -(self.volume - average) / (standard_deviation + 1e-8)

    def alpha_014(self) -> pd.Series:
        """Negative distance between short and long moving averages."""
        average_10 = self.ts_mean(self.close, 10)
        average_30 = self.ts_mean(self.close, 30)
        return -(average_10 - average_30) / (average_30 + 1e-8)

    def alpha_015(self) -> pd.Series:
        """Negative average true range relative to close."""
        previous_close = self.close.shift(1)
        true_range = np.maximum(
            self.high - self.low,
            np.maximum((self.high - previous_close).abs(), (self.low - previous_close).abs()),
        )
        atr_14 = self.ts_mean(pd.Series(true_range, index=self.close.index), 14)
        return -atr_14 / (self.close + 1e-8)

    def alpha_016(self) -> pd.Series:
        """Negative 20-period volume time-series rank."""
        return -self.ts_rank(self.volume, 20)

    def alpha_017(self) -> pd.Series:
        """Negative upper-shadow size relative to close."""
        upper_shadow = self.high - np.maximum(self.open, self.close)
        return -upper_shadow / (self.close + 1e-8)

    def alpha_018(self) -> pd.Series:
        """Negative rolling volume-weighted return."""
        volume_weight = self.volume / (self.ts_sum(self.volume, 20) + 1e-8)
        return -self.ts_sum(self.returns * volume_weight, 20)

    def alpha_019(self) -> pd.Series:
        """Low-turnover stability proxy."""
        turnover_proxy = self.volume / (self.ts_mean(self.volume, 20) + 1e-8)
        instability = self.ts_std(turnover_proxy, 20)
        return -turnover_proxy * instability

    def alpha_020(self) -> pd.Series:
        """Negative correlation between turnover and recent return."""
        turnover_proxy = self.volume / (self.ts_mean(self.volume, 20) + 1e-8)
        return_10 = self.close.pct_change(10)
        return -self.ts_corr(turnover_proxy, return_10, 20)

    def alpha_021(self) -> pd.Series:
        """Price reversal scaled by relative volume."""
        return_20 = self.close.pct_change(20)
        relative_volume = self.volume / (self.ts_mean(self.volume, 20) + 1e-8)
        return -return_20 * relative_volume

    def alpha_022(self) -> pd.Series:
        """Negative average absolute overnight-gap proxy."""
        previous_close = self.close.shift(1)
        gap = (self.open - previous_close) / (previous_close + 1e-8)
        return -self.ts_mean(gap.abs(), 20)

    def alpha_023(self) -> pd.Series:
        """Return accumulated on unusually high-volume observations."""
        high_volume = self.volume > self.ts_mean(self.volume, 20) * 1.5
        conditional_return = np.where(high_volume, self.returns, 0.0)
        series = pd.Series(conditional_return, index=self.close.index)
        return -self.ts_sum(series, 20)

    def alpha_024(self) -> pd.Series:
        """Negative triple-exponential moving-average rate of change."""
        ema_1 = self.close.ewm(span=12, adjust=False).mean()
        ema_2 = ema_1.ewm(span=12, adjust=False).mean()
        ema_3 = ema_2.ewm(span=12, adjust=False).mean()
        return -ema_3.pct_change()

    def alpha_025(self) -> pd.Series:
        """Negative frequency of two-sigma returns."""
        volatility_60 = self.ts_std(self.returns, 60)
        extreme = (self.returns.abs() > 2.0 * volatility_60).astype(float)
        return -self.ts_sum(extreme, 20)


def _validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"missing panel columns: {sorted(missing)}")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    duplicates = data.duplicated(["symbol", "date"])
    if duplicates.any():
        raise ValueError("each symbol/date pair must be unique")

    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (data["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be the largest OHLC value")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be the smallest OHLC value")
    return data.sort_values(["symbol", "date"]).reset_index(drop=True)


def calculate_factor_frame(
    panel: pd.DataFrame,
    factor_names: Iterable[str] = DEMO_FACTORS,
) -> pd.DataFrame:
    """Return a sorted panel with requested public factors appended."""
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
