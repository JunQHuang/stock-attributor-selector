"""Factor diagnostics for cross-sectional research."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .factors import DEMO_FACTORS


def daily_information_coefficient(
    frame: pd.DataFrame,
    factor: str,
    target: str = "target_return",
    min_assets: int = 8,
) -> pd.DataFrame:
    """Calculate daily Pearson IC and Spearman RankIC for one factor."""
    if min_assets < 3:
        raise ValueError("min_assets must be at least 3")
    missing = {"date", factor, target} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for date, day in frame[["date", factor, target]].dropna().groupby("date", sort=True):
        if len(day) < min_assets or day[factor].nunique() < 2 or day[target].nunique() < 2:
            continue
        rows.append(
            {
                "date": date,
                "ic": day[factor].corr(day[target], method="pearson"),
                "rank_ic": day[factor].corr(day[target], method="spearman"),
                "assets": len(day),
            }
        )
    return pd.DataFrame(rows, columns=["date", "ic", "rank_ic", "assets"])


def factor_report(
    frame: pd.DataFrame,
    factors: Iterable[str] = DEMO_FACTORS,
    target: str = "target_return",
    min_assets: int = 8,
) -> pd.DataFrame:
    """Summarize IC level, stability, and sign consistency for many factors."""
    rows: list[dict[str, object]] = []
    for factor in factors:
        daily = daily_information_coefficient(frame, factor, target, min_assets)
        ic_std = daily["ic"].std(ddof=1)
        rank_ic_std = daily["rank_ic"].std(ddof=1)
        rows.append(
            {
                "factor": factor,
                "dates": len(daily),
                "ic_mean": daily["ic"].mean(),
                "icir": daily["ic"].mean() / ic_std if ic_std and np.isfinite(ic_std) else np.nan,
                "rank_ic_mean": daily["rank_ic"].mean(),
                "rank_icir": (
                    daily["rank_ic"].mean() / rank_ic_std
                    if rank_ic_std and np.isfinite(rank_ic_std)
                    else np.nan
                ),
                "ic_positive_ratio": (daily["ic"] > 0).mean() if len(daily) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_mean", key=lambda values: values.abs(), ascending=False)


def quantile_return_table(
    frame: pd.DataFrame,
    factor: str,
    quantiles: int = 5,
    target: str = "target_return",
) -> pd.DataFrame:
    """Return mean target returns for within-date factor quantiles."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    missing = {"date", factor, target} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    pieces: list[pd.DataFrame] = []
    for date, day in frame[["date", factor, target]].dropna().groupby("date", sort=True):
        if len(day) < quantiles:
            continue
        ranked = day[factor].rank(method="first")
        labelled = day.copy()
        labelled["quantile"] = pd.qcut(ranked, q=quantiles, labels=False) + 1
        pieces.append(labelled.assign(date=date))

    if not pieces:
        return pd.DataFrame(columns=["quantile", "mean_return", "observations"])
    combined = pd.concat(pieces, ignore_index=True)
    return (
        combined.groupby("quantile", sort=True)[target]
        .agg(mean_return="mean", observations="size")
        .reset_index()
    )
