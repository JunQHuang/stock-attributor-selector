"""Run the public research workflow with deterministic synthetic market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from multifactor_demo import (
    backtest_top_n,
    factor_report,
    latest_signal,
    make_supervised_dataset,
    walk_forward_predict,
)


def make_synthetic_panel(symbols: int = 24, periods: int = 360, seed: int = 7) -> pd.DataFrame:
    """Create a reproducible panel with a shared market component."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    common_market = rng.normal(0.0003, 0.009, periods)
    rows: list[pd.DataFrame] = []

    for number in range(symbols):
        idiosyncratic = rng.normal(0.0, 0.007 + number * 0.00005, periods)
        mean_reversion = np.zeros(periods)
        for index in range(1, periods):
            mean_reversion[index] = -0.08 * idiosyncratic[index - 1]
        close = 30.0 * np.exp(np.cumsum(common_market + idiosyncratic + mean_reversion))
        overnight = rng.normal(0.0, 0.003, periods)
        open_price = close * (1.0 + overnight)
        spread = np.abs(rng.normal(0.008, 0.003, periods))

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": f"SYN{number:03d}",
                    "open": open_price,
                    "high": np.maximum(open_price, close) * (1.0 + spread),
                    "low": np.minimum(open_price, close) * (1.0 - spread),
                    "close": close,
                    "volume": rng.lognormal(mean=13.0, sigma=0.35, size=periods),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    panel = make_synthetic_panel()
    dataset = make_supervised_dataset(panel, forward_periods=5)

    diagnostics = factor_report(dataset)
    print("Top factors by absolute synthetic RankIC")
    print(diagnostics.head(5).to_string(index=False))

    scored = walk_forward_predict(
        dataset,
        min_train_dates=140,
        test_block_dates=25,
        max_train_dates=220,
    )
    holdings, returns, summary = backtest_top_n(
        scored,
        top_n=5,
        rebalance_every=5,
        transaction_cost_bps=10.0,
    )
    signal = latest_signal(scored, top_n=5)

    print("\nWalk-forward synthetic demo completed")
    print(f"dataset rows: {len(dataset):,}")
    print(f"out-of-sample rows: {len(scored):,}")
    print(f"rebalance periods: {len(returns):,}")
    print(f"holding rows: {len(holdings):,}")
    for name, value in summary.items():
        print(f"{name}: {value:.6f}")
    print("\nLatest synthetic signal")
    print(signal.to_string(index=False))
    print("\nIllustration only; synthetic results are not investment performance.")


if __name__ == "__main__":
    main()
