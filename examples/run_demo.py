"""Run the public pipeline with deterministic, entirely synthetic market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from multifactor_demo import (
    backtest_top_n,
    chronological_split,
    make_supervised_dataset,
    train_and_predict,
)


def make_synthetic_panel(symbols: int = 24, periods: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    rows: list[pd.DataFrame] = []

    for number in range(symbols):
        market = rng.normal(0.0003, 0.012, periods)
        idiosyncratic = rng.normal(0.0, 0.008 + number * 0.00005, periods)
        close = 30.0 * np.exp(np.cumsum(market + idiosyncratic))
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
    split = chronological_split(dataset)
    _, scored = train_and_predict(split)
    _, summary = backtest_top_n(scored, top_n=5)

    print("Synthetic demo completed")
    print(f"train rows: {len(split.train):,}")
    print(f"validation rows: {len(split.validation):,}")
    print(f"test rows: {len(split.test):,}")
    for name, value in summary.items():
        print(f"{name}: {value:.6f}")
    print("Illustration only; synthetic results are not investment performance.")


if __name__ == "__main__":
    main()
