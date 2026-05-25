"""Find the cleanest EUR_USD and GBP_USD intraday signals."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles


def per_hour(symbol):
    df = get_candles(symbol, "H1")
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    wd = df[df["weekday"] < 5]
    out = wd.groupby("hour")["ret"].agg(["mean", "std", "count"])
    out["t"] = out["mean"] / out["std"] * np.sqrt(out["count"])
    out["mean_bps"] = out["mean"] * 1e4
    return out[["mean_bps", "count", "t"]]


def main():
    pd.set_option("display.width", 200)
    for sym in ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]:
        print(f"\n=== {sym} (weekdays only, by hour UTC) ===")
        s = per_hour(sym)
        print(s.to_string(float_format=lambda x: f"{x:>7.2f}"))


if __name__ == "__main__":
    main()
