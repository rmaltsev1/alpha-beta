"""Per-hour breakdown of the Monday NY effect on the 3 US indices."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles


def per_hour_stats(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "H1")
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    ts = df["timestamp"]
    df["weekday"] = ts.dt.weekday
    df["hour"] = ts.dt.hour
    mon = df[df["weekday"] == 0]
    out = mon.groupby("hour")["ret"].agg(["mean", "std", "count"])
    out["t"] = out["mean"] / out["std"] * np.sqrt(out["count"])
    out["mean_bps"] = out["mean"] * 1e4
    out["std_bps"] = out["std"] * 1e4
    return out[["mean_bps", "std_bps", "count", "t"]]


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 30)
    for sym in ["SPX500_USD", "NAS100_USD", "US30_USD"]:
        print(f"\n=== {sym} (H1, Mondays only) ===")
        out = per_hour_stats(sym)
        print(out.to_string(float_format=lambda x: f"{x:>7.2f}"))


if __name__ == "__main__":
    main()
