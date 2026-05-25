"""Is the 23:00 UTC effect every weekday or only Monday?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles


def stats_for(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "H1")
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    h23 = df[df["hour"] == 23]
    out = h23.groupby("weekday")["ret"].agg(["mean", "std", "count"])
    out["t"] = out["mean"] / out["std"] * np.sqrt(out["count"])
    out["mean_bps"] = out["mean"] * 1e4
    return out[["mean_bps", "count", "t"]]


def main():
    pd.set_option("display.width", 200)
    for sym in ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR",
                "UK100_GBP", "JP225_USD", "XAU_USD",
                "EUR_USD", "GBP_USD", "USD_JPY"]:
        print(f"\n=== {sym} (hour 23 UTC by weekday) ===")
        s = stats_for(sym)
        print(s.to_string(float_format=lambda x: f"{x:>7.2f}"))


if __name__ == "__main__":
    main()
