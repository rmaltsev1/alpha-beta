"""XAU evening: hour 23 only vs 21-23 block vs 22-23 block, all weekdays + Sun."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles
from alphabeta.backtest import backtest, split_is_oos

SPLIT_DATE = "2024-01-01"
TARGET_VOL = 0.05


def _bpy(df):
    s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    return len(df) / s * 365.25 if s > 0 else 252.0


def _scale(df_is, pos_is, target):
    ret = np.log(df_is["close"] / df_is["close"].shift(1)).fillna(0)
    raw = pos_is.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(df_is)))
    return target/av if av > 1e-9 else 0


def trial(label, build_pos):
    df = get_candles("XAU_USD", "H1")
    is_df, oos_df = split_is_oos(df, SPLIT_DATE)
    is_pos = pd.Series(build_pos(is_df).values, index=is_df.index)
    scale = _scale(is_df, is_pos, TARGET_VOL)
    out = []
    for tag, frame in [("FULL", df), ("IS", is_df), ("OOS", oos_df)]:
        pos = pd.Series(build_pos(frame).values, index=frame.index) * scale
        res = backtest(frame, pos, symbol="XAU_USD", timeframe="H1", name=label)
        s = res.stats
        out.append((label, tag, scale, s["exposure"], s["ann_return"], s["ann_vol"],
                    s["sharpe"], s["max_dd"], s["n_trades"]))
    return out


def main():
    def block(hours, weekdays=range(0, 5)):
        def fn(df):
            ts = df["timestamp"]
            pos = pd.Series(0.0, index=df.index)
            pos[ts.dt.hour.isin(hours) & ts.dt.weekday.isin(weekdays)] = 1.0
            return pos
        return fn

    rows = []
    rows += trial("h23_only_wk",  block([23], range(0, 5)))
    rows += trial("h22_only_wk",  block([22], range(0, 5)))
    rows += trial("h23_only_all", block([23], range(0, 7)))    # include Sun (t=1.76)
    rows += trial("h21_23_wk",    block([21, 22, 23], range(0, 5)))
    rows += trial("h22_23_wk",    block([22, 23], range(0, 5)))
    rows += trial("h21_only_wk",  block([21], range(0, 5)))

    df = pd.DataFrame(rows, columns=["label","period","scale","exp","ret","vol","sharpe","dd","trades"])
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={
        "scale": lambda x: f"{x:5.2f}", "exp": lambda x: f"{x:.1%}",
        "ret": lambda x: f"{x:+.1%}", "vol": lambda x: f"{x:.1%}",
        "sharpe": lambda x: f"{x:+.2f}", "dd": lambda x: f"{x:+.1%}"}))


if __name__ == "__main__":
    main()
