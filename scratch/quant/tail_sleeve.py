"""Tail-protection sleeve: short SPX when 30d realized vol breaks out
above a walk-forward 80th-percentile threshold. Hold for 5 days (or
until vol drops below median).

Designed as explicit insurance:
  - Made +1.50 Sharpe in 2022 (defensive agent's S5)
  - Loses in calm years (-1.19 OOS Sharpe)
  - Add at 2-3% portfolio weight for tail protection
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles
from alphabeta.backtest import backtest

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05


def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2: continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0
        out[f"{tag}_ret"] = ar
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


def build_tail_signal(df: pd.DataFrame) -> pd.Series:
    """SPX vol-breakout short. -1 when rv30 > rolling-252d-trailing 80th pctile.
    Walk-forward: each day's threshold uses only data strictly before it.
    Hold for 5 days max; exit if vol drops below trailing median.
    """
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv30"] = df["ret"].rolling(30).std()
    # Rolling 252-day p80 and p50 (walk-forward)
    df["p80"] = df["rv30"].rolling(252, min_periods=126).quantile(0.80).shift(1)
    df["p50"] = df["rv30"].rolling(252, min_periods=126).quantile(0.50).shift(1)
    df["rv30_lag"] = df["rv30"].shift(1)

    pos = pd.Series(0.0, index=df.index)
    in_trade = False
    bars_held = 0
    for i in range(len(df)):
        if pd.isna(df["p80"].iloc[i]) or pd.isna(df["rv30_lag"].iloc[i]):
            continue
        if not in_trade:
            if df["rv30_lag"].iloc[i] > df["p80"].iloc[i]:
                pos.iloc[i] = -1.0
                in_trade = True
                bars_held = 1
        else:
            # Already short — check exit conditions
            if df["rv30_lag"].iloc[i] < df["p50"].iloc[i] or bars_held >= 5:
                in_trade = False
                bars_held = 0
            else:
                pos.iloc[i] = -1.0
                bars_held += 1
    return pos


def _scale_is(df_full, build_position, target):
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df["timestamp"].values)))
    return target/av if av > 1e-9 else 0


def main():
    df = get_candles("SPX500_USD", "D1")
    scale = _scale_is(df, build_tail_signal, TARGET_VOL)
    print(f"Scale to 5% IS vol: {scale:.2f}")
    pos = pd.Series(build_tail_signal(df).values, index=df.index) * scale
    res = backtest(df, pos, symbol="SPX500_USD", timeframe="D1",
                   name="TAIL_VOL_SHORT")
    idx = pd.to_datetime(df["timestamp"].values, utc=True)
    rets = pd.Series(res.returns.values, index=idx)

    # Stats
    s = stats("TAIL", rets)
    print(f"{'Period':<6} {'Sharpe':>7} {'Ret':>7} {'DD':>7}")
    for tag in ["FULL", "IS", "OOS"]:
        sh = s.get(f"{tag}_sharpe", 0); rt = s.get(f"{tag}_ret", 0); dd = s.get(f"{tag}_dd", 0)
        print(f"{tag:<6} {sh:>+7.2f} {rt:>+7.2%} {dd:>+7.2%}")

    # Year-by-year
    print(f"\nYear-by-year Sharpe + return:")
    for year, sub in rets.groupby(rets.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60: continue
        bpy = len(sub) / span * 365.25
        ar = sub.mean() * bpy
        av = sub.std(ddof=0) * np.sqrt(bpy)
        sh = ar / av if av > 0 else 0
        print(f"  {year}  Sharpe={sh:+5.2f}  Return={ar:+.2%}")

    # Save
    pd.DataFrame({"timestamp": idx, "ret": res.returns.values}).to_parquet(
        OUT / "tail_returns.parquet", index=False)
    print(f"\nSaved → {OUT / 'tail_returns.parquet'}")


if __name__ == "__main__":
    main()
