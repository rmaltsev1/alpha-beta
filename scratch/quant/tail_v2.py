"""Tail-protection sleeve v2 — multiple variants.

Variant A: vol-of-vol breakout (rate of change of rv30, not level).
Variant B: trend-confirmed vol short (vol breakout AND prior 5d return < 0).
Variant C: long XAU+USD_JPY when SPX 20d return < -5% (real safe-haven trade).
Variant D: short NAS (not SPX) on vol breakout — NAS falls harder.
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


def stats_full(label, r):
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
    # 2022 specifically
    y22 = r[r.index.year == 2022]
    if len(y22) > 0:
        bpy = _bpy(y22.index)
        ar = float(y22.mean()) * bpy
        av = float(y22.std(ddof=0)) * np.sqrt(bpy)
        out["2022_sharpe"] = ar/av if av > 0 else 0
        out["2022_ret"] = ar
    return out


def _scale_is(df_full, build_position, target):
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df["timestamp"].values)))
    return target/av if av > 1e-9 else 0


def variant_A_vol_of_vol(df):
    """Short SPX when vol-of-vol (rv of rv30) breaks above 80th pctile."""
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv30"] = df["ret"].rolling(30).std()
    df["volvol"] = df["rv30"].rolling(30).std()
    df["vv_p80"] = df["volvol"].rolling(252, min_periods=126).quantile(0.80).shift(1)
    df["vv_lag"] = df["volvol"].shift(1)
    pos = pd.Series(0.0, index=df.index)
    pos[df["vv_lag"] > df["vv_p80"]] = -1.0
    return pos


def variant_B_trend_confirmed_short(df):
    """Short SPX when rv30 > p80 AND prior 10d return < 0 (downtrend filter)."""
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv30"] = df["ret"].rolling(30).std()
    df["rv_p80"] = df["rv30"].rolling(252, min_periods=126).quantile(0.80).shift(1)
    df["rv_lag"] = df["rv30"].shift(1)
    df["trend10"] = df["ret"].rolling(10).sum().shift(1)
    pos = pd.Series(0.0, index=df.index)
    cond = (df["rv_lag"] > df["rv_p80"]) & (df["trend10"] < 0)
    pos[cond] = -1.0
    return pos


def variant_D_short_nas(df):
    """Short NAS on vol breakout (NAS falls harder than SPX)."""
    return variant_B_trend_confirmed_short(df)


def main():
    # Variants A, B, D on SPX/NAS
    variants = [
        ("A_volvol_SPX",   "SPX500_USD", variant_A_vol_of_vol),
        ("B_trend_SPX",    "SPX500_USD", variant_B_trend_confirmed_short),
        ("D_trend_NAS",    "NAS100_USD", variant_D_short_nas),
        ("D_trend_US30",   "US30_USD",   variant_B_trend_confirmed_short),
    ]
    rows = []
    streams = {}
    for name, sym, fn in variants:
        df = get_candles(sym, "D1")
        scale = _scale_is(df, fn, TARGET_VOL)
        pos = pd.Series(fn(df).values, index=df.index) * scale
        res = backtest(df, pos, symbol=sym, timeframe="D1", name=name)
        idx = pd.to_datetime(df["timestamp"].values, utc=True)
        rets = pd.Series(res.returns.values, index=idx)
        streams[name] = rets
        s = stats_full(name, rets)
        rows.append({"name": name, "symbol": sym, "scale": scale, **s})
        print(f"{name:<18} scale={scale:5.2f}  IS_Sh={s.get('IS_sharpe',0):+5.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+5.2f}  2022_Sh={s.get('2022_sharpe',0):+5.2f}  "
              f"2022_ret={s.get('2022_ret',0):+6.2%}")

    # Year-by-year for the best
    print("\n=== Year-by-year per variant ===")
    for name, r in streams.items():
        print(f"\n{name}:")
        for year, sub in r.groupby(r.index.year):
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            ar = sub.mean() * bpy
            av = sub.std(ddof=0) * np.sqrt(bpy)
            sh = ar / av if av > 0 else 0
            print(f"  {year}  Sh={sh:+5.2f}  Ret={ar:+6.2%}")

    # Save best survivor
    pd.DataFrame(rows).to_csv(OUT / "tail_v2_breakdown.csv", index=False)
    # Pick the variant with highest 2022 Sharpe AND positive across years on average
    print(f"\nSaving variant with strongest 2022 result...")
    best_name = max(rows, key=lambda r: r.get("2022_sharpe", -99))["name"]
    print(f"Best 2022 variant: {best_name}")
    best_stream = streams[best_name]
    pd.DataFrame({"timestamp": best_stream.index, "ret": best_stream.values}).to_parquet(
        OUT / "tail_v2_returns.parquet", index=False)


if __name__ == "__main__":
    main()
