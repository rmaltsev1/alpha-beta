"""Crypto microstructure: hour-of-week, day-of-month patterns.

Crypto trades 24/7, so it has a fuller hour-of-week space than FX/indices.
Hunting for previously-unexplored calendar effects.
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
    return out


# ---- Diagnostic: per-hour-of-week stats ----
def per_hour_of_week(symbol):
    df = get_candles(symbol, "H1").copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["dow"] = df["timestamp"].dt.weekday
    df["hour"] = df["timestamp"].dt.hour
    df["how"] = df["dow"] * 24 + df["hour"]  # 0=Mon 00 .. 167=Sun 23
    # IS only — to find candidate signals
    is_df = df[df["timestamp"] < SPLIT]
    grp = is_df.groupby("how")["ret"].agg(["mean", "std", "count"])
    grp["t"] = grp["mean"] / grp["std"] * np.sqrt(grp["count"])
    grp["mean_bps"] = grp["mean"] * 1e4
    return grp.sort_values("t", ascending=False)


# ---- Diagnostic: per-day-of-month ----
def per_day_of_month(symbol):
    df = get_candles(symbol, "D1").copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["dom"] = df["timestamp"].dt.day
    is_df = df[df["timestamp"] < SPLIT]
    grp = is_df.groupby("dom")["ret"].agg(["mean", "std", "count"])
    grp["t"] = grp["mean"] / grp["std"] * np.sqrt(grp["count"])
    grp["mean_bps"] = grp["mean"] * 1e4
    return grp.sort_values("t", ascending=False)


def _scale_is(df_full, build_position, target):
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df["timestamp"].values)))
    return target/av if av > 1e-9 else 0


def hour_of_week_signal(df, hows_with_side):
    """hows_with_side: {how (0-167): side}"""
    df = df.copy()
    df["dow"] = df["timestamp"].dt.weekday
    df["hour"] = df["timestamp"].dt.hour
    df["how"] = df["dow"] * 24 + df["hour"]
    pos = pd.Series(0.0, index=df.index)
    for how, side in hows_with_side.items():
        pos[df["how"] == how] = float(side)
    return pos


def main():
    print("=" * 70)
    print("Step 1: scan top 20 hour-of-week buckets per crypto symbol (IS only)")
    print("=" * 70)

    candidates = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        grp = per_hour_of_week(sym)
        top = grp.head(10)
        bot = grp.tail(10)
        print(f"\n=== {sym} top 10 (long candidates) ===")
        print(top[["mean_bps", "count", "t"]].to_string(float_format=lambda x: f"{x:>7.2f}"))
        print(f"\n=== {sym} bottom 10 (short candidates) ===")
        print(bot[["mean_bps", "count", "t"]].to_string(float_format=lambda x: f"{x:>7.2f}"))
        # Keep top-3 long and bottom-3 short candidates if |t| > 2.0
        candidates[sym] = {}
        for how, row in top.iterrows():
            if row["t"] > 2.0:
                candidates[sym][how] = +1
        for how, row in bot.iterrows():
            if row["t"] < -2.0:
                candidates[sym][how] = -1
        print(f"  → {sym} candidates: {candidates[sym]}")

    print("\n" + "=" * 70)
    print("Step 2: backtest each symbol's hour-of-week sleeve (IS-discovered, OOS-tested)")
    print("=" * 70)
    sleeves = {}
    rows = []
    for sym, sides in candidates.items():
        if not sides:
            continue
        df = get_candles(sym, "H1")
        builder = lambda df: hour_of_week_signal(df, sides)
        scale = _scale_is(df, builder, TARGET_VOL)
        pos = pd.Series(builder(df).values, index=df.index) * scale
        res = backtest(df, pos, symbol=sym, timeframe="H1", name=f"HOW_{sym}")
        idx = pd.to_datetime(df["timestamp"].values, utc=True)
        rets = pd.Series(res.returns.values, index=idx)
        # Collapse to D1
        daily = rets.groupby(rets.index.floor("D")).sum()
        daily.index = pd.to_datetime(daily.index, utc=True)
        s = stats(sym, daily)
        sleeves[sym] = daily
        rows.append({"symbol": sym, "scale": scale, "n_buckets": len(sides), **s})
        print(f"{sym:<10}  scale={scale:5.2f}  IS_Sh={s.get('IS_sharpe',0):+5.2f}  OOS_Sh={s.get('OOS_sharpe',0):+5.2f}")

    # ---- Day-of-month ----
    print("\n" + "=" * 70)
    print("Step 3: day-of-month scan (D1, IS only)")
    print("=" * 70)
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        grp = per_day_of_month(sym)
        print(f"\n=== {sym} top 5 / bottom 5 days-of-month ===")
        print(grp.head(5)[["mean_bps", "count", "t"]].to_string(float_format=lambda x: f"{x:>7.2f}"))
        print("  ...")
        print(grp.tail(5)[["mean_bps", "count", "t"]].to_string(float_format=lambda x: f"{x:>7.2f}"))

    # ---- Combine surviving sleeves ----
    if sleeves:
        survivors = [s for s in rows
                     if s.get("IS_sharpe", 0) > 0.5 and s.get("OOS_sharpe", 0) > 0]
        print(f"\nSurvivors (IS≥0.5 AND OOS>0): {[s['symbol'] for s in survivors]}")
        if survivors:
            keep = [s["symbol"] for s in survivors]
            combined = pd.concat({s: sleeves[s] for s in keep}, axis=1, sort=True).fillna(0.0).mean(axis=1)
            s_combined = stats("CRYPTO_HOW", combined)
            print(f"\nCombined sleeve:")
            for tag in ["FULL", "IS", "OOS"]:
                print(f"  {tag:<4} Sharpe={s_combined.get(f'{tag}_sharpe',0):+.2f}  "
                      f"Return={s_combined.get(f'{tag}_ret',0):+.2%}")
            for year, sub in combined.groupby(combined.index.year):
                if len(sub) < 50: continue
                bpy = _bpy(sub.index)
                sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
                print(f"  {year}  Sharpe={sh:+.2f}")
            pd.DataFrame({"timestamp": combined.index, "ret": combined.values}).to_parquet(
                OUT / "crypto_micro_returns.parquet", index=False)
        else:
            print("No survivors — saving zero series.")
            zero = pd.Series(0.0, index=sleeves[list(sleeves.keys())[0]].index)
            pd.DataFrame({"timestamp": zero.index, "ret": zero.values}).to_parquet(
                OUT / "crypto_micro_returns.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUT / "crypto_micro_breakdown.csv", index=False)


if __name__ == "__main__":
    main()
