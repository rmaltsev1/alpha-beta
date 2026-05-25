"""V4 — same 7 sleeves as v3, plus a regime overlay.

Hypothesis: 2022 was the only losing year (Sharpe −0.23). The reason is the
risk-off block where gold, crypto, and equity intraday-reversion all
correlated to the downside *together*. A simple SPX-vol regime filter
that halves portfolio gross when SPX realized vol is in the top quartile
should protect 2022 with minimal cost in calmer years.

Regime: compute SPX D1 30-day realized vol. Compare to the IS distribution.
If above the IS 80th percentile, sleeve weights are halved for that bar.
The cutoff is fit on IS data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles
from alphabeta.backtest import backtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies_v2 as S2

SPLIT_DATE = "2024-01-01"
TARGET_VOL = 0.05


def _bpy(df):
    s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    return len(df) / s * 365.25 if s > 0 else 252.0


def _scale_is(df_full, build_pos, target):
    split = pd.Timestamp(SPLIT_DATE, tz="UTC")
    is_df = df_full[df_full["timestamp"] < split].reset_index(drop=True)
    is_pos = pd.Series(build_pos(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df)))
    return target/av if av > 1e-9 else 0


def sleeve_returns(symbol, timeframe, build_pos):
    df = get_candles(symbol, timeframe)
    scale = _scale_is(df, build_pos, TARGET_VOL)
    full_pos = pd.Series(build_pos(df).values, index=df.index) * scale
    res = backtest(df, full_pos, symbol=symbol, timeframe=timeframe, name="")
    idx = pd.to_datetime(df["timestamp"].values, utc=True)
    return pd.Series(res.returns.values, index=idx)


def regime_series(percentile=80) -> pd.Series:
    """Boolean series indexed by H1 timestamps — True when SPX 30d rv exceeds
    the IS percentile threshold.
    """
    spx = get_candles("SPX500_USD", "D1").copy()
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["ret"].rolling(30).std()
    split = pd.Timestamp(SPLIT_DATE, tz="UTC")
    cutoff = spx.loc[spx["timestamp"] < split, "rv30"].quantile(percentile / 100)
    spx["high_vol"] = spx["rv30"] > cutoff
    # Index by date, then reindex to any bar timestamp via merge_asof.
    return spx[["timestamp", "high_vol", "rv30"]], cutoff


def main():
    sleeves = [
        ("EVE_XAU",   "XAU_USD",    "H1",  S2.evening_long),
        ("WED_BTC",   "BTCUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_ETH",   "ETHUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_SOL",   "SOLUSDT",    "D1",  S2.crypto_wed_long),
        ("D1REV_NAS", "NAS100_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_UK",  "UK100_GBP",  "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_SPX", "SPX500_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
    ]
    streams = {n: sleeve_returns(s, t, f) for n, s, t, f in sleeves}
    df = pd.concat(streams, axis=1, sort=True).fillna(0.0)

    # Build a per-bar regime mask. SPX D1 closes at 21:00 UTC; the high_vol
    # flag is "as of the most recent SPX close before this bar".
    regime_df, cutoff = regime_series(percentile=80)
    regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"], utc=True)
    print(f"IS SPX 30d-rv 80th-percentile cutoff: {cutoff:.4f}  ({cutoff*np.sqrt(252)*100:.1f}% ann)")

    # Forward-fill the regime onto df.index. Each H1 / D1 bar gets the most
    # recent SPX close's high_vol state.
    aligned_regime = pd.merge_asof(
        pd.DataFrame({"timestamp": df.index}).sort_values("timestamp"),
        regime_df.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )["high_vol"].fillna(False).astype(bool)
    mask = pd.Series(aligned_regime.values, index=df.index)

    # Regime overlay: halve gross when mask is True.
    weight = pd.Series(np.where(mask, 0.5, 1.0), index=df.index)
    sleeve_means = df.mean(axis=1)
    portfolio_v3 = sleeve_means  # unchanged baseline
    portfolio_v4 = sleeve_means * weight

    out = pd.DataFrame({
        "timestamp": df.index, "high_vol": mask.values,
        "v3_ret": portfolio_v3.values, "v4_ret": portfolio_v4.values,
        "v3_eq": (1 + portfolio_v3).cumprod().values,
        "v4_eq": (1 + portfolio_v4).cumprod().values,
    })
    out.to_parquet(Path(__file__).resolve().parent / "portfolio_v4.parquet", index=False)

    split = pd.Timestamp(SPLIT_DATE, tz="UTC")

    def stats(label, r):
        for tag, m in [("FULL", pd.Series(True, index=r.index)),
                       ("IS",   r.index < split),
                       ("OOS",  r.index >= split)]:
            s = r[m]
            if len(s) < 2: continue
            span = (s.index[-1] - s.index[0]).total_seconds() / 86400
            bpy = len(s) / span * 365.25
            ar = float(s.mean()) * bpy
            av = float(s.std(ddof=0)) * np.sqrt(bpy)
            sh = ar / av if av > 0 else 0
            eq = (1 + s).cumprod()
            dd = eq / eq.cummax() - 1
            print(f"  {label:<6} {tag:<4}  ann_ret={ar:+6.1%}  vol={av:5.1%}  Sharpe={sh:+5.2f}  MaxDD={dd.min():+6.1%}")

    stats("V3", portfolio_v3)
    print()
    stats("V4", portfolio_v4)

    # By calendar year
    print("\nSharpe by year (V3 vs V4):")
    yearly = pd.DataFrame({"v3": portfolio_v3, "v4": portfolio_v4, "regime": mask.astype(int)})
    for year, sub in yearly.groupby(yearly.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60: continue
        bpy = len(sub) / span * 365.25
        sh3 = sub["v3"].mean()*bpy / (sub["v3"].std(ddof=0)*np.sqrt(bpy)) if sub["v3"].std() > 0 else 0
        sh4 = sub["v4"].mean()*bpy / (sub["v4"].std(ddof=0)*np.sqrt(bpy)) if sub["v4"].std() > 0 else 0
        rg = sub["regime"].mean()
        print(f"  {year}  V3={sh3:+5.2f}  V4={sh4:+5.2f}  high_vol_bars={rg:.0%}")


if __name__ == "__main__":
    main()
