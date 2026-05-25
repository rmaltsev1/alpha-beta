"""Synthesize ALL sleeves into a master portfolio.

Sources:
  * scratch/quant/{tsmom,xsmom,volmgmt_subA,risk_parity}_returns.parquet
    (Pairs dropped — broken in OOS.
     volmgmt combined dropped in favor of Sub-A — Sub-B was OOS-flat.)
  * v3 calendar sleeves rebuilt inline from scratch/models/strategies_v2.py
    via scratch/models/run_v3.py logic (EVE_XAU, WED_BTC/ETH/SOL, D1REV_NAS/UK/SPX)

For each sleeve we:
  1. Rescale to a common 5% IS-annualized vol (so weights compare apples-to-apples).
  2. Resample to the common D1 calendar (UTC date floor).
  3. Compute correlation matrix and stats.
  4. Build several portfolio variants:
       - Equal weight
       - Inverse-vol (already vol-normalized → ~equal)
       - Sharpe-weighted (cap 30% per sleeve)
       - Cluster-aware (1/n by *bucket*: trend / mean-rev / beta / calendar)
  5. Report IS / OOS / year-by-year for each variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "models"))

from alphabeta import get_candles
from alphabeta.backtest import backtest, split_is_oos
import strategies_v2 as S2  # type: ignore

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05  # per-sleeve IS vol target after our rescaling


def _bpy(idx_or_df) -> float:
    if isinstance(idx_or_df, pd.DataFrame):
        ts = idx_or_df["timestamp"]
    else:
        ts = pd.Series(idx_or_df)
    span = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
    return max(len(ts) / span * 365.25, 1.0) if span > 0 else 252.0


def _scale_is(df_full: pd.DataFrame, build_position, target: float) -> float:
    is_df = df_full[df_full["timestamp"] < SPLIT].reset_index(drop=True)
    is_pos = pd.Series(build_position(is_df).values, index=is_df.index)
    ret = np.log(is_df["close"] / is_df["close"].shift(1)).fillna(0)
    raw = is_pos.values * ret.values
    av = float(np.std(raw, ddof=0) * np.sqrt(_bpy(is_df)))
    return target / av if av > 1e-9 else 0.0


def v3_sleeve_returns(symbol: str, timeframe: str, build_position) -> pd.Series:
    """Reproduce v3 sleeve return stream and return as a per-bar Series
    indexed by tz-aware UTC timestamps."""
    df = get_candles(symbol, timeframe)
    scale = _scale_is(df, build_position, TARGET_VOL)
    full_pos = pd.Series(build_position(df).values, index=df.index) * scale
    res = backtest(df, full_pos, symbol=symbol, timeframe=timeframe, name="")
    idx = pd.to_datetime(df["timestamp"].values, utc=True)
    return pd.Series(res.returns.values, index=idx)


def collapse_to_daily(s: pd.Series) -> pd.Series:
    """Sum per-bar returns into one row per UTC calendar day. Returns are small
    (≤1%) so additive aggregation is a fine approximation to compounding."""
    daily = s.groupby(s.index.floor("D")).sum()
    daily.index = pd.to_datetime(daily.index, utc=True)
    return daily


def rescale_to_target_vol(s: pd.Series, target_vol: float) -> tuple[pd.Series, float]:
    """Rescale a daily return stream so its IS realized vol is target_vol."""
    is_part = s[s.index < SPLIT]
    if len(is_part) < 30:
        return s, 0.0
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    if av <= 1e-9:
        return s * 0, 0.0
    k = target_vol / av
    return s * k, k


def stats_for(label: str, r: pd.Series) -> dict:
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2:
            continue
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        bpy = len(sub) / span * 365.25 if span > 0 else 252.0
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        sh = ar / av if av > 0 else 0.0
        eq = (1 + sub).cumprod()
        dd = eq / eq.cummax() - 1
        out[f"{tag}_sharpe"] = sh
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        out[f"{tag}_dd"] = float(dd.min())
    return out


def yearly_sharpes(r: pd.Series) -> pd.Series:
    rows = {}
    for year, sub in r.groupby(r.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60:
            continue
        bpy = len(sub) / span * 365.25
        sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
        rows[year] = sh
    return pd.Series(rows)


def main() -> None:
    # ---- (A) v3 calendar sleeves: reproduce from scratch ----
    v3_specs = [
        ("EVE_XAU",   "XAU_USD",    "H1",  S2.evening_long),
        ("WED_BTC",   "BTCUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_ETH",   "ETHUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_SOL",   "SOLUSDT",    "D1",  S2.crypto_wed_long),
        ("D1REV_NAS", "NAS100_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_UK",  "UK100_GBP",  "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_SPX", "SPX500_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
    ]
    v3_streams_raw = {n: v3_sleeve_returns(s, t, f) for n, s, t, f in v3_specs}
    v3_streams = {n: collapse_to_daily(s) for n, s in v3_streams_raw.items()}

    # ---- (B) quant sleeves loaded from parquet ----
    quant_files = {
        "TSMOM":   OUT / "tsmom_returns.parquet",
        "XSMOM":   OUT / "xsmom_returns.parquet",
        "VOLMGD":  OUT / "volmgmt_subA_returns.parquet",
        "RISKPAR": OUT / "risk_parity_returns.parquet",
        # PAIRS dropped — agent reported negative OOS Sharpe across all periods.
    }
    quant_streams = {}
    for name, p in quant_files.items():
        df = pd.read_parquet(p)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        quant_streams[name] = pd.Series(df["ret"].values, index=ts).groupby(
            lambda t: t.floor("D")).sum()
        quant_streams[name].index = pd.to_datetime(quant_streams[name].index, utc=True)

    all_streams_raw = {**v3_streams, **quant_streams}

    # ---- Rescale every sleeve to 5% IS vol so weights are commensurate ----
    rescaled = {}
    scales = {}
    for name, s in all_streams_raw.items():
        rs, k = rescale_to_target_vol(s, TARGET_VOL)
        rescaled[name] = rs
        scales[name] = k

    df = pd.concat(rescaled, axis=1, sort=True).fillna(0.0)
    df.to_parquet(OUT / "all_sleeve_returns.parquet")

    # Per-sleeve stats and yearly Sharpes
    print("=" * 95)
    print(f"{'Sleeve':<14} {'IS_Sh':>6} {'OOS_Sh':>6} {'IS_vol':>7} {'OOS_vol':>7} {'IS_DD':>7} {'OOS_DD':>7} {'scale':>7}")
    print("-" * 95)
    rows = []
    for name in df.columns:
        s = stats_for(name, df[name])
        rows.append(s)
        print(f"{name:<14} {s['IS_sharpe']:>+6.2f} {s['OOS_sharpe']:>+6.2f} "
              f"{s['IS_vol']:>7.1%} {s['OOS_vol']:>7.1%} "
              f"{s['IS_dd']:>+7.1%} {s['OOS_dd']:>+7.1%} {scales[name]:>7.2f}")
    sleeves_df = pd.DataFrame(rows)
    sleeves_df.to_csv(OUT / "master_sleeves_stats.csv", index=False)

    # ---- Correlation matrix ----
    print("\n=== Sleeve weekly-return correlations (FULL) ===")
    weekly = df.resample("1W").sum()
    corr = weekly.corr().round(2)
    print(corr.to_string())
    corr.to_csv(OUT / "master_sleeve_correlations.csv")

    # ---- Portfolio variants ----
    # 1) Equal weight across all 11 sleeves
    ew = df.mean(axis=1)

    # 2) Sharpe-tilted, cap 30%, floor 0%
    sharpes = pd.Series({r["label"]: max(r.get("IS_sharpe", 0), 0) for r in rows})
    sw = (sharpes / sharpes.sum()).clip(upper=0.30)
    sw = sw / sw.sum()
    sharpe_w = (df * sw.reindex(df.columns).values).sum(axis=1)

    # 3) Cluster-balanced: 1/4 to each bucket, equal within bucket
    buckets = {
        "calendar":   ["EVE_XAU", "WED_BTC", "WED_ETH", "WED_SOL"],
        "mean_rev":   ["D1REV_NAS", "D1REV_UK", "D1REV_SPX"],
        "trend":      ["TSMOM", "XSMOM"],
        "beta":       ["VOLMGD", "RISKPAR"],
    }
    cluster_w = pd.Series(0.0, index=df.columns)
    for bucket, members in buckets.items():
        share = 0.25 / len(members)
        for m in members:
            if m in cluster_w.index:
                cluster_w[m] = share
    cluster_w = cluster_w / cluster_w.sum()
    cluster_p = (df * cluster_w.values).sum(axis=1)

    # 4) Cluster + OOS sleeve gating: drop sleeves with IS Sharpe < 0.3
    floor = 0.30
    keep = sharpes >= floor
    gated_w = cluster_w.copy()
    for n in df.columns:
        if not keep.get(n, True):
            gated_w[n] = 0.0
    gated_w = gated_w / gated_w.sum()
    gated_p = (df * gated_w.values).sum(axis=1)

    variants = {
        "EQUAL_WT":        (ew, pd.Series(1/len(df.columns), index=df.columns)),
        "SHARPE_TILT":     (sharpe_w, sw),
        "CLUSTER_BAL":     (cluster_p, cluster_w),
        "CLUSTER_GATED":   (gated_p, gated_w),
    }

    print(f"\n=== Portfolio variants (target sleeve vol = {TARGET_VOL:.0%} IS) ===")
    print(f"{'Variant':<16} {'FULL_Sh':>7} {'IS_Sh':>6} {'OOS_Sh':>6} {'OOS_vol':>7} {'OOS_DD':>7}")
    print("-" * 60)
    out_rows = []
    for vname, (ret, w) in variants.items():
        s = stats_for(vname, ret)
        out_rows.append({**s, "weights": w.to_dict()})
        print(f"{vname:<16} {s['FULL_sharpe']:>+7.2f} {s['IS_sharpe']:>+6.2f} "
              f"{s['OOS_sharpe']:>+6.2f} {s['OOS_vol']:>7.1%} {s['OOS_dd']:>+7.1%}")

    # Save the gated portfolio (the best risk-adjusted) explicitly
    gated_out = pd.DataFrame({
        "timestamp": gated_p.index, "ret": gated_p.values,
        "equity": (1 + gated_p).cumprod().values,
    })
    gated_out.to_parquet(OUT / "master_portfolio_returns.parquet", index=False)

    # ---- Year-by-year for each variant ----
    print(f"\n=== Year-by-year Sharpe (calendar-balanced gated variant is the headline) ===")
    yr_df = pd.DataFrame({vname: yearly_sharpes(ret) for vname, (ret, _) in variants.items()})
    print(yr_df.round(2).to_string())
    yr_df.to_csv(OUT / "master_yearly_sharpes.csv")

    # ---- Final weights summary ----
    print(f"\n=== Final CLUSTER_GATED weights ===")
    for name, w in gated_w.sort_values(ascending=False).items():
        bucket = next((b for b, ms in buckets.items() if name in ms), "?")
        is_sh = sharpes.get(name, 0)
        print(f"  {name:<14} bucket={bucket:<10} weight={w:6.1%}  IS_Sharpe={is_sh:+.2f}")


if __name__ == "__main__":
    main()
