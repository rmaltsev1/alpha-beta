"""Volatility-forecasting + dynamic-sizing sleeve (wave3).

Builds 4 sub-sleeves per symbol that use a walk-forward EWMA vol forecast
to scale exposure. The premise (from agent-1 analysis): |returns| are
highly persistent (Hurst > 0.7 on all 13 symbols; AR(1) ~0.985 on 30d
realized vol), so a vol forecast is a *predictable* state variable that
should pay off when used to dynamically size positions.

Sub-sleeves:
  1. EWMA vol-targeted long  (Moreira-Muir, predictive)
  2. Vol-regime conditional TSMOM (1.5x in low vol, 0.5x in high vol)
  3. Vol-of-vol regime filter (cash when vol-of-vol elevated)
  4. Predictive vol-skew (long contraction, exit expansion)

Methodology:
  - D1, 13 symbols, 2020-01-01..2026-05-23
  - IS < 2024-01-01, OOS >= 2024-01-01
  - EWMA lambda=0.94 on |daily log returns|
  - Walk-forward percentiles (rolling)
  - Vol-scale to 5% IS ann vol
  - Keep sub-sleeves with IS Sharpe >= 0.4 AND OOS Sharpe > 0
  - Combine survivors equal-weight per asset class, then EW across asset classes
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import backtest, cost_for

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
EWMA_LAMBDA = 0.94
TRADING_DAYS = 252


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ewma_abs(absret: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Walk-forward EWMA of |returns|. EWMA variance recursion:
        var_t = lam * var_{t-1} + (1-lam) * ret_{t-1}^2
    We forecast vol at t using info up to t-1 (no look-ahead).
    """
    r2 = (absret ** 2).fillna(0.0).values
    var = np.zeros_like(r2)
    # seed with first non-null value's r^2
    var[0] = r2[0] if r2[0] > 0 else 1e-8
    for i in range(1, len(r2)):
        var[i] = lam * var[i - 1] + (1 - lam) * r2[i - 1]
    vol = np.sqrt(var)
    return pd.Series(vol, index=absret.index)


def realized_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized vol over `window` days, on log returns."""
    return ret.rolling(window).std(ddof=0)


def ann_stats(r: pd.Series, freq: int = TRADING_DAYS) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
    av = float(r.std(ddof=0) * np.sqrt(freq))
    ar = float(r.mean() * freq)
    sh = ar / av if av > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": sh, "ann_ret": ar, "ann_vol": av, "max_dd": dd}


def split_stats(r: pd.Series) -> dict:
    out = {}
    full = ann_stats(r)
    out["full_sharpe"] = full["sharpe"]
    out["full_dd"] = full["max_dd"]
    out["full_vol"] = full["ann_vol"]
    is_r = r[r.index < SPLIT]
    oos_r = r[r.index >= SPLIT]
    out["is_sharpe"] = ann_stats(is_r)["sharpe"]
    out["is_vol"] = ann_stats(is_r)["ann_vol"]
    out["oos_sharpe"] = ann_stats(oos_r)["sharpe"]
    out["oos_vol"] = ann_stats(oos_r)["ann_vol"]
    y22 = r[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
            (r.index < pd.Timestamp("2023-01-01", tz="UTC"))]
    out["y2022_sharpe"] = ann_stats(y22)["sharpe"]
    return out


# -----------------------------------------------------------------------------
# Sub-sleeve signal builders.
# Each returns a position Series aligned to df.index (NOT yet vol-scaled to 5%).
# All inputs are time-aligned and computed walk-forward.
# -----------------------------------------------------------------------------
def sleeve_ewma_voltarget(df: pd.DataFrame) -> pd.Series:
    """1. EWMA-vol-forecasted long: pos = target_vol / forecast_vol, capped [0,3].
    target_vol set internally to the EWMA-vol IS median so the *average* position
    is ~1 in-sample. Later we vol-scale the whole sleeve to 5% ann.
    """
    ret = np.log(df["close"] / df["close"].shift(1))
    abs_ret = ret.abs()
    vol_fcst = ewma_abs(abs_ret)  # daily vol forecast for t (uses info up to t-1)
    # target = IS median of forecast (computed walk-forward via expanding median up to t-1)
    # Use expanding median for "what would have been the target then?"
    target = vol_fcst.shift(1).expanding(min_periods=60).median()
    pos = (target / vol_fcst).clip(lower=0.0, upper=3.0)
    pos = pos.shift(1).fillna(0.0)  # avoid look-ahead: act on t with vol-forecast made at t
    return pos


def sleeve_volregime_tsmom(df: pd.DataFrame, mom_lookback: int = 252) -> pd.Series:
    """2. Vol-regime conditional TSMOM.
    - TSMOM signal: sign of trailing `mom_lookback` log-return.
    - Vol regime: rolling 252d percentile of EWMA-vol forecast.
       - Bottom tercile (pct<0.33) -> 1.5x
       - Top tercile  (pct>=0.67)  -> 0.5x
       - Middle                     -> 1.0x
    """
    ret = np.log(df["close"] / df["close"].shift(1))
    mom = np.sign(np.log(df["close"] / df["close"].shift(mom_lookback)))
    vol_fcst = ewma_abs(ret.abs())
    # rolling 252d percentile
    vol_pct = vol_fcst.rolling(252, min_periods=60).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )
    w = pd.Series(1.0, index=df.index)
    w[vol_pct < 0.33] = 1.5
    w[vol_pct >= 0.67] = 0.5
    pos = (mom * w).shift(1).fillna(0.0)
    return pos


def sleeve_volofvol_filter(df: pd.DataFrame, mom_lookback: int = 60) -> pd.Series:
    """3. Vol-of-vol filter: rolling std of log(EWMA vol).
    When vol-of-vol > 252d rolling 70th pct, go to cash. Else follow short-mom (60d) sign.
    """
    ret = np.log(df["close"] / df["close"].shift(1))
    vol_fcst = ewma_abs(ret.abs())
    log_vol = np.log(vol_fcst.replace(0, np.nan)).ffill()
    vov = log_vol.diff().rolling(20).std(ddof=0)
    # walk-forward 70th percentile
    vov_pct = vov.rolling(252, min_periods=60).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )
    mom = np.sign(np.log(df["close"] / df["close"].shift(mom_lookback)))
    pos = mom.copy()
    pos[vov_pct >= 0.7] = 0.0
    return pos.shift(1).fillna(0.0)


def sleeve_predictive_skew(df: pd.DataFrame, rv_window: int = 20) -> pd.Series:
    """4. Predictive vol-skew:
    - forecast vol / current realized vol ratio
    - ratio > 1.5  -> expecting expansion -> exit longs (flat)
    - ratio < 0.7  -> expecting contraction -> long
    - else carry prev state (sticky)
    """
    ret = np.log(df["close"] / df["close"].shift(1))
    vol_fcst = ewma_abs(ret.abs())
    rv = realized_vol(ret, rv_window)
    ratio = vol_fcst / rv.replace(0, np.nan)
    pos = pd.Series(np.nan, index=df.index)
    pos[ratio < 0.7] = 1.0
    pos[ratio > 1.5] = 0.0
    pos = pos.ffill().fillna(0.0)
    return pos.shift(1).fillna(0.0)


SLEEVE_BUILDERS = {
    "ewma_voltarget": sleeve_ewma_voltarget,
    "volregime_tsmom": sleeve_volregime_tsmom,
    "volofvol_filter": sleeve_volofvol_filter,
    "predictive_skew": sleeve_predictive_skew,
}


# -----------------------------------------------------------------------------
# Per-symbol runner
# -----------------------------------------------------------------------------
def run_symbol(symbol: str) -> dict[str, pd.Series]:
    """Return dict of {sleeve_name: daily-return-series (UTC tz-aware index)}."""
    df = get_candles(symbol, "D1", start="2020-01-01", end="2026-05-24")
    df = df.reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    out: dict[str, pd.Series] = {}
    for name, builder in SLEEVE_BUILDERS.items():
        try:
            pos = builder(df)
        except Exception as e:
            print(f"  ! {symbol}/{name} build failed: {e}")
            continue
        pos = pos.reindex(df.index).fillna(0.0)

        # Vol-scale to 5% ann using IS subset
        is_mask = ts < SPLIT
        ret_raw = df["close"].pct_change().fillna(0.0)
        raw_strat = pos * ret_raw
        is_strat = raw_strat[is_mask]
        if len(is_strat) < 60:
            continue
        is_vol = float(is_strat.std(ddof=0) * np.sqrt(TRADING_DAYS))
        if is_vol < 1e-6:
            continue
        scale = TARGET_VOL / is_vol
        # cap scale to a sane range so we don't blow up sizing on a flat IS sleeve
        scale = float(np.clip(scale, 0.0, 10.0))
        pos_scaled = pos * scale

        res = backtest(df, pos_scaled, symbol=symbol, timeframe="D1",
                       name=f"{symbol}_{name}")
        rets = pd.Series(res.returns.values, index=ts)
        out[name] = rets
    return out


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main():
    print(f"Wave3: volatility-forecasting sleeve")
    print(f"  Symbols: {len(CRYPTO + FOREX + INDEX)}")
    print(f"  Split:   {SPLIT.date()}")
    print(f"  Target:  {TARGET_VOL:.1%} ann vol per sub-sleeve (IS-fit)")
    print()

    all_symbols = CRYPTO + FOREX + INDEX
    per_sleeve_returns: dict[str, pd.Series] = {}  # key: "{sym}__{name}"
    rows = []

    for sym in all_symbols:
        cls = SYMBOL_TYPE[sym].value
        try:
            sleeves = run_symbol(sym)
        except Exception as e:
            print(f"!! {sym} failed: {e}")
            continue
        for name, ret in sleeves.items():
            key = f"{sym}__{name}"
            per_sleeve_returns[key] = ret
            s = split_stats(ret)
            row = {"symbol": sym, "class": cls, "sleeve": name, **s}
            rows.append(row)
            print(f"  {sym:<10} {name:<18}  IS={s['is_sharpe']:+.2f}  "
                  f"OOS={s['oos_sharpe']:+.2f}  "
                  f"DD={s['full_dd']:+.1%}  Y22={s['y2022_sharpe']:+.2f}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "volforecast_breakdown.csv", index=False)
    print(f"\nWrote {OUT / 'volforecast_breakdown.csv'}")

    # Filter survivors: IS >= 0.4 AND OOS > 0
    keep_mask = (breakdown["is_sharpe"] >= 0.4) & (breakdown["oos_sharpe"] > 0)
    survivors = breakdown[keep_mask].copy()
    print(f"\nSurvivors: {len(survivors)} / {len(breakdown)}")
    if len(survivors):
        print(survivors.groupby(["class", "sleeve"]).size().to_string())

    # Combine survivors equal-weight inside asset-class, then equal-weight across classes.
    # Stage 1: per-class EW mean of survivor returns
    class_returns: dict[str, pd.Series] = {}
    for cls in ["crypto", "forex", "index"]:
        cls_survivors = survivors[survivors["class"] == cls]
        if len(cls_survivors) == 0:
            continue
        keys = [f"{r.symbol}__{r.sleeve}" for r in cls_survivors.itertuples()]
        streams = [per_sleeve_returns[k] for k in keys]
        aligned = pd.concat(streams, axis=1).fillna(0.0)
        cls_ret = aligned.mean(axis=1)
        class_returns[cls] = cls_ret
        s = split_stats(cls_ret)
        print(f"  class {cls:<6}: n={len(cls_survivors):>2}  "
              f"IS={s['is_sharpe']:+.2f}  OOS={s['oos_sharpe']:+.2f}  "
              f"DD={s['full_dd']:+.1%}")

    # Stage 2: equal-weight across asset classes
    if class_returns:
        aligned = pd.concat(class_returns.values(), axis=1).fillna(0.0)
        sleeve_ret = aligned.mean(axis=1)
    else:
        sleeve_ret = pd.Series(dtype=float, name="ret")

    sleeve_ret.name = "ret"
    sleeve_ret.index.name = "timestamp"
    df_out = sleeve_ret.reset_index()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], utc=True)
    df_out.to_parquet(OUT / "volforecast_returns.parquet", index=False)
    print(f"\nWrote {OUT / 'volforecast_returns.parquet'}  ({len(df_out)} rows)")

    # Headline
    print("\nHEADLINE")
    print("-" * 60)
    s = split_stats(sleeve_ret)
    print(f"  FULL Sharpe: {s['full_sharpe']:+.2f}  Vol={s['full_vol']:.1%}  DD={s['full_dd']:+.1%}")
    print(f"  IS   Sharpe: {s['is_sharpe']:+.2f}  Vol={s['is_vol']:.1%}")
    print(f"  OOS  Sharpe: {s['oos_sharpe']:+.2f}  Vol={s['oos_vol']:.1%}")
    print(f"  2022 Sharpe: {s['y2022_sharpe']:+.2f}")

    # Per-sleeve aggregate (across all symbols, survivors only)
    if len(survivors):
        print("\nPER-SUB-SLEEVE (survivors aggregated EW across symbols):")
        print(f"{'sleeve':<18} {'n':>3} {'IS':>6} {'OOS':>6} {'DD':>7} {'Y22':>6}")
        for slv in ["ewma_voltarget", "volregime_tsmom", "volofvol_filter", "predictive_skew"]:
            sub = survivors[survivors["sleeve"] == slv]
            if len(sub) == 0:
                print(f"{slv:<18} {0:>3} {'-':>6} {'-':>6} {'-':>7} {'-':>6}")
                continue
            keys = [f"{r.symbol}__{r.sleeve}" for r in sub.itertuples()]
            aligned = pd.concat([per_sleeve_returns[k] for k in keys], axis=1).fillna(0.0)
            slv_ret = aligned.mean(axis=1)
            ss = split_stats(slv_ret)
            print(f"{slv:<18} {len(sub):>3} {ss['is_sharpe']:>+6.2f} {ss['oos_sharpe']:>+6.2f} "
                  f"{ss['full_dd']:>+7.1%} {ss['y2022_sharpe']:>+6.2f}")


if __name__ == "__main__":
    main()
