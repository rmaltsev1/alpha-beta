"""VWAP-based mean-reversion, trend, and breakout strategies (wave 3).

Five strategy families:

  1. Daily VWAP mean-reversion (H1, per symbol, all 13).
     - Compute intraday cumulative VWAP within UTC day.
     - close > VWAP * 1.005  -> short 3 H1 bars
     - close < VWAP * 0.995  -> long  3 H1 bars

  2. VWAP-trend follow (H1, per symbol).
     - Slope of daily VWAP over last 6 H1 bars > threshold AND price > VWAP
       -> long for rest of day.

  3. Weekly VWAP on D1 (per symbol).
     - 7-day rolling volume-weighted average + 2-sigma bands.
     - Long if close < VWAP - 2 sigma; short if > VWAP + 2 sigma. Hold 3 days.

  4. VWAP breakout, crypto only (H1).
     - Weekly VWAP. After >=5 bars below, break above -> long 8 bars.
       Mirror for short.

  5. High-volume reversal (H1, all 13, but flagged on real-volume vs tick).
     - Volume > 95th percentile (rolling 500-bar) AND bar closes against
       prior bar direction -> hold reversal direction 2 bars.

Methodology
-----------
- IS  <  2024-01-01
- OOS >= 2024-01-01
- Percentiles and rolling stats are walk-forward (no future leak).
- Each sub-sleeve vol-scaled to 5% IS annualized vol (D1).
- Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
- Combine equal-weight, daily UTC.

Outputs
-------
  scratch/wave3/vwap_returns.parquet   -- daily UTC combined sleeve
  scratch/wave3/vwap_breakdown.csv     -- per-strategy stats
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import backtest


# -- knobs --------------------------------------------------------------------
SPLIT = "2024-01-01"
TF_H1 = "H1"
TF_D1 = "D1"
BPY_DAILY = 365.25
IS_VOL_TARGET = 0.05
IS_SHARPE_MIN = 0.5
OOS_SHARPE_MIN = 0.0
OUT_DIR = Path(__file__).resolve().parent


# -- helpers ------------------------------------------------------------------
def _stats(returns: pd.Series, freq: float = BPY_DAILY) -> dict:
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0, "n": int(len(r))}
    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_dd": float(dd),
        "n": int(len(r)),
    }


def _to_utc_index(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    out = pd.Series(s.values, index=pd.DatetimeIndex(df["timestamp"]))
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _vol_scale_is(daily_ret: pd.Series, split_ts: pd.Timestamp,
                  target: float = IS_VOL_TARGET) -> float:
    is_r = daily_ret.loc[daily_ret.index < split_ts].dropna()
    if is_r.empty:
        return 0.0
    sd = is_r.std(ddof=0) * np.sqrt(BPY_DAILY)
    if sd <= 0:
        return 0.0
    return float(target / sd)


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return ((df["high"] + df["low"] + df["close"]) / 3.0).astype("float64")


# -- VWAP computations --------------------------------------------------------
def daily_vwap_h1(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP, reset at UTC day boundary.

    Uses typical price * volume / cum volume. No look-ahead: the value at
    bar t aggregates the bar itself; strategies must shift when using it
    as a *signal* (and we do, in the position functions).
    """
    tp = _typical_price(df)
    vol = df["volume"].astype("float64").clip(lower=0.0)
    pv = tp * vol
    day = pd.DatetimeIndex(df["timestamp"]).tz_convert("UTC").normalize()
    grp = pd.Series(day, index=df.index)
    cum_pv = pv.groupby(grp).cumsum()
    cum_v = vol.groupby(grp).cumsum()
    # Where cum_v is 0 (e.g. pre-market with zero ticks), fall back to TP.
    vwap = np.where(cum_v > 0, cum_pv / cum_v.replace(0, np.nan), tp)
    return pd.Series(vwap, index=df.index, dtype="float64")


def rolling_vwap(df: pd.DataFrame, lookback_bars: int) -> pd.Series:
    """Rolling volume-weighted average price over `lookback_bars`."""
    tp = _typical_price(df)
    vol = df["volume"].astype("float64").clip(lower=0.0)
    pv = (tp * vol).rolling(lookback_bars, min_periods=max(1, lookback_bars // 2)).sum()
    cv = vol.rolling(lookback_bars, min_periods=max(1, lookback_bars // 2)).sum()
    out = pv / cv.replace(0, np.nan)
    return out.ffill()


# -- strategy 1: Daily VWAP mean-reversion (H1) -------------------------------
def daily_vwap_reversion_position(df: pd.DataFrame,
                                  upper: float = 1.005,
                                  lower: float = 0.995,
                                  hold_bars: int = 3) -> pd.Series:
    """When close > VWAP * upper -> short; < VWAP * lower -> long. Hold N bars.

    Signal is generated from close[t-1] vs VWAP[t-1] (i.e. fully observable
    by start of bar t). The position is then held for `hold_bars`.
    """
    vwap = daily_vwap_h1(df)
    close = df["close"].astype("float64")

    # shift to make decision observable at start of next bar
    ratio = (close / vwap).shift(1)
    raw = pd.Series(0.0, index=df.index, dtype="float64")
    raw[ratio > upper] = -1.0
    raw[ratio < lower] = +1.0

    # Hold for hold_bars: once a new non-zero signal fires, override.
    # Otherwise carry forward up to hold_bars then reset.
    pos = np.zeros(len(df), dtype="float64")
    counter = 0
    last_sig = 0.0
    raw_arr = raw.to_numpy()
    for i in range(len(df)):
        sig = raw_arr[i]
        if sig != 0.0:
            last_sig = sig
            counter = hold_bars
        if counter > 0:
            pos[i] = last_sig
            counter -= 1
        else:
            pos[i] = 0.0
            last_sig = 0.0
    return pd.Series(pos, index=df.index)


# -- strategy 2: VWAP-trend follow (H1) ---------------------------------------
def vwap_trend_position(df: pd.DataFrame,
                        slope_bars: int = 6,
                        slope_bps_per_hr: float = 5.0) -> pd.Series:
    """Long when daily-VWAP slope > threshold AND close > VWAP; mirror short.

    Slope: regress last `slope_bars` VWAP values vs time index; threshold
    in bps/hr. Hold until end-of-day (UTC).
    """
    vwap = daily_vwap_h1(df).astype("float64")
    close = df["close"].astype("float64")
    n = len(df)

    # Rolling OLS slope of VWAP vs bar index (1 unit per H1 bar = 1 hr).
    # Use the closed-form: slope = cov(x,y)/var(x). With x = [0..L-1], var
    # is constant; we compute via rolling mean / sum.
    L = slope_bars
    x = np.arange(L, dtype="float64")
    x_mean = x.mean()
    x_dev = x - x_mean
    var_x = (x_dev ** 2).sum()

    vw_arr = vwap.to_numpy()
    slope = np.full(n, np.nan, dtype="float64")
    for i in range(L - 1, n):
        y = vw_arr[i - L + 1 : i + 1]
        y_dev = y - y.mean()
        slope[i] = float((x_dev * y_dev).sum() / var_x)
    # Normalize slope as bps/hr of the VWAP level itself.
    slope_bps = (slope / vw_arr) * 1e4  # change-per-bar / level * 10000

    # Decision uses bar t-1 info; therefore shift everything by 1.
    slope_prev = pd.Series(slope_bps, index=df.index).shift(1)
    price_above = (close.shift(1) > vwap.shift(1))
    price_below = (close.shift(1) < vwap.shift(1))

    long_sig = (slope_prev > slope_bps_per_hr) & price_above
    short_sig = (slope_prev < -slope_bps_per_hr) & price_below

    # Build position: when long_sig fires, hold long until end of UTC day.
    # Same for short.
    day = pd.DatetimeIndex(df["timestamp"]).tz_convert("UTC").normalize()
    pos = np.zeros(n, dtype="float64")
    cur = 0.0
    cur_day = pd.NaT
    for i in range(n):
        d = day[i]
        if d != cur_day:
            cur = 0.0
            cur_day = d
        if long_sig.iat[i]:
            cur = 1.0
        elif short_sig.iat[i]:
            cur = -1.0
        pos[i] = cur
    return pd.Series(pos, index=df.index)


# -- strategy 3: Weekly VWAP on D1 with sigma bands ---------------------------
def weekly_vwap_d1_position(df: pd.DataFrame,
                            lookback: int = 7,
                            sigma_n: float = 2.0,
                            hold_bars: int = 3) -> pd.Series:
    """Long when close drops > N sigma below 7d rolling VWAP, short if above."""
    close = df["close"].astype("float64")
    vwap = rolling_vwap(df, lookback)
    # sigma: rolling stdev of (close - vwap), walk-forward only with shift.
    spread = close - vwap
    # Use 30d rolling stdev for the band scale.
    sigma = spread.rolling(30, min_periods=15).std(ddof=0)
    # all decision signals must be derivable at start of bar t -> shift 1.
    spread_prev = spread.shift(1)
    sigma_prev = sigma.shift(1)
    long_trig = spread_prev < -sigma_n * sigma_prev
    short_trig = spread_prev > sigma_n * sigma_prev

    raw = pd.Series(0.0, index=df.index, dtype="float64")
    raw[long_trig.fillna(False)] = 1.0
    raw[short_trig.fillna(False)] = -1.0

    pos = np.zeros(len(df), dtype="float64")
    counter = 0
    last = 0.0
    raw_arr = raw.to_numpy()
    for i in range(len(df)):
        s = raw_arr[i]
        if s != 0.0:
            last = s
            counter = hold_bars
        if counter > 0:
            pos[i] = last
            counter -= 1
        else:
            pos[i] = 0.0
            last = 0.0
    return pd.Series(pos, index=df.index)


# -- strategy 4: VWAP breakout (H1, crypto) -----------------------------------
def vwap_breakout_position(df: pd.DataFrame,
                           lookback_h1: int = 168,  # 7d * 24h
                           sustained_below: int = 5,
                           hold_bars: int = 8) -> pd.Series:
    """Long on first bar crossing above weekly VWAP after sustained below.

    Mirror for short side: cross below after sustained above.
    """
    close = df["close"].astype("float64")
    wvwap = rolling_vwap(df, lookback_h1)
    above = (close > wvwap).astype("float64")
    below = (close < wvwap).astype("float64")

    # "Sustained below" must be observed BEFORE the cross. We require bars
    # t-sustained_below-1 .. t-2 to all be below, and bar t-1 to cross above.
    # sus_below_prior counts how many of the previous `sustained_below` bars
    # ENDING at t-2 were below VWAP.
    sus_below_prior = below.shift(2).rolling(sustained_below, min_periods=sustained_below).sum()
    sus_above_prior = above.shift(2).rolling(sustained_below, min_periods=sustained_below).sum()

    # Cross at bar t-1: close[t-1] above VWAP[t-1]; close[t-2] below VWAP[t-2].
    above_prev = (close.shift(1) > wvwap.shift(1))
    below_prev = (close.shift(1) < wvwap.shift(1))

    long_trig = above_prev & (sus_below_prior >= sustained_below)
    short_trig = below_prev & (sus_above_prior >= sustained_below)

    pos = np.zeros(len(df), dtype="float64")
    counter = 0
    last = 0.0
    lt = long_trig.fillna(False).to_numpy()
    st = short_trig.fillna(False).to_numpy()
    for i in range(len(df)):
        if lt[i]:
            last = 1.0
            counter = hold_bars
        elif st[i]:
            last = -1.0
            counter = hold_bars
        if counter > 0:
            pos[i] = last
            counter -= 1
        else:
            pos[i] = 0.0
            last = 0.0
    return pd.Series(pos, index=df.index)


# -- strategy 5: high-volume reversal (H1) ------------------------------------
def high_vol_reversal_position(df: pd.DataFrame,
                               pctl_win: int = 500,
                               pctl: float = 0.95,
                               hold_bars: int = 2) -> pd.Series:
    """If vol[t-1] > rolling 95th percentile AND bar[t-1] closed against
    bar[t-2] direction, hold the reversal direction for `hold_bars`.

    "Closed against prior direction" = sign(close[t-1] - open[t-1]) opposite
    to sign(close[t-2] - open[t-2]).
    """
    vol = df["volume"].astype("float64")
    op = df["open"].astype("float64")
    cl = df["close"].astype("float64")

    # Walk-forward percentile: rolling over PRIOR pctl_win bars (exclude self).
    pctl_thresh = vol.shift(1).rolling(pctl_win, min_periods=max(50, pctl_win // 2)).quantile(pctl)
    hi_vol = vol.shift(1) > pctl_thresh

    prev_dir = np.sign(cl.shift(1) - op.shift(1))   # bar t-1 direction
    prev2_dir = np.sign(cl.shift(2) - op.shift(2))  # bar t-2 direction
    reversal = (prev_dir != 0) & (prev2_dir != 0) & (prev_dir != prev2_dir)

    trig_long = hi_vol & reversal & (prev_dir > 0)
    trig_short = hi_vol & reversal & (prev_dir < 0)

    pos = np.zeros(len(df), dtype="float64")
    counter = 0
    last = 0.0
    tl = trig_long.fillna(False).to_numpy()
    ts = trig_short.fillna(False).to_numpy()
    for i in range(len(df)):
        if tl[i]:
            last = 1.0
            counter = hold_bars
        elif ts[i]:
            last = -1.0
            counter = hold_bars
        if counter > 0:
            pos[i] = last
            counter -= 1
        else:
            pos[i] = 0.0
            last = 0.0
    return pd.Series(pos, index=df.index)


# -- driver -------------------------------------------------------------------
@dataclass
class SubResult:
    name: str
    variant: str
    symbol: str
    returns: pd.Series  # UTC-indexed per-bar returns (whatever TF the strat ran on)
    daily: pd.Series    # UTC-day aggregated returns (pre vol-scale)
    stats_full: dict
    stats_is: dict
    stats_oos: dict
    vol_scale: float


def _backtest_position(symbol: str, df: pd.DataFrame, pos: pd.Series,
                       name: str, tf: str) -> pd.Series:
    res = backtest(df, pos, symbol=symbol, timeframe=tf, name=name)
    return _to_utc_index(res.returns, df)


def _daily_collapse(s: pd.Series) -> pd.Series:
    out = s.groupby(s.index.normalize()).sum()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _collect_strategies() -> list[SubResult]:
    split_ts = pd.Timestamp(SPLIT, tz="UTC")
    subs: list[SubResult] = []

    h1: dict[str, pd.DataFrame] = {s: get_candles(s, TF_H1) for s in ALL_SYMBOLS}
    d1: dict[str, pd.DataFrame] = {s: get_candles(s, TF_D1) for s in ALL_SYMBOLS}

    def add(family: str, variant: str, symbol: str, pos: pd.Series,
            df: pd.DataFrame, tf: str) -> None:
        ret = _backtest_position(symbol, df, pos, f"{family}:{variant}", tf=tf)
        daily = _daily_collapse(ret)
        scale = _vol_scale_is(daily, split_ts)
        sf = _stats(daily)
        si = _stats(daily.loc[daily.index < split_ts])
        so = _stats(daily.loc[daily.index >= split_ts])
        subs.append(SubResult(
            name=family, variant=variant, symbol=symbol,
            returns=ret, daily=daily,
            stats_full=sf, stats_is=si, stats_oos=so,
            vol_scale=scale,
        ))

    # -- 1. Daily VWAP mean-reversion (H1) ---------------------------------
    for s in ALL_SYMBOLS:
        df = h1[s]
        pos = daily_vwap_reversion_position(df, upper=1.005, lower=0.995, hold_bars=3)
        add("daily_vwap_reversion", "u005_l005_h3", s, pos, df, TF_H1)

    # -- 2. VWAP-trend follow (H1) -----------------------------------------
    # Use a tighter slope threshold for FX (tiny bps moves), looser for crypto.
    for s in ALL_SYMBOLS:
        df = h1[s]
        # asset-class-dependent slope threshold
        if SYMBOL_TYPE[s].value == "crypto":
            thresh = 20.0  # bps/hr
        elif SYMBOL_TYPE[s].value == "index":
            thresh = 5.0
        else:
            thresh = 2.0
        pos = vwap_trend_position(df, slope_bars=6, slope_bps_per_hr=thresh)
        add("vwap_trend", f"slope6_t{thresh:g}", s, pos, df, TF_H1)

    # -- 3. Weekly VWAP D1 -------------------------------------------------
    for s in ALL_SYMBOLS:
        df = d1[s]
        pos = weekly_vwap_d1_position(df, lookback=7, sigma_n=2.0, hold_bars=3)
        add("weekly_vwap_d1", "w7_s2_h3", s, pos, df, TF_D1)

    # -- 4. VWAP breakout (H1, crypto only) --------------------------------
    for s in CRYPTO:
        df = h1[s]
        pos = vwap_breakout_position(df, lookback_h1=168, sustained_below=5, hold_bars=8)
        add("vwap_breakout", "w168_s5_h8", s, pos, df, TF_H1)

    # -- 5. High-volume reversal (H1, all 13) ------------------------------
    for s in ALL_SYMBOLS:
        df = h1[s]
        pos = high_vol_reversal_position(df, pctl_win=500, pctl=0.95, hold_bars=2)
        add("high_vol_reversal", "p95_w500_h2", s, pos, df, TF_H1)

    return subs


def _combine_survivors(subs: list[SubResult]) -> tuple[pd.Series, list[SubResult]]:
    survivors = [
        sr for sr in subs
        if sr.stats_is["sharpe"] >= IS_SHARPE_MIN
        and sr.stats_oos["sharpe"] >= OOS_SHARPE_MIN
        and sr.vol_scale > 0
        and np.isfinite(sr.vol_scale)
    ]
    if not survivors:
        return pd.Series(dtype="float64"), survivors
    streams = []
    for sr in survivors:
        s = sr.daily * sr.vol_scale
        s.name = f"{sr.name}|{sr.variant}|{sr.symbol}"
        streams.append(s)
    mat = pd.concat(streams, axis=1).fillna(0.0)
    sleeve = mat.mean(axis=1)
    sleeve.name = "ret"
    return sleeve, survivors


def main() -> None:
    print("Building VWAP sub-strategies...")
    subs = _collect_strategies()
    print(f"  total sub-strategies: {len(subs)}")

    sleeve, survivors = _combine_survivors(subs)
    print(f"  survivors (IS>={IS_SHARPE_MIN}, OOS>={OOS_SHARPE_MIN}): {len(survivors)}")

    # -- save sleeve parquet ------------------------------------------------
    out_parquet = OUT_DIR / "vwap_returns.parquet"
    if not sleeve.empty:
        out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    else:
        out_df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[us, UTC]"),
                               "ret": pd.Series(dtype="float64")})
    out_df.to_parquet(out_parquet, index=False)

    # -- per-strategy CSV ---------------------------------------------------
    rows = []
    for sr in subs:
        rows.append({
            "family": sr.name,
            "variant": sr.variant,
            "symbol": sr.symbol,
            "asset_class": SYMBOL_TYPE[sr.symbol].value,
            "full_sharpe": sr.stats_full["sharpe"],
            "full_ann_return": sr.stats_full["ann_return"],
            "full_ann_vol": sr.stats_full["ann_vol"],
            "full_max_dd": sr.stats_full["max_dd"],
            "is_sharpe": sr.stats_is["sharpe"],
            "is_ann_return": sr.stats_is["ann_return"],
            "is_ann_vol": sr.stats_is["ann_vol"],
            "is_max_dd": sr.stats_is["max_dd"],
            "oos_sharpe": sr.stats_oos["sharpe"],
            "oos_ann_return": sr.stats_oos["ann_return"],
            "oos_ann_vol": sr.stats_oos["ann_vol"],
            "oos_max_dd": sr.stats_oos["max_dd"],
            "vol_scale_is": sr.vol_scale,
            "survivor": int(sr.stats_is["sharpe"] >= IS_SHARPE_MIN
                            and sr.stats_oos["sharpe"] >= OOS_SHARPE_MIN
                            and sr.vol_scale > 0
                            and np.isfinite(sr.vol_scale)),
        })
    bd = pd.DataFrame(rows)
    bd.to_csv(OUT_DIR / "vwap_breakdown.csv", index=False)

    split_ts = pd.Timestamp(SPLIT, tz="UTC")

    def block(name: str, r: pd.Series) -> None:
        st = _stats(r)
        print(f"  {name:<10} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print("=== Combined VWAP sleeve (daily UTC) ===")
    if not sleeve.empty:
        block("FULL", sleeve)
        block("IS",   sleeve.loc[sleeve.index < split_ts])
        block("OOS",  sleeve.loc[sleeve.index >= split_ts])
        s2022 = sleeve.loc[(sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                           (sleeve.index < pd.Timestamp("2023-01-01", tz="UTC"))]
        if len(s2022):
            block("2022", s2022)
        print("  Sharpe by year:")
        for yr, sub in sleeve.groupby(sleeve.index.year):
            st = _stats(sub)
            print(f"    {yr}  Sharpe={st['sharpe']:+5.2f}  "
                  f"Ret={st['ann_return']:+7.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    # Family-level aggregates
    print("=== Per-family (vol-scaled, equal-weight within family across survivors) ===")
    families = bd["family"].unique().tolist()
    fam_rows = []
    for fam in families:
        fam_subs = [s for s in subs if s.name == fam
                    and s.stats_is["sharpe"] >= IS_SHARPE_MIN
                    and s.stats_oos["sharpe"] >= OOS_SHARPE_MIN
                    and s.vol_scale > 0 and np.isfinite(s.vol_scale)]
        has_surv = bool(fam_subs)
        if not fam_subs:
            fam_subs = [s for s in subs if s.name == fam
                        and s.vol_scale > 0 and np.isfinite(s.vol_scale)]
        if not fam_subs:
            continue
        streams = []
        for sr in fam_subs:
            streams.append(sr.daily * sr.vol_scale)
        mat = pd.concat(streams, axis=1).fillna(0.0)
        fr = mat.mean(axis=1)
        full = _stats(fr)
        is_ = _stats(fr.loc[fr.index < split_ts])
        oos = _stats(fr.loc[fr.index >= split_ts])
        s2022 = fr.loc[(fr.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                       (fr.index < pd.Timestamp("2023-01-01", tz="UTC"))]
        st2022 = _stats(s2022) if len(s2022) else {"sharpe": 0.0}
        fam_rows.append({
            "family": fam, "n_members": len(fam_subs), "has_survivors": has_surv,
            "is_sharpe": is_["sharpe"], "oos_sharpe": oos["sharpe"],
            "full_sharpe": full["sharpe"], "2022_sharpe": st2022["sharpe"],
            "full_ann_vol": full["ann_vol"], "full_max_dd": full["max_dd"],
        })
    fam_df = pd.DataFrame(fam_rows)
    print(fam_df.to_string(index=False))

    if survivors:
        srv_rows = []
        for sr in survivors:
            srv_rows.append({
                "family": sr.name, "variant": sr.variant, "symbol": sr.symbol,
                "is_sharpe": sr.stats_is["sharpe"],
                "oos_sharpe": sr.stats_oos["sharpe"],
                "vol_scale": sr.vol_scale,
            })
        srv_df = pd.DataFrame(srv_rows).sort_values("oos_sharpe", ascending=False)
        print(f"\n=== Top survivors by OOS Sharpe ({len(survivors)} total) ===")
        print(srv_df.head(20).to_string(index=False))

    print(f"\nWrote {out_parquet}")
    print(f"Wrote {OUT_DIR / 'vwap_breakdown.csv'}")


if __name__ == "__main__":
    main()
