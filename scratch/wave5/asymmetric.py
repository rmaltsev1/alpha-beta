"""Asymmetric long/short strategies.

Markets are not symmetric — skew, regime-dependent drift, short-squeeze
risk all argue for treating the long and the short side differently. This
sleeve builds 7 candidate sub-sleeves, each implementing a different
flavor of asymmetry, then keeps survivors.

Methodology:
    - IS: < 2024-01-01.  OOS: >= 2024-01-01.
    - All thresholds set on IS data only (walk-forward by construction:
      we use trailing rolling stats, but tuning quantiles only on IS).
    - Each survivor vol-scaled to 5% ann vol on IS.
    - Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
    - Combine survivors equal-weight.

Outputs:
    scratch/wave5/asymmetric.py                     (this file)
    scratch/wave5/asymmetric_returns.parquet        (combined sleeve returns)
    scratch/wave5/asymmetric_breakdown.csv          (per-sub-sleeve table)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import CRYPTO, FOREX, INDEX, get_candles
from alphabeta.backtest import cost_for


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scratch" / "wave5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05    # 5% per-sub-sleeve annualised IS vol

INDICES_EQ = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP", "DE30_EUR", "JP225_USD"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ann_factor(symbol: str) -> float:
    return 365.0 if symbol in CRYPTO else 252.0


def load_d1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "D1").sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    df["ret"] = df["close"].pct_change()
    return df


def load_h1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "H1").sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    df["ret"] = df["close"].pct_change()
    return df


def perf_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return dict(sharpe=np.nan, ann_return=np.nan, ann_vol=np.nan, max_dd=np.nan, n=0)
    ann_ret = r.mean() * bpy
    ann_vol = r.std(ddof=0) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(sharpe=float(sharpe), ann_return=float(ann_ret),
                ann_vol=float(ann_vol), max_dd=float(dd), n=int(len(r)))


def split_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_mask = idx < SPLIT
    return {
        "FULL": perf_stats(r, bpy),
        "IS":   perf_stats(r[is_mask], bpy),
        "OOS":  perf_stats(r[~is_mask], bpy),
        "Y2022": perf_stats(r[(idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
                              (idx < pd.Timestamp("2023-01-01", tz="UTC"))], bpy),
        "Y2024_25": perf_stats(r[idx >= pd.Timestamp("2024-01-01", tz="UTC")], bpy),
    }


def to_daily(s: pd.Series) -> pd.Series:
    """Collapse a per-symbol return series to calendar day."""
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).floor("1D")
    s = s.groupby(level=0).sum()
    return s


def vol_scale(ret: pd.Series, target: float = TARGET_SUB_VOL,
              bpy: float = 252.0) -> tuple[pd.Series, float]:
    """Scale a return stream so IS realised ann vol == target. Single scalar."""
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_r = r[idx < SPLIT]
    iv = is_r.std(ddof=0) * np.sqrt(bpy)
    if iv <= 0 or not np.isfinite(iv):
        return ret * 0.0, 0.0
    k = target / iv
    return ret * k, float(k)


def position_to_returns(df: pd.DataFrame, pos: pd.Series, symbol: str) -> pd.Series:
    """Apply position to price series with per-side cost. Indexed by timestamp."""
    pos = pos.astype("float64").fillna(0.0)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos * bar_ret
    cps = cost_for(symbol)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = gross - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


def is_quantile(series: pd.Series, q: float, timestamps: pd.Series) -> float:
    """Quantile computed only on in-sample data (timestamps < SPLIT)."""
    is_mask = timestamps < SPLIT
    is_vals = series[is_mask].dropna()
    if len(is_vals) == 0:
        return np.nan
    return float(is_vals.quantile(q))


# ---------------------------------------------------------------------------
# Sub-sleeve 1: Asymmetric trend-following (long-only)
# Long when 252d return > 0 AND 21d return > 0. No shorts.
# ---------------------------------------------------------------------------
def sleeve1_asym_trend_long(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = ["SPX500_USD", "NAS100_USD", "US30_USD",
            "UK100_GBP", "DE30_EUR", "JP225_USD",
            "BTCUSDT", "ETHUSDT", "XAU_USD"]
    streams = []
    for sym in syms:
        df = data[sym]
        r252 = df["close"].pct_change(252)
        r21 = df["close"].pct_change(21)
        long_sig = (r252 > 0) & (r21 > 0)
        # shift by 1 — known at open of bar t
        pos = long_sig.shift(1).fillna(False).astype(float)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# Symmetric counterpart for direct comparison
def sleeve1b_sym_trend(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = ["SPX500_USD", "NAS100_USD", "US30_USD",
            "UK100_GBP", "DE30_EUR", "JP225_USD",
            "BTCUSDT", "ETHUSDT", "XAU_USD"]
    streams = []
    for sym in syms:
        df = data[sym]
        r252 = df["close"].pct_change(252)
        r21 = df["close"].pct_change(21)
        long_sig  = ((r252 > 0) & (r21 > 0)).astype(float)
        short_sig = ((r252 < 0) & (r21 < 0)).astype(float)
        pos = (long_sig - short_sig).shift(1).fillna(0.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 2: Long-only D1 mean-reversion on big down-days
# Enter long when prior log_ret < -1.5 * std60, hold 3 bars. Never short.
# ---------------------------------------------------------------------------
def sleeve2_long_mr(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = ["SPX500_USD", "NAS100_USD", "US30_USD",
            "UK100_GBP", "DE30_EUR", "JP225_USD",
            "BTCUSDT", "ETHUSDT"]
    streams = []
    for sym in syms:
        df = data[sym]
        std60 = df["log_ret"].rolling(60).std(ddof=1)
        prior_ret = df["log_ret"].shift(1)
        prior_thr = (-1.5 * std60).shift(1)  # threshold known at open of t
        trig = prior_ret < prior_thr

        n = len(df)
        pos = np.zeros(n)
        hold = 0
        for t in range(n):
            if hold > 0:
                pos[t] = 1.0
                hold -= 1
                continue
            if bool(trig.iloc[t]):
                pos[t] = 1.0
                hold = 2  # 1 bar now + 2 more = 3 total
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 3: Asymmetric breakout
# Long on 20d high; exit if close below 10d low. Indices only.
# ---------------------------------------------------------------------------
def sleeve3_asym_breakout(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = INDICES_EQ
    streams = []
    for sym in syms:
        df = data[sym]
        hh20 = df["close"].rolling(20).max()
        ll10 = df["close"].rolling(10).min()
        close = df["close"]

        n = len(df)
        pos = np.zeros(n)
        # at open of bar t we use info up to t-1
        hh20_s = hh20.shift(1).values
        ll10_s = ll10.shift(1).values
        close_s = close.shift(1).values

        long_in = False
        for t in range(n):
            c = close_s[t]; hh = hh20_s[t]; ll = ll10_s[t]
            if not np.isfinite(c) or not np.isfinite(hh) or not np.isfinite(ll):
                pos[t] = 1.0 if long_in else 0.0
                continue
            if not long_in:
                if c >= hh:
                    long_in = True
            else:
                if c <= ll:
                    long_in = False
            pos[t] = 1.0 if long_in else 0.0
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# Symmetric breakout for comparison: long 20d high, short 20d low, exit on
# opposite 10d band — uses symmetric stops.
def sleeve3b_sym_breakout(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = INDICES_EQ
    streams = []
    for sym in syms:
        df = data[sym]
        hh20 = df["close"].rolling(20).max()
        ll20 = df["close"].rolling(20).min()
        hh10 = df["close"].rolling(10).max()
        ll10 = df["close"].rolling(10).min()
        close = df["close"]

        n = len(df)
        pos = np.zeros(n)
        hh20_s = hh20.shift(1).values
        ll20_s = ll20.shift(1).values
        hh10_s = hh10.shift(1).values
        ll10_s = ll10.shift(1).values
        close_s = close.shift(1).values

        state = 0  # -1 short, 0 flat, +1 long
        for t in range(n):
            c = close_s[t]
            h20 = hh20_s[t]; l20 = ll20_s[t]
            h10 = hh10_s[t]; l10 = ll10_s[t]
            if not np.isfinite(c):
                pos[t] = state
                continue
            if state == 0:
                if np.isfinite(h20) and c >= h20:
                    state = 1
                elif np.isfinite(l20) and c <= l20:
                    state = -1
            elif state == 1:
                if np.isfinite(l10) and c <= l10:
                    state = 0
            elif state == -1:
                if np.isfinite(h10) and c >= h10:
                    state = 0
            pos[t] = float(state)
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 4: Skew-conditional fade
# Trailing 60d skew gates side. When skew > +0.5 (right-tailed): short
# local 5d highs. When skew < -0.5 (left-tailed): long local 5d lows.
# ---------------------------------------------------------------------------
def sleeve4_skew_fade(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = ["SPX500_USD", "NAS100_USD", "US30_USD",
            "UK100_GBP", "DE30_EUR", "JP225_USD",
            "BTCUSDT", "ETHUSDT", "XAU_USD",
            "EUR_USD", "GBP_USD", "USD_JPY"]
    streams = []
    for sym in syms:
        df = data[sym]
        skew60 = df["log_ret"].rolling(60).skew()
        hh5 = df["close"].rolling(5).max()
        ll5 = df["close"].rolling(5).min()
        close = df["close"]

        # signals shifted by 1 bar
        skew_s = skew60.shift(1)
        at_high = (close.shift(1) >= hh5.shift(1)).astype(float)
        at_low  = (close.shift(1) <= ll5.shift(1)).astype(float)

        short_sig = ((skew_s > 0.5) & (at_high > 0)).astype(float)
        long_sig  = ((skew_s < -0.5) & (at_low > 0)).astype(float)

        # Hold for 3 bars then exit
        n = len(df)
        pos = np.zeros(n)
        hold = 0
        cur = 0.0
        for t in range(n):
            if hold > 0:
                pos[t] = cur
                hold -= 1
                continue
            ls = long_sig.iloc[t]; ss = short_sig.iloc[t]
            if ls > 0 and ss == 0:
                cur = 1.0; hold = 2; pos[t] = 1.0
            elif ss > 0 and ls == 0:
                cur = -1.0; hold = 2; pos[t] = -1.0
            else:
                cur = 0.0; pos[t] = 0.0
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 5: Asymmetric pair-trade leg sizing
# Pairs: NAS-US30, NAS-SPX, EUR-GBP. Z-score of log-spread.
# Enter long-cheap leg at z < -1.5, enter short-rich leg at z > +2.0.
# Exit on |z| < 0.5.
# ---------------------------------------------------------------------------
def sleeve5_asym_pairs(data: dict[str, pd.DataFrame]) -> pd.Series:
    pairs = [
        ("NAS100_USD", "US30_USD"),
        ("NAS100_USD", "SPX500_USD"),
        ("EUR_USD",    "GBP_USD"),
    ]
    streams = []
    for sym_a, sym_b in pairs:
        a = data[sym_a].set_index("timestamp")["close"].rename("a")
        b = data[sym_b].set_index("timestamp")["close"].rename("b")
        df = pd.concat([a, b], axis=1, join="inner").dropna()
        la = np.log(df["a"]); lb = np.log(df["b"])
        # rolling hedge ratio via simple mean-spread (use IS-period beta = 1 for
        # log prices since both index/forex are similar scale)
        spread = la - lb
        mu = spread.rolling(60).mean()
        sd = spread.rolling(60).std(ddof=1)
        z = (spread - mu) / sd
        z_s = z.shift(1)

        n = len(df)
        pos_a = np.zeros(n); pos_b = np.zeros(n)
        state = 0  # +1 long-a-short-b, -1 short-a-long-b, 0 flat
        z_vals = z_s.values
        for t in range(n):
            zv = z_vals[t]
            if not np.isfinite(zv):
                pos_a[t] = state; pos_b[t] = -state
                continue
            if state == 0:
                if zv < -1.5:
                    state = 1   # spread low: long a, short b
                elif zv > 2.0:
                    state = -1  # spread high: short a, long b
            elif state == 1:
                if zv > -0.5:
                    state = 0
            elif state == -1:
                if zv < 0.5:
                    state = 0
            pos_a[t] = float(state); pos_b[t] = float(-state)
        pos_a = pd.Series(pos_a, index=df.index)
        pos_b = pd.Series(pos_b, index=df.index)

        # bar returns and costs
        ra = df["a"].pct_change().fillna(0.0)
        rb = df["b"].pct_change().fillna(0.0)
        cpa = cost_for(sym_a); cpb = cost_for(sym_b)
        net = (pos_a * ra - pos_a.diff().fillna(pos_a.iloc[0]).abs() * cpa) + \
              (pos_b * rb - pos_b.diff().fillna(pos_b.iloc[0]).abs() * cpb)
        # 0.5 weight per leg so the pair has unit gross
        net = net * 0.5
        net.name = f"{sym_a}-{sym_b}"
        streams.append(to_daily(net).rename(f"{sym_a}-{sym_b}"))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# Symmetric pair counterpart: |z|>1.5 entry both sides, |z|<0.5 exit
def sleeve5b_sym_pairs(data: dict[str, pd.DataFrame]) -> pd.Series:
    pairs = [
        ("NAS100_USD", "US30_USD"),
        ("NAS100_USD", "SPX500_USD"),
        ("EUR_USD",    "GBP_USD"),
    ]
    streams = []
    for sym_a, sym_b in pairs:
        a = data[sym_a].set_index("timestamp")["close"].rename("a")
        b = data[sym_b].set_index("timestamp")["close"].rename("b")
        df = pd.concat([a, b], axis=1, join="inner").dropna()
        la = np.log(df["a"]); lb = np.log(df["b"])
        spread = la - lb
        mu = spread.rolling(60).mean()
        sd = spread.rolling(60).std(ddof=1)
        z = (spread - mu) / sd
        z_s = z.shift(1)
        n = len(df)
        pos_a = np.zeros(n); pos_b = np.zeros(n)
        state = 0
        z_vals = z_s.values
        for t in range(n):
            zv = z_vals[t]
            if not np.isfinite(zv):
                pos_a[t] = state; pos_b[t] = -state
                continue
            if state == 0:
                if zv < -1.5: state = 1
                elif zv > 1.5: state = -1
            elif state == 1:
                if zv > -0.5: state = 0
            elif state == -1:
                if zv < 0.5: state = 0
            pos_a[t] = float(state); pos_b[t] = float(-state)
        pos_a = pd.Series(pos_a, index=df.index)
        pos_b = pd.Series(pos_b, index=df.index)
        ra = df["a"].pct_change().fillna(0.0)
        rb = df["b"].pct_change().fillna(0.0)
        cpa = cost_for(sym_a); cpb = cost_for(sym_b)
        net = (pos_a * ra - pos_a.diff().fillna(pos_a.iloc[0]).abs() * cpa) + \
              (pos_b * rb - pos_b.diff().fillna(pos_b.iloc[0]).abs() * cpb)
        net = net * 0.5
        streams.append(to_daily(net).rename(f"{sym_a}-{sym_b}"))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 6: Time-of-day asymmetric momentum on XAU
# Long XAU during evening hours (asia/europe overnight 18-23 UTC), but
# explicitly SHORT XAU during the US session consolidation (15-17 UTC) on
# days that already saw strong morning rallies.
# ---------------------------------------------------------------------------
def sleeve6_xau_tod(data_h1: dict[str, pd.DataFrame]) -> pd.Series:
    df = data_h1["XAU_USD"].copy()
    df["hour"] = df["timestamp"].dt.hour
    # daily morning return: from 06:00 UTC up to 14:00 UTC of same date
    df["date"] = df["timestamp"].dt.floor("1D")
    # compute morning return per date as close@14 / close@06 - 1
    close_at_h = df.pivot_table(index="date", columns="hour", values="close", aggfunc="last")
    if 14 in close_at_h.columns and 6 in close_at_h.columns:
        morn_ret = (close_at_h[14] / close_at_h[6] - 1).rename("morn_ret")
    else:
        morn_ret = pd.Series(0.0, index=close_at_h.index, name="morn_ret")
    df = df.merge(morn_ret.reset_index(), on="date", how="left")

    # base long position during evening UTC 18..23 (5 hours)
    long_hours = df["hour"].isin([18, 19, 20, 21, 22, 23]).astype(float)
    # short position during 15..17 UTC IF morning was up strongly (>+0.4% by 14 UTC)
    short_hours = (df["hour"].isin([15, 16, 17]) & (df["morn_ret"].shift(1) > 0.004)).astype(float)
    pos = (long_hours - short_hours).shift(1).fillna(0.0)
    ret = position_to_returns(df, pos, "XAU_USD")
    # keep H1 hourly returns then daily collapse
    return to_daily(ret)


# Symmetric counterpart: long evening only, no short
def sleeve6b_sym_eve_long(data_h1: dict[str, pd.DataFrame]) -> pd.Series:
    df = data_h1["XAU_USD"].copy()
    df["hour"] = df["timestamp"].dt.hour
    long_hours = df["hour"].isin([18, 19, 20, 21, 22, 23]).astype(float)
    pos = long_hours.shift(1).fillna(0.0)
    ret = position_to_returns(df, pos, "XAU_USD")
    return to_daily(ret)


# ---------------------------------------------------------------------------
# Sub-sleeve 7: Reversal regime detector
# Trade D1 mean-reversion in LOW-vol regimes (vol < IS p33 of rv30).
# Trade trend in HIGH-vol regimes (vol > IS p66). Indices.
# ---------------------------------------------------------------------------
def sleeve7_regime(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = INDICES_EQ
    streams = []
    for sym in syms:
        df = data[sym]
        rv30 = df["log_ret"].rolling(30).std(ddof=1) * np.sqrt(252.0)
        p33 = is_quantile(rv30, 0.33, df["timestamp"])
        p66 = is_quantile(rv30, 0.66, df["timestamp"])

        # MR signal: short prior up-day, long prior down-day (1-bar hold) — applied in LOW vol
        std60 = df["log_ret"].rolling(60).std(ddof=1)
        mr_long = ((df["log_ret"].shift(1) < -1.0 * std60.shift(1)) & (rv30.shift(1) < p33)).astype(float)
        mr_short = ((df["log_ret"].shift(1) > +1.0 * std60.shift(1)) & (rv30.shift(1) < p33)).astype(float)

        # Trend signal: sign(252d return) in HIGH vol
        r252 = df["close"].pct_change(252).shift(1)
        tr_long = ((r252 > 0) & (rv30.shift(1) > p66)).astype(float)
        tr_short = ((r252 < 0) & (rv30.shift(1) > p66)).astype(float)

        pos = (mr_long - mr_short + tr_long - tr_short).fillna(0.0)
        pos = pos.clip(-1.0, 1.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading D1 data ===")
    d1_syms = sorted(set(INDICES_EQ + ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY",
                                       "BTCUSDT", "ETHUSDT", "SOLUSDT"]))
    data = {s: load_d1(s) for s in d1_syms}
    print("=== Loading H1 data (XAU only) ===")
    data_h1 = {"XAU_USD": load_h1("XAU_USD")}

    builders = {
        "S1_asym_trend_long":     lambda: sleeve1_asym_trend_long(data),
        "S1b_sym_trend":          lambda: sleeve1b_sym_trend(data),       # for comparison only
        "S2_long_only_mr":        lambda: sleeve2_long_mr(data),
        "S3_asym_breakout":       lambda: sleeve3_asym_breakout(data),
        "S3b_sym_breakout":       lambda: sleeve3b_sym_breakout(data),    # for comparison only
        "S4_skew_fade":           lambda: sleeve4_skew_fade(data),
        "S5_asym_pairs":          lambda: sleeve5_asym_pairs(data),
        "S5b_sym_pairs":          lambda: sleeve5b_sym_pairs(data),       # for comparison only
        "S6_xau_tod_asym":        lambda: sleeve6_xau_tod(data_h1),
        "S6b_xau_eve_long":       lambda: sleeve6b_sym_eve_long(data_h1), # for comparison only
        "S7_reversal_regime":     lambda: sleeve7_regime(data),
    }
    # primary asymmetric strategies eligible for combination
    ASYM_NAMES = ["S1_asym_trend_long", "S2_long_only_mr", "S3_asym_breakout",
                  "S4_skew_fade", "S5_asym_pairs", "S6_xau_tod_asym",
                  "S7_reversal_regime"]

    print("=== Building sub-sleeves ===")
    raw_streams: dict[str, pd.Series] = {}
    for name, fn in builders.items():
        s = fn()
        s = s.sort_index()
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
        raw_streams[name] = s
        print(f"  built {name}: {len(s)} bars, "
              f"first={s.index.min().date()}, last={s.index.max().date()}")

    # Vol-scale each to 5% ann on IS, then re-evaluate stats.
    scaled_streams: dict[str, pd.Series] = {}
    rows = []
    for name, s in raw_streams.items():
        scaled, k = vol_scale(s, target=TARGET_SUB_VOL, bpy=252.0)
        scaled_streams[name] = scaled
        stats = split_stats(scaled, bpy=252.0)
        rows.append({
            "sleeve": name,
            "scale_factor": k,
            "FULL_sharpe":  stats["FULL"]["sharpe"],
            "FULL_ret":     stats["FULL"]["ann_return"],
            "FULL_vol":     stats["FULL"]["ann_vol"],
            "FULL_dd":      stats["FULL"]["max_dd"],
            "IS_sharpe":    stats["IS"]["sharpe"],
            "IS_ret":       stats["IS"]["ann_return"],
            "IS_dd":        stats["IS"]["max_dd"],
            "OOS_sharpe":   stats["OOS"]["sharpe"],
            "OOS_ret":      stats["OOS"]["ann_return"],
            "OOS_dd":       stats["OOS"]["max_dd"],
            "Y2022_sharpe": stats["Y2022"]["sharpe"],
            "Y2022_ret":    stats["Y2022"]["ann_return"],
            "Y2024_25_sharpe": stats["Y2024_25"]["sharpe"],
            "Y2024_25_ret":    stats["Y2024_25"]["ann_return"],
        })
    breakdown = pd.DataFrame(rows).set_index("sleeve")

    # Survivors: IS Sharpe >= 0.5 AND OOS Sharpe >= 0  — among ASYM_NAMES only.
    eligible = breakdown.loc[ASYM_NAMES]
    is_ok = eligible["IS_sharpe"] >= 0.5
    oos_ok = eligible["OOS_sharpe"] >= 0.0
    survivors = eligible[is_ok & oos_ok].index.tolist()
    breakdown["survivor"] = breakdown.index.isin(survivors)

    print("\n=== Per sub-sleeve breakdown ===")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print(breakdown)

    # Combine survivors equal-weight on union calendar.
    if survivors:
        panel = pd.concat({k: scaled_streams[k] for k in survivors},
                          axis=1, sort=True).fillna(0.0)
        combined = panel.mean(axis=1)
    else:
        all_idx = pd.DatetimeIndex(sorted(set(
            ts for s in scaled_streams.values() for ts in s.index
        )), tz="UTC")
        combined = pd.Series(0.0, index=all_idx)

    combined = combined.sort_index()
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC")

    combined_stats = split_stats(combined, bpy=252.0)
    print("\n=== Survivors:", survivors)
    print("=== Combined sleeve stats ===")
    for k, v in combined_stats.items():
        print(f"  {k:>9} : sharpe={v['sharpe']:+.3f} ret={v['ann_return']:+.3%} "
              f"vol={v['ann_vol']:.3%} dd={v['max_dd']:+.3%} n={v['n']}")

    # Append COMBINED row
    combined_row = {
        "scale_factor": 1.0,
        "FULL_sharpe":  combined_stats["FULL"]["sharpe"],
        "FULL_ret":     combined_stats["FULL"]["ann_return"],
        "FULL_vol":     combined_stats["FULL"]["ann_vol"],
        "FULL_dd":      combined_stats["FULL"]["max_dd"],
        "IS_sharpe":    combined_stats["IS"]["sharpe"],
        "IS_ret":       combined_stats["IS"]["ann_return"],
        "IS_dd":        combined_stats["IS"]["max_dd"],
        "OOS_sharpe":   combined_stats["OOS"]["sharpe"],
        "OOS_ret":      combined_stats["OOS"]["ann_return"],
        "OOS_dd":       combined_stats["OOS"]["max_dd"],
        "Y2022_sharpe": combined_stats["Y2022"]["sharpe"],
        "Y2022_ret":    combined_stats["Y2022"]["ann_return"],
        "Y2024_25_sharpe": combined_stats["Y2024_25"]["sharpe"],
        "Y2024_25_ret":    combined_stats["Y2024_25"]["ann_return"],
        "survivor": True,
    }
    breakdown.loc["COMBINED"] = combined_row
    breakdown.to_csv(OUT_DIR / "asymmetric_breakdown.csv", float_format="%.4f")

    out = combined.rename("ret").to_frame()
    out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
    out = out.reset_index()
    out.to_parquet(OUT_DIR / "asymmetric_returns.parquet", index=False)
    print("\nSaved:")
    print(f"  {OUT_DIR / 'asymmetric_returns.parquet'}")
    print(f"  {OUT_DIR / 'asymmetric_breakdown.csv'}")


if __name__ == "__main__":
    main()
