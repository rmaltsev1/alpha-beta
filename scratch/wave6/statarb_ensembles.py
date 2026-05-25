"""Wave-6: Statistical-arbitrage ensemble of mean-reversion signals.

We already have a "vanilla D1REV" family (see adaptive_reversion.py). This
file builds *complementary* mean-reversion sub-sleeves that operate on
signals not yet exercised:

    S1  OU (Ornstein-Uhlenbeck) reversion.
            Fit OU on 60d rolling window of log-price per symbol; build
            (theta, sigma_OU, half-life). Enter at +/- 2 sigma_OU. Hold
            for the (capped) half-life. Walk-forward, per symbol.
    S2  Half-life filtered reversion.
            Same OU half-life on 60d window. Vanilla D1REV-style fade only
            when half-life < 10 days (i.e. fast mean-reverters). Slower ->
            do not trade.
    S3  Multi-horizon reversion confirmation.
            Long when price < 5d MA AND price < 21d MA AND price < 63d MA.
            Short opposite. Multi-scale agreement.
    S4  Cross-sectional dollar-neutral 5d-return reversion.
            Each day rank 13 symbols by trailing 5d return. Long bottom 3,
            short top 3, equal-weight, dollar-neutral. Rebalance daily.
    S5  Bollinger-z relative ranking.
            For every symbol compute z = (close - MA20) / std20. Long the
            3 most-negative-z, short the 3 most-positive-z, equal-weight
            basket. Cross-sectional reversion.
    S6  Vol-adjusted reversion intensity.
            Standard D1 fade but the trigger is adaptive: high-vol periods
            require 2 sigma daily moves, low-vol periods require 1 sigma.
            Vol regime split on IS 30d realised-vol median.
    S7  Reversion-of-momentum.
            When trailing 21d momentum > IS 252d 90th percentile (over-
            extended), fade for the next 5 bars. Per symbol, symmetric.

Methodology:
    IS  < 2024-01-01;  OOS >= 2024-01-01.
    All adaptive thresholds/quantiles set on IS data only ("walk-forward"
    in the sense that any parameter that needs an empirical distribution
    uses IS history).
    Each sub-sleeve vol-scaled so IS realised vol == 5% ann.
    Survival: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.

Outputs (all in scratch/wave6/):
    statarb_ensembles.py            (this file)
    statarb_returns.parquet         (combined survivor returns)
    statarb_breakdown.csv           (per sub-sleeve + COMBINED table)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import ALL_SYMBOLS, CRYPTO, get_candles
from alphabeta.backtest import cost_for


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scratch" / "wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05  # 5% ann vol per sub-sleeve

# Walk-forward window for OU fit
OU_WIN = 60
# Cross-sectional baskets
XS_TOP = 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_d1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "D1").sort_values("timestamp").reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["log_ret"] = df["log_close"].diff()
    df["ret"] = df["close"].pct_change()
    return df


def perf_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return dict(sharpe=np.nan, ann_return=np.nan, ann_vol=np.nan,
                    max_dd=np.nan, n=0)
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
        "FULL":  perf_stats(r, bpy),
        "IS":    perf_stats(r[is_mask], bpy),
        "OOS":   perf_stats(r[~is_mask], bpy),
        "Y2022": perf_stats(r[(idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
                              (idx < pd.Timestamp("2023-01-01", tz="UTC"))], bpy),
        "Y2024_25": perf_stats(r[idx >= pd.Timestamp("2024-01-01", tz="UTC")], bpy),
    }


def to_daily(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).floor("1D")
    s = s.groupby(level=0).sum()
    return s


def vol_scale(ret: pd.Series, target: float = TARGET_SUB_VOL,
              bpy: float = 252.0) -> tuple[pd.Series, float]:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_r = r[idx < SPLIT]
    iv = is_r.std(ddof=0) * np.sqrt(bpy)
    if iv <= 0 or not np.isfinite(iv):
        return ret * 0.0, 0.0
    k = target / iv
    return ret * k, float(k)


def position_to_returns(df: pd.DataFrame, pos: pd.Series, symbol: str) -> pd.Series:
    """Per-bar net return as a Series indexed by timestamp."""
    pos = pos.astype("float64").fillna(0.0)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos * bar_ret
    cps = cost_for(symbol)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = gross - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


# ---------------------------------------------------------------------------
# OU fit on rolling 60d window of log-prices.
# AR(1) on log-price:  x[t] = a + b*x[t-1] + eps
# Continuous-time:   theta = a/(1-b), kappa = -log(b)/dt, half-life = log(2)/kappa
# Equilibrium sigma_OU = std(x_t) (stationary unconditional std under OU).
# All computed *only* on info up to t-1; positions enter on t.
# ---------------------------------------------------------------------------
def rolling_ou_params(log_close: pd.Series, win: int = OU_WIN
                      ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (theta, sigma_OU, half_life), aligned to the input index.

    Each value at index t uses the win observations ending at t-1 (so it's
    usable to position bar t).
    """
    n = len(log_close)
    theta = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    hl = np.full(n, np.nan)
    x = log_close.values
    for t in range(win + 1, n):
        # use bars [t-win-1 ... t-1] (win+1 points -> win lagged pairs)
        seg = x[t - win - 1: t]
        if np.any(~np.isfinite(seg)):
            continue
        x_lag = seg[:-1]
        x_now = seg[1:]
        # OLS  x_now = a + b * x_lag
        # closed form: cov / var
        xm = x_lag.mean()
        ym = x_now.mean()
        denom = ((x_lag - xm) ** 2).sum()
        if denom <= 0:
            continue
        b = ((x_lag - xm) * (x_now - ym)).sum() / denom
        a = ym - b * xm
        # need |b| < 1 for stationarity; clamp
        if not (0.0 < b < 1.0):
            continue
        thet = a / (1.0 - b)
        kappa = -np.log(b)   # per-bar
        if kappa <= 0 or not np.isfinite(kappa):
            continue
        half_life = np.log(2.0) / kappa
        # OU stationary std under AR(1):  sigma_eps / sqrt(1 - b^2)
        resid = x_now - (a + b * x_lag)
        sig_eps = resid.std(ddof=1)
        sig_eq = sig_eps / np.sqrt(max(1.0 - b * b, 1e-12))
        theta[t] = thet
        sigma[t] = sig_eq
        hl[t] = half_life
    return (pd.Series(theta, index=log_close.index),
            pd.Series(sigma, index=log_close.index),
            pd.Series(hl, index=log_close.index))


# ---------------------------------------------------------------------------
# S1: OU reversion.
# When log-price > theta + 2*sigma_OU -> short for half-life days (cap 10).
# When log-price < theta - 2*sigma_OU -> long  for half-life days (cap 10).
# A new signal overrides any in-flight position.
# ---------------------------------------------------------------------------
def sleeve1_ou_reversion(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        theta, sig_ou, hl = rolling_ou_params(df["log_close"], win=OU_WIN)
        # signals known at end of bar t-1  ->  shift(1)
        lx_prev = df["log_close"].shift(1)
        theta_p = theta.shift(1)
        sigma_p = sig_ou.shift(1)
        hl_p = hl.shift(1).clip(upper=10.0)  # cap hold
        upper = theta_p + 2.0 * sigma_p
        lower = theta_p - 2.0 * sigma_p
        trig_short = (lx_prev > upper).fillna(False).values
        trig_long = (lx_prev < lower).fillna(False).values
        hold_n = hl_p.fillna(0.0).clip(lower=0.0).round().astype(int).values
        n = len(df)
        pos = np.zeros(n)
        remaining = 0
        cur = 0.0
        for t in range(n):
            if trig_short[t]:
                cur = -1.0
                remaining = max(int(hold_n[t]), 1)
            elif trig_long[t]:
                cur = +1.0
                remaining = max(int(hold_n[t]), 1)
            if remaining > 0:
                pos[t] = cur
                remaining -= 1
            else:
                pos[t] = 0.0
                cur = 0.0
        pos_s = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos_s, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S2: Half-life filtered D1 reversion.
# Compute OU half-life on 60d window. Only fade prior-day move when
# half_life < 10 days. Otherwise stand aside.
# ---------------------------------------------------------------------------
def sleeve2_halflife_filter(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        _, _, hl = rolling_ou_params(df["log_close"], win=OU_WIN)
        log_ret = df["log_ret"]
        thresh = 50.0 / 10_000.0  # match vanilla D1REV threshold
        raw_dir = pd.Series(0.0, index=df.index)
        raw_dir[log_ret > thresh] = -1.0
        raw_dir[log_ret < -thresh] = +1.0
        # position for bar t = signal at t-1, gated by HL at t-1 < 10
        gate = (hl.shift(1) < 10.0).fillna(False).astype(float)
        pos = raw_dir.shift(1).fillna(0.0) * gate
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S3: Multi-horizon reversion confirmation.
# Long when close < SMA5 AND close < SMA21 AND close < SMA63 (all three agree
# "cheap"). Short opposite. Hold for 1 bar each day (continuous).
# ---------------------------------------------------------------------------
def sleeve3_multihorizon(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        c = df["close"]
        ma5 = c.rolling(5, min_periods=5).mean()
        ma21 = c.rolling(21, min_periods=21).mean()
        ma63 = c.rolling(63, min_periods=63).mean()
        cheap = ((c < ma5) & (c < ma21) & (c < ma63)).astype(float)
        rich = ((c > ma5) & (c > ma21) & (c > ma63)).astype(float)
        raw = cheap - rich  # +1/0/-1
        pos = raw.shift(1).fillna(0.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S4: Cross-sectional dollar-neutral 5d-reversion.
# Each day rank the 13 symbols by trailing 5-day return (close.pct_change(5)).
# Long bottom 3 (worst performers), short top 3 (best performers), equal-
# weight inside each side. Rebalance daily. Combine on union calendar.
# Each side is sized 1/3, so total gross = 2.0 and net = 0.0 (dollar-neutral).
# We then *normalise to gross-exposure 1.0* for vol-scaling consistency.
# ---------------------------------------------------------------------------
def sleeve4_xs_5d(data: dict[str, pd.DataFrame]) -> pd.Series:
    # Build a wide panel of trailing 5d returns (each column = a symbol),
    # indexed by daily timestamp. Use t-1 info to position bar t.
    streams_close = {}
    for sym, df in data.items():
        c = df["close"].copy()
        c.index = pd.DatetimeIndex(df["timestamp"]).floor("1D")
        # collapse same-day duplicates (shouldn't be any on D1, but defensive)
        c = c.groupby(level=0).last()
        streams_close[sym] = c
    panel = pd.DataFrame(streams_close).sort_index()
    rets5 = panel.pct_change(5)
    daily_ret = panel.pct_change()  # bar return same calendar
    # signal at the start of day t uses rets5 known at end of t-1.
    sig5 = rets5.shift(1)
    # Per-day rank (only over non-NaN entries)
    pos_panel = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    # iterate by row -> small (~1700) so cheap
    syms = panel.columns.tolist()
    for t, row in sig5.iterrows():
        valid = row.dropna()
        if len(valid) < 2 * XS_TOP:
            continue
        sorted_syms = valid.sort_values()
        longs = sorted_syms.index[:XS_TOP]
        shorts = sorted_syms.index[-XS_TOP:]
        w = 1.0 / XS_TOP
        for s in longs:
            pos_panel.at[t, s] = +w
        for s in shorts:
            pos_panel.at[t, s] = -w
    # gross PnL: positions * daily_ret; subtract cost = |dpos| * cost_per_side
    gross = (pos_panel * daily_ret).fillna(0.0)
    dpos = pos_panel.diff().abs().fillna(pos_panel.abs())
    cost_vec = pd.Series({s: cost_for(s) for s in syms})
    cost = dpos * cost_vec
    net_per_sym = gross - cost
    # Sum across symbols -> portfolio return per day. Net total exposure = 0,
    # gross = 2/3 * (top+bottom) = 2.0 in absolute terms? Actually each leg is
    # sized 1/3 across XS_TOP names; long-leg gross = 1, short-leg gross = 1,
    # total gross = 2.0. We'll *not* renormalise here — vol_scale handles it.
    return net_per_sym.sum(axis=1)


# ---------------------------------------------------------------------------
# S5: Bollinger-z relative ranking.
# For each symbol compute z = (close - MA20) / std20. Cross-sectionally each
# day, long the 3 most-negative-z (oversold), short the 3 most-positive-z
# (overbought). Equal-weight basket, rebalance daily.
# ---------------------------------------------------------------------------
def sleeve5_bollinger_xs(data: dict[str, pd.DataFrame]) -> pd.Series:
    z_panel = {}
    close_panel = {}
    for sym, df in data.items():
        c = df["close"]
        ma = c.rolling(20, min_periods=20).mean()
        sd = c.rolling(20, min_periods=20).std(ddof=0)
        z = (c - ma) / sd
        z.index = pd.DatetimeIndex(df["timestamp"]).floor("1D")
        c2 = c.copy()
        c2.index = z.index
        z = z.groupby(level=0).last()
        c2 = c2.groupby(level=0).last()
        z_panel[sym] = z
        close_panel[sym] = c2
    zp = pd.DataFrame(z_panel).sort_index()
    cp = pd.DataFrame(close_panel).sort_index()
    daily_ret = cp.pct_change()
    sig = zp.shift(1)  # z known at end of t-1 -> position for t
    pos_panel = pd.DataFrame(0.0, index=zp.index, columns=zp.columns)
    syms = zp.columns.tolist()
    for t, row in sig.iterrows():
        valid = row.dropna()
        if len(valid) < 2 * XS_TOP:
            continue
        sorted_syms = valid.sort_values()
        longs = sorted_syms.index[:XS_TOP]   # most negative z
        shorts = sorted_syms.index[-XS_TOP:]  # most positive z
        w = 1.0 / XS_TOP
        for s in longs:
            pos_panel.at[t, s] = +w
        for s in shorts:
            pos_panel.at[t, s] = -w
    gross = (pos_panel * daily_ret).fillna(0.0)
    dpos = pos_panel.diff().abs().fillna(pos_panel.abs())
    cost_vec = pd.Series({s: cost_for(s) for s in syms})
    cost = dpos * cost_vec
    net_per_sym = gross - cost
    return net_per_sym.sum(axis=1)


# ---------------------------------------------------------------------------
# S6: Vol-adjusted reversion intensity.
# Standard fade-yesterday but the trigger size scales with realised vol:
#     rv30 (30d std of log_ret, annualised)
#     IS median rv30 -> the regime split per symbol
#     trigger = (1 sigma) of yesterday's |log_ret| in low-vol, (2 sigma) in
#     high-vol, where sigma here is the realised-vol per-bar (rv30 / sqrt(252)).
# Per-symbol, then equal-weight basket across 13.
# ---------------------------------------------------------------------------
def sleeve6_vol_adj(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        log_ret = df["log_ret"]
        rv30 = log_ret.rolling(30, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        # IS median
        is_mask = df["timestamp"] < SPLIT
        med = float(rv30[is_mask].dropna().median())
        # per-bar sigma
        sigma_bar = rv30 / np.sqrt(252.0)
        # high-vol regime requires 2 sigma_bar, low-vol 1 sigma_bar
        high = rv30 >= med
        k = pd.Series(1.0, index=df.index)
        k[high] = 2.0
        trig = (k * sigma_bar).shift(1)
        lr_prev = log_ret.shift(1)
        pos = pd.Series(0.0, index=df.index)
        pos[lr_prev > trig] = -1.0
        pos[lr_prev < -trig] = +1.0
        # already shifted (using lr_prev, trig already at t-1) -> pos[t] valid
        ret = position_to_returns(df, pos.fillna(0.0), sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S7: Reversion-of-momentum.
# Per symbol: trailing 21d momentum = close.pct_change(21). If, at end of
# bar t-1, momentum is ABOVE the 90th percentile of its trailing 252d history
# (using only IS-period quantile for the threshold), then fade for 5 bars
# (short). Symmetric on the downside (below 10th pct  -> long for 5).
# Threshold = IS-period 90th percentile of |21d momentum| values.
# ---------------------------------------------------------------------------
def sleeve7_rev_of_mom(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        mom21 = df["close"].pct_change(21)
        is_mask = df["timestamp"] < SPLIT
        is_vals = mom21[is_mask].dropna()
        if len(is_vals) < 50:
            streams.append(pd.Series(dtype=float, name=sym))
            continue
        p90 = float(is_vals.quantile(0.90))
        p10 = float(is_vals.quantile(0.10))
        trig_short = (mom21.shift(1) > p90).fillna(False).values
        trig_long = (mom21.shift(1) < p10).fillna(False).values
        n = len(df)
        pos = np.zeros(n)
        remaining = 0
        cur = 0.0
        for t in range(n):
            if trig_short[t]:
                cur = -1.0
                remaining = 5
            elif trig_long[t]:
                cur = +1.0
                remaining = 5
            if remaining > 0:
                pos[t] = cur
                remaining -= 1
            else:
                pos[t] = 0.0
                cur = 0.0
        pos_s = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos_s, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading D1 data for 13 symbols ===")
    data = {s: load_d1(s) for s in ALL_SYMBOLS}
    for s in ALL_SYMBOLS:
        df = data[s]
        print(f"  {s:<12}: {len(df)} bars  "
              f"{df['timestamp'].iloc[0].date()} .. {df['timestamp'].iloc[-1].date()}")

    builders = {
        "S1_OU_reversion":   lambda: sleeve1_ou_reversion(data),
        "S2_halflife_filt":  lambda: sleeve2_halflife_filter(data),
        "S3_multi_horizon":  lambda: sleeve3_multihorizon(data),
        "S4_xs_5d_neutral":  lambda: sleeve4_xs_5d(data),
        "S5_bollinger_xs":   lambda: sleeve5_bollinger_xs(data),
        "S6_vol_adj_rev":    lambda: sleeve6_vol_adj(data),
        "S7_rev_of_mom":     lambda: sleeve7_rev_of_mom(data),
    }

    print("\n=== Building sub-sleeves ===")
    raw_streams: dict[str, pd.Series] = {}
    for name, fn in builders.items():
        s = fn().sort_index()
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
        raw_streams[name] = s
        print(f"  built {name:<22} n={len(s)} first={s.index.min().date()}"
              f" last={s.index.max().date()}")

    print("\n=== Vol-scale + per-sleeve stats ===")
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

    # Survival filter
    is_ok = breakdown["IS_sharpe"] >= 0.5
    oos_ok = breakdown["OOS_sharpe"] >= 0.0
    survivors = breakdown[is_ok & oos_ok].index.tolist()
    breakdown["survivor"] = breakdown.index.isin(survivors)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print("\n=== Per-sub-sleeve breakdown ===")
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
    print("=== Combined statarb ensemble stats ===")
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

    # Correlation between sub-sleeves (on FULL period, scaled).
    print("\n=== Pairwise correlation of vol-scaled sub-sleeves (FULL) ===")
    corr_panel = pd.concat(scaled_streams, axis=1).fillna(0.0)
    corr_panel.columns = list(scaled_streams.keys())
    print(corr_panel.corr().round(3))

    # Save.
    breakdown.to_csv(OUT_DIR / "statarb_breakdown.csv", float_format="%.4f")
    out = combined.rename("ret").to_frame()
    out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
    out = out.reset_index()
    out.to_parquet(OUT_DIR / "statarb_returns.parquet", index=False)

    print("\nSaved:")
    print(f"  {OUT_DIR / 'statarb_returns.parquet'}")
    print(f"  {OUT_DIR / 'statarb_breakdown.csv'}")


if __name__ == "__main__":
    main()
