"""FX-specific strategies — wave 6.

The 4 FX pairs (EUR_USD, GBP_USD, USD_JPY, XAU_USD) have been under-exploited
in prior waves. This sleeve builds 8 strategies tailored to FX-specific
microstructure: rate-differential proxies (carry), DXY momentum, JPY-vs-
everything baskets, EUR-GBP cross mean-reversion, post-Brexit GBP vol,
gold Donchian breakouts, FX risk-off coordination, and Asian-session FX
momentum.

Strategies
----------
  1. USDJPY_CARRY      — long USD_JPY when 252d slope > 0 AND realized vol
                         below IS p75 (smooth uptrend = carry working).
  2. EURGBP_MEANREV    — synthetic EUR/GBP cross, 60d z-score extreme MR.
  3. JPY_BASKET        — long USD_JPY + short EUR_USD + short GBP_USD
                         (long-USD/short-JPY basket). On when SPX vol low.
  4. GBP_BOLLINGER     — GBP_USD H1 60-day Bollinger band breakout.
  5. XAU_DONCHIAN      — Daily Donchian: long on 20d-high break, exit at 10d low.
  6. DXY_TSMOM         — DXY proxy = -.58*EUR -.12*GBP +.14*USDJPY; 63/252 TSMOM
                         ensemble; long when both lookbacks positive.
  7. RISK_OFF_DXY      — when SPX 5d ret < -3% AND USD_JPY rising, long DXY.
  8. ASIAN_FX_MOM      — EUR_USD / GBP_USD Asian-session (00-06 UTC) > 0.5 SD
                         move → follow direction next H1 bar.

Methodology
-----------
  - D1 + H1 parquets, FX cost = 1bp / side (engine default).
  - All signals walk-forward (shift(1), expanding stats, no IS-only fits on OOS).
  - IS:  timestamp <  2024-01-01
  - OOS: timestamp >= 2024-01-01.
  - Each surviving sub-sleeve vol-scaled to 5% IS ann vol.
  - Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/fx_specific.py
  scratch/wave6/fx_specific_returns.parquet
  scratch/wave6/fx_specific_breakdown.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles  # noqa: E402
from alphabeta.backtest import backtest  # noqa: E402

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
SUB_TARGET_VOL = 0.05
BPY_D = 252.0   # FX trading days/yr (D1 ~ business days)
BPY_H = 252.0 * 24  # H1 bars/yr proxy

FX_SYMS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]


# --- helpers ---------------------------------------------------------------

def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label: str, r: pd.Series) -> dict:
    out = {"label": label}
    r = pd.Series(r).dropna()
    tags = ["FULL", "IS", "OOS", "Y2022", "Y2024", "Y2025"]
    if len(r) < 5:
        for t in tags:
            out[f"{t}_sharpe"] = 0.0
            out[f"{t}_ret"] = 0.0
            out[f"{t}_vol"] = 0.0
        return out
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        r.index = idx
    masks = {
        "FULL": np.ones(len(idx), dtype=bool),
        "IS": np.asarray(idx < SPLIT),
        "OOS": np.asarray(idx >= SPLIT),
        "Y2022": np.asarray((idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
                            (idx < pd.Timestamp("2023-01-01", tz="UTC"))),
        "Y2024": np.asarray((idx >= pd.Timestamp("2024-01-01", tz="UTC")) &
                            (idx < pd.Timestamp("2025-01-01", tz="UTC"))),
        "Y2025": np.asarray((idx >= pd.Timestamp("2025-01-01", tz="UTC")) &
                            (idx < pd.Timestamp("2026-01-01", tz="UTC"))),
    }
    for tag, mask in masks.items():
        sub = r[mask]
        if len(sub) < 5:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            out[f"{tag}_vol"] = 0.0
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
    return out


def scale_to_is_vol(rets: pd.Series, target: float = SUB_TARGET_VOL) -> float:
    is_r = rets[rets.index < SPLIT].dropna()
    if len(is_r) < 30:
        return 0.0
    bpy = _bpy(is_r.index)
    av = float(is_r.std(ddof=0) * np.sqrt(bpy))
    return target / av if av > 1e-9 else 0.0


def run_position(df: pd.DataFrame, pos, *, name: str, symbol: str, timeframe: str = "D1"):
    """Run backtest and return (daily_returns, raw_result)."""
    p = pd.Series(np.asarray(pos), index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, p, symbol=symbol, timeframe=timeframe, name=name)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    if rets.index.tz is None:
        rets.index = rets.index.tz_localize("UTC")
    return rets, res


# --- data load -------------------------------------------------------------

print("Loading FX D1 + H1 data...")
D1 = {s: get_candles(s, "D1") for s in FX_SYMS}
D1_SPX = get_candles("SPX500_USD", "D1")
H1 = {s: get_candles(s, "H1") for s in ["EUR_USD", "GBP_USD"]}

for s, df in D1.items():
    print(f"  D1 {s:8}  {len(df):5}  {df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")
print(f"  D1 SPX500   {len(D1_SPX):5}  {D1_SPX.timestamp.iloc[0].date()} -> {D1_SPX.timestamp.iloc[-1].date()}")
for s, df in H1.items():
    print(f"  H1 {s:8}  {len(df):6}  {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")


# ---------------------------------------------------------------------------
# Strategy 1 — USD_JPY carry mode (long when 252d slope > 0 AND vol < IS p75)
# ---------------------------------------------------------------------------

def strat_usdjpy_carry():
    print("\n=== Strategy 1: USD_JPY carry mode ===")
    sym = "USD_JPY"
    df = D1[sym].copy()
    close = df["close"].astype("float64")
    log_close = np.log(close)
    log_ret = log_close.diff()

    # 252d slope via OLS of log price on time
    win = 252
    x = np.arange(win, dtype="float64")
    x_dev = x - x.mean()
    sxx = (x_dev * x_dev).sum()
    slopes = np.full(len(df), np.nan)
    lc = log_close.values
    for t in range(win - 1, len(df)):
        y = lc[t - win + 1: t + 1]
        if np.isnan(y).any():
            continue
        y_dev = y - y.mean()
        slopes[t] = (x_dev * y_dev).sum() / sxx
    slope_s = pd.Series(slopes, index=df.index).shift(1)

    # 60d realized vol (ann)
    rv = log_ret.rolling(60, min_periods=60).std(ddof=0) * np.sqrt(BPY_D)
    rv = rv.shift(1)

    # IS p75 vol threshold (walk-forward: use IS-only value as the threshold)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    is_mask = ts < SPLIT
    p75 = float(rv[is_mask].quantile(0.75))
    print(f"  IS p75 realized vol = {p75:.3f}")

    # Long when slope > 0 AND rv < p75
    long_gate = (slope_s > 0) & (rv < p75)
    pos = pd.Series(np.where(long_gate.fillna(False), 1.0, 0.0), index=df.index)
    rets, _ = run_position(df, pos, name="USDJPY_CARRY", symbol=sym)
    s = stats("USDJPY_CARRY", rets)
    print(f"  exposure={float(long_gate.mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Strategy 2 — Synthetic EUR/GBP cross, 60d z-score mean-reversion
# ---------------------------------------------------------------------------

def strat_eurgbp_meanrev():
    """Trade synthetic EUR/GBP via EUR_USD long / GBP_USD short basket.

    Z-score of synthetic cross at 60-day rolling mean/std (walk-forward).
    Long EURGBP cross (long EUR_USD, short GBP_USD) when z < -1.5,
    short when z > +1.5. Exit when |z| < 0.5.

    Decompose into per-symbol positions and backtest each leg, then sum.
    """
    print("\n=== Strategy 2: EUR/GBP synthetic mean-reversion ===")
    eu = D1["EUR_USD"].copy()
    gb = D1["GBP_USD"].copy()
    # Align by timestamp
    eu_ts = pd.to_datetime(eu["timestamp"], utc=True)
    gb_ts = pd.to_datetime(gb["timestamp"], utc=True)
    eu_close = pd.Series(eu["close"].values, index=eu_ts)
    gb_close = pd.Series(gb["close"].values, index=gb_ts)
    common = eu_close.index.intersection(gb_close.index)
    eu_c = eu_close.loc[common]
    gb_c = gb_close.loc[common]
    cross = eu_c / gb_c

    win = 60
    mu = cross.rolling(win, min_periods=win).mean()
    sd = cross.rolling(win, min_periods=win).std(ddof=0)
    z = ((cross - mu) / sd).shift(1)

    long_entry = z < -1.5
    short_entry = z > 1.5
    flat_zone = z.abs() < 0.5

    # State machine: position based on bands
    state = pd.Series(0.0, index=common)
    cur = 0.0
    for i, t in enumerate(common):
        zv = z.iloc[i]
        if np.isnan(zv):
            state.iloc[i] = 0.0
            continue
        if cur == 0:
            if zv < -1.5:
                cur = 1.0
            elif zv > 1.5:
                cur = -1.0
        else:
            if abs(zv) < 0.5:
                cur = 0.0
        state.iloc[i] = cur

    # Build per-symbol positions: long-cross = long EUR, short GBP
    eu_pos = pd.Series(0.0, index=eu.index)
    gb_pos = pd.Series(0.0, index=gb.index)

    eu_pos_aligned = pd.Series(state.values, index=common)
    gb_pos_aligned = pd.Series(-state.values, index=common)

    # Map back to symbol df rows
    eu_pos_full = eu_pos_aligned.reindex(eu_ts).fillna(0.0)
    gb_pos_full = gb_pos_aligned.reindex(gb_ts).fillna(0.0)
    eu_pos = pd.Series(eu_pos_full.values, index=eu.index)
    gb_pos = pd.Series(gb_pos_full.values, index=gb.index)

    eu_rets, _ = run_position(eu, eu_pos, name="EURGBP_MR_EUR", symbol="EUR_USD")
    gb_rets, _ = run_position(gb, gb_pos, name="EURGBP_MR_GBP", symbol="GBP_USD")
    # Combine equal-weighted (single position vol-target later)
    combo = eu_rets.add(gb_rets, fill_value=0.0) * 0.5
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    s = stats("EURGBP_MR", combo)
    print(f"  state-changes={int((state.diff().abs() > 0).sum())}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return combo


# ---------------------------------------------------------------------------
# Strategy 3 — JPY-vs-everything basket, gated on SPX vol regime
# ---------------------------------------------------------------------------

def strat_jpy_basket():
    """Long USD_JPY + short EUR_USD + short GBP_USD (long-USD-vs-rest).

    On when SPX 60d realized vol is below its IS median (carry-on).
    """
    print("\n=== Strategy 3: JPY-basket gated by SPX vol regime ===")
    # SPX 60d vol
    spx = D1_SPX.copy()
    spx_close = spx["close"].astype("float64")
    spx_ret = spx_close.pct_change()
    rv = spx_ret.rolling(60, min_periods=60).std(ddof=0) * np.sqrt(252)
    rv = rv.shift(1)
    spx_ts = pd.to_datetime(spx["timestamp"], utc=True)
    is_mask = spx_ts < SPLIT
    spx_thr = float(rv[is_mask].quantile(0.50))
    print(f"  IS median SPX 60d vol = {spx_thr:.3f}")

    # Series of "on" flag by date
    spx_date = spx_ts.dt.normalize()
    on_by_date = pd.Series((rv < spx_thr).values, index=pd.DatetimeIndex(spx_date))
    on_by_date = on_by_date.groupby(on_by_date.index).last().fillna(False)

    streams = []
    for sym, sign in [("USD_JPY", +1.0), ("EUR_USD", -1.0), ("GBP_USD", -1.0)]:
        df = D1[sym].copy()
        df_ts = pd.to_datetime(df["timestamp"], utc=True)
        df_date = df_ts.dt.normalize()
        gate = pd.Series(on_by_date.reindex(pd.DatetimeIndex(df_date)).fillna(False).values,
                         index=df.index)
        pos = pd.Series(np.where(gate, sign, 0.0), index=df.index)
        rets, _ = run_position(df, pos, name=f"JPYB_{sym}", symbol=sym)
        streams.append(rets)
    combo = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1) / 3.0
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    s = stats("JPY_BASKET", combo)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return combo


# ---------------------------------------------------------------------------
# Strategy 4 — GBP_USD H1 60-day Bollinger band breakout
# ---------------------------------------------------------------------------

def strat_gbp_bollinger():
    """H1 60-day (60*24 ~ 1440 H1 bars) Bollinger band breakout on GBP_USD.

    Breakout: when close > upper-band → long; close < lower-band → short.
    Hold while inside the band; flip on opposite break.
    """
    print("\n=== Strategy 4: GBP_USD H1 60-day Bollinger breakout ===")
    df = H1["GBP_USD"].copy()
    close = df["close"].astype("float64")
    # 60 D1 bars at H1 ~ 60*24 = 1440 (but FX has ~24 H1 bars/day weekdays)
    win = 60 * 24
    mu = close.rolling(win, min_periods=win).mean()
    sd = close.rolling(win, min_periods=win).std(ddof=0)
    upper = (mu + 2.0 * sd).shift(1)
    lower = (mu - 2.0 * sd).shift(1)
    cs = close.shift(0)  # close at end of bar t

    # Breakout signal: when prior close > upper at bar t (so we go long at start of t+1).
    # Use shifted close to ensure no look-ahead at bar t open: we use close[t-1] vs upper[t-1].
    cs_prev = close.shift(1)
    long_break = cs_prev > upper
    short_break = cs_prev < lower

    state = np.zeros(len(df))
    cur = 0.0
    lb = long_break.fillna(False).values
    sb = short_break.fillna(False).values
    for i in range(len(df)):
        if lb[i]:
            cur = 1.0
        elif sb[i]:
            cur = -1.0
        state[i] = cur
    pos = pd.Series(state, index=df.index)
    rets, _ = run_position(df, pos, name="GBP_BOLL", symbol="GBP_USD", timeframe="H1")
    s = stats("GBP_BOLL", rets)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Strategy 5 — XAU_USD Donchian breakout (20d high / 10d low exit, long-only)
# ---------------------------------------------------------------------------

def strat_xau_donchian():
    print("\n=== Strategy 5: XAU Donchian 20d/10d (long-only) ===")
    df = D1["XAU_USD"].copy()
    close = df["close"].astype("float64")

    # Close-based Donchian: at start of bar t we compare close[t-1] to
    # max(close[t-21..t-2]) so the close being compared is NOT in the rolling max.
    prev_close = close.shift(1)
    hi20 = close.shift(2).rolling(20, min_periods=20).max()
    lo10 = close.shift(2).rolling(10, min_periods=10).min()

    state = np.zeros(len(df))
    cur = 0.0
    cp = prev_close.values
    hi = hi20.values
    lo = lo10.values
    for i in range(len(df)):
        if np.isnan(cp[i]) or np.isnan(hi[i]) or np.isnan(lo[i]):
            state[i] = 0.0
            continue
        if cur == 0:
            if cp[i] >= hi[i]:
                cur = 1.0
        else:
            if cp[i] <= lo[i]:
                cur = 0.0
        state[i] = cur
    pos = pd.Series(state, index=df.index)
    rets, _ = run_position(df, pos, name="XAU_DONCH", symbol="XAU_USD")
    s = stats("XAU_DONCH", rets)
    print(f"  exposure={pos.abs().mean():.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Strategy 6 — DXY-proxy TSMOM ensemble
# ---------------------------------------------------------------------------

def strat_dxy_tsmom():
    """Construct DXY proxy = -.58*EUR_USD -.12*GBP_USD +.14*USD_JPY (signed log-return basket).

    Apply 63d and 252d TSMOM. Long DXY when BOTH lookbacks are positive,
    short when both negative, flat otherwise.
    """
    print("\n=== Strategy 6: DXY-proxy TSMOM ensemble (63d AND 252d) ===")
    # Align EUR/GBP/JPY on common timestamp
    syms = ["EUR_USD", "GBP_USD", "USD_JPY"]
    closes = {}
    for s in syms:
        df = D1[s]
        ts = pd.to_datetime(df["timestamp"], utc=True)
        closes[s] = pd.Series(df["close"].astype("float64").values,
                              index=pd.DatetimeIndex(ts))
    panel = pd.concat(closes, axis=1).dropna()
    log_rets = np.log(panel).diff()

    # DXY proxy log-return = -.58 * EUR - .12 * GBP + .14 * USDJPY
    dxy_ret = -0.58 * log_rets["EUR_USD"] - 0.12 * log_rets["GBP_USD"] + 0.14 * log_rets["USD_JPY"]
    dxy_idx = (1.0 + dxy_ret.fillna(0.0)).cumprod()
    log_dxy = np.log(dxy_idx)

    mom_63 = (log_dxy - log_dxy.shift(63)).shift(1)
    mom_252 = (log_dxy - log_dxy.shift(252)).shift(1)
    long_g = (mom_63 > 0) & (mom_252 > 0)
    short_g = (mom_63 < 0) & (mom_252 < 0)
    direction = pd.Series(0.0, index=panel.index)
    direction[long_g.fillna(False)] = 1.0
    direction[short_g.fillna(False)] = -1.0

    # Express position via each underlying with its DXY weight (sign)
    weights = {"EUR_USD": -0.58, "GBP_USD": -0.12, "USD_JPY": +0.14}
    # Normalize so weights sum-abs to ~1 in the basket; here sum-abs = 0.84
    streams = []
    for s, w in weights.items():
        df = D1[s].copy()
        df_ts = pd.to_datetime(df["timestamp"], utc=True)
        dir_aligned = direction.reindex(pd.DatetimeIndex(df_ts)).fillna(0.0)
        pos = pd.Series(w * dir_aligned.values, index=df.index)
        rets, _ = run_position(df, pos, name=f"DXY_TSMOM_{s}", symbol=s)
        streams.append(rets)
    combo = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    s = stats("DXY_TSMOM", combo)
    print(f"  long-frac={float(long_g.mean()):.2%}  short-frac={float(short_g.mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return combo


# ---------------------------------------------------------------------------
# Strategy 7 — FX risk-off coordinator (SPX 5d < -3% AND USD_JPY rising → long DXY)
# ---------------------------------------------------------------------------

def strat_risk_off_dxy():
    print("\n=== Strategy 7: FX risk-off DXY follow ===")
    # SPX 5d return
    spx = D1_SPX.copy()
    spx_close = spx["close"].astype("float64")
    spx_5d = spx_close.pct_change(5).shift(1)
    spx_ts = pd.to_datetime(spx["timestamp"], utc=True)
    spx_date = spx_ts.dt.normalize()
    spx_5d_by_date = pd.Series(spx_5d.values, index=pd.DatetimeIndex(spx_date))
    spx_5d_by_date = spx_5d_by_date.groupby(spx_5d_by_date.index).last()

    # USD_JPY 5d return rising
    jpy = D1["USD_JPY"].copy()
    jpy_close = jpy["close"].astype("float64")
    jpy_5d = jpy_close.pct_change(5).shift(1)
    jpy_ts = pd.to_datetime(jpy["timestamp"], utc=True)
    jpy_date = jpy_ts.dt.normalize()
    jpy_5d_by_date = pd.Series(jpy_5d.values, index=pd.DatetimeIndex(jpy_date))
    jpy_5d_by_date = jpy_5d_by_date.groupby(jpy_5d_by_date.index).last()

    # Risk-off trigger by date
    trig_dates = (spx_5d_by_date < -0.03) & (jpy_5d_by_date > 0)
    trig_dates = trig_dates.fillna(False)
    print(f"  trigger days: {int(trig_dates.sum())}")

    # When triggered, hold position for next 3 bars (3-day holding window).
    # Position is "long DXY" — expressed as -EUR, -GBP, +JPY (signed weights).
    weights = {"EUR_USD": -0.58, "GBP_USD": -0.12, "USD_JPY": +0.14}
    trig_idx = trig_dates[trig_dates].index

    streams = []
    for sym, w in weights.items():
        df = D1[sym].copy()
        df_ts = pd.to_datetime(df["timestamp"], utc=True)
        df_date = df_ts.dt.normalize()
        df_date_idx = pd.DatetimeIndex(df_date)

        pos = pd.Series(0.0, index=df.index)
        # For each trigger date, fire t+1 through t+3 (3 bars)
        # Build a map: date -> position weight
        target_weights = pd.Series(0.0, index=df_date_idx)
        for td in trig_idx:
            # Hold for 3 bars after td (date of trigger inferred from data shift)
            for h in [1, 2, 3]:
                td_h = td + pd.Timedelta(days=h)
                # advance by trading days; use search
                mask = df_date_idx == td_h
                if mask.any():
                    target_weights.loc[mask] = w
        pos = pd.Series(target_weights.values, index=df.index)
        rets, _ = run_position(df, pos, name=f"RISKOFF_{sym}", symbol=sym)
        streams.append(rets)
    combo = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    s = stats("RISK_OFF_DXY", combo)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return combo


# ---------------------------------------------------------------------------
# Strategy 8 — Asian-session FX momentum (EUR/GBP H1, 00-06 UTC)
# ---------------------------------------------------------------------------

def strat_asian_fx_momentum():
    """For EUR_USD and GBP_USD H1:
    Compute the cumulative log-return over the Asian session (00:00-06:00 UTC)
    of each calendar day. If |Asia_ret| > 0.5 SD of trailing Asian sessions,
    follow direction for the next H1 bar (07:00 UTC).
    """
    print("\n=== Strategy 8: Asian-session FX momentum (EUR/GBP) ===")
    streams = []
    for sym in ["EUR_USD", "GBP_USD"]:
        df = H1[sym].copy()
        ts = pd.to_datetime(df["timestamp"], utc=True)
        hour = ts.dt.hour
        date = ts.dt.normalize()
        close = df["close"].astype("float64")
        log_close = np.log(close)
        log_ret = log_close.diff()

        # Asia mask: hour in [0, 5] inclusive (00-05 = 6 bars, 00:00-05:59)
        asia_mask = hour.isin([0, 1, 2, 3, 4, 5])
        # Sum of log returns within Asia session per date
        asia_ret_per_bar = log_ret.where(asia_mask, 0.0)
        asia_cum = asia_ret_per_bar.groupby(pd.DatetimeIndex(date)).cumsum()

        # End-of-Asia: last bar where hour == 5 of each date
        is_asia_end = (hour == 5)
        # asia_total per date = log_ret sum 00:00-05:59
        asia_total_by_date = (log_ret.where(asia_mask)
                              .groupby(pd.DatetimeIndex(date))
                              .sum())

        # Walk-forward SD of trailing N=60 Asia sessions
        asia_sd = asia_total_by_date.shift(1).rolling(60, min_periods=30).std(ddof=0)
        # Threshold per date
        big_mask = (asia_total_by_date.shift(1).abs() > 0.5 * asia_sd)
        # NOTE: this is "yesterday's Asia" — we need *today's* Asia ending at 06:00 to
        # trade the 07:00 bar. So we use asia_total_by_date itself (unshifted),
        # but the threshold SD must be walk-forward (shifted).
        asia_sd_today = asia_total_by_date.rolling(60, min_periods=30).std(ddof=0).shift(1)
        big_today = asia_total_by_date.abs() > 0.5 * asia_sd_today
        direction_by_date = np.sign(asia_total_by_date) * big_today.astype(float)

        # Position at H1 bar with hour == 6 (07:00 UTC = first post-Asia bar);
        # at hour 6 we observe close[5] which already includes 00:00-05:59 close.
        # Strictly: at start of hour-6 bar, hour-5 bar is closed → asia sum is known.
        df_pos = pd.Series(0.0, index=df.index)
        # Map each bar's date (UTC, tz-aware) to a direction value.
        # direction_by_date.index is tz-aware UTC; convert date_arr to match.
        date_arr = pd.DatetimeIndex(date.values)
        if date_arr.tz is None:
            date_arr = date_arr.tz_localize("UTC")
        dir_idx_unique = direction_by_date.copy()
        if dir_idx_unique.index.tz is None:
            dir_idx_unique.index = dir_idx_unique.index.tz_localize("UTC")
        # Build per-bar via map dict
        dir_map = {pd.Timestamp(k): float(v) for k, v in dir_idx_unique.items()}
        dir_per_bar = np.array([dir_map.get(d, 0.0) for d in date_arr],
                               dtype="float64")
        is_post_asia = (hour == 6).values
        df_pos.loc[is_post_asia] = dir_per_bar[is_post_asia]

        rets, _ = run_position(df, df_pos, name=f"ASIA_{sym}", symbol=sym, timeframe="H1")
        streams.append(rets)
    combo = pd.concat(streams, axis=1, sort=True).fillna(0.0).mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    s = stats("ASIAN_FX_MOM", combo)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return combo


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    raw = {}
    raw["USDJPY_CARRY"]   = strat_usdjpy_carry()
    raw["EURGBP_MEANREV"] = strat_eurgbp_meanrev()
    raw["JPY_BASKET"]     = strat_jpy_basket()
    raw["GBP_BOLLINGER"]  = strat_gbp_bollinger()
    raw["XAU_DONCHIAN"]   = strat_xau_donchian()
    raw["DXY_TSMOM"]      = strat_dxy_tsmom()
    raw["RISK_OFF_DXY"]   = strat_risk_off_dxy()
    raw["ASIAN_FX_MOM"]   = strat_asian_fx_momentum()

    scaled = {}
    breakdown_rows = []
    for name, r in raw.items():
        scale = scale_to_is_vol(r, SUB_TARGET_VOL)
        sc = r * scale
        scaled[name] = sc
        s = stats(name, sc)
        s["scale"] = scale
        s["sleeve"] = name
        breakdown_rows.append(s)
        print(f"  scale[{name:<16}] = {scale:.3f}")

    breakdown = pd.DataFrame(breakdown_rows)
    cols = ["sleeve"] + [c for c in breakdown.columns if c not in ("sleeve", "label")]
    breakdown = breakdown[cols]

    # Survivor filter: IS Sharpe >= 0.5 AND OOS >= 0
    breakdown["survived"] = ((breakdown["IS_sharpe"] >= 0.5) &
                             (breakdown["OOS_sharpe"] >= 0.0))
    breakdown.to_csv(OUT / "fx_specific_breakdown.csv", index=False)

    print("\n=== Breakdown (vol-scaled to 5% IS ann vol) ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "Y2024_sharpe", "Y2025_sharpe",
                     "FULL_sharpe", "FULL_ret", "scale", "survived"]]
          .to_string(index=False))

    survivors = breakdown[breakdown["survived"]]["sleeve"].tolist()
    print(f"\nSurvivors ({len(survivors)} / {len(breakdown)}): {survivors}")

    # Save panel (all sleeves + survivors_mean)
    panel = pd.concat(scaled, axis=1, sort=True).fillna(0.0)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    if survivors:
        panel["survivors_mean"] = panel[survivors].mean(axis=1)
    else:
        panel["survivors_mean"] = 0.0

    out_df = panel.reset_index().rename(columns={"index": "timestamp"})
    out_df.to_parquet(OUT / "fx_specific_returns.parquet", index=False)

    # Combined sleeve stats
    combined = panel["survivors_mean"]
    sc = stats("FX_SPECIFIC_COMBINED", combined)
    print("\n=== Combined survivor sleeve (equal-weight) ===")
    for tag in ["FULL", "IS", "OOS", "Y2022", "Y2024", "Y2025"]:
        sh = sc.get(f"{tag}_sharpe", 0)
        rt = sc.get(f"{tag}_ret", 0)
        vv = sc.get(f"{tag}_vol", 0)
        print(f"  {tag:<6}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  AnnVol={vv:.2%}")

    # Yearly
    print("\n=== Combined yearly Sharpe ===")
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 20:
            continue
        bpy = _bpy(sub.index)
        sd = sub.std(ddof=0)
        sh = (sub.mean() * bpy) / (sd * np.sqrt(bpy)) if sd > 0 else 0.0
        rt = sub.mean() * bpy
        print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")

    print(f"\nWrote {OUT/'fx_specific_returns.parquet'}")
    print(f"Wrote {OUT/'fx_specific_breakdown.csv'}")


if __name__ == "__main__":
    main()
