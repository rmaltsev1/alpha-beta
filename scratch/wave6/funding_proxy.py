"""Crypto funding-rate proxies — wave 6.

We have no perpetual-funding data, so we *construct proxies from spot* that
correlate with funding-rate behaviour, then trade their extremes for
mean-reversion.  Theory: persistently positive funding => crowded longs =>
unwind => short opportunity (and mirror).

Six sub-sleeves (D1 unless noted):
  1. MOM_DIVERGENCE  D1   — leader vs. laggard cross-coin pair trades when
                            5d returns diverge sharply (BTC vs ETH, ETH vs SOL,
                            BTC vs SOL).
  2. VOL_ADJ_FUND    D1   — funding_proxy = 5d_ret / 5d_vol; fade when |z| > 2.
  3. CROSS_FUND_SPRD D1   — (BTC vol-adj) - (ETH vol-adj) spread; trade
                            convergence between BTC and ETH.
  4. WKND_FUND_SPIKE H1   — short BTC at Fri 20:00 UTC if Wed->Fri gain > 5%;
                            hold 48h.  Mirror long if 48h dump > 5%.
  5. VOL_OI_PROXY    D1   — when 24h volume in top 5% AND price at 90d-high
                            => short for 5 days (overleveraged-long flush).
                            Mirror at 90d-low + vol top 5% => long 5d.
  6. EIGHTHOUR_CYCLE H1   — funding settles 00/08/16 UTC.  Test cyclic
                            pre/post 8h-window directional drift on H1 returns,
                            IS-best per-coin sign + hour.

Methodology
-----------
- IS: timestamp < 2024-01-01.  OOS: >= 2024-01-01.
- All thresholds picked from IS only.  Per sub-sleeve & per symbol where
  applicable.  Sign-of-trade picked IS-best from {+1, -1}.
- Each sub-sleeve combined (mean across coins/legs), then vol-scaled to
  5% IS ann vol.
- Survivor filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/funding_proxy.py                 (this file)
  scratch/wave6/funding_proxy_returns.parquet    (per-sleeve daily streams)
  scratch/wave6/funding_proxy_breakdown.csv      (per-sleeve summary stats)
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

CRYPTO_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# --- helpers ---------------------------------------------------------------

def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    r = pd.Series(r).dropna()
    tags = ["FULL", "IS", "OOS", "Y2022", "Y2024_25"]
    if len(r) < 5:
        for tag in tags:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            out[f"{tag}_vol"] = 0.0
        return out
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        r.index = idx
    windows = [
        ("FULL", np.ones(len(idx), dtype=bool)),
        ("IS", np.asarray(idx < SPLIT)),
        ("OOS", np.asarray(idx >= SPLIT)),
        ("Y2022", np.asarray(
            (idx >= pd.Timestamp("2022-01-01", tz="UTC"))
            & (idx < pd.Timestamp("2023-01-01", tz="UTC"))
        )),
        ("Y2024_25", np.asarray(idx >= pd.Timestamp("2024-01-01", tz="UTC"))),
    ]
    for tag, mask in windows:
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


def run_position(df, pos, *, name, symbol, timeframe="D1"):
    p = pd.Series(np.asarray(pos), index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, p, symbol=symbol, timeframe=timeframe, name=name)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    # Aggregate to D1 for cross-strategy comparability (H1 sleeves too).
    if timeframe in ("M15", "M5", "M1", "H1", "H4"):
        ts = ts.dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    if rets.index.tz is None:
        rets.index = rets.index.tz_localize("UTC")
    return rets, res


# --- data load -------------------------------------------------------------

print("Loading data...")
DATA_D1 = {s: get_candles(s, "D1") for s in CRYPTO_SYMS}
DATA_H1 = {s: get_candles(s, "H1") for s in CRYPTO_SYMS}
for s in CRYPTO_SYMS:
    df = DATA_D1[s]
    print(f"  D1 {s:10} {len(df):5}  "
          f"{df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


def _align_d1(symbol: str) -> pd.DataFrame:
    df = DATA_D1[symbol].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ===========================================================================
# Strategy 1 — Momentum divergence (leader vs. laggard)
# ===========================================================================

def strat_mom_divergence():
    """For each ordered pair (A, B) of coins:
      Compute 5d log returns rA, rB.  Form spread d = rA - rB.
      When |d| > IS-fitted threshold (top 10% of |d| on IS), trade convergence:
        short the leader (sign +d -> short A, long B) for next 3 days.

    Combine BTC-ETH, ETH-SOL, BTC-SOL (each leg = -leader + laggard).
    Per-pair IS-best sign picked from {+1, -1} (data decides if convergence or
    continuation).  Run as long/short cross-coin spread; entry costs are paid
    per leg, per side.
    """
    print("\n=== Strategy 1: Momentum divergence (pair convergence) ===")
    streams = {}
    detail = []
    pairs = [("BTCUSDT", "ETHUSDT"),
             ("ETHUSDT", "SOLUSDT"),
             ("BTCUSDT", "SOLUSDT")]
    horizon = 3

    # Align to common D1 index (intersection).
    ts_common = None
    closes = {}
    for s in CRYPTO_SYMS:
        d = _align_d1(s)
        ts = pd.to_datetime(d["timestamp"], utc=True)
        c = pd.Series(d["close"].astype("float64").values, index=ts)
        closes[s] = c
        ts_common = c.index if ts_common is None else ts_common.intersection(c.index)

    panel = pd.concat({k: closes[k].reindex(ts_common) for k in closes}, axis=1)
    log_ret_5 = np.log(panel / panel.shift(5))

    for A, B in pairs:
        spread = (log_ret_5[A] - log_ret_5[B]).dropna()
        # IS-only threshold: top 10% of |spread| in absolute.
        is_mask = spread.index < SPLIT
        thr = float(spread[is_mask].abs().quantile(0.90))
        # signal at end of day t: pos for bars t+1..t+horizon
        sig = pd.Series(0.0, index=spread.index)
        sig[spread > thr] = -1.0   # A is the leader => short A, long B
        sig[spread < -thr] = +1.0  # A is laggard

        # Propagate to horizon: position at t = max-magnitude latest signal
        # within last `horizon` bars (signal known at end of t-1).
        sig_lead = sig.shift(1).fillna(0.0)
        held = pd.Series(0.0, index=spread.index)
        cur = 0.0
        days_left = 0
        for ts, s_ in sig_lead.items():
            if s_ != 0.0:
                cur = s_
                days_left = horizon
            if days_left > 0:
                held.loc[ts] = cur
                days_left -= 1

        # We run as two single-symbol backtests so transaction costs apply
        # correctly per leg.
        # Position for A = held (sign means: short leader A scenario).
        # Sign convention: +1 means long-A and short-B convergence trade
        # (i.e., A is the laggard).  Reverse for B.
        pos_A = held.reindex(panel.index, fill_value=0.0)
        pos_B = -held.reindex(panel.index, fill_value=0.0)

        # Build dfs aligned to ts_common for each leg
        df_A = _align_d1(A).copy()
        df_A = df_A[df_A["timestamp"].isin(pos_A.index)].reset_index(drop=True)
        df_B = _align_d1(B).copy()
        df_B = df_B[df_B["timestamp"].isin(pos_B.index)].reset_index(drop=True)
        pA = pd.Series(pos_A.reindex(pd.DatetimeIndex(df_A["timestamp"])).values,
                       index=df_A.index)
        pB = pd.Series(pos_B.reindex(pd.DatetimeIndex(df_B["timestamp"])).values,
                       index=df_B.index)

        n_events = int((sig != 0).sum())

        best = None
        for sgn in [+1, -1]:
            rA, _ = run_position(df_A, sgn * pA, name=f"MD_{A}_{B}_A_{sgn}",
                                 symbol=A)
            rB, _ = run_position(df_B, sgn * pB, name=f"MD_{A}_{B}_B_{sgn}",
                                 symbol=B)
            combo = pd.concat([rA, rB], axis=1).fillna(0.0).sum(axis=1) * 0.5
            s = stats(f"MD_{A}_{B}_{sgn}", combo)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, combo)
        sgn, s, combo = best
        key = f"{A[:3]}_{B[:3]}"
        streams[key] = combo
        detail.append({
            "pair": key, "sign": sgn, "threshold": thr, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {key:<10} sgn={sgn:+d}  thr={thr:.3f}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["pair"].tolist()
    if surv:
        panel_s = pd.concat({k: streams[k] for k in surv}, axis=1).fillna(0.0)
        combo = panel_s.mean(axis=1)
    else:
        panel_s = pd.concat(streams, axis=1).fillna(0.0)
        combo = panel_s.mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ===========================================================================
# Strategy 2 — Vol-adjusted return as funding proxy (per-coin mean-revert)
# ===========================================================================

def strat_vol_adj_funding():
    """funding_proxy(t) = 5d_log_return(t) / 5d_realized_vol(t).
    When |proxy| > 2 (IS-quantile-calibrated), fade for 3 days:
      proxy > +threshold => short for 3 days (longs overheated)
      proxy < -threshold => long for 3 days (shorts overpaid)
    Per-coin IS-best sign and threshold.
    """
    print("\n=== Strategy 2: Vol-adjusted funding proxy (per-coin fade) ===")
    streams = {}
    detail = []
    horizon = 3
    for sym in CRYPTO_SYMS:
        df = _align_d1(sym)
        c = df["close"].astype("float64")
        lr = np.log(c / c.shift(1)).fillna(0.0)
        r5 = np.log(c / c.shift(5))
        vol5 = lr.rolling(5, min_periods=5).std()
        proxy = (r5 / (vol5 * np.sqrt(5))).fillna(0.0)

        # IS threshold from quantile (cap at min 1.5 to keep meaningful)
        is_mask = pd.to_datetime(df["timestamp"], utc=True) < SPLIT
        thr = float(np.nanquantile(proxy[is_mask].abs().values, 0.90))
        thr = max(thr, 1.5)

        sig = pd.Series(0.0, index=df.index)
        sig[proxy > thr] = -1.0
        sig[proxy < -thr] = +1.0
        # propagate to horizon
        sig_lead = sig.shift(1).fillna(0.0)
        held = np.zeros(len(df))
        cur = 0.0
        days_left = 0
        for i, s_ in enumerate(sig_lead.values):
            if s_ != 0.0:
                cur = s_
                days_left = horizon
            if days_left > 0:
                held[i] = cur
                days_left -= 1
        pos_base = pd.Series(held, index=df.index)
        n_events = int((sig != 0).sum())

        best = None
        for sgn in [+1, -1]:
            rets, _ = run_position(df, sgn * pos_base,
                                   name=f"VAF_{sym}_{sgn}", symbol=sym)
            s = stats(f"VAF_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "threshold": thr, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<10} sgn={sgn:+d}  thr={thr:.2f}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    if surv:
        combo = pd.concat({k: streams[k] for k in surv}, axis=1).fillna(0.0).mean(axis=1)
    else:
        combo = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ===========================================================================
# Strategy 3 — Cross-coin funding spread (BTC vs ETH vol-adj convergence)
# ===========================================================================

def strat_cross_fund_spread():
    """funding_proxy_X = 5d_ret_X / (5d_vol_X * sqrt(5)).
    spread = proxy_BTC - proxy_ETH.
    When |spread| > IS 90% quantile, trade convergence:
      spread > thr => short BTC, long ETH for 3 days.
    Run as two single-symbol legs.
    """
    print("\n=== Strategy 3: Cross-coin funding spread (BTC vs ETH) ===")
    detail = []
    horizon = 3

    def _proxy(sym):
        df = _align_d1(sym)
        c = df["close"].astype("float64")
        lr = np.log(c / c.shift(1)).fillna(0.0)
        r5 = np.log(c / c.shift(5))
        vol5 = lr.rolling(5, min_periods=5).std()
        proxy = (r5 / (vol5 * np.sqrt(5))).fillna(0.0)
        proxy.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        return proxy

    pBTC = _proxy("BTCUSDT")
    pETH = _proxy("ETHUSDT")
    common = pBTC.index.intersection(pETH.index)
    spread = (pBTC.reindex(common) - pETH.reindex(common)).dropna()

    is_mask = spread.index < SPLIT
    thr = float(spread[is_mask].abs().quantile(0.90))

    sig = pd.Series(0.0, index=spread.index)
    sig[spread > thr] = -1.0   # short BTC, long ETH
    sig[spread < -thr] = +1.0  # long BTC, short ETH

    sig_lead = sig.shift(1).fillna(0.0)
    held = pd.Series(0.0, index=spread.index)
    cur = 0.0
    days_left = 0
    for ts, s_ in sig_lead.items():
        if s_ != 0.0:
            cur = s_
            days_left = horizon
        if days_left > 0:
            held.loc[ts] = cur
            days_left -= 1

    df_B = _align_d1("BTCUSDT")
    df_E = _align_d1("ETHUSDT")
    df_B = df_B[df_B["timestamp"].isin(spread.index)].reset_index(drop=True)
    df_E = df_E[df_E["timestamp"].isin(spread.index)].reset_index(drop=True)
    held_B = held.reindex(pd.DatetimeIndex(df_B["timestamp"])).fillna(0.0).values
    held_E = -held.reindex(pd.DatetimeIndex(df_E["timestamp"])).fillna(0.0).values
    pB = pd.Series(held_B, index=df_B.index)
    pE = pd.Series(held_E, index=df_E.index)
    n_events = int((sig != 0).sum())

    best = None
    for sgn in [+1, -1]:
        rB, _ = run_position(df_B, sgn * pB, name=f"CFS_B_{sgn}", symbol="BTCUSDT")
        rE, _ = run_position(df_E, sgn * pE, name=f"CFS_E_{sgn}", symbol="ETHUSDT")
        combo = pd.concat([rB, rE], axis=1).fillna(0.0).sum(axis=1) * 0.5
        s = stats(f"CFS_{sgn}", combo)
        if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
            best = (sgn, s, combo)
    sgn, s, combo = best
    detail.append({
        "label": "CFS_BTC_ETH", "sign": sgn, "threshold": thr,
        "n_events": n_events,
        "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
        "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
    })
    print(f"  CFS_BTC_ETH sgn={sgn:+d}  thr={thr:.2f}  n={n_events:3d}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")

    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    return combo, pd.DataFrame(detail)


# ===========================================================================
# Strategy 4 — Weekend funding spike (Friday 20:00 UTC short if Wed-Fri up)
# ===========================================================================

def strat_weekend_spike():
    """At Friday 20:00 UTC, if BTC has gained >5% from Wed 00:00 UTC to now,
    short for 48 hours.  Mirror: if it dropped >5%, long for 48 hours.
    Run on H1 timeframe.  Per-coin IS-best sign.
    Threshold floored at 5% (5pct gain), but tuned IS-best per coin from
    {3, 4, 5, 7}%.
    """
    print("\n=== Strategy 4: Weekend funding spike (H1, Fri 20:00 UTC) ===")
    streams = {}
    detail = []
    holding_hours = 48
    candidate_thr = [0.03, 0.04, 0.05, 0.07]
    for sym in CRYPTO_SYMS:
        df = DATA_H1[sym].copy()
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = ts
        c = df["close"].astype("float64")
        # For each row find close at *previous* Wed 00:00 UTC.
        # Strategy: only triggers at Fri 20:00 UTC anyway.
        weekday = ts.dt.weekday
        hour = ts.dt.hour
        is_fri_20 = (weekday == 4) & (hour == 20)

        # Build a "wed_close_of_this_week" series: forward-fill close at
        # Wed 00:00 UTC across the week.
        wed_close = c.where((weekday == 2) & (hour == 0)).ffill()

        gain = (c / wed_close - 1.0).fillna(0.0)
        n = len(df)

        best = None
        for thr in candidate_thr:
            up_trig = is_fri_20.values & (gain.values > thr)
            down_trig = is_fri_20.values & (gain.values < -thr)
            pos_base = np.zeros(n, dtype="float64")
            for i in np.where(up_trig)[0]:
                for h in range(1, holding_hours + 1):
                    if i + h < n:
                        pos_base[i + h] = -1.0
            for i in np.where(down_trig)[0]:
                for h in range(1, holding_hours + 1):
                    if i + h < n:
                        pos_base[i + h] = +1.0
            n_events = int(up_trig.sum() + down_trig.sum())
            for sgn in [+1, -1]:
                pos = sgn * pos_base
                rets, _ = run_position(df, pos, name=f"WSP_{sym}_{thr}_{sgn}",
                                       symbol=sym, timeframe="H1")
                s = stats(f"WSP_{sym}_{thr}_{sgn}", rets)
                key = (thr, sgn, n_events, s, rets)
                if best is None or s["IS_sharpe"] > best[3]["IS_sharpe"]:
                    best = key

        thr, sgn, n_events, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "threshold": thr, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<10} sgn={sgn:+d}  thr={thr:.2f}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    if surv:
        combo = pd.concat({k: streams[k] for k in surv}, axis=1).fillna(0.0).mean(axis=1)
    else:
        combo = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ===========================================================================
# Strategy 5 — Open-interest proxy via volume + multi-month-high (D1)
# ===========================================================================

def strat_vol_oi_proxy():
    """At 24h, when D1 volume in IS top 5% AND price within 1% of 90d-high,
    short for 5 days (overleveraged-long flush).  Mirror at 90d-low + top-5%
    vol => long for 5 days.  Per-coin IS-best sign.
    """
    print("\n=== Strategy 5: Volume/OI proxy (top vol + extreme) ===")
    streams = {}
    detail = []
    horizon = 5
    for sym in CRYPTO_SYMS:
        df = _align_d1(sym)
        c = df["close"].astype("float64")
        vol = df["volume"].astype("float64").fillna(0.0)
        high_90 = c.rolling(90, min_periods=20).max()
        low_90 = c.rolling(90, min_periods=20).min()
        # IS-only vol threshold (top 5%)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        is_mask = ts < SPLIT
        vol_thr = float(np.nanquantile(vol[is_mask].values, 0.95))
        # also use rolling vs 30d mean as a robustness factor
        vol_30 = vol.rolling(30, min_periods=10).mean().shift(1)

        near_high = (c >= 0.99 * high_90).fillna(False)
        near_low = (c <= 1.01 * low_90).fillna(False)
        high_vol = ((vol > vol_thr) | (vol > 2.5 * vol_30)).fillna(False)

        top_event = (high_vol & near_high).values
        bot_event = (high_vol & near_low).values
        n = len(df)
        pos_base = np.zeros(n, dtype="float64")
        for i in np.where(top_event)[0]:
            for h in range(1, horizon + 1):
                if i + h < n:
                    pos_base[i + h] = -1.0
        for i in np.where(bot_event)[0]:
            for h in range(1, horizon + 1):
                if i + h < n:
                    pos_base[i + h] = +1.0
        n_events = int(top_event.sum() + bot_event.sum())

        best = None
        for sgn in [+1, -1]:
            rets, _ = run_position(df, sgn * pos_base,
                                   name=f"OIV_{sym}_{sgn}", symbol=sym)
            s = stats(f"OIV_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "vol_thr": vol_thr, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<10} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    if surv:
        combo = pd.concat({k: streams[k] for k in surv}, axis=1).fillna(0.0).mean(axis=1)
    else:
        combo = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ===========================================================================
# Strategy 6 — 8h funding-settlement cycle (H1 pattern hunt)
# ===========================================================================

def strat_eight_hour_cycle():
    """Real funding settles at 00, 08, 16 UTC.  Test H1 mean return at each
    UTC hour on IS; for top |mean-t-stat| hour per coin, take direction from
    IS and trade that hour always (active only that hour).  IS-best per coin.
    """
    print("\n=== Strategy 6: 8h cycle (per-hour H1 patterns) ===")
    streams = {}
    detail = []
    for sym in CRYPTO_SYMS:
        df = DATA_H1[sym].copy()
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = ts
        c = df["close"].astype("float64")
        lr = np.log(c / c.shift(1)).fillna(0.0)
        hours = ts.dt.hour

        is_mask = (ts < SPLIT).values
        # Compute t-stat per hour on IS
        best_hour = None
        best_t = 0.0
        best_sign = 0
        for h in range(24):
            sel = (hours.values == h) & is_mask
            if sel.sum() < 30:
                continue
            x = lr.values[sel]
            mu = x.mean()
            sd = x.std(ddof=0)
            t_stat = mu / (sd / np.sqrt(len(x))) if sd > 0 else 0.0
            if abs(t_stat) > abs(best_t):
                best_t = t_stat
                best_hour = h
                best_sign = int(np.sign(t_stat))

        # Build position: active only at the chosen hour, in best-sign direction.
        n = len(df)
        pos = np.zeros(n, dtype="float64")
        if best_hour is not None:
            mask = (hours.values == best_hour)
            pos[mask] = float(best_sign)

        rets, _ = run_position(df, pd.Series(pos, index=df.index),
                               name=f"H8_{sym}", symbol=sym, timeframe="H1")
        s = stats(f"H8_{sym}", rets)
        streams[sym] = rets
        n_events = int((pos != 0).sum())
        detail.append({
            "symbol": sym, "best_hour": best_hour, "sign": best_sign,
            "is_t_stat": float(best_t), "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<10} hr={best_hour}  sgn={best_sign:+d}  t={best_t:+.2f}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    if surv:
        combo = pd.concat({k: streams[k] for k in surv}, axis=1).fillna(0.0).mean(axis=1)
    else:
        combo = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Cross-check against existing sleeves (WED_BTC, CRYPTO_DOM)
# ---------------------------------------------------------------------------

def cross_correlations(combined):
    out = []
    panel_path = ROOT / "scratch" / "quant" / "all_sleeve_returns_v15.parquet"
    if not panel_path.exists():
        return pd.DataFrame()
    panel = pd.read_parquet(panel_path)
    # Standardize tz
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")

    targets = ["WED_BTC", "WED_ETH", "WED_SOL", "CRYPTO_DOM",
               "CRYPTO_vs_SPX", "TSMOM", "XSMOM"]
    for t in targets:
        if t not in panel.columns:
            continue
        joint = pd.concat([combined.rename("us"),
                           panel[t].rename(t)], axis=1).dropna()
        if len(joint) < 60:
            continue
        c_full = float(joint.corr().iloc[0, 1])
        is_m = joint.index < SPLIT
        oos_m = joint.index >= SPLIT
        c_is = float(joint[is_m].corr().iloc[0, 1]) if is_m.sum() > 30 else float("nan")
        c_oos = float(joint[oos_m].corr().iloc[0, 1]) if oos_m.sum() > 30 else float("nan")
        out.append({"sleeve": t, "corr_full": c_full,
                    "corr_is": c_is, "corr_oos": c_oos})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_sleeves = {}
    details = {}

    raw_sleeves["MOM_DIVERGENCE"], details["MOM_DIVERGENCE"] = strat_mom_divergence()
    raw_sleeves["VOL_ADJ_FUND"], details["VOL_ADJ_FUND"] = strat_vol_adj_funding()
    raw_sleeves["CROSS_FUND_SPRD"], details["CROSS_FUND_SPRD"] = strat_cross_fund_spread()
    raw_sleeves["WKND_FUND_SPIKE"], details["WKND_FUND_SPIKE"] = strat_weekend_spike()
    raw_sleeves["VOL_OI_PROXY"], details["VOL_OI_PROXY"] = strat_vol_oi_proxy()
    raw_sleeves["EIGHTHOUR_CYCLE"], details["EIGHTHOUR_CYCLE"] = strat_eight_hour_cycle()

    # Vol-scale each sub-sleeve to 5% IS ann vol
    scaled = {}
    breakdown_rows = []
    print("\n=== Sub-sleeve scaling ===")
    for name, r in raw_sleeves.items():
        scale = scale_to_is_vol(r, SUB_TARGET_VOL)
        sc = r * scale
        scaled[name] = sc
        s = stats(name, sc)
        s["scale"] = scale
        s["sleeve"] = name
        breakdown_rows.append(s)
        print(f"  scale[{name:<20}] = {scale:.3f}")

    breakdown = pd.DataFrame(breakdown_rows)
    cols = ["sleeve"] + [c for c in breakdown.columns
                        if c not in ("sleeve", "label")]
    breakdown = breakdown[cols]
    breakdown.to_csv(OUT / "funding_proxy_breakdown.csv", index=False)

    print("\n=== Breakdown ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "Y2024_25_sharpe", "FULL_sharpe",
                     "FULL_ret", "FULL_vol", "scale"]].to_string(index=False))

    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.4)
                          & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) "
          f"[IS Sharpe >= 0.4 AND OOS Sharpe >= 0] ===")
    if not survivors.empty:
        print(survivors[["sleeve", "IS_sharpe", "OOS_sharpe",
                         "Y2022_sharpe", "Y2024_25_sharpe",
                         "FULL_sharpe", "FULL_ret"]].to_string(index=False))
    else:
        print("(no sleeves passed both gates)")

    surv_names = survivors["sleeve"].tolist()

    panel_df = pd.concat(scaled, axis=1, sort=True).fillna(0.0)
    if panel_df.index.tz is None:
        panel_df.index = panel_df.index.tz_localize("UTC")
    if surv_names:
        panel_df["survivors_mean"] = panel_df[surv_names].mean(axis=1)
    else:
        panel_df["survivors_mean"] = 0.0
    out_df = panel_df.reset_index().rename(columns={"index": "timestamp",
                                                    "date": "timestamp"})
    out_df.to_parquet(OUT / "funding_proxy_returns.parquet", index=False)

    if surv_names:
        combined = panel_df[surv_names].mean(axis=1)
    else:
        combined = panel_df["survivors_mean"]
    sc = stats("FUNDING_PROXY_COMBINED", combined)
    print("\n=== Combined survivor sleeve (equal-weight) ===")
    for tag in ["FULL", "IS", "OOS", "Y2022", "Y2024_25"]:
        sh = sc.get(f"{tag}_sharpe", 0)
        rt = sc.get(f"{tag}_ret", 0)
        vv = sc.get(f"{tag}_vol", 0)
        print(f"  {tag:<8}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  AnnVol={vv:.2%}")

    print("\n=== Combined yearly Sharpe ===")
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 20:
            continue
        bpy = _bpy(sub.index)
        std = sub.std(ddof=0)
        sh = (sub.mean() * bpy) / (std * np.sqrt(bpy)) if std > 0 else 0.0
        rt = sub.mean() * bpy
        print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")

    # --- Correlations vs existing sleeves ---
    print("\n=== Correlations vs existing master sleeves ===")
    corr_df = cross_correlations(combined)
    if not corr_df.empty:
        print(corr_df.to_string(index=False))
    else:
        print("(no master panel)")


if __name__ == "__main__":
    main()
