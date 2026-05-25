"""Multi-day pattern recognition strategies — wave 6.

Seven 2-5-day pattern sub-sleeves on D1.  These are folklore patterns
that traders cite constantly: 3-bar reversals, hammers, momentum
acceleration, absorption days, trend confirmation, climaxes, range
contraction breakouts.  We test them systematically.

Sub-sleeves
-----------
  1. THREE_BAR_REVERSAL  — close[t-2]<close[t-1] AND close[t-1]>close[t]
                           AND close[t]>close[t-2] (bearish reversal candidate)
                           => short for 2 bars; mirror long.
  2. HAMMER_SHOOTING     — Body in upper 10% of range -> hammer (long 2 bars);
                           lower 10% -> shooting star (short 2 bars).
  3. MOM_ACCEL_5         — trailing 2-bar ret > 2 * trailing 5-bar ret
                           (acceleration) => follow direction for 3 bars.
  4. ABSORPTION_2BAR     — Day1 large bar w/ high volume; Day2 small bar that
                           absorbs prior move (closes near day-1 low after
                           up-bar or near day-1 high after down-bar).
                           Expect reversal for 2 bars.
  5. TREND_CONFIRM_5     — Long when 3 of last 5 bars closed higher AND each
                           bar's high > prior bar's high.  Mirror short.
  6. VOL_CLIMAX_CRYPTO   — Crypto only: vol > 2x 20d avg AND close in lower
                           30% of range => long for 5 days.  Mirror short.
  7. RANGE_CONTRACT_BO   — 5-bar TR < 50% of 20-bar TR (contraction); wait
                           for breakout bar (TR > 1.5x 20-bar).  Follow.

Methodology
-----------
- IS:  timestamp < 2024-01-01.  OOS: >= 2024-01-01.
- All thresholds and IS-best sign-picks fit on IS only.  Gating uses
  shift(1) trailing windows (walk-forward by construction).
- Each sub-sleeve scaled to 5% IS ann vol.
- Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/multiday_patterns.py             (this file)
  scratch/wave6/multiday_returns.parquet         (per-sleeve daily streams)
  scratch/wave6/multiday_breakdown.csv           (per-sleeve summary stats)
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
FX_SYMS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
INDEX_SYMS = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP",
              "DE30_EUR", "JP225_USD"]
ALL_D1 = CRYPTO_SYMS + FX_SYMS + INDEX_SYMS


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
    if len(r) < 5:
        for tag in ["FULL", "IS", "OOS", "Y2022"]:
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
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    if rets.index.tz is None:
        rets.index = rets.index.tz_localize("UTC")
    return rets, res


def _equal_weight_combo(streams, surv):
    if not surv:
        if not streams:
            return pd.Series(dtype=float)
        panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    else:
        panel = pd.concat({s: streams[s] for s in surv}, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    return combo


def _true_range(df):
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    prev_close = df["close"].astype("float64").shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _spread_position_to_horizon(events_idx, sign_at_event, horizons, n):
    """Set position at event_idx+h for each h in horizons."""
    pos = np.zeros(n, dtype="float64")
    for idx0, sgn in zip(events_idx, sign_at_event):
        for h in horizons:
            target = idx0 + h
            if 0 <= target < n:
                pos[target] = float(sgn)
    return pos


# --- data load -------------------------------------------------------------

print("Loading D1 data...")
DATA_D1 = {s: get_candles(s, "D1") for s in ALL_D1}
for s, df in DATA_D1.items():
    print(f"  D1 {s:12} {len(df):5}  "
          f"{df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


# ---------------------------------------------------------------------------
# Strategy 1 — Three-bar reversal pattern
# ---------------------------------------------------------------------------

def strat_three_bar_reversal():
    """Bearish setup: close[t-2] < close[t-1] AND close[t-1] > close[t]
                      AND close[t] > close[t-2]  ("lower high after pop")
    Bullish mirror: close[t-2] > close[t-1] AND close[t-1] < close[t]
                    AND close[t] < close[t-2]    ("higher low after drop")
    Action: hold position for 2 bars in reversal direction.  Per symbol
    IS-best sign (continuation vs reversal lets data decide).
    """
    print("\n=== Strategy 1: Three-bar reversal ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        c = df["close"].astype("float64")
        c1 = c.shift(1)
        c2 = c.shift(2)
        # bearish reversal candidate at bar t (use only info known *by end* of t,
        # so position is set for t+1, t+2).
        bear = ((c2 < c1) & (c1 > c) & (c > c2)).fillna(False).values
        bull = ((c2 > c1) & (c1 < c) & (c < c2)).fillna(False).values
        bear_evt = np.where(bear)[0]
        bull_evt = np.where(bull)[0]
        n = len(df)

        # Base position: short 2 bars after bear setup, long 2 bars after bull.
        pos_base = np.zeros(n, dtype="float64")
        for idx0 in bear_evt:
            for h in (1, 2):
                if idx0 + h < n:
                    pos_base[idx0 + h] = -1.0
        for idx0 in bull_evt:
            for h in (1, 2):
                if idx0 + h < n:
                    pos_base[idx0 + h] = +1.0

        best = None
        for sgn in [+1, -1]:
            pos = sgn * pos_base
            rets, _ = run_position(df, pos, name=f"3BR_{sym}_{sgn}", symbol=sym)
            s = stats(f"3BR_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": int(bear.sum() + bull.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={int(bear.sum() + bull.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 2 — Hammer / shooting star (body location)
# ---------------------------------------------------------------------------

def strat_hammer_shooting():
    """Hammer = body in upper 10% of bar's range (long lower wick).
    Shooting star = body in lower 10% (long upper wick).
    Hammer -> long 2 bars; shooting star -> short 2 bars.  IS-best sign.

    Body = [min(open,close), max(open,close)].  Body-top = max(o,c),
    body-bot = min(o,c).  "Body in upper 10%" means body-bot > low + 0.9*range.
    """
    print("\n=== Strategy 2: Hammer / shooting star ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        o = df["open"].astype("float64")
        c = df["close"].astype("float64")
        h = df["high"].astype("float64")
        l = df["low"].astype("float64")
        rng = (h - l).replace(0, np.nan)
        body_top = np.maximum(o, c)
        body_bot = np.minimum(o, c)
        # Hammer: small body sitting at the top => body_bot above 90% of range.
        hammer = (((body_bot - l) / rng) > 0.90).fillna(False).values
        shoot = (((body_top - l) / rng) < 0.10).fillna(False).values
        n = len(df)

        # Position for bars t+1, t+2: +1 after hammer, -1 after shooting star.
        pos_base = np.zeros(n, dtype="float64")
        for idx0 in np.where(hammer)[0]:
            for hh in (1, 2):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = +1.0
        for idx0 in np.where(shoot)[0]:
            for hh in (1, 2):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = -1.0

        best = None
        for sgn in [+1, -1]:
            pos = sgn * pos_base
            rets, _ = run_position(df, pos, name=f"HMR_{sym}_{sgn}", symbol=sym)
            s = stats(f"HMR_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": int(hammer.sum() + shoot.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={int(hammer.sum() + shoot.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 3 — 5-bar momentum acceleration
# ---------------------------------------------------------------------------

def strat_mom_accel():
    """Trailing 2-bar return ratio > 2 * trailing 5-bar return (in absolute,
    same sign).  Interpreted: short-window momentum has accelerated relative
    to medium-window.  Hold direction for 3 bars.  IS-best sign per symbol.
    """
    print("\n=== Strategy 3: Momentum acceleration ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        c = df["close"].astype("float64")
        # 2-bar ret = c[t]/c[t-2] - 1; 5-bar ret = c[t]/c[t-5] - 1.
        r2 = (c / c.shift(2) - 1).fillna(0.0)
        r5 = (c / c.shift(5) - 1).fillna(0.0)
        same_sign = (np.sign(r2) == np.sign(r5)) & (np.sign(r2) != 0)
        accel = same_sign & (r2.abs() > 2 * r5.abs())
        accel = accel.fillna(False).values
        n = len(df)

        sgn_arr = np.sign(r2.values)
        events = np.where(accel)[0]
        pos_base = np.zeros(n, dtype="float64")
        for idx0 in events:
            s_here = sgn_arr[idx0]
            for hh in (1, 2, 3):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = float(s_here)

        best = None
        for sgn in [+1, -1]:
            pos = sgn * pos_base
            rets, _ = run_position(df, pos, name=f"ACC_{sym}_{sgn}", symbol=sym)
            s = stats(f"ACC_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": int(accel.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={int(accel.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 4 — 2-bar absorption
# ---------------------------------------------------------------------------

def strat_absorption():
    """Day 1 (t-1): large bar (|ret| > 1.5 * 20d-mean-|ret|, shift(1)) with
    high volume (vol > 1.5 * 20d-mean-vol, shift(1)).
    Day 2 (t): small bar (range < 70% of day-1 range) that closes near
    day-1 low after up-bar (close < low[t-1] + 0.3*range[t-1]) or
    near day-1 high after down-bar (close > high[t-1] - 0.3*range[t-1]).
    Expect reversal: short 2 bars after up-absorb, long 2 bars after
    down-absorb.  IS-best sign per symbol.
    """
    print("\n=== Strategy 4: 2-bar absorption ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        c = df["close"].astype("float64")
        h = df["high"].astype("float64")
        l = df["low"].astype("float64")
        vol = df["volume"].astype("float64").fillna(0.0)
        ret = c.pct_change().fillna(0.0)
        abs_ret = ret.abs()
        avg_abs = abs_ret.rolling(20, min_periods=20).mean().shift(1)
        avg_vol = vol.rolling(20, min_periods=20).mean().shift(1)
        rng = (h - l).replace(0, np.nan)

        prev_ret = ret.shift(1)
        prev_rng = rng.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)
        prev_abs_ret = abs_ret.shift(1)
        prev_vol = vol.shift(1)

        day1_large = (prev_abs_ret > 1.5 * avg_abs.shift(1)).fillna(False)
        day1_high_vol = (prev_vol > 1.5 * avg_vol.shift(1)).fillna(False)
        day1_up = (prev_ret > 0).fillna(False)
        day1_down = (prev_ret < 0).fillna(False)

        day2_small = (rng < 0.7 * prev_rng).fillna(False)
        # close near day-1 low (after up bar)
        close_near_low = (c < (prev_low + 0.3 * prev_rng)).fillna(False)
        close_near_high = (c > (prev_high - 0.3 * prev_rng)).fillna(False)

        up_absorb = (day1_large & day1_high_vol & day1_up
                     & day2_small & close_near_low).fillna(False).values
        down_absorb = (day1_large & day1_high_vol & day1_down
                       & day2_small & close_near_high).fillna(False).values
        n = len(df)

        pos_base = np.zeros(n, dtype="float64")
        for idx0 in np.where(up_absorb)[0]:
            for hh in (1, 2):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = -1.0
        for idx0 in np.where(down_absorb)[0]:
            for hh in (1, 2):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = +1.0

        best = None
        for sgn in [+1, -1]:
            pos = sgn * pos_base
            rets, _ = run_position(df, pos, name=f"ABS_{sym}_{sgn}", symbol=sym)
            s = stats(f"ABS_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        n_events = int(up_absorb.sum() + down_absorb.sum())
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 5 — Bar-by-bar trend confirmation
# ---------------------------------------------------------------------------

def strat_trend_confirm():
    """Long when:
      - at least 3 of the last 5 bars closed higher (close > prev close), AND
      - each of last 5 bars has high[t-k] > high[t-k-1] (rising highs).
    Mirror: short when 3 of last 5 closed lower AND each low < prior low.
    Hold for the next bar.  IS-best sign per symbol.
    """
    print("\n=== Strategy 5: Trend confirmation (5-bar) ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        c = df["close"].astype("float64")
        h = df["high"].astype("float64")
        l = df["low"].astype("float64")
        up_close = (c > c.shift(1)).astype(int).fillna(0)
        dn_close = (c < c.shift(1)).astype(int).fillna(0)
        up_count = up_close.rolling(5, min_periods=5).sum()
        dn_count = dn_close.rolling(5, min_periods=5).sum()

        # rising highs: high > prior high for last 5 bars all true
        h_rises = (h > h.shift(1)).astype(int).fillna(0)
        l_falls = (l < l.shift(1)).astype(int).fillna(0)
        h_rises_5 = h_rises.rolling(5, min_periods=5).sum()
        l_falls_5 = l_falls.rolling(5, min_periods=5).sum()

        long_setup = ((up_count >= 3) & (h_rises_5 == 5)).fillna(False)
        short_setup = ((dn_count >= 3) & (l_falls_5 == 5)).fillna(False)
        # Position on bar t+1
        long_next = long_setup.shift(1).fillna(False).astype(bool)
        short_next = short_setup.shift(1).fillna(False).astype(bool)
        base = pd.Series(0.0, index=df.index)
        base = base.where(~long_next, 1.0)
        base = base.where(~short_next, -1.0)
        n_events = int(long_setup.sum() + short_setup.sum())

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"TC5_{sym}_{sgn}", symbol=sym)
            s = stats(f"TC5_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 6 — Volume climax + reversal (crypto only)
# ---------------------------------------------------------------------------

def strat_vol_climax():
    """Crypto only: D1 volume > 2x 20d avg AND close in lower 30% of range
    (panic-bottom proxy) -> long for 5 days.  Mirror: vol > 2x 20d AND close
    in upper 30% -> short for 5 days (euphoria-top).  IS-best sign per symbol.
    """
    print("\n=== Strategy 6: Volume climax (crypto) ===")
    streams = {}
    detail = []
    for sym in CRYPTO_SYMS:
        df = DATA_D1[sym].copy()
        c = df["close"].astype("float64")
        h = df["high"].astype("float64")
        l = df["low"].astype("float64")
        vol = df["volume"].astype("float64").fillna(0.0)
        avg_vol = vol.rolling(20, min_periods=20).mean().shift(1)
        rng = (h - l).replace(0, np.nan)
        pos_in_rng = (c - l) / rng
        vol_spike = (vol > 2.0 * avg_vol).fillna(False)

        bottom_setup = (vol_spike & (pos_in_rng < 0.30)).fillna(False).values
        top_setup = (vol_spike & (pos_in_rng > 0.70)).fillna(False).values
        n = len(df)

        pos_base = np.zeros(n, dtype="float64")
        for idx0 in np.where(bottom_setup)[0]:
            for hh in (1, 2, 3, 4, 5):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = +1.0
        for idx0 in np.where(top_setup)[0]:
            for hh in (1, 2, 3, 4, 5):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = -1.0

        best = None
        for sgn in [+1, -1]:
            pos = sgn * pos_base
            rets, _ = run_position(df, pos, name=f"VCX_{sym}_{sgn}", symbol=sym)
            s = stats(f"VCX_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        n_events = int(bottom_setup.sum() + top_setup.sum())
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 7 — Range contraction breakout
# ---------------------------------------------------------------------------

def strat_range_contract_bo():
    """5-bar mean TR < 50% of 20-bar mean TR (contraction).  When TR on bar t
    > 1.5 * 20-bar mean TR (breakout), follow the breakout direction for
    next 2 bars.  Direction = sign(close[t] - mid[t]) where mid = (h+l)/2.
    Per symbol IS-best sign.
    """
    print("\n=== Strategy 7: Range contraction breakout ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        c = df["close"].astype("float64")
        h = df["high"].astype("float64")
        l = df["low"].astype("float64")
        tr = _true_range(df)
        tr5 = tr.rolling(5, min_periods=5).mean()
        tr20 = tr.rolling(20, min_periods=20).mean()
        contracted = (tr5.shift(1) < 0.5 * tr20.shift(1)).fillna(False)
        breakout = (tr > 1.5 * tr20.shift(1)).fillna(False)
        trigger = (contracted & breakout).fillna(False).values

        mid = (h + l) / 2.0
        direction = np.sign((c - mid).fillna(0.0)).values
        n = len(df)
        pos_base = np.zeros(n, dtype="float64")
        for idx0 in np.where(trigger)[0]:
            for hh in (1, 2):
                if idx0 + hh < n:
                    pos_base[idx0 + hh] = float(direction[idx0])

        best = None
        for sgn in [+1, -1]:
            pos = sgn * pos_base
            rets, _ = run_position(df, pos, name=f"RCB_{sym}_{sgn}", symbol=sym)
            s = stats(f"RCB_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        n_events = int(trigger.sum())
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_sleeves = {}
    details = {}

    raw_sleeves["THREE_BAR_REVERSAL"], details["THREE_BAR_REVERSAL"] = strat_three_bar_reversal()
    raw_sleeves["HAMMER_SHOOTING"], details["HAMMER_SHOOTING"] = strat_hammer_shooting()
    raw_sleeves["MOM_ACCEL_5"], details["MOM_ACCEL_5"] = strat_mom_accel()
    raw_sleeves["ABSORPTION_2BAR"], details["ABSORPTION_2BAR"] = strat_absorption()
    raw_sleeves["TREND_CONFIRM_5"], details["TREND_CONFIRM_5"] = strat_trend_confirm()
    raw_sleeves["VOL_CLIMAX_CRYPTO"], details["VOL_CLIMAX_CRYPTO"] = strat_vol_climax()
    raw_sleeves["RANGE_CONTRACT_BO"], details["RANGE_CONTRACT_BO"] = strat_range_contract_bo()

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
        print(f"  scale[{name:<22}] = {scale:.3f}")

    breakdown = pd.DataFrame(breakdown_rows)
    cols = ["sleeve"] + [c for c in breakdown.columns
                        if c not in ("sleeve", "label")]
    breakdown = breakdown[cols]
    breakdown.to_csv(OUT / "multiday_breakdown.csv", index=False)

    print("\n=== Breakdown ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "FULL_sharpe",
                     "FULL_ret", "FULL_vol", "scale"]].to_string(index=False))

    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.5)
                          & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) "
          f"[IS Sharpe >= 0.5 AND OOS Sharpe >= 0] ===")
    if not survivors.empty:
        print(survivors[["sleeve", "IS_sharpe", "OOS_sharpe",
                         "Y2022_sharpe", "FULL_sharpe",
                         "FULL_ret"]].to_string(index=False))
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
    out_df.to_parquet(OUT / "multiday_returns.parquet", index=False)

    if surv_names:
        combined = panel_df[surv_names].mean(axis=1)
    else:
        combined = panel_df["survivors_mean"]
    sc = stats("MULTIDAY_COMBINED", combined)
    print("\n=== Combined survivor sleeve (equal-weight) ===")
    for tag in ["FULL", "IS", "OOS", "Y2022"]:
        sh = sc.get(f"{tag}_sharpe", 0)
        rt = sc.get(f"{tag}_ret", 0)
        vv = sc.get(f"{tag}_vol", 0)
        print(f"  {tag:<6}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  AnnVol={vv:.2%}")

    print("\n=== Combined yearly Sharpe ===")
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 20:
            continue
        bpy = _bpy(sub.index)
        std = sub.std(ddof=0)
        sh = (sub.mean() * bpy) / (std * np.sqrt(bpy)) if std > 0 else 0.0
        rt = sub.mean() * bpy
        print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")

    # --- Correlation with MICROSTRUCTURE_D1 (if available) ---
    micro_path = OUT / "microstructure_returns.parquet"
    if micro_path.exists() and surv_names:
        try:
            micro = pd.read_parquet(micro_path)
            # find timestamp column
            ts_col = "timestamp" if "timestamp" in micro.columns else micro.columns[0]
            micro[ts_col] = pd.to_datetime(micro[ts_col], utc=True)
            micro = micro.set_index(ts_col)
            if "survivors_mean" in micro.columns:
                micro_combo = micro["survivors_mean"]
            else:
                micro_combo = micro.mean(axis=1)
            joint = pd.concat([combined.rename("multi"),
                               micro_combo.rename("micro")], axis=1).dropna()
            if len(joint) > 30:
                corr_full = joint.corr().iloc[0, 1]
                is_mask = joint.index < SPLIT
                oos_mask = joint.index >= SPLIT
                corr_is = joint[is_mask].corr().iloc[0, 1] if is_mask.sum() > 30 else float("nan")
                corr_oos = joint[oos_mask].corr().iloc[0, 1] if oos_mask.sum() > 30 else float("nan")
                print("\n=== Correlation with MICROSTRUCTURE_D1 survivors_mean ===")
                print(f"  FULL: {corr_full:+.3f}   IS: {corr_is:+.3f}   OOS: {corr_oos:+.3f}")
        except Exception as e:
            print(f"  (couldn't compute microstructure corr: {e})")


if __name__ == "__main__":
    main()
