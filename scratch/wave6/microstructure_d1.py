"""Microstructure D1 strategies — wave 6.

Eight microstructure / pattern-recognition sub-sleeves at D1.  Whereas the
intraday sleeves run on H1/M15 timestamps, these all consume the D1 OHLC bar:
the open, the high, the low, and the close of each session.

Sub-sleeves
-----------
  1.  GAP_FILL_INDEX      — Overnight gap >0.5% on US indices, fade toward
                            prior close.  Per index, IS-best sign.
  2.  GAP_AND_GO          — Gap >+1% with positive prior-day momentum: long
                            continuation (mirror short).  Per US index.
  3.  RANGE_EXPANSION     — True range > 2x trailing-20-day average; fade
                            next bar.  Per symbol, IS-best sign.
  4.  INSIDE_BAR          — Inside bar => trade breakout next session.  Long
                            if next bar's high > prior high; short if
                            low < prior low.
  5.  THREE_BAR_DRIVE     — 3 consecutive same-direction closes with no gaps.
                            Test continuation (bars +1,+2) vs reversal (bar +4).
                            Pick IS-best per symbol.
  6.  CLOSE_NEAR_EXTREME  — Close in upper 10% of range -> long next bar;
                            lower 10% -> short.  Per symbol.
  7.  DOJI                — |close-open| < 20% range -> reversal proxy.  Pick
                            IS-best of long/short/fade-prior-bar.  Per symbol.
  8.  PIVOT_EQ            — Close/Pivot ratio (H+L+C)/3: >1.01 long,
                            <0.99 short, +/- 1 bar.  Per symbol.

Methodology
-----------
- IS:  timestamp < 2024-01-01.  OOS: >= 2024-01-01.
- All thresholds & sign-picks fit on IS only (walk-forward by construction;
  trailing windows shift(1)).
- Each sub-sleeve scaled to 5% IS ann vol.
- Survivor filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/microstructure_d1.py                 (this file)
  scratch/wave6/microstructure_returns.parquet       (per-sleeve daily streams)
  scratch/wave6/microstructure_breakdown.csv         (per-sleeve summary stats)
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

INDEX_SYMS = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP",
              "DE30_EUR", "JP225_USD"]
US_INDEX_SYMS = ["SPX500_USD", "NAS100_USD", "US30_USD"]
FX_SYMS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
CRYPTO_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ALL_D1 = INDEX_SYMS + FX_SYMS + CRYPTO_SYMS


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


def run_position(df: pd.DataFrame, pos, *, name: str, symbol: str, timeframe="D1"):
    p = pd.Series(np.asarray(pos), index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, p, symbol=symbol, timeframe=timeframe, name=name)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    if rets.index.tz is None:
        rets.index = rets.index.tz_localize("UTC")
    return rets, res


def _equal_weight_combo(streams: dict[str, pd.Series], surv: list[str]) -> pd.Series:
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


# --- data load -------------------------------------------------------------

print("Loading D1 data...")
DATA_D1 = {s: get_candles(s, "D1") for s in ALL_D1}
for s, df in DATA_D1.items():
    print(f"  D1 {s:12} {len(df):5}  "
          f"{df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


# ---------------------------------------------------------------------------
# Strategy 1 — Gap-fill on US indices
# ---------------------------------------------------------------------------

def strat_gap_fill():
    """Overnight gap = (open[t] - close[t-1]) / close[t-1].  |gap|>0.5% on a US
    index => fade the gap (target close[t-1]).  Position held during bar t,
    sign = -sign(gap).  IS-tunable sign per symbol (some may be momentum-like).
    """
    print("\n=== Strategy 1: Gap-fill on US indices ===")
    streams = {}
    detail = []
    for sym in US_INDEX_SYMS:
        df = DATA_D1[sym].copy()
        open_ = df["open"].astype("float64")
        close = df["close"].astype("float64")
        prev_close = close.shift(1)
        gap = (open_ - prev_close) / prev_close
        # Trigger threshold |gap| > 0.5%
        trigger = gap.abs() > 0.005
        # When triggered, position[t] = sgn * -sign(gap[t]).  sgn IS-tunable.
        base = -np.sign(gap.fillna(0.0))
        base = base.where(trigger.fillna(False), 0.0)

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"GAPFILL_{sym}_{sgn}",
                                   symbol=sym)
            s = stats(f"GAPFILL_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn,
            "n_events": int(trigger.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d} (fade={'YES' if sgn==1 else 'NO'})  "
              f"n={int(trigger.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE (fallback equal-wt all)'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 2 — Gap-and-go (continuation) on US indices
# ---------------------------------------------------------------------------

def strat_gap_and_go():
    """gap[t] = (open[t]-close[t-1])/close[t-1].  When gap > +1% AND prior
    bar's body was positive (close[t-1] > open[t-1]), go LONG bar t.  When
    gap < -1% AND prior bar negative, go SHORT.  IS-tunable sign per symbol
    (symbol-by-symbol either confirms continuation or fades it).
    """
    print("\n=== Strategy 2: Gap-and-go on US indices ===")
    streams = {}
    detail = []
    for sym in US_INDEX_SYMS:
        df = DATA_D1[sym].copy()
        open_ = df["open"].astype("float64")
        close = df["close"].astype("float64")
        prev_close = close.shift(1)
        prev_open = open_.shift(1)
        gap = (open_ - prev_close) / prev_close
        prior_body = (close - open_).shift(1)
        prior_mom = np.sign(prior_body.fillna(0.0))

        long_setup = (gap > 0.01) & (prior_mom > 0)
        short_setup = (gap < -0.01) & (prior_mom < 0)
        base = pd.Series(0.0, index=df.index)
        base = base.where(~long_setup, 1.0)
        base = base.where(~short_setup, -1.0)
        n_events = int(long_setup.sum() + short_setup.sum())

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"GNG_{sym}_{sgn}", symbol=sym)
            s = stats(f"GNG_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE (fallback)'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 3 — Range expansion D1 (TR > 2x 20d avg => mean-revert)
# ---------------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    prev_close = df["close"].astype("float64").shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def strat_range_expansion():
    """TR > 2x trailing-20d mean TR => fade direction of bar t on bar t+1.
    Per symbol, pick IS-best sign (the "fade" hypothesis is a prior; if the
    data prefers continuation we let IS flip it)."""
    print("\n=== Strategy 3: Range expansion D1 ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        tr = _true_range(df)
        avg20 = tr.rolling(20, min_periods=20).mean().shift(1)
        ratio = tr / avg20
        spike = (ratio > 2.0).fillna(False)
        close = df["close"].astype("float64")
        ret = close.pct_change()
        sgn_today = np.sign(ret.where(spike, 0.0))
        # Position[t+1] = -sgn_today (fade by default).  Tune sign IS.
        base = pd.Series(0.0, index=df.index)
        idx_events = np.where(spike.values)[0]
        for idx0 in idx_events:
            nxt = idx0 + 1
            if nxt >= len(df):
                continue
            base.iloc[nxt] = -float(sgn_today.iloc[idx0])

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"RE_{sym}_{sgn}", symbol=sym)
            s = stats(f"RE_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "n_events": int(spike.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={int(spike.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 4 — Inside-bar breakout
# ---------------------------------------------------------------------------

def strat_inside_bar():
    """A D1 bar is "inside" if high[t]<=high[t-1] AND low[t]>=low[t-1].
    On the next bar t+1, if high[t+1] > high[t] => long bar t+1 (close-to-close
    from the breakout signal moment we *enter at open*-equivalent; we can't
    do mid-bar so we approximate: position[t+1] = +1 if bar t+1's full range
    breaks the inside bar's high, -1 if it breaks the low, 0 otherwise).
    Because we only know the break ex-post within bar t+1, this is a
    *biased* implementation — to avoid look-ahead we instead set:
    position[t+1] = +1 if open[t+1] > high[t], -1 if open[t+1] < low[t],
    else 0.  This is the cleanest D1 implementation: hold the breakout bar.
    """
    print("\n=== Strategy 4: Inside-bar breakout ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        open_ = df["open"].astype("float64")
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        # inside bar on bar t:
        inside = ((high <= prev_high) & (low >= prev_low)).fillna(False)
        # Trigger on bar t+1:  open above prior(=inside) high OR below low.
        # Inside-bar bar t has range [low[t], high[t]].  Use *inside bar's*
        # range, not the parent bar's.  Open of t+1 vs high/low of t.
        inside_high = high.where(inside).ffill(limit=1).shift(1)
        inside_low = low.where(inside).ffill(limit=1).shift(1)
        had_inside = inside.shift(1).fillna(False)
        long_brk = had_inside & (open_ > inside_high)
        short_brk = had_inside & (open_ < inside_low)
        base = pd.Series(0.0, index=df.index)
        base = base.where(~long_brk, 1.0)
        base = base.where(~short_brk, -1.0)
        n_events = int(long_brk.sum() + short_brk.sum())

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"IB_{sym}_{sgn}", symbol=sym)
            s = stats(f"IB_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 5 — Three-bar drive (Marubozu pattern)
# ---------------------------------------------------------------------------

def strat_three_bar_drive():
    """3 consecutive same-direction closes (sign(close-open)) with no gaps
    between them (|gap[t]| < 25 bp).  Test:
       (a) continuation: long sign on bar t+1 and t+2
       (b) reversal: short sign on bar t+4 (mean-revert 4th)
    IS-pick best of {cont_long, cont_short, rev_long, rev_short, none} per
    symbol (i.e. let IS choose which hypothesis dominates).
    """
    print("\n=== Strategy 5: Three-bar drive (Marubozu) ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        open_ = df["open"].astype("float64")
        close = df["close"].astype("float64")
        body_sgn = np.sign(close - open_)
        gap = (open_ - close.shift(1)) / close.shift(1)
        no_gap = gap.abs() < 0.0025
        same_sign = (body_sgn != 0) & \
                    (body_sgn == body_sgn.shift(1)) & \
                    (body_sgn == body_sgn.shift(2))
        # Pattern complete at bar t when bar t, t-1, t-2 share sign,
        # gaps between (t-1,t-2) and (t,t-1) are small.
        no_gap_2 = no_gap & no_gap.shift(1)
        complete = (same_sign & no_gap_2).fillna(False)
        drive_sign = body_sgn.where(complete, 0.0)

        events = np.where(complete.values)[0]
        # Build 4 candidate positions:
        cand = {}
        for label, horizons, mult in [
            ("CONT12", [1, 2], +1),     # long the move on +1 and +2
            ("REV4",   [4],    -1),     # short the move on +4
        ]:
            pos = pd.Series(0.0, index=df.index)
            for idx0 in events:
                for h in horizons:
                    target = idx0 + h
                    if target >= len(df):
                        continue
                    pos.iloc[target] = mult * float(drive_sign.iloc[idx0])
            cand[label] = pos

        # Test each cand at +1 / -1 multiplier
        best = None
        for label, pos_base in cand.items():
            for sgn in [+1, -1]:
                pos = (sgn * pos_base).astype("float64")
                rets, _ = run_position(df, pos, name=f"3BD_{sym}_{label}_{sgn}",
                                       symbol=sym)
                s = stats(f"3BD_{sym}_{label}_{sgn}", rets)
                if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                    best = ((label, sgn), s, rets)
        (label, sgn), s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "pattern": label, "sign": sgn,
            "n_events": int(complete.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} pick={label}_{sgn:+d}  n={int(complete.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 6 — Close near high/low (momentum signal)
# ---------------------------------------------------------------------------

def strat_close_near_extreme():
    """Close in top 10% of the day's range -> long bar t+1.
    Close in bottom 10% -> short bar t+1.  Per symbol, IS-pick sign."""
    print("\n=== Strategy 6: Close-near-extreme ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        close = df["close"].astype("float64")
        rng = (high - low).replace(0, np.nan)
        pos_in_rng = (close - low) / rng
        long_setup = (pos_in_rng > 0.90).fillna(False)
        short_setup = (pos_in_rng < 0.10).fillna(False)
        base = pd.Series(0.0, index=df.index)
        # Position on bar t+1 (shift signal forward)
        long_next = long_setup.shift(1).fillna(False).astype(bool)
        short_next = short_setup.shift(1).fillna(False).astype(bool)
        base = base.where(~long_next, 1.0)
        base = base.where(~short_next, -1.0)
        n_events = int(long_setup.sum() + short_setup.sum())

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"CNE_{sym}_{sgn}", symbol=sym)
            s = stats(f"CNE_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 7 — Doji reversal
# ---------------------------------------------------------------------------

def strat_doji():
    """|close-open| < 20% of (high-low) => doji.  Test direction of next 2
    bars.  We try four candidate signals: long+1, long+1&+2, fade-prior+1,
    fade-prior+1&+2.  IS-pick best per symbol."""
    print("\n=== Strategy 7: Doji ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        open_ = df["open"].astype("float64")
        close = df["close"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        rng = (high - low).replace(0, np.nan)
        body = (close - open_).abs()
        is_doji = ((body / rng) < 0.20).fillna(False)
        # "fade prior": sign opposite to prior bar's body.
        prior_sgn = np.sign((close - open_).shift(1).fillna(0.0))
        # bar t's doji predicts bar t+1, t+2.
        events = np.where(is_doji.values)[0]

        cand = {}
        # long_1 / long_2: long the next 1 or 2 bars
        for h_max in [1, 2]:
            pos = pd.Series(0.0, index=df.index)
            for idx0 in events:
                for h in range(1, h_max + 1):
                    target = idx0 + h
                    if target >= len(df):
                        continue
                    pos.iloc[target] = 1.0
            cand[f"LONG_{h_max}"] = pos
        # fade prior body direction
        for h_max in [1, 2]:
            pos = pd.Series(0.0, index=df.index)
            for idx0 in events:
                for h in range(1, h_max + 1):
                    target = idx0 + h
                    if target >= len(df):
                        continue
                    pos.iloc[target] = -float(prior_sgn.iloc[idx0])
            cand[f"FADE_PRIOR_{h_max}"] = pos

        best = None
        for label, pos_base in cand.items():
            for sgn in [+1, -1]:
                pos = (sgn * pos_base).astype("float64")
                rets, _ = run_position(df, pos, name=f"DOJI_{sym}_{label}_{sgn}",
                                       symbol=sym)
                s = stats(f"DOJI_{sym}_{label}_{sgn}", rets)
                if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                    best = ((label, sgn), s, rets)
        (label, sgn), s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "pattern": label, "sign": sgn,
            "n_events": int(is_doji.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} pick={label}_{sgn:+d}  n={int(is_doji.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 8 — Pivot-point equilibrium
# ---------------------------------------------------------------------------

def strat_pivot_eq():
    """Pivot = (H+L+C)/3.  Close/pivot > 1.01 => long bar t+1.
    Close/pivot < 0.99 => short bar t+1.  Per symbol, IS-pick sign."""
    print("\n=== Strategy 8: Pivot-point equilibrium ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        close = df["close"].astype("float64")
        pivot = (high + low + close) / 3.0
        ratio = close / pivot
        long_setup = (ratio > 1.01).fillna(False)
        short_setup = (ratio < 0.99).fillna(False)
        base = pd.Series(0.0, index=df.index)
        long_next = long_setup.shift(1).fillna(False).astype(bool)
        short_next = short_setup.shift(1).fillna(False).astype(bool)
        base = base.where(~long_next, 1.0)
        base = base.where(~short_next, -1.0)
        n_events = int(long_setup.sum() + short_setup.sum())

        best = None
        for sgn in [+1, -1]:
            pos = (sgn * base).astype("float64")
            rets, _ = run_position(df, pos, name=f"PIV_{sym}_{sgn}", symbol=sym)
            s = stats(f"PIV_{sym}_{sgn}", rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (sgn, s, rets)
        sgn, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "sign": sgn, "n_events": n_events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} sgn={sgn:+d}  n={n_events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    combo = _equal_weight_combo(streams, surv)
    print(f"  -> IS survivors: {surv if surv else 'NONE'}")
    return combo, det


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_sleeves = {}
    details = {}

    r1, d1 = strat_gap_fill()
    raw_sleeves["GAP_FILL_INDEX"] = r1
    details["GAP_FILL_INDEX"] = d1

    r2, d2 = strat_gap_and_go()
    raw_sleeves["GAP_AND_GO"] = r2
    details["GAP_AND_GO"] = d2

    r3, d3 = strat_range_expansion()
    raw_sleeves["RANGE_EXPANSION"] = r3
    details["RANGE_EXPANSION"] = d3

    r4, d4 = strat_inside_bar()
    raw_sleeves["INSIDE_BAR"] = r4
    details["INSIDE_BAR"] = d4

    r5, d5 = strat_three_bar_drive()
    raw_sleeves["THREE_BAR_DRIVE"] = r5
    details["THREE_BAR_DRIVE"] = d5

    r6, d6 = strat_close_near_extreme()
    raw_sleeves["CLOSE_NEAR_EXTREME"] = r6
    details["CLOSE_NEAR_EXTREME"] = d6

    r7, d7 = strat_doji()
    raw_sleeves["DOJI"] = r7
    details["DOJI"] = d7

    r8, d8 = strat_pivot_eq()
    raw_sleeves["PIVOT_EQ"] = r8
    details["PIVOT_EQ"] = d8

    # Vol-scale each sub-sleeve to 5% IS ann vol
    scaled = {}
    breakdown_rows = []
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
    breakdown.to_csv(OUT / "microstructure_breakdown.csv", index=False)

    print("\n=== Breakdown ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "FULL_sharpe",
                     "FULL_ret", "FULL_vol", "scale"]].to_string(index=False))

    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.4)
                          & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) ===")
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
    out_df.to_parquet(OUT / "microstructure_returns.parquet", index=False)

    if surv_names:
        combined = panel_df[surv_names].mean(axis=1)
    else:
        combined = panel_df["survivors_mean"]
    sc = stats("MICROSTRUCTURE_COMBINED", combined)
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


if __name__ == "__main__":
    main()
