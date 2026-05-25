"""Calendar-event clustering strategies — wave 6.

We don't have explicit FOMC / ECB / BoJ event dates, so we *infer* event days
from anomalous price-action footprints (vol spikes, calendar heuristics) and
trade the day-after follow-through or fade.

Sub-sleeves
-----------
  1. VOL_SPIKE_DAYAFTER     — per symbol, when |D1 ret| > 3 SD trade the t+1 day.
  2. FOMC_DETECT_SPX        — SPX trades the day-after "last-Wed-of-month"
                              (FOMC heuristic, ex-December).
  3. CRYPTO_VOL_CLUSTER_FADE — BTC: 3 consecutive |D1 ret| > 2σ → fade next day.
  4. FRIDAY_INDEX_EFFECT    — long indices on Fridays (with 3rd-Friday overlay).
  5. MONTHEND_REBALANCE     — last 2 D1 bars of month → long top-3 indices by
                              trailing 21d return.
  6. POST_HOLIDAY_CATCHUP   — first US trading day after 3-day weekend → SPX.
  7. EURUSD_DE30_SPILLOVER  — EUR_USD |D1 ret| > 2σ → DE30 next 4 H1 bars.

Methodology
-----------
- All SD/quantile gates are walk-forward (use only data prior to bar t via
  shift(1) and expanding mean / std).
- IS: timestamp < 2024-01-01.  OOS: timestamp >= 2024-01-01.
- Each surviving sub-sleeve is vol-scaled to 5% IS ann vol uniformly applied.
- Survivor filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/event_clusters.py                 (this file)
  scratch/wave6/event_clusters_returns.parquet    (per-sleeve daily streams)
  scratch/wave6/event_clusters_breakdown.csv      (per-sleeve stats)
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
EQ_BARS_PER_YEAR = 252.0
CRYPTO_BARS_PER_YEAR = 365.25

INDEX_SYMS = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP", "DE30_EUR", "JP225_USD"]
FX_SYMS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
CRYPTO_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ALL_D1 = INDEX_SYMS + FX_SYMS + CRYPTO_SYMS


# --- helpers ---------------------------------------------------------------

def _bpy(idx):
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


def trading_date(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the *trading-session date* per bar as a DatetimeIndex.

    OANDA D1 candles for FX/indices are stamped 21:00 or 22:00 UTC at the bar
    *start*, so the bar at Thu 22:00 is Friday's NY session. Normalize by
    adding 6 hours then taking the date — that puts crypto (00:00 UTC) into
    the same calendar day and FX/index bars into the trading day they close.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return pd.DatetimeIndex((ts + pd.Timedelta(hours=6)).dt.normalize().values, tz="UTC")


def run_position(df: pd.DataFrame, pos, *, name: str, symbol: str, timeframe="D1"):
    p = pd.Series(np.asarray(pos), index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, p, symbol=symbol, timeframe=timeframe, name=name)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    if rets.index.tz is None:
        rets.index = rets.index.tz_localize("UTC")
    return rets, res


# --- data load -------------------------------------------------------------

print("Loading D1 + H1 data...")
DATA_D1 = {s: get_candles(s, "D1") for s in ALL_D1}
DATA_H1 = {}
for s in ["EUR_USD", "DE30_EUR", "SPX500_USD"]:
    DATA_H1[s] = get_candles(s, "H1")

for s, df in DATA_D1.items():
    print(f"  D1 {s:12} {len(df):5}  "
          f"{df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")
for s, df in DATA_H1.items():
    print(f"  H1 {s:12} {len(df):6} {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")


# ---------------------------------------------------------------------------
# Strategy 1 — Vol-spike day-after
# ---------------------------------------------------------------------------

def strat_vol_spike_dayafter():
    """Per-symbol: when D1 |ret| > 3 walk-forward SD, look at t+1, t+2, t+3.

    We pick the lookahead horizon and the sign (long/short) using IS only:
    for each (horizon, sign) compute IS Sharpe of the resulting sleeve, pick
    the best. Then the sleeve trades that direction every event.
    """
    print("\n=== Strategy 1: Vol-spike day-after ===")
    streams = {}
    detail_rows = []
    for sym, df in DATA_D1.items():
        close = df["close"].astype("float64")
        ret = close.pct_change()
        # walk-forward expanding SD shifted by 1
        exp_std = ret.expanding(min_periods=60).std().shift(1)
        spike = (ret.abs() > 3.0 * exp_std).fillna(False)
        spike_sign = np.sign(ret.where(spike, 0.0))

        # build candidate positions for each horizon (t+1..t+3) and pick best IS
        ts_utc = pd.to_datetime(df["timestamp"], utc=True)
        ts_norm = ts_utc.dt.normalize()
        best = None
        for h in [1, 2, 3]:
            for sgn in [+1, -1]:
                pos = pd.Series(0.0, index=df.index)
                # When we "see" a spike on bar t, we want to hold position
                # during bar t+h (so position[t+h] is set, position[t] is not).
                # We must set it from info observable at start of bar t+h,
                # which is fine since spike[t] is observable end-of-bar t.
                events = np.where(spike.values)[0]
                for idx0 in events:
                    target_idx = idx0 + h
                    if target_idx >= len(df):
                        continue
                    pos.iloc[target_idx] = sgn * spike_sign.iloc[idx0]
                rets, _ = run_position(df, pos, name=f"VS_{sym}_h{h}_s{sgn}",
                                       symbol=sym)
                s = stats(f"VS_{sym}_h{h}_s{sgn}", rets)
                key = (h, sgn)
                if best is None or s["IS_sharpe"] > best[2]["IS_sharpe"]:
                    best = (key, rets, s, pos)

        (h, sgn), rets, s, pos = best
        streams[sym] = rets
        detail_rows.append({
            "symbol": sym, "horizon": h, "sign": sgn,
            "n_events_total": int(spike.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} h={h}  s={sgn:+d}  events={int(spike.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    detail = pd.DataFrame(detail_rows)

    # Equal-weight across IS-survivor symbols
    surv = detail[detail["IS_sharpe"] >= 0.4]["symbol"].tolist()
    if surv:
        panel = pd.concat({s: streams[s] for s in surv}, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    else:
        all_panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        combo = all_panel.mean(axis=1)
        surv = list(streams.keys())
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv}")
    return combo, detail


# ---------------------------------------------------------------------------
# Strategy 2 — FOMC-detection on SPX (last-Wed-of-month, ex-December)
# ---------------------------------------------------------------------------

def _is_last_wednesday(ts: pd.Timestamp) -> bool:
    """True if ts is the last Wednesday of its month."""
    if ts.weekday() != 2:        # 2 = Wednesday
        return False
    # next Wednesday is in the next month?
    nxt = ts + pd.Timedelta(days=7)
    return nxt.month != ts.month


def strat_fomc_spx():
    print("\n=== Strategy 2: FOMC-detection on SPX (last-Wed-of-month) ===")
    sym = "SPX500_USD"
    df = DATA_D1[sym].copy()
    td = trading_date(df)

    is_fomc_day = pd.Series(
        [_is_last_wednesday(d) and d.month != 12 for d in td],
        index=df.index,
    )
    # We trade the day-after the FOMC: hold position during t+1 close-to-close.
    # Bar i is t+1 if the previous calendar trading day was an FOMC-candidate.
    # Simpler approach: position[i] = sign if is_fomc_day[i-1].
    fomc_prev = is_fomc_day.shift(1).fillna(False).astype(bool)

    # Direction: pick IS-best (long vs short).
    best = None
    for sgn in [+1, -1]:
        pos = pd.Series(np.where(fomc_prev.values, sgn, 0.0), index=df.index)
        rets, _ = run_position(df, pos, name=f"FOMC_{sgn}", symbol=sym)
        s = stats(f"FOMC_{sgn}", rets)
        if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
            best = (sgn, s, rets)
    sgn, s, rets = best
    print(f"  best dir = {sgn:+d}  n_events={int(fomc_prev.sum())}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Strategy 3 — Crypto vol-cluster fade (BTC)
# ---------------------------------------------------------------------------

def strat_crypto_vol_cluster_fade():
    print("\n=== Strategy 3: Crypto vol-cluster fade (BTC, 3 cons. >2σ) ===")
    sym = "BTCUSDT"
    df = DATA_D1[sym].copy()
    close = df["close"].astype("float64")
    ret = close.pct_change()
    # walk-forward expanding SD
    exp_std = ret.expanding(min_periods=60).std().shift(1)
    big = (ret.abs() > 2.0 * exp_std).fillna(False)
    # 3 consecutive big moves
    rolled = big.rolling(3).sum()
    cluster_end = (rolled == 3).fillna(False)
    # The "next day" fade: position[i+1] = -sign(ret[i]).
    sign_today = np.sign(ret.where(cluster_end, 0.0))
    pos = pd.Series(0.0, index=df.index)
    events = np.where(cluster_end.values)[0]
    for idx0 in events:
        nxt = idx0 + 1
        if nxt >= len(df):
            continue
        pos.iloc[nxt] = -float(sign_today.iloc[idx0])
    rets, _ = run_position(df, pos, name="CRYPTO_VCFADE", symbol=sym)
    s = stats("CRYPTO_VCFADE", rets)
    print(f"  events={int(cluster_end.sum())}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Strategy 4 — Friday effect on indices (with 3rd-Friday overlay)
# ---------------------------------------------------------------------------

def _is_third_friday(ts: pd.Timestamp) -> bool:
    if ts.weekday() != 4:
        return False
    return 15 <= ts.day <= 21


def strat_friday_indices():
    """Long indices on Fridays. Also test 3rd-Friday (options expiry) overlay.

    For each index, pick IS-best of {all-Fri long, all-Fri short, 3rd-Fri long,
    3rd-Fri short}. Then equal-weight across IS-survivors.
    """
    print("\n=== Strategy 4: Friday-effect on indices ===")
    streams = {}
    detail = []
    for sym in INDEX_SYMS:
        df = DATA_D1[sym].copy()
        td = trading_date(df)
        is_fri = pd.Series(td.weekday == 4, index=df.index)
        is_3rd_fri = pd.Series([_is_third_friday(d) for d in td], index=df.index)

        variants = {
            "ALL_FRI_LONG": (is_fri, +1),
            "ALL_FRI_SHORT": (is_fri, -1),
            "3RD_FRI_LONG": (is_3rd_fri, +1),
            "3RD_FRI_SHORT": (is_3rd_fri, -1),
        }
        best = None
        for label, (mask, sgn) in variants.items():
            pos = pd.Series(np.where(mask.values, sgn, 0.0), index=df.index)
            rets, _ = run_position(df, pos, name=f"FRI_{sym}_{label}", symbol=sym)
            s = stats(label, rets)
            if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
                best = (label, s, rets)
        label, s, rets = best
        streams[sym] = rets
        detail.append({
            "symbol": sym, "variant": label,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} pick={label:<15}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.4]["symbol"].tolist()
    if surv:
        panel = pd.concat({s: streams[s] for s in surv}, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    else:
        panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
        surv = list(streams.keys())
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 5 — Month-end rebalance flow
# ---------------------------------------------------------------------------

def strat_monthend_rebalance():
    """Last 2 trading bars of month: long top-3 indices by trailing 21d return.

    We pick the months on a per-index level — each index has its own month-end
    bars. For each bar that is one of the last 2 trading days of its month,
    rank the indices by trailing 21d return (using only data observable at
    start of bar), long top-3, short bottom-0 (long-only basket).
    """
    print("\n=== Strategy 5: Month-end rebalance flow (long top-3 indices) ===")
    # Build a common trading-date panel of close prices for ranking.
    panel_close = {}
    for sym in INDEX_SYMS:
        df = DATA_D1[sym].copy()
        td = trading_date(df)
        close = df["close"].astype("float64")
        s_close = pd.Series(close.values, index=pd.DatetimeIndex(td))
        s_close = s_close.groupby(s_close.index).last()
        panel_close[sym] = s_close

    close_panel = pd.concat(panel_close, axis=1, sort=True).ffill()
    close_panel.index = pd.DatetimeIndex(close_panel.index)
    if close_panel.index.tz is None:
        close_panel.index = close_panel.index.tz_localize("UTC")

    # 21d trailing return (shifted by 1 so observable at bar start)
    trail21 = close_panel.pct_change(21).shift(1)

    # Last 2 trading bars of month: a bar where the next-bar or 2nd-next-bar
    # lies in a different (year, month) than this bar.
    dates = pd.DatetimeIndex(close_panel.index)
    month_id = pd.Series(dates.year * 12 + dates.month, index=dates).astype(int)
    next1 = month_id.shift(-1).ffill().astype(int)
    next2 = month_id.shift(-2).ffill().astype(int)
    is_last2 = (next1 != month_id) | (next2 != month_id)

    # Rank each row of trail21; top-3 → weight 1/3, else 0
    ranks = trail21.rank(axis=1, ascending=False)
    weights = (ranks <= 3).astype(float) / 3.0
    weights = weights.mul(is_last2.astype(float), axis=0)

    # Per-symbol daily strategy returns net of cost: use backtest engine.
    streams = {}
    for sym in INDEX_SYMS:
        df = DATA_D1[sym].copy()
        td = trading_date(df)
        w_series = weights[sym].reindex(pd.DatetimeIndex(td)).fillna(0.0)
        pos = pd.Series(w_series.values, index=df.index)
        rets, _ = run_position(df, pos, name=f"ME_REB_{sym}", symbol=sym)
        streams[sym] = rets

    combined = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    s = stats("MONTHEND_REB", combined)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")
    return combined


# ---------------------------------------------------------------------------
# Strategy 6 — Post-holiday catch-up on SPX
# ---------------------------------------------------------------------------

# Hard-code US Memorial Day / July 4 / Labor Day / Thanksgiving observed dates
# in our data range. We don't need them all — we *detect* a 3+ day weekend by
# the gap between successive SPX trading bars.

def strat_post_holiday_catchup():
    """First SPX trading bar after >=3 calendar-day gap (3-day weekend)."""
    print("\n=== Strategy 6: Post-holiday catch-up SPX ===")
    sym = "SPX500_USD"
    df = DATA_D1[sym].copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    diffs = ts.diff().dt.total_seconds().div(86400.0)
    # gap > 2 days = at least one weekday skipped (i.e. Monday holiday or
    # extended holiday weekend). Most weekends are 3 days (Fri->Mon = 3 days).
    # 3-day weekend = Mon holiday, Fri->Tue = 4 calendar days; or
    # Fri holiday + weekend = Thu->Mon = 4 days; or
    # Thanksgiving (Thu off, plus Fri short, Sat/Sun) = Wed->Mon = 5 days.
    long_gap = (diffs > 3.5).fillna(False)
    n_events = int(long_gap.sum())

    best = None
    for sgn in [+1, -1]:
        pos = pd.Series(np.where(long_gap.values, sgn, 0.0), index=df.index)
        rets, _ = run_position(df, pos, name=f"HOLIDAY_{sgn}", symbol=sym)
        s = stats(f"HOLIDAY_{sgn}", rets)
        if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
            best = (sgn, s, rets)
    sgn, s, rets = best
    print(f"  best dir={sgn:+d}  events={n_events}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Strategy 7 — EUR/USD shock -> DE30 spillover next 4 H1 bars
# ---------------------------------------------------------------------------

def strat_eurusd_de30_spillover():
    """When EUR_USD |D1 ret| > 2 walk-forward SD on day t, trade DE30 H1 next
    4 bars (Frankfurt morning following the European close)."""
    print("\n=== Strategy 7: EUR_USD shock -> DE30 H1 spillover ===")
    eu = DATA_D1["EUR_USD"].copy()
    eu_close = eu["close"].astype("float64")
    eu_ret = eu_close.pct_change()
    exp_std = eu_ret.expanding(min_periods=60).std().shift(1)
    big = (eu_ret.abs() > 2.0 * exp_std).fillna(False)
    sign_today = np.sign(eu_ret.where(big, 0.0))

    eu_td = trading_date(eu)
    # Trigger date list with sign (indexed by EUR_USD trading date)
    trig = pd.Series(sign_today.values, index=pd.DatetimeIndex(eu_td))
    trig = trig[trig != 0]

    de = DATA_H1["DE30_EUR"].copy()
    de_ts = pd.to_datetime(de["timestamp"], utc=True)
    # For H1 we use the actual UTC calendar date (DE30 cash session
    # is 07:00-15:30 UTC, falls in same calendar day).
    de_date = de_ts.dt.normalize()

    # For each trigger date d, fire long-or-short on the first 4 H1 bars whose
    # date is d+1 (UTC). The "sign" — we test both directions (pick IS-best).
    pos = pd.Series(0.0, index=de.index)
    # Map: H1 bar index -> trigger sign (if applicable)
    # Build a date-indexed series of trigger signs for "next day"
    nxt_date = trig.copy()
    nxt_date.index = nxt_date.index + pd.Timedelta(days=1)
    # If multiple triggers map to same day, take the last (rare).
    nxt_date = nxt_date.groupby(nxt_date.index).last()

    # For each next-day, mark the 4 H1 bars covering the Frankfurt cash open
    # (07:00, 08:00, 09:00, 10:00 UTC). DE30 H1 trades 24h, so we pick those
    # specifically rather than "first 4 bars of UTC day" (00:00-03:00 UTC).
    base_pos = pd.Series(0.0, index=de.index)
    bar_date = pd.DatetimeIndex(de_date)
    bar_hour = pd.to_datetime(de["timestamp"], utc=True).dt.hour.values
    target_hours = {7, 8, 9, 10}
    for d, sgn in nxt_date.items():
        mask = (bar_date == d) & np.isin(bar_hour, list(target_hours))
        if not mask.any():
            continue
        base_pos.loc[mask] = float(sgn)

    # Test long-spillover (same-direction) vs fade (opposite). Walk-forward IS pick.
    best = None
    for direction in [+1, -1]:
        pos_dir = base_pos * direction
        rets_h1, res = run_position(de, pos_dir, name=f"SPILL_{direction}",
                                    symbol="DE30_EUR", timeframe="H1")
        s = stats(f"SPILL_{direction}", rets_h1)
        if best is None or s["IS_sharpe"] > best[1]["IS_sharpe"]:
            best = (direction, s, rets_h1)
    direction, s, rets = best
    n_events = int(big.sum())
    print(f"  best dir={direction:+d}  EU_events={n_events}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")
    return rets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_sleeves = {}

    rets1, det1 = strat_vol_spike_dayafter()
    raw_sleeves["VOL_SPIKE_DAYAFTER"] = rets1
    det1.to_csv(OUT / "event_clusters_volspike_detail.csv", index=False)

    rets2 = strat_fomc_spx()
    raw_sleeves["FOMC_DETECT_SPX"] = rets2

    rets3 = strat_crypto_vol_cluster_fade()
    raw_sleeves["CRYPTO_VOL_CLUSTER_FADE"] = rets3

    rets4, det4 = strat_friday_indices()
    raw_sleeves["FRIDAY_INDEX_EFFECT"] = rets4
    det4.to_csv(OUT / "event_clusters_friday_detail.csv", index=False)

    rets5 = strat_monthend_rebalance()
    raw_sleeves["MONTHEND_REBALANCE"] = rets5

    rets6 = strat_post_holiday_catchup()
    raw_sleeves["POST_HOLIDAY_CATCHUP"] = rets6

    rets7 = strat_eurusd_de30_spillover()
    raw_sleeves["EURUSD_DE30_SPILLOVER"] = rets7

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
        print(f"  scale[{name:<24}] = {scale:.3f}")

    breakdown = pd.DataFrame(breakdown_rows)
    cols = ["sleeve"] + [c for c in breakdown.columns
                        if c not in ("sleeve", "label")]
    breakdown = breakdown[cols]
    breakdown.to_csv(OUT / "event_clusters_breakdown.csv", index=False)
    print("\n=== Breakdown ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "FULL_sharpe",
                     "FULL_ret", "FULL_vol", "scale"]].to_string(index=False))

    # Survivor filter
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

    # Save panel of scaled streams
    panel_df = pd.concat(scaled, axis=1, sort=True).fillna(0.0)
    if panel_df.index.tz is None:
        panel_df.index = panel_df.index.tz_localize("UTC")
    if surv_names:
        panel_df["survivors_mean"] = panel_df[surv_names].mean(axis=1)
    else:
        panel_df["survivors_mean"] = 0.0
    out_df = panel_df.reset_index().rename(columns={"index": "timestamp",
                                                    "date": "timestamp"})
    out_df.to_parquet(OUT / "event_clusters_returns.parquet", index=False)

    # Combined survivor stats
    if surv_names:
        combined = panel_df[surv_names].mean(axis=1)
    else:
        combined = panel_df["survivors_mean"]
    sc = stats("EVENT_CLUSTERS_COMBINED", combined)
    print("\n=== Combined survivor sleeve (equal-weight) ===")
    for tag in ["FULL", "IS", "OOS", "Y2022"]:
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
        std = sub.std(ddof=0)
        sh = (sub.mean() * bpy) / (std * np.sqrt(bpy)) if std > 0 else 0.0
        rt = sub.mean() * bpy
        print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")


if __name__ == "__main__":
    main()
