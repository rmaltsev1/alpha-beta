"""Statistical anomaly-detection strategies — wave 6.

Profit from unusual market behavior: when normal price relationships break
down (correlations, beta, residuals, cross-sectional ordering, vol-regime,
range, asset-class dispersion), fade or convergence-trade the dislocation.

Sub-sleeves
-----------
  1. CORR_BREAKDOWN_CONVERGE   — pair-wise 60d rolling corr; if corr drops
                                  > 2σ below historical mean → expect
                                  convergence (long laggard / short leader).
                                  Pairs: BTC-ETH, SPX-NAS, EUR-GBP.
  2. BETA_DRIFT_SPX_NAS        — rolling 60d beta(NAS, SPX) vs 252d avg;
                                  |drift| > 0.3 → mean-revert relative.
  3. CUMRES_ZSCORE_FADE        — per symbol: 90d cumulative residual-from-
                                  trend; |z| > 2.5 → fade.
  4. CROSS_SECTION_OUTLIER     — daily winner of |z-score return vs xs median|
                                  → short for 2 days.
  5. VOL_REGIME_FADE           — short-vol exposure when 5d vol / calendar-
                                  week historical-5d-vol-avg is in top 5%.
  6. RANGE_EXPANSION_FADE      — D1 range > 3× 20d avg AND close in extreme
                                  half of range → fade for 2 days.
  7. ASSET_CLASS_CALM          — lean into the basket (crypto/fx/index)
                                  whose 21d std vs 252d-rolling std ratio
                                  is the *lowest* (avoiding chaos), weekly.

Methodology
-----------
- IS:   timestamp < 2024-01-01.
- OOS:  timestamp >= 2024-01-01.
- Walk-forward: gates use expanding statistics shift(1).
- Vol-scale each sub-sleeve to 5% IS ann vol.
- Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/anomaly_detection.py
  scratch/wave6/anomaly_returns.parquet
  scratch/wave6/anomaly_breakdown.csv
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
INDEX_SYMS = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP", "DE30_EUR", "JP225_USD"]
ALL_D1 = CRYPTO_SYMS + FX_SYMS + INDEX_SYMS


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
    """Normalize bar-stamps to a trading-session date for cross-symbol joins."""
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

print("Loading D1 data...")
DATA_D1 = {s: get_candles(s, "D1") for s in ALL_D1}
for s, df in DATA_D1.items():
    print(f"  D1 {s:12} {len(df):5}  "
          f"{df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


def close_panel(symbols):
    """Trading-date-indexed close-price panel (forward-filled)."""
    cols = {}
    for sym in symbols:
        df = DATA_D1[sym]
        td = trading_date(df)
        s = pd.Series(df["close"].astype("float64").values, index=pd.DatetimeIndex(td))
        s = s.groupby(s.index).last()
        cols[sym] = s
    panel = pd.concat(cols, axis=1, sort=True).ffill()
    panel.index = pd.DatetimeIndex(panel.index)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    return panel


# ---------------------------------------------------------------------------
# Strategy 1 — Correlation breakdown convergence
# ---------------------------------------------------------------------------

def strat_corr_breakdown():
    """For each historical pair: when 60d corr drops > 2σ below its expanding
    historical mean, expect convergence. Identify laggard = recent 20d
    underperformer; long laggard, short leader. Hold until corr recovers
    above its expanding mean OR 10 bars (whichever first)."""
    print("\n=== Strategy 1: Correlation breakdown convergence ===")
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("SPX500_USD", "NAS100_USD"),
        ("EUR_USD", "GBP_USD"),
    ]

    streams = []
    detail = []
    for a, b in pairs:
        panel = close_panel([a, b])
        ra = panel[a].pct_change()
        rb = panel[b].pct_change()
        # 60d rolling correlation
        roll = ra.rolling(60).corr(rb)
        # Expanding mean / std of historical rolling-corr (lookback only)
        exp_mu = roll.expanding(min_periods=200).mean().shift(1)
        exp_sd = roll.expanding(min_periods=200).std().shift(1)
        z = (roll.shift(1) - exp_mu) / exp_sd
        # Break = z < -2 (corr dropped >2σ below mean)
        break_event = (z < -2.0).fillna(False)

        # Recent 20d performance gap -> laggard/leader
        # rel = a-return - b-return cumulative over 20d (observed at start of t)
        rel20 = (panel[a].pct_change(20).shift(1) - panel[b].pct_change(20).shift(1))
        # if rel20 < 0 → a is laggard → long a, short b → sign +1 means long-a/short-b
        sign_pair = -np.sign(rel20)

        # Build per-symbol position from event triggers with 10-bar hold
        sym_pos = {a: pd.Series(0.0, index=panel.index),
                   b: pd.Series(0.0, index=panel.index)}
        events_n = 0
        in_trade = False
        bars_left = 0
        cur_sign = 0.0
        for i, ts in enumerate(panel.index):
            # decrement hold
            if in_trade:
                bars_left -= 1
                # exit when corr recovers above expanding mean
                if not pd.isna(roll.iloc[i]) and not pd.isna(exp_mu.iloc[i]) \
                        and roll.iloc[i] >= exp_mu.iloc[i]:
                    in_trade = False
                    bars_left = 0
                elif bars_left <= 0:
                    in_trade = False
            if in_trade:
                sym_pos[a].iloc[i] = +cur_sign
                sym_pos[b].iloc[i] = -cur_sign
            # check entry at bar t: signal observed before bar (using shift(1))
            elif break_event.iloc[i] and not pd.isna(sign_pair.iloc[i]) \
                    and sign_pair.iloc[i] != 0:
                cur_sign = float(sign_pair.iloc[i])
                in_trade = True
                bars_left = 10
                events_n += 1
                sym_pos[a].iloc[i] = +cur_sign
                sym_pos[b].iloc[i] = -cur_sign

        # Map sym_pos back onto each symbol's df and backtest
        pair_streams = []
        for sym in (a, b):
            df = DATA_D1[sym]
            td = trading_date(df)
            p_aligned = sym_pos[sym].reindex(pd.DatetimeIndex(td)).fillna(0.0)
            pos = pd.Series(p_aligned.values, index=df.index)
            rets, _ = run_position(df, pos, name=f"CORR_BRK_{a}_{b}_{sym}",
                                   symbol=sym)
            pair_streams.append(rets)
        combo = pd.concat(pair_streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
        s = stats(f"CORR_{a}_{b}", combo)
        detail.append({
            "pair": f"{a}/{b}", "events": events_n,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        streams.append(combo)
        print(f"  pair={a}/{b:<11}  events={events_n:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    combined = pd.concat(streams, axis=1, sort=True).fillna(0.0).mean(axis=1)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    return combined, pd.DataFrame(detail)


# ---------------------------------------------------------------------------
# Strategy 2 — Beta drift on SPX-NAS
# ---------------------------------------------------------------------------

def strat_beta_drift():
    """Beta(NAS, SPX): 60d rolling. When |60d beta - 252d avg beta| > 0.3,
    expect mean reversion of beta. If 60d beta > 252d → NAS has been
    *too sensitive* → bet on lower future NAS-SPX dispersion: short
    NAS-vs-SPX (long SPX, short NAS) scaled by relative-beta sign."""
    print("\n=== Strategy 2: Beta drift SPX/NAS ===")
    panel = close_panel(["SPX500_USD", "NAS100_USD"])
    spx = panel["SPX500_USD"].pct_change()
    nas = panel["NAS100_USD"].pct_change()

    # Rolling 60d beta of NAS on SPX
    cov60 = nas.rolling(60).cov(spx)
    var60 = spx.rolling(60).var()
    beta60 = cov60 / var60
    beta252 = beta60.rolling(252).mean()
    # Observable at start of bar t
    drift = (beta60.shift(1) - beta252.shift(1))

    # When drift > +0.3 -> beta inflated -> short NAS / long SPX (sign = -1 on NAS)
    # When drift < -0.3 -> beta deflated -> long NAS / short SPX (sign = +1 on NAS)
    sig_nas = np.where(drift > 0.3, -1.0, np.where(drift < -0.3, +1.0, 0.0))
    sig_nas = pd.Series(sig_nas, index=panel.index)

    # Hold while |drift| > 0.15 (hysteresis); else flat
    held = sig_nas.copy()
    cur = 0.0
    for i in range(len(held)):
        d = drift.iloc[i]
        if pd.isna(d):
            cur = 0.0
        elif cur == 0.0 and abs(d) > 0.3:
            cur = float(sig_nas.iloc[i])
        elif cur != 0.0 and abs(d) < 0.15:
            cur = 0.0
        held.iloc[i] = cur

    n_active = int((held != 0).sum())

    # Position NAS = held, SPX = -held (relative trade)
    streams = []
    for sym, sgn in [("NAS100_USD", +1), ("SPX500_USD", -1)]:
        df = DATA_D1[sym]
        td = trading_date(df)
        p = (held * sgn).reindex(pd.DatetimeIndex(td)).fillna(0.0)
        pos = pd.Series(p.values, index=df.index)
        rets, _ = run_position(df, pos, name=f"BETA_DRIFT_{sym}", symbol=sym)
        streams.append(rets)
    combined = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    s = stats("BETA_DRIFT", combined)
    print(f"  active_bars={n_active}  IS={s['IS_sharpe']:+.2f}  "
          f"OOS={s['OOS_sharpe']:+.2f}  2022={s['Y2022_sharpe']:+.2f}")
    return combined


# ---------------------------------------------------------------------------
# Strategy 3 — Cumulative residual z-score fade
# ---------------------------------------------------------------------------

def strat_cumres_zscore_fade():
    """Per symbol: regress log-price on time in 90d window. Compute residuals.
    Take cumulative sum of residuals over 90d. Z-score that against
    expanding history; |z| > 2.5 → fade the direction of cumulative dev.
    Hold 5 bars."""
    print("\n=== Strategy 3: Cumulative residual z-score fade ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        close = df["close"].astype("float64")
        log_p = np.log(close)
        n = len(df)
        cum_res = np.full(n, np.nan)
        # rolling 90d OLS via efficient computation
        w = 90
        # use rolling apply for a simple linear residual: detrend via
        # x - mean - slope*(t - t_mean). Slope = cov(x,t)/var(t).
        if n >= w:
            t = np.arange(n, dtype="float64")
            for i in range(w - 1, n):
                lo = i - w + 1
                xs = log_p.values[lo:i + 1]
                ts = t[lo:i + 1]
                tm = ts.mean()
                xm = xs.mean()
                tc = ts - tm
                xc = xs - xm
                denom = (tc * tc).sum()
                if denom <= 0:
                    continue
                slope = (tc * xc).sum() / denom
                resid = xc - slope * tc
                cum_res[i] = float(resid.sum())
        cum_res = pd.Series(cum_res, index=df.index)
        # Expanding mean/std of cum_res
        exp_mu = cum_res.expanding(min_periods=200).mean().shift(1)
        exp_sd = cum_res.expanding(min_periods=200).std().shift(1)
        z = (cum_res.shift(1) - exp_mu) / exp_sd

        # Fade: sign opposite to z. If z > 2.5 → short (price drifted up); if
        # z < -2.5 → long.
        trig = pd.Series(np.where(z > 2.5, -1.0,
                                  np.where(z < -2.5, +1.0, 0.0)),
                         index=df.index)

        # Hold for 5 bars from trigger
        pos = pd.Series(0.0, index=df.index)
        hold = 5
        bars_left = 0
        cur = 0.0
        for i in range(len(df)):
            if bars_left > 0:
                pos.iloc[i] = cur
                bars_left -= 1
            else:
                t = trig.iloc[i]
                if t != 0 and not pd.isna(t):
                    cur = float(t)
                    pos.iloc[i] = cur
                    bars_left = hold - 1
        rets, _ = run_position(df, pos, name=f"CUMRES_{sym}", symbol=sym)
        s = stats(f"CUMRES_{sym}", rets)
        streams[sym] = rets
        detail.append({
            "symbol": sym,
            "events": int((trig != 0).sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} events={int((trig != 0).sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    if surv:
        panel = pd.concat({s: streams[s] for s in surv}, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    else:
        all_panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        combo = all_panel.mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 4 — Cross-sectional return outlier (short 2 days)
# ---------------------------------------------------------------------------

def strat_xs_outlier():
    """Each day, compute z-score of each symbol's return vs cross-section
    median (using cross-section median + MAD as scale). Identify max-|z|.
    Short the outlier symbol in the direction of its return for 2 days
    (extreme moves mean-revert). Both directions: long if move was down,
    short if move was up."""
    print("\n=== Strategy 4: Cross-sectional return outlier ===")
    panel = close_panel(ALL_D1)
    rets_panel = panel.pct_change()

    xs_med = rets_panel.median(axis=1)
    # MAD per row (robust scale)
    abs_dev = (rets_panel.sub(xs_med, axis=0)).abs()
    mad = abs_dev.median(axis=1).replace(0, np.nan)
    z = rets_panel.sub(xs_med, axis=0).div(mad * 1.4826, axis=0)

    # Identify per-row max-|z| symbol
    # we want trades observable at start of t+1 (use today's z, hold during t+1, t+2)
    # Find the column with the largest |z| per row
    abs_z = z.abs()
    # rank to find argmax (skipping all-NA rows)
    any_valid = abs_z.notna().any(axis=1)
    argmax_col = pd.Series(index=abs_z.index, dtype=object)
    if any_valid.any():
        argmax_col.loc[any_valid] = abs_z.loc[any_valid].idxmax(axis=1)
    # And whether |z| > 2 (require minimum extremity)
    valid = (abs_z.max(axis=1) > 2.0).fillna(False)

    # Build per-symbol short-2-day position
    sym_pos = {s: pd.Series(0.0, index=panel.index) for s in ALL_D1}
    # When valid[i] is True, for symbol argmax_col[i], set
    # pos[i+1] = -sign(z[i, sym]) and pos[i+2] = same.
    sign_z = np.sign(z.fillna(0.0))
    idx_list = panel.index
    n = len(idx_list)
    events = 0
    for i in range(n):
        if not valid.iloc[i]:
            continue
        sym = argmax_col.iloc[i]
        if sym not in sym_pos:
            continue
        sgn = float(sign_z.iloc[i].loc[sym])
        if sgn == 0:
            continue
        # short the outlier in its direction → position = -sgn
        for k in (1, 2):
            j = i + k
            if j < n:
                # accumulate (in case multiple events overlap, prefer latest)
                sym_pos[sym].iloc[j] = -sgn
        events += 1

    # Backtest per symbol
    streams = []
    for sym in ALL_D1:
        df = DATA_D1[sym]
        td = trading_date(df)
        p = sym_pos[sym].reindex(pd.DatetimeIndex(td)).fillna(0.0)
        pos = pd.Series(p.values, index=df.index)
        rets, _ = run_position(df, pos, name=f"XS_OUT_{sym}", symbol=sym)
        streams.append(rets)
    combined = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    s = stats("XS_OUTLIER", combined)
    print(f"  events={events}  IS={s['IS_sharpe']:+.2f}  "
          f"OOS={s['OOS_sharpe']:+.2f}  2022={s['Y2022_sharpe']:+.2f}")
    return combined


# ---------------------------------------------------------------------------
# Strategy 5 — Vol regime fade (calendar-week relative vol)
# ---------------------------------------------------------------------------

def strat_vol_regime_fade():
    """Per symbol: compute 5d realized vol. For each calendar ISO week,
    track historical mean 5d-vol. Ratio = current 5d-vol / historical-week-
    avg. Top 5% → short vol via fading the recent direction (inverse-vol
    sized) for 3 days."""
    print("\n=== Strategy 5: Vol regime fade ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        close = df["close"].astype("float64")
        ret = close.pct_change()
        vol5 = ret.rolling(5).std()
        ts = pd.to_datetime(df["timestamp"], utc=True)
        iso_week = ts.dt.isocalendar().week

        # Walk-forward calendar-week historical mean of 5d-vol
        # For each bar i: take mean of vol5[j] for j<i where iso_week[j]==iso_week[i].
        # Vectorize via expanding groupby trick: cum_sum and cum_count.
        week_arr = iso_week.values
        vol_arr = vol5.values
        # Build expanding sum/count by week
        sums = np.zeros(54, dtype="float64")
        cnts = np.zeros(54, dtype="int64")
        week_mean = np.full(len(df), np.nan)
        for i in range(len(df)):
            w = int(week_arr[i])
            if cnts[w] > 0:
                week_mean[i] = sums[w] / cnts[w]
            v = vol_arr[i]
            if not np.isnan(v):
                sums[w] += v
                cnts[w] += 1
        week_mean = pd.Series(week_mean, index=df.index)

        # Ratio at start of bar (shift(1) for vol5)
        ratio = vol5.shift(1) / week_mean.shift(1)
        # Top 5% of historical ratio distribution → walk-forward quantile
        exp_q = ratio.expanding(min_periods=200).quantile(0.95).shift(1)
        trig = (ratio > exp_q).fillna(False)

        # Fade direction: -sign of recent 5d cumulative return.
        sign_recent = np.sign(close.pct_change(5).shift(1))
        # Inverse-vol size: 1/vol5 normalized by expanding median
        inv_vol = (1.0 / vol5.replace(0, np.nan)).shift(1)
        # Cap with expanding-median rescale so units ≈ 1
        inv_med = inv_vol.expanding(min_periods=60).median().shift(1)
        size = (inv_vol / inv_med).clip(0.25, 2.0).fillna(0.0)

        raw = -sign_recent * size  # fade direction sized inversely to vol
        # Trigger position for 3 bars
        pos = pd.Series(0.0, index=df.index)
        bars_left = 0
        cur = 0.0
        for i in range(len(df)):
            if trig.iloc[i] and not pd.isna(raw.iloc[i]) and raw.iloc[i] != 0:
                cur = float(raw.iloc[i])
                bars_left = 3
            if bars_left > 0:
                pos.iloc[i] = cur
                bars_left -= 1
                if bars_left == 0:
                    cur = 0.0
        rets, _ = run_position(df, pos, name=f"VOLREG_{sym}", symbol=sym)
        s = stats(f"VOLREG_{sym}", rets)
        streams[sym] = rets
        detail.append({
            "symbol": sym, "events": int(trig.sum()),
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} events={int(trig.sum()):3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    if surv:
        panel = pd.concat({s: streams[s] for s in surv}, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    else:
        panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 6 — Range expansion + close-direction reversal fade
# ---------------------------------------------------------------------------

def strat_range_expansion_fade():
    """When D1 range > 3× 20d avg range AND close is in extreme half of range
    (relative to its midpoint), fade for 2 days."""
    print("\n=== Strategy 6: Range expansion + close-direction fade ===")
    streams = {}
    detail = []
    for sym in ALL_D1:
        df = DATA_D1[sym].copy()
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        close = df["close"].astype("float64")
        rng = (high - low)
        avg20 = rng.rolling(20).mean().shift(1)
        big = (rng > 3.0 * avg20).fillna(False)

        mid = (high + low) / 2.0
        # close in extreme half: True if close > 0.75 of range from low or
        # close < 0.25. Encode position-in-range:
        pir = (close - low) / (high - low).replace(0, np.nan)
        extreme_up = pir > 0.75
        extreme_dn = pir < 0.25

        # Fade direction: if close was high in range -> short next 2 bars;
        # if low -> long next 2 bars.
        sgn_today = pd.Series(0.0, index=df.index)
        sgn_today[big & extreme_up] = -1.0
        sgn_today[big & extreme_dn] = +1.0

        # Build position[t+1], pos[t+2]
        pos = pd.Series(0.0, index=df.index)
        events = 0
        for i in range(len(df)):
            s_t = sgn_today.iloc[i]
            if s_t != 0:
                events += 1
                for k in (1, 2):
                    j = i + k
                    if j < len(df):
                        pos.iloc[j] = float(s_t)
        rets, _ = run_position(df, pos, name=f"RNG_EXP_{sym}", symbol=sym)
        s = stats(f"RNG_EXP_{sym}", rets)
        streams[sym] = rets
        detail.append({
            "symbol": sym, "events": events,
            "IS_sharpe": s["IS_sharpe"], "OOS_sharpe": s["OOS_sharpe"],
            "FULL_sharpe": s["FULL_sharpe"], "Y2022_sharpe": s["Y2022_sharpe"],
        })
        print(f"  {sym:<12} events={events:3d}  "
              f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
              f"2022={s['Y2022_sharpe']:+.2f}")

    det = pd.DataFrame(detail)
    surv = det[det["IS_sharpe"] >= 0.5]["symbol"].tolist()
    if surv:
        panel = pd.concat({s: streams[s] for s in surv}, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    else:
        panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        combo = panel.mean(axis=1)
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    print(f"  -> IS survivors: {surv}")
    return combo, det


# ---------------------------------------------------------------------------
# Strategy 7 — Asset-class outlier rotation (lean into calmest basket)
# ---------------------------------------------------------------------------

def strat_asset_class_calm():
    """Weekly: compute each basket's 21d std of basket-average-return / its
    252d-rolling std. Lean LONG into the basket with lowest ratio (calmest
    relative to its own history) and short the basket with highest ratio.

    Direction: classical "calm = continuation" — long calmest basket
    (mean of its constituents), short chaos basket. Carry weekly."""
    print("\n=== Strategy 7: Asset-class calm rotation ===")
    baskets = {
        "crypto": CRYPTO_SYMS,
        "fx": FX_SYMS,
        "index": INDEX_SYMS,
    }

    # Build basket-average-return panel
    panel_close = close_panel(ALL_D1)
    panel_ret = panel_close.pct_change()
    basket_ret = pd.DataFrame(index=panel_ret.index)
    for bn, syms in baskets.items():
        basket_ret[bn] = panel_ret[syms].mean(axis=1)

    std21 = basket_ret.rolling(21).std()
    std252 = basket_ret.rolling(252).std()
    ratio = (std21 / std252).shift(1)  # observable at bar start

    # Each Monday-equivalent (weekly), choose winners.
    # Use ISO-week change to mark rebalance bars
    iso_yw = panel_ret.index.to_series().apply(
        lambda t: f"{t.isocalendar().year}-{t.isocalendar().week:02d}")
    is_new_week = iso_yw != iso_yw.shift(1)

    # Position vector per basket per bar — persist between rebal bars
    basket_names = list(baskets.keys())
    pos_arr = np.zeros((len(panel_ret), len(basket_names)), dtype="float64")
    last_alloc = {b: 0.0 for b in basket_names}
    for i in range(len(panel_ret)):
        if is_new_week.iloc[i]:
            row = ratio.iloc[i]
            if row.notna().all():
                calm = row.idxmin()
                chaos = row.idxmax()
                last_alloc = {b: 0.0 for b in basket_names}
                last_alloc[calm] = +1.0
                last_alloc[chaos] = -1.0
        for j, b in enumerate(basket_names):
            pos_arr[i, j] = last_alloc[b]
    pos_basket = pd.DataFrame(pos_arr, index=panel_ret.index, columns=basket_names)

    # Convert basket positions to per-symbol positions (equal-weight within basket)
    sym_pos = {s: pd.Series(0.0, index=panel_ret.index) for s in ALL_D1}
    for bn, syms in baskets.items():
        w = 1.0 / len(syms)
        for sym in syms:
            sym_pos[sym] = pos_basket[bn] * w

    # Backtest per symbol
    streams = []
    for sym in ALL_D1:
        df = DATA_D1[sym]
        td = trading_date(df)
        p = sym_pos[sym].reindex(pd.DatetimeIndex(td)).fillna(0.0)
        pos = pd.Series(p.values, index=df.index)
        rets, _ = run_position(df, pos, name=f"AC_CALM_{sym}", symbol=sym)
        streams.append(rets)
    combined = pd.concat(streams, axis=1, sort=True).fillna(0.0).sum(axis=1)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    s = stats("AC_CALM", combined)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}")
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_sleeves = {}

    rets1, det1 = strat_corr_breakdown()
    raw_sleeves["CORR_BREAKDOWN"] = rets1

    rets2 = strat_beta_drift()
    raw_sleeves["BETA_DRIFT_SPX_NAS"] = rets2

    rets3, det3 = strat_cumres_zscore_fade()
    raw_sleeves["CUMRES_ZSCORE_FADE"] = rets3

    rets4 = strat_xs_outlier()
    raw_sleeves["XS_OUTLIER_2D"] = rets4

    rets5, det5 = strat_vol_regime_fade()
    raw_sleeves["VOL_REGIME_FADE"] = rets5

    rets6, det6 = strat_range_expansion_fade()
    raw_sleeves["RANGE_EXPANSION_FADE"] = rets6

    rets7 = strat_asset_class_calm()
    raw_sleeves["ASSET_CLASS_CALM"] = rets7

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
    breakdown.to_csv(OUT / "anomaly_breakdown.csv", index=False)

    print("\n=== Breakdown ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "FULL_sharpe",
                     "FULL_ret", "FULL_vol", "scale"]].to_string(index=False))

    # Survivor filter: IS >= 0.5 AND OOS >= 0
    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.5)
                          & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) ===")
    if not survivors.empty:
        print(survivors[["sleeve", "IS_sharpe", "OOS_sharpe",
                         "Y2022_sharpe", "FULL_sharpe",
                         "FULL_ret"]].to_string(index=False))
    else:
        print("(no sleeves passed both gates)")

    surv_names = survivors["sleeve"].tolist()

    # Panel
    panel_df = pd.concat(scaled, axis=1, sort=True).fillna(0.0)
    if panel_df.index.tz is None:
        panel_df.index = panel_df.index.tz_localize("UTC")
    if surv_names:
        panel_df["survivors_mean"] = panel_df[surv_names].mean(axis=1)
    else:
        panel_df["survivors_mean"] = 0.0
    out_df = panel_df.reset_index().rename(columns={"index": "timestamp"})
    out_df.to_parquet(OUT / "anomaly_returns.parquet", index=False)

    # Combined survivor stats
    if surv_names:
        combined = panel_df[surv_names].mean(axis=1)
    else:
        combined = panel_df["survivors_mean"]
    sc = stats("ANOMALY_COMBINED", combined)
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
