"""Intraday strategies sleeve (H1 / M15).

Explores 4 strategy families. Survivors must clear IS Sharpe >= 0.5 AND
OOS Sharpe >= 0.0 (the IS/OOS split is 2024-01-01). Costs are charged per
bar by alphabeta.backtest at the per-side rate for the asset class; the
intraday horizons make those costs particularly painful, which is the
whole point of the filter.

Families:
  1. Opening-range breakout (ORB) on H1 for the 6 index symbols.
  2. Intraday momentum on M15 for BTC / ETH / SOL (lookbacks 16 / 32 / 64).
  3. RSI(14) on H1 across all 13 symbols, classic 30/70 extremes with 50 exit.
  4. Asia-session mean-reversion on crypto M15 (00-08 UTC reverts the
     prior US 12-20 UTC move).

The survivors are vol-scaled (IS=10% ann) within their family, family
mean is taken, then the combined sleeve is rescaled to 5% IS ann vol so
it slots into the master portfolio alongside the other 5%-vol sleeves.

Outputs:
  scratch/quant/intraday_returns.parquet    -- D1 sleeve returns (timestamp UTC, ret)
  scratch/quant/intraday_breakdown.csv      -- per-strategy IS/OOS/2022/FULL Sharpe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, CRYPTO, INDEX, ALL_SYMBOLS, SYMBOL_TYPE
from alphabeta.backtest import backtest, split_is_oos


# -- knobs --------------------------------------------------------------------
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
FAMILY_TARGET_VOL = 0.10        # per-family IS target
SLEEVE_TARGET_VOL = 0.05        # combined sleeve IS target (matches the rest)
POS_CAP = 3.0
MIN_IS_SHARPE = 0.5             # survival filter
MIN_OOS_SHARPE = 0.0
OUT_DIR = Path(__file__).resolve().parent


# Index local session "open" hours in UTC. Convention: the opening-range bar
# is the one whose `timestamp` equals this hour (OHLC for the hour starting
# at this label). Cash equity opens are 14:30 UTC for US; the 14:00 H1 bar
# covers 14:00-15:00 and captures the open + first 30 min of price action.
INDEX_SESSIONS = {
    # symbol     : (open_hour_utc, close_hour_utc)
    "SPX500_USD" : (14, 20),    # cash 14:30-21:00 → bars 14..20
    "NAS100_USD" : (14, 20),
    "US30_USD"   : (14, 20),
    "DE30_EUR"   : (8, 15),     # XETRA 09:00 CET → 08:00 UTC; close 16:30 CET ~ 15:00 UTC
    "UK100_GBP"  : (8, 15),     # LSE 08:00 UTC → 15:30 UTC
    "JP225_USD"  : (0, 5),      # Tokyo 09:00 JST = 00:00 UTC → close 15:00 JST = 06:00 UTC
}


# -- helpers ------------------------------------------------------------------
def _bpy_idx(idx: pd.DatetimeIndex) -> float:
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats_for(label: str, r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {"label": label}
    out = {"label": label}
    for tag, mask in [
        ("FULL", pd.Series(True, index=r.index)),
        ("IS",   r.index < SPLIT),
        ("OOS",  r.index >= SPLIT),
    ]:
        sub = r[mask] if isinstance(mask, pd.Series) else r[mask]
        if len(sub) < 2 or sub.std(ddof=0) == 0:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            out[f"{tag}_vol"] = 0.0
            out[f"{tag}_dd"] = 0.0
            continue
        bpy = _bpy_idx(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    # 2022 Sharpe
    y22 = r[r.index.year == 2022]
    if len(y22) > 1 and y22.std(ddof=0) > 0:
        bpy = _bpy_idx(y22.index)
        out["Y2022_sharpe"] = float(y22.mean() * bpy / (y22.std(ddof=0) * np.sqrt(bpy)))
    else:
        out["Y2022_sharpe"] = np.nan
    return out


def to_utc_series(rets: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Build a tz-aware UTC indexed series from raw returns + a timestamp col."""
    idx = pd.to_datetime(timestamps, utc=True)
    out = pd.Series(rets.values, index=idx, name="ret").sort_index()
    # dedupe (shouldn't happen but be safe)
    out = out[~out.index.duplicated(keep="last")]
    return out


def daily_collapse(s: pd.Series) -> pd.Series:
    """Sum intraday returns to one row per UTC calendar day."""
    s = s.copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    d = s.groupby(s.index.floor("D")).sum()
    d.index = pd.to_datetime(d.index, utc=True)
    d.name = "ret"
    return d


def vol_scale_is(s: pd.Series, target_vol: float) -> pd.Series:
    """Scale `s` (per-bar returns w/ tz-aware index) so IS ann vol == target."""
    is_part = s[s.index < SPLIT].dropna()
    if len(is_part) < 30 or is_part.std(ddof=0) == 0:
        return s * 0.0
    bpy = _bpy_idx(is_part.index)
    av = float(is_part.std(ddof=0)) * np.sqrt(bpy)
    if av <= 1e-9:
        return s * 0.0
    return s * (target_vol / av)


# =============================================================================
# Gross / net helper: same backtest path, allows cost_per_side override.
# =============================================================================
def _gross_net_streams(df: pd.DataFrame, pos: pd.Series, sym: str, tf: str, name: str) -> tuple[pd.Series, pd.Series, dict, dict]:
    """Run two backtests (gross and net) and return tz-UTC per-bar return
    series for each, plus stat dicts. Used so we can record cost drag."""
    from alphabeta.backtest import cost_for as _cps
    cps = _cps(sym)
    net = backtest(df, pos, symbol=sym, timeframe=tf, name=name)
    gross = backtest(df, pos, symbol=sym, timeframe=tf, name=name + "_gross", cost_per_side=0.0)
    g = to_utc_series(gross.returns, df["timestamp"])
    n = to_utc_series(net.returns, df["timestamp"])
    return g, n, gross.stats, net.stats


# =============================================================================
# Family 1: Opening-range breakout (ORB) on H1, index symbols
# =============================================================================
def orb_position(df: pd.DataFrame, open_hr: int, close_hr: int) -> pd.Series:
    """For each session day, look at the bar whose hour == open_hr (the
    "opening range" bar). Next bar: if its close > opening bar's high, go
    long; if < low, go short. Hold until the close_hr bar (inclusive), then
    flat. Position is computed strictly from prior-bar info (we mark the
    signal *one bar after* the opening bar; the engine uses position[t] for
    bar t's return, so we must set position[t] using info <= t-1).
    """
    n = len(df)
    pos = np.zeros(n, dtype="float64")
    ts = df["timestamp"]
    hours = ts.dt.hour.to_numpy()
    dates = ts.dt.date.to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    # Identify, for each day, the index of the open_hr bar.
    # We iterate sessionwise. For speed we precompute lookup: dict day->open_idx.
    open_idx_by_day: dict = {}
    for i in range(n):
        if hours[i] == open_hr:
            open_idx_by_day.setdefault(dates[i], i)

    # For each day with an open bar present:
    for day, oi in open_idx_by_day.items():
        # Decision bar is oi+1: at the START of bar oi+1 we know close[oi].
        # So position[oi+1] is set, then we hold to the bar whose hour ==
        # close_hr on the same day, inclusive. Position is exited (=0) on the
        # bar AFTER close_hr.
        signal_start = oi + 1
        if signal_start >= n:
            continue
        # If decision bar isn't same date (gap), skip.
        if dates[signal_start] != day:
            continue
        # Decide direction from bar oi+1's *open* relative to the OR. But
        # rules say "next bar closes above/below" → we use close[oi+1] to
        # decide direction, hold from oi+2 onward. Engine convention: pos[t]
        # uses info <= t-1. So set pos[oi+2 .. close_idx] = sign.
        o_high = highs[oi]
        o_low = lows[oi]
        decision_bar = signal_start
        decision_close = closes[decision_bar]
        if decision_close > o_high:
            direction = 1.0
        elif decision_close < o_low:
            direction = -1.0
        else:
            direction = 0.0
        if direction == 0.0:
            continue
        # Find the close_hr bar idx on the same day.
        # Walk forward until hour == close_hr OR day changes.
        close_idx = None
        j = decision_bar + 1
        while j < n and dates[j] == day:
            if hours[j] == close_hr:
                close_idx = j
                break
            j += 1
        if close_idx is None:
            # No close bar today; just hold to end-of-day
            close_idx = j - 1 if j > decision_bar + 1 else decision_bar + 1
        # Set position from (decision_bar+1) through close_idx inclusive.
        start = decision_bar + 1
        end = min(close_idx + 1, n)
        if start < n:
            pos[start:end] = direction
    return pd.Series(pos, index=df.index)


def run_family_orb() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Returns: ({symbol: net stream}, {symbol: gross stream}) — both tz-UTC indexed."""
    net_streams: dict[str, pd.Series] = {}
    gross_streams: dict[str, pd.Series] = {}
    for sym, (oh, ch) in INDEX_SESSIONS.items():
        df = get_candles(sym, "H1")
        pos = orb_position(df, oh, ch)
        g, n, _, _ = _gross_net_streams(df, pos, sym, "H1", f"ORB_{sym}")
        net_streams[sym] = n
        gross_streams[sym] = g
    return net_streams, gross_streams


# =============================================================================
# Family 2: Intraday momentum on crypto, M15
# =============================================================================
def momentum_position(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Position = sign of trailing N-bar log return, held for lookback//2 bars.

    Implementation: sample the trailing-return-sign every `hold = lookback//2`
    bars and hold the position constant between samples. This keeps the
    turnover at ~ (1 / hold) round-trips per bar instead of recomputing every
    bar — i.e. a true "look, decide, hold" rule.

    Shifted by 1 to avoid look-ahead (engine convention).
    """
    close = df["close"].astype("float64").to_numpy()
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    s = pd.Series(log_ret, index=df.index)
    trail = s.rolling(lookback, min_periods=lookback).sum().to_numpy()
    sig = np.sign(trail)
    hold = max(1, lookback // 2)
    n = len(df)
    pos = np.zeros(n, dtype="float64")
    # We "look" every `hold` bars starting at index `lookback`.
    current = 0.0
    for i in range(n):
        if i >= lookback and (i - lookback) % hold == 0:
            x = sig[i]
            if not np.isnan(x):
                current = float(x)
        pos[i] = current
    # Shift by 1 — engine reads pos[t] against bar-t return; we must use only
    # information available at start-of-bar t (i.e. up to t-1).
    s_pos = pd.Series(pos, index=df.index).shift(1).fillna(0.0).clip(-POS_CAP, POS_CAP)
    return s_pos


def run_family_momentum() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """One stream per symbol = mean of net returns across L in {16,32,64}.
    We also return the per-symbol gross-stream sibling (mean across lookbacks)."""
    net_streams: dict[str, pd.Series] = {}
    gross_streams: dict[str, pd.Series] = {}
    lookbacks = [16, 32, 64]
    for sym in CRYPTO:
        df = get_candles(sym, "M15")
        per_lb_net = []
        per_lb_gross = []
        for L in lookbacks:
            pos = momentum_position(df, L)
            g, n, _, _ = _gross_net_streams(df, pos, sym, "M15", f"MOM_{sym}_{L}")
            per_lb_net.append(n)
            per_lb_gross.append(g)
        net_streams[sym] = pd.concat(per_lb_net, axis=1).mean(axis=1).rename("ret")
        gross_streams[sym] = pd.concat(per_lb_gross, axis=1).mean(axis=1).rename("ret")
    return net_streams, gross_streams


# =============================================================================
# Family 3: RSI(14) extremes on H1
# =============================================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    # Wilder's smoothing ≈ EMA with alpha = 1/period
    ru = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rd = dn.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = ru / rd.replace(0.0, np.nan)
    r = 100.0 - 100.0 / (1.0 + rs)
    return r.fillna(50.0)


def rsi_position(df: pd.DataFrame) -> pd.Series:
    """Long when RSI<30 until RSI>50; short when RSI>70 until RSI<50.
    State machine on the *prior* RSI value to avoid look-ahead."""
    r = rsi(df["close"].astype("float64"), 14)
    r_prev = r.shift(1)  # position[t] uses RSI at bar t-1
    n = len(df)
    pos = np.zeros(n, dtype="float64")
    state = 0.0
    rp = r_prev.to_numpy()
    for i in range(n):
        x = rp[i]
        if np.isnan(x):
            pos[i] = 0.0
            continue
        if state == 0.0:
            if x < 30.0:
                state = 1.0
            elif x > 70.0:
                state = -1.0
        elif state > 0.0:
            if x > 50.0:
                state = 0.0
        elif state < 0.0:
            if x < 50.0:
                state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def run_family_rsi() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    net_streams: dict[str, pd.Series] = {}
    gross_streams: dict[str, pd.Series] = {}
    for sym in ALL_SYMBOLS:
        df = get_candles(sym, "H1")
        pos = rsi_position(df)
        g, n, _, _ = _gross_net_streams(df, pos, sym, "H1", f"RSI_{sym}")
        net_streams[sym] = n
        gross_streams[sym] = g
    return net_streams, gross_streams


# =============================================================================
# Family 4: Asia-session crypto reversal
# =============================================================================
def asia_reversal_position(df: pd.DataFrame) -> pd.Series:
    """On M15: at the start of the Asia session each day (00:00 UTC), take a
    position OPPOSITE to the prior day's US-session move (12-20 UTC, computed
    on the *previous* calendar day). Hold through Asia session (00-08 UTC),
    flat for rest of day.

    Strictly walk-forward: the US move is fully observed by 20:00 UTC on day
    D-1, so the 00:00 UTC bar on day D can use it.
    """
    n = len(df)
    pos = np.zeros(n, dtype="float64")
    ts = df["timestamp"]
    hours = ts.dt.hour.to_numpy()
    dates = ts.dt.date.to_numpy()
    closes = df["close"].astype("float64").to_numpy()
    # Build per-day US-session move using close at 20:00 UTC vs close at 12:00 UTC.
    # We'll do a one-pass scan.
    # First, for each day, find idx of 12:00 (first M15 bar in that hour) and 20:00 (last).
    day_to_us_open: dict = {}
    day_to_us_close: dict = {}
    for i in range(n):
        d = dates[i]
        h = hours[i]
        if h == 12:
            day_to_us_open.setdefault(d, i)
        if h == 20 or (h > 20 and d not in day_to_us_close):
            # we want the last bar inside [12, 20] window — use 20-hour bars
            pass
        if h == 20:
            day_to_us_close[d] = i  # overwrite, keep last 20-hour bar
    # Compute prior US move per day
    sorted_days = sorted(set(dates))
    us_move = {}
    for d in sorted_days:
        if d in day_to_us_open and d in day_to_us_close:
            oi = day_to_us_open[d]
            ci = day_to_us_close[d]
            if ci > oi and closes[oi] > 0:
                us_move[d] = (closes[ci] / closes[oi]) - 1.0
    # Now apply: on day D, for bars in 00:00-07:45 UTC, position = -sign(us_move[D-1]).
    # We need to map prior trading day quickly.
    day_idx = {d: k for k, d in enumerate(sorted_days)}
    for i in range(n):
        d = dates[i]
        h = hours[i]
        if 0 <= h < 8:
            k = day_idx.get(d)
            if k is None or k == 0:
                continue
            prior = sorted_days[k - 1]
            if prior in us_move:
                direction = -np.sign(us_move[prior])
                pos[i] = direction
    # Shift by 1 (engine convention: pos[t] uses <= t-1 info). For 00:00 bar,
    # the prior US session ends at 20:00 the prior day; that's already strictly
    # before 00:00, so no shift needed *for the signal itself*. But the engine
    # multiplies pos[t] against close.pct_change[t], so to be safe we shift 1.
    s = pd.Series(pos, index=df.index).shift(1).fillna(0.0)
    return s


def run_family_asia() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    net_streams: dict[str, pd.Series] = {}
    gross_streams: dict[str, pd.Series] = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        df = get_candles(sym, "M15")
        pos = asia_reversal_position(df)
        g, n, _, _ = _gross_net_streams(df, pos, sym, "M15", f"ASIA_REV_{sym}")
        net_streams[sym] = n
        gross_streams[sym] = g
    return net_streams, gross_streams


# =============================================================================
# Driver
# =============================================================================
def survives(stats: dict) -> bool:
    return (
        stats.get("IS_sharpe", 0.0) >= MIN_IS_SHARPE
        and stats.get("OOS_sharpe", 0.0) >= MIN_OOS_SHARPE
    )


def build_family_stream(
    net_by_name: dict[str, pd.Series],
    gross_by_name: dict[str, pd.Series],
    family_label: str,
) -> tuple[pd.Series | None, list[dict], list[dict]]:
    """Compute per-strategy stats (net + gross context), filter survivors by
    net IS/OOS gates, vol-scale each survivor, equal-weight average within
    family. Returns (family_daily_stream, all_breakdown_rows, surviving_rows).
    """
    all_rows = []
    survivors: list[tuple[str, pd.Series]] = []
    for name, s in net_by_name.items():
        d = daily_collapse(s)
        st = stats_for(name, d)
        # Attach gross IS Sharpe for context (lets us see whether the strategy
        # has gross edge that got eaten by costs vs no edge at all).
        if name in gross_by_name:
            g = daily_collapse(gross_by_name[name])
            gross_st = stats_for(name + "_gross", g)
            st["gross_IS_sharpe"] = gross_st.get("IS_sharpe", 0.0)
            st["gross_OOS_sharpe"] = gross_st.get("OOS_sharpe", 0.0)
            st["gross_FULL_sharpe"] = gross_st.get("FULL_sharpe", 0.0)
        st["family"] = family_label
        st["survived"] = bool(
            st.get("IS_sharpe", 0.0) >= MIN_IS_SHARPE
            and st.get("OOS_sharpe", 0.0) >= MIN_OOS_SHARPE
        )
        all_rows.append(st)
        if st["survived"]:
            scaled = vol_scale_is(d, FAMILY_TARGET_VOL)
            survivors.append((name, scaled))

    if not survivors:
        return None, all_rows, []

    mat = pd.concat([s.rename(n) for n, s in survivors], axis=1).fillna(0.0)
    fam_stream = mat.mean(axis=1)
    fam_stream.name = f"FAM_{family_label}"
    surv_rows = [r for r in all_rows if r["survived"]]
    return fam_stream, all_rows, surv_rows


def main() -> None:
    print("=" * 70)
    print("INTRADAY SLEEVE — H1/M15 strategies")
    print("=" * 70)

    # ---- Run all four families ---------------------------------------------
    print("\n[1/4] ORB on index H1...")
    orb_net, orb_gross = run_family_orb()

    print("[2/4] Crypto momentum M15...")
    mom_net, mom_gross = run_family_momentum()

    print("[3/4] RSI(14) extremes H1...")
    rsi_net, rsi_gross = run_family_rsi()

    print("[4/4] Asia-session crypto reversal M15...")
    asia_net, asia_gross = run_family_asia()

    # ---- Per-family processing ---------------------------------------------
    fam_streams: dict[str, pd.Series] = {}
    breakdown_rows: list[dict] = []
    fam_summary_rows: list[dict] = []

    for fam_label, fam_net, fam_gross in [
        ("ORB",       orb_net,  orb_gross),
        ("MOM",       mom_net,  mom_gross),
        ("RSI",       rsi_net,  rsi_gross),
        ("ASIA_REV",  asia_net, asia_gross),
    ]:
        fam_stream, rows, surv = build_family_stream(fam_net, fam_gross, fam_label)
        breakdown_rows.extend(rows)
        n_surv = len(surv)
        n_total = len(fam_net)
        # Compute family-level NET stats by averaging the per-strategy net
        # streams (no survival filtering): this lets us compare family NET
        # to family GROSS even when no strategy individually survives.
        net_panel = pd.concat(
            [daily_collapse(s).rename(n) for n, s in fam_net.items()],
            axis=1, sort=True,
        ).fillna(0.0)
        fam_net_avg = net_panel.mean(axis=1)
        fam_st = stats_for(fam_label, fam_net_avg)
        fam_st["family"] = fam_label
        fam_st["n_total"] = n_total
        fam_st["n_survivors"] = n_surv

        gross_panel = pd.concat(
            [daily_collapse(s).rename(n) for n, s in fam_gross.items()],
            axis=1, sort=True,
        ).fillna(0.0)
        fam_gross_avg = gross_panel.mean(axis=1)
        fam_gross_st = stats_for(fam_label + "_GROSS", fam_gross_avg)
        fam_st["gross_FULL_sharpe"] = fam_gross_st.get("FULL_sharpe", 0.0)
        fam_st["gross_IS_sharpe"]   = fam_gross_st.get("IS_sharpe", 0.0)
        fam_st["gross_OOS_sharpe"]  = fam_gross_st.get("OOS_sharpe", 0.0)
        fam_summary_rows.append(fam_st)
        if fam_stream is not None and n_surv > 0:
            fam_streams[fam_label] = fam_stream
            print(f"  {fam_label}: {n_surv}/{n_total} survived. "
                  f"NET IS Sh={fam_st['IS_sharpe']:+.2f}  OOS Sh={fam_st['OOS_sharpe']:+.2f}  "
                  f"FULL Sh={fam_st['FULL_sharpe']:+.2f}  "
                  f"GROSS IS={fam_st['gross_IS_sharpe']:+.2f}")
        else:
            print(f"  {fam_label}: 0/{n_total} survived  "
                  f"(family-avg GROSS IS={fam_st['gross_IS_sharpe']:+.2f} "
                  f"OOS={fam_st['gross_OOS_sharpe']:+.2f}) — costs / no IS edge.")

    # ---- Combine surviving families into sleeve ----------------------------
    if not fam_streams:
        print("\nNo surviving families.")
        # Write a zero-return sleeve over a daily index from BTC H1 so master_v3
        # can still join us with weight=0 if desired. The recommended weight in
        # the master portfolio is 0 — see the breakdown CSV for the diagnosis.
        ref = get_candles("BTCUSDT", "H1")
        ref_idx = pd.DatetimeIndex(pd.to_datetime(ref["timestamp"], utc=True))
        day_idx = pd.DatetimeIndex(sorted(set(ref_idx.floor("D"))))
        sleeve_zero = pd.Series(0.0, index=day_idx, name="ret")
        out_df = pd.DataFrame({"timestamp": sleeve_zero.index, "ret": sleeve_zero.values})
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
        out_df.to_parquet(OUT_DIR / "intraday_returns.parquet", index=False)
        # Breakdown CSV: per-strategy rows + per-family summary.
        bd = pd.DataFrame(breakdown_rows)
        fam_df = pd.DataFrame(fam_summary_rows)
        bd_out = pd.concat(
            [bd.assign(scope="strategy"), fam_df.assign(scope="family")],
            ignore_index=True,
        )
        bd_out.to_csv(OUT_DIR / "intraday_breakdown.csv", index=False)
        print(f"\nWrote {OUT_DIR/'intraday_returns.parquet'} (zero-sleeve, {len(out_df)} daily rows)")
        print(f"Wrote {OUT_DIR/'intraday_breakdown.csv'} ({len(bd_out)} rows)")

        # Print final per-strategy summary for the report.
        print("\n--- Per-strategy net Sharpe (sorted by IS) ---")
        cols = ["family", "label", "IS_sharpe", "OOS_sharpe", "FULL_sharpe",
                "Y2022_sharpe", "gross_IS_sharpe", "gross_OOS_sharpe", "survived"]
        avail = [c for c in cols if c in bd.columns]
        print(bd[avail].sort_values("IS_sharpe", ascending=False).to_string(index=False))

        print("\n--- Per-family summary (NET vs GROSS) ---")
        fam_cols = ["family", "n_total", "n_survivors", "IS_sharpe", "OOS_sharpe",
                    "FULL_sharpe", "gross_IS_sharpe", "gross_OOS_sharpe", "gross_FULL_sharpe"]
        fam_avail = [c for c in fam_cols if c in fam_df.columns]
        print(fam_df[fam_avail].to_string(index=False))
        return

    # Equal-weight across surviving families, then rescale to 5% IS vol.
    fam_mat = pd.concat(list(fam_streams.values()), axis=1).fillna(0.0)
    sleeve_raw = fam_mat.mean(axis=1)
    sleeve = vol_scale_is(sleeve_raw, SLEEVE_TARGET_VOL)
    sleeve.name = "ret"

    # ---- Stats on combined sleeve ------------------------------------------
    s = stats_for("INTRADAY_SLEEVE", sleeve)
    print("\n=== Combined intraday sleeve (D1-collapsed, 5% IS vol target) ===")
    for tag in ["FULL", "IS", "OOS"]:
        print(f"  {tag:<5}: Sharpe={s[f'{tag}_sharpe']:+.2f}  "
              f"Ret={s[f'{tag}_ret']:+.2%}  Vol={s[f'{tag}_vol']:.2%}  "
              f"DD={s[f'{tag}_dd']:+.2%}")
    print(f"  2022 : Sharpe={s.get('Y2022_sharpe', np.nan):+.2f}")

    # Year-by-year
    print("\n--- Year-by-year sleeve Sharpe ---")
    for yr, sub in sleeve.groupby(sleeve.index.year):
        if len(sub) < 30 or sub.std(ddof=0) == 0:
            continue
        bpy = _bpy_idx(sub.index)
        sh = float(sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)))
        print(f"  {yr}  Sharpe={sh:+5.2f}  Ret={sub.mean()*bpy:+7.2%}  n={len(sub)}")

    # ---- Save outputs ------------------------------------------------------
    out_path = OUT_DIR / "intraday_returns.parquet"
    out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out_df)} daily rows)")

    # Breakdown CSV: per-strategy rows + per-family summary at bottom
    bd = pd.DataFrame(breakdown_rows)
    fam_df = pd.DataFrame(fam_summary_rows)
    fam_df["survived"] = fam_df["n_survivors"] > 0 if "n_survivors" in fam_df else False
    # Stack them: per-strategy rows, then a separator row, then per-family rows.
    bd_path = OUT_DIR / "intraday_breakdown.csv"
    bd_out = pd.concat(
        [bd.assign(scope="strategy"), fam_df.assign(scope="family")],
        ignore_index=True,
    )
    bd_out.to_csv(bd_path, index=False)
    print(f"Wrote {bd_path}  ({len(bd_out)} rows)")

    # ---- Quick per-family table to stdout ----------------------------------
    print("\n--- Per-strategy survival ---")
    cols = ["family", "label", "IS_sharpe", "OOS_sharpe", "FULL_sharpe",
            "Y2022_sharpe", "gross_IS_sharpe", "gross_OOS_sharpe", "survived"]
    avail = [c for c in cols if c in bd.columns]
    print(bd[avail].sort_values("IS_sharpe", ascending=False).to_string(index=False))

    print("\n--- Per-family summary (NET vs GROSS) ---")
    fam_cols = ["family", "n_total", "n_survivors", "IS_sharpe", "OOS_sharpe",
                "FULL_sharpe", "gross_IS_sharpe", "gross_OOS_sharpe", "gross_FULL_sharpe"]
    fam_avail = [c for c in fam_cols if c in fam_df.columns]
    print(fam_df[fam_avail].to_string(index=False))


if __name__ == "__main__":
    main()
