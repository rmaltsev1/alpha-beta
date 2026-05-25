"""Session-opens momentum-follow sleeve (wave3 — inverse of reversion).

Hypothesis (flip): the wave-3 reversion sleeve produced strongly *negative*
Sharpe on session opens (Asia open Sharpe ~ -1.24 for fading). If fading
loses, then *following* the open's direction is the better bet — session
opens are MOMENTUM events, not reversion events.

Same six sub-strategies, same scopes, same hold horizons, same costs — only
the sign of the direction is flipped. Survivors and combination use the
same gates as the reversion run so results are directly comparable.

  1. Asia-open momentum follow        (JP225, USD_JPY, XAU_USD, BTCUSDT) H1
  2. London-open momentum follow      (EUR_USD, GBP_USD)         H1
  3. NY-open continuation             (SPX, NAS, US30)           H1
  4. NY first-bar continuation (top)  (SPX, NAS, US30)           H1
  5. Gap continuation                  (SPX, NAS, US30)           D1->H1
  6. Sunday-evening crypto follow      (BTC, ETH)                 H1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles, SYMBOL_TYPE
from alphabeta.backtest import backtest, cost_for

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
TRADING_DAYS = 252


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def ann_stats(r: pd.Series, freq: int = TRADING_DAYS) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
    av = float(r.std(ddof=0) * np.sqrt(freq))
    ar = float(r.mean() * freq)
    sh = ar / av if av > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": sh, "ann_ret": ar, "ann_vol": av, "max_dd": dd}


def split_stats(r: pd.Series) -> dict:
    """`r` is a D1-indexed (UTC) returns series."""
    out = {}
    full = ann_stats(r)
    out["full_sharpe"] = full["sharpe"]
    out["full_dd"] = full["max_dd"]
    out["full_vol"] = full["ann_vol"]
    out["full_ret"] = full["ann_ret"]
    is_r = r[r.index < SPLIT]
    oos_r = r[r.index >= SPLIT]
    out["is_sharpe"] = ann_stats(is_r)["sharpe"]
    out["is_vol"] = ann_stats(is_r)["ann_vol"]
    out["oos_sharpe"] = ann_stats(oos_r)["sharpe"]
    out["oos_vol"] = ann_stats(oos_r)["ann_vol"]
    for yr in (2022, 2024):
        yr_r = r[(r.index >= pd.Timestamp(f"{yr}-01-01", tz="UTC")) &
                 (r.index < pd.Timestamp(f"{yr+1}-01-01", tz="UTC"))]
        out[f"y{yr}_sharpe"] = ann_stats(yr_r)["sharpe"]
    return out


def collapse_to_d1(r_h1: pd.Series) -> pd.Series:
    """Sum H1 returns within UTC calendar day -> D1 returns."""
    r = r_h1.copy()
    if r.empty:
        return r
    r.index = pd.to_datetime(r.index, utc=True)
    daily = r.groupby(r.index.normalize()).sum()
    daily.index = pd.to_datetime(daily.index, utc=True)
    return daily


# -----------------------------------------------------------------------------
# Helpers shared with the reversion script
# -----------------------------------------------------------------------------
def build_position(
    df: pd.DataFrame,
    trigger: pd.Series,
    direction: pd.Series,
    hold: int,
) -> pd.Series:
    """Position is set for `hold` bars following each trigger.
    direction[t] in {-1,0,+1} = sign we want when the trigger fires at t.
    Signal acts on bar t+1 (executes on close[t]; held over [t+1 .. t+hold]).
    """
    pos = np.zeros(len(df), dtype="float64")
    trig_idx = np.where(trigger.fillna(False).values & (direction.fillna(0).values != 0))[0]
    dir_vals = direction.fillna(0).values.astype("float64")
    for i in trig_idx:
        d = dir_vals[i]
        if d == 0:
            continue
        s = i + 1
        e = min(i + 1 + hold, len(df))
        if s >= len(df):
            continue
        pos[s:e] = d
    return pd.Series(pos, index=df.index)


def rolling_session_stats(
    session_ret: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """Rolling mean+std over the sparse session-only series, shift(1) to avoid
    look-ahead, then will be ffill-broadcast back to H1 by the caller."""
    m = session_ret.rolling(window, min_periods=max(10, window // 3)).mean().shift(1)
    s = session_ret.rolling(window, min_periods=max(10, window // 3)).std(ddof=0).shift(1)
    return m, s


# -----------------------------------------------------------------------------
# Inverse (momentum-follow) sub-strategy builders.
# Each is structurally identical to the reversion version, but with the
# direction sign flipped — `+np.sign(z)` instead of `-np.sign(z)` etc.
# -----------------------------------------------------------------------------
def asia_open_momentum(df: pd.DataFrame, sd_thresh: float = 2.0, hold: int = 5) -> pd.Series:
    """1. 00:00 UTC bar: if |z| > 2 SD vs prior-30 session-day stats, FOLLOW
    the direction of the move for `hold` H1 bars (4-6h ≈ 5).
    """
    ts = df["timestamp"]
    is_open = (ts.dt.hour == 0)
    bar_ret = df["close"].pct_change()
    sess = bar_ret.where(is_open)
    m, s = rolling_session_stats(sess.dropna(), window=30)
    m = m.reindex(df.index).ffill()
    s = s.reindex(df.index).ffill()
    z = (bar_ret - m) / s
    z = z.where(is_open)
    trigger = (z.abs() > sd_thresh) & is_open
    direction = np.sign(z)  # FOLLOW (flip of -sign(z))
    return build_position(df, trigger.fillna(False), direction.fillna(0), hold)


def london_open_fx_momentum(df: pd.DataFrame, abs_thresh: float = 0.005, hold: int = 3) -> pd.Series:
    """2. London open 08:00 UTC bar. If |ret| > 0.5%, FOLLOW for `hold` bars
    (2-4h, default 3)."""
    ts = df["timestamp"]
    is_open = (ts.dt.hour == 8)
    bar_ret = df["close"].pct_change()
    trigger = is_open & (bar_ret.abs() > abs_thresh)
    direction = np.sign(bar_ret).where(is_open, 0)  # FOLLOW
    return build_position(df, trigger.fillna(False), direction.fillna(0), hold)


def ny_open_continuation(df: pd.DataFrame, sd_thresh: float = 1.5, hold: int = 2) -> pd.Series:
    """3. 14:00 UTC bar. If bar moves > 1.5 SD AGAINST prior 5-bar trend,
    FOLLOW the open's direction (instead of fading) for `hold` bars.
    """
    ts = df["timestamp"]
    is_open = (ts.dt.hour == 14)
    bar_ret = df["close"].pct_change()
    sess = bar_ret.where(is_open)
    m, s = rolling_session_stats(sess.dropna(), window=60)
    m = m.reindex(df.index).ffill()
    s = s.reindex(df.index).ffill()
    z = (bar_ret - m) / s

    trend5 = np.sign(df["close"].shift(1) / df["close"].shift(6) - 1.0)
    against = (np.sign(z) != trend5) & (trend5 != 0)
    trigger = is_open & (z.abs() > sd_thresh) & against
    direction = np.sign(z).where(is_open, 0)  # FOLLOW
    return build_position(df, trigger.fillna(False), direction.fillna(0), hold)


def ny_first_bar_continuation(df: pd.DataFrame, range_pct: float = 0.6,
                              close_pos: float = 0.75, hold: int = 2) -> pd.Series:
    """4. 14:00 UTC bar. Big range AND close near top of bar → FOLLOW long.
    Big range AND close near bottom → FOLLOW short. (Same as reversion's
    structurally-identical signal: this strategy was already a continuation.)
    """
    ts = df["timestamp"]
    is_open = (ts.dt.hour == 14)
    bar_range = (df["high"] - df["low"]) / df["close"].shift(1)
    sess_range = bar_range.where(is_open).dropna()
    sess_pct = sess_range.rolling(60, min_periods=20).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    ).shift(1)
    pct = sess_pct.reindex(df.index).ffill()

    bar_pos = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    big_range = is_open & (pct > range_pct)
    long_trig = big_range & (bar_pos > close_pos)
    short_trig = big_range & (bar_pos < (1 - close_pos))
    direction = pd.Series(0.0, index=df.index)
    direction[long_trig] = 1.0
    direction[short_trig] = -1.0
    trigger = long_trig | short_trig
    return build_position(df, trigger.fillna(False), direction, hold)


def gap_continuation(df_h1: pd.DataFrame, gap_thresh: float = 0.005, hold: int = 6) -> pd.Series:
    """5. Overnight gap CONTINUATION (flip of gap-fill). D1 gap > +0.5%
    long; < -0.5% short. Triggered at the first H1 bar of the UTC day.
    """
    ts = df_h1["timestamp"]
    date = ts.dt.normalize()
    d_open = df_h1.groupby(date)["open"].first()
    d_close = df_h1.groupby(date)["close"].last()
    prior_close = d_close.shift(1)
    gap = (d_open / prior_close - 1.0)

    direction_per_day = pd.Series(0.0, index=gap.index)
    direction_per_day[gap > gap_thresh] = 1.0   # FOLLOW long
    direction_per_day[gap < -gap_thresh] = -1.0  # FOLLOW short

    first_bar_idx = df_h1.groupby(date)["timestamp"].idxmin()
    trigger = pd.Series(False, index=df_h1.index)
    direction = pd.Series(0.0, index=df_h1.index)
    for d, idx in first_bar_idx.items():
        di = direction_per_day.get(d, 0.0)
        if di == 0.0:
            continue
        trigger.iloc[idx] = True
        direction.iloc[idx] = di
    return build_position(df_h1, trigger, direction, hold)


def crypto_sunday_follow(df: pd.DataFrame, gap_thresh: float = 0.03, hold: int = 8) -> pd.Series:
    """6. Friday close → Sunday Asia open gap > 3% → FOLLOW the gap
    direction for 8 H1 bars (instead of fading)."""
    ts = df["timestamp"]
    dow = ts.dt.dayofweek
    hr = ts.dt.hour
    sun_open_mask = (dow == 6) & (hr == 0)
    last_fri = (dow == 4) & (hr == 21)
    fri_close = df["close"].where(last_fri).ffill()

    sun_open = df["open"]
    gap = (sun_open / fri_close - 1.0).where(sun_open_mask)
    direction = pd.Series(0.0, index=df.index)
    direction[gap > gap_thresh] = 1.0   # FOLLOW
    direction[gap < -gap_thresh] = -1.0  # FOLLOW
    trigger = sun_open_mask & (gap.abs() > gap_thresh)
    return build_position(df, trigger.fillna(False), direction, hold)


# -----------------------------------------------------------------------------
# Strategy registry. Asia adds BTCUSDT as instructed.
# -----------------------------------------------------------------------------
STRATEGIES = {
    "asia_momentum": {
        "fn": asia_open_momentum,
        "symbols": ["JP225_USD", "USD_JPY", "XAU_USD", "BTCUSDT"],
        "session": "Asia (00z)",
        "hold": 5,
    },
    "london_fx_momentum": {
        "fn": london_open_fx_momentum,
        "symbols": ["EUR_USD", "GBP_USD"],
        "session": "London (08z)",
        "hold": 3,
    },
    "ny_open_continuation": {
        "fn": ny_open_continuation,
        "symbols": ["SPX500_USD", "NAS100_USD", "US30_USD"],
        "session": "NY (14z)",
        "hold": 2,
    },
    "ny_first_bar_cont": {
        "fn": ny_first_bar_continuation,
        "symbols": ["SPX500_USD", "NAS100_USD", "US30_USD"],
        "session": "NY (14z)",
        "hold": 2,
    },
    "gap_continuation": {
        "fn": gap_continuation,
        "symbols": ["SPX500_USD", "NAS100_USD", "US30_USD"],
        "session": "Daily open",
        "hold": 6,
    },
    "crypto_sunday_follow": {
        "fn": crypto_sunday_follow,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "session": "Asia Sun (00z)",
        "hold": 8,
    },
}


# -----------------------------------------------------------------------------
# Per-symbol-strategy runner
# -----------------------------------------------------------------------------
def run_one(symbol: str, strat_name: str, cfg: dict):
    df = get_candles(symbol, "H1", start="2020-01-01", end="2026-05-25")
    df = df.reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    pos = cfg["fn"](df)
    pos = pos.reindex(df.index).fillna(0.0)

    is_mask = ts < SPLIT
    bar_ret = df["close"].pct_change().fillna(0.0)
    raw_strat = pos * bar_ret
    is_strat = raw_strat[is_mask]
    bars_per_year = (len(df) / max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400, 1)) * 365.25
    is_vol = float(is_strat.std(ddof=0) * np.sqrt(bars_per_year)) if len(is_strat) else 0.0
    if is_vol < 1e-8:
        return None
    scale = TARGET_VOL / is_vol
    scale = float(np.clip(scale, 0.0, 25.0))
    pos_scaled = pos * scale

    res = backtest(df, pos_scaled, symbol=symbol, timeframe="H1",
                   name=f"{symbol}_{strat_name}")
    rets_h1 = pd.Series(res.returns.values, index=ts)
    rets_d1 = collapse_to_d1(rets_h1)
    stats = split_stats(rets_d1)
    n_trigger = int((pos.diff().abs() > 1e-12).sum())
    stats["n_trigger_events"] = n_trigger
    stats["exposure"] = float((pos != 0).mean())
    stats["scale"] = scale
    return rets_d1, stats


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Wave3: session-opens MOMENTUM-FOLLOW sleeve (inverse of reversion)")
    print(f"  Split: {SPLIT.date()}")
    print(f"  Target per-sub-sleeve vol (IS): {TARGET_VOL:.1%}")
    print()

    rows = []
    per_returns: dict[str, pd.Series] = {}

    for strat_name, cfg in STRATEGIES.items():
        for sym in cfg["symbols"]:
            try:
                out = run_one(sym, strat_name, cfg)
            except Exception as e:
                print(f"  ! {sym}/{strat_name} failed: {e}")
                continue
            if out is None:
                continue
            rets_d1, stats = out
            key = f"{sym}__{strat_name}"
            per_returns[key] = rets_d1
            cls = SYMBOL_TYPE[sym].value
            row = {
                "strategy": strat_name,
                "session": cfg["session"],
                "symbol": sym,
                "class": cls,
                "hold_bars": cfg["hold"],
                **stats,
            }
            rows.append(row)
            print(f"  {strat_name:<22} {sym:<12} IS={stats['is_sharpe']:+.2f}  "
                  f"OOS={stats['oos_sharpe']:+.2f}  "
                  f"2022={stats['y2022_sharpe']:+.2f}  2024={stats['y2024_sharpe']:+.2f}  "
                  f"trig={stats['n_trigger_events']:>4d}  exp={stats['exposure']:.1%}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "session_momentum_breakdown.csv", index=False)
    print(f"\nWrote {OUT / 'session_momentum_breakdown.csv'}  ({len(breakdown)} rows)")

    if not len(breakdown):
        print("No strategies ran.")
        return
    keep = (breakdown["is_sharpe"] >= 0.4) & (breakdown["oos_sharpe"] >= 0)
    survivors = breakdown[keep].copy()
    print(f"\nSurvivors: {len(survivors)} / {len(breakdown)}")
    if len(survivors):
        print(survivors[["strategy", "symbol", "is_sharpe", "oos_sharpe",
                         "y2022_sharpe", "y2024_sharpe"]].to_string(index=False))

    if len(survivors):
        keys = [f"{r.symbol}__{r.strategy}" for r in survivors.itertuples()]
        aligned = pd.concat([per_returns[k] for k in keys], axis=1).fillna(0.0)
        sleeve_ret = aligned.mean(axis=1)
    else:
        sleeve_ret = pd.Series(dtype=float)

    sleeve_ret.name = "ret"
    sleeve_ret.index.name = "timestamp"
    df_out = sleeve_ret.reset_index()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], utc=True)
    df_out.to_parquet(OUT / "session_momentum_returns.parquet", index=False)
    print(f"\nWrote {OUT / 'session_momentum_returns.parquet'}  ({len(df_out)} rows)")

    print("\nHEADLINE")
    print("-" * 60)
    s = split_stats(sleeve_ret)
    print(f"  FULL  Sharpe: {s['full_sharpe']:+.2f}  Vol={s['full_vol']:.1%}  "
          f"AnnRet={s['full_ret']:+.1%}  DD={s['full_dd']:+.1%}")
    print(f"  IS    Sharpe: {s['is_sharpe']:+.2f}  Vol={s['is_vol']:.1%}")
    print(f"  OOS   Sharpe: {s['oos_sharpe']:+.2f}  Vol={s['oos_vol']:.1%}")
    print(f"  2022  Sharpe: {s['y2022_sharpe']:+.2f}")
    print(f"  2024  Sharpe: {s['y2024_sharpe']:+.2f}")

    print("\nPER-STRATEGY (EW across symbols, full universe):")
    print(f"{'strategy':<22} {'sess':<16} {'n':>3} {'IS':>6} {'OOS':>6} {'2022':>6} {'2024':>6} {'DD':>7}")
    for strat_name, cfg in STRATEGIES.items():
        sub = breakdown[breakdown["strategy"] == strat_name]
        if not len(sub):
            continue
        keys = [f"{r.symbol}__{r.strategy}" for r in sub.itertuples()]
        avail = [k for k in keys if k in per_returns]
        if not avail:
            continue
        aligned = pd.concat([per_returns[k] for k in avail], axis=1).fillna(0.0)
        avg = aligned.mean(axis=1)
        ss = split_stats(avg)
        print(f"{strat_name:<22} {cfg['session']:<16} {len(avail):>3} "
              f"{ss['is_sharpe']:>+6.2f} {ss['oos_sharpe']:>+6.2f} "
              f"{ss['y2022_sharpe']:>+6.2f} {ss['y2024_sharpe']:>+6.2f} "
              f"{ss['full_dd']:>+7.1%}")


if __name__ == "__main__":
    main()
