"""Strategy library — each function returns a position series aligned to df.

Conventions:
  * position[t] ∈ {-1, 0, +1}, held *during* bar t. Cost charged on |Δpos|.
  * All decisions must be derivable from data at *start* of bar t (i.e. from
    df rows 0..t-1, plus deterministic calendar info from df["timestamp"][t]).
  * Never reference future bars or even df["close"][t] when sizing position[t].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# Hour-of-day biases (from agent2 findings)
# =====================================================================

def hour_of_day(df: pd.DataFrame, hour_to_side: dict[int, int]) -> pd.Series:
    """Hold the given side at the given UTC hour(s). Flat otherwise.

    hour_to_side: e.g. {23: +1} or {8: +1, 11: -1}.
    """
    h = df["timestamp"].dt.hour
    pos = pd.Series(0.0, index=df.index)
    for hour, side in hour_to_side.items():
        pos[h == hour] = float(side)
    return pos


# =====================================================================
# D1 mean-reversion on equity / large-cap crypto (agent3 finding)
# =====================================================================

def daily_reversion(df: pd.DataFrame, threshold_bps: float = 0.0) -> pd.Series:
    """Fade yesterday's D1 sign.

    threshold_bps: only act when |yesterday's return| exceeds this (in bps).
    A small threshold filters out the noise-floor mean-reversion which is
    cheap noise.
    """
    ret = np.log(df["close"] / df["close"].shift(1))
    thresh = threshold_bps / 10_000.0
    sig = pd.Series(0.0, index=df.index)
    sig[ret > thresh] = -1.0
    sig[ret < -thresh] = +1.0
    # Shift by one so the *previous* bar's return drives today's position.
    return sig.shift(1).fillna(0.0)


# =====================================================================
# Monday-open long on US indices (agent2 finding)
# =====================================================================

def monday_session(df: pd.DataFrame, *, ny_only: bool = True) -> pd.Series:
    """Long during Monday NY hours (12-20 UTC). Flat else.

    OANDA D1 closes at 21:00 UTC. The "Monday gap-up" in indices is actually
    the Sunday-evening / week-open NY-session bar. We just hold long during
    every NY hour on a Monday.
    """
    ts = df["timestamp"]
    is_monday = ts.dt.weekday == 0
    h = ts.dt.hour
    if ny_only:
        active = is_monday & (h >= 12) & (h < 20)
    else:
        active = is_monday
    return active.astype(float)


# =====================================================================
# NY-session long bias on US indices (agent2 finding)
# =====================================================================

def ny_session_long(df: pd.DataFrame) -> pd.Series:
    """Long during NY hours (12-20 UTC) every weekday."""
    ts = df["timestamp"]
    h = ts.dt.hour
    wd = ts.dt.weekday
    active = (h >= 12) & (h < 20) & (wd < 5)
    return active.astype(float)


# =====================================================================
# London-open EUR_USD intraday: long 08:00, short 11:00 (agent2 finding)
# =====================================================================

def eur_london_pattern(df: pd.DataFrame) -> pd.Series:
    """+1 at 08:00 UTC, -1 at 11:00 UTC, weekdays only."""
    ts = df["timestamp"]
    h = ts.dt.hour
    wd = ts.dt.weekday
    pos = pd.Series(0.0, index=df.index)
    pos[(h == 8) & (wd < 5)] = +1.0
    pos[(h == 11) & (wd < 5)] = -1.0
    return pos


# =====================================================================
# Crypto midweek long (agent2 finding: BTC/ETH Wed positive D1)
# =====================================================================

def crypto_midweek(df: pd.DataFrame) -> pd.Series:
    """Long on Wednesday D1 bars."""
    ts = df["timestamp"]
    return (ts.dt.weekday == 2).astype(float)


# =====================================================================
# JP225 Asia open volatility breakout (agent2 finding: 00:00 UTC range 2x)
# =====================================================================

def asia_open_breakout(df: pd.DataFrame, ref_hour: int = 23, signal_hour: int = 0) -> pd.Series:
    """Trade the JP225 Asia open in the direction of the previous bar.

    At signal_hour (00:00 UTC), take the sign of the close[signal_hour-1] -
    open[signal_hour-1] (the bar just before the Tokyo open) and hold it
    for one bar. The hypothesis: the Asia-open bar extends the immediately
    preceding move.
    """
    ts = df["timestamp"]
    h = ts.dt.hour
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    direction = np.sign(prev_close - prev_open).fillna(0.0)
    pos = pd.Series(0.0, index=df.index)
    mask = (h == signal_hour)
    pos[mask] = direction[mask]
    return pos
