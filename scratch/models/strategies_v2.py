"""V2 strategies — concentrated, vol-targeted, combinable.

Each `make_*` returns a position series in [-1, +1]. Use scale_to_vol() to
risk-equalize across strategies before combining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# Utility: scale a position series so the strategy hits a target vol
# =====================================================================

def scale_to_vol(
    df: pd.DataFrame, position: pd.Series, target_ann_vol: float,
) -> pd.Series:
    """Multiply position by a constant chosen so the *historical* strategy
    return hits target_ann_vol on this data.

    Backward-looking by construction — for real walk-forward we'd estimate
    the scale on rolling pre-period data only. For sleeve sizing this is OK.
    """
    ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    raw = position.fillna(0) * ret
    span_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    bars_per_year = len(df) / span_days * 365.25
    ann_vol = raw.std(ddof=0) * np.sqrt(bars_per_year)
    if ann_vol <= 1e-9:
        return position
    return position * (target_ann_vol / ann_vol)


# =====================================================================
# US-INDEX MONDAY-EVENING SLEEVE: long at 23:00 UTC on Mondays
# =====================================================================

def mon_23_long(df: pd.DataFrame) -> pd.Series:
    """+1 at hour 23 UTC on Mondays. 1-bar hold. Concentrated bet on the
    single highest-t hour of the week for SPX/NAS/US30."""
    ts = df["timestamp"]
    pos = pd.Series(0.0, index=df.index)
    pos[(ts.dt.weekday == 0) & (ts.dt.hour == 23)] = 1.0
    return pos


# =====================================================================
# EVENING-USD-SOFTNESS SLEEVE: 21-23 UTC USD weakens vs EUR / GBP / XAU,
# strengthens vs JPY. Concentrated 3-bar window across multiple symbols.
# =====================================================================

def evening_long(df: pd.DataFrame, hours: tuple[int, ...] = (21, 22, 23)) -> pd.Series:
    """+1 during the given evening UTC hours on weekdays."""
    ts = df["timestamp"]
    pos = pd.Series(0.0, index=df.index)
    pos[(ts.dt.weekday < 5) & ts.dt.hour.isin(hours)] = 1.0
    return pos


def evening_short(df: pd.DataFrame, hours: tuple[int, ...] = (21, 22, 23)) -> pd.Series:
    """-1 during the given evening UTC hours on weekdays. Use for USD_JPY."""
    ts = df["timestamp"]
    pos = pd.Series(0.0, index=df.index)
    pos[(ts.dt.weekday < 5) & ts.dt.hour.isin(hours)] = -1.0
    return pos


# =====================================================================
# D1 MEAN-REVERSION SLEEVE: fade yesterday's D1 close on NAS / UK100
# =====================================================================

def d1_reversion(df: pd.DataFrame, threshold_bps: float = 0.0) -> pd.Series:
    ret = np.log(df["close"] / df["close"].shift(1))
    thresh = threshold_bps / 10_000.0
    sig = pd.Series(0.0, index=df.index)
    sig[ret > thresh] = -1.0
    sig[ret < -thresh] = +1.0
    return sig.shift(1).fillna(0.0)


# =====================================================================
# CRYPTO WEDNESDAY SLEEVE: long BTC/ETH/SOL on Wednesday D1
# =====================================================================

def crypto_wed_long(df: pd.DataFrame) -> pd.Series:
    return (df["timestamp"].dt.weekday == 2).astype(float)


# =====================================================================
# USD_JPY WEDNESDAY-EVENING SHORT: hour 23 UTC on Wednesdays only (t=-2.7)
# =====================================================================

def jpy_wed_23_short(df: pd.DataFrame) -> pd.Series:
    ts = df["timestamp"]
    pos = pd.Series(0.0, index=df.index)
    pos[(ts.dt.weekday == 2) & (ts.dt.hour == 23)] = -1.0
    return pos
