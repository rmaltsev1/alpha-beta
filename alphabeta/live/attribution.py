"""Sleeve attribution for exposure shifts.

For each instrument, we want to know which sleeves drive its exposure.
Approach:
  1. Pre-compute correlation matrix (24 sleeves × 13 instruments) on trailing 60d.
  2. For each (sleeve, instrument) pair, signed_corr ≈ "this sleeve's natural
     direction on this instrument" (positive = long, negative = short).
  3. For today, contribution_i_per_sleeve ≈ sleeve_return_today × signed_corr.
  4. Sleeves with highest |contribution| on the day are the likely drivers.

Caveats:
  - This is statistical, not the actual sleeve-level position vector.
  - Sleeves with low correlation to any instrument (basket strategies) attribute
    poorly.
  - Cross-correlation between instruments can mis-attribute (e.g., SPX/NAS).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alphabeta import get_candles
from alphabeta.symbols import ALL_SYMBOLS


REPO_ROOT = Path(__file__).resolve().parents[2]
ALL_SLEEVES_PARQUET = REPO_ROOT / "scratch" / "quant" / "all_sleeve_returns_v16.parquet"

CORR_WINDOW_DAYS = 60
MIN_ABS_CORR = 0.15  # below this, the sleeve-instrument link is too weak


@dataclass(frozen=True)
class SleeveContribution:
    sleeve: str
    instrument: str
    corr: float            # trailing 60d Spearman correlation (signed)
    sleeve_ret_today: float
    contribution: float    # ≈ sleeve_ret_today × corr (sign-aware)


def _load_sleeve_panel() -> pd.DataFrame:
    df = pd.read_parquet(ALL_SLEEVES_PARQUET)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _load_instrument_panel() -> pd.DataFrame:
    cols = {}
    for s in ALL_SYMBOLS:
        try:
            df = get_candles(s, "D1").copy()
        except Exception:
            continue
        df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
        df = df.drop_duplicates(subset="date", keep="last").set_index("date")
        cols[s] = np.log(df["close"].astype(float) / df["close"].shift(1))
    panel = pd.concat(cols, axis=1).sort_index()
    if panel.index.tz is None:
        panel.index = pd.DatetimeIndex(panel.index).tz_localize("UTC")
    return panel


def compute_attribution(
    instrument: str,
    *,
    bar_ts: pd.Timestamp,
    n_top: int = 3,
    corr_window: int = CORR_WINDOW_DAYS,
) -> list[SleeveContribution]:
    """Return top-N sleeves likely contributing to the exposure on `instrument`.

    Algorithm: rank sleeves by |corr(sleeve, instrument)| × |sleeve_ret_today|.
    Sign of contribution = sign(corr) × sign(sleeve_ret_today).
    """
    sleeves = _load_sleeve_panel()
    instruments = _load_instrument_panel()
    if instrument not in instruments.columns:
        return []

    # Align on common index
    end_idx = sleeves.index[sleeves.index <= bar_ts]
    if len(end_idx) < corr_window + 1:
        return []
    window_end = end_idx[-1]

    sleeve_w = sleeves.loc[sleeves.index <= window_end].tail(corr_window)
    inst_w = instruments.reindex(sleeve_w.index, method="nearest").fillna(0.0)
    inst_series = inst_w[instrument]

    # Today's row
    today_row = sleeves.loc[window_end]

    contributions: list[SleeveContribution] = []
    for sleeve in sleeve_w.columns:
        s = sleeve_w[sleeve].fillna(0.0)
        if s.std() < 1e-9:
            continue
        # Pearson correlation on log returns is fine for our purposes
        corr = float(np.corrcoef(s.values, inst_series.values)[0, 1])
        if abs(corr) < MIN_ABS_CORR:
            continue
        today_r = float(today_row.get(sleeve, 0.0))
        if pd.isna(today_r):
            continue
        contrib = today_r * np.sign(corr) * abs(corr)
        contributions.append(SleeveContribution(
            sleeve=sleeve,
            instrument=instrument,
            corr=corr,
            sleeve_ret_today=today_r,
            contribution=contrib,
        ))

    contributions.sort(key=lambda c: -abs(c.contribution))
    return contributions[:n_top]


def format_attribution_lines(
    instrument: str,
    contributions: list[SleeveContribution],
    *,
    indent: str = "  ",
) -> list[str]:
    """Format attribution as readable text lines.

    Example:
      ↳ D1REV_UK   (corr=+0.42, ret=+0.18%) contributed +0.08
      ↳ TREND_NEW  (corr=+0.31, ret=+0.12%) contributed +0.04
    """
    if not contributions:
        return [f"{indent}↳ <i>no strong sleeve driver identified</i>"]
    lines = []
    for c in contributions:
        bps_ret = c.sleeve_ret_today * 10000
        sign = "+" if c.contribution >= 0 else "-"
        lines.append(
            f"{indent}↳ <code>{c.sleeve:<14}</code> "
            f"(corr={c.corr:+.2f}, ret={bps_ret:+5.1f} bps) "
            f"{sign}{abs(c.contribution * 100):.2f}%"
        )
    return lines
