"""Per-instrument net exposure estimator (v2).

Problem: the 24 sleeves don't persist positions, only returns. Refactoring
each one would be heavy. Instead we infer the portfolio's *implied* current
exposure to each of the 13 instruments via rolling multivariate regression:

    portfolio_return_t  =  sum_i  beta_i * instrument_return_{i,t}  +  epsilon_t

Fit on the trailing N-day window ending at the latest bar. The beta vector
is the portfolio's implied dollar exposure per dollar of instrument move —
exactly what a paper-trader would care about.

This is the same technique used to factor-decompose hedge fund returns
without seeing their books (Sharpe 1992, "Asset Allocation: Management Style
and Performance Measurement"). It's an *estimate*, not the exact position
vector, but it's directionally correct and stable.

Limitations:
  - When the strategy turns over fast, the rolling window blurs recent shifts.
  - Cross-instrument correlations can confound (e.g., SPX and NAS both load
    on the same systematic moves). Use ridge regularization to stabilize.
  - The R^2 should be high (>0.6) for the betas to be trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alphabeta import get_candles
from alphabeta.symbols import ALL_SYMBOLS


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_PARQUET = REPO_ROOT / "scratch" / "quant" / "PRODUCTION_v16_V4.parquet"

WINDOW_DAYS = 60          # rolling regression window
RIDGE_LAMBDA = 0.001      # tiny ridge for numerical stability
MIN_R2 = 0.50             # below this, betas are too uncertain to publish
SIGNAL_BETA_THRESHOLD = 0.05  # |Δbeta| ≥ 5% triggers a per-instrument signal


@dataclass(frozen=True)
class ExposureSnapshot:
    bar_iso: str
    window_days: int
    r_squared: float
    betas: dict[str, float]            # implied exposure per instrument
    last_known_betas: dict[str, float] # previous snapshot for diffing
    deltas: dict[str, float]           # betas - last_known_betas

    def to_dict(self) -> dict:
        return asdict(self)


def _load_portfolio_returns() -> pd.Series:
    df = pd.read_parquet(PRODUCTION_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return pd.Series(df["ret"].values, index=df["timestamp"]).sort_index()


def _load_instrument_returns() -> pd.DataFrame:
    """Daily log returns for all 13 instruments, UTC-aligned."""
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
    panel.index = pd.DatetimeIndex(panel.index, name="date").tz_localize("UTC") if panel.index.tz is None else panel.index
    return panel


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge regression. Returns (betas, R²)."""
    XtX = X.T @ X
    XtX_reg = XtX + lam * np.eye(X.shape[1])
    XtY = X.T @ y
    betas = np.linalg.solve(XtX_reg, XtY)
    y_pred = X @ betas
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return betas, r2


def compute_exposure(
    *,
    end_date: Optional[pd.Timestamp] = None,
    window_days: int = WINDOW_DAYS,
    last_known: Optional[dict[str, float]] = None,
) -> ExposureSnapshot:
    """Fit rolling ridge regression of portfolio returns on instrument returns.

    end_date: include data up to (and including) this date. None = use latest.
    last_known: previous betas for delta computation.
    """
    port = _load_portfolio_returns()
    instr = _load_instrument_returns()

    if end_date is None:
        end_date = port.index[-1]

    # Align: take dates present in both
    port_w = port[port.index <= end_date].iloc[-window_days:]
    instr_w = instr.reindex(port_w.index, method=None).fillna(0.0)
    syms = [c for c in instr_w.columns if instr_w[c].abs().sum() > 1e-9]
    if len(syms) < 3:
        raise ValueError(f"Not enough instruments with data in window: {syms}")
    X = instr_w[syms].values
    y = port_w.values

    betas, r2 = _ridge_fit(X, y, RIDGE_LAMBDA)
    beta_dict = {s: float(b) for s, b in zip(syms, betas)}
    last_known = last_known or {}
    deltas = {s: beta_dict[s] - last_known.get(s, 0.0) for s in beta_dict}

    return ExposureSnapshot(
        bar_iso=end_date.isoformat(),
        window_days=window_days,
        r_squared=r2,
        betas=beta_dict,
        last_known_betas=last_known,
        deltas=deltas,
    )


def format_exposure_html(snap: ExposureSnapshot) -> str:
    """Format an exposure snapshot as a Telegram HTML block."""
    bar_short = snap.bar_iso[:10]
    rows = sorted(snap.betas.items(), key=lambda x: -abs(x[1]))
    lines = [
        f"<b>📍 Net portfolio exposure — {bar_short}</b>",
        f"<i>Rolling {snap.window_days}d implied beta (R² = {snap.r_squared:.2f})</i>",
        "",
        "<pre>",
        f"{'Instrument':<12} {'Beta':>8}  {'Δ':>8}  Bias",
    ]
    for sym, b in rows:
        if abs(b) < 0.005:
            continue
        d = snap.deltas.get(sym, 0.0)
        bias = "🟢LONG " if b > 0.005 else ("🔴SHORT" if b < -0.005 else "  flat")
        lines.append(f"{sym:<12} {b:+8.3f}  {d:+8.3f}  {bias}")
    lines.append("</pre>")
    if snap.r_squared < MIN_R2:
        lines.append(f"<i>⚠️ R² below {MIN_R2} — exposure estimates uncertain</i>")
    return "\n".join(lines)


def detect_signal_events(snap: ExposureSnapshot) -> list[tuple[str, float, float]]:
    """Return list of (instrument, new_beta, delta) where |delta| ≥ threshold.

    These are the per-instrument "signal events" worth flagging.
    """
    events = []
    for sym, b in snap.betas.items():
        d = snap.deltas.get(sym, 0.0)
        if abs(d) >= SIGNAL_BETA_THRESHOLD:
            events.append((sym, b, d))
    events.sort(key=lambda x: -abs(x[2]))
    return events


def format_signal_events_html(snap: ExposureSnapshot, events: list[tuple[str, float, float]]) -> str:
    """Format detected signal events as a Telegram message."""
    if not events:
        return ""
    bar_short = snap.bar_iso[:10]
    lines = [
        f"<b>🚦 Material exposure shifts — {bar_short}</b>",
        f"<i>|Δbeta| ≥ {SIGNAL_BETA_THRESHOLD:.2f} (R² {snap.r_squared:.2f})</i>",
        "",
        "<pre>",
    ]
    for sym, b, d in events:
        prev = b - d
        arrow = "→"
        if abs(prev) < 0.005:
            action = "OPEN " + ("LONG" if b > 0 else "SHORT")
        elif abs(b) < 0.005:
            action = "CLOSE"
        elif prev * b < 0:
            action = "FLIP " + ("LONG" if b > 0 else "SHORT")
        elif abs(b) > abs(prev):
            action = "ADD  " + ("LONG" if b > 0 else "SHORT")
        else:
            action = "TRIM " + ("LONG" if b > 0 else "SHORT")
        lines.append(f"{sym:<12} {prev:+6.2f} {arrow} {b:+6.2f}  [{action}]")
    lines.append("</pre>")
    lines.append("<i>Estimates from rolling regression — not exact positions.</i>")
    return "\n".join(lines)
