"""Vectorized signal-based backtester.

A "strategy" boils down to: given OHLCV data, produce a `position` series
(values in [-1, +1]) where position[t] is the position held *during* bar t.
The engine multiplies that against the bar return, subtracts costs on
position changes, and reports stats.

Convention: position[t] must be derivable from data observable at the
*start* of bar t — i.e. from close[t-1] and earlier, plus deterministic
calendar info. Strategies must shift their own signals; the engine does
not enforce no-look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .symbols import SYMBOL_TYPE, AssetType


# Per-side cost as a fraction of notional. These are conservative round-trip
# estimates from public retail quotes; they're meant to be slightly pessimistic
# so a strategy that survives them has real edge.
DEFAULT_COSTS_BPS = {
    AssetType.FOREX: 1.0,   # ~0.5 pip on EUR_USD per side → 1 bp
    AssetType.INDEX: 1.5,   # ~1 pt spread on SPX / NAS
    AssetType.CRYPTO: 5.0,  # Binance taker ≈ 10 bps round-trip → 5 bps/side
}


def cost_for(symbol: str) -> float:
    """Return per-side cost (as fraction) for `symbol`."""
    bps = DEFAULT_COSTS_BPS[SYMBOL_TYPE[symbol]]
    return bps / 10_000.0


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    name: str
    equity: pd.Series                    # cumulative equity, starts at 1.0
    returns: pd.Series                   # per-bar strategy returns (net of cost)
    position: pd.Series                  # the position series fed in
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (
            f"{self.name:<30} {self.symbol:<10} {self.timeframe:<3} "
            f"Sharpe={s['sharpe']:>5.2f}  Ret={s['ann_return']:>+6.1%}  "
            f"Vol={s['ann_vol']:>5.1%}  DD={s['max_dd']:>+6.1%}  "
            f"Trades={s['n_trades']:>4d}  Hit={s['hit_rate']:>4.1%}"
        )


def _bars_per_year(df: pd.DataFrame) -> float:
    """Empirical bars-per-year, used for annualization."""
    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    if span <= 0:
        return 252.0
    return len(df) / span * 365.25


def backtest(
    df: pd.DataFrame,
    position: pd.Series,
    *,
    symbol: str,
    timeframe: str,
    name: str,
    cost_per_side: float | None = None,
) -> BacktestResult:
    """Run a single-symbol vectorized backtest.

    `df` must have the canonical columns from alphabeta.storage.
    `position` must be aligned to df's index (0..N-1). Values in [-1, +1].
    Cost is charged on |Δposition| at each bar.
    """
    assert len(df) == len(position), f"position length {len(position)} != df length {len(df)}"
    if cost_per_side is None:
        cost_per_side = cost_for(symbol)

    pos = pd.Series(position.values, index=df.index, dtype="float64").fillna(0.0)
    close = df["close"].astype("float64")
    bar_ret = close.pct_change().fillna(0.0)

    gross = pos * bar_ret
    # Δposition: cost is proportional to *changes*. Flat→long is one cost,
    # long→short crosses zero (two costs' worth of notional turnover).
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    cost = dpos * cost_per_side
    net = gross - cost

    equity = (1.0 + net).cumprod()
    bars_per_year = _bars_per_year(df)
    ann_ret = net.mean() * bars_per_year
    ann_vol = net.std(ddof=0) * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    # A "trade" = a non-zero Δposition event. Round-trip = 2 events.
    trade_events = (dpos > 0).sum()
    n_trades = max(int(trade_events // 2), 1)

    # Hit-rate: per-bar net return positive given an active position.
    active = pos.abs() > 0
    hit_rate = (net[active] > 0).mean() if active.any() else 0.0

    # Profit factor: sum of positive trade pnl / |sum of negative|.
    pnl_pos = net[net > 0].sum()
    pnl_neg = -net[net < 0].sum()
    profit_factor = (pnl_pos / pnl_neg) if pnl_neg > 0 else float("inf")

    # Exposure: fraction of bars where we hold a position.
    exposure = active.mean()

    stats = {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_trades": int(n_trades),
        "hit_rate": float(hit_rate),
        "profit_factor": float(profit_factor),
        "exposure": float(exposure),
        "n_bars": int(len(df)),
        "bars_per_year": float(bars_per_year),
        "cost_per_side": float(cost_per_side),
        "first": df["timestamp"].iloc[0],
        "last": df["timestamp"].iloc[-1],
    }
    return BacktestResult(
        symbol=symbol, timeframe=timeframe, name=name,
        equity=equity, returns=net, position=pos, stats=stats,
    )


def split_is_oos(df: pd.DataFrame, split: str = "2024-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a candle dataframe into in-sample / out-of-sample by date."""
    split_ts = pd.Timestamp(split, tz="UTC")
    is_mask = df["timestamp"] < split_ts
    is_df = df[is_mask].reset_index(drop=True)
    oos_df = df[~is_mask].reset_index(drop=True)
    return is_df, oos_df


def combine_returns(results: list[BacktestResult], weights: list[float] | None = None) -> pd.Series:
    """Daily-or-bar-aligned aggregation of multiple strategy return streams.

    The streams can be on different timeframes / symbols. We reindex to the
    union of all timestamps, forward-fill nothing (zero on missing), then
    weight and sum.
    """
    streams = []
    for r in results:
        s = r.returns.copy()
        s.index = r.equity.index  # already aligned to df index
        # We need timestamp index to align across strategies. Pull from result.
        # The BacktestResult doesn't carry timestamps directly; reconstruct.
        # (We embed timestamp in the index via the caller; see run_strategy.)
        streams.append(s)
    if weights is None:
        weights = [1.0 / len(streams)] * len(streams)
    aligned = pd.concat(streams, axis=1).fillna(0.0)
    return (aligned * np.asarray(weights)).sum(axis=1)
