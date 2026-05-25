"""Sophisticated trend-following strategies (wave 3).

Seven strategy families on D1:
  1. Donchian breakout (20-bar and 55-bar) on every symbol.
  2. TSMOM-252 with vol-regime filter (only trade in low-vol regimes).
  3. Triple-screen momentum (252d > 0 AND 63d > 0 AND 5d > 0, long-only).
  4. MACD-style (12,26,9) trend.
  5. Momentum reversal at extremes (252d momentum, halve at +30% / -20% 30d).
  6. Cross-asset trend confirmation (basket on if >= 2/3 members have 90d>0).
  7. Trailing-stop trend on indices (long when 50d > +5%, exit on -10% from peak).

All sub-sleeves are vol-targeted to 5% IS annualized vol, then survivors
(IS Sharpe >= 0.4, OOS Sharpe >= 0) are combined equal-weight into a
single trend sleeve. Outputs:

  scratch/wave3/trend_returns.parquet   -- one row per UTC calendar day
  scratch/wave3/trend_breakdown.csv     -- per-strategy stats
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import backtest, cost_for


# -- knobs --------------------------------------------------------------------
SPLIT = "2024-01-01"
TF = "D1"
BPY = 365.25                  # bars-per-year for annualization
IS_VOL_TARGET = 0.05          # 5% IS ann vol per sub-sleeve
IS_SHARPE_MIN = 0.4
OOS_SHARPE_MIN = 0.0
OUT_DIR = Path(__file__).resolve().parent


# -- helpers ------------------------------------------------------------------
def _stats(returns: pd.Series, freq: float = BPY) -> dict:
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0, "n": int(len(r))}
    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_dd": float(dd),
        "n": int(len(r)),
    }


def _to_utc_index(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Re-index a per-bar return series by the symbol's UTC timestamps."""
    out = pd.Series(s.values, index=pd.DatetimeIndex(df["timestamp"]))
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _vol_scale_is(ret: pd.Series, split_ts: pd.Timestamp, target: float = IS_VOL_TARGET) -> float:
    """One scalar from IS sub-window so OOS is genuinely OOS for the scaler."""
    is_r = ret.loc[ret.index < split_ts].dropna()
    if is_r.empty:
        return 0.0
    sd = is_r.std(ddof=0) * np.sqrt(BPY)
    if sd <= 0:
        return 0.0
    return float(target / sd)


# -- strategies ---------------------------------------------------------------
def donchian_position(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Long (+1) on close > trailing N-bar high, short (-1) on close < N-bar low.

    Hold until reversed. Uses *prior* high/low to avoid look-ahead. The
    breakout is evaluated on the close of bar t-1; the position is held on
    bar t.
    """
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    # Trailing N-bar high/low EXCLUDING the bar itself: shift(1) first.
    trail_hi = high.shift(1).rolling(lookback, min_periods=lookback).max()
    trail_lo = low.shift(1).rolling(lookback, min_periods=lookback).min()
    long_break = close > trail_hi
    short_break = close < trail_lo
    sig = pd.Series(np.nan, index=df.index, dtype="float64")
    sig[long_break] = 1.0
    sig[short_break] = -1.0
    sig = sig.ffill().fillna(0.0)
    # position[t] from info at close of t-1
    return sig.shift(1).fillna(0.0)


def tsmom_vol_filter_position(df: pd.DataFrame, lookback: int = 252,
                              vol_win: int = 30, regime_win: int = 252,
                              percentile: float = 0.75) -> pd.Series:
    """Standard sign(252d) TSMOM, gated off when 30d realized vol >= 252d 75th pctile."""
    close = df["close"].astype("float64")
    log_ret = np.log(close).diff()
    trail_sum = log_ret.rolling(lookback, min_periods=lookback).sum()
    sig = np.sign(trail_sum)

    rv30 = log_ret.rolling(vol_win, min_periods=vol_win).std(ddof=0)
    rv30_threshold = rv30.rolling(regime_win, min_periods=regime_win).quantile(percentile)
    # gate: trade only when current 30d vol is BELOW the 252d 75th pctile of that 30d vol
    gate = (rv30 < rv30_threshold).astype(float)
    pos = (sig * gate).fillna(0.0)
    return pos.shift(1).fillna(0.0)


def triple_screen_position(df: pd.DataFrame) -> pd.Series:
    """Long-only: +1 when trailing 252d, 63d, and 5d returns are all positive."""
    close = df["close"].astype("float64")
    r252 = close.pct_change(252)
    r63 = close.pct_change(63)
    r5 = close.pct_change(5)
    sig = ((r252 > 0) & (r63 > 0) & (r5 > 0)).astype(float)
    return sig.shift(1).fillna(0.0)


def macd_position(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """+1 when MACD > signal, -1 otherwise. EMA-based."""
    close = df["close"].astype("float64")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig_line = macd.ewm(span=signal, adjust=False).mean()
    pos = np.where(macd > sig_line, 1.0, -1.0)
    pos = pd.Series(pos, index=df.index)
    # First slow+signal bars are warmup; flatten to 0 to avoid garbage early.
    warmup = slow + signal
    pos.iloc[:warmup] = 0.0
    return pos.shift(1).fillna(0.0)


def momentum_reversal_position(df: pd.DataFrame, lookback: int = 252,
                               win30: int = 30,
                               hi_thresh: float = 0.30,
                               lo_thresh: float = -0.20) -> pd.Series:
    """sign(252d) momentum; cut position by 50% when 30d return is in extreme zone."""
    close = df["close"].astype("float64")
    log_ret = np.log(close).diff()
    trail_sum = log_ret.rolling(lookback, min_periods=lookback).sum()
    sig = np.sign(trail_sum)
    r30 = close.pct_change(win30)
    extreme = (r30 > hi_thresh) | (r30 < lo_thresh)
    scale = pd.Series(np.where(extreme, 0.5, 1.0), index=df.index)
    pos = (sig * scale).fillna(0.0)
    return pos.shift(1).fillna(0.0)


def trailing_stop_position(df: pd.DataFrame, entry_win: int = 50,
                           entry_thresh: float = 0.05,
                           stop_pct: float = 0.10) -> pd.Series:
    """Long when trailing 50d return > +5%; exit when close drops 10% from peak.

    Implemented as a stateful daily loop because path-dependence kills any
    pure vectorization. Position is +1 (in) or 0 (flat). No shorts.
    """
    close = df["close"].astype("float64").to_numpy()
    r50 = pd.Series(close, index=df.index).pct_change(entry_win).to_numpy()
    n = len(close)
    pos = np.zeros(n, dtype="float64")
    in_trade = False
    peak = 0.0
    for t in range(n):
        if in_trade:
            peak = max(peak, close[t])
            # exit if drawdown from peak >= stop_pct
            if (peak - close[t]) / peak >= stop_pct:
                in_trade = False
                pos[t] = 0.0
            else:
                pos[t] = 1.0
        else:
            # Entry signal based on data UP TO and including t-1 only:
            # use r50 at t-1 to decide whether to be in on bar t.
            if t >= 1 and not np.isnan(r50[t - 1]) and r50[t - 1] > entry_thresh:
                in_trade = True
                peak = close[t]
                pos[t] = 1.0
    # The exits above use the bar's close; shift to enforce "no look-ahead"
    # at the entry point. The exit is therefore lagged by one bar — that
    # matches the engine's convention that position[t] uses info up to t-1.
    s = pd.Series(pos, index=df.index)
    return s.shift(1).fillna(0.0)


# -- driver -------------------------------------------------------------------
@dataclass
class SubResult:
    name: str           # strategy family name
    variant: str        # specific variant id (e.g. "donchian20", "macd")
    symbol: str
    returns: pd.Series  # UTC-indexed daily returns (pre-vol-scale)
    stats_full: dict
    stats_is: dict
    stats_oos: dict
    vol_scale: float    # scalar to hit IS_VOL_TARGET on IS


def _backtest_position(symbol: str, df: pd.DataFrame, pos: pd.Series,
                       name: str) -> pd.Series:
    res = backtest(df, pos, symbol=symbol, timeframe=TF, name=name)
    return _to_utc_index(res.returns, df)


def _collect_strategies() -> list[SubResult]:
    """Run every variant on every (eligible) symbol and return the sub-results.

    Strategies 1, 3, 4, 5 are per-symbol. Strategy 2 (vol-regime TSMOM) is
    per-symbol. Strategy 6 is per *basket*. Strategy 7 is index-only.
    """
    split_ts = pd.Timestamp(SPLIT, tz="UTC")
    subs: list[SubResult] = []

    # Cache D1 frames once.
    dfs: dict[str, pd.DataFrame] = {s: get_candles(s, TF) for s in ALL_SYMBOLS}

    def add(variant: str, family: str, symbol: str, pos: pd.Series, df: pd.DataFrame) -> None:
        ret = _backtest_position(symbol, df, pos, name=f"{family}:{variant}")
        scale = _vol_scale_is(ret, split_ts)
        sf = _stats(ret)
        si = _stats(ret.loc[ret.index < split_ts])
        so = _stats(ret.loc[ret.index >= split_ts])
        subs.append(SubResult(
            name=family, variant=variant, symbol=symbol,
            returns=ret, stats_full=sf, stats_is=si, stats_oos=so,
            vol_scale=scale,
        ))

    # -- 1. Donchian breakout 20 & 55 on all 13 ----------------------------
    for s in ALL_SYMBOLS:
        df = dfs[s]
        for L in (20, 55):
            pos = donchian_position(df, L)
            add(variant=f"donchian{L}", family="donchian", symbol=s, pos=pos, df=df)

    # -- 2. TSMOM-252 with vol-regime filter, per symbol --------------------
    for s in ALL_SYMBOLS:
        df = dfs[s]
        pos = tsmom_vol_filter_position(df)
        add(variant="tsmom252_volfilter", family="vol_regime_tsmom",
            symbol=s, pos=pos, df=df)

    # -- 3. Triple-screen momentum, per symbol ------------------------------
    for s in ALL_SYMBOLS:
        df = dfs[s]
        pos = triple_screen_position(df)
        add(variant="triple_screen", family="triple_screen", symbol=s, pos=pos, df=df)

    # -- 4. MACD trend, per symbol ------------------------------------------
    for s in ALL_SYMBOLS:
        df = dfs[s]
        pos = macd_position(df)
        add(variant="macd_12_26_9", family="macd", symbol=s, pos=pos, df=df)

    # -- 5. Momentum reversal at extremes, per symbol -----------------------
    for s in ALL_SYMBOLS:
        df = dfs[s]
        pos = momentum_reversal_position(df)
        add(variant="momrev_30d", family="mom_reversal", symbol=s, pos=pos, df=df)

    # -- 6. Cross-asset trend confirmation (basket gating) ------------------
    # For each basket: compute per-symbol sign(90d) gates. Activate the basket
    # when at least 2 members have positive 90d returns. Trade each member's
    # 252d-momentum sign while basket gate is on; otherwise flat.
    for basket_name, members in (("crypto_basket", CRYPTO),
                                  ("fx_basket", FOREX),
                                  ("index_basket", INDEX)):
        # Build per-symbol 90d gate aligned on each symbol's own calendar; then
        # take the union of timestamps and count how many of the basket members
        # are "in" at each daily bar (each symbol's gate persists on its own
        # calendar). Simpler: align on each member's index, since most baskets
        # share the same trading calendar.
        member_gates = {}
        member_signs = {}
        for s in members:
            df = dfs[s]
            close = df["close"].astype("float64")
            log_ret = np.log(close).diff()
            r90 = close.pct_change(90)
            member_gates[s] = (r90 > 0).astype(float)
            sign252 = np.sign(log_ret.rolling(252, min_periods=252).sum())
            member_signs[s] = sign252

        # Build a basket-on indicator on each symbol's own calendar by counting
        # how many members have a positive 90d return at the same bar index.
        # We align by reindex on each symbol's timestamps (forward-fill from
        # the cross-asset basket gate aggregated on UTC daily).
        # Aggregate basket gate on a union daily index:
        aligned = {}
        for s in members:
            df = dfs[s]
            g = pd.Series(member_gates[s].values, index=pd.DatetimeIndex(df["timestamp"]))
            if g.index.tz is None:
                g.index = g.index.tz_localize("UTC")
            aligned[s] = g
        # union by calendar day
        gate_df = pd.concat(aligned, axis=1)
        # Daily normalize by groupby calendar day, take max (any bar that day)
        gate_df.index = gate_df.index.tz_convert("UTC")
        daily_gate = gate_df.groupby(gate_df.index.normalize()).max()
        daily_gate.index = daily_gate.index.tz_convert("UTC") if daily_gate.index.tz else daily_gate.index.tz_localize("UTC")
        # Carry the last seen value forward (gate only updates on a symbol's own bars).
        daily_gate = daily_gate.ffill().fillna(0.0)
        basket_on = (daily_gate.sum(axis=1) >= 2).astype(float)

        for s in members:
            df = dfs[s]
            sign = member_signs[s].fillna(0.0)
            # Map basket_on (daily) to symbol's own bar index by normalizing the
            # symbol's timestamps to date and looking up.
            sym_ts = pd.DatetimeIndex(df["timestamp"])
            if sym_ts.tz is None:
                sym_ts = sym_ts.tz_localize("UTC")
            else:
                sym_ts = sym_ts.tz_convert("UTC")
            day_keys = sym_ts.normalize()
            sym_gate = pd.Series(
                basket_on.reindex(day_keys).fillna(0.0).values,
                index=df.index,
            )
            pos = (sign * sym_gate).fillna(0.0).shift(1).fillna(0.0)
            add(variant=f"xbasket_{basket_name}", family="cross_asset",
                symbol=s, pos=pos, df=df)

    # -- 7. Trailing-stop trend on indices ----------------------------------
    for s in INDEX:
        df = dfs[s]
        pos = trailing_stop_position(df)
        add(variant="trailstop_50d_10pct", family="trailing_stop",
            symbol=s, pos=pos, df=df)

    return subs


def _combine_survivors(subs: list[SubResult]) -> tuple[pd.Series, list[SubResult]]:
    """Apply IS vol-scale, survivor filter, then equal-weight combine.

    The combined sleeve is on a *daily UTC* index (calendar days).
    """
    survivors = [
        sr for sr in subs
        if sr.stats_is["sharpe"] >= IS_SHARPE_MIN
        and sr.stats_oos["sharpe"] >= OOS_SHARPE_MIN
        and sr.vol_scale > 0
        and np.isfinite(sr.vol_scale)
    ]
    if not survivors:
        return pd.Series(dtype="float64"), survivors

    # Scale each survivor by its IS vol-scale, daily-normalize, combine EW.
    streams = []
    for sr in survivors:
        s = sr.returns * sr.vol_scale
        # Collapse to one row per UTC calendar day.
        s_daily = s.groupby(s.index.normalize()).sum()
        if s_daily.index.tz is None:
            s_daily.index = s_daily.index.tz_localize("UTC")
        else:
            s_daily.index = s_daily.index.tz_convert("UTC")
        s_daily.name = f"{sr.name}|{sr.variant}|{sr.symbol}"
        streams.append(s_daily)
    mat = pd.concat(streams, axis=1).fillna(0.0)
    sleeve = mat.mean(axis=1)
    sleeve.name = "ret"
    return sleeve, survivors


def main() -> None:
    print("Building trend sub-strategies...")
    subs = _collect_strategies()
    print(f"  total sub-strategies: {len(subs)}")

    sleeve, survivors = _combine_survivors(subs)
    print(f"  survivors (IS>={IS_SHARPE_MIN}, OOS>={OOS_SHARPE_MIN}): {len(survivors)}")

    # -- save sleeve parquet ------------------------------------------------
    out_parquet = OUT_DIR / "trend_returns.parquet"
    if not sleeve.empty:
        out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    else:
        out_df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[us, UTC]"),
                               "ret": pd.Series(dtype="float64")})
    out_df.to_parquet(out_parquet, index=False)

    # -- per-strategy CSV ---------------------------------------------------
    rows = []
    for sr in subs:
        rows.append({
            "family": sr.name,
            "variant": sr.variant,
            "symbol": sr.symbol,
            "asset_class": SYMBOL_TYPE[sr.symbol].value,
            "full_sharpe": sr.stats_full["sharpe"],
            "full_ann_return": sr.stats_full["ann_return"],
            "full_ann_vol": sr.stats_full["ann_vol"],
            "full_max_dd": sr.stats_full["max_dd"],
            "is_sharpe": sr.stats_is["sharpe"],
            "is_ann_return": sr.stats_is["ann_return"],
            "is_ann_vol": sr.stats_is["ann_vol"],
            "is_max_dd": sr.stats_is["max_dd"],
            "oos_sharpe": sr.stats_oos["sharpe"],
            "oos_ann_return": sr.stats_oos["ann_return"],
            "oos_ann_vol": sr.stats_oos["ann_vol"],
            "oos_max_dd": sr.stats_oos["max_dd"],
            "vol_scale_is": sr.vol_scale,
            "survivor": int(sr.stats_is["sharpe"] >= IS_SHARPE_MIN
                            and sr.stats_oos["sharpe"] >= OOS_SHARPE_MIN
                            and sr.vol_scale > 0
                            and np.isfinite(sr.vol_scale)),
        })
    bd = pd.DataFrame(rows)
    bd.to_csv(OUT_DIR / "trend_breakdown.csv", index=False)

    # -- print report -------------------------------------------------------
    split_ts = pd.Timestamp(SPLIT, tz="UTC")

    def block(name: str, r: pd.Series) -> None:
        st = _stats(r)
        print(f"  {name:<10} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print("=== Combined TREND sleeve (daily UTC) ===")
    if not sleeve.empty:
        block("FULL", sleeve)
        block("IS",   sleeve.loc[sleeve.index < split_ts])
        block("OOS",  sleeve.loc[sleeve.index >= split_ts])
        print("  Sharpe by year:")
        for yr, sub in sleeve.groupby(sleeve.index.year):
            st = _stats(sub)
            print(f"    {yr}  Sharpe={st['sharpe']:+5.2f}  "
                  f"Ret={st['ann_return']:+7.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    # Family-level aggregates (equal-weight combine within family across
    # survivors; if nothing survives in a family, report all-members combine).
    print("=== Per-family (vol-scaled, equal-weight within family) ===")
    families = bd["family"].unique().tolist()
    fam_rows = []
    for fam in families:
        # Take survivors only; if none, take all in family.
        fam_subs = [s for s in subs if s.name == fam
                    and s.stats_is["sharpe"] >= IS_SHARPE_MIN
                    and s.stats_oos["sharpe"] >= OOS_SHARPE_MIN
                    and s.vol_scale > 0 and np.isfinite(s.vol_scale)]
        is_survivor_set = bool(fam_subs)
        if not fam_subs:
            fam_subs = [s for s in subs if s.name == fam
                        and s.vol_scale > 0 and np.isfinite(s.vol_scale)]
        if not fam_subs:
            continue
        streams = []
        for sr in fam_subs:
            s = sr.returns * sr.vol_scale
            s_daily = s.groupby(s.index.normalize()).sum()
            if s_daily.index.tz is None:
                s_daily.index = s_daily.index.tz_localize("UTC")
            streams.append(s_daily)
        mat = pd.concat(streams, axis=1).fillna(0.0)
        fr = mat.mean(axis=1)
        full = _stats(fr)
        is_ = _stats(fr.loc[fr.index < split_ts])
        oos = _stats(fr.loc[fr.index >= split_ts])
        fam_rows.append({
            "family": fam, "n_members": len(fam_subs), "has_survivors": is_survivor_set,
            "is_sharpe": is_["sharpe"], "oos_sharpe": oos["sharpe"],
            "full_sharpe": full["sharpe"], "full_ann_vol": full["ann_vol"],
            "full_max_dd": full["max_dd"],
        })
    fam_df = pd.DataFrame(fam_rows)
    print(fam_df.to_string(index=False))

    # 2022 chop year
    if not sleeve.empty:
        s2022 = sleeve.loc[(sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                           (sleeve.index < pd.Timestamp("2023-01-01", tz="UTC"))]
        if len(s2022):
            st = _stats(s2022)
            print(f"\n2022 sleeve: Sharpe={st['sharpe']:+5.2f}  "
                  f"Ret={st['ann_return']:+7.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    # Top survivors
    if survivors:
        srv_rows = []
        for sr in survivors:
            srv_rows.append({
                "family": sr.name, "variant": sr.variant, "symbol": sr.symbol,
                "is_sharpe": sr.stats_is["sharpe"], "oos_sharpe": sr.stats_oos["sharpe"],
                "vol_scale": sr.vol_scale,
            })
        srv_df = pd.DataFrame(srv_rows).sort_values("oos_sharpe", ascending=False)
        print(f"\n=== Top 15 survivors by OOS Sharpe ===")
        print(srv_df.head(15).to_string(index=False))

    print(f"\nWrote {out_parquet}")
    print(f"Wrote {OUT_DIR / 'trend_breakdown.csv'}")


if __name__ == "__main__":
    main()
