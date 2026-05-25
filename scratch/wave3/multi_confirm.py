"""Multi-symbol momentum-confirmation sleeve (wave 3).

Idea: only trade a symbol's momentum signal when correlated symbols in the
same basket ALSO show the same momentum. The cross-basket agreement should
filter out single-symbol noise and amplify durable regime-level trends.

Six sub-sleeves:

  1. crypto_basket    -- long all of BTC/ETH/SOL when 63d_ret > 0 for ALL 3,
                         short all when 63d_ret < 0 for ALL 3, flat else.
                         5-day minimum hold.
  2. us_equity_basket -- long all of SPX/NAS/US30 when 126d > 0 AND 21d > 0
                         for ALL 3 (combined ensemble, long-only).
  3. fx_usd_basket    -- long all of EUR_USD + GBP_USD + USD_JPY (long-USD
                         after flipping the inverse pairs) when >= 2/3 have
                         positive trailing 63d return.
  4. cross_basket_rel -- long crypto basket when crypto_basket_63d_ret >
                         index_basket_63d_ret AND both > 0. ("crypto leading
                         in an up market" regime.)
  5. safe_haven       -- long XAU + USD_JPY when SPX 21d < 0 AND SPX 30d vol
                         > IS 70th percentile. Coordinated risk-off bet.
  6. crash_continue   -- when ALL 6 equity indices have 21d < -2%, short the
                         equity basket for 10 days. Coordinated crash trade.

Methodology:
  - IS  <= 2024-01-01, OOS >= 2024-01-01.
  - Vol-scale each sub-sleeve to 5% IS ann vol (using IS-only stats).
  - Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
  - Combine survivors equal-weight on the UTC daily calendar.

Outputs:
  scratch/wave3/multi_confirm_returns.parquet  -- combined sleeve, daily UTC
  scratch/wave3/multi_confirm_breakdown.csv    -- per-sub-sleeve stats
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, CRYPTO, INDEX
from alphabeta.backtest import backtest, cost_for


# -- knobs --------------------------------------------------------------------
SPLIT = "2024-01-01"
TF = "D1"
BPY = 365.25
IS_VOL_TARGET = 0.05
IS_SHARPE_MIN = 0.5
OOS_SHARPE_MIN = 0.0
OUT_DIR = Path(__file__).resolve().parent

US_EQUITY = ["SPX500_USD", "NAS100_USD", "US30_USD"]
ALL_EQUITY = ["SPX500_USD", "NAS100_USD", "US30_USD",
              "UK100_GBP", "DE30_EUR", "JP225_USD"]
USD_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]  # sign flips applied below


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


def _to_utc_daily(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Index a per-bar series by UTC calendar day (one bar per day on D1)."""
    out = pd.Series(s.values, index=pd.DatetimeIndex(df["timestamp"]))
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    # collapse to one row per calendar day
    out = out.groupby(out.index.normalize()).sum()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out


def _vol_scale_is(ret: pd.Series, split_ts: pd.Timestamp, target: float = IS_VOL_TARGET) -> float:
    is_r = ret.loc[ret.index < split_ts].dropna()
    if is_r.empty:
        return 0.0
    sd = is_r.std(ddof=0) * np.sqrt(BPY)
    if sd <= 0:
        return 0.0
    return float(target / sd)


def _enforce_min_hold(pos: pd.Series, min_hold: int) -> pd.Series:
    """When position changes to a non-zero value, force it to hold for `min_hold`
    bars regardless of fresh signal. Re-entries after a flat segment trigger a
    fresh hold window. Vectorization is awkward; loop is fine on D1 lengths."""
    if min_hold <= 1:
        return pos
    p = pos.to_numpy(dtype="float64", copy=True)
    n = len(p)
    held = 0
    cur = 0.0
    for t in range(n):
        if held > 0:
            p[t] = cur
            held -= 1
        else:
            if p[t] != 0.0:
                cur = p[t]
                held = min_hold - 1
            else:
                cur = 0.0
    return pd.Series(p, index=pos.index)


def _hold_n_after_trigger(trigger: pd.Series, hold: int, sign: float) -> pd.Series:
    """When `trigger` is True at bar t, set the next `hold` bars to `sign`.
    Overlapping triggers reset the hold window. Trigger uses info up to t-1 in
    callers (we shift inside)."""
    p = np.zeros(len(trigger), dtype="float64")
    t_arr = trigger.fillna(False).to_numpy().astype(bool)
    remaining = 0
    for t in range(len(p)):
        if t_arr[t]:
            remaining = hold
        if remaining > 0:
            p[t] = sign
            remaining -= 1
    return pd.Series(p, index=trigger.index)


# -- daily-aligned price utilities -------------------------------------------
def _daily_close(df: pd.DataFrame) -> pd.Series:
    """Per-symbol close indexed by UTC calendar day."""
    s = pd.Series(df["close"].astype("float64").values,
                  index=pd.DatetimeIndex(df["timestamp"]))
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    # take the last bar of each calendar day (already 1/day on D1, but safe)
    s = s.groupby(s.index.normalize()).last()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


# -- sub-sleeve builders ------------------------------------------------------
@dataclass
class Sleeve:
    name: str
    returns: pd.Series          # UTC-daily pre-vol-scale net returns
    vol_scale: float
    stats_full: dict
    stats_is: dict
    stats_oos: dict


def _sleeve_from_positions(name: str,
                           positions: dict[str, pd.Series],
                           dfs: dict[str, pd.DataFrame],
                           split_ts: pd.Timestamp) -> Sleeve:
    """Backtest each symbol's position, equal-weight combine on UTC daily, build Sleeve."""
    streams = []
    for sym, pos in positions.items():
        df = dfs[sym]
        # Make sure position is bar-aligned with df (engine asserts equal length)
        pos = pos.reindex(df.index).fillna(0.0)
        # Already shifted by builder; engine just multiplies into bar_ret.
        res = backtest(df, pos, symbol=sym, timeframe=TF, name=f"{name}:{sym}")
        daily = _to_utc_daily(res.returns, df)
        streams.append(daily)
    if not streams:
        ret = pd.Series(dtype="float64")
    else:
        mat = pd.concat(streams, axis=1).fillna(0.0)
        # equal-weight within the sub-sleeve (the sub-sleeve already imposes the
        # *gate*; here we simply average the gated per-symbol return streams).
        ret = mat.mean(axis=1)
    full = _stats(ret)
    is_ = _stats(ret.loc[ret.index < split_ts]) if not ret.empty else _stats(ret)
    oos = _stats(ret.loc[ret.index >= split_ts]) if not ret.empty else _stats(ret)
    vs = _vol_scale_is(ret, split_ts) if not ret.empty else 0.0
    return Sleeve(name=name, returns=ret, vol_scale=vs,
                  stats_full=full, stats_is=is_, stats_oos=oos)


# 1. CRYPTO BASKET ------------------------------------------------------------
def sleeve_crypto_basket(dfs: dict[str, pd.DataFrame],
                         split_ts: pd.Timestamp) -> Sleeve:
    closes = {s: _daily_close(dfs[s]) for s in CRYPTO}
    # Align all on union of dates.
    aligned = pd.concat(closes, axis=1).ffill()
    r63 = aligned.pct_change(63)
    all_pos = (r63 > 0).all(axis=1)
    all_neg = (r63 < 0).all(axis=1)
    basket = pd.Series(0.0, index=aligned.index)
    basket[all_pos] = 1.0
    basket[all_neg] = -1.0
    # Use only info up to t-1 to set position on bar t.
    basket = basket.shift(1).fillna(0.0)
    # Minimum 5-day hold.
    basket = _enforce_min_hold(basket, min_hold=5)
    # Map onto each symbol's own bar index.
    positions = {}
    for s in CRYPTO:
        df = dfs[s]
        day_keys = _daily_close(df).index  # canonical daily index for this symbol
        pos_daily = basket.reindex(day_keys).fillna(0.0)
        # Now map back to the df's bar index (1-to-1 for D1).
        sym_ts = pd.DatetimeIndex(df["timestamp"])
        if sym_ts.tz is None:
            sym_ts = sym_ts.tz_localize("UTC")
        else:
            sym_ts = sym_ts.tz_convert("UTC")
        day_lookup = sym_ts.normalize()
        pos_on_bars = pd.Series(pos_daily.reindex(day_lookup).fillna(0.0).values,
                                index=df.index)
        positions[s] = pos_on_bars
    return _sleeve_from_positions("crypto_basket", positions, dfs, split_ts)


# 2. US EQUITY BASKET ---------------------------------------------------------
def sleeve_us_equity_basket(dfs: dict[str, pd.DataFrame],
                             split_ts: pd.Timestamp) -> Sleeve:
    closes = {s: _daily_close(dfs[s]) for s in US_EQUITY}
    aligned = pd.concat(closes, axis=1).ffill()
    r126 = aligned.pct_change(126)
    r21 = aligned.pct_change(21)
    long_gate = ((r126 > 0).all(axis=1) & (r21 > 0).all(axis=1)).astype(float)
    # Long-only, shifted to use t-1 info.
    basket = long_gate.shift(1).fillna(0.0)
    positions = {}
    for s in US_EQUITY:
        df = dfs[s]
        sym_ts = pd.DatetimeIndex(df["timestamp"])
        if sym_ts.tz is None:
            sym_ts = sym_ts.tz_localize("UTC")
        else:
            sym_ts = sym_ts.tz_convert("UTC")
        day_lookup = sym_ts.normalize()
        pos = pd.Series(basket.reindex(day_lookup).fillna(0.0).values, index=df.index)
        positions[s] = pos
    return _sleeve_from_positions("us_equity_basket", positions, dfs, split_ts)


# 3. FX USD BASKET ------------------------------------------------------------
def sleeve_fx_usd_basket(dfs: dict[str, pd.DataFrame],
                         split_ts: pd.Timestamp) -> Sleeve:
    """Long-USD basket. EUR_USD/GBP_USD are USD-quoted (price up = USD down), so
    to express "long USD" we flip the sign on those. USD_JPY is USD-base, sign
    stays +1. The basket activates (sign +1 on the USD direction) when >=2/3
    pairs show positive 63d USD-strength momentum."""
    sign_map = {"EUR_USD": -1.0, "GBP_USD": -1.0, "USD_JPY": +1.0}
    closes = {s: _daily_close(dfs[s]) for s in USD_PAIRS}
    aligned = pd.concat(closes, axis=1).ffill()
    # USD-strength return = sign_map[s] * pair_ret.
    usd_strength_63 = pd.DataFrame({s: sign_map[s] * aligned[s].pct_change(63)
                                     for s in USD_PAIRS})
    gate = (usd_strength_63 > 0).sum(axis=1) >= 2
    long_usd = gate.astype(float).shift(1).fillna(0.0)
    positions = {}
    for s in USD_PAIRS:
        df = dfs[s]
        sym_ts = pd.DatetimeIndex(df["timestamp"])
        if sym_ts.tz is None:
            sym_ts = sym_ts.tz_localize("UTC")
        else:
            sym_ts = sym_ts.tz_convert("UTC")
        day_lookup = sym_ts.normalize()
        gate_on_bars = pd.Series(long_usd.reindex(day_lookup).fillna(0.0).values,
                                 index=df.index)
        # Apply the per-pair sign so we are net long USD.
        positions[s] = gate_on_bars * sign_map[s]
    return _sleeve_from_positions("fx_usd_basket", positions, dfs, split_ts)


# 4. CROSS-BASKET RELATIVE CONFIRMATION ---------------------------------------
def sleeve_cross_basket_rel(dfs: dict[str, pd.DataFrame],
                             split_ts: pd.Timestamp) -> Sleeve:
    """Long crypto basket only when crypto_basket_63d > index_basket_63d AND
    both are > 0. Index basket = SPX/NAS/US30 equal-weight."""
    crypto_closes = pd.concat({s: _daily_close(dfs[s]) for s in CRYPTO}, axis=1).ffill()
    eq_closes = pd.concat({s: _daily_close(dfs[s]) for s in US_EQUITY}, axis=1).ffill()
    # Equal-weight basket "price" via cumulative simple returns.
    crypto_ret = crypto_closes.pct_change().mean(axis=1)
    eq_ret = eq_closes.pct_change().mean(axis=1)
    crypto_idx = (1 + crypto_ret.fillna(0.0)).cumprod()
    eq_idx = (1 + eq_ret.fillna(0.0)).cumprod()
    crypto_63 = crypto_idx.pct_change(63)
    eq_63 = eq_idx.pct_change(63)
    # Align both series onto a common UTC daily index before comparison.
    both = pd.concat({"c": crypto_63, "e": eq_63}, axis=1).ffill()
    c63 = both["c"]
    e63 = both["e"]
    gate = ((c63 > e63) & (c63 > 0) & (e63 > 0)).astype(float)
    gate = gate.shift(1).fillna(0.0)
    positions = {}
    for s in CRYPTO:
        df = dfs[s]
        sym_ts = pd.DatetimeIndex(df["timestamp"])
        if sym_ts.tz is None:
            sym_ts = sym_ts.tz_localize("UTC")
        else:
            sym_ts = sym_ts.tz_convert("UTC")
        day_lookup = sym_ts.normalize()
        pos = pd.Series(gate.reindex(day_lookup).fillna(0.0).values, index=df.index)
        positions[s] = pos
    return _sleeve_from_positions("cross_basket_rel", positions, dfs, split_ts)


# 5. SAFE-HAVEN CONFIRMATION --------------------------------------------------
def sleeve_safe_haven(dfs: dict[str, pd.DataFrame],
                       split_ts: pd.Timestamp) -> Sleeve:
    """Long XAU + USD_JPY when SPX 21d < 0 AND SPX 30d vol > IS 70th pctile."""
    spx = _daily_close(dfs["SPX500_USD"])
    spx_ret = np.log(spx).diff()
    spx_21 = spx.pct_change(21)
    spx_vol30 = spx_ret.rolling(30, min_periods=30).std(ddof=0)
    # IS percentile: compute on IS slice only, single scalar threshold (no future leak).
    is_vol = spx_vol30.loc[spx_vol30.index < split_ts].dropna()
    if len(is_vol) == 0:
        threshold = np.inf
    else:
        threshold = float(np.quantile(is_vol.values, 0.70))
    gate = ((spx_21 < 0) & (spx_vol30 > threshold)).astype(float).shift(1).fillna(0.0)
    havens = ["XAU_USD", "USD_JPY"]
    positions = {}
    for s in havens:
        df = dfs[s]
        sym_ts = pd.DatetimeIndex(df["timestamp"])
        if sym_ts.tz is None:
            sym_ts = sym_ts.tz_localize("UTC")
        else:
            sym_ts = sym_ts.tz_convert("UTC")
        day_lookup = sym_ts.normalize()
        positions[s] = pd.Series(gate.reindex(day_lookup).fillna(0.0).values, index=df.index)
    return _sleeve_from_positions("safe_haven", positions, dfs, split_ts)


# 6. ALL-EQUITY CRASH CONTINUATION --------------------------------------------
def sleeve_crash_continue(dfs: dict[str, pd.DataFrame],
                           split_ts: pd.Timestamp) -> Sleeve:
    """When ALL 6 equity indices have 21d return < -2%, short the equity basket
    for the next 10 days. Re-trigger resets the window."""
    closes = pd.concat({s: _daily_close(dfs[s]) for s in ALL_EQUITY}, axis=1).ffill()
    r21 = closes.pct_change(21)
    trigger = (r21 < -0.02).all(axis=1)
    # Use only past info: shift trigger by 1 so position[t] is set from t-1 info.
    trigger = trigger.shift(1).fillna(False)
    short_window = _hold_n_after_trigger(trigger, hold=10, sign=-1.0)
    positions = {}
    for s in ALL_EQUITY:
        df = dfs[s]
        sym_ts = pd.DatetimeIndex(df["timestamp"])
        if sym_ts.tz is None:
            sym_ts = sym_ts.tz_localize("UTC")
        else:
            sym_ts = sym_ts.tz_convert("UTC")
        day_lookup = sym_ts.normalize()
        positions[s] = pd.Series(short_window.reindex(day_lookup).fillna(0.0).values,
                                  index=df.index)
    return _sleeve_from_positions("crash_continue", positions, dfs, split_ts)


# -- driver -------------------------------------------------------------------
def main() -> None:
    print("Loading D1 data...")
    needed = sorted(set(CRYPTO + US_EQUITY + ALL_EQUITY + USD_PAIRS + ["XAU_USD"]))
    dfs = {s: get_candles(s, TF) for s in needed}

    split_ts = pd.Timestamp(SPLIT, tz="UTC")

    print("Building sub-sleeves...")
    sleeves = [
        sleeve_crypto_basket(dfs, split_ts),
        sleeve_us_equity_basket(dfs, split_ts),
        sleeve_fx_usd_basket(dfs, split_ts),
        sleeve_cross_basket_rel(dfs, split_ts),
        sleeve_safe_haven(dfs, split_ts),
        sleeve_crash_continue(dfs, split_ts),
    ]

    # -- breakdown CSV (per sub-sleeve, raw + vol-scaled) ---------------------
    rows = []
    for sl in sleeves:
        scaled = sl.returns * sl.vol_scale if sl.vol_scale > 0 else sl.returns * 0.0
        sf = _stats(scaled)
        si = _stats(scaled.loc[scaled.index < split_ts])
        so = _stats(scaled.loc[scaled.index >= split_ts])
        s22 = scaled.loc[(scaled.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                         (scaled.index < pd.Timestamp("2023-01-01", tz="UTC"))]
        s22_stats = _stats(s22)
        survivor = int(sl.stats_is["sharpe"] >= IS_SHARPE_MIN
                       and sl.stats_oos["sharpe"] >= OOS_SHARPE_MIN
                       and sl.vol_scale > 0
                       and np.isfinite(sl.vol_scale))
        rows.append({
            "sleeve": sl.name,
            "is_sharpe_raw": sl.stats_is["sharpe"],
            "oos_sharpe_raw": sl.stats_oos["sharpe"],
            "full_sharpe_raw": sl.stats_full["sharpe"],
            "is_ann_vol_raw": sl.stats_is["ann_vol"],
            "vol_scale_is": sl.vol_scale,
            "full_sharpe_scaled": sf["sharpe"],
            "full_ann_return_scaled": sf["ann_return"],
            "full_ann_vol_scaled": sf["ann_vol"],
            "full_max_dd_scaled": sf["max_dd"],
            "is_sharpe_scaled": si["sharpe"],
            "is_ann_return_scaled": si["ann_return"],
            "is_max_dd_scaled": si["max_dd"],
            "oos_sharpe_scaled": so["sharpe"],
            "oos_ann_return_scaled": so["ann_return"],
            "oos_max_dd_scaled": so["max_dd"],
            "y2022_sharpe": s22_stats["sharpe"],
            "y2022_ann_return": s22_stats["ann_return"],
            "y2022_max_dd": s22_stats["max_dd"],
            "survivor": survivor,
            "n_obs": sl.stats_full["n"],
        })
    bd = pd.DataFrame(rows)
    bd.to_csv(OUT_DIR / "multi_confirm_breakdown.csv", index=False)

    # -- combine survivors equal-weight ---------------------------------------
    survivors = [sl for sl in sleeves
                 if sl.stats_is["sharpe"] >= IS_SHARPE_MIN
                 and sl.stats_oos["sharpe"] >= OOS_SHARPE_MIN
                 and sl.vol_scale > 0 and np.isfinite(sl.vol_scale)]
    print(f"  survivors (IS>={IS_SHARPE_MIN}, OOS>={OOS_SHARPE_MIN}): "
          f"{[s.name for s in survivors]}")

    if survivors:
        streams = []
        for sl in survivors:
            r = sl.returns * sl.vol_scale
            r.name = sl.name
            streams.append(r)
        mat = pd.concat(streams, axis=1).fillna(0.0)
        sleeve = mat.mean(axis=1)
        sleeve.name = "ret"
    else:
        sleeve = pd.Series(dtype="float64", name="ret")

    out_parquet = OUT_DIR / "multi_confirm_returns.parquet"
    if not sleeve.empty:
        out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    else:
        out_df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[us, UTC]"),
                               "ret": pd.Series(dtype="float64")})
    out_df.to_parquet(out_parquet, index=False)

    # -- print report ---------------------------------------------------------
    def block(label: str, r: pd.Series) -> None:
        st = _stats(r)
        print(f"  {label:<6} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print("\n=== Combined MULTI_CONFIRM sleeve ===")
    if not sleeve.empty:
        block("FULL", sleeve)
        block("IS",   sleeve.loc[sleeve.index < split_ts])
        block("OOS",  sleeve.loc[sleeve.index >= split_ts])
        s22 = sleeve.loc[(sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                         (sleeve.index < pd.Timestamp("2023-01-01", tz="UTC"))]
        block("2022", s22)
        print("  Sharpe by year:")
        for yr, sub in sleeve.groupby(sleeve.index.year):
            st = _stats(sub)
            print(f"    {yr}  Sharpe={st['sharpe']:+5.2f}  "
                  f"Ret={st['ann_return']:+7.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")
    else:
        print("  (no survivors)")

    print("\n=== Per-sub-sleeve breakdown (vol-scaled) ===")
    print(bd[["sleeve", "is_sharpe_scaled", "oos_sharpe_scaled",
              "full_sharpe_scaled", "y2022_sharpe", "vol_scale_is",
              "survivor"]].to_string(index=False))

    print(f"\nWrote {out_parquet}")
    print(f"Wrote {OUT_DIR / 'multi_confirm_breakdown.csv'}")


if __name__ == "__main__":
    main()
