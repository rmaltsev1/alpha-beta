"""Classic Moskowitz/AQR time-series momentum sleeve, D1.

For each symbol, for each lookback in {21, 63, 126, 252}:
  - signal[t]   = sign(sum of log returns over the trailing `L` bars ending at t-1)
  - vol scale   = TARGET_VOL / realized vol of *log returns* over trailing 60 bars
                  (estimator is walk-forward: uses bars strictly before t)
  - position[t] = clip(signal[t] * vol_scale[t], -POS_CAP, +POS_CAP)
Per-symbol TSMOM = mean of those 4 sub-sleeves (with their per-bar positions).
Sleeve = mean across the three asset-class baskets (crypto / fx / index),
which prevents the 6-name index basket from dominating just by count.

Outputs:
  scratch/quant/tsmom_returns.parquet  -- timestamp (UTC), ret (sleeve D1 return)
  scratch/quant/tsmom_breakdown.csv    -- per-symbol IS/OOS stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import backtest, cost_for, split_is_oos


# -- knobs --------------------------------------------------------------------
LOOKBACKS = [21, 63, 126, 252]
VOL_WIN = 60                # bars for trailing realized-vol estimator
TARGET_VOL = 0.10           # 10% annualized per (symbol, lookback) sub-sleeve
POS_CAP = 2.0               # safety cap on |position| after vol-targeting
SPLIT = "2024-01-01"
TF = "D1"
BARS_PER_YEAR = 365.25      # match backtest._bars_per_year for crypto-ish calendars
OUT_DIR = Path(__file__).resolve().parent


# -- helpers ------------------------------------------------------------------
def sleeve_stats(returns: pd.Series, freq: float) -> dict:
    """Sharpe / ann-return / max-DD on a per-bar return series."""
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
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
    }


def build_position(df: pd.DataFrame, lookback: int) -> pd.Series:
    """TSMOM position for one (symbol, lookback). Strictly walk-forward.

    Convention from alphabeta.backtest: position[t] uses info up to t-1.
    We compute trailing-sum log returns and trailing realized vol *of log
    returns* on the data, then shift by 1 before exposing them as signals.
    """
    close = df["close"].astype("float64").to_numpy()
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    s_log_ret = pd.Series(log_ret, index=df.index)

    # trailing sum of log returns over `lookback` bars, ending at bar t (uses
    # bars 1..t). Direction = sign of that sum.
    trail_sum = s_log_ret.rolling(lookback, min_periods=lookback).sum()
    raw_signal = np.sign(trail_sum)

    # Realized vol of *log returns* over trailing VOL_WIN bars, annualized.
    # min_periods=VOL_WIN -> NaN for first VOL_WIN bars: those become 0 below.
    rv = s_log_ret.rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=0) * np.sqrt(BARS_PER_YEAR)
    vol_scale = (TARGET_VOL / rv).where(rv > 0)

    # Position before look-ahead shift: sign * vol_scale, capped.
    pos_raw = (raw_signal * vol_scale).clip(-POS_CAP, POS_CAP).fillna(0.0)

    # Shift by 1 bar so position[t] uses info up to t-1 only.
    pos = pos_raw.shift(1).fillna(0.0)
    return pos


def per_symbol_sleeve(symbol: str) -> tuple[pd.DataFrame, dict]:
    """Returns:
       - DataFrame indexed by timestamp with columns: ret_<L> for each L, and 'ret_avg'.
       - dict with diagnostics per lookback + the average.
    """
    df = get_candles(symbol, TF)
    cps = cost_for(symbol)

    out_rets = {}
    diag = {}
    for L in LOOKBACKS:
        pos = build_position(df, L)
        res = backtest(df, pos, symbol=symbol, timeframe=TF, name=f"tsmom_{L}")
        out_rets[f"ret_{L}"] = res.returns.values
        diag[L] = res.stats

    rdf = pd.DataFrame(out_rets)
    rdf["ret_avg"] = rdf.mean(axis=1)
    rdf["timestamp"] = df["timestamp"].values
    rdf = rdf.set_index("timestamp")
    return rdf, diag


# -- driver -------------------------------------------------------------------
def main() -> None:
    per_sym: dict[str, pd.DataFrame] = {}
    per_sym_diag: dict[str, dict] = {}
    for s in ALL_SYMBOLS:
        rdf, diag = per_symbol_sleeve(s)
        per_sym[s] = rdf
        per_sym_diag[s] = diag

    # ---- union timestamp index across the whole basket ---------------------
    all_ts = sorted({pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC")
                     for rdf in per_sym.values() for ts in rdf.index})
    union_idx = pd.DatetimeIndex(all_ts)
    # Also normalize per-symbol indexes to tz-aware UTC for clean reindex.
    for s, rdf in per_sym.items():
        idx = pd.DatetimeIndex(rdf.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        rdf.index = idx

    # ---- group baskets (crypto / fx / index) -------------------------------
    groups = {"crypto": CRYPTO, "fx": FOREX, "index": INDEX}
    basket_rets: dict[str, pd.Series] = {}
    for gname, syms in groups.items():
        # Average per-symbol "ret_avg" within group; days with no symbol = 0.
        mat = pd.concat(
            [per_sym[s]["ret_avg"].reindex(union_idx).fillna(0.0).rename(s) for s in syms],
            axis=1,
        )
        # Per-day: average across symbols that *traded today* (have nonzero),
        # then fall back to simple mean if all zero. Simple mean is fine since
        # missing days are exactly 0 and would dilute equally for all symbols.
        basket_rets[gname] = mat.mean(axis=1)

    # ---- sleeve = mean across the 3 baskets --------------------------------
    # On the union (sub-)daily index different bars come from different
    # symbols (crypto closes 00:00 UTC, FX/Index ~21:00 UTC). For the
    # downstream master combiner we want ONE row per calendar day, so we
    # collapse multiple intra-day bars by *summing* per-day -- the per-bar
    # streams across symbols are zero on bars that aren't their own close, so
    # summing reproduces "average across baskets on this calendar day".
    sleeve_bar = pd.concat(basket_rets, axis=1).mean(axis=1)
    sleeve_bar.name = "ret"
    sleeve = sleeve_bar.groupby(sleeve_bar.index.normalize()).sum()
    sleeve.index = sleeve.index.tz_convert("UTC") if sleeve.index.tz else sleeve.index.tz_localize("UTC")
    sleeve.name = "ret"

    # ---- save sleeve parquet (D1, one row per calendar day) ---------------
    out_parquet = OUT_DIR / "tsmom_returns.parquet"
    out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(out_parquet, index=False)

    # ---- per-symbol breakdown CSV -----------------------------------------
    rows = []
    split_ts = pd.Timestamp(SPLIT, tz="UTC")
    for s in ALL_SYMBOLS:
        rdf = per_sym[s]
        idx = pd.to_datetime(rdf.index, utc=True)
        is_mask = idx < split_ts
        # Per-symbol "ret_avg" is the symbol's TSMOM stream on its own calendar.
        full = sleeve_stats(rdf["ret_avg"], BARS_PER_YEAR)
        is_  = sleeve_stats(rdf.loc[is_mask, "ret_avg"], BARS_PER_YEAR)
        oos  = sleeve_stats(rdf.loc[~is_mask, "ret_avg"], BARS_PER_YEAR)
        # Best lookback per symbol by full-period Sharpe.
        best_L, best_sharpe = max(
            ((L, per_sym_diag[s][L]["sharpe"]) for L in LOOKBACKS),
            key=lambda kv: kv[1],
        )
        rows.append({
            "symbol": s,
            "asset_class": SYMBOL_TYPE[s].value,
            "full_sharpe": full["sharpe"],
            "full_ann_return": full["ann_return"],
            "full_max_dd": full["max_dd"],
            "is_sharpe": is_["sharpe"],
            "is_ann_return": is_["ann_return"],
            "is_max_dd": is_["max_dd"],
            "oos_sharpe": oos["sharpe"],
            "oos_ann_return": oos["ann_return"],
            "oos_max_dd": oos["max_dd"],
            "best_lookback": best_L,
            "best_lookback_sharpe": best_sharpe,
            "sharpe_21":  per_sym_diag[s][21]["sharpe"],
            "sharpe_63":  per_sym_diag[s][63]["sharpe"],
            "sharpe_126": per_sym_diag[s][126]["sharpe"],
            "sharpe_252": per_sym_diag[s][252]["sharpe"],
        })
    bd = pd.DataFrame(rows)
    bd.to_csv(OUT_DIR / "tsmom_breakdown.csv", index=False)

    # ---- print a tidy report ----------------------------------------------
    def block(name: str, r: pd.Series) -> None:
        st = sleeve_stats(r, BARS_PER_YEAR)
        print(f"  {name:<6} Sharpe={st['sharpe']:+5.2f} "
              f"Ret={st['ann_return']:+7.2%} Vol={st['ann_vol']:6.2%} "
              f"DD={st['max_dd']:+7.2%} bars={len(r)}")

    is_mask  = sleeve.index < pd.Timestamp(SPLIT, tz="UTC")
    print("=== TSMOM SLEEVE (one row per calendar day) ===")
    block("FULL", sleeve)
    block("IS",   sleeve.loc[is_mask])
    block("OOS",  sleeve.loc[~is_mask])

    print("--- Sleeve Sharpe by year ---")
    by_year = sleeve.groupby(sleeve.index.year)
    for yr, sub in by_year:
        st = sleeve_stats(sub, BARS_PER_YEAR)
        print(f"  {yr}  Sharpe={st['sharpe']:+5.2f} "
              f"Ret={st['ann_return']:+7.2%} DD={st['max_dd']:+7.2%} n={len(sub)}")

    print("--- Basket stats (FULL, daily-aggregated) ---")
    for gname, r in basket_rets.items():
        r_daily = r.groupby(r.index.normalize()).sum()
        block(gname, r_daily)

    print("--- Best lookback per symbol (by FULL sharpe of sub-sleeve) ---")
    print(bd[["symbol","asset_class","best_lookback","best_lookback_sharpe",
              "sharpe_21","sharpe_63","sharpe_126","sharpe_252"]]
          .to_string(index=False))

    print(f"\nWrote {out_parquet}")
    print(f"Wrote {OUT_DIR / 'tsmom_breakdown.csv'}")


if __name__ == "__main__":
    main()
