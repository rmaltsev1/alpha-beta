"""Volatility Risk Premium (VRP) proxy strategy — wave 6.

Classic VRP = "sell options, collect premium, hedge tail." We have no options,
so we approximate the long-side: sell realized vol (i.e., bet on vol coming
down) when an implied-vol-proxy is elevated.

Five sub-strategies:
  1. VRP long (sell vol when high) — inverse-vol-position when RV > 80th pctile
  2. Realized-vs-implied trade — long calm after vol-expansion event
  3. Vol-of-vol spread — short eq when 30d RV rising fast, long when falling
  4. Term-structure proxy — 7d RV vs 30d RV mean-reversion
  5. Crypto perpetual basis proxy — short BTC after >15% 5d rally

Methodology:
  IS  < 2024-01-01;  OOS >= 2024-01-01.
  Walk-forward vol estimates (rolling, lagged).
  Vol-scale each sub-sleeve to 5% IS ann vol.
  Survivor filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.
  Combine equal-weight.

Outputs:
  scratch/wave6/vrp_returns.parquet
  scratch/wave6/vrp_breakdown.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles
from alphabeta.backtest import cost_for
from alphabeta.symbols import CRYPTO, FOREX, INDEX

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05

ALL_ASSETS = CRYPTO + FOREX + INDEX  # 13 symbols
EQUITY_BASKET = ["SPX500_USD", "NAS100_USD", "US30_USD",
                 "DE30_EUR", "UK100_GBP", "JP225_USD"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def sharpe_of(r: pd.Series) -> float:
    if len(r) < 5:
        return 0.0
    bpy = _bpy(r.index)
    av = r.std(ddof=0) * np.sqrt(bpy)
    if av <= 0:
        return 0.0
    return float(r.mean() * bpy / av)


def ann_ret_of(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    return float(r.mean() * _bpy(r.index))


def ann_vol_of(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(_bpy(r.index)))


def max_dd_of(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def stats_block(label: str, r: pd.Series) -> dict:
    out = {"name": label}
    for tag, sub in [
        ("FULL", r),
        ("IS",   r[r.index <  SPLIT]),
        ("OOS",  r[r.index >= SPLIT]),
        ("2020", r[r.index.year == 2020]),
        ("2021", r[r.index.year == 2021]),
        ("2022", r[r.index.year == 2022]),
        ("2023", r[r.index.year == 2023]),
        ("2024", r[r.index.year == 2024]),
        ("2025", r[r.index.year == 2025]),
    ]:
        out[f"{tag}_sharpe"] = sharpe_of(sub)
        out[f"{tag}_ret"] = ann_ret_of(sub)
        out[f"{tag}_dd"] = max_dd_of(sub)
    return out


def vol_scale_is(r: pd.Series, target: float = TARGET_VOL) -> float:
    is_ret = r[r.index < SPLIT]
    av = ann_vol_of(is_ret)
    return target / av if av > 1e-9 else 0.0


def to_ts_series(df: pd.DataFrame, vals: np.ndarray) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"].values, utc=True))
    return pd.Series(vals, index=idx).sort_index()


def cost_apply(pos: pd.Series, ret: pd.Series, cps: float) -> pd.Series:
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    return pos * ret - dpos * cps


def _load_d1(symbol: str) -> pd.DataFrame:
    return get_candles(symbol, "D1").reset_index(drop=True)


def _ret_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        df["close"].pct_change().fillna(0).values,
        index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)),
    )


# ---------------------------------------------------------------------------
# 1. VRP long — sell-vol-when-high (inverse-vol position)
#
#   Predict next-day RV with an EWMA. When 30d RV > rolling 80th-pctile of
#   itself (vol regime "high"), take long position sized inversely to the
#   predicted vol. When vol normalizes the realized vol drops and the long
#   position (kept on through the calm-down) profits from drift. Net: a
#   stylized "short vol" sleeve that earns when RV mean-reverts down.
# ---------------------------------------------------------------------------

def strat1_vrp_inverse_vol(symbol: str,
                           rv_win: int = 30,
                           ewma_span: int = 10,
                           pct_win: int = 252,
                           pct: float = 0.80) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < pct_win + 60:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)

    # 30d realized vol (annualized)
    rv30 = ret.rolling(rv_win).std() * np.sqrt(252)
    # EWMA predicted vol (also annualized) — walk-forward, then lag by 1
    ewma_vol = ret.ewm(span=ewma_span, adjust=False).std() * np.sqrt(252)
    pred_vol = ewma_vol.shift(1)

    # high-vol regime threshold: rolling 80th pctile of past rv30, lagged
    thresh = rv30.rolling(pct_win, min_periods=126).quantile(pct).shift(1)
    rv30_lag = rv30.shift(1)
    high_vol = (rv30_lag > thresh)

    # Position inversely scaled to predicted vol, gated by high-vol regime.
    # Target a fixed "vol-budget" — when pred_vol is small the position is
    # large, when large the position shrinks.
    target_pred = 0.20  # 20% ann vol budget
    raw_size = (target_pred / pred_vol.replace(0, np.nan)).clip(0, 1.0)
    pos = np.where(high_vol & raw_size.notna(), raw_size, 0.0)
    pos = pd.Series(pos, index=close.index).fillna(0)

    cps = cost_for(symbol)
    sleeve = cost_apply(pos, ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 2. Realized-vs-implied — long-the-calm after a vol-expansion event
#
#   When today's |ret| >> 21-day mean |ret| (a vol-expansion event), expect
#   next-day vol to come back down. Be PAID to be long the calm: buy
#   underlying with vol-matched size = 1 / predicted_vol, capped.
# ---------------------------------------------------------------------------

def strat2_realized_vs_implied(symbol: str,
                               mean_win: int = 21,
                               expansion_mult: float = 2.0,
                               hold_days: int = 1,
                               ewma_span: int = 10) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < mean_win + 60:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    abs_ret = ret.abs()

    # rolling avg |ret| over 21d, lagged
    avg_abs = abs_ret.rolling(mean_win).mean().shift(1)
    abs_ret_lag = abs_ret.shift(1)
    expansion = (abs_ret_lag > expansion_mult * avg_abs)

    # predicted vol (EWMA) — walk-forward, lagged
    pred_vol = (ret.ewm(span=ewma_span, adjust=False).std() *
                np.sqrt(252)).shift(1)
    target = 0.20
    size = (target / pred_vol.replace(0, np.nan)).clip(0, 1.0)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    exp_vals = expansion.values
    size_vals = size.fillna(0).values
    for t in range(n):
        if hold > 0:
            pos[t] = pos[t - 1]
            hold -= 1
        elif exp_vals[t] and size_vals[t] > 0:
            pos[t] = size_vals[t]
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 3. Vol-of-vol spread — short when 30d RV rising fast, long when falling
#
#   vol_of_vol = std of 30d RV over last 10 days. Sign of (rv30 - rv30.shift(5))
#   gives the direction of vol motion. Short equity when accelerating up,
#   long when accelerating down. Bet on vol normalization.
# ---------------------------------------------------------------------------

def strat3_vol_of_vol(symbol: str,
                      rv_win: int = 30,
                      delta_win: int = 5,
                      vov_win: int = 10,
                      vov_thresh_pct: float = 0.70) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < rv_win + 252:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    rv30 = ret.rolling(rv_win).std() * np.sqrt(252)
    rv_change = (rv30 - rv30.shift(delta_win))
    vov = rv30.rolling(vov_win).std()

    # rolling pctile threshold for "high" vol-of-vol, lagged
    vov_thresh = vov.rolling(252, min_periods=126).quantile(vov_thresh_pct).shift(1)
    vov_lag = vov.shift(1)
    rv_change_lag = rv_change.shift(1)

    high_vov = vov_lag > vov_thresh
    pos = np.where(high_vov & (rv_change_lag > 0), -1.0,
            np.where(high_vov & (rv_change_lag < 0),  1.0, 0.0))
    pos = pd.Series(pos, index=close.index).fillna(0)

    cps = cost_for(symbol)
    sleeve = cost_apply(pos, ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 4. Term-structure proxy — 7d RV vs 30d RV mean reversion
#
#   ratio = 7d RV / 30d RV.
#     ratio > 1.5  -> short-term vol stress; expect mean reversion; LONG
#     ratio < 0.7  -> too compressed; expect expansion; SHORT (small, this
#     is the risk-bearing leg)
# ---------------------------------------------------------------------------

def strat4_term_structure(symbol: str,
                          short_win: int = 7,
                          long_win: int = 30,
                          high_thresh: float = 1.5,
                          low_thresh: float = 0.7,
                          hold_days: int = 5) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < long_win + 60:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    rv_s = ret.rolling(short_win).std() * np.sqrt(252)
    rv_l = ret.rolling(long_win).std() * np.sqrt(252)
    ratio = (rv_s / rv_l).shift(1)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    rvals = ratio.values
    for t in range(n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
            continue
        v = rvals[t]
        if np.isnan(v):
            continue
        if v > high_thresh:
            sign = 1.0
            pos[t] = sign
            hold = hold_days - 1
        elif v < low_thresh:
            sign = -0.5  # smaller short — this is risk-bearing
            pos[t] = sign
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 5. Crypto perpetual basis proxy — short BTC after >15% 5d rally
#
#   When BTC 5d return > 15%, perp funding typically goes positive (long bias)
#   and over-leveraged longs unwind. Short BTC for 3 days.
# ---------------------------------------------------------------------------

def strat5_crypto_basis(symbol: str = "BTCUSDT",
                        ret_thresh: float = 0.15,
                        hold_days: int = 3) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < 60:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    ret5_lag = (close / close.shift(5) - 1).shift(1)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    for t in range(n):
        if hold > 0:
            pos[t] = -1.0
            hold -= 1
        elif (not pd.isna(ret5_lag.iloc[t])) and ret5_lag.iloc[t] > ret_thresh:
            pos[t] = -1.0
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# Build all candidates, score, filter, combine.
# ---------------------------------------------------------------------------

# For the per-asset strategies, run across the equity basket + crypto majors
# (vol mean-reversion is cleanest in mature markets). Forex tends to have
# weaker vol cycles but we include majors for completeness.
PER_ASSET_UNIVERSE = (EQUITY_BASKET +
                      ["BTCUSDT", "ETHUSDT", "SOLUSDT"] +
                      ["XAU_USD", "USD_JPY", "EUR_USD", "GBP_USD"])


def build_candidates() -> dict[str, pd.Series]:
    cands: dict[str, pd.Series] = {}
    for sym in PER_ASSET_UNIVERSE:
        try:
            cands[f"1_vrp_inv_vol__{sym}"] = strat1_vrp_inverse_vol(sym)
            cands[f"2_real_vs_impl__{sym}"] = strat2_realized_vs_implied(sym)
            cands[f"3_vol_of_vol__{sym}"]   = strat3_vol_of_vol(sym)
            cands[f"4_term_struct__{sym}"]  = strat4_term_structure(sym)
        except FileNotFoundError:
            continue
    # crypto basis: just BTC (and ETH as bonus)
    cands["5_crypto_basis__BTCUSDT"] = strat5_crypto_basis("BTCUSDT")
    cands["5_crypto_basis__ETHUSDT"] = strat5_crypto_basis("ETHUSDT")
    return cands


def asset_class_of(name: str) -> str:
    sym = name.split("__")[-1]
    if sym in CRYPTO:
        return "crypto"
    if sym in FOREX:
        return "forex"
    if sym in INDEX:
        return "index"
    return "unknown"


def strat_family(name: str) -> str:
    return name.split("__")[0]


def main():
    candidates = build_candidates()
    print(f"\nBuilt {len(candidates)} candidate sleeves across "
          f"{len(PER_ASSET_UNIVERSE)} assets.")

    rows = []
    scaled_streams = {}
    print(f"\n{'name':<30} {'IS_Sh':>6} {'OOS_Sh':>6} {'2022_Sh':>7} "
          f"{'2022_Ret':>8} {'OOS_Ret':>8} {'scale':>6}")
    print("-" * 90)
    for name, raw in candidates.items():
        raw = raw.dropna().sort_index()
        if len(raw) < 10:
            continue
        scale = vol_scale_is(raw, TARGET_VOL)
        if scale == 0.0 or not np.isfinite(scale):
            continue
        scaled = raw * scale
        scaled_streams[name] = scaled
        s = stats_block(name, scaled)
        s["raw_is_vol"] = ann_vol_of(raw[raw.index < SPLIT])
        s["scale"] = float(scale)
        s["family"] = strat_family(name)
        s["asset_class"] = asset_class_of(name)
        rows.append(s)
        print(f"{name:<30} {s['IS_sharpe']:>+6.2f} {s['OOS_sharpe']:>+6.2f} "
              f"{s['2022_sharpe']:>+7.2f} {s['2022_ret']:>+8.2%} "
              f"{s['OOS_ret']:>+8.2%} {scale:>6.2f}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "vrp_breakdown.csv", index=False)

    # ---- Survivor selection ----
    survivors = []
    for s in rows:
        is_ok = s["IS_sharpe"] >= 0.4
        oos_ok = s["OOS_sharpe"] >= 0.0
        if is_ok and oos_ok:
            survivors.append(s["name"])
    print(f"\nSurvivors (IS >= 0.4 AND OOS >= 0):  ({len(survivors)})")
    for n in survivors:
        print(f"   {n}")

    # ---- Combined equal-weight (rescaled to 5% IS vol) ----
    if survivors:
        aligned = pd.concat(
            [scaled_streams[n].rename(n) for n in survivors], axis=1
        ).sort_index().fillna(0)
        combined = aligned.mean(axis=1)
        scl = vol_scale_is(combined, TARGET_VOL)
        combined_scaled = combined * scl
    else:
        # fallback: take top-3 by IS Sharpe (so we still produce *something*)
        ranked = sorted(rows, key=lambda r: r["IS_sharpe"], reverse=True)[:3]
        top = [r["name"] for r in ranked]
        print(f"\nNo strict survivors — falling back to top-3 IS: {top}")
        aligned = pd.concat(
            [scaled_streams[n].rename(n) for n in top], axis=1
        ).sort_index().fillna(0)
        combined = aligned.mean(axis=1)
        scl = vol_scale_is(combined, TARGET_VOL)
        combined_scaled = combined * scl
        survivors = top  # for reporting

    print(f"\nCombined sleeve (equal-weight, rescaled to 5% IS vol):")
    cs = stats_block("combined", combined_scaled)
    for tag in ["IS","OOS","2020","2021","2022","2023","2024","2025"]:
        print(f"  {tag:<6} Sh={cs[tag+'_sharpe']:+5.2f}  "
              f"Ret={cs[tag+'_ret']:+6.2%}  DD={cs[tag+'_dd']:+6.2%}")

    out_df = pd.DataFrame({
        "timestamp": combined_scaled.index,
        "ret": combined_scaled.values,
    })
    out_df.to_parquet(OUT / "vrp_returns.parquet", index=False)
    print(f"\nSaved combined returns: {OUT / 'vrp_returns.parquet'}")
    print(f"Saved breakdown:        {OUT / 'vrp_breakdown.csv'}")

    # ---- Family / asset-class summary ----
    print("\n=== Family summary (mean IS / OOS / 2022 Sharpe) ===")
    bk = pd.DataFrame(rows)
    fam = bk.groupby("family").agg(
        n=("name", "count"),
        IS_sharpe=("IS_sharpe", "mean"),
        OOS_sharpe=("OOS_sharpe", "mean"),
        s2022=("2022_sharpe", "mean"),
    )
    print(fam.to_string())
    print("\n=== Asset-class summary (mean IS / OOS / 2022 Sharpe) ===")
    ac = bk.groupby("asset_class").agg(
        n=("name", "count"),
        IS_sharpe=("IS_sharpe", "mean"),
        OOS_sharpe=("OOS_sharpe", "mean"),
        s2022=("2022_sharpe", "mean"),
    )
    print(ac.to_string())


if __name__ == "__main__":
    main()
