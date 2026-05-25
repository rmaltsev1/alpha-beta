"""Volume-based signal sleeves.

Five sub-sleeves, each evaluated with strict no-look-ahead (positions are
shifted so position[t] uses info up to t-1 only). Surviving sub-sleeves
(IS Sharpe >= 0.5 AND OOS mean > 0) get vol-scaled to 5% IS ann-vol and
averaged equal-weight into the combined volume sleeve.

Strategies:
  1. CRYPTO vol-spike + price-confirm long (D1)
  2. CRYPTO low-vol large-move fade (D1)
  3. INDEX bearish vol-divergence short (H1 -> D1)
  4. FOREX tick-activity follow (H1 -> D1)
  5. CRYPTO vol-confirmed D1 mean reversion (D1) -- compared to vanilla MR

Outputs:
  scratch/quant/volume_returns.parquet     (combined sleeve, D1, UTC ts)
  scratch/quant/volume_breakdown.csv       (per-sub-sleeve IS/OOS table)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphabeta import get_candles, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import backtest, cost_for

# ---------------- knobs --------------------------------------------------
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
BPY_D = 365.25                # crypto-style daily annualization
BPY_BUS = 252.0               # fx/index daily annualization
SURVIVE_IS_SHARPE = 0.5
OUT_DIR = Path(__file__).resolve().parent

# Forex tickers we use for tick-activity follow. Exclude USD_JPY because
# 23 UTC has a dedicated rule elsewhere; we still keep it here for diversity.
FX_LIST = FOREX
INDEX_LIST = ["SPX500_USD", "NAS100_USD", "US30_USD"]   # focus on US per spec


# ---------------- stat utils --------------------------------------------
def _bpy_for(returns: pd.Series, default: float) -> float:
    if len(returns) < 3:
        return default
    span = (returns.index[-1] - returns.index[0]).total_seconds() / 86400
    if span <= 0:
        return default
    return len(returns) / span * 365.25


def sleeve_stats(r: pd.Series, bpy: float | None = None) -> dict:
    r = r.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0,
                "n": int(len(r)), "exposure": 0.0}
    if bpy is None:
        bpy = _bpy_for(r, BPY_D)
    mu = float(r.mean()) * bpy
    sd = float(r.std(ddof=0)) * np.sqrt(bpy)
    sharpe = mu / sd if sd > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": float(sharpe), "ann_return": mu, "ann_vol": sd,
            "max_dd": dd, "n": int(len(r)), "exposure": float((r != 0).mean())}


def is_oos_split(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    return r[r.index < SPLIT], r[r.index >= SPLIT]


def normalize_to_daily(r: pd.Series) -> pd.Series:
    """Aggregate (sum) per UTC calendar day, tz-aware."""
    r = r.copy()
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    r.index = idx
    daily = r.groupby(r.index.floor("D")).sum()
    daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
    return daily


def vol_scale_5pct_is(r: pd.Series) -> pd.Series:
    """Scale series so IS portion has 5% annualized vol; oos uses same scalar."""
    is_, _ = is_oos_split(r)
    if len(is_) < 30:
        return r * 0.0
    bpy = _bpy_for(is_, BPY_D)
    sd = float(is_.std(ddof=0)) * np.sqrt(bpy)
    if sd <= 0:
        return r * 0.0
    scalar = TARGET_VOL / sd
    return r * scalar


# ---------------- strategy 1: crypto vol-spike + price confirm ----------
def strat1_position(df: pd.DataFrame, vol_mult: float = 1.5, ret_thr: float = 0.01,
                    hold: int = 3) -> pd.Series:
    """D1: today's vol > vol_mult * MA20(vol) AND today's ret > ret_thr -> long for `hold` bars.

    Strict shift: signal computed on bar t (close of day t) is exposed on t+1..t+hold.
    """
    close = df["close"].astype("float64")
    vol = df["volume"].astype("float64")
    ret_today = close.pct_change()
    vol_ma = vol.rolling(20, min_periods=20).mean().shift(1)
    # use trailing 20-day ma ending at t-1 to compare against today's vol
    trigger = (vol > vol_mult * vol_ma) & (ret_today > ret_thr)
    pos = pd.Series(0.0, index=df.index)
    # Hold from t+1 to t+hold inclusive
    # We'll set position by max over forward window
    fired = trigger.fillna(False).to_numpy()
    arr = np.zeros(len(df), dtype="float64")
    n = len(df)
    for i in range(n):
        if fired[i]:
            for k in range(1, hold + 1):
                if i + k < n:
                    arr[i + k] = 1.0
    pos = pd.Series(arr, index=df.index)
    return pos


# ---------------- strategy 2: crypto low-vol large-move fade ------------
def strat2_position(df: pd.DataFrame, q: float = 0.20, window: int = 60,
                    abs_thr: float = 0.02) -> pd.Series:
    """Fade yesterday's move when yesterday's volume was in the bottom quintile
    of trailing 60d AND |yesterday's ret| > 2%. Hold 1 day."""
    close = df["close"].astype("float64")
    vol = df["volume"].astype("float64")
    ret_y = close.pct_change()                                   # yesterday's ret at index t-1
    # volume quintile of trailing 60d ending at t-1
    quintile_lo = vol.rolling(window, min_periods=window).quantile(q)
    low_vol_y = (vol <= quintile_lo)
    big_move_y = ret_y.abs() > abs_thr
    fade_dir = -np.sign(ret_y)
    raw = pd.Series(0.0, index=df.index)
    mask = (low_vol_y & big_move_y).fillna(False)
    raw[mask] = fade_dir[mask]
    # shift by 1 so position[t] uses info up to t-1
    return raw.shift(1).fillna(0.0)


# ---------------- strategy 3: index bearish vol-divergence --------------
def strat3_position_h1(df_h1: pd.DataFrame, lookback: int = 24, hold_hours: int = 48,
                       hh_window: int = 100) -> pd.Series:
    """Short when price makes a new high (over hh_window bars) and rolling volume
    slope (over `lookback` bars) is negative. Hold short for `hold_hours` H1 bars.
    """
    close = df_h1["close"].astype("float64")
    vol = df_h1["volume"].astype("float64").replace(0, np.nan)
    # rolling vol-slope via least-squares on x=arange(lookback), y=log(vol)
    logv = np.log(vol)
    x = np.arange(lookback, dtype="float64")
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / x_var)

    slope = logv.rolling(lookback, min_periods=lookback).apply(_slope, raw=True)
    new_high = (close == close.rolling(hh_window, min_periods=hh_window).max())
    trigger = (new_high & (slope < 0)).fillna(False).to_numpy()

    arr = np.zeros(len(df_h1), dtype="float64")
    n = len(df_h1)
    for i in range(n):
        if trigger[i]:
            for k in range(1, hold_hours + 1):
                if i + k < n:
                    arr[i + k] = min(arr[i + k] - 1.0, -1.0)  # stay short / extend
    # cap at -1
    arr = np.clip(arr, -1.0, 0.0)
    return pd.Series(arr, index=df_h1.index)


# ---------------- strategy 4: FX tick-activity follow -------------------
def strat4_position_h1(df_h1: pd.DataFrame, q: float = 0.90, hold_hours: int = 5,
                       window: int = 24 * 20) -> pd.Series:
    """When H1 tick-count is in top decile of trailing `window` bars, follow
    the H1 bar direction for `hold_hours` bars."""
    close = df_h1["close"].astype("float64")
    vol = df_h1["volume"].astype("float64")
    bar_ret = np.log(close / close.shift(1))
    q_hi = vol.rolling(window, min_periods=window).quantile(q)
    hi_act = (vol >= q_hi).fillna(False)
    direction = np.sign(bar_ret).fillna(0.0)
    trigger = (hi_act & (direction != 0)).to_numpy()

    n = len(df_h1)
    arr = np.zeros(n, dtype="float64")
    dir_arr = direction.to_numpy()
    for i in range(n):
        if trigger[i]:
            d = dir_arr[i]
            for k in range(1, hold_hours + 1):
                if i + k < n:
                    # overwrite to current direction (most recent signal wins)
                    arr[i + k] = d
    return pd.Series(arr, index=df_h1.index)


# ---------------- strategy 5: vol-confirmed D1 mean reversion -----------
def strat5_position(df: pd.DataFrame, ret_thr: float = 0.02,
                    vol_quantile: float = 0.70, vol_win: int = 60) -> pd.Series:
    """Fade yesterday's move only if yesterday's volume was above 70th pctile
    of trailing `vol_win` days."""
    close = df["close"].astype("float64")
    vol = df["volume"].astype("float64")
    ret_y = close.pct_change()
    q_hi = vol.rolling(vol_win, min_periods=vol_win).quantile(vol_quantile)
    high_vol_y = vol >= q_hi
    big_move = ret_y.abs() > ret_thr
    raw = pd.Series(0.0, index=df.index)
    mask = (high_vol_y & big_move).fillna(False)
    raw[mask] = -np.sign(ret_y[mask])
    return raw.shift(1).fillna(0.0)


def strat5_vanilla(df: pd.DataFrame, ret_thr: float = 0.02) -> pd.Series:
    """Vanilla mean-reversion baseline (no volume filter), for comparison."""
    close = df["close"].astype("float64")
    ret_y = close.pct_change()
    big = ret_y.abs() > ret_thr
    raw = pd.Series(0.0, index=df.index)
    raw[big] = -np.sign(ret_y[big])
    return raw.shift(1).fillna(0.0)


# ---------------- runners -----------------------------------------------
def run_d1(symbol: str, pos_builder, name: str) -> pd.Series:
    df = get_candles(symbol, "D1")
    pos = pos_builder(df)
    res = backtest(df, pos, symbol=symbol, timeframe="D1", name=name)
    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    r = pd.Series(res.returns.values, index=ts, name=name)
    return r


def run_h1_to_daily(symbol: str, pos_builder, name: str) -> pd.Series:
    df = get_candles(symbol, "H1")
    pos = pos_builder(df)
    res = backtest(df, pos, symbol=symbol, timeframe="H1", name=name)
    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    r = pd.Series(res.returns.values, index=ts, name=name)
    # aggregate per UTC calendar day
    daily = r.groupby(r.index.floor("D")).sum()
    daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
    daily.name = name
    return daily


# ---------------- main --------------------------------------------------
def main() -> None:
    rows = []
    all_streams: dict[str, pd.Series] = {}

    # ---- Strategy 1: vol-spike + price confirm long (crypto) ------------
    print("=== Strategy 1: crypto vol-spike + price confirm (D1 long, 3d hold) ===")
    s1_streams = []
    for sym in CRYPTO:
        r = run_d1(sym, lambda df: strat1_position(df), f"S1_{sym}")
        s1_streams.append(r)
        is_, oos = is_oos_split(r)
        st_full = sleeve_stats(r, BPY_D); st_is = sleeve_stats(is_, BPY_D); st_oos = sleeve_stats(oos, BPY_D)
        rows.append({"sub_sleeve": f"S1_{sym}", "asset": "crypto", "symbol": sym, "tf": "D1",
                     "is_sharpe": st_is["sharpe"], "oos_sharpe": st_oos["sharpe"],
                     "full_sharpe": st_full["sharpe"], "is_ret": st_is["ann_return"],
                     "oos_ret": st_oos["ann_return"], "exposure": st_full["exposure"]})
        all_streams[f"S1_{sym}"] = r
        print(f"  {sym}: IS Sh={st_is['sharpe']:+.2f}  OOS Sh={st_oos['sharpe']:+.2f}  "
              f"Full Sh={st_full['sharpe']:+.2f}  expo={st_full['exposure']:.3f}")

    # ---- Strategy 2: low-vol large-move fade (crypto) -------------------
    print("\n=== Strategy 2: crypto low-vol large-move fade (D1) ===")
    for sym in CRYPTO:
        r = run_d1(sym, lambda df: strat2_position(df), f"S2_{sym}")
        is_, oos = is_oos_split(r)
        st_full = sleeve_stats(r, BPY_D); st_is = sleeve_stats(is_, BPY_D); st_oos = sleeve_stats(oos, BPY_D)
        rows.append({"sub_sleeve": f"S2_{sym}", "asset": "crypto", "symbol": sym, "tf": "D1",
                     "is_sharpe": st_is["sharpe"], "oos_sharpe": st_oos["sharpe"],
                     "full_sharpe": st_full["sharpe"], "is_ret": st_is["ann_return"],
                     "oos_ret": st_oos["ann_return"], "exposure": st_full["exposure"]})
        all_streams[f"S2_{sym}"] = r
        print(f"  {sym}: IS Sh={st_is['sharpe']:+.2f}  OOS Sh={st_oos['sharpe']:+.2f}  "
              f"Full Sh={st_full['sharpe']:+.2f}  expo={st_full['exposure']:.3f}")

    # ---- Strategy 3: index bearish vol-divergence (H1) -----------------
    print("\n=== Strategy 3: index bearish vol-divergence (H1 -> daily) ===")
    for sym in INDEX_LIST:
        r = run_h1_to_daily(sym, lambda df: strat3_position_h1(df), f"S3_{sym}")
        is_, oos = is_oos_split(r)
        st_full = sleeve_stats(r, BPY_BUS); st_is = sleeve_stats(is_, BPY_BUS); st_oos = sleeve_stats(oos, BPY_BUS)
        rows.append({"sub_sleeve": f"S3_{sym}", "asset": "index", "symbol": sym, "tf": "H1",
                     "is_sharpe": st_is["sharpe"], "oos_sharpe": st_oos["sharpe"],
                     "full_sharpe": st_full["sharpe"], "is_ret": st_is["ann_return"],
                     "oos_ret": st_oos["ann_return"], "exposure": st_full["exposure"]})
        all_streams[f"S3_{sym}"] = r
        print(f"  {sym}: IS Sh={st_is['sharpe']:+.2f}  OOS Sh={st_oos['sharpe']:+.2f}  "
              f"Full Sh={st_full['sharpe']:+.2f}  expo={st_full['exposure']:.3f}")

    # ---- Strategy 4: FX tick-activity follow (H1) -----------------------
    print("\n=== Strategy 4: FX tick-activity follow (H1 -> daily) ===")
    for sym in FX_LIST:
        r = run_h1_to_daily(sym, lambda df: strat4_position_h1(df), f"S4_{sym}")
        is_, oos = is_oos_split(r)
        st_full = sleeve_stats(r, BPY_BUS); st_is = sleeve_stats(is_, BPY_BUS); st_oos = sleeve_stats(oos, BPY_BUS)
        rows.append({"sub_sleeve": f"S4_{sym}", "asset": "fx", "symbol": sym, "tf": "H1",
                     "is_sharpe": st_is["sharpe"], "oos_sharpe": st_oos["sharpe"],
                     "full_sharpe": st_full["sharpe"], "is_ret": st_is["ann_return"],
                     "oos_ret": st_oos["ann_return"], "exposure": st_full["exposure"]})
        all_streams[f"S4_{sym}"] = r
        print(f"  {sym}: IS Sh={st_is['sharpe']:+.2f}  OOS Sh={st_oos['sharpe']:+.2f}  "
              f"Full Sh={st_full['sharpe']:+.2f}  expo={st_full['exposure']:.3f}")

    # ---- Strategy 5: vol-confirmed D1 reversion (crypto) ---------------
    print("\n=== Strategy 5: crypto vol-confirmed D1 mean reversion (D1) ===")
    for sym in ["BTCUSDT", "ETHUSDT"]:
        r = run_d1(sym, lambda df: strat5_position(df), f"S5_{sym}")
        is_, oos = is_oos_split(r)
        st_full = sleeve_stats(r, BPY_D); st_is = sleeve_stats(is_, BPY_D); st_oos = sleeve_stats(oos, BPY_D)
        rows.append({"sub_sleeve": f"S5_{sym}", "asset": "crypto", "symbol": sym, "tf": "D1",
                     "is_sharpe": st_is["sharpe"], "oos_sharpe": st_oos["sharpe"],
                     "full_sharpe": st_full["sharpe"], "is_ret": st_is["ann_return"],
                     "oos_ret": st_oos["ann_return"], "exposure": st_full["exposure"]})
        all_streams[f"S5_{sym}"] = r
        # vanilla baseline (informational only, not added to combined sleeve)
        r_v = run_d1(sym, lambda df: strat5_vanilla(df), f"S5v_{sym}")
        is_v, oos_v = is_oos_split(r_v)
        st_v_full = sleeve_stats(r_v, BPY_D); st_v_is = sleeve_stats(is_v, BPY_D); st_v_oos = sleeve_stats(oos_v, BPY_D)
        rows.append({"sub_sleeve": f"S5v_{sym}_vanilla", "asset": "crypto", "symbol": sym, "tf": "D1",
                     "is_sharpe": st_v_is["sharpe"], "oos_sharpe": st_v_oos["sharpe"],
                     "full_sharpe": st_v_full["sharpe"], "is_ret": st_v_is["ann_return"],
                     "oos_ret": st_v_oos["ann_return"], "exposure": st_v_full["exposure"]})
        print(f"  {sym} vol-conf: IS Sh={st_is['sharpe']:+.2f}  OOS Sh={st_oos['sharpe']:+.2f}  Full Sh={st_full['sharpe']:+.2f}")
        print(f"  {sym} vanilla : IS Sh={st_v_is['sharpe']:+.2f}  OOS Sh={st_v_oos['sharpe']:+.2f}  Full Sh={st_v_full['sharpe']:+.2f}")

    # ---- Filter survivors and combine ---------------------------------
    bd = pd.DataFrame(rows)
    bd["survives"] = (bd["is_sharpe"] >= SURVIVE_IS_SHARPE) & (bd["oos_ret"] > 0) & (~bd["sub_sleeve"].str.endswith("_vanilla"))
    bd.to_csv(OUT_DIR / "volume_breakdown.csv", index=False)

    survivors = bd[bd["survives"]]["sub_sleeve"].tolist()
    print("\n=== SURVIVORS (IS Sharpe >= 0.5 AND OOS ret > 0) ===")
    for n in survivors:
        print("  ", n)
    if not survivors:
        print("  (none)")

    # Build combined sleeve: vol-scale each survivor to 5% IS vol then equal-weight average
    if survivors:
        scaled = []
        for n in survivors:
            s = all_streams[n].copy()
            s = vol_scale_5pct_is(s)
            scaled.append(s)
        all_idx = sorted(set().union(*[s.index for s in scaled]))
        all_idx = pd.DatetimeIndex(all_idx)
        mat = pd.concat([s.reindex(all_idx).fillna(0.0) for s in scaled], axis=1)
        combined_bar = mat.mean(axis=1)
        combined = combined_bar.groupby(combined_bar.index.floor("D")).sum()
        combined.index = pd.DatetimeIndex(combined.index, tz="UTC")
    else:
        combined = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))

    # Headline stats
    print("\n=== COMBINED VOLUME SLEEVE ===")
    if len(combined) > 5:
        for tag, mask in [("FULL", pd.Series(True, index=combined.index)),
                          ("IS",   combined.index < SPLIT),
                          ("OOS",  combined.index >= SPLIT)]:
            sub = combined[mask]
            st = sleeve_stats(sub, BPY_D)
            print(f"  {tag:<4} Sharpe={st['sharpe']:+5.2f}  Ret={st['ann_return']:+7.2%}  "
                  f"Vol={st['ann_vol']:6.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")
    else:
        print("  (no survivors)")

    # Save parquet
    out_parquet = OUT_DIR / "volume_returns.parquet"
    out_df = pd.DataFrame({"timestamp": combined.index, "ret": combined.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"\nWrote {out_parquet}")
    print(f"Wrote {OUT_DIR / 'volume_breakdown.csv'}")


if __name__ == "__main__":
    main()
