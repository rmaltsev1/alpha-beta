"""Volatility-breakout strategies — wave 6.

Hypothesis: when vol *breaks out* of a compressed/quiet regime, momentum on
the underlying tends to work better than mean-reversion. We profit FROM vol
expansion (the opposite of VRP / vol mean-reversion).

Seven sub-strategy families (each ran across a relevant universe):

  1. Vol-regime-switched momentum — 30d RV > 60d MA of itself → switch to
     fast (21d) momentum; otherwise slow (126d) momentum.
  2. Compression-then-expansion — 30d RV < 60d avg for 10+ days, then
     >20% jump → enter in the direction of the breakout for N days.
  3. NR7 breakout — narrowest-range bar of last 7. Trade next break of
     prior high/low; hold to opposite break.
  4. BB squeeze + expansion — 20d Bollinger width at 60d 25th pctile
     (squeeze). Close > upper -> long; close < lower -> short. Hold 5d.
  5. Inside-day vol-breakout combo — inside day + low recent vol = setup;
     trade direction of next break for 3 days.
  6. Cross-asset vol leadership — BTC vol-breakout > 2σ → next-day long
     SPX-like equity index vol-proxy (a flat long-vol position via
     gamma-style long-then-flatten exposure on the index).
  7. Vol-of-vol momentum — rising vol-of-vol confirms TSMOM signals;
     filter classic 60d TSMOM by vov regime.

Methodology:
  IS  < 2024-01-01;  OOS >= 2024-01-01.
  All thresholds & vol estimates rolling/lagged (walk-forward).
  Vol-scale each sub-sleeve to 5% IS ann vol.
  Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
  Combine survivors equal-weight, rescale combined to 5% IS vol.

Outputs:
  scratch/wave6/vol_breakout_returns.parquet
  scratch/wave6/vol_breakout_breakdown.csv
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

EQUITY_BASKET = ["SPX500_USD", "NAS100_USD", "US30_USD",
                 "DE30_EUR", "UK100_GBP", "JP225_USD"]
CRYPTO_BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FX_BASKET = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]


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


# ---------------------------------------------------------------------------
# 1. Vol-regime-switched momentum
#   When 30d RV > 60d MA of itself → fast momentum (21d). Else slow (126d).
# ---------------------------------------------------------------------------

def strat1_regime_switched_mom(symbol: str,
                               rv_win: int = 30,
                               rv_ma_win: int = 60,
                               fast_win: int = 21,
                               slow_win: int = 126) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < slow_win + rv_ma_win + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)

    rv30 = ret.rolling(rv_win).std() * np.sqrt(252)
    rv30_ma = rv30.rolling(rv_ma_win).mean()
    # both lagged: regime is known *as of* yesterday's close
    high_vol = (rv30.shift(1) > rv30_ma.shift(1))

    # Momentum signals: sign of N-day return, lagged
    mom_fast = np.sign((close / close.shift(fast_win) - 1).shift(1))
    mom_slow = np.sign((close / close.shift(slow_win) - 1).shift(1))

    pos = np.where(high_vol, mom_fast, mom_slow)
    pos = pd.Series(pos, index=close.index).fillna(0).astype(float)

    cps = cost_for(symbol)
    sleeve = cost_apply(pos, ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 2. Compression-then-expansion
#   30d RV < 60d avg AND below for 10+ days; then next-day RV jumps > 20%
#   → enter in direction of the breakout move; hold N days.
# ---------------------------------------------------------------------------

def strat2_compression_expansion(symbol: str,
                                 rv_win: int = 30,
                                 rv_avg_win: int = 60,
                                 below_days: int = 10,
                                 jump_pct: float = 0.20,
                                 hold_days: int = 5) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < rv_avg_win + below_days + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)

    rv30 = ret.rolling(rv_win).std() * np.sqrt(252)
    rv_avg = rv30.rolling(rv_avg_win).mean()
    below = (rv30 < rv_avg).astype(int)
    # rolling count of consecutive 'below' days, but cheap proxy: sum over
    # last `below_days` is == below_days when ALL of them were below.
    compressed = below.rolling(below_days).sum() >= below_days

    # vol-jump: rv30[t-1] > 1.2 * rv30[t-2], with compressed state at t-2
    rv_jump = (rv30.shift(1) > (1 + jump_pct) * rv30.shift(2))
    compressed_yesterday = compressed.shift(2).fillna(False)
    trigger = rv_jump & compressed_yesterday

    # direction = sign of yesterday's return (the move that broke vol out)
    dir_sig = np.sign(ret.shift(1))

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    tvals = trigger.values
    dvals = dir_sig.fillna(0).values
    for t in range(n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
        elif tvals[t] and dvals[t] != 0:
            sign = float(dvals[t])
            pos[t] = sign
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 3. NR7 breakout
#   The bar with the smallest high-low range of the last 7 sets up a vol
#   compression. Trade the next break of that bar's high/low (continuation).
# ---------------------------------------------------------------------------

def strat3_nr7_breakout(symbol: str,
                        win: int = 7,
                        max_hold: int = 5) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < win + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    ret = close.pct_change().fillna(0)
    rng = (high - low)

    # NR7: today's range is the min of the last 7 ranges
    is_nr7 = (rng == rng.rolling(win).min())
    # We need to act NEXT day. So at bar t, setup = is_nr7[t-1].
    # The setup-bar's high/low are the levels to break.
    setup_high = high.where(is_nr7).ffill().shift(1)
    setup_low = low.where(is_nr7).ffill().shift(1)
    setup_age = (~is_nr7).astype(int).groupby((is_nr7.cumsum())).cumcount().shift(1)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    hi = high.values
    lo = low.values
    cl = close.values
    sh = setup_high.values
    sl = setup_low.values
    sa = setup_age.fillna(99).values
    for t in range(1, n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
            # exit early if reverse break of setup
            if sign > 0 and (not np.isnan(sl[t])) and cl[t] < sl[t]:
                hold = 0
            elif sign < 0 and (not np.isnan(sh[t])) and cl[t] > sh[t]:
                hold = 0
            continue
        # Only fresh setups: setup_age <= 3 days old
        if np.isnan(sh[t]) or np.isnan(sl[t]) or sa[t] > 3:
            continue
        if cl[t] > sh[t]:
            sign = 1.0
            pos[t] = sign
            hold = max_hold - 1
        elif cl[t] < sl[t]:
            sign = -1.0
            pos[t] = sign
            hold = max_hold - 1
        else:
            pos[t] = 0.0

    # Position derived from close[t] (today's close). To avoid look-ahead,
    # the position must be active *from t+1*. Shift by 1.
    pos = pd.Series(pos, index=close.index).shift(1).fillna(0)

    cps = cost_for(symbol)
    sleeve = cost_apply(pos, ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 4. Bollinger band squeeze + expansion
#   20d BB width at 60d 25th pctile (squeeze). Close > upper → long;
#   close < lower → short. Hold 5d. All signals computed on close[t-1].
# ---------------------------------------------------------------------------

def strat4_bb_squeeze(symbol: str,
                      bb_win: int = 20,
                      pct_win: int = 60,
                      pct: float = 0.25,
                      hold_days: int = 5,
                      k: float = 2.0) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < bb_win + pct_win + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)

    ma = close.rolling(bb_win).mean()
    sd = close.rolling(bb_win).std()
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma

    # 60d 25th pctile of width, lagged
    thresh = width.rolling(pct_win).quantile(pct).shift(1)
    width_lag = width.shift(1)
    squeezed = (width_lag <= thresh)

    # Breakout signals: use *yesterday's* close vs *yesterday's* upper/lower
    close_lag = close.shift(1)
    upper_lag = upper.shift(1)
    lower_lag = lower.shift(1)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    sq = squeezed.fillna(False).values
    cu = (close_lag > upper_lag).fillna(False).values
    cl_ = (close_lag < lower_lag).fillna(False).values
    for t in range(n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
            continue
        if sq[t] and cu[t]:
            sign = 1.0
            pos[t] = sign
            hold = hold_days - 1
        elif sq[t] and cl_[t]:
            sign = -1.0
            pos[t] = sign
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 5. Inside-day vol-breakout combo
#   Inside day (today's range strictly inside yesterday's range) + low
#   recent vol (30d RV < 60d median). Trade direction of next-day breakout
#   of yesterday's high/low; hold 3 days.
# ---------------------------------------------------------------------------

def strat5_inside_day(symbol: str,
                      rv_win: int = 30,
                      rv_med_win: int = 60,
                      hold_days: int = 3) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < rv_med_win + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    ret = close.pct_change().fillna(0)
    rv30 = ret.rolling(rv_win).std() * np.sqrt(252)
    rv_med = rv30.rolling(rv_med_win).median()

    # Inside-day at time t: high[t] < high[t-1] AND low[t] > low[t-1].
    inside = (high < high.shift(1)) & (low > low.shift(1))
    low_vol = (rv30 < rv_med)
    setup = inside & low_vol  # detected at end of bar t

    # Breakout direction at t+1: relative to bar-t's high/low.
    # We act starting bar t+1: if close[t+1] > high[t] → long, etc.
    # In array form: at bar tt (=t+1), trigger = setup[tt-1], and
    # direction = sign(close[tt] - high[tt-1]) / (close[tt] - low[tt-1]).
    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    setup_v = setup.fillna(False).values
    cl_ = close.values
    hi = high.values
    lo = low.values
    for t in range(1, n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
            continue
        if setup_v[t - 1]:
            if cl_[t] > hi[t - 1]:
                sign = 1.0
                pos[t] = sign
                hold = hold_days - 1
            elif cl_[t] < lo[t - 1]:
                sign = -1.0
                pos[t] = sign
                hold = hold_days - 1
            else:
                pos[t] = 0.0
        else:
            pos[t] = 0.0

    # The position determined at close[t] takes effect at t+1. Shift.
    pos = pd.Series(pos, index=close.index).shift(1).fillna(0)
    cps = cost_for(symbol)
    sleeve = cost_apply(pos, ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 6. Cross-asset vol leadership: BTC vol-breakout → next-day long equity-vol
#   We can't trade VIX. Proxy "long vol exposure" on the index by going
#   LONG-AT-BREAK and SHORT-AT-BREAK based on next-day return sign, which
#   is unknown — so we use a vol-mimicking long-gamma proxy: take the
#   ABSOLUTE return direction implied by the last day's BTC move. Concretely:
#   when BTC's 30d RV jumps > 2σ (rolling 252d), next day go LONG the index
#   in BTC's signed-move direction (a momentum spillover bet). Hold 1d.
# ---------------------------------------------------------------------------

def strat6_cross_asset(equity_symbol: str,
                       btc_symbol: str = "BTCUSDT",
                       rv_win: int = 30,
                       z_win: int = 252,
                       z_thresh: float = 2.0) -> pd.Series:
    df_eq = _load_d1(equity_symbol)
    df_bt = _load_d1(btc_symbol)
    if len(df_eq) < z_win + 30 or len(df_bt) < z_win + 30:
        return pd.Series(dtype=float)

    bt_close = pd.Series(
        df_bt["close"].astype(float).values,
        index=pd.DatetimeIndex(pd.to_datetime(df_bt["timestamp"], utc=True)),
    )
    bt_ret = bt_close.pct_change().fillna(0)
    rv_bt = bt_ret.rolling(rv_win).std() * np.sqrt(252)
    drv = rv_bt - rv_bt.shift(1)
    drv_z = (drv - drv.rolling(z_win).mean()) / drv.rolling(z_win).std()
    btc_break = (drv_z > z_thresh)
    # direction = sign of BTC's return on the breakout day
    btc_dir = np.sign(bt_ret)

    # build signal aligned to equity timestamps
    eq_idx = pd.DatetimeIndex(pd.to_datetime(df_eq["timestamp"], utc=True))
    eq_close = pd.Series(df_eq["close"].astype(float).values, index=eq_idx)
    eq_ret = eq_close.pct_change().fillna(0)

    # for each equity bar, find prior BTC bar (BTC has weekends, eq doesn't —
    # use as-of forward fill onto eq idx, then lag by 1 day).
    sig_break = btc_break.reindex(
        btc_break.index.union(eq_idx)
    ).ffill().reindex(eq_idx)
    sig_dir = btc_dir.reindex(
        btc_dir.index.union(eq_idx)
    ).ffill().reindex(eq_idx)

    trig = sig_break.shift(1).fillna(False).astype(bool)
    dirn = sig_dir.shift(1).fillna(0)
    pos = (trig.astype(float) * dirn).fillna(0)

    cps = cost_for(equity_symbol)
    sleeve = cost_apply(pos, eq_ret, cps)
    return to_ts_series(df_eq, sleeve.values)


# ---------------------------------------------------------------------------
# 7. Vol-of-vol momentum / TSMOM confirmation
#   vol-of-vol = std of 30d RV over last 20 days. When vov is RISING
#   (vov > vov.rolling(60).mean()), TSMOM (sign of 60d ret) should be more
#   reliable. Take TSMOM only when vov regime is rising.
# ---------------------------------------------------------------------------

def strat7_vov_tsmom(symbol: str,
                     rv_win: int = 30,
                     vov_win: int = 20,
                     vov_ma_win: int = 60,
                     mom_win: int = 60) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < mom_win + vov_ma_win + rv_win + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    rv30 = ret.rolling(rv_win).std() * np.sqrt(252)
    vov = rv30.rolling(vov_win).std()
    vov_ma = vov.rolling(vov_ma_win).mean()
    vov_rising = (vov.shift(1) > vov_ma.shift(1))

    mom = np.sign((close / close.shift(mom_win) - 1).shift(1))
    pos = np.where(vov_rising, mom, 0.0)
    pos = pd.Series(pos, index=close.index).fillna(0).astype(float)

    cps = cost_for(symbol)
    sleeve = cost_apply(pos, ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# Build, scale, filter, combine.
# ---------------------------------------------------------------------------

# Universes per family — vol-breakout is cleanest on equities & crypto.
# Family 2 explicitly per-spec runs on indices and crypto.
INDICES_AND_CRYPTO = EQUITY_BASKET + CRYPTO_BASKET
ALL_RELEVANT = EQUITY_BASKET + CRYPTO_BASKET + FX_BASKET


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


def build_candidates() -> dict[str, pd.Series]:
    cands: dict[str, pd.Series] = {}
    # 1, 7: TSMOM-flavoured — run across all relevant assets
    for sym in ALL_RELEVANT:
        try:
            cands[f"1_regime_mom__{sym}"] = strat1_regime_switched_mom(sym)
            cands[f"7_vov_tsmom__{sym}"]  = strat7_vov_tsmom(sym)
        except FileNotFoundError:
            continue

    # 2: compression-expansion — indices + crypto (per spec)
    for sym in INDICES_AND_CRYPTO:
        try:
            cands[f"2_compress_expand__{sym}"] = strat2_compression_expansion(sym)
        except FileNotFoundError:
            continue

    # 3: NR7 — all relevant
    for sym in ALL_RELEVANT:
        try:
            cands[f"3_nr7__{sym}"] = strat3_nr7_breakout(sym)
        except FileNotFoundError:
            continue

    # 4: BB squeeze — all relevant
    for sym in ALL_RELEVANT:
        try:
            cands[f"4_bb_squeeze__{sym}"] = strat4_bb_squeeze(sym)
        except FileNotFoundError:
            continue

    # 5: inside-day — all relevant
    for sym in ALL_RELEVANT:
        try:
            cands[f"5_inside_day__{sym}"] = strat5_inside_day(sym)
        except FileNotFoundError:
            continue

    # 6: cross-asset BTC→equity — equity basket only
    for sym in EQUITY_BASKET:
        try:
            cands[f"6_btc_to_eq__{sym}"] = strat6_cross_asset(sym, "BTCUSDT")
        except FileNotFoundError:
            continue

    return cands


def main():
    candidates = build_candidates()
    print(f"\nBuilt {len(candidates)} candidate sleeves.")

    rows = []
    scaled_streams = {}
    print(f"\n{'name':<35} {'IS_Sh':>6} {'OOS_Sh':>6} {'2022_Sh':>7} "
          f"{'2022_Ret':>8} {'OOS_Ret':>8} {'scale':>6}")
    print("-" * 95)
    for name, raw in candidates.items():
        raw = raw.dropna().sort_index()
        if len(raw) < 30:
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
        print(f"{name:<35} {s['IS_sharpe']:>+6.2f} {s['OOS_sharpe']:>+6.2f} "
              f"{s['2022_sharpe']:>+7.2f} {s['2022_ret']:>+8.2%} "
              f"{s['OOS_ret']:>+8.2%} {scale:>6.2f}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "vol_breakout_breakdown.csv", index=False)

    # ---- Survivor selection ----
    survivors = []
    for s in rows:
        if s["IS_sharpe"] >= 0.5 and s["OOS_sharpe"] >= 0.0:
            survivors.append(s["name"])
    print(f"\nSurvivors (IS >= 0.5 AND OOS >= 0):  ({len(survivors)})")
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
        ranked = sorted(rows, key=lambda r: r["IS_sharpe"], reverse=True)[:3]
        top = [r["name"] for r in ranked]
        print(f"\nNo strict survivors — falling back to top-3 IS: {top}")
        aligned = pd.concat(
            [scaled_streams[n].rename(n) for n in top], axis=1
        ).sort_index().fillna(0)
        combined = aligned.mean(axis=1)
        scl = vol_scale_is(combined, TARGET_VOL)
        combined_scaled = combined * scl
        survivors = top

    print("\nCombined sleeve (equal-weight survivors, rescaled to 5% IS vol):")
    cs = stats_block("combined", combined_scaled)
    for tag in ["IS","OOS","2020","2021","2022","2023","2024","2025"]:
        print(f"  {tag:<6} Sh={cs[tag+'_sharpe']:+5.2f}  "
              f"Ret={cs[tag+'_ret']:+6.2%}  DD={cs[tag+'_dd']:+6.2%}")

    out_df = pd.DataFrame({
        "timestamp": combined_scaled.index,
        "ret": combined_scaled.values,
    })
    out_df.to_parquet(OUT / "vol_breakout_returns.parquet", index=False)
    print(f"\nSaved combined returns: {OUT / 'vol_breakout_returns.parquet'}")
    print(f"Saved breakdown:        {OUT / 'vol_breakout_breakdown.csv'}")

    # ---- Family / asset-class summary ----
    print("\n=== Family summary (mean IS / OOS / 2022 Sharpe) ===")
    bk = pd.DataFrame(rows)
    fam = bk.groupby("family").agg(
        n=("name", "count"),
        IS_sharpe=("IS_sharpe", "mean"),
        OOS_sharpe=("OOS_sharpe", "mean"),
        s2022=("2022_sharpe", "mean"),
        OOS_ret=("OOS_ret", "mean"),
    )
    print(fam.to_string())
    print("\n=== Asset-class summary (mean IS / OOS / 2022 Sharpe) ===")
    ac = bk.groupby("asset_class").agg(
        n=("name", "count"),
        IS_sharpe=("IS_sharpe", "mean"),
        OOS_sharpe=("OOS_sharpe", "mean"),
        s2022=("2022_sharpe", "mean"),
        OOS_ret=("OOS_ret", "mean"),
    )
    print(ac.to_string())

    # ---- Per-family best individual sleeve ----
    print("\n=== Best sleeve per family (by IS Sharpe, OOS >= 0) ===")
    for fam_name, grp in bk[bk["OOS_sharpe"] >= 0].groupby("family"):
        best = grp.sort_values("IS_sharpe", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            print(f"  {fam_name:<22} -> {row['name']:<40} "
                  f"IS={row['IS_sharpe']:+.2f} OOS={row['OOS_sharpe']:+.2f} "
                  f"2022={row['2022_sharpe']:+.2f}")


if __name__ == "__main__":
    main()
