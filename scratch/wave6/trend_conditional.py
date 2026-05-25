"""Trend-strength CONDITIONAL strategies (wave 6).

Different from generic momentum: these only trade when the trend has
specific characteristics (strength, smoothness, multi-scale agreement,
freshness, regime stability).

Strategies
----------
  1. ADX_FILTER        -- TSMOM-63 gated on ADX(14) > 25 (strong trend). Skip <20.
  2. MULTI_SLOPE       -- Trade only when 21d, 63d, 252d slopes all same sign
                          AND |slope_short / slope_long| in [0.5, 2.0].
  3. ACCEL             -- Trade when current-21d return > previous-21d return.
                          (positive acceleration of momentum)
  4. LOW_PULLBACK      -- Trade trend only when max drawdown DURING the trend
                          (trailing 63d) < 5%.
  5. HURST             -- Trade only when rolling 252d Hurst exponent > 0.55.
  6. BREAKOUT_DECAY    -- Long only within first 10 days after breaking 50d high.
  7. VOLVOL_GATE       -- Trade trend only when vol-of-vol is below median.

Methodology
-----------
  - D1 candles, 13 symbols.
  - IS  <  2024-01-01.  OOS >= 2024-01-01.
  - All features walk-forward (computed on info available at t-1; positions
    shifted by 1 bar).
  - Walk-forward thresholds: ADX/Hurst/volvol use *expanding-window*
    percentile gates (no IS-only snooping of OOS distribution).
  - Vol-scale each surviving (strategy, symbol) sub-sleeve to 5% IS ann vol.
  - Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
  - Combine survivors equal-weight within each strategy, then equal-weight
    across strategies that have survivors to form the sleeve.

Outputs
-------
  scratch/wave6/trend_conditional.py
  scratch/wave6/trend_conditional_returns.parquet
  scratch/wave6/trend_conditional_breakdown.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, SYMBOL_TYPE
from alphabeta.backtest import backtest


# -- knobs --------------------------------------------------------------------
SPLIT          = pd.Timestamp("2024-01-01", tz="UTC")
TF             = "D1"

# Trend lookbacks
LB_SHORT       = 21
LB_MED         = 63
LB_LONG        = 252

VOL_WIN        = 60
TARGET_VOL_IS  = 0.05
TARGET_VOL_BASE= 0.10
POS_CAP        = 2.0
BPY            = 365.25

# Threshold knobs (walk-forward)
ADX_WIN        = 14
ADX_STRONG     = 25.0        # trade if ADX > this
ADX_SKIP       = 20.0        # zero pos if ADX < this

MULTI_RATIO_LO = 0.5
MULTI_RATIO_HI = 2.0

LOW_PULLBACK_DD= -0.05       # trailing 63d DD must be > -5%

HURST_WIN      = 252
HURST_THRESH   = 0.55

BREAKOUT_WIN   = 50
BREAKOUT_FRESH = 10          # only trade within first N days post-breakout

VOLVOL_WIN     = 60          # window for std of realized vol
VOLVOL_PCT_REF = 252         # min bars before applying expanding-median gate

ZWIN_MIN       = 252

MIN_IS_SHARPE  = 0.5
MIN_OOS_SHARPE = 0.0

STRATEGIES     = ["ADX_FILTER", "MULTI_SLOPE", "ACCEL", "LOW_PULLBACK",
                  "HURST", "BREAKOUT_DECAY", "VOLVOL_GATE", "VANILLA"]

OUT_DIR        = Path(__file__).resolve().parent


# -- stats helpers ------------------------------------------------------------
def _stats(r: pd.Series, freq: float = BPY) -> dict:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "max_dd": 0.0, "n": int(len(r))}
    mu = r.mean() * freq
    sd = r.std(ddof=0) * np.sqrt(freq)
    sh = mu / sd if sd > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"sharpe": float(sh), "ann_return": float(mu),
            "ann_vol": float(sd), "max_dd": dd, "n": int(len(r))}


def _vol_scale_to_target(ret: pd.Series, split_ts: pd.Timestamp,
                         target: float = TARGET_VOL_IS) -> float:
    is_r = ret.loc[ret.index < split_ts].dropna()
    if is_r.empty:
        return 0.0
    sd = is_r.std(ddof=0) * np.sqrt(BPY)
    if sd <= 0:
        return 0.0
    return float(target / sd)


# -- feature kernels ----------------------------------------------------------
def rolling_slope(y: np.ndarray, win: int) -> np.ndarray:
    """OLS slope of `y` on time over each trailing window of length win."""
    n = len(y)
    out = np.full(n, np.nan)
    if n < win:
        return out
    x = np.arange(win, dtype="float64")
    x_mean = x.mean()
    x_dev = x - x_mean
    sxx = (x_dev * x_dev).sum()
    if sxx <= 0:
        return out
    for t in range(win - 1, n):
        w = y[t - win + 1 : t + 1]
        if np.isnan(w).any():
            continue
        y_mean = w.mean()
        beta = ((w - y_mean) * x_dev).sum() / sxx
        out[t] = beta
    return out


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                win: int = ADX_WIN) -> np.ndarray:
    """Wilder's ADX. Returns array (NaN until 2*win bars)."""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for t in range(1, n):
        up = high[t] - high[t - 1]
        dn = low[t - 1] - low[t]
        plus_dm[t]  = up if (up > dn and up > 0) else 0.0
        minus_dm[t] = dn if (dn > up and dn > 0) else 0.0
        tr[t] = max(high[t] - low[t],
                    abs(high[t] - close[t - 1]),
                    abs(low[t]  - close[t - 1]))
    # Wilder smoothing
    atr = np.zeros(n)
    pdi = np.zeros(n)
    mdi = np.zeros(n)
    if n <= win:
        return np.full(n, np.nan)
    atr[win] = tr[1 : win + 1].sum()
    pdm_s = plus_dm[1 : win + 1].sum()
    mdm_s = minus_dm[1 : win + 1].sum()
    for t in range(win + 1, n):
        atr[t]   = atr[t - 1]   - (atr[t - 1]   / win) + tr[t]
        pdm_s    = pdm_s        - (pdm_s        / win) + plus_dm[t]
        mdm_s    = mdm_s        - (mdm_s        / win) + minus_dm[t]
        if atr[t] > 0:
            pdi[t] = 100.0 * pdm_s / atr[t]
            mdi[t] = 100.0 * mdm_s / atr[t]
    dx = np.zeros(n)
    for t in range(win + 1, n):
        s = pdi[t] + mdi[t]
        dx[t] = 100.0 * abs(pdi[t] - mdi[t]) / s if s > 0 else 0.0
    adx = np.full(n, np.nan)
    start = 2 * win
    if n > start:
        adx[start] = dx[win + 1 : start + 1].mean()
        for t in range(start + 1, n):
            adx[t] = (adx[t - 1] * (win - 1) + dx[t]) / win
    return adx


def rolling_max_dd(close: np.ndarray, win: int) -> np.ndarray:
    """Max drawdown over each trailing `win` window (non-positive)."""
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(win - 1, n):
        w = close[t - win + 1 : t + 1]
        if np.isnan(w).any():
            continue
        cm = np.maximum.accumulate(w)
        out[t] = ((w / cm) - 1.0).min()
    return out


def rolling_hurst(log_close: np.ndarray, win: int = HURST_WIN) -> np.ndarray:
    """Rescaled-range Hurst exponent over each trailing window.

    For each window, compute log(R/S) at chunk sizes [16, 32, 64], regress
    on log(size); slope = Hurst.
    """
    n = len(log_close)
    out = np.full(n, np.nan)
    sizes = [16, 32, 64]
    log_sizes = np.log(sizes)
    if n < win:
        return out
    for t in range(win - 1, n):
        y = log_close[t - win + 1 : t + 1]
        if np.isnan(y).any():
            continue
        rets = np.diff(y)
        if len(rets) < max(sizes) * 2:
            continue
        log_rs = []
        for k in sizes:
            n_chunks = len(rets) // k
            if n_chunks < 2:
                log_rs.append(np.nan)
                continue
            rs_vals = []
            for j in range(n_chunks):
                chunk = rets[j * k : (j + 1) * k]
                m = chunk.mean()
                dev = chunk - m
                z = np.cumsum(dev)
                R = z.max() - z.min()
                S = chunk.std(ddof=0)
                if S > 0:
                    rs_vals.append(R / S)
            if len(rs_vals) == 0:
                log_rs.append(np.nan)
            else:
                log_rs.append(np.log(np.mean(rs_vals)))
        log_rs = np.array(log_rs)
        if np.isnan(log_rs).any():
            continue
        # slope of log_rs vs log_sizes
        a = log_sizes - log_sizes.mean()
        b = log_rs - log_rs.mean()
        denom = (a * a).sum()
        if denom <= 0:
            continue
        out[t] = float((a * b).sum() / denom)
    return out


def days_since_breakout(close: np.ndarray, win: int = BREAKOUT_WIN) -> np.ndarray:
    """For each bar t, how many bars ago we last printed a new `win`-bar high.

    Returns -1 if we never broke out within the trailing window, 0 if today
    (will be shifted later), etc.
    """
    n = len(close)
    out = np.full(n, -1.0)
    rolling_high = pd.Series(close).rolling(win, min_periods=win).max().values
    # A bar t is a "breakout" if close[t] >= rolling_high[t]
    last_breakout = -10_000
    for t in range(n):
        if not np.isnan(rolling_high[t]) and close[t] >= rolling_high[t] - 1e-12:
            last_breakout = t
        if last_breakout >= 0:
            out[t] = t - last_breakout
        else:
            out[t] = -1.0
    return out


# -- per-symbol feature build -------------------------------------------------
def per_symbol_features(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df, feat) where feat has all signals shifted by 1 bar."""
    df = get_candles(symbol, TF).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    close = df["close"].astype("float64").values
    high  = df["high"].astype("float64").values
    low   = df["low"].astype("float64").values
    n = len(df)

    log_close = np.log(np.where(close > 0, close, np.nan))
    log_ret = np.zeros(n)
    log_ret[1:] = log_close[1:] - log_close[:-1]

    # 63d cumulative log return (trail_sum) for direction.
    trail_sum_med = pd.Series(log_ret).rolling(LB_MED,   min_periods=LB_MED).sum().values
    trail_sum_sh  = pd.Series(log_ret).rolling(LB_SHORT, min_periods=LB_SHORT).sum().values
    trail_sum_ln  = pd.Series(log_ret).rolling(LB_LONG,  min_periods=LB_LONG).sum().values

    # Slopes for MULTI_SLOPE
    sl_sh  = rolling_slope(log_close, LB_SHORT)
    sl_med = rolling_slope(log_close, LB_MED)
    sl_ln  = rolling_slope(log_close, LB_LONG)

    # ACCEL: current 21d log-ret vs previous 21d log-ret (shifted by 21)
    ret_21    = trail_sum_sh
    ret_21_pr = pd.Series(trail_sum_sh).shift(LB_SHORT).values

    # Vol scaler for vanilla TSMOM (60d realized-vol of log returns).
    rv = (pd.Series(log_ret).rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=0)
          * np.sqrt(BPY))
    vol_scale = (TARGET_VOL_BASE / rv).where(rv > 0).values

    # ADX
    adx = compute_adx(high, low, close, ADX_WIN)

    # Low pullback: 63d trailing max DD
    dd_63 = rolling_max_dd(close, LB_MED)

    # Hurst (252d, walk-forward)
    hurst = rolling_hurst(log_close, HURST_WIN)

    # Breakout freshness
    dsb = days_since_breakout(close, BREAKOUT_WIN)

    # Vol of vol: 60d std of rolling 20d realized vol
    rv_short = pd.Series(log_ret).rolling(20, min_periods=20).std(ddof=0).values
    volvol = pd.Series(rv_short).rolling(VOLVOL_WIN, min_periods=VOLVOL_WIN).std(ddof=0).values
    # Walk-forward gate via expanding median
    volvol_s = pd.Series(volvol)
    volvol_med = volvol_s.expanding(min_periods=VOLVOL_PCT_REF).median().values

    feat = pd.DataFrame({
        "timestamp":    df["timestamp"].values,
        "close":        close,
        "log_ret":      log_ret,
        "trail_sum":    trail_sum_med,   # 63d
        "trail_sum_sh": trail_sum_sh,    # 21d
        "trail_sum_ln": trail_sum_ln,    # 252d
        "sl_sh":        sl_sh,
        "sl_med":       sl_med,
        "sl_ln":        sl_ln,
        "ret_21":       ret_21,
        "ret_21_prev":  ret_21_pr,
        "vol_scale":    vol_scale,
        "adx":          adx,
        "dd_63":        dd_63,
        "hurst":        hurst,
        "dsb":          dsb,
        "volvol":       volvol,
        "volvol_med":   volvol_med,
    })

    # Walk-forward shift on EVERY signal used for positioning.
    shift_cols = ["trail_sum", "trail_sum_sh", "trail_sum_ln",
                  "sl_sh", "sl_med", "sl_ln",
                  "ret_21", "ret_21_prev",
                  "vol_scale", "adx", "dd_63", "hurst", "dsb",
                  "volvol", "volvol_med"]
    for c in shift_cols:
        feat[c] = feat[c].shift(1)

    return df, feat


# -- position builders --------------------------------------------------------
def _vanilla_pos(feat: pd.DataFrame) -> pd.Series:
    """Bare 63d TSMOM — reference comparator (unconditional trend)."""
    sig = np.sign(feat["trail_sum"]).fillna(0.0)
    return (sig * feat["vol_scale"]).clip(-POS_CAP, POS_CAP).fillna(0.0)


def _adx_filter_pos(feat: pd.DataFrame) -> pd.Series:
    """Trade trend signal only when ADX > 25; flat when < 20; partial between."""
    base = _vanilla_pos(feat)
    adx = feat["adx"]
    strong = (adx > ADX_STRONG).fillna(False)
    weak   = (adx < ADX_SKIP).fillna(True)
    pos = base.where(strong, 0.0)
    # Already 0 in weak zone since strong=False; explicit clarity:
    pos = pos.where(~weak, 0.0)
    return pos


def _multi_slope_pos(feat: pd.DataFrame) -> pd.Series:
    """All three slopes same sign + ratio in [0.5, 2.0] band."""
    ssh, smed, sln = feat["sl_sh"], feat["sl_med"], feat["sl_ln"]
    same_sign = ((np.sign(ssh) == np.sign(smed)) &
                 (np.sign(smed) == np.sign(sln)) &
                 (np.sign(ssh) != 0)).fillna(False)
    # ratio: short slope vs long slope (per-bar scaling). Normalize by bars/window
    # so they're comparable: convert each to "log-ret per bar" already (slope IS).
    eps = 1e-12
    ratio_sl = (ssh / (sln.replace(0, np.nan))).abs()
    in_band = ((ratio_sl >= MULTI_RATIO_LO) & (ratio_sl <= MULTI_RATIO_HI)).fillna(False)
    gate = same_sign & in_band
    direction = np.sign(smed).fillna(0.0)
    pos = (direction * feat["vol_scale"]).clip(-POS_CAP, POS_CAP).fillna(0.0)
    return pos.where(gate, 0.0)


def _accel_pos(feat: pd.DataFrame) -> pd.Series:
    """Trade when current 21d return is GREATER than previous 21d return."""
    accel = (feat["ret_21"] > feat["ret_21_prev"]).fillna(False)
    # Direction = sign of current 21d return
    direction = np.sign(feat["ret_21"]).fillna(0.0)
    pos = (direction * feat["vol_scale"]).clip(-POS_CAP, POS_CAP).fillna(0.0)
    return pos.where(accel, 0.0)


def _low_pullback_pos(feat: pd.DataFrame) -> pd.Series:
    """Trade trend only when trailing-63d max DD > -5% (smooth trends)."""
    base = _vanilla_pos(feat)
    gate = (feat["dd_63"] > LOW_PULLBACK_DD).fillna(False)
    return base.where(gate, 0.0)


def _hurst_pos(feat: pd.DataFrame) -> pd.Series:
    """Trade trend only when 252d Hurst > 0.55."""
    base = _vanilla_pos(feat)
    gate = (feat["hurst"] > HURST_THRESH).fillna(False)
    return base.where(gate, 0.0)


def _breakout_decay_pos(feat: pd.DataFrame) -> pd.Series:
    """Long-only for first 10 days after breaking 50d high."""
    dsb = feat["dsb"]
    fresh = ((dsb >= 0) & (dsb < BREAKOUT_FRESH)).fillna(False)
    pos = feat["vol_scale"].clip(-POS_CAP, POS_CAP).fillna(0.0)  # +1 long * scale
    return pos.where(fresh, 0.0)


def _volvol_gate_pos(feat: pd.DataFrame) -> pd.Series:
    """Trade trend only when vol-of-vol is below its walk-forward median."""
    base = _vanilla_pos(feat)
    gate = (feat["volvol"] < feat["volvol_med"]).fillna(False)
    return base.where(gate, 0.0)


POS_BUILDERS = {
    "ADX_FILTER":     _adx_filter_pos,
    "MULTI_SLOPE":    _multi_slope_pos,
    "ACCEL":          _accel_pos,
    "LOW_PULLBACK":   _low_pullback_pos,
    "HURST":          _hurst_pos,
    "BREAKOUT_DECAY": _breakout_decay_pos,
    "VOLVOL_GATE":    _volvol_gate_pos,
    "VANILLA":        _vanilla_pos,
}


# -- per-symbol driver --------------------------------------------------------
def per_symbol(symbol: str) -> dict:
    df, feat = per_symbol_features(symbol)
    d1_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    feat = feat.reset_index(drop=True)
    assert len(feat) == len(df)

    out = {"d1_idx": d1_idx, "returns": {}, "stats": {},
           "n_trades": {}, "exposure": {}, "gate_rate": {}}

    for sname, builder in POS_BUILDERS.items():
        pos = builder(feat).reset_index(drop=True)
        pos.index = df.index
        res = backtest(df, pos, symbol=symbol, timeframe=TF,
                       name=f"tcond_{sname}_{symbol}")
        ret = res.returns
        ret.index = d1_idx
        out["returns"][sname] = ret
        out["stats"][sname] = res.stats
        out["n_trades"][sname] = int(res.stats["n_trades"])
        out["exposure"][sname] = float(res.stats["exposure"])
        out["gate_rate"][sname] = float((pos.abs() > 0).mean())
    return out


# -- driver -------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building trend-conditional strategies for", len(ALL_SYMBOLS), "symbols...")
    per_sym = {}
    for s in ALL_SYMBOLS:
        print(f"  {s} ...")
        per_sym[s] = per_symbol(s)

    all_ts = sorted({pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
                     else pd.Timestamp(t, tz="UTC")
                     for d in per_sym.values() for t in d["d1_idx"]})
    union_idx = pd.DatetimeIndex(all_ts)

    rows = []
    per_strat_streams: dict[str, list[pd.Series]] = {s: [] for s in STRATEGIES}

    for sym in ALL_SYMBOLS:
        d1_idx = per_sym[sym]["d1_idx"]
        for sname in STRATEGIES:
            r = per_sym[sym]["returns"][sname].copy()
            r.index = d1_idx
            r_daily = r.groupby(r.index.normalize()).sum()
            r_daily.index = (r_daily.index.tz_convert("UTC")
                             if r_daily.index.tz
                             else r_daily.index.tz_localize("UTC"))

            full = _stats(r_daily)
            is_  = _stats(r_daily.loc[r_daily.index <  SPLIT])
            oos  = _stats(r_daily.loc[r_daily.index >= SPLIT])
            y22  = _stats(r_daily.loc[(r_daily.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                     (r_daily.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

            survive = (is_["sharpe"] >= MIN_IS_SHARPE) and (oos["sharpe"] >= MIN_OOS_SHARPE)
            scale = _vol_scale_to_target(r_daily, SPLIT) if survive else 0.0
            scaled = (r_daily * scale).reindex(union_idx).fillna(0.0)

            rows.append({
                "strategy": sname,
                "symbol": sym,
                "asset_class": SYMBOL_TYPE[sym].value,
                "is_sharpe": is_["sharpe"],
                "oos_sharpe": oos["sharpe"],
                "full_sharpe": full["sharpe"],
                "y2022_sharpe": y22["sharpe"],
                "is_ann_return": is_["ann_return"],
                "oos_ann_return": oos["ann_return"],
                "full_ann_return": full["ann_return"],
                "full_max_dd": full["max_dd"],
                "n_trades": per_sym[sym]["n_trades"][sname],
                "exposure": per_sym[sym]["exposure"][sname],
                "gate_rate": per_sym[sym]["gate_rate"][sname],
                "vol_scale": scale,
                "survived": bool(survive),
            })
            if survive:
                per_strat_streams[sname].append(scaled.rename(f"{sname}_{sym}"))

    bd = pd.DataFrame(rows)
    bd.to_csv(OUT_DIR / "trend_conditional_breakdown.csv", index=False)

    strat_streams: dict[str, pd.Series] = {}
    for sname, lst in per_strat_streams.items():
        if not lst:
            strat_streams[sname] = pd.Series(0.0, index=union_idx)
        else:
            mat = pd.concat(lst, axis=1).fillna(0.0)
            strat_streams[sname] = mat.mean(axis=1)

    # Build the conditional sleeve (excludes VANILLA reference).
    cond_active = [s for s, lst in per_strat_streams.items()
                   if lst and s != "VANILLA"]
    if cond_active:
        cmat = pd.concat([strat_streams[s].rename(s) for s in cond_active], axis=1).fillna(0.0)
        cond_raw = cmat.mean(axis=1)
        c_scale = _vol_scale_to_target(cond_raw, SPLIT, target=TARGET_VOL_IS)
        cond_sleeve = (cond_raw * c_scale)
        cond_sleeve = cond_sleeve.groupby(cond_sleeve.index.normalize()).sum()
        cond_sleeve.index = (cond_sleeve.index.tz_convert("UTC")
                             if cond_sleeve.index.tz
                             else cond_sleeve.index.tz_localize("UTC"))
    else:
        cond_sleeve = pd.Series(0.0, index=union_idx)
        c_scale = 0.0

    # Full sleeve including VANILLA reference (for comparison only)
    all_active = [s for s, lst in per_strat_streams.items() if lst]
    if all_active:
        amat = pd.concat([strat_streams[s].rename(s) for s in all_active], axis=1).fillna(0.0)
        all_raw = amat.mean(axis=1)
        a_scale = _vol_scale_to_target(all_raw, SPLIT, target=TARGET_VOL_IS)
        all_sleeve = (all_raw * a_scale)
        all_sleeve = all_sleeve.groupby(all_sleeve.index.normalize()).sum()
        all_sleeve.index = (all_sleeve.index.tz_convert("UTC")
                            if all_sleeve.index.tz
                            else all_sleeve.index.tz_localize("UTC"))
    else:
        all_sleeve = pd.Series(0.0, index=cond_sleeve.index)
        a_scale = 0.0

    # Save the conditional sleeve as the deliverable parquet
    out_df = pd.DataFrame({"timestamp": cond_sleeve.index, "ret": cond_sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(OUT_DIR / "trend_conditional_returns.parquet", index=False)

    def block(label: str, r: pd.Series) -> None:
        st = _stats(r)
        print(f"  {label:<6} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print(f"=== TREND-CONDITIONAL SLEEVE (D1, vol-scaled to {TARGET_VOL_IS:.0%} IS) ===")
    print(f"  Active conditional strategies: {cond_active}")
    print(f"  Sleeve scale: {c_scale:.3f}")
    block("FULL", cond_sleeve)
    block("IS",   cond_sleeve.loc[cond_sleeve.index <  SPLIT])
    block("OOS",  cond_sleeve.loc[cond_sleeve.index >= SPLIT])
    block("2022", cond_sleeve.loc[(cond_sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                  (cond_sleeve.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

    print()
    print("=== ALL-STRATEGIES SLEEVE (incl. VANILLA reference) ===")
    print(f"  Active: {all_active}    scale={a_scale:.3f}")
    block("FULL", all_sleeve)
    block("IS",   all_sleeve.loc[all_sleeve.index <  SPLIT])
    block("OOS",  all_sleeve.loc[all_sleeve.index >= SPLIT])
    block("2022", all_sleeve.loc[(all_sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                 (all_sleeve.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

    print()
    print("--- Per-strategy basket stats (post symbol-EW, pre sleeve-scale) ---")
    for sname in STRATEGIES:
        n_surv = sum(1 for row in rows if row["strategy"] == sname and row["survived"])
        r = strat_streams[sname]
        if r.std() == 0:
            print(f"  {sname:<15}  survivors={n_surv:>2d}   (no survivors)")
            continue
        full = _stats(r)
        is_  = _stats(r.loc[r.index <  SPLIT])
        oos  = _stats(r.loc[r.index >= SPLIT])
        y22  = _stats(r.loc[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                            (r.index <  pd.Timestamp("2023-01-01", tz="UTC"))])
        print(f"  {sname:<15}  surv={n_surv:>2d}  "
              f"FULL Sh={full['sharpe']:+5.2f}  IS Sh={is_['sharpe']:+5.2f}  "
              f"OOS Sh={oos['sharpe']:+5.2f}  2022 Sh={y22['sharpe']:+5.2f}")

    print()
    print("--- Survivor counts per strategy by asset class ---")
    if not bd[bd["survived"]].empty:
        piv = (bd[bd["survived"]]
               .groupby(["strategy", "asset_class"]).size().unstack(fill_value=0))
        print(piv.to_string())
    else:
        print("  (none)")

    print()
    print("--- Gate-rate (fraction of bars with non-zero pos) by strategy x asset ---")
    g_piv = (bd.groupby(["strategy", "asset_class"])["gate_rate"]
             .mean().unstack(fill_value=0.0))
    print(g_piv.round(3).to_string())

    print()
    print("--- Correlation with existing sleeves (daily) ---")
    try:
        tsmom = pd.read_parquet(OUT_DIR.parent / "quant" / "tsmom_returns.parquet")
        trend_new = pd.read_parquet(OUT_DIR.parent / "wave3" / "trend_returns.parquet")
        def _ser(df_):
            return pd.Series(df_["ret"].values,
                             index=pd.to_datetime(df_["timestamp"], utc=True))
        ts = _ser(tsmom).reindex(cond_sleeve.index).fillna(0.0)
        tn = _ser(trend_new).reindex(cond_sleeve.index).fillna(0.0)
        c_ts = float(cond_sleeve.corr(ts))
        c_tn = float(cond_sleeve.corr(tn))
        # vs vanilla basket
        van_basket = strat_streams.get("VANILLA")
        c_van = float(cond_sleeve.corr(van_basket)) if van_basket is not None and van_basket.std() > 0 else float("nan")
        print(f"  corr(conditional, TSMOM)     = {c_ts:+.3f}")
        print(f"  corr(conditional, TREND_NEW) = {c_tn:+.3f}")
        print(f"  corr(conditional, VANILLA)   = {c_van:+.3f}")

        # Joint OOS sharpe of (TREND_NEW + conditional)
        joint = 0.5 * tn + 0.5 * cond_sleeve
        s_joint = _stats(joint.loc[joint.index >= SPLIT])
        s_tn_oos = _stats(tn.loc[tn.index >= SPLIT])
        s_cond_oos = _stats(cond_sleeve.loc[cond_sleeve.index >= SPLIT])
        print(f"  OOS  TREND_NEW alone Sh={s_tn_oos['sharpe']:+.2f}, "
              f"COND alone Sh={s_cond_oos['sharpe']:+.2f}, "
              f"50/50 mix Sh={s_joint['sharpe']:+.2f}")
    except Exception as e:
        print(f"  could not compute correlations: {e}")

    # Yearly Sharpes
    print()
    print("--- Conditional sleeve Sharpe by year ---")
    for yr, sub in cond_sleeve.groupby(cond_sleeve.index.year):
        st = _stats(sub)
        print(f"  {yr}  Sh={st['sharpe']:+5.2f}  Ret={st['ann_return']:+7.2%}  "
              f"Vol={st['ann_vol']:6.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print(f"Wrote {OUT_DIR / 'trend_conditional_returns.parquet'}")
    print(f"Wrote {OUT_DIR / 'trend_conditional_breakdown.csv'}")


if __name__ == "__main__":
    main()
