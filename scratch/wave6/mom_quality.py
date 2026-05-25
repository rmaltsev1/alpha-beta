"""Momentum-quality scoring system (wave 6).

A *quality filter* on top of vanilla 63d TSMOM. Premise: not all trends are
equal. A steady, low-noise, shallow-drawdown trend has higher expected
continuation than a jerky one. We measure trend quality with four metrics
(R^2 of linear log-price fit, daily-direction hit-rate, in-trend max
drawdown, and crypto-only rising-volume confirmation), then build six
strategies that USE those quality scores in different ways.

Strategies
----------
  1. R2_FILTER       -- vanilla 63d TSMOM, gated on R^2 > 0.7.
  2. HIT_FILTER      -- vanilla 63d TSMOM, gated on direction hit-rate > 0.60.
  3. DD_FILTER       -- vanilla 63d TSMOM, gated on trailing-63d max DD > -0.10.
  4. COMBO_GATE      -- z-score each metric (z_r2 + z_hit - z_dd + z_vol),
                        binary gate when composite > +1.
  5. Q_WEIGHTED      -- vanilla 63d TSMOM scaled BY the composite quality
                        (linearly mapped to [0, 1]) — bigger positions when
                        trend is cleaner, no hard on/off.
  6. ANTI_QUALITY    -- fade direction of the LOW-quality trends (composite <
                        -1). Choppy trends mean revert.
  + VANILLA          -- reference: bare 63d TSMOM with no quality filter
                        (for like-for-like quality-helps-or-not comparison).

Methodology
-----------
  - D1 candles, 13 symbols.
  - IS  <  2024-01-01.  OOS >= 2024-01-01.
  - All quality metrics computed walk-forward (rolling 63d on info
    available at t-1; positions shifted by 1 bar).
  - Z-scores for the composite use *expanding-window mean/std* per symbol
    (also walk-forward — no IS-only "snooping" of the OOS distribution).
  - Vol-scale each surviving (strategy, symbol) sub-sleeve to 5% IS ann vol.
  - Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
  - Combine survivors equal-weight inside each strategy; then equal-weight
    across strategies that have any survivors to form the sleeve.

Outputs
-------
  scratch/wave6/mom_quality.py
  scratch/wave6/mom_quality_returns.parquet
  scratch/wave6/mom_quality_breakdown.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO, SYMBOL_TYPE
from alphabeta.backtest import backtest


# -- knobs --------------------------------------------------------------------
SPLIT          = pd.Timestamp("2024-01-01", tz="UTC")
TF             = "D1"
LOOKBACK       = 63                # 63d momentum
VOL_WIN        = 60                # realized-vol estimator
TARGET_VOL_IS  = 0.05              # 5% IS ann vol target per sub-sleeve
TARGET_VOL_BASE= 0.10              # 10% pre-scale on each TSMOM stream
POS_CAP        = 2.0
BPY            = 365.25

R2_THRESHOLD   = 0.70
HIT_THRESHOLD  = 0.60
DD_THRESHOLD   = -0.10             # trailing-63d max DD must be > -10%
COMBO_HIGH     = 1.0
COMBO_LOW      = -1.0
ZWIN_MIN       = 126               # need this many bars before first z-score

MIN_IS_SHARPE  = 0.5
MIN_OOS_SHARPE = 0.0

STRATEGIES     = ["VANILLA", "R2_FILTER", "HIT_FILTER", "DD_FILTER",
                  "COMBO_GATE", "Q_WEIGHTED", "ANTI_QUALITY"]

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


# -- quality metric kernels ---------------------------------------------------
def rolling_r2(log_close: np.ndarray, win: int) -> np.ndarray:
    """R^2 of a linear regression of log price on time within each `win` window.

    Returns array of length len(log_close), NaN for the first `win-1` bars.
    """
    n = len(log_close)
    out = np.full(n, np.nan)
    if n < win:
        return out
    x = np.arange(win, dtype="float64")
    x_mean = x.mean()
    x_dev = x - x_mean
    sxx = (x_dev * x_dev).sum()
    for t in range(win - 1, n):
        y = log_close[t - win + 1 : t + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        y_dev = y - y_mean
        sxy = (x_dev * y_dev).sum()
        syy = (y_dev * y_dev).sum()
        if syy <= 0 or sxx <= 0:
            out[t] = 0.0
            continue
        # R^2 of OLS y ~ a + b x  =  sxy^2 / (sxx * syy)
        out[t] = (sxy * sxy) / (sxx * syy)
    return out


def rolling_hit_rate(log_ret: np.ndarray, trail_sum: np.ndarray,
                     win: int) -> np.ndarray:
    """Fraction of the last `win` log-returns whose sign matches the sign of
    the trailing sum of log-returns over the same window. Walk-forward at t."""
    n = len(log_ret)
    out = np.full(n, np.nan)
    if n < win:
        return out
    sign_lr = np.sign(log_ret)
    # rolling count of positive bars and negative bars
    s = pd.Series(sign_lr)
    n_up = (s > 0).astype("float64").rolling(win, min_periods=win).sum().values
    n_dn = (s < 0).astype("float64").rolling(win, min_periods=win).sum().values
    sign_trend = np.sign(trail_sum)
    for t in range(win - 1, n):
        if np.isnan(sign_trend[t]):
            continue
        if sign_trend[t] > 0:
            out[t] = n_up[t] / win
        elif sign_trend[t] < 0:
            out[t] = n_dn[t] / win
        else:
            out[t] = 0.0
    return out


def rolling_max_dd(close: np.ndarray, win: int) -> np.ndarray:
    """Worst peak-to-trough drawdown of `close` over each trailing window.

    Returned as a non-positive number (-0.10 == 10% drawdown).
    Implementation: for each window, dd_t = c_t / cummax(c_in_window) - 1,
    then take min within the window.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < win:
        return out
    for t in range(win - 1, n):
        w = close[t - win + 1 : t + 1]
        if np.isnan(w).any():
            continue
        cummax = np.maximum.accumulate(w)
        dd = (w / cummax) - 1.0
        out[t] = dd.min()
    return out


def rolling_volume_slope(volume: np.ndarray, win: int) -> np.ndarray:
    """OLS slope of log(volume+1) vs time within each trailing window.

    Positive = volume trending up (confirmation). NaN for non-crypto handled
    by caller (passing zeros for non-crypto).
    """
    n = len(volume)
    out = np.full(n, np.nan)
    if n < win:
        return out
    lv = np.log(np.maximum(volume, 1.0))
    x = np.arange(win, dtype="float64")
    x_mean = x.mean()
    x_dev = x - x_mean
    sxx = (x_dev * x_dev).sum()
    for t in range(win - 1, n):
        y = lv[t - win + 1 : t + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        y_dev = y - y_mean
        if sxx <= 0:
            out[t] = 0.0
            continue
        beta = (x_dev * y_dev).sum() / sxx
        out[t] = beta
    return out


def expanding_zscore(s: pd.Series, min_periods: int = ZWIN_MIN) -> pd.Series:
    """Walk-forward z-score using *expanding* mean/std (no leakage)."""
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


# -- per-symbol position builders ---------------------------------------------
def per_symbol_metrics(symbol: str) -> pd.DataFrame:
    """Build a DataFrame of per-bar features needed for all strategies.

    All metric columns are *shifted by 1 bar* to enforce walk-forward.
    """
    df = get_candles(symbol, TF).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    close = df["close"].astype("float64").values
    volume = df["volume"].astype("float64").values
    n = len(df)

    log_close = np.log(np.where(close > 0, close, np.nan))
    log_ret = np.zeros(n)
    log_ret[1:] = log_close[1:] - log_close[:-1]

    # 63d trailing log-return sum (sign = trend direction).
    trail_sum = pd.Series(log_ret).rolling(LOOKBACK, min_periods=LOOKBACK).sum().values

    # Vol scaler for vanilla TSMOM (60d realized-vol of log returns).
    rv = (pd.Series(log_ret).rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=0)
          * np.sqrt(BPY))
    vol_scale = (TARGET_VOL_BASE / rv).where(rv > 0).values

    # Quality metrics
    r2     = rolling_r2(log_close, LOOKBACK)
    hit    = rolling_hit_rate(log_ret, trail_sum, LOOKBACK)
    mxdd   = rolling_max_dd(close, LOOKBACK)
    if SYMBOL_TYPE[symbol].value == "crypto":
        vslope = rolling_volume_slope(volume, LOOKBACK)
    else:
        # neutral: zeros instead of NaN so they don't blow up composite for FX/Index
        vslope = np.zeros(n)
        vslope[:LOOKBACK - 1] = np.nan

    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "close":     close,
        "log_ret":   log_ret,
        "trail_sum": trail_sum,
        "vol_scale": vol_scale,
        "r2":        r2,
        "hit":       hit,
        "mxdd":      mxdd,
        "vslope":    vslope,
    })

    # Walk-forward shift on EVERY signal/feature used by positions.
    for col in ["trail_sum", "vol_scale", "r2", "hit", "mxdd", "vslope"]:
        out[col] = out[col].shift(1)

    # Expanding z-scores for the composite (using already-shifted features).
    out["z_r2"]   = expanding_zscore(out["r2"])
    out["z_hit"]  = expanding_zscore(out["hit"])
    out["z_dd"]   = expanding_zscore(out["mxdd"])  # higher = less negative DD
    if SYMBOL_TYPE[symbol].value == "crypto":
        out["z_vol"] = expanding_zscore(out["vslope"])
    else:
        out["z_vol"] = 0.0

    # Composite (higher = cleaner trend). z_dd is already aligned (less DD =
    # bigger value = higher z), so we ADD it.
    out["composite"] = (out["z_r2"].fillna(0.0)
                        + out["z_hit"].fillna(0.0)
                        + out["z_dd"].fillna(0.0)
                        + out["z_vol"].fillna(0.0))
    return out


def _vanilla_pos(feat: pd.DataFrame) -> pd.Series:
    """Bare 63d TSMOM: sign(trail_sum) * vol_scale, capped."""
    sig = np.sign(feat["trail_sum"]).fillna(0.0)
    pos = (sig * feat["vol_scale"]).clip(-POS_CAP, POS_CAP).fillna(0.0)
    return pos


def _r2_filter_pos(feat: pd.DataFrame) -> pd.Series:
    base = _vanilla_pos(feat)
    gate = (feat["r2"] > R2_THRESHOLD).fillna(False)
    return base.where(gate, 0.0)


def _hit_filter_pos(feat: pd.DataFrame) -> pd.Series:
    base = _vanilla_pos(feat)
    gate = (feat["hit"] > HIT_THRESHOLD).fillna(False)
    return base.where(gate, 0.0)


def _dd_filter_pos(feat: pd.DataFrame) -> pd.Series:
    base = _vanilla_pos(feat)
    gate = (feat["mxdd"] > DD_THRESHOLD).fillna(False)
    return base.where(gate, 0.0)


def _combo_gate_pos(feat: pd.DataFrame) -> pd.Series:
    base = _vanilla_pos(feat)
    gate = (feat["composite"] > COMBO_HIGH).fillna(False)
    return base.where(gate, 0.0)


def _q_weighted_pos(feat: pd.DataFrame) -> pd.Series:
    """Scale vanilla TSMOM by the composite mapped to [0, 1]: anything below
    -1 sigma gets size 0, anything above +1 sigma gets size 1, linear between."""
    base = _vanilla_pos(feat)
    w = ((feat["composite"] - COMBO_LOW) / (COMBO_HIGH - COMBO_LOW)).clip(0.0, 1.0)
    return (base * w.fillna(0.0)).clip(-POS_CAP, POS_CAP)


def _anti_quality_pos(feat: pd.DataFrame) -> pd.Series:
    """Fade direction of low-quality trends: when composite < -1, short the
    trailing-trend direction. Position size keeps the same vol scaler."""
    sig = np.sign(feat["trail_sum"]).fillna(0.0)
    gate = (feat["composite"] < COMBO_LOW).fillna(False)
    pos = (-sig * feat["vol_scale"]).clip(-POS_CAP, POS_CAP).fillna(0.0)
    return pos.where(gate, 0.0)


POS_BUILDERS = {
    "VANILLA":      _vanilla_pos,
    "R2_FILTER":    _r2_filter_pos,
    "HIT_FILTER":   _hit_filter_pos,
    "DD_FILTER":    _dd_filter_pos,
    "COMBO_GATE":   _combo_gate_pos,
    "Q_WEIGHTED":   _q_weighted_pos,
    "ANTI_QUALITY": _anti_quality_pos,
}


# -- per-symbol driver --------------------------------------------------------
def per_symbol(symbol: str) -> dict:
    df = get_candles(symbol, TF).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    d1_idx = pd.DatetimeIndex(df["timestamp"])

    feat = per_symbol_metrics(symbol)
    # Align feat index to df.index (positional 0..N-1) so backtest matches.
    feat = feat.reset_index(drop=True)
    assert len(feat) == len(df)

    out = {"d1_idx": d1_idx, "returns": {}, "stats": {},
           "n_trades": {}, "exposure": {}, "gate_rate": {}}

    for sname, builder in POS_BUILDERS.items():
        pos = builder(feat).reset_index(drop=True)
        pos.index = df.index
        res = backtest(df, pos, symbol=symbol, timeframe=TF,
                       name=f"momq_{sname}_{symbol}")
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

    print("Building momentum-quality strategies for", len(ALL_SYMBOLS), "symbols...")
    per_sym = {}
    for s in ALL_SYMBOLS:
        print(f"  {s} ...")
        per_sym[s] = per_symbol(s)

    # union UTC index across all symbols
    all_ts = sorted({pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
                     else pd.Timestamp(t, tz="UTC")
                     for d in per_sym.values() for t in d["d1_idx"]})
    union_idx = pd.DatetimeIndex(all_ts)

    # ---- breakdown rows + per-strategy aggregation -------------------------
    rows = []
    per_strat_streams: dict[str, list[pd.Series]] = {s: [] for s in STRATEGIES}

    for sym in ALL_SYMBOLS:
        d1_idx = per_sym[sym]["d1_idx"]
        for sname in STRATEGIES:
            r = per_sym[sym]["returns"][sname].copy()
            r.index = d1_idx
            # collapse to one row per UTC day
            r_daily = r.groupby(r.index.normalize()).sum()
            r_daily.index = (r_daily.index.tz_convert("UTC")
                             if r_daily.index.tz else r_daily.index.tz_localize("UTC"))

            full = _stats(r_daily)
            is_  = _stats(r_daily.loc[r_daily.index < SPLIT])
            oos  = _stats(r_daily.loc[r_daily.index >= SPLIT])
            y22  = _stats(r_daily.loc[(r_daily.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                     (r_daily.index < pd.Timestamp("2023-01-01", tz="UTC"))])

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
    bd.to_csv(OUT_DIR / "mom_quality_breakdown.csv", index=False)

    # ---- per-strategy basket (EW across surviving symbols) -----------------
    strat_streams: dict[str, pd.Series] = {}
    for sname, lst in per_strat_streams.items():
        if not lst:
            strat_streams[sname] = pd.Series(0.0, index=union_idx)
        else:
            mat = pd.concat(lst, axis=1).fillna(0.0)
            strat_streams[sname] = mat.mean(axis=1)

    # ---- sleeve: equal-weight across strategies that have any survivor -----
    # NOTE: VANILLA is a reference comparator; we still include it if it survives,
    # but the headline "quality sleeve" reports both with and without it.
    active = [s for s, lst in per_strat_streams.items() if lst]
    if not active:
        sleeve_raw = pd.Series(0.0, index=union_idx)
    else:
        mat = pd.concat([strat_streams[s].rename(s) for s in active], axis=1).fillna(0.0)
        sleeve_raw = mat.mean(axis=1)

    sleeve_scale = _vol_scale_to_target(sleeve_raw, SPLIT, target=TARGET_VOL_IS)
    sleeve_bar = (sleeve_raw * sleeve_scale).rename("ret")
    sleeve = sleeve_bar.groupby(sleeve_bar.index.normalize()).sum()
    sleeve.index = (sleeve.index.tz_convert("UTC")
                    if sleeve.index.tz else sleeve.index.tz_localize("UTC"))
    sleeve.name = "ret"

    # Quality-only sleeve excludes VANILLA (for the "does the filter help?" check)
    quality_active = [s for s in active if s != "VANILLA"]
    if quality_active:
        qmat = pd.concat([strat_streams[s].rename(s) for s in quality_active], axis=1).fillna(0.0)
        quality_raw = qmat.mean(axis=1)
        q_scale = _vol_scale_to_target(quality_raw, SPLIT, target=TARGET_VOL_IS)
        quality_sleeve = (quality_raw * q_scale).groupby(quality_raw.index.normalize()).sum()
        quality_sleeve.index = (quality_sleeve.index.tz_convert("UTC")
                                if quality_sleeve.index.tz
                                else quality_sleeve.index.tz_localize("UTC"))
    else:
        quality_sleeve = pd.Series(0.0, index=sleeve.index)
        q_scale = 0.0

    # Persist the quality-only sleeve as the deliverable parquet (it's the
    # value-add over vanilla, which already exists as scratch/quant/tsmom).
    out_df = pd.DataFrame({"timestamp": quality_sleeve.index, "ret": quality_sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(OUT_DIR / "mom_quality_returns.parquet", index=False)

    # ---- report ------------------------------------------------------------
    def block(label: str, r: pd.Series) -> None:
        st = _stats(r)
        print(f"  {label:<6} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print(f"=== MOM-QUALITY QUALITY-ONLY SLEEVE (D1, vol-scaled to {TARGET_VOL_IS:.0%} IS) ===")
    print(f"  Active strategies in quality sleeve: {quality_active}")
    print(f"  Quality sleeve scale: {q_scale:.3f}")
    block("FULL", quality_sleeve)
    block("IS",   quality_sleeve.loc[quality_sleeve.index <  SPLIT])
    block("OOS",  quality_sleeve.loc[quality_sleeve.index >= SPLIT])
    block("2022", quality_sleeve.loc[(quality_sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                     (quality_sleeve.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

    print()
    print("=== ALL-STRATEGIES SLEEVE (incl. VANILLA reference) ===")
    print(f"  Active strategies: {active}    scale={sleeve_scale:.3f}")
    block("FULL", sleeve)
    block("IS",   sleeve.loc[sleeve.index <  SPLIT])
    block("OOS",  sleeve.loc[sleeve.index >= SPLIT])
    block("2022", sleeve.loc[(sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                             (sleeve.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

    print()
    print("--- Per-strategy sleeve stats (post symbol-EW, pre sleeve-scale) ---")
    for sname in STRATEGIES:
        n_surv = sum(1 for row in rows if row["strategy"] == sname and row["survived"])
        r = strat_streams[sname]
        if r.std() == 0:
            print(f"  {sname:<13}  survivors={n_surv:>2d}   (no survivors)")
            continue
        full = _stats(r)
        is_ = _stats(r.loc[r.index < SPLIT])
        oos = _stats(r.loc[r.index >= SPLIT])
        y22 = _stats(r.loc[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                          (r.index < pd.Timestamp("2023-01-01", tz="UTC"))])
        print(f"  {sname:<13}  surv={n_surv:>2d}  "
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
    g_piv = (bd.groupby(["strategy", "asset_class"])["gate_rate"].mean().unstack(fill_value=0.0))
    print(g_piv.round(3).to_string())

    # Correlation of quality sleeve with TSMOM and TREND_NEW
    print()
    print("--- Correlation with existing sleeves (post-2020-04 daily) ---")
    try:
        tsmom = pd.read_parquet(OUT_DIR.parent / "quant" / "tsmom_returns.parquet")
        trend_new = pd.read_parquet(OUT_DIR.parent / "wave3" / "trend_returns.parquet")
        def _ser(df_):
            return pd.Series(df_["ret"].values, index=pd.to_datetime(df_["timestamp"], utc=True))
        ts = _ser(tsmom).reindex(quality_sleeve.index).fillna(0.0)
        tn = _ser(trend_new).reindex(quality_sleeve.index).fillna(0.0)
        c_ts = float(quality_sleeve.corr(ts))
        c_tn = float(quality_sleeve.corr(tn))
        print(f"  corr(quality, TSMOM)     = {c_ts:+.3f}")
        print(f"  corr(quality, TREND_NEW) = {c_tn:+.3f}")
    except Exception as e:
        print(f"  could not compute correlations: {e}")

    # Yearly Sharpes of quality sleeve
    print()
    print("--- Quality sleeve Sharpe by year ---")
    for yr, sub in quality_sleeve.groupby(quality_sleeve.index.year):
        st = _stats(sub)
        print(f"  {yr}  Sh={st['sharpe']:+5.2f}  Ret={st['ann_return']:+7.2%}  "
              f"Vol={st['ann_vol']:6.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print(f"Wrote {OUT_DIR / 'mom_quality_returns.parquet'}")
    print(f"Wrote {OUT_DIR / 'mom_quality_breakdown.csv'}")


if __name__ == "__main__":
    main()
