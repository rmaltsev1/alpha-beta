"""Per-asset multi-timeframe ensembles (wave 5).

For each of the 13 symbols, stack M15/H1/H4/D1 signals into a SINGLE D1-rebalanced
position, vol-managed. Costs are bounded by D1 turnover rather than M15 noise,
even though the signals look at fine timeframes.

Five ensembles:
  1. SIGNVOTE     -- sign(trailing-N return) per TF (N: M15=96, H1=24, H4=12, D1=21).
                     position = +1 / -1 if >=3 TFs agree on sign, else 0.
  2. ZSCORE       -- z = current-bar return / 20-bar rolling std per TF; avg across TFs.
                     position = +/-1 when |avg z| > 0.5.
  3. TRENDQUAL    -- "all TFs trending up AND slopes rising" (or symmetrically down).
                     Stricter than SIGNVOTE.
  4. MEANREV      -- short setup: H1 1-day-return > +2 sigma AND H4 trend <= 0 AND D1 RSI < 40.
                     Symmetric long setup with the signs flipped.
  5. VOLREGIME    -- D1 sign(21d) momentum, halved when M15 realized vol > IS p80.

Methodology:
  - IS  <  2024-01-01,  OOS >= 2024-01-01.
  - All signals walk-forward (shift(1) per TF before re-sampling to D1).
  - Vol-scale each surviving per-(ensemble, symbol) stream to 5% IS ann vol.
  - Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
  - Combine survivors equal-weight across symbols within each ensemble.
  - Equal-weight across the 5 ensembles -> final sleeve, rescaled to 5% IS ann vol.

Outputs:
  scratch/wave5/multi_tf.py                 -- this script
  scratch/wave5/multi_tf_returns.parquet    -- D1 sleeve, timestamp (UTC), ret
  scratch/wave5/multi_tf_breakdown.csv      -- per-(ensemble,symbol) IS/OOS/FULL/2022 stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, SYMBOL_TYPE
from alphabeta.backtest import backtest, cost_for


# -- knobs --------------------------------------------------------------------
SPLIT       = pd.Timestamp("2024-01-01", tz="UTC")
TFS         = ["M15", "H1", "H4", "D1"]
NBARS       = {"M15": 96, "H1": 24, "H4": 12, "D1": 21}   # signal lookbacks per TF
ZWIN        = 20                                          # rolling std for z-score
RSI_WIN     = 14
RSI_LOW     = 40.0
RSI_HIGH    = 60.0
H1_SIGMA_K  = 2.0
TARGET_VOL  = 0.05
POS_CAP     = 3.0
MIN_IS_SHRP = 0.5
MIN_OOS_SHRP = 0.0
BPY         = 365.25                                       # daily annualization (calendar)
OUT_DIR     = Path(__file__).resolve().parent


# -- helpers ------------------------------------------------------------------
def _stats(r: pd.Series, freq: float = BPY) -> dict:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0, "n": int(len(r))}
    mu = r.mean() * freq
    sd = r.std(ddof=0) * np.sqrt(freq)
    sh = mu / sd if sd > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"sharpe": float(sh), "ann_return": float(mu), "ann_vol": float(sd),
            "max_dd": float(dd), "n": int(len(r))}


def _utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _vol_scale_is(ret: pd.Series, split_ts: pd.Timestamp, target: float = TARGET_VOL) -> float:
    is_r = ret.loc[ret.index < split_ts].dropna()
    if is_r.empty:
        return 0.0
    sd = is_r.std(ddof=0) * np.sqrt(BPY)
    if sd <= 0:
        return 0.0
    return float(target / sd)


def _rsi(close: pd.Series, win: int = RSI_WIN) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.rolling(win, min_periods=win).mean()
    roll_dn = down.rolling(win, min_periods=win).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _slope(close: pd.Series, win: int) -> pd.Series:
    """Trailing OLS slope over `win` bars on log price; positive = rising trend."""
    lp = np.log(close.astype("float64").replace(0, np.nan))
    # quick approximation: change in 'win'-bar moving average vs win/2 bars ago.
    ma = lp.rolling(win, min_periods=win).mean()
    return ma - ma.shift(max(1, win // 4))


# -- per-TF signal builders ---------------------------------------------------
# All builders return per-(TF-bar) series aligned to df.index, AND a timestamp series.
# They are walk-forward: position[t] uses info up to t-1 (final shift(1)).

def _per_tf_signals(symbol: str) -> dict[str, pd.DataFrame]:
    """Compute everything we need from each TF, returned as DataFrames keyed by TF.

    Each row has: timestamp (tz-aware UTC), close, sign_n, z_n, trend_pos, slope_pos,
    ret_1d (1-day return ~ N bars), sigma_1d, rsi.
    """
    out = {}
    for tf in TFS:
        df = get_candles(symbol, tf).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        c = df["close"].astype("float64")
        lr = np.log(c / c.shift(1))
        n = NBARS[tf]

        # 1) sign of trailing N-bar return (log-return sum)
        sign_n = np.sign(lr.rolling(n, min_periods=n).sum()).fillna(0.0)

        # 2) z-score of current bar return vs rolling std
        sd = lr.rolling(ZWIN, min_periods=ZWIN).std(ddof=0)
        z_n = (lr / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # 3) trend positive (sign of N-bar return) + slope rising
        slope = _slope(c, n)
        trend_pos = (sign_n > 0).astype("float64") - (sign_n < 0).astype("float64")
        slope_pos = np.sign(slope).fillna(0.0)

        # 4) one-"day"-ish return on this TF + its rolling std (for mean reversion)
        # H1 1-day == 24 bars; H4 1-day == 6 bars; D1 1-day == 1 bar; M15 1-day == 96 bars
        day_bars = {"M15": 96, "H1": 24, "H4": 6, "D1": 1}[tf]
        ret_1d = c.pct_change(day_bars)
        sigma_1d = ret_1d.rolling(max(20, day_bars), min_periods=max(20, day_bars)).std(ddof=0)

        # 5) RSI on close
        rsi = _rsi(c, RSI_WIN)

        # Walk-forward shift: every feature is shifted by 1 bar of its own TF so
        # they're observable at bar t (i.e. at the close of t-1).
        feat = pd.DataFrame({
            "timestamp": df["timestamp"].values,
            "close": c.values,
            "sign_n": sign_n.shift(1).fillna(0.0).values,
            "z_n":    z_n.shift(1).fillna(0.0).values,
            "trend_pos": trend_pos.shift(1).fillna(0.0).values,
            "slope_pos": slope_pos.shift(1).fillna(0.0).values,
            "ret_1d":  ret_1d.shift(1).fillna(0.0).values,
            "sigma_1d": sigma_1d.shift(1).values,
            "rsi":     rsi.shift(1).fillna(50.0).values,
        })
        out[tf] = feat
    return out


def _resample_to_d1(feat: pd.DataFrame, d1_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """For a per-TF feature frame, take the LAST value at-or-before each D1 bar.

    Concretely: build a series indexed by feature timestamps, then asof-merge onto
    d1_idx. Because the feature is already shift(1)-ed, the value at d1_idx[t] is
    a feature observable at the close of the previous D1 bar (one D1 bar of lag).
    """
    out = pd.DataFrame(index=d1_idx)
    feat = feat.sort_values("timestamp")
    feat_idx = pd.DatetimeIndex(pd.to_datetime(feat["timestamp"], utc=True))
    for col in ["sign_n", "z_n", "trend_pos", "slope_pos", "ret_1d", "sigma_1d", "rsi"]:
        s = pd.Series(feat[col].values, index=feat_idx).sort_index()
        # asof: last value at-or-before each d1 timestamp
        # use reindex with method='ffill' after concat to align cleanly
        aligned = pd.merge_asof(
            pd.DataFrame({"ts": d1_idx}).sort_values("ts"),
            pd.DataFrame({"ts": feat_idx, col: feat[col].values}).sort_values("ts"),
            on="ts", direction="backward",
        )
        out[col] = aligned[col].values
    return out


# -- ensemble position constructors ------------------------------------------
def pos_signvote(feats_at_d1: dict[str, pd.DataFrame]) -> pd.Series:
    # majority across the 4 TF signs
    signs = pd.concat([feats_at_d1[tf]["sign_n"].rename(tf) for tf in TFS], axis=1).fillna(0.0)
    pos_count = (signs > 0).sum(axis=1)
    neg_count = (signs < 0).sum(axis=1)
    pos = pd.Series(0.0, index=signs.index)
    pos[pos_count >= 3] = 1.0
    pos[neg_count >= 3] = -1.0
    return pos


def pos_zscore(feats_at_d1: dict[str, pd.DataFrame]) -> pd.Series:
    zs = pd.concat([feats_at_d1[tf]["z_n"].rename(tf) for tf in TFS], axis=1)
    avg_z = zs.mean(axis=1)
    pos = pd.Series(0.0, index=avg_z.index)
    pos[avg_z > 0.5] = 1.0
    pos[avg_z < -0.5] = -1.0
    return pos


def pos_trendqual(feats_at_d1: dict[str, pd.DataFrame]) -> pd.Series:
    # all TF trends agree AND all slopes agree.
    tps = pd.concat([feats_at_d1[tf]["trend_pos"].rename(tf) for tf in TFS], axis=1).fillna(0.0)
    sps = pd.concat([feats_at_d1[tf]["slope_pos"].rename(tf) for tf in TFS], axis=1).fillna(0.0)
    all_up = (tps > 0).all(axis=1) & (sps > 0).all(axis=1)
    all_dn = (tps < 0).all(axis=1) & (sps < 0).all(axis=1)
    pos = pd.Series(0.0, index=tps.index)
    pos[all_up] = 1.0
    pos[all_dn] = -1.0
    return pos


def pos_meanrev(feats_at_d1: dict[str, pd.DataFrame]) -> pd.Series:
    h1 = feats_at_d1["H1"]
    h4 = feats_at_d1["H4"]
    d1 = feats_at_d1["D1"]
    # short: H1 1d-return > +2sigma  &  H4 trend <= 0  &  D1 RSI < 40
    h1_blowoff_up = (h1["ret_1d"] > H1_SIGMA_K * h1["sigma_1d"]) & h1["sigma_1d"].notna()
    h4_trend_dn   = (h4["trend_pos"] <= 0)
    d1_oversold   = (d1["rsi"] < RSI_LOW)
    short_sig = h1_blowoff_up & h4_trend_dn & d1_oversold
    # long: H1 1d-return < -2sigma  &  H4 trend >= 0  &  D1 RSI > 60
    h1_blowoff_dn = (h1["ret_1d"] < -H1_SIGMA_K * h1["sigma_1d"]) & h1["sigma_1d"].notna()
    h4_trend_up   = (h4["trend_pos"] >= 0)
    d1_overbought = (d1["rsi"] > RSI_HIGH)
    long_sig = h1_blowoff_dn & h4_trend_up & d1_overbought
    pos = pd.Series(0.0, index=h1.index)
    pos[short_sig] = -1.0
    pos[long_sig]  =  1.0
    return pos


def pos_volregime(feats_at_d1: dict[str, pd.DataFrame], m15_rv: pd.Series,
                  rv_threshold: float) -> pd.Series:
    """D1 sign(21d) momentum, halved when M15 realized vol > IS p80 threshold."""
    d1_dir = feats_at_d1["D1"]["sign_n"]
    pos = d1_dir.copy()
    if rv_threshold > 0:
        hot = (m15_rv > rv_threshold).reindex(pos.index).fillna(False)
        pos[hot] = pos[hot] * 0.5
    return pos


# -- driver for one symbol ----------------------------------------------------
ENSEMBLES = ["SIGNVOTE", "ZSCORE", "TRENDQUAL", "MEANREV", "VOLREGIME"]


def _per_symbol(symbol: str) -> dict:
    df_d1 = get_candles(symbol, "D1").copy()
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"], utc=True)
    d1_idx = pd.DatetimeIndex(df_d1["timestamp"])

    feats = _per_tf_signals(symbol)
    feats_at_d1 = {tf: _resample_to_d1(feats[tf], d1_idx) for tf in TFS}

    # Realized vol from M15: stdev of M15 log returns within each D1 day, then
    # forward-shifted by 1 D1 bar (walk-forward).
    m15 = feats["M15"][["timestamp", "close"]].copy()
    m15["lr"] = np.log(m15["close"] / m15["close"].shift(1))
    m15["day"] = pd.to_datetime(m15["timestamp"], utc=True).dt.normalize()
    rv_daily = m15.groupby("day")["lr"].std(ddof=0)
    # threshold based on IS window only
    is_rv = rv_daily.loc[rv_daily.index < SPLIT].dropna()
    rv_threshold = float(np.quantile(is_rv, 0.80)) if len(is_rv) else 0.0

    # Align M15 daily RV to D1 calendar (shift 1 day so it's only observable next bar)
    m15_rv = rv_daily.shift(1)
    m15_rv.index = pd.DatetimeIndex(m15_rv.index)
    if m15_rv.index.tz is None:
        m15_rv.index = m15_rv.index.tz_localize("UTC")
    # match d1_idx via asof
    m15_rv_d1 = pd.merge_asof(
        pd.DataFrame({"ts": d1_idx}).sort_values("ts"),
        pd.DataFrame({"ts": m15_rv.index, "rv": m15_rv.values}).sort_values("ts"),
        on="ts", direction="backward",
    )["rv"].values
    m15_rv_d1 = pd.Series(m15_rv_d1, index=d1_idx)

    # Build positions for each ensemble
    positions = {
        "SIGNVOTE":  pos_signvote(feats_at_d1),
        "ZSCORE":    pos_zscore(feats_at_d1),
        "TRENDQUAL": pos_trendqual(feats_at_d1),
        "MEANREV":   pos_meanrev(feats_at_d1),
        "VOLREGIME": pos_volregime(feats_at_d1, m15_rv_d1, rv_threshold),
    }

    # Backtest each ensemble on D1 with proper costs.
    out = {"d1_idx": d1_idx, "returns": {}, "stats": {}, "trades": {}, "exposure": {}}
    for ename, pos in positions.items():
        pos = pos.clip(-POS_CAP, POS_CAP).fillna(0.0).reset_index(drop=True)
        # ensure index aligned to df_d1
        pos.index = df_d1.index
        res = backtest(df_d1, pos, symbol=symbol, timeframe="D1",
                       name=f"multi_tf_{ename}_{symbol}")
        ret = res.returns
        ret.index = d1_idx
        out["returns"][ename] = ret
        out["stats"][ename] = res.stats
        out["trades"][ename] = int(res.stats["n_trades"])
        out["exposure"][ename] = float(res.stats["exposure"])
    return out


# -- driver -------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building per-symbol multi-TF ensembles for", len(ALL_SYMBOLS), "symbols...")
    per_sym = {}
    for s in ALL_SYMBOLS:
        print(f"  {s} ...")
        per_sym[s] = _per_symbol(s)

    # ---- Filter survivors and assemble per-ensemble streams ----------------
    # Build a union UTC-daily index across all symbols.
    all_ts = sorted({pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo
                     else pd.Timestamp(t, tz="UTC")
                     for d in per_sym.values() for t in d["d1_idx"]})
    union_idx = pd.DatetimeIndex(all_ts)

    survival = []         # rows for breakdown CSV
    per_ens_streams = {e: [] for e in ENSEMBLES}

    def _is_oos_stats(r: pd.Series) -> dict:
        # r is already UTC tz-aware indexed
        is_r = r.loc[r.index < SPLIT]
        oos_r = r.loc[r.index >= SPLIT]
        full = _stats(r)
        is_ = _stats(is_r)
        oos = _stats(oos_r)
        y2022 = _stats(r.loc[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                             (r.index < pd.Timestamp("2023-01-01", tz="UTC"))])
        return full, is_, oos, y2022

    for sym in ALL_SYMBOLS:
        d1_idx = pd.DatetimeIndex(_utc(per_sym[sym]["d1_idx"]))
        for ename in ENSEMBLES:
            r = per_sym[sym]["returns"][ename]
            r.index = d1_idx  # tz-aware
            # collapse to one row per calendar day (sum -- many TFs already at D1)
            r_daily = r.groupby(r.index.normalize()).sum()
            r_daily.index = _utc(r_daily.index)
            full, is_, oos, y22 = _is_oos_stats(r_daily)

            survive = (is_["sharpe"] >= MIN_IS_SHRP) and (oos["sharpe"] >= MIN_OOS_SHRP)
            scale = _vol_scale_is(r_daily, SPLIT) if survive else 0.0
            scaled = (r_daily * scale).reindex(union_idx).fillna(0.0)

            survival.append({
                "ensemble": ename,
                "symbol": sym,
                "asset_class": SYMBOL_TYPE[sym].value,
                "is_sharpe": is_["sharpe"],
                "oos_sharpe": oos["sharpe"],
                "full_sharpe": full["sharpe"],
                "yr2022_sharpe": y22["sharpe"],
                "is_ann_return": is_["ann_return"],
                "oos_ann_return": oos["ann_return"],
                "full_ann_return": full["ann_return"],
                "full_max_dd": full["max_dd"],
                "n_trades": per_sym[sym]["trades"][ename],
                "exposure": per_sym[sym]["exposure"][ename],
                "survived": bool(survive),
                "vol_scale": scale,
            })
            if survive:
                per_ens_streams[ename].append(scaled.rename(f"{ename}_{sym}"))

    bd = pd.DataFrame(survival)
    bd.to_csv(OUT_DIR / "multi_tf_breakdown.csv", index=False)

    # ---- Ensemble streams: equal weight across surviving symbols -----------
    ens_streams = {}
    for ename, lst in per_ens_streams.items():
        if not lst:
            ens_streams[ename] = pd.Series(0.0, index=union_idx)
            continue
        mat = pd.concat(lst, axis=1).fillna(0.0)
        ens_streams[ename] = mat.mean(axis=1)

    # ---- Sleeve = equal-weight across ensembles that have any survivors ----
    active_ensembles = [e for e, lst in per_ens_streams.items() if lst]
    if not active_ensembles:
        sleeve_raw = pd.Series(0.0, index=union_idx)
    else:
        mat = pd.concat([ens_streams[e].rename(e) for e in active_ensembles], axis=1).fillna(0.0)
        sleeve_raw = mat.mean(axis=1)

    sleeve_scale = _vol_scale_is(sleeve_raw, SPLIT, target=TARGET_VOL)
    sleeve_bar = (sleeve_raw * sleeve_scale).rename("ret")
    # Collapse to one row per UTC calendar day. Within a day some symbols close
    # at 00:00 and others at ~22:00; per-bar streams are zero on bars that
    # aren't their close, so summing matches "mean across baskets that day".
    sleeve = sleeve_bar.groupby(sleeve_bar.index.normalize()).sum()
    sleeve.index = _utc(sleeve.index)
    sleeve.name = "ret"

    # ---- Save sleeve parquet (D1, UTC tz-aware timestamp, ret) -------------
    out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(OUT_DIR / "multi_tf_returns.parquet", index=False)

    # ---- Headline reporting ------------------------------------------------
    def block(label: str, r: pd.Series) -> None:
        st = _stats(r)
        print(f"  {label:<6} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print("=== MULTI-TF SLEEVE (D1, vol-scaled to {:.0%} IS) ===".format(TARGET_VOL))
    print(f"Sleeve IS vol scale factor: {sleeve_scale:.3f}")
    print(f"Active ensembles: {active_ensembles}")
    block("FULL", sleeve)
    block("IS",   sleeve.loc[sleeve.index <  SPLIT])
    block("OOS",  sleeve.loc[sleeve.index >= SPLIT])
    block("2022",
          sleeve.loc[(sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                     (sleeve.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

    print("\n--- Per-ensemble sleeve stats (post symbol-EW, pre sleeve-scale) ---")
    for ename in ENSEMBLES:
        n_surv = sum(1 for row in survival
                     if row["ensemble"] == ename and row["survived"])
        if n_surv == 0:
            print(f"  {ename:<10}  0 survivors")
            continue
        r = ens_streams[ename]
        full = _stats(r); is_ = _stats(r.loc[r.index < SPLIT]); oos = _stats(r.loc[r.index >= SPLIT])
        print(f"  {ename:<10}  survivors={n_surv:>2d}  "
              f"FULL Sh={full['sharpe']:+5.2f}  IS Sh={is_['sharpe']:+5.2f}  "
              f"OOS Sh={oos['sharpe']:+5.2f}")

    print("\n--- Survivors per ensemble per symbol ---")
    pivot = bd.pivot_table(index="symbol", columns="ensemble", values="survived", aggfunc="first")
    print(pivot.fillna(False).astype(bool).to_string())

    print("\n--- Per-ensemble survival counts by asset class ---")
    pivot2 = (bd[bd["survived"]]
              .groupby(["ensemble", "asset_class"]).size().unstack(fill_value=0))
    print(pivot2.to_string())

    print("\n--- Avg trades / yr per (ensemble, asset class) for survivors ---")
    # trades counted full-period; bars are calendar days approx 365
    tmp = bd[bd["survived"]].copy()
    if not tmp.empty:
        tmp["trades_yr"] = tmp["n_trades"] / 6.4   # ~6.4 years of data
        piv = tmp.pivot_table(index="ensemble", columns="asset_class",
                              values="trades_yr", aggfunc="mean").round(1)
        print(piv.to_string())

    # Yearly sleeve sharpes
    print("\n--- Sleeve sharpes by year ---")
    yr_groups = sleeve.groupby(sleeve.index.year)
    for yr, sub in yr_groups:
        st = _stats(sub)
        print(f"  {yr}  Sh={st['sharpe']:+5.2f}  Ret={st['ann_return']:+7.2%}  "
              f"Vol={st['ann_vol']:6.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    print(f"\nWrote {OUT_DIR / 'multi_tf_returns.parquet'}")
    print(f"Wrote {OUT_DIR / 'multi_tf_breakdown.csv'}")


if __name__ == "__main__":
    main()
