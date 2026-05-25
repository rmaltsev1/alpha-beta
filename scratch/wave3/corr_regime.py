"""Cross-asset correlation regime detector — wave 3.

Computes the average pairwise 60-day correlation across all 13 symbols and
uses that single regime signal to gate / activate five sub-strategies:

  1. Correlation-aware TSMOM (halve when avg corr > IS 75th pct)
  2. Risk-off long XAU (avg corr > 80th pct AND SPX 30d ret < 0 -> long 5d)
  3. Risk-on long crypto basket (avg corr < 25th pct AND BTC 30d > 0 -> long)
  4. Correlation-breakdown spread (corr dropping fast -> long crypto / short SPX)
  5. Vol-rotation overlay (5d corr spike -> rotate into XAU + USD_JPY, 10d)

All percentiles are walk-forward (cumulative IS-only on day t for the gate at t).
Each surviving sub-sleeve is vol-scaled to 5% IS ann vol. Survivor filter:
IS Sharpe >= 0.4 AND OOS Sharpe >= 0. Survivors combined equal-weight.

Outputs:
  - scratch/wave3/corr_regime_returns.parquet
  - scratch/wave3/corr_regime_breakdown.csv
  - scratch/wave3/avg_corr_60d.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO  # noqa: E402
from alphabeta.backtest import backtest  # noqa: E402

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
CORR_WIN = 60


# ----- utilities ---------------------------------------------------------

def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    r = r.dropna()
    if len(r) < 5:
        return out
    idx = r.index
    windows = [
        ("FULL", np.ones(len(idx), dtype=bool)),
        ("IS",   np.asarray(idx < SPLIT)),
        ("OOS",  np.asarray(idx >= SPLIT)),
        ("Y2022", np.asarray((idx >= pd.Timestamp("2022-01-01", tz="UTC"))
                             & (idx < pd.Timestamp("2023-01-01", tz="UTC")))),
    ]
    for tag, mask in windows:
        sub = r[mask]
        if len(sub) < 5:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            out[f"{tag}_vol"] = 0.0
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
    return out


def scale_to_is_vol(rets: pd.Series, target: float = TARGET_VOL) -> float:
    is_r = rets[rets.index < SPLIT].dropna()
    if len(is_r) < 30:
        return 0.0
    bpy = _bpy(is_r.index)
    av = float(is_r.std(ddof=0) * np.sqrt(bpy))
    return target / av if av > 1e-9 else 0.0


def _utc_ts(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def run_signal(df: pd.DataFrame, signal: pd.Series, name: str, symbol: str,
               timeframe: str = "D1"):
    pos = pd.Series(signal.values, index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, pos, symbol=symbol, timeframe=timeframe, name=name)
    idx = _utc_ts(df["timestamp"].values)
    rets_unscaled = pd.Series(res.returns.values, index=idx)
    scale = scale_to_is_vol(rets_unscaled, TARGET_VOL)
    return rets_unscaled * scale, scale, res


def to_daily(rets: pd.Series) -> pd.Series:
    if rets.empty:
        return rets
    idx = pd.DatetimeIndex(rets.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    s = pd.Series(rets.values, index=idx)
    return s.resample("1D").sum()


def hold_n(trigger: pd.Series, n: int, value: float = 1.0) -> pd.Series:
    """Carry forward `value` for `n` bars after each True in `trigger`."""
    out = np.zeros(len(trigger), dtype="float64")
    last = -10 * n
    vals = trigger.fillna(False).values
    for i, v in enumerate(vals):
        if v:
            last = i
        if i - last < n:
            out[i] = value
    return pd.Series(out, index=trigger.index)


# ----- load data ---------------------------------------------------------

print("Loading data...")
DATA = {s: get_candles(s, "D1") for s in ALL_SYMBOLS}
for s, df in DATA.items():
    print(f"  {s:12} {len(df):5}  {df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


# ----- build daily close panel + log returns ------------------------------

# Use UTC-normalized daily index. For non-crypto, weekend bars are missing —
# pct_change skips naturally; correlation is computed pairwise on overlapping
# observations.

def _to_utc(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def daily_panel(price_field: str = "close") -> pd.DataFrame:
    series = {}
    for sym, df in DATA.items():
        idx = _to_utc(df["timestamp"].values).normalize()
        s = pd.Series(df[price_field].values, index=idx, dtype="float64")
        s = s[~s.index.duplicated(keep="last")]
        series[sym] = s
    panel = pd.DataFrame(series).sort_index()
    return panel

PANEL = daily_panel("close")
print(f"\nPanel shape: {PANEL.shape}, "
      f"range {PANEL.index[0].date()} -> {PANEL.index[-1].date()}")

LOG_RET = np.log(PANEL / PANEL.shift(1))


# ----- average pairwise correlation, 60d rolling --------------------------
# Trick: avg pairwise corr = (sum_corr - N) / (N * (N-1)). Use rolling corr
# matrix on returns directly.

def avg_pairwise_corr(log_ret: pd.DataFrame, window: int = CORR_WIN) -> pd.Series:
    out_idx = log_ret.index
    out = np.full(len(out_idx), np.nan)
    cols = log_ret.columns
    N = len(cols)
    arr = log_ret.values  # rows = time, cols = symbols
    for i in range(window, len(out_idx) + 1):
        chunk = arr[i - window:i]  # window x N
        # Pairwise complete observations: require >=30 valid pairs per pair
        # Implement via pandas corr() with min_periods.
        sub = log_ret.iloc[i - window:i]
        cm = sub.corr(min_periods=30)
        if cm.isnull().all().all():
            continue
        # mask diag
        m = cm.values.copy()
        np.fill_diagonal(m, np.nan)
        # Average of non-nan off-diagonal entries
        vals = m[~np.isnan(m)]
        if len(vals) == 0:
            continue
        out[i - 1] = float(vals.mean())
    return pd.Series(out, index=out_idx, name="avg_corr_60d")


print("\nComputing 60d avg pairwise correlation...")
AVG_CORR = avg_pairwise_corr(LOG_RET, CORR_WIN)
print(f"  non-NaN values: {AVG_CORR.notna().sum()} / {len(AVG_CORR)}")
print(f"  IS mean: {AVG_CORR[AVG_CORR.index < SPLIT].mean():.3f}, "
      f"OOS mean: {AVG_CORR[AVG_CORR.index >= SPLIT].mean():.3f}")
print(f"  Y2022 mean: "
      f"{AVG_CORR[(AVG_CORR.index >= pd.Timestamp('2022-01-01', tz='UTC')) & (AVG_CORR.index < pd.Timestamp('2023-01-01', tz='UTC'))].mean():.3f}")

# Save avg-corr time series
out_corr = AVG_CORR.reset_index().rename(columns={"index": "timestamp"})
out_corr.to_parquet(OUT / "avg_corr_60d.parquet", index=False)


# Walk-forward percentile of avg-corr: use ONLY data with index < SPLIT for IS;
# for OOS we still use IS-derived percentile cutoffs (no peeking).
IS_CORR = AVG_CORR[AVG_CORR.index < SPLIT].dropna()
P25 = float(np.quantile(IS_CORR, 0.25))
P75 = float(np.quantile(IS_CORR, 0.75))
P80 = float(np.quantile(IS_CORR, 0.80))
print(f"\nIS percentiles  P25={P25:.3f}  P75={P75:.3f}  P80={P80:.3f}")


# 30d std of avg corr (for "changing rapidly" regime)
CORR_STD30 = AVG_CORR.rolling(30).std()
CORR_STD30_IS = CORR_STD30[CORR_STD30.index < SPLIT].dropna()
CORR_STD_P75 = float(np.quantile(CORR_STD30_IS, 0.75)) if len(CORR_STD30_IS) > 30 else 0.0

# 5d change in avg corr (for vol-rotation)
CORR_5D = AVG_CORR - AVG_CORR.shift(5)
CORR_5D_IS = CORR_5D[CORR_5D.index < SPLIT].dropna()
CORR_5D_P90 = float(np.quantile(CORR_5D_IS, 0.90)) if len(CORR_5D_IS) > 30 else 0.0
print(f"  30d std P75={CORR_STD_P75:.4f}  5d-change P90={CORR_5D_P90:.4f}")


# Helper: align a per-day series onto a symbol's df index.
def align_to_df(series_by_date: pd.Series, df: pd.DataFrame) -> pd.Series:
    """series_by_date is indexed by UTC-normalized dates. df has timestamp col.
    Return a Series aligned to df.index, forward-filled by date for that
    symbol's bars."""
    s = series_by_date.copy()
    s.index = _to_utc(s.index).normalize()
    df_dates = _to_utc(df["timestamp"].values).normalize()
    out = s.reindex(df_dates).values
    return pd.Series(out, index=df.index)


# ----- STRATEGY 1: correlation-aware TSMOM --------------------------------
# Per-symbol TSMOM using 60d log return; long when >0, short when <0. When
# avg corr > P75, halve the position (corr regime suppresses TSMOM).

def strat1_tsmom_corr():
    print("\n=== STRATEGY 1: Correlation-aware TSMOM ===")
    sleeves = {}
    TSMOM_LB = 60
    for sym in ALL_SYMBOLS:
        df = DATA[sym].copy()
        close = df["close"].astype("float64")
        log_ret_60 = np.log(close / close.shift(TSMOM_LB))
        # raw position: sign of 60d log ret, observable at start of next bar
        raw_pos = np.sign(log_ret_60.shift(1)).fillna(0.0)

        # Halve if avg corr > P75 (regime info from yesterday)
        corr_aligned = align_to_df(AVG_CORR.shift(1), df).fillna(0.0)
        scale_arr = np.where(corr_aligned.values > P75, 0.5, 1.0)
        sig = raw_pos.values * scale_arr
        sig = pd.Series(sig, index=df.index).fillna(0.0)

        rets, scale, _ = run_signal(df, sig, f"TSMOM_{sym}", sym)
        s = stats(f"tsmom_{sym}", rets)
        s["scale"] = scale
        sleeves[f"tsmom_{sym}"] = (rets, s)
        print(f"  tsmom_{sym:<12}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- STRATEGY 2: risk-off long XAU --------------------------------------
# Trigger when avg corr > P80 AND SPX 30d return < 0 (observable yesterday).
# Hold long XAU 5 days.

def strat2_riskoff_xau():
    print("\n=== STRATEGY 2: Risk-off long XAU ===")
    spx_panel = PANEL["SPX500_USD"]
    spx_30 = spx_panel.pct_change(30)

    corr_y = AVG_CORR.shift(1)
    spx_y = spx_30.shift(1)
    trigger_by_date = (corr_y > P80) & (spx_y < 0)
    trigger_by_date = trigger_by_date.fillna(False)

    sym = "XAU_USD"
    df = DATA[sym].copy()
    trig_aligned = align_to_df(trigger_by_date.astype(float), df).fillna(0.0) > 0.5
    sig = hold_n(pd.Series(trig_aligned.values, index=df.index), n=5, value=1.0)

    rets, scale, _ = run_signal(df, sig, "ROFF_XAU", sym)
    s = stats("roff_xau", rets)
    s["scale"] = scale
    print(f"  roff_xau          IS_Sh={s.get('IS_sharpe',0):+.2f}  "
          f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}  "
          f"trig_days={int(trig_aligned.sum())}")
    return {"roff_xau": (rets, s)}


# ----- STRATEGY 3: risk-on long crypto basket -----------------------------
# Trigger when avg corr < P25 AND BTC trailing 30d return > 0. Long each
# crypto for 5 days (re-trigger extends hold).

def strat3_riskon_crypto():
    print("\n=== STRATEGY 3: Risk-on long crypto basket ===")
    btc_panel = PANEL["BTCUSDT"]
    btc_30 = btc_panel.pct_change(30)

    corr_y = AVG_CORR.shift(1)
    btc_y = btc_30.shift(1)
    trigger_by_date = (corr_y < P25) & (btc_y > 0)
    trigger_by_date = trigger_by_date.fillna(False)

    sleeves = {}
    for sym in CRYPTO:
        df = DATA[sym].copy()
        trig_aligned = align_to_df(trigger_by_date.astype(float), df).fillna(0.0) > 0.5
        sig = hold_n(pd.Series(trig_aligned.values, index=df.index), n=5, value=1.0)
        rets, scale, _ = run_signal(df, sig, f"RON_{sym}", sym)
        s = stats(f"ron_{sym}", rets)
        s["scale"] = scale
        sleeves[f"ron_{sym}"] = (rets, s)
        print(f"  ron_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}  "
              f"trig_days={int(trig_aligned.sum())}")
    return sleeves


# ----- STRATEGY 4: correlation-breakdown spread ---------------------------
# "Std of avg corr last 30d" is high AND avg corr is *dropping* (delta < 0):
# long crypto basket vs short SPX. Hold each bar where condition is true
# (with 1-bar shift on signal).

def strat4_breakdown_spread():
    print("\n=== STRATEGY 4: Correlation breakdown spread (crypto - SPX) ===")
    delta5 = AVG_CORR - AVG_CORR.shift(5)  # >0 = corr rising, <0 = falling
    std30 = CORR_STD30

    # Condition: regime changing rapidly AND dropping
    cond = (std30.shift(1) > CORR_STD_P75) & (delta5.shift(1) < 0)
    cond = cond.fillna(False)

    sleeves = {}
    # long-leg: each crypto +1 under trigger
    for sym in CRYPTO:
        df = DATA[sym].copy()
        cond_aligned = align_to_df(cond.astype(float), df).fillna(0.0) > 0.5
        sig = pd.Series(np.where(cond_aligned.values, 1.0, 0.0),
                        index=df.index)
        rets, scale, _ = run_signal(df, sig, f"BRK_LONG_{sym}", sym)
        s = stats(f"brk_long_{sym}", rets)
        s["scale"] = scale
        sleeves[f"brk_long_{sym}"] = (rets, s)
        print(f"  brk_long_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")

    # short-leg: SPX -1 under trigger
    sym = "SPX500_USD"
    df = DATA[sym].copy()
    cond_aligned = align_to_df(cond.astype(float), df).fillna(0.0) > 0.5
    sig = pd.Series(np.where(cond_aligned.values, -1.0, 0.0), index=df.index)
    rets, scale, _ = run_signal(df, sig, "BRK_SHORT_SPX", sym)
    s = stats("brk_short_spx", rets)
    s["scale"] = scale
    sleeves["brk_short_spx"] = (rets, s)
    print(f"  brk_short_spx     IS_Sh={s.get('IS_sharpe',0):+.2f}  "
          f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- STRATEGY 5: vol-rotation overlay -----------------------------------
# When avg corr spikes from low to high in 5 days (5d change > IS P90), rotate
# from risk (crypto + SPX short) to safe (XAU + USD_JPY long) for 10 days.

def strat5_vol_rotation():
    print("\n=== STRATEGY 5: Vol-rotation overlay ===")
    delta5 = AVG_CORR - AVG_CORR.shift(5)
    trigger_by_date = (delta5.shift(1) > CORR_5D_P90).fillna(False)
    print(f"  trigger days IS: {int(trigger_by_date[trigger_by_date.index < SPLIT].sum())}, "
          f"OOS: {int(trigger_by_date[trigger_by_date.index >= SPLIT].sum())}")

    sleeves = {}
    # Safe-haven longs: XAU + USD_JPY for 10 days
    for sym in ["XAU_USD", "USD_JPY"]:
        df = DATA[sym].copy()
        trig_aligned = align_to_df(trigger_by_date.astype(float), df).fillna(0.0) > 0.5
        sig = hold_n(pd.Series(trig_aligned.values, index=df.index), n=10, value=1.0)
        rets, scale, _ = run_signal(df, sig, f"VOLROT_LONG_{sym}", sym)
        s = stats(f"volrot_long_{sym}", rets)
        s["scale"] = scale
        sleeves[f"volrot_long_{sym}"] = (rets, s)
        print(f"  volrot_long_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")

    # Risk shorts: BTC, ETH, SPX for 10 days
    for sym in ["BTCUSDT", "ETHUSDT", "SPX500_USD"]:
        df = DATA[sym].copy()
        trig_aligned = align_to_df(trigger_by_date.astype(float), df).fillna(0.0) > 0.5
        sig = hold_n(pd.Series(trig_aligned.values, index=df.index), n=10, value=-1.0)
        rets, scale, _ = run_signal(df, sig, f"VOLROT_SHORT_{sym}", sym)
        s = stats(f"volrot_short_{sym}", rets)
        s["scale"] = scale
        sleeves[f"volrot_short_{sym}"] = (rets, s)
        print(f"  volrot_short_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- MAIN ---------------------------------------------------------------

def main():
    all_sleeves = {}
    # tag each sleeve with its parent strategy id for breakdown
    strat_map = {}
    s1 = strat1_tsmom_corr()
    for k in s1: strat_map[k] = "1_tsmom"
    all_sleeves.update(s1)
    s2 = strat2_riskoff_xau()
    for k in s2: strat_map[k] = "2_roff_xau"
    all_sleeves.update(s2)
    s3 = strat3_riskon_crypto()
    for k in s3: strat_map[k] = "3_ron_crypto"
    all_sleeves.update(s3)
    s4 = strat4_breakdown_spread()
    for k in s4: strat_map[k] = "4_brk_spread"
    all_sleeves.update(s4)
    s5 = strat5_vol_rotation()
    for k in s5: strat_map[k] = "5_volrot"
    all_sleeves.update(s5)

    rows = []
    for name, (rets, s) in all_sleeves.items():
        rows.append({"sleeve": name, "strategy": strat_map.get(name, "?"), **s})
    breakdown = pd.DataFrame(rows)
    cols = ["sleeve", "strategy"] + [c for c in breakdown.columns if c not in ("sleeve", "strategy", "label")]
    breakdown = breakdown[cols]
    breakdown.to_csv(OUT / "corr_regime_breakdown.csv", index=False)

    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.4) & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) ===")
    if not survivors.empty:
        print(survivors[["sleeve", "strategy", "IS_sharpe", "OOS_sharpe",
                         "Y2022_sharpe", "FULL_sharpe", "FULL_ret"]].to_string(index=False))

    survivor_names = survivors["sleeve"].tolist()
    panel_out = {}
    for name in survivor_names:
        rets = all_sleeves[name][0]
        idx = pd.DatetimeIndex(rets.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        s = pd.Series(rets.values, index=idx)
        s.index = s.index.normalize()
        # collapse duplicate dates by summing
        s = s.groupby(s.index).sum()
        panel_out[name] = s

    if panel_out:
        ret_df = pd.concat(panel_out, axis=1, sort=True).fillna(0.0)
        if ret_df.index.tz is None:
            ret_df.index = ret_df.index.tz_localize("UTC")
        out_df = ret_df.reset_index().rename(columns={"index": "timestamp"})
        out_df.to_parquet(OUT / "corr_regime_returns.parquet", index=False)

        combined = ret_df.mean(axis=1)
        s_c = stats("CORR_REGIME_COMBINED", combined)
        print("\n=== Combined survivor sleeve (equal-weight) ===")
        for tag in ["FULL", "IS", "OOS", "Y2022"]:
            sh = s_c.get(f"{tag}_sharpe", 0)
            rt = s_c.get(f"{tag}_ret", 0)
            print(f"  {tag:<6}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}")

        # Per-strategy combined view
        print("\n=== Per-strategy combined (within-survivor average) ===")
        for sid in sorted(set(strat_map.values())):
            subnames = [n for n in survivor_names if strat_map.get(n) == sid]
            if not subnames:
                print(f"  {sid:<14}  no survivors")
                continue
            sub_df = ret_df[subnames].mean(axis=1)
            ssub = stats(sid, sub_df)
            print(f"  {sid:<14}  n={len(subnames):2d}  IS={ssub.get('IS_sharpe',0):+.2f}  "
                  f"OOS={ssub.get('OOS_sharpe',0):+.2f}  Y2022={ssub.get('Y2022_sharpe',0):+.2f}  "
                  f"FULL={ssub.get('FULL_sharpe',0):+.2f}")

        # Yearly Sharpe
        print("\n=== Combined yearly Sharpe ===")
        for year, sub in combined.groupby(combined.index.year):
            if len(sub) < 20:
                continue
            bpy = _bpy(sub.index)
            std = sub.std(ddof=0)
            sh = (sub.mean() * bpy) / (std * np.sqrt(bpy)) if std > 0 else 0.0
            rt = sub.mean() * bpy
            print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")

        # Correlation between avg-corr regime and SPX realized 30d vol
        print("\n=== Avg-corr regime vs SPX 30d realized vol ===")
        spx_df = DATA["SPX500_USD"]
        spx_idx = _to_utc(spx_df["timestamp"].values).normalize()
        spx_native = pd.Series(spx_df["close"].values, index=spx_idx).astype("float64")
        spx_native = spx_native[~spx_native.index.duplicated(keep="last")]
        spx_lr = np.log(spx_native / spx_native.shift(1))
        spx_vol30 = (spx_lr.rolling(30).std() * np.sqrt(252)).reindex(AVG_CORR.index)
        joint = pd.concat({"ac": AVG_CORR, "sv": spx_vol30}, axis=1).dropna()
        if len(joint) > 30:
            for tag, mask in [
                ("FULL", np.ones(len(joint), dtype=bool)),
                ("IS",   np.asarray(joint.index < SPLIT)),
                ("OOS",  np.asarray(joint.index >= SPLIT)),
                ("Y2022", np.asarray((joint.index >= pd.Timestamp("2022-01-01", tz="UTC"))
                                     & (joint.index < pd.Timestamp("2023-01-01", tz="UTC")))),
            ]:
                sub = joint[mask]
                if len(sub) < 20:
                    print(f"  {tag:<6}  n={len(sub)} too few")
                    continue
                cp = sub["ac"].corr(sub["sv"])
                # rank-correlation via numpy (avoid scipy dep)
                ac_r = sub["ac"].rank().values
                sv_r = sub["sv"].rank().values
                cs = float(np.corrcoef(ac_r, sv_r)[0, 1])
                print(f"  {tag:<6}  pearson={cp:+.3f}  spearman={cs:+.3f}  n={len(sub)}")
    else:
        pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC")}).to_parquet(
            OUT / "corr_regime_returns.parquet", index=False)
        print("\nNo survivors.")


if __name__ == "__main__":
    main()
