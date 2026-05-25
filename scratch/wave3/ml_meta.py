"""ML meta-signal: walk-forward ridge regression that dynamically weights sleeves.

Inputs
------
  scratch/quant/all_sleeve_returns_v3.parquet   (12 sleeves, daily, vol-scaled 5% IS)
  scratch/quant/pairs_expanded_returns.parquet  (1 sleeve)
  data/SPX500_USD/D1.parquet                    (SPX D1 for regime feats)

Features (computed walk-forward per bar):
  - SPX 30d realized vol percentile (IS distribution)
  - SPX 30d cumulative return (regime up/down)
  - Trailing-30d Sharpe per sleeve (13-dim if PAIRS, else 12)
  - 60d sleeve-pair correlation summary (avg + max abs)
  - Day-of-week one-hot (5 cols, Mon..Fri)
  - Day-of-month standardized

Target: next-day return per sleeve.

Model: ridge regression solved with numpy normal equations,
       w = (X'X + lambda*I)^-1 X'y. One model per sleeve. Re-fit monthly.
       Trailing 252-day window.

Construction: at each rebalance, predicted next-day sleeve return ->
       softmax(clip(pred,0,1) / T) -> long-only weights.

Baselines:
  - Equal-weight TOP8+PAIRS
  - Static TOP9 (no ML, equal-weight) -- same as above since TOP9 = TOP8 + PAIRS_EXP
  - Sharpe-tilted (rolling 12-month Sharpe -> softmax)

Overlay: existing per-sleeve regime gate, vol-target 10%, drawdown control.

Outputs
-------
  scratch/wave3/ml_meta_returns.parquet
  scratch/wave3/ml_meta_weights.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

from master_v4 import (
    fast_decay_tripwire, vol_target_overlay, drawdown_control,
    build_regime_mask, stats, _bpy, GATES_HIGH,
)

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

TOP9 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
        "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP"]


# --------------------------- data utilities ---------------------------

def normalize_idx(s):
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s.groupby(s.index.floor("D")).sum()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def load_pairs(path, name):
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return normalize_idx(pd.Series(df["ret"].values, index=ts).rename(name))


def rescale_is(s, target):
    is_part = s[s.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    return s * (target / av) if av > 1e-9 else s * 0


def load_panel():
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v3.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    pairs = load_pairs(QUANT / "pairs_expanded_returns.parquet", "PAIRS_EXP")
    pairs = rescale_is(pairs, 0.05)
    panel = panel.join(pairs.rename("PAIRS_EXP"), how="outer").fillna(0.0)
    panel = panel.sort_index()
    return panel


def load_spx():
    spx = pd.read_parquet(ROOT / "data" / "SPX500_USD" / "D1.parquet").copy()
    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)
    spx = spx.set_index("timestamp").sort_index()
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["ret"].rolling(30).std() * np.sqrt(252)
    spx["cum30"] = spx["close"] / spx["close"].shift(30) - 1
    return spx[["ret", "rv30", "cum30"]]


# --------------------------- features ---------------------------

def build_features(panel: pd.DataFrame, sleeves: list[str]) -> pd.DataFrame:
    """Build daily feature matrix, walk-forward safe (no future data)."""
    spx = load_spx()
    # Align SPX to panel calendar via merge_asof (backward)
    idx_df = pd.DataFrame({"timestamp": panel.index}).sort_values("timestamp")
    spx_r = spx.reset_index().rename(columns={"timestamp": "ts"}).sort_values("ts")
    spx_aligned = pd.merge_asof(idx_df, spx_r, left_on="timestamp", right_on="ts",
                                direction="backward")
    spx_aligned.index = panel.index

    # rv30 percentile within trailing 504d window (walk-forward)
    rv30 = spx_aligned["rv30"].copy()
    rv30 = rv30.shift(1)  # lag to be safe (use yesterday's rv30 for today's feature)
    # Build expanding/trailing percentile within IS distribution that grows over time
    rv30_pct = pd.Series(np.nan, index=rv30.index)
    arr = rv30.values
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        # Use a long-trailing window for percentile; expanding up to t-1
        # but cap window at 2 years for efficiency
        start = max(0, i - 504)
        hist = arr[start:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 30:
            rv30_pct.iloc[i] = 0.5
        else:
            rv30_pct.iloc[i] = float((hist < arr[i]).mean())
    cum30 = spx_aligned["cum30"].shift(1)

    feats = pd.DataFrame(index=panel.index)
    feats["rv30_pct"] = rv30_pct
    feats["cum30"] = cum30
    feats["cum30_sign"] = (cum30 > 0).astype(float)

    # Trailing-30d Sharpe per sleeve
    for s in sleeves:
        r = panel[s].shift(1)  # lag: today's feat uses up to yesterday
        mu = r.rolling(30).mean()
        sd = r.rolling(30).std(ddof=0)
        sh = (mu / sd.replace(0, np.nan)) * np.sqrt(252)
        feats[f"sh30_{s}"] = sh.fillna(0)

    # Cross-sleeve correlation summary (60d): avg pairwise corr + max abs
    pan_lag = panel[sleeves].shift(1)
    # Rolling correlation summary: compute scalar per day from a 60d window
    avg_corr = pd.Series(np.nan, index=panel.index)
    max_abs = pd.Series(np.nan, index=panel.index)
    arr = pan_lag.values
    for i in range(60, len(arr)):
        w = arr[i - 60:i]
        if np.isnan(w).any() or w.std(axis=0).min() == 0:
            continue
        c = np.corrcoef(w, rowvar=False)
        n = c.shape[0]
        iu = np.triu_indices(n, k=1)
        offdiag = c[iu]
        avg_corr.iloc[i] = float(np.nanmean(offdiag))
        max_abs.iloc[i] = float(np.nanmax(np.abs(offdiag)))
    feats["avg_corr"] = avg_corr.fillna(0)
    feats["max_abs_corr"] = max_abs.fillna(0)

    # Day-of-week one-hot (Mon..Fri)
    dow = feats.index.dayofweek
    for d, name in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        feats[f"dow_{name}"] = (dow == d).astype(float)
    # Day-of-month standardized to [-1,1]
    dom = feats.index.day.values.astype(float)
    feats["dom_std"] = (dom - 15.5) / 14.5

    # Intercept added in ridge solve
    return feats


# --------------------------- ridge ---------------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """w = (X'X + lam*I)^-1 X'y, with intercept column already in X."""
    n_feat = X.shape[1]
    I = np.eye(n_feat)
    I[0, 0] = 0.0  # don't penalize intercept
    A = X.T @ X + lam * I
    b = X.T @ y
    # solve
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w


# --------------------------- walk-forward ---------------------------

def walk_forward_ridge(
    panel: pd.DataFrame,
    feats: pd.DataFrame,
    sleeves: list[str],
    train_window: int = 252,
    refit_freq: str = "ME",
    lam: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """For each rebalance month-end, train ridge per sleeve on trailing
    train_window days, then predict the next-day return for each subsequent
    day until next rebalance. Returns predictions DataFrame (sleeves cols)
    and dict of coefficients per refit timestamp."""
    feat_cols = feats.columns.tolist()
    # add intercept column
    X_all = np.column_stack([np.ones(len(feats)), feats.values])
    preds = pd.DataFrame(np.nan, index=panel.index, columns=sleeves)
    coefs_history = {}  # sleeve -> list of (ts, coef vector)
    for s in sleeves:
        coefs_history[s] = []

    # Identify rebalance dates (month-ends within panel)
    month_ends = pd.date_range(panel.index[0].normalize(), panel.index[-1].normalize(),
                               freq=refit_freq, tz="UTC")

    # Targets: next-day returns; aligned so feats[t] predicts panel[t]
    # In our build_features, features are lagged (use up to t-1) so
    # predicting panel[t] (today's return) from feats[t] is causal.
    y_all = panel[sleeves].values  # rows aligned to feats

    # We need a sentinel that records, for each bar in panel, which model
    # (which refit timestamp) governs prediction on that bar.
    current_w = {s: None for s in sleeves}
    current_refit_ts = None

    # Pre-compute refit positions
    refit_positions = []
    for me in month_ends:
        pos = panel.index.searchsorted(me, side="right") - 1
        if pos < train_window + 30:  # need enough warmup
            continue
        refit_positions.append((pos, me))

    # Walk forward
    for k, (pos, me) in enumerate(refit_positions):
        # Train on rows [pos - train_window + 1, pos]
        start = pos - train_window + 1
        Xtr = X_all[start:pos + 1]
        Ytr = y_all[start:pos + 1]
        # Drop rows with any NaN in features
        ok = ~np.isnan(Xtr).any(axis=1) & ~np.isnan(Ytr).any(axis=1)
        if ok.sum() < 50:
            continue
        Xtr_c = Xtr[ok]
        Ytr_c = Ytr[ok]
        # Fit per sleeve
        for j, s in enumerate(sleeves):
            ytr = Ytr_c[:, j]
            w = ridge_fit(Xtr_c, ytr, lam)
            current_w[s] = w
            coefs_history[s].append({"ts": me, **{c: w[i + 1] for i, c in enumerate(feat_cols)},
                                     "intercept": w[0]})
        current_refit_ts = me

        # Predict from pos+1 up to the next refit position (exclusive)
        if k + 1 < len(refit_positions):
            next_pos = refit_positions[k + 1][0]
        else:
            next_pos = len(panel) - 1
        for t in range(pos + 1, next_pos + 1):
            xrow = X_all[t]
            if np.isnan(xrow).any():
                continue
            for j, s in enumerate(sleeves):
                preds.iat[t, j] = float(current_w[s] @ xrow)

    return preds, coefs_history


# --------------------------- weights ---------------------------

def softmax(x: np.ndarray, T: float) -> np.ndarray:
    """Stable softmax along last axis."""
    x = x / max(T, 1e-9)
    x = x - np.nanmax(x, axis=-1, keepdims=True)
    e = np.exp(x)
    e[np.isnan(e)] = 0.0
    s = e.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e / s


def ml_weights_from_preds(preds: pd.DataFrame, T: float) -> pd.DataFrame:
    """Long-only weights via softmax(clip(pred,0,inf)/T). NaN rows -> equal weights."""
    P = preds.values.copy()
    P = np.where(np.isnan(P), 0.0, P)
    P = np.clip(P, 0.0, None)
    # If all zero on a row, fall back to equal weight
    row_sums = P.sum(axis=1)
    W = softmax(P, T)
    all_zero = row_sums == 0
    W[all_zero] = 1.0 / preds.shape[1]
    # Also blank out rows where preds were originally all NaN -> equal weight
    blank = preds.isna().all(axis=1).values
    W[blank] = 1.0 / preds.shape[1]
    return pd.DataFrame(W, index=preds.index, columns=preds.columns)


def sharpe_tilted_weights(panel: pd.DataFrame, sleeves: list[str],
                          lookback: int = 252, T: float = 0.5) -> pd.DataFrame:
    """Rolling 12-month Sharpe per sleeve -> softmax to get weights.
    No ML, but adaptive."""
    r = panel[sleeves]
    mu = r.rolling(lookback).mean().shift(1)
    sd = r.rolling(lookback).std(ddof=0).shift(1)
    sh = (mu / sd.replace(0, np.nan)) * np.sqrt(252)
    sh = sh.fillna(0).clip(lower=0)  # long-only via clip
    W = softmax(sh.values, T)
    # Rows still NaN -> equal weight
    blank = r.rolling(lookback).mean().shift(1).isna().all(axis=1).values
    W[blank] = 1.0 / len(sleeves)
    return pd.DataFrame(W, index=panel.index, columns=sleeves)


# --------------------------- driver ---------------------------

def compose_portfolio(panel_gated: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio return = sum(weight_i * sleeve_i_gated).
    Weights are aligned to panel; weights at row t are decided pre-bar t."""
    common = panel_gated.columns.intersection(weights.columns)
    return (panel_gated[common] * weights[common]).sum(axis=1)


def yearly_sharpe(r: pd.Series) -> dict[int, float]:
    out = {}
    for y, sub in r.groupby(r.index.year):
        if len(sub) < 50:
            continue
        bpy = _bpy(sub.index)
        sd = sub.std(ddof=0)
        out[int(y)] = float(sub.mean() * bpy / (sd * np.sqrt(bpy))) if sd > 0 else 0.0
    return out


def main():
    print("Loading panel...")
    panel = load_panel()
    panel = panel[TOP9]  # use TOP9 = TOP8 + PAIRS_EXP

    print("Building features (walk-forward, may take ~20s)...")
    feats = build_features(panel, TOP9)
    print(f"  feature matrix: {feats.shape}, cols: {feats.columns.tolist()}")

    print("Walk-forward ridge fits (per sleeve, monthly refit, 252d train)...")
    preds, coefs_hist = walk_forward_ridge(panel, feats, TOP9,
                                           train_window=252,
                                           refit_freq="ME",
                                           lam=10.0)
    print(f"  prediction matrix: {preds.notna().sum().sum()} non-NaN cells "
          f"of {preds.size} total")

    # Regime gate for the underlying sleeves
    print("Applying regime gate + decay tripwire to sleeve returns...")
    high_vol, _ = build_regime_mask(panel.index, 80)
    gates = dict(GATES_HIGH)
    gates["PAIRS_EXP"] = 1.5
    gated = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in gated.columns:
            gated.loc[high_vol, sleeve] *= mult
    after_decay, _ = fast_decay_tripwire(gated, TOP9)

    # Build a few ML variants
    ml_variants = {}
    for T in (0.005, 0.02):
        W_ml = ml_weights_from_preds(preds, T=T)
        ml_variants[f"ML_T{T}"] = W_ml

    # Baselines
    eq_w = pd.DataFrame(1.0 / len(TOP9),
                        index=panel.index, columns=TOP9)
    sh_w = sharpe_tilted_weights(panel, TOP9, lookback=252, T=0.5)

    weight_dict = {
        "EQUAL_WEIGHT": eq_w,
        "STATIC_TOP9": eq_w,  # same as equal weight on TOP9
        "SHARPE_TILT": sh_w,
        **ml_variants,
    }

    # Compose returns -> apply vol-target -> apply DD control
    SPLIT_TS = SPLIT
    print(f"\n{'Variant':<22} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
    print("-" * 70)
    results = {}
    rows = []
    yearly_rows = {}
    for name, W in weight_dict.items():
        r_raw = compose_portfolio(after_decay, W)
        vt, _ = vol_target_overlay(r_raw, target_vol=0.10)
        final, _ = drawdown_control(vt)
        results[name] = final
        s = stats(name, final)
        y22 = final[final.index.year == 2022]
        sh22 = 0.0
        if len(y22) > 1 and y22.std() > 0:
            bpy = _bpy(y22.index)
            sh22 = float(y22.mean() * bpy / (y22.std(ddof=0) * np.sqrt(bpy)))
        rows.append({"variant": name, **s, "Sh2022": sh22})
        yearly_rows[name] = yearly_sharpe(final)
        print(f"{name:<22} {s.get('FULL_sharpe',0):>+6.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+6.2f} {sh22:>+7.2f} "
              f"{s.get('OOS_vol',0):>8.1%} {s.get('OOS_dd',0):>+7.1%}")

    pd.DataFrame(rows).to_csv(OUT / "ml_meta_variants.csv", index=False)
    print("\nYear-by-year Sharpe:")
    yr_df = pd.DataFrame(yearly_rows).T
    print(yr_df.round(2).to_string())
    yr_df.to_csv(OUT / "ml_meta_yearly.csv")

    # Save the headline ML variant returns (use T=0.02 — milder, more robust)
    chosen = "ML_T0.02"
    final_r = results[chosen]
    eq = (1 + final_r).cumprod()
    pd.DataFrame({"timestamp": final_r.index, "ret": final_r.values,
                  "equity": eq.values}).to_parquet(
        OUT / "ml_meta_returns.parquet", index=False)
    print(f"\nSaved sleeve returns: {OUT / 'ml_meta_returns.parquet'}")

    # Save weights CSV: take month-end snapshots (one row per rebalance day)
    W_ml = weight_dict[chosen]
    me_idx = pd.date_range(panel.index[0].normalize(), panel.index[-1].normalize(),
                           freq="ME", tz="UTC")
    me_positions = []
    for me in me_idx:
        pos = panel.index.searchsorted(me, side="right") - 1
        if 0 <= pos < len(panel):
            me_positions.append(pos)
    W_snap = W_ml.iloc[me_positions]
    W_snap.to_csv(OUT / "ml_meta_weights.csv")
    print(f"Saved per-rebalance weights: {OUT / 'ml_meta_weights.csv'}")

    # Feature importance: average absolute coefficient across refits across sleeves
    print("\nFeature importance (mean |coef| across sleeves x refits):")
    feat_cols = feats.columns.tolist()
    all_records = []
    for s, lst in coefs_hist.items():
        for rec in lst:
            for c in feat_cols:
                all_records.append({"sleeve": s, "feat": c, "coef": rec[c]})
    cdf = pd.DataFrame(all_records)
    imp = cdf.groupby("feat")["coef"].apply(lambda x: float(np.mean(np.abs(x)))).sort_values(
        ascending=False)
    print(imp.to_string())
    imp.to_csv(OUT / "ml_meta_feature_importance.csv")
    # Save per-sleeve coefs too
    cdf.to_csv(OUT / "ml_meta_coef_history.csv", index=False)

    print("\nDONE.")


if __name__ == "__main__":
    main()
