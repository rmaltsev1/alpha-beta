"""Wave 6 — Non-linear ML meta-signal via gradient-boosted decision stumps.

Prior linear ML failed. Here we try richer features + non-linearity via
depth-1 trees (stumps) trained sequentially with gradient boosting.
Pure numpy implementation (no scikit-learn).

Pipeline:
  1. Build rich feature set from D1 prices (SPX/NAS/BTC/EUR_USD/USD_JPY) and
     from the TOP21 base portfolio's own history.
  2. Target = next-day TOP21 base return.
  3. Fit a gradient-boosted stump ensemble on a rolling window, retrain monthly,
     predict OOS daily.
  4. Walk-forward — only past data used for features and fit.
  5. Strategy variants:
       - HALVE (defensive): if pred < 0, halve position.
       - DOUBLE (aggressive): if pred > 0, double position.
       - SCALED (continuous): scale by tanh(pred / scale).
  6. Apply existing regime gates + decay tripwire on the underlying panel.
  7. Vol-target to 10% ann vol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

from master_v4 import (  # noqa: E402
    fast_decay_tripwire, build_regime_mask, drawdown_control,
    stats, _bpy, GATES_HIGH, vol_target_overlay,
)

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.10
SEED = 7

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_d1_close(symbol: str) -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / symbol / "D1.parquet")
    s = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamp"], utc=True))
    s.index = s.index.floor("D")
    s = s.groupby(s.index).last()
    return s.sort_index()


def daily_log_ret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(panel_index: pd.DatetimeIndex,
                   top21_base: pd.Series) -> pd.DataFrame:
    """Walk-forward features computed on D1 prices.

    All features at time t use only data <= t (shift(1) on raw inputs is not
    necessary because we're predicting t+1 from features at t).
    """
    spx = load_d1_close("SPX500_USD")
    nas = load_d1_close("NAS100_USD")
    btc = load_d1_close("BTCUSDT")
    eur = load_d1_close("EUR_USD")
    jpy = load_d1_close("USD_JPY")

    spx_r = daily_log_ret(spx)
    nas_r = daily_log_ret(nas)
    btc_r = daily_log_ret(btc)
    eur_r = daily_log_ret(eur)
    jpy_r = daily_log_ret(jpy)

    feats = {}

    # --- SPX returns ---
    feats["spx_r5"] = spx_r.rolling(5).sum()
    feats["spx_r30"] = spx_r.rolling(30).sum()
    feats["spx_r90"] = spx_r.rolling(90).sum()

    # --- SPX vol & vol-of-vol ---
    spx_rv30 = spx_r.rolling(30).std(ddof=0)
    feats["spx_rv30"] = spx_rv30
    feats["spx_vov"] = spx_rv30.rolling(30).std(ddof=0)

    # --- BTC ---
    feats["btc_r30"] = btc_r.rolling(30).sum()
    feats["btc_rv30"] = btc_r.rolling(30).std(ddof=0)

    # --- FX vol ---
    feats["eur_rv60"] = eur_r.rolling(60).std(ddof=0)
    feats["jpy_rv60"] = jpy_r.rolling(60).std(ddof=0)

    # --- VIX-proxy quartile (regime indicator from SPX RV30) ---
    # IS quartile cutoffs computed from data <= 2024-01-01 (no look-ahead).
    is_rv = spx_rv30[spx_rv30.index < SPLIT].dropna()
    q25 = float(is_rv.quantile(0.25))
    q75 = float(is_rv.quantile(0.75))
    rv = spx_rv30
    quart = pd.Series(1, index=rv.index, dtype=float)  # middle
    quart[rv <= q25] = 0
    quart[rv >= q75] = 2
    feats["vix_quartile"] = quart

    # --- Calendar features ---
    cal_idx = panel_index
    feats_cal_dow = pd.Series(cal_idx.dayofweek.values, index=cal_idx, dtype=float)
    feats_cal_dom = pd.Series(cal_idx.day.values, index=cal_idx, dtype=float)
    feats_cal_month = pd.Series(cal_idx.month.values, index=cal_idx, dtype=float)

    # --- Trailing TOP21 30d Sharpe ---
    top21_ret = top21_base
    mu30 = top21_ret.rolling(30).mean()
    sd30 = top21_ret.rolling(30).std(ddof=0)
    feats["top21_sharpe30"] = (mu30 / sd30.replace(0, np.nan)) * np.sqrt(365.25)

    # --- Cross-asset correlation regime (avg 60d corr across pairs) ---
    rets_df = pd.concat({
        "spx": spx_r, "nas": nas_r, "btc": btc_r, "eur": eur_r, "jpy": jpy_r,
    }, axis=1).dropna(how="all")
    # Avg pairwise rolling 60d corr
    pairs = [("spx","nas"),("spx","btc"),("spx","eur"),("spx","jpy"),
             ("btc","eur"),("btc","jpy"),("eur","jpy"),("nas","btc")]
    corr_list = []
    for a, b in pairs:
        c = rets_df[a].rolling(60).corr(rets_df[b])
        corr_list.append(c)
    feats["xcorr60"] = pd.concat(corr_list, axis=1).mean(axis=1)

    # --- Sign-vol interactions ---
    feats["spx_signvol"] = np.sign(feats["spx_r30"]) * feats["spx_rv30"]
    feats["btc_signvol"] = np.sign(feats["btc_r30"]) * feats["btc_rv30"]
    feats["spx_vol_xcorr"] = feats["spx_rv30"] * feats["xcorr60"]

    # --- Assemble, align to panel index ---
    feat_df = pd.DataFrame(feats)
    # Forward-fill price-based features to calendar (panel may have weekends)
    feat_df = feat_df.reindex(panel_index, method="ffill")

    # Calendar features come directly from panel index
    feat_df["dow"] = feats_cal_dow
    feat_df["dom"] = feats_cal_dom
    feat_df["month"] = feats_cal_month

    # Replace inf with NaN, then forward fill, then 0
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.ffill().fillna(0.0)
    return feat_df


# =============================================================================
# GRADIENT BOOSTED STUMPS (numpy only)
# =============================================================================

class GBStumps:
    """Gradient-boosted depth-1 decision trees (stumps).

    Squared-error loss. Each stump picks the best feature j and threshold t,
    producing 2 leaf values L (x[j] <= t) and R (x[j] > t).

    Speed: per fit O(n_iter * n_features * n_samples * log n_samples) — fine
    for ~1000 obs and ~20 features.
    """

    def __init__(self, n_iter: int = 50, lr: float = 0.05,
                 n_thresh: int = 12, rng: np.random.Generator | None = None):
        self.n_iter = n_iter
        self.lr = lr
        self.n_thresh = n_thresh
        self.stumps: list[tuple[int, float, float, float]] = []
        self.base = 0.0
        self.rng = rng if rng is not None else np.random.default_rng(SEED)

    def _best_stump(self, X: np.ndarray, g: np.ndarray) -> tuple[int, float, float, float, float]:
        """Find the stump that minimises sum (g - pred)^2.
        Returns (feature_idx, threshold, left_val, right_val, sse).
        """
        n, p = X.shape
        best = (-1, 0.0, 0.0, 0.0, np.inf)
        for j in range(p):
            col = X[:, j]
            # Threshold candidates from quantiles to keep fast
            qs = np.quantile(col, np.linspace(0.1, 0.9, self.n_thresh))
            qs = np.unique(qs)
            for t in qs:
                mask = col <= t
                nL = mask.sum()
                nR = n - nL
                if nL < 5 or nR < 5:
                    continue
                gL = g[mask]
                gR = g[~mask]
                muL = gL.mean()
                muR = gR.mean()
                sse = ((gL - muL) ** 2).sum() + ((gR - muR) ** 2).sum()
                if sse < best[4]:
                    best = (j, float(t), float(muL), float(muR), float(sse))
        return best

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.stumps = []
        self.base = float(np.mean(y))
        pred = np.full_like(y, self.base, dtype=float)
        for _ in range(self.n_iter):
            resid = y - pred
            j, t, lv, rv_, _sse = self._best_stump(X, resid)
            if j < 0:
                break
            self.stumps.append((j, t, lv, rv_))
            col = X[:, j]
            pred = pred + self.lr * np.where(col <= t, lv, rv_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(X.shape[0], self.base, dtype=float)
        for (j, t, lv, rv_) in self.stumps:
            col = X[:, j]
            pred = pred + self.lr * np.where(col <= t, lv, rv_)
        return pred

    def feature_importance(self, n_features: int) -> np.ndarray:
        imp = np.zeros(n_features)
        for (j, _t, lv, rv_) in self.stumps:
            # Use squared output range as a crude importance.
            imp[j] += (rv_ - lv) ** 2
        if imp.sum() > 0:
            imp = imp / imp.sum()
        return imp


# =============================================================================
# WALK-FORWARD ML
# =============================================================================

def walk_forward_predict(features: pd.DataFrame, target: pd.Series,
                         min_train: int = 252, retrain_freq: str = "ME",
                         n_iter: int = 50, lr: float = 0.05
                         ) -> tuple[pd.Series, np.ndarray, list[str]]:
    """Predict target[t+1] at every t using a GBStump model retrained monthly.

    - At each retrain date, fit on all data through that date.
    - Use that model for all bars until the next retrain.
    """
    idx = features.index
    X_full = features.values.astype(float)
    y_full = target.values.astype(float)

    preds = np.full(len(idx), np.nan)
    feat_names = list(features.columns)
    n_feat = len(feat_names)
    cum_importance = np.zeros(n_feat)
    n_fits = 0

    retrain_dates = pd.date_range(idx[0], idx[-1], freq=retrain_freq, tz="UTC")
    retrain_dates = [d for d in retrain_dates if d >= idx[0]]

    last_model = None
    last_retrain_pos = -1

    rng = np.random.default_rng(SEED)

    for k, rd in enumerate(retrain_dates):
        # Position of last bar with data <= rd. We use data through index pos
        # (inclusive) for training. Predictions for bars > pos use this model
        # until the next retrain date.
        pos = idx.searchsorted(rd, side="right") - 1
        if pos < min_train:
            continue
        # Training data: rows 0..pos with valid target (target = next-day ret,
        # so row i's target uses panel return at i+1 — we must drop row pos if
        # it has no next-day target, but since target is precomputed, just use
        # rows with non-NaN).
        X_tr = X_full[: pos + 1]
        y_tr = y_full[: pos + 1]
        valid = ~np.isnan(y_tr) & np.all(~np.isnan(X_tr), axis=1)
        X_tr = X_tr[valid]
        y_tr = y_tr[valid]
        if len(X_tr) < min_train:
            continue
        model = GBStumps(n_iter=n_iter, lr=lr, rng=rng)
        model.fit(X_tr, y_tr)
        # Predict from (pos+1) to next retrain date (exclusive).
        if k + 1 < len(retrain_dates):
            nxt_pos = idx.searchsorted(retrain_dates[k + 1], side="right") - 1
        else:
            nxt_pos = len(idx) - 1
        # Use this model for bars (pos+1) .. nxt_pos (inclusive)
        if nxt_pos > pos:
            X_pred = X_full[pos + 1 : nxt_pos + 1]
            preds[pos + 1 : nxt_pos + 1] = model.predict(X_pred)
        last_model = model
        last_retrain_pos = pos
        cum_importance += model.feature_importance(n_feat)
        n_fits += 1

    if n_fits > 0:
        cum_importance /= n_fits

    return pd.Series(preds, index=idx, name="pred"), cum_importance, feat_names


# =============================================================================
# STRATEGY VARIANTS
# =============================================================================

def apply_strategy(base: pd.Series, pred: pd.Series, mode: str) -> pd.Series:
    """Apply ML signal to base returns. Signal at t scales return at t+1
    (executable). We assume base index == pred index; pred[t] is the prediction
    made using info up to t for the t->t+1 return. So weight at t+1 = f(pred[t]).

    Because the base portfolio has a strong positive drift, the raw GB
    predictions are almost never negative. We therefore convert "neg/pos"
    semantics into "below/above the trailing predictive mean" — i.e., the
    model thinks tomorrow will be worse than average vs. better than average.
    The trailing mean uses an IS-anchored expanding window with a 252-day
    cap, walk-forward.
    """
    pred_lag = pred.shift(1)
    # Walk-forward rolling mean of pred — only past pred values.
    roll_mu = pred_lag.expanding(min_periods=63).mean()
    # Cap influence: use rolling 252-day mean once enough history.
    roll_mu_252 = pred_lag.rolling(252, min_periods=63).mean()
    ref = roll_mu_252.fillna(roll_mu)
    delta = pred_lag - ref  # negative = below-average prediction

    w = pd.Series(1.0, index=base.index)
    if mode == "HALVE":
        w[delta < 0] = 0.5
    elif mode == "DOUBLE":
        w[delta > 0] = 2.0
    elif mode == "BOTH":
        w[delta < 0] = 0.5
        w[delta > 0] = 1.5
    elif mode == "SCALED":
        # Standardise delta on its trailing 252d std, weight = 1 + tanh(z/2)
        sd = pred_lag.rolling(252, min_periods=63).std(ddof=0)
        z = (delta / sd.replace(0, np.nan)).fillna(0)
        w = (1.0 + np.tanh(z / 2.0)).clip(lower=0.0, upper=2.0)
    elif mode == "BASE":
        pass
    else:
        raise ValueError(mode)
    return base * w.fillna(1.0)


# =============================================================================
# REPORTING UTILS
# =============================================================================

def yearly_sharpe(r: pd.Series, year: int) -> float:
    sub = r[r.index.year == year].dropna()
    if len(sub) < 30 or sub.std(ddof=0) == 0:
        return 0.0
    bpy = _bpy(sub.index)
    return float(sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)))


# =============================================================================
# MAIN
# =============================================================================

def main():
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v13.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    print(f"Panel: {panel.shape[0]} bars, {panel.shape[1]} sleeves")

    # ---- Define TOP21 (from master_v13) ----
    TOP14 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX", "CORR_REGIME", "SESSION_MOM"]
    TOP16 = TOP14 + ["W1_STRATS", "EVENT_VOLSPIKE"]
    TOP21 = TOP16 + ["STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT", "TERM_SPREADS", "EURGBP_MR"]

    missing = [s for s in TOP21 if s not in panel.columns]
    if missing:
        print(f"WARN: missing sleeves: {missing}")
    TOP21 = [s for s in TOP21 if s in panel.columns]

    # ---- Apply regime gates + decay tripwire to underlying sleeves ----
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS", "MICROSTR_D1",
              "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    for s in ["TREND_NEW", "TREND_COND"]:
        gates[s] = 0.5
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    after_decay, _ = fast_decay_tripwire(g[TOP21], TOP21)
    base = after_decay.mean(axis=1).rename("base")
    print(f"Base TOP21 portfolio: {len(base)} bars, "
          f"mean={base.mean():.5f}, std={base.std(ddof=0):.5f}")

    # ---- Build features ----
    feats = build_features(base.index, base)
    print(f"Features: {feats.shape}, cols: {list(feats.columns)}")

    # ---- Target: next-day base return ----
    target = base.shift(-1)  # at time t, target = base[t+1]

    # ---- Walk-forward predictions ----
    pred, importance, feat_names = walk_forward_predict(
        feats, target, min_train=252, retrain_freq="ME",
        n_iter=50, lr=0.05,
    )
    print(f"OOS predictions: {(~pred.isna()).sum()} non-NaN bars")

    # ---- Save feature importance ----
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(OUT / "ml_features.csv", index=False)
    print(f"\nTop features by importance:")
    print(fi_df.head(10).to_string(index=False))

    # ---- Strategy variants ----
    variants = {}
    for mode in ["BASE", "HALVE", "DOUBLE", "BOTH", "SCALED"]:
        r = apply_strategy(base, pred, mode)
        variants[mode] = r

    # ---- Vol-target each variant to 10% ----
    vt_variants = {}
    for mode, r in variants.items():
        vt, _ = vol_target_overlay(r, target_vol=TARGET_VOL, lookback=60, max_lev=5.0)
        dd, _ = drawdown_control(vt)
        vt_variants[mode] = dd

    # ---- Stats ----
    print(f"\n{'Variant':<10} {'FULL_Sh':>8} {'IS_Sh':>8} {'OOS_Sh':>8} {'2022_Sh':>9}")
    print("-" * 50)
    rows = []
    for mode, r in vt_variants.items():
        s = stats(mode, r)
        sh22 = yearly_sharpe(r, 2022)
        print(f"{mode:<10} {s.get('FULL_sharpe',0):>+8.2f} "
              f"{s.get('IS_sharpe',0):>+8.2f} {s.get('OOS_sharpe',0):>+8.2f} "
              f"{sh22:>+9.2f}")
        rows.append({"variant": mode,
                     "FULL_sharpe": s.get("FULL_sharpe", 0.0),
                     "IS_sharpe": s.get("IS_sharpe", 0.0),
                     "OOS_sharpe": s.get("OOS_sharpe", 0.0),
                     "y2022_sharpe": sh22,
                     "FULL_ret": s.get("FULL_ret", 0.0),
                     "OOS_ret": s.get("OOS_ret", 0.0),
                     "OOS_dd": s.get("OOS_dd", 0.0)})

    summary = pd.DataFrame(rows)
    summary_path = OUT / "ml_nonlinear_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")

    # ---- Best variant returns parquet ----
    # Use the strongest OOS-Sharpe variant excluding BASE
    non_base = summary[summary["variant"] != "BASE"]
    if len(non_base) > 0:
        best_mode = non_base.loc[non_base["OOS_sharpe"].idxmax(), "variant"]
    else:
        best_mode = "HALVE"
    best_r = vt_variants[best_mode]
    eq = (1 + best_r.fillna(0)).cumprod()
    out_df = pd.DataFrame({
        "timestamp": best_r.index,
        "ret": best_r.values,
        "equity": eq.values,
        "pred": pred.reindex(best_r.index).values,
    })
    # Also stash every variant's daily ret for downstream analysis
    for mode, r in vt_variants.items():
        out_df[f"ret_{mode}"] = r.reindex(best_r.index).values

    out_path = OUT / "ml_nonlinear_returns.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Best variant: {best_mode} -> {out_path}")

    # ---- Year-by-year for headline variant ----
    print(f"\nYear-by-year ({best_mode}, vol-targeted):")
    for year, sub in best_r.groupby(best_r.index.year):
        if len(sub) < 30:
            continue
        bpy = _bpy(sub.index)
        ar = sub.mean() * bpy
        av = sub.std(ddof=0) * np.sqrt(bpy)
        sh = ar / av if av > 0 else 0
        eq_y = (1 + sub).cumprod()
        dd = (eq_y / eq_y.cummax() - 1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  DD={dd:+.1%}")

    # ---- Baseline TOP21 vol-targeted for direct comparison ----
    print(f"\nBaseline (TOP21 no-ML, vol-targeted to {TARGET_VOL:.0%}):")
    base_vt, _ = vol_target_overlay(base, target_vol=TARGET_VOL, lookback=60, max_lev=5.0)
    base_dd, _ = drawdown_control(base_vt)
    sb = stats("BASELINE", base_dd)
    sh22b = yearly_sharpe(base_dd, 2022)
    print(f"  FULL_Sh={sb.get('FULL_sharpe',0):+.2f} "
          f"IS_Sh={sb.get('IS_sharpe',0):+.2f} "
          f"OOS_Sh={sb.get('OOS_sharpe',0):+.2f} "
          f"2022_Sh={sh22b:+.2f}")


if __name__ == "__main__":
    main()
