"""Adaptive sleeve-weighting layer (wave5).

Goal: improve on the v11 equal-weight TOP14 portfolio (OOS Sharpe +2.82) by
tilting toward sleeves with strong recent risk-adjusted performance, but in a
*careful* way that doesn't repeat the Kelly / sharpe-tilt failures.

Six variants are evaluated, all walk-forward, monthly rebalance:

  1. Shrinkage-toward-equal-weight Sharpe tilt.
     w_i = alpha * sharpe_tilt_i + (1 - alpha) * (1/N). Sweep alpha in
     {0.1, 0.2, 0.3, 0.5}.

  2. Bucket-EW + within-bucket Sharpe tilt.
     5 buckets (calendar, mean-rev, trend, beta, pairs). Across buckets
     equal-weight; within bucket Sharpe-tilt (positive part, mild).

  3. Inverse-vol within bucket.
     Same buckets, inverse-vol within. Across buckets equal-weight.

  4. Risk-budget allocation.
     Target risk contribution per bucket: calendar 20%, MR 20%, trend 25%,
     beta 20%, pairs 15%. Solve so each bucket contributes the target sigma
     to the portfolio. Within bucket equal-weight (vols already similar).

  5. Rolling-window Sharpe with momentum filter.
     Tilt toward sleeves with BOTH trailing-3m and trailing-12m positive
     Sharpe. Skip (zero out) sleeves with trailing-3m < 0 — same as the
     decay tripwire signal but applied as a *weight* zero, not a state.

  6. Bayesian shrinkage of trailing Sharpe.
     Prior = mean trailing Sharpe across all sleeves. Posterior =
     (n*sample + k*prior) / (n + k). Use posterior for tilt.

The same overlay stack as v11 is applied to every variant:
  - high-vol regime gates (GATES_HIGH + v11 overrides)
  - fast decay tripwire (63d trailing-Sharpe state, 2 consec checks)
  - vol-target overlay (18% target, max 15x)
  - drawdown control (halve at 3% trailing 30d DD, recover at 1%)

The weighted combine happens AFTER regime-gate + tripwire (so the weights act
on the gated/tripwired contributions). This keeps the comparison fair to EW.

Output:
  scratch/wave5/adaptive_weighting_returns.parquet     (best variant daily)
  scratch/wave5/adaptive_weighting_comparison.csv      (all variants vs EW)
  scratch/wave5/adaptive_weighting_weights.csv         (monthly weight log)
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
    build_regime_mask, _bpy, GATES_HIGH,
)

OUT = ROOT / "scratch" / "wave5"
PANEL_PATH = ROOT / "scratch" / "quant" / "all_sleeve_returns_v11.parquet"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
VOL_TARGET = 0.18
MAX_LEV = 15.0

TOP14 = ["VOLFORECAST", "EVE_XAU", "RISKPAR", "XSMOM", "CORR_REGIME",
         "TREND_NEW", "WED_BTC", "SESSION_MOM", "D1REV_NAS", "PAIRS_EXP",
         "DEFEND", "H4_SLEEVE", "D1REV_UK", "CRYPTO_vs_SPX"]

# Bucket assignment for the 14 production sleeves.
BUCKETS = {
    "calendar": ["EVE_XAU", "WED_BTC", "SESSION_MOM"],
    "mean_rev": ["D1REV_NAS", "D1REV_UK"],
    "trend":    ["TREND_NEW", "XSMOM", "H4_SLEEVE"],
    "beta":     ["RISKPAR", "DEFEND", "VOLFORECAST"],
    "pairs":    ["PAIRS_EXP", "CORR_REGIME", "CRYPTO_vs_SPX"],
}
# Risk-budget (variant 4)
RISK_BUDGET = {"calendar": 0.20, "mean_rev": 0.20, "trend": 0.25,
               "beta": 0.20, "pairs": 0.15}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stats_block(r: pd.Series) -> dict:
    out = {}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2:
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    # 2022 sub
    y22 = r[r.index.year == 2022]
    if len(y22) > 10:
        bpy22 = _bpy(y22.index)
        ar22 = float(y22.mean()) * bpy22
        av22 = float(y22.std(ddof=0)) * np.sqrt(bpy22)
        out["YR2022_sharpe"] = ar22/av22 if av22 > 0 else 0.0
        eq22 = (1 + y22).cumprod()
        out["YR2022_dd"] = float((eq22 / eq22.cummax() - 1).min())
    return out


def trailing_sharpe(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling Sharpe per column (uses past window bars)."""
    mu = panel.rolling(window).mean()
    sd = panel.rolling(window).std(ddof=0)
    sh = (mu * 365.25) / (sd * np.sqrt(365.25))
    return sh.replace([np.inf, -np.inf], np.nan)


def trailing_vol(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    return panel.rolling(window).std(ddof=0) * np.sqrt(365.25)


def month_end_index_positions(idx: pd.DatetimeIndex) -> list[int]:
    """Return integer positions corresponding to last bar of each month."""
    if len(idx) == 0:
        return []
    s = pd.Series(np.arange(len(idx)), index=idx)
    by_month = s.groupby([idx.year, idx.month]).last()
    return sorted(int(v) for v in by_month.tolist())


# ---------------------------------------------------------------------------
# Weight builders. Each takes the gated+tripwired panel (sleeve cols, daily
# rows) and returns a same-shape DataFrame of weights. Weights at row t apply
# to the return at row t (we build them so they look only at past bars).
# ---------------------------------------------------------------------------
def _stepwise_weights(panel: pd.DataFrame, weight_at_monthend_fn) -> pd.DataFrame:
    """Build a weights DataFrame by calling weight_at_monthend_fn(month_end_pos)
    at each month-end and broadcasting that vector to the NEXT month's rows.
    Bars before the first usable month-end get equal weights."""
    n_sleeves = panel.shape[1]
    sleeves = list(panel.columns)
    W = pd.DataFrame(1.0 / n_sleeves, index=panel.index, columns=sleeves)
    me_pos = month_end_index_positions(panel.index)
    last_w = np.full(n_sleeves, 1.0 / n_sleeves)
    cursor = 0  # we'll fill rows [cursor : next_me_pos] with last_w
    for pos in me_pos:
        # fill rows from cursor up to AND INCLUDING pos with last_w
        W.iloc[cursor:pos + 1, :] = last_w
        # compute next-period weights using data through pos
        nw = weight_at_monthend_fn(pos)
        if nw is not None:
            last_w = nw
        cursor = pos + 1
    # tail
    W.iloc[cursor:, :] = last_w
    return W


def _normalize_pos(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s <= 1e-12:
        return np.full_like(v, 1.0 / len(v))
    return v / s


def _sharpe_tilt_from_trailing(sh: np.ndarray) -> np.ndarray:
    """Convert trailing Sharpes into a tilt vector via clip-to-positive and
    normalize. Bounded, no Kelly-style blow-ups."""
    n = len(sh)
    if not np.isfinite(sh).any():
        return np.full(n, 1.0 / n)
    sh = np.where(np.isfinite(sh), sh, 0.0)
    pos = np.clip(sh, 0.0, None)
    if pos.sum() <= 1e-12:
        return np.full(n, 1.0 / n)
    return pos / pos.sum()


# ----- Variant 1: shrinkage-toward-EW Sharpe tilt -----------------------------
def make_shrinkage_weights(panel: pd.DataFrame, alpha: float,
                           lookback: int = 126) -> pd.DataFrame:
    sleeves = list(panel.columns)
    n = len(sleeves)
    sh = trailing_sharpe(panel, lookback)
    ew = np.full(n, 1.0 / n)

    def w_at(pos: int) -> np.ndarray | None:
        if pos < lookback:
            return None
        s = sh.iloc[pos].values
        tilt = _sharpe_tilt_from_trailing(s)
        return alpha * tilt + (1 - alpha) * ew

    return _stepwise_weights(panel, w_at)


# ----- Variant 2: bucket EW across, sharpe-tilt within ------------------------
def make_bucket_sharpe_tilt_weights(panel: pd.DataFrame,
                                    lookback: int = 126,
                                    within_alpha: float = 0.5) -> pd.DataFrame:
    sleeves = list(panel.columns)
    col_idx = {s: i for i, s in enumerate(sleeves)}
    sh = trailing_sharpe(panel, lookback)
    n_buckets = len(BUCKETS)

    def w_at(pos: int) -> np.ndarray | None:
        if pos < lookback:
            return None
        w = np.zeros(len(sleeves))
        bucket_share = 1.0 / n_buckets
        for _, members in BUCKETS.items():
            members = [m for m in members if m in col_idx]
            if not members:
                continue
            s_vec = np.array([sh.iloc[pos][m] for m in members])
            tilt = _sharpe_tilt_from_trailing(s_vec)
            ew = np.full(len(members), 1.0 / len(members))
            blended = within_alpha * tilt + (1 - within_alpha) * ew
            blended = blended / blended.sum()
            for w_member, member in zip(blended, members):
                w[col_idx[member]] = bucket_share * w_member
        return w

    return _stepwise_weights(panel, w_at)


# ----- Variant 3: inverse-vol within bucket -----------------------------------
def make_bucket_invvol_weights(panel: pd.DataFrame,
                               lookback: int = 63) -> pd.DataFrame:
    sleeves = list(panel.columns)
    col_idx = {s: i for i, s in enumerate(sleeves)}
    vol = trailing_vol(panel, lookback)
    n_buckets = len(BUCKETS)

    def w_at(pos: int) -> np.ndarray | None:
        if pos < lookback:
            return None
        w = np.zeros(len(sleeves))
        bucket_share = 1.0 / n_buckets
        for _, members in BUCKETS.items():
            members = [m for m in members if m in col_idx]
            if not members:
                continue
            v_vec = np.array([vol.iloc[pos][m] for m in members])
            v_vec = np.where((v_vec > 1e-9) & np.isfinite(v_vec), v_vec, np.nan)
            if np.isnan(v_vec).all():
                inv = np.full(len(members), 1.0 / len(members))
            else:
                inv = np.where(np.isfinite(v_vec), 1.0 / v_vec, 0.0)
                s = inv.sum()
                inv = inv / s if s > 1e-12 else np.full(len(members), 1.0 / len(members))
            for w_member, member in zip(inv, members):
                w[col_idx[member]] = bucket_share * w_member
        return w

    return _stepwise_weights(panel, w_at)


# ----- Variant 4: risk-budget across buckets ----------------------------------
def make_risk_budget_weights(panel: pd.DataFrame,
                             lookback: int = 63) -> pd.DataFrame:
    """Allocate so each bucket contributes RISK_BUDGET[b] of the *standalone*
    portfolio vol (ignoring cross-bucket correlations, since EW already has
    decent diversification). Within bucket: inverse-vol (so each sleeve in
    the bucket contributes equal risk to the bucket)."""
    sleeves = list(panel.columns)
    col_idx = {s: i for i, s in enumerate(sleeves)}
    vol = trailing_vol(panel, lookback)

    def w_at(pos: int) -> np.ndarray | None:
        if pos < lookback:
            return None
        w = np.zeros(len(sleeves))
        # Compute each bucket's "risk per unit weight" (its vol when members
        # are inverse-vol blended). Then scale buckets so contributions match
        # the risk budget.
        bucket_w = {}
        bucket_internal_vol = {}
        for b, members in BUCKETS.items():
            members = [m for m in members if m in col_idx]
            if not members:
                continue
            v_vec = np.array([vol.iloc[pos][m] for m in members])
            v_vec = np.where((v_vec > 1e-9) & np.isfinite(v_vec), v_vec, np.nan)
            if np.isnan(v_vec).all():
                inv = np.full(len(members), 1.0 / len(members))
                bv = float(np.nanmean(v_vec)) if np.isfinite(v_vec).any() else 0.05
            else:
                inv = np.where(np.isfinite(v_vec), 1.0 / v_vec, 0.0)
                s = inv.sum()
                inv = inv / s if s > 1e-12 else np.full(len(members), 1.0 / len(members))
                # Internal bucket vol if uncorrelated: sqrt(sum w_i^2 * v_i^2)
                v_eff = np.where(np.isfinite(v_vec), v_vec, np.nanmean(v_vec))
                bv = float(np.sqrt(np.sum((inv * v_eff) ** 2)))
                bv = max(bv, 1e-6)
            bucket_w[b] = (members, inv)
            bucket_internal_vol[b] = bv

        # Each bucket should contribute risk_target_b to total portfolio vol.
        # Bucket contribution = scale_b * bucket_internal_vol[b]. We don't need
        # the total to be any specific number (vol-target overlay will scale),
        # so just set scale_b proportional to RISK_BUDGET[b] / bucket_internal_vol[b].
        scales = {}
        for b in bucket_w:
            scales[b] = RISK_BUDGET[b] / bucket_internal_vol[b]
        total = sum(scales.values())
        if total <= 1e-12:
            return np.full(len(sleeves), 1.0 / len(sleeves))
        for b in bucket_w:
            members, inv = bucket_w[b]
            bucket_total = scales[b] / total
            for w_m, m in zip(inv, members):
                w[col_idx[m]] = bucket_total * w_m
        return w

    return _stepwise_weights(panel, w_at)


# ----- Variant 5: 3m+12m positive Sharpe filter ------------------------------
def make_momfilter_weights(panel: pd.DataFrame,
                           short: int = 63, long: int = 252) -> pd.DataFrame:
    sleeves = list(panel.columns)
    sh_s = trailing_sharpe(panel, short)
    sh_l = trailing_sharpe(panel, long)
    n = len(sleeves)

    def w_at(pos: int) -> np.ndarray | None:
        if pos < long:
            return None
        s_short = sh_s.iloc[pos].values
        s_long = sh_l.iloc[pos].values
        # Keep sleeves where both > 0; if too few survive (<3), fall back to EW
        # over the long-positive set, then to global EW.
        keep_strict = np.where(
            np.isfinite(s_short) & np.isfinite(s_long)
            & (s_short > 0) & (s_long > 0),
            1.0, 0.0)
        if keep_strict.sum() >= 3:
            base = keep_strict
        else:
            relax = np.where(
                np.isfinite(s_long) & (s_long > 0), 1.0, 0.0)
            base = relax if relax.sum() >= 3 else np.ones(n)
        # Mild tilt by long-Sharpe for the kept set (cap relative tilt)
        tilt_raw = np.where((base > 0) & np.isfinite(s_long),
                            np.clip(s_long, 0.0, 3.0), 0.0)
        if tilt_raw.sum() <= 1e-12:
            tilt = base / base.sum() if base.sum() > 0 else np.full(n, 1.0/n)
        else:
            tilt = tilt_raw / tilt_raw.sum()
        # Mild shrinkage to within-kept EW (alpha = 0.5)
        if base.sum() > 0:
            kept_ew = base / base.sum()
            return 0.5 * tilt + 0.5 * kept_ew
        return np.full(n, 1.0 / n)

    return _stepwise_weights(panel, w_at)


# ----- Variant 6: Bayesian shrinkage of trailing Sharpe ----------------------
def make_bayes_sharpe_weights(panel: pd.DataFrame,
                              lookback: int = 126, k: float = 60.0,
                              alpha: float = 0.5) -> pd.DataFrame:
    """Posterior Sharpe = (n * sample + k * prior) / (n + k), prior =
    cross-sectional mean trailing Sharpe at the same time. n = lookback in
    business days (~ effective sample size). alpha mixes posterior-tilt with
    EW so we don't over-concentrate even with shrinkage."""
    sleeves = list(panel.columns)
    sh = trailing_sharpe(panel, lookback)
    n_sample = float(lookback)
    n_total = len(sleeves)

    def w_at(pos: int) -> np.ndarray | None:
        if pos < lookback:
            return None
        s = sh.iloc[pos].values
        s = np.where(np.isfinite(s), s, np.nan)
        prior = np.nanmean(s) if np.isfinite(s).any() else 0.0
        s_filled = np.where(np.isfinite(s), s, prior)
        post = (n_sample * s_filled + k * prior) / (n_sample + k)
        tilt = _sharpe_tilt_from_trailing(post)
        ew = np.full(n_total, 1.0 / n_total)
        return alpha * tilt + (1 - alpha) * ew

    return _stepwise_weights(panel, w_at)


# ---------------------------------------------------------------------------
# Pipeline: gated panel -> weights -> portfolio -> overlays -> stats
# ---------------------------------------------------------------------------
def build_gated_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply high-vol regime gates and decay tripwire — matches v11 exactly."""
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME"]:
        gates[s] = 1.5
    gates["VOLFORECAST"] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["SESSION_MOM"] = 1.0

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m
    after_decay, _ = fast_decay_tripwire(g[TOP14], TOP14)
    return after_decay


def evaluate(panel_gated: pd.DataFrame, W: pd.DataFrame, label: str
             ) -> tuple[pd.Series, dict]:
    """Weighted combine -> 18% vol-target -> DD control -> stats."""
    # Sleeve weights are sized to *sum to 1* (i.e. they are shares of a
    # 1-unit portfolio). Equal-weight uses mean(axis=1) which is also share-
    # based. So the weighted version is (W * R).sum(axis=1).
    Ws = W.reindex_like(panel_gated).fillna(1.0 / panel_gated.shape[1])
    port = (Ws * panel_gated).sum(axis=1)
    vt, _ = vol_target_overlay(port, target_vol=VOL_TARGET, max_lev=MAX_LEV)
    prod, _ = drawdown_control(vt)
    s = stats_block(prod)
    s["label"] = label
    return prod, s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    panel = pd.read_parquet(PANEL_PATH)
    panel.index = pd.to_datetime(panel.index, utc=True)
    panel = panel[TOP14]
    print(f"Loaded panel: {panel.shape[0]} bars x {panel.shape[1]} sleeves")

    panel_gated = build_gated_panel(panel)

    # ---- Equal-weight baseline (re-run for apples-to-apples comparison) ----
    n = panel_gated.shape[1]
    W_ew = pd.DataFrame(1.0 / n, index=panel_gated.index, columns=panel_gated.columns)
    ew_ret, ew_stats = evaluate(panel_gated, W_ew, "EW_baseline")

    # ---- Build all variants ----
    variants = {}

    # 1) Shrinkage sweep
    for a in [0.1, 0.2, 0.3, 0.5]:
        W = make_shrinkage_weights(panel_gated, alpha=a, lookback=126)
        ret, st = evaluate(panel_gated, W, f"V1_shrink_a={a:.1f}")
        variants[f"V1_shrink_a={a:.1f}"] = (W, ret, st)

    # 2) Bucket + sharpe within
    W = make_bucket_sharpe_tilt_weights(panel_gated, lookback=126, within_alpha=0.5)
    ret, st = evaluate(panel_gated, W, "V2_bucket_sharpe")
    variants["V2_bucket_sharpe"] = (W, ret, st)

    # 3) Inverse-vol within bucket
    W = make_bucket_invvol_weights(panel_gated, lookback=63)
    ret, st = evaluate(panel_gated, W, "V3_bucket_invvol")
    variants["V3_bucket_invvol"] = (W, ret, st)

    # 4) Risk-budget
    W = make_risk_budget_weights(panel_gated, lookback=63)
    ret, st = evaluate(panel_gated, W, "V4_risk_budget")
    variants["V4_risk_budget"] = (W, ret, st)

    # 5) Momentum filter (3m+12m positive Sharpe)
    W = make_momfilter_weights(panel_gated, short=63, long=252)
    ret, st = evaluate(panel_gated, W, "V5_mom_filter")
    variants["V5_mom_filter"] = (W, ret, st)

    # 6) Bayesian shrinkage of trailing Sharpe
    W = make_bayes_sharpe_weights(panel_gated, lookback=126, k=60.0, alpha=0.5)
    ret, st = evaluate(panel_gated, W, "V6_bayes_sharpe")
    variants["V6_bayes_sharpe"] = (W, ret, st)

    # ---- Comparison table ----
    rows = [ew_stats] + [v[2] for v in variants.values()]
    df = pd.DataFrame(rows)
    cols = ["label", "FULL_sharpe", "IS_sharpe", "OOS_sharpe", "YR2022_sharpe",
            "OOS_ret", "OOS_vol", "OOS_dd", "FULL_dd"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT / "adaptive_weighting_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print(f"{'Variant':<24} {'FULL_Sh':>8} {'IS_Sh':>7} {'OOS_Sh':>8} "
          f"{'2022_Sh':>8} {'OOS_DD':>8}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['label']:<24} {row.get('FULL_sharpe',0):>+8.3f} "
              f"{row.get('IS_sharpe',0):>+7.2f} {row.get('OOS_sharpe',0):>+8.3f} "
              f"{row.get('YR2022_sharpe',0):>+8.2f} {row.get('OOS_dd',0):>+8.1%}")
    print("=" * 90)
    print(f"\nEW baseline OOS Sharpe = {ew_stats['OOS_sharpe']:+.3f}")

    # ---- Pick best by OOS Sharpe ----
    best_name = None
    best_oos = ew_stats["OOS_sharpe"]
    for name, (W, ret, st) in variants.items():
        if st["OOS_sharpe"] > best_oos:
            best_oos = st["OOS_sharpe"]
            best_name = name
    if best_name is None:
        print("\nNo variant beat EW OOS Sharpe.")
        best_name = max(variants, key=lambda k: variants[k][2]["OOS_sharpe"])
    print(f"\nBest variant: {best_name}  (OOS Sh = {variants[best_name][2]['OOS_sharpe']:+.3f}, "
          f"EW = {ew_stats['OOS_sharpe']:+.3f}, "
          f"delta = {variants[best_name][2]['OOS_sharpe'] - ew_stats['OOS_sharpe']:+.3f})")

    # ---- Save best variant returns + monthly weight log ----
    W_best, ret_best, st_best = variants[best_name]
    eq = (1 + ret_best).cumprod()
    pd.DataFrame({"timestamp": ret_best.index, "ret": ret_best.values,
                  "equity": eq.values}).to_parquet(
        OUT / "adaptive_weighting_returns.parquet", index=False)

    # Monthly weight log: sample at each month-end across ALL variants for traceability
    me_pos = month_end_index_positions(panel_gated.index)
    log_rows = []
    for name, (W, _, _) in [("EW_baseline", (W_ew, None, None))] + \
                           [(n, (v[0], None, None)) for n, v in variants.items()]:
        for pos in me_pos:
            ts = panel_gated.index[pos]
            row = {"variant": name, "month_end": ts}
            for s in panel_gated.columns:
                row[s] = float(W.iloc[pos][s])
            log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(OUT / "adaptive_weighting_weights.csv", index=False)
    print(f"\nSaved: adaptive_weighting_returns.parquet (best={best_name})")
    print(f"Saved: adaptive_weighting_comparison.csv")
    print(f"Saved: adaptive_weighting_weights.csv")

    # ---- Diagnostic: how concentrated are best variant's OOS weights? ----
    oos_mask = pd.Index(panel_gated.index) >= SPLIT
    W_oos = W_best.loc[oos_mask]
    avg_w = W_oos.mean()
    top3 = avg_w.sort_values(ascending=False).head(5)
    bot3 = avg_w.sort_values(ascending=False).tail(3)
    print(f"\nBest variant OOS avg weights — top 5:")
    for s, v in top3.items():
        print(f"  {s:<18} {v:.3f}  (EW={1/n:.3f})")
    print(f"Bottom 3:")
    for s, v in bot3.items():
        print(f"  {s:<18} {v:.3f}")
    # Effective N
    eff_n = 1.0 / (avg_w ** 2).sum()
    print(f"\nEffective N (OOS mean): {eff_n:.2f}  (EW = {n})")

    # Persistence diagnostic: top-3 sleeves by half-year
    full_idx = W_best.index
    h1_m = (full_idx >= pd.Timestamp("2024-01-01", tz="UTC")) & \
           (full_idx <  pd.Timestamp("2024-07-01", tz="UTC"))
    h2_m = (full_idx >= pd.Timestamp("2024-07-01", tz="UTC")) & \
           (full_idx <  pd.Timestamp("2025-01-01", tz="UTC"))
    y25_m = (full_idx >= pd.Timestamp("2025-01-01", tz="UTC"))
    h1 = W_best.loc[h1_m].mean()
    h2 = W_best.loc[h2_m].mean()
    y25 = W_best.loc[y25_m].mean()
    print(f"\nTop-3 by half-year (best variant):")
    print(f"  2024H1: {list(h1.sort_values(ascending=False).head(3).index)}")
    print(f"  2024H2: {list(h2.sort_values(ascending=False).head(3).index)}")
    print(f"  2025+:  {list(y25.sort_values(ascending=False).head(3).index)}")

    # Persistence diagnostic for V1_shrink_a=0.3 and V6 also (higher tilt)
    for diag_name in ["V1_shrink_a=0.3", "V1_shrink_a=0.5", "V6_bayes_sharpe", "V5_mom_filter"]:
        if diag_name in variants:
            W_d = variants[diag_name][0]
            h1d = W_d.loc[h1_m].mean()
            h2d = W_d.loc[h2_m].mean()
            y25d = W_d.loc[y25_m].mean()
            print(f"\nTop-3 by half-year ({diag_name}):")
            print(f"  2024H1: {list(h1d.sort_values(ascending=False).head(3).index)}")
            print(f"  2024H2: {list(h2d.sort_values(ascending=False).head(3).index)}")
            print(f"  2025+:  {list(y25d.sort_values(ascending=False).head(3).index)}")


if __name__ == "__main__":
    main()
