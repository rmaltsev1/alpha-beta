"""Asymmetric per-asset vol targeting.

Approach
--------
1. Build a sleeve -> symbol weight matrix from `*_breakdown.csv` files.
   Each sleeve's exposure to a given symbol is proportional to that symbol's
   |OOS Sharpe| (or full Sharpe fallback) within the sleeve, normalized to 1.
   Hard-coded single-asset sleeves (EVE_XAU, WED_BTC, ...) are mapped directly.

2. Decompose each v15 sleeve return into per-symbol returns:
       sym_ret[s,t] = sum_over_sleeves( w[sleeve, s] * sleeve_ret[sleeve, t] )

   The column sums of `sym_ret` reproduce the equal-weight TOP23 portfolio of
   `all_sleeve_returns_v15.parquet` to within rounding.

3. Compute per-symbol edge metrics (OOS Sharpe, 2022 Sharpe, 2024-25 Sharpe,
   symbol-level vol). These drive the 5 vol-budget variants.

4. For each variant, build a vector of per-symbol multipliers `m[s]` that
   normalize across symbols so the gross vol budget stays at the symmetric
   baseline. Final portfolio = mean over sleeves of  sum_s m[s] * sleeve_sym_part.
   In practice we just scale the per-symbol returns by m[s] and average.

5. Compare variants on OOS Sharpe, OOS return, OOS vol, OOS DD.

Outputs
-------
* per_asset_returns.parquet  - daily returns for each variant
* per_asset_edge.csv         - per-symbol edge ranking
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "scratch" / "quant"
WAVE3 = ROOT / "scratch" / "wave3"
WAVE6 = ROOT / "scratch" / "wave6"
OUT_DIR = WAVE6

SPLIT_OOS = pd.Timestamp("2024-01-01", tz="UTC")

# 13 underlying symbols (basket / multi-asset rows are excluded)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD",
    "SPX500_USD", "NAS100_USD", "US30_USD",
    "UK100_GBP", "DE30_EUR", "JP225_USD",
]

# TOP23 v15 production sleeves (excludes HMM trio that requires extra steps)
TOP23 = [
    "RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
    "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
    "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
    "CORR_REGIME", "SESSION_MOM",
    "W1_STRATS", "EVENT_VOLSPIKE",
    "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
    "TERM_SPREADS", "EURGBP_MR", "MULTIDAY",
    "HMM_BULL_TSMOM",
]

# Hard-coded single/dual asset sleeves (sleeve -> {symbol: weight})
HARDCODED = {
    "EVE_XAU":          {"XAU_USD": 1.0},
    "WED_BTC":          {"BTCUSDT": 1.0},
    "WED_ETH":          {"ETHUSDT": 1.0},
    "WED_SOL":          {"SOLUSDT": 1.0},
    "D1REV_NAS":        {"NAS100_USD": 1.0},
    "D1REV_UK":         {"UK100_GBP": 1.0},
    "D1REV_SPX":        {"SPX500_USD": 1.0},
    "EURGBP_MR":        {"EUR_USD": 0.5, "GBP_USD": 0.5},
    "CRYPTO_vs_SPX":    {"BTCUSDT": 0.5, "SPX500_USD": 0.5},
}

# Map v15 sleeve names -> breakdown CSV path (best inference)
SLEEVE_BREAKDOWN = {
    "TSMOM":           QUANT / "tsmom_breakdown.csv",
    "VOLMGD":          QUANT / "volmgmt_breakdown.csv",
    "VOLFORECAST":     WAVE3 / "volforecast_breakdown.csv",
    "H4_SLEEVE":       WAVE3 / "h4_breakdown.csv",
    "TREND_NEW":       WAVE3 / "trend_breakdown.csv",
    "SESSION_MOM":     WAVE3 / "session_momentum_breakdown.csv",
    "W1_STRATS":       WAVE6 / "w1_breakdown.csv",
    "MOM_QUALITY":     WAVE6 / "mom_quality_breakdown.csv",
    "TREND_COND":      WAVE6 / "trend_conditional_breakdown.csv",
    "TERM_SPREADS":    WAVE6 / "term_spreads_breakdown.csv",
    "TAIL_SAFEHAVEN":  QUANT / "tail_v2_breakdown.csv",
    "CRYPTO_DOM":      QUANT / "crypto_dom_breakdown.csv",
}

# Sleeves with no per-symbol breakdown: fall back to equal-weight over a
# heuristic universe.
SLEEVE_FALLBACK_UNIVERSE = {
    # default: all 13 (cross-asset diversified)
    "RISKPAR":         SYMBOLS,
    "XSMOM":           SYMBOLS,
    "DEFEND":          SYMBOLS,
    "PAIRS_EXP":       SYMBOLS,
    "CORR_REGIME":     SYMBOLS,
    "MULTI_CONFIRM":   SYMBOLS,
    "EVENT_VOLSPIKE":  SYMBOLS,
    "VRP_PROXY":       ["SPX500_USD", "NAS100_USD", "US30_USD",
                         "DE30_EUR", "UK100_GBP", "JP225_USD"],
    "STATARB_XS":      ["SPX500_USD", "NAS100_USD", "US30_USD",
                         "DE30_EUR", "UK100_GBP", "JP225_USD"],
    "MICROSTR_D1":     SYMBOLS,
    "VOL_BREAKOUT":    SYMBOLS,
    "MULTIDAY":        SYMBOLS,
    "HMM_BULL_TSMOM":  SYMBOLS,
    "HMM_SLEEVE_MIX":  SYMBOLS,
    "HMM_BEAR_REV":    SYMBOLS,
}

ANN_FACTOR = 365.25  # daily UTC index


# ---------------------------------------------------------------------------
def load_breakdown(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cl = {c.lower(): c for c in df.columns}
    sym_col = cl.get("symbol") or cl.get("asset")
    if sym_col is None:
        return None
    # Pick best Sharpe column: prefer OOS, then full
    oos = cl.get("oos_sharpe") or cl.get("sharpe_oos")
    full = cl.get("full_sharpe") or cl.get("sharpe_full")
    score_col = oos or full
    if score_col is None:
        return None
    out = df[[sym_col, score_col]].copy()
    out.columns = ["symbol", "score"]
    out["symbol"] = out["symbol"].astype(str)
    return out


def sleeve_weights(sleeve: str) -> dict:
    """Return {symbol: weight} for a sleeve, normalized to sum=1 over SYMBOLS."""
    if sleeve in HARDCODED:
        w = HARDCODED[sleeve]
    elif sleeve in SLEEVE_BREAKDOWN and SLEEVE_BREAKDOWN[sleeve].exists():
        bd = load_breakdown(SLEEVE_BREAKDOWN[sleeve])
        if bd is None or len(bd) == 0:
            w = {s: 1.0 for s in SLEEVE_FALLBACK_UNIVERSE.get(sleeve, SYMBOLS)}
        else:
            # Aggregate to symbol-level: sum |score| (using absolute as proxy)
            bd = bd.groupby("symbol")["score"].apply(lambda x: np.nanmean(np.abs(x))).reset_index()
            bd = bd[bd["symbol"].isin(SYMBOLS)]
            if bd["score"].sum() <= 0 or bd.empty:
                w = {s: 1.0 for s in SLEEVE_FALLBACK_UNIVERSE.get(sleeve, SYMBOLS)}
            else:
                w = dict(zip(bd["symbol"], bd["score"]))
    else:
        w = {s: 1.0 for s in SLEEVE_FALLBACK_UNIVERSE.get(sleeve, SYMBOLS)}

    # Fill missing symbols with 0, normalize to sum=1
    w = {s: float(w.get(s, 0.0)) for s in SYMBOLS}
    tot = sum(w.values())
    if tot <= 0:
        return {s: 1.0 / len(SYMBOLS) for s in SYMBOLS}
    return {s: v / tot for s, v in w.items()}


# ---------------------------------------------------------------------------
def per_symbol_edge() -> pd.DataFrame:
    """Aggregate Sharpe contributions per symbol across all breakdowns."""
    files = sorted(
        glob.glob(str(WAVE3 / "*_breakdown.csv"))
        + glob.glob(str(WAVE6 / "*_breakdown.csv"))
        + glob.glob(str(QUANT / "*_breakdown.csv"))
    )
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cl = {c.lower(): c for c in df.columns}
        sym_col = cl.get("symbol") or cl.get("asset")
        if sym_col is None:
            continue
        oos_col = cl.get("oos_sharpe") or cl.get("sharpe_oos")
        is_col = cl.get("is_sharpe") or cl.get("sharpe_is")
        y22_col = (cl.get("y2022_sharpe") or cl.get("yr2022_sharpe")
                   or cl.get("2022_sharpe") or cl.get("sharpe_2022"))
        y2425_col = cl.get("y2024_25_sharpe")
        for _, r in df.iterrows():
            sym = str(r[sym_col])
            if sym not in SYMBOLS:
                continue
            rows.append({
                "symbol": sym,
                "sleeve": os.path.basename(f).replace("_breakdown.csv", ""),
                "oos_sharpe": float(r[oos_col]) if oos_col and pd.notna(r[oos_col]) else np.nan,
                "is_sharpe": float(r[is_col]) if is_col and pd.notna(r[is_col]) else np.nan,
                "y2022_sharpe": float(r[y22_col]) if y22_col and pd.notna(r[y22_col]) else np.nan,
                "y2024_25_sharpe": float(r[y2425_col]) if y2425_col and pd.notna(r[y2425_col]) else np.nan,
            })
    long = pd.DataFrame(rows)
    agg = long.groupby("symbol").agg(
        n_sleeves=("sleeve", "nunique"),
        oos_sharpe_mean=("oos_sharpe", "mean"),
        oos_sharpe_sum=("oos_sharpe", "sum"),
        is_sharpe_mean=("is_sharpe", "mean"),
        y2022_sharpe_mean=("y2022_sharpe", "mean"),
        y2024_25_sharpe_mean=("y2024_25_sharpe", "mean"),
    ).reindex(SYMBOLS)
    return agg.reset_index()


# ---------------------------------------------------------------------------
def build_weight_matrix(sleeves: list[str]) -> pd.DataFrame:
    """Return DataFrame indexed by sleeve, columns = SYMBOLS, weights sum=1 per row."""
    rows = {}
    for sl in sleeves:
        rows[sl] = sleeve_weights(sl)
    return pd.DataFrame(rows).T[SYMBOLS]


def stats(name: str, ret: pd.Series) -> dict:
    full = ret.dropna()
    is_ = full[full.index < SPLIT_OOS]
    oos = full[full.index >= SPLIT_OOS]
    y22 = full[full.index.year == 2022]
    def sh(x):
        if len(x) == 0 or x.std() == 0: return 0.0
        return float(x.mean() * ANN_FACTOR / (x.std(ddof=0) * np.sqrt(ANN_FACTOR)))
    def ar(x):
        return float(x.mean() * ANN_FACTOR) if len(x) else 0.0
    def vol(x):
        return float(x.std(ddof=0) * np.sqrt(ANN_FACTOR)) if len(x) else 0.0
    def dd(x):
        if len(x) == 0: return 0.0
        eq = (1 + x).cumprod()
        return float((eq / eq.cummax() - 1).min())
    return {
        "variant": name,
        "OOS_sharpe": sh(oos),
        "OOS_ret": ar(oos),
        "OOS_vol": vol(oos),
        "OOS_dd": dd(oos),
        "IS_sharpe": sh(is_),
        "Y2022_sharpe": sh(y22),
        "Full_sharpe": sh(full),
    }


# ---------------------------------------------------------------------------
def main():
    # ---------- Load v15 sleeve panel ----------
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v15.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    sleeves = [s for s in TOP23 if s in panel.columns]
    missing = [s for s in TOP23 if s not in panel.columns]
    if missing:
        print(f"WARN missing in panel: {missing}")
    print(f"Using {len(sleeves)} sleeves: {sleeves}")

    # ---------- Build sleeve x symbol weight matrix ----------
    W = build_weight_matrix(sleeves)
    print("\nWeight matrix (rows=sleeves, cols=symbols):")
    print(W.round(3))

    # ---------- Decompose: sym_ret[s,t] = sum_sl  (1/n_sleeves) * W[sl,s] * panel[sl,t]
    # The v15 baseline portfolio is panel[sleeves].mean(axis=1).
    # Decomposing as a sum over symbols: baseline_t = sum_s  per_sym_contrib[s,t]
    # where per_sym_contrib[s,t] = (1/n_sleeves) * sum_sl  W[sl,s] * panel[sl,t]
    n = len(sleeves)
    contrib = pd.DataFrame(index=panel.index, columns=SYMBOLS, dtype=float)
    R = panel[sleeves].values  # (T, n)
    Wm = W.values               # (n, |SYMBOLS|)
    contrib_vals = (R @ Wm) / n
    contrib = pd.DataFrame(contrib_vals, index=panel.index, columns=SYMBOLS)
    baseline = contrib.sum(axis=1)
    baseline_direct = panel[sleeves].mean(axis=1)
    # Sanity check
    diff = (baseline - baseline_direct).abs().max()
    print(f"\nDecomposition sanity: |reconstructed - direct| max = {diff:.2e} (should be 0)")

    # ---------- Per-asset edge ----------
    edge = per_symbol_edge()

    # Symbol vol over IS window (annualized)
    is_part = contrib[contrib.index < SPLIT_OOS]
    sym_vol_is = is_part.std(ddof=0) * np.sqrt(ANN_FACTOR)
    sym_sharpe_oos = {}
    sym_sharpe_2022 = {}
    sym_sharpe_oos_window = {}
    oos_part = contrib[contrib.index >= SPLIT_OOS]
    y22_part = contrib[contrib.index.year == 2022]
    for s in SYMBOLS:
        x = is_part[s]
        sym_sharpe_oos[s] = (oos_part[s].mean() * ANN_FACTOR /
                              (oos_part[s].std(ddof=0) * np.sqrt(ANN_FACTOR))
                              if oos_part[s].std() > 0 else 0.0)
        sym_sharpe_2022[s] = (y22_part[s].mean() * ANN_FACTOR /
                               (y22_part[s].std(ddof=0) * np.sqrt(ANN_FACTOR))
                               if y22_part[s].std() > 0 else 0.0)
    edge["panel_contrib_OOS_sharpe"] = edge["symbol"].map(sym_sharpe_oos)
    edge["panel_contrib_2022_sharpe"] = edge["symbol"].map(sym_sharpe_2022)
    edge["panel_contrib_IS_vol"] = edge["symbol"].map(sym_vol_is.to_dict())

    edge = edge.sort_values("panel_contrib_OOS_sharpe", ascending=False)
    edge.to_csv(OUT_DIR / "per_asset_edge.csv", index=False)
    print("\nPer-asset edge ranking (descending by panel OOS sharpe):")
    print(edge[["symbol", "panel_contrib_OOS_sharpe", "panel_contrib_2022_sharpe",
                 "oos_sharpe_mean", "y2022_sharpe_mean", "y2024_25_sharpe_mean",
                 "panel_contrib_IS_vol", "n_sleeves"]].round(3).to_string(index=False))

    # ---------- Build variants ----------
    # Each variant defines a multiplier m[s] for symbol s. We normalize so that
    # sum_s m[s] * w_eff[s] is preserved (i.e. the average multiplier is 1
    # across symbols weighted by their current exposure).

    # exposure proxy = sum of |W[sl,s]| / n  (i.e. average weight)
    sym_exposure = W.mean(axis=0)  # already normalized within each sleeve
    sym_exposure = sym_exposure / sym_exposure.sum()

    def normalize(m: pd.Series) -> pd.Series:
        # Normalize so weighted (by current exposure) avg = 1
        avg = (m * sym_exposure).sum()
        if avg <= 0:
            return pd.Series(1.0, index=m.index)
        return m / avg

    def apply_mults(mults: pd.Series) -> pd.Series:
        m = normalize(mults).reindex(SYMBOLS).fillna(1.0).values
        return (contrib.values * m).sum(axis=1)

    variants_returns = {"baseline_sleeve_equal": baseline_direct.values}

    # V1: equal per-asset vol -> m[s] = 1/sym_vol[s] but only to equalize total vol
    # Actually V1 "equal per-asset vol budget" means each symbol contributes
    # same vol. That is m[s] = (target_vol / sym_vol[s]). Use 1/sym_vol_is as
    # the multiplier (then normalize).
    v1 = pd.Series({s: 1.0 / max(sym_vol_is[s], 1e-9) for s in SYMBOLS})
    variants_returns["v1_equal_per_asset_vol"] = apply_mults(v1)

    # V2: edge-weighted. m[s] proportional to sleeve sharpe contribution.
    # Use mean OOS Sharpe across breakdowns (clip negatives to 0.05 to avoid sign flips)
    edge_idx = edge.set_index("symbol")
    score = edge_idx["oos_sharpe_mean"].reindex(SYMBOLS).fillna(0)
    score = score.clip(lower=0.05)  # avoid negative or zero
    variants_returns["v2_edge_weighted"] = apply_mults(score)

    # V3: inverse-vol per-asset
    v3 = pd.Series({s: 1.0 / max(sym_vol_is[s], 1e-9) for s in SYMBOLS})
    variants_returns["v3_inverse_vol"] = apply_mults(v3)  # same form as V1 but
    # V1 and V3 are essentially equivalent here; keep both but distinguish:
    # Redefine V1 to be truly equal per-symbol budget (m = 1 for each)
    # since both definitions collapse otherwise.
    variants_returns["v1_equal_per_asset_vol"] = apply_mults(pd.Series(1.0, index=SYMBOLS))

    # V4: 2022-weighted
    score22 = edge_idx["y2022_sharpe_mean"].reindex(SYMBOLS).fillna(0)
    score22 = score22.clip(lower=0.05)
    variants_returns["v4_2022_weighted"] = apply_mults(score22)

    # V5: OOS 2024-25 weighted -> use y2024_25_sharpe_mean if available, else
    # use the panel-decomposition OOS sharpe.
    score2425 = edge_idx["y2024_25_sharpe_mean"].reindex(SYMBOLS)
    score2425 = score2425.fillna(edge_idx["panel_contrib_OOS_sharpe"].reindex(SYMBOLS))
    score2425 = score2425.fillna(0).clip(lower=0.05)
    variants_returns["v5_oos_weighted"] = apply_mults(score2425)

    out_df = pd.DataFrame(variants_returns, index=panel.index)
    out_df.to_parquet(OUT_DIR / "per_asset_returns.parquet")

    # ---------- Compare ----------
    print("\n\n===== Variant comparison =====")
    header = f"{'variant':<30} {'OOS_Sh':>7} {'2022_Sh':>8} {'OOS_Ret':>8} {'OOS_Vol':>8} {'OOS_DD':>8} {'Full_Sh':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for name in ["baseline_sleeve_equal", "v1_equal_per_asset_vol", "v2_edge_weighted",
                  "v3_inverse_vol", "v4_2022_weighted", "v5_oos_weighted"]:
        ret = pd.Series(variants_returns[name], index=panel.index)
        st = stats(name, ret)
        rows.append(st)
        print(f"{name:<30} {st['OOS_sharpe']:>+7.2f} {st['Y2022_sharpe']:>+8.2f} "
              f"{st['OOS_ret']:>+8.1%} {st['OOS_vol']:>8.1%} {st['OOS_dd']:>+8.1%} "
              f"{st['Full_sharpe']:>+8.2f}")

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_DIR / "per_asset_variants.csv", index=False)

    # ---------- Print multiplier vectors for transparency ----------
    print("\nMultiplier vectors (after normalize):")
    mults_view = {
        "v1_equal":  normalize(pd.Series(1.0, index=SYMBOLS)),
        "v2_edge":   normalize(score),
        "v3_invvol": normalize(v3),
        "v4_y2022":  normalize(score22),
        "v5_oos":    normalize(score2425),
    }
    mt = pd.DataFrame(mults_view).round(2)
    print(mt)

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'per_asset_returns.parquet'}")
    print(f"  {OUT_DIR / 'per_asset_edge.csv'}")
    print(f"  {OUT_DIR / 'per_asset_variants.csv'}")


if __name__ == "__main__":
    main()
