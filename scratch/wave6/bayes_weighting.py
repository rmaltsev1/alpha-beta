"""Wave 6 -- Bayesian-shrinkage sleeve weighting (numpy only).

Prior linear / ML / Kelly attempts failed against the equal-weight TOP22
baseline. This script tries a proper hierarchical Bayesian approach:

    posterior_Sharpe_i = (n_i * sample_Sharpe_i + tau0 * mu0) /
                         (n_i + tau0)

where
    mu0  : cross-sectional grand mean of trailing-12m Sharpes (per-rebalance).
    tau0 : prior strength -- empirical-Bayes scaled by the cross-sectional
           variance of sleeve Sharpes; small variance -> high tau0
           (strong shrinkage), large variance -> low tau0 (let data speak).
    n_i  : effective sample size (number of bars in the trailing-12m
           window divided by a per-sleeve "Sharpe stability" factor).

Three families are run:

  A) `bayes_sh_alpha{0.5,1.0,2.0}`  -- empirical-Bayes shrinkage with
       weight  ~  max(0, posterior_Sharpe)^alpha, 25% cap.
  B) `bayes_vol_alpha1`              -- vol-of-Sharpe shrinkage.  Sleeves
       whose trailing-3m Sharpe is stable (low std-of-rolling-Sharpes)
       get a *lower* tau0 (less shrinkage); volatile sleeves get a
       *higher* tau0 (more shrinkage to the grand mean).
  C) `bayes_tau_{LOW,MED,HIGH}`      -- a tau0-sensitivity sweep at a
       fixed alpha=1.0 to confirm the EB choice isn't a sweet spot.

Compared against:
  - `EW_TOP22`           : equal-weight TOP22 (current production baseline).
  - `Sharpe_tilt_alpha1` : naive trailing-12m-Sharpe weighting, no shrinkage.
  - `Half_Kelly`         : already known to fail; included for completeness.

Methodology
-----------
* Panel  : scratch/quant/all_sleeve_returns_v14.parquet  (TOP22 sleeves).
* Split  : IS < 2024-01-01,  OOS >= 2024-01-01.
* Walk-forward EVERYTHING (no look-ahead): every Sharpe, prior, and weight
  is computed strictly from bars prior to the rebalance month-end.
* Monthly rebalance, weights applied next bar onwards.
* Regime gate (v14 GATES_HIGH adjustments) + fast decay tripwire.
* Vol-target the final portfolio to 10% annualised, max-lev 15x.
* Drawdown control: halve gross when trailing 30d DD < -3%, recover at -1%.

Outputs
-------
* scratch/wave6/bayes_weighting.py       -- this script.
* scratch/wave6/bayes_returns.parquet    -- daily returns of every variant.
* scratch/wave6/bayes_weights.csv        -- per-month, per-sleeve weights
                                            + diagnostics (mu0, tau0, sd).
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
    build_regime_mask, fast_decay_tripwire, vol_target_overlay,
    drawdown_control, GATES_HIGH, _bpy,
)

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.10
MAX_LEV = 15.0
ANN = 365.25
SQRT_ANN = np.sqrt(ANN)
CAP = 0.25
LOOKBACK_12M = 252
LOOKBACK_3M = 63

# v14 TOP22 sleeve set (production baseline).
TOP22 = [
    "RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
    "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
    "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
    "CORR_REGIME", "SESSION_MOM",
    "W1_STRATS", "EVENT_VOLSPIKE",
    "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
    "TERM_SPREADS", "EURGBP_MR", "MULTIDAY",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def month_end_iterator(idx: pd.DatetimeIndex):
    me = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="ME", tz="UTC")
    for ts in me:
        pos = idx.searchsorted(ts, side="right") - 1
        if pos <= 0:
            continue
        yield ts, pos


def trailing_sharpe(window: pd.Series) -> float:
    mu = float(window.mean()) * ANN
    sd = float(window.std(ddof=0)) * SQRT_ANN
    return mu / sd if sd > 1e-9 else 0.0


def vol_of_sharpe(panel_window: pd.DataFrame, sleeve: str,
                   subwin: int = LOOKBACK_3M, n_chunks: int = 4) -> float:
    """Standard deviation of trailing-3m Sharpe across `n_chunks` consecutive
    non-overlapping 63-day blocks of the trailing-12m window."""
    r = panel_window[sleeve].values
    if len(r) < subwin * 2:
        return 0.0
    # Take the last n_chunks non-overlapping subwin blocks.
    n_take = min(n_chunks, len(r) // subwin)
    sh = []
    for k in range(n_take):
        block = r[-subwin * (k + 1): len(r) - subwin * k] if k > 0 else r[-subwin:]
        if block.std(ddof=0) > 1e-9:
            sh.append(block.mean() * ANN / (block.std(ddof=0) * SQRT_ANN))
        else:
            sh.append(0.0)
    return float(np.std(sh, ddof=0))


def apply_weights_to_panel(panel: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    """Multiply each daily sleeve return by its forward monthly weight.

    Weights are set only on month-end rows. Zero-rows -> NaN before ffill so
    weights propagate forward, then shift(1) so a weight decided on date t
    is applied to the return on t+1 onward.
    """
    w = weights_df.copy()
    zero_rows = (w.abs().sum(axis=1) < 1e-12)
    w[zero_rows] = np.nan
    w = w.reindex(panel.index).ffill().shift(1).fillna(0.0)
    return (panel * w).sum(axis=1)


def normalise_cap(raw: np.ndarray, cap: float) -> np.ndarray:
    """Floor at 0, cap at `cap`, renormalise. Iterative -- needed because
    capping breaks the sum, so we redistribute the residual to uncapped
    sleeves up to the cap (max 10 passes)."""
    w = np.maximum(raw, 0.0)
    if w.sum() < 1e-12:
        return np.ones_like(w) / len(w)
    w = w / w.sum()
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        remaining = ~over & (w > 0)
        if not remaining.any():
            # All sleeves capped -- distribute evenly.
            return np.ones_like(w) / len(w)
        # Distribute excess proportional to remaining weights.
        w[remaining] += excess * (w[remaining] / w[remaining].sum())
    return w / w.sum()


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
def weights_equal(panel: pd.DataFrame, sleeves: list[str]) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    n = len(sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < LOOKBACK_12M:
            continue
        w.iloc[pos] = np.ones(n) / n
    return w


def weights_sharpe_tilt(panel: pd.DataFrame, sleeves: list[str],
                         alpha: float = 1.0) -> pd.DataFrame:
    """Naive trailing-12m-Sharpe weighting, no shrinkage."""
    w = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < LOOKBACK_12M:
            continue
        win = panel.iloc[pos - LOOKBACK_12M + 1: pos + 1]
        sh = np.array([trailing_sharpe(win[s]) for s in sleeves])
        raw = np.power(np.maximum(sh, 0.0), alpha)
        w.iloc[pos] = normalise_cap(raw, CAP)
    return w


def weights_half_kelly(panel: pd.DataFrame, sleeves: list[str]) -> pd.DataFrame:
    """Half-Kelly: f = 0.5 * mu / sigma^2, cap=CAP, renormalise."""
    w = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < LOOKBACK_12M:
            continue
        win = panel.iloc[pos - LOOKBACK_12M + 1: pos + 1]
        f = []
        for s in sleeves:
            mu = float(win[s].mean()) * ANN
            var = float(win[s].var(ddof=0)) * ANN
            f.append(0.0 if var <= 1e-9 else max(0.0, 0.5 * mu / var))
        w.iloc[pos] = normalise_cap(np.array(f), CAP)
    return w


def weights_bayes_shrunk(panel: pd.DataFrame, sleeves: list[str],
                          alpha: float = 1.0, tau0_mode: str = "EB",
                          tau0_value: float | None = None,
                          vol_aware: bool = False,
                          log_rows: list | None = None) -> pd.DataFrame:
    """Bayesian shrinkage variant.

    Posterior_i = (n_eff_i * sample_Sharpe_i + tau0_i * mu0) / (n_eff_i + tau0_i)

    tau0_mode:
        "EB"     -- empirical Bayes: tau0 = LOOKBACK_12M / (1 + k * var(sample_Sharpes))
                    so that low cross-sectional variance => strong shrinkage.
        "fixed"  -- tau0 = tau0_value (constant).

    vol_aware:
        If True, scale the *per-sleeve* tau0 multiplicatively by vol-of-Sharpe
        relative to the cross-sectional median: stable sleeves get less
        shrinkage, volatile ones get more.

    log_rows: optional list to which we append per-month diagnostics.
    """
    w = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < LOOKBACK_12M:
            continue
        win = panel.iloc[pos - LOOKBACK_12M + 1: pos + 1]
        n = LOOKBACK_12M

        sample_sh = np.array([trailing_sharpe(win[s]) for s in sleeves])
        mu0 = float(sample_sh.mean())
        var0 = float(sample_sh.var(ddof=0))

        # tau0 base
        if tau0_mode == "EB":
            # Empirical Bayes: high cross-sectional variance -> low tau0.
            # k controls calibration; var of Sharpes is typically O(0.2-1.0).
            tau0_base = n / (1.0 + 5.0 * var0)
        else:
            tau0_base = float(tau0_value if tau0_value is not None else n)

        tau0_arr = np.full(len(sleeves), tau0_base)

        if vol_aware:
            vol_sh = np.array([vol_of_sharpe(win, s) for s in sleeves])
            med = np.median(vol_sh) if np.median(vol_sh) > 1e-9 else 1.0
            # ratio > 1 -> volatile -> more shrinkage. ratio < 1 -> stable -> less.
            ratio = vol_sh / med
            tau0_arr = tau0_arr * (0.5 + ratio)  # range ~[0.5x, 2x]

        post = (n * sample_sh + tau0_arr * mu0) / (n + tau0_arr)
        raw = np.power(np.maximum(post, 0.0), alpha)
        weights = normalise_cap(raw, CAP)
        w.iloc[pos] = weights

        if log_rows is not None:
            for i, s in enumerate(sleeves):
                log_rows.append({
                    "month_end": me, "sleeve": s,
                    "sample_sharpe": sample_sh[i],
                    "posterior_sharpe": post[i],
                    "tau0": tau0_arr[i],
                    "mu0": mu0,
                    "var0": var0,
                    "weight": weights[i],
                })
    return w


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------
def evaluate(name: str, port: pd.Series) -> tuple[dict, pd.Series]:
    """Apply vol-target + DD control overlays, then compute IS/OOS stats."""
    vt, _ = vol_target_overlay(port, target_vol=TARGET_VOL, max_lev=MAX_LEV)
    dd_ctrl, _ = drawdown_control(vt)

    out = {"variant": name}
    for tag, mask in [("FULL", pd.Series(True, index=dd_ctrl.index)),
                       ("IS",   dd_ctrl.index < SPLIT),
                       ("OOS",  dd_ctrl.index >= SPLIT)]:
        sub = dd_ctrl[mask]
        if len(sub) < 2:
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        sh = ar / av if av > 1e-9 else 0.0
        eq = (1 + sub).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        out[f"{tag}_sharpe"] = sh
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        out[f"{tag}_maxdd"] = dd

    y22 = dd_ctrl[dd_ctrl.index.year == 2022]
    if len(y22) > 10 and y22.std() > 1e-9:
        bpy22 = _bpy(y22.index)
        out["Y2022_sharpe"] = float(y22.mean()) * bpy22 / (float(y22.std(ddof=0)) * np.sqrt(bpy22))
    else:
        out["Y2022_sharpe"] = 0.0
    return out, dd_ctrl


def concentration_stats(weights: pd.DataFrame, mask) -> tuple[float, float]:
    sub = weights[mask]
    active = sub.loc[(sub.abs().sum(axis=1) > 1e-9)]
    if len(active) == 0:
        return 0.0, 0.0
    return float(active.max(axis=1).mean()), float((active ** 2).sum(axis=1).mean())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    panel_all = pd.read_parquet(QUANT / "all_sleeve_returns_v14.parquet")
    panel_all.index = pd.to_datetime(panel_all.index, utc=True)

    # ---- Regime gates (v14 setup) ----
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE", "MULTIDAY"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, _ = build_regime_mask(panel_all.index, 80)
    g = panel_all.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    # ---- Fast decay tripwire ----
    after_decay, _ = fast_decay_tripwire(g[TOP22], TOP22)
    panel = after_decay[TOP22].copy()

    # ---- Build variants ----
    log_rows: list[dict] = []

    variants: dict[str, pd.DataFrame] = {}
    print("Building variants ...")
    variants["EW_TOP22"]          = weights_equal(panel, TOP22)
    variants["Sharpe_tilt_a05"]   = weights_sharpe_tilt(panel, TOP22, alpha=0.5)
    variants["Sharpe_tilt_a10"]   = weights_sharpe_tilt(panel, TOP22, alpha=1.0)
    variants["Sharpe_tilt_a20"]   = weights_sharpe_tilt(panel, TOP22, alpha=2.0)
    variants["Half_Kelly"]        = weights_half_kelly(panel, TOP22)

    variants["bayes_sh_a05"] = weights_bayes_shrunk(panel, TOP22, alpha=0.5,
                                                     tau0_mode="EB",
                                                     log_rows=log_rows if False else None)
    variants["bayes_sh_a10"] = weights_bayes_shrunk(panel, TOP22, alpha=1.0,
                                                     tau0_mode="EB",
                                                     log_rows=log_rows)
    variants["bayes_sh_a20"] = weights_bayes_shrunk(panel, TOP22, alpha=2.0,
                                                     tau0_mode="EB")
    variants["bayes_vol_a10"] = weights_bayes_shrunk(panel, TOP22, alpha=1.0,
                                                      tau0_mode="EB",
                                                      vol_aware=True)
    # tau0 sensitivity sweep (alpha=1.0)
    variants["bayes_tauLOW_a10"]  = weights_bayes_shrunk(panel, TOP22, alpha=1.0,
                                                          tau0_mode="fixed",
                                                          tau0_value=50.0)
    variants["bayes_tauMED_a10"]  = weights_bayes_shrunk(panel, TOP22, alpha=1.0,
                                                          tau0_mode="fixed",
                                                          tau0_value=252.0)
    variants["bayes_tauHIGH_a10"] = weights_bayes_shrunk(panel, TOP22, alpha=1.0,
                                                          tau0_mode="fixed",
                                                          tau0_value=1000.0)

    # ---- Evaluate ----
    daily = pd.DataFrame(index=panel.index)
    rows: list[dict] = []
    oos_mask = panel.index >= SPLIT
    for name, w in variants.items():
        port = apply_weights_to_panel(panel, w)
        stats_, dd_ret = evaluate(name, port)
        max_w, hhi = concentration_stats(w, oos_mask)
        stats_["OOS_avg_maxweight"] = max_w
        stats_["OOS_avg_HHI"] = hhi
        rows.append(stats_)
        daily[name] = dd_ret
        print(f"{name:<22}  FULL={stats_.get('FULL_sharpe',0):+.2f}  "
              f"IS={stats_.get('IS_sharpe',0):+.2f}  "
              f"OOS={stats_.get('OOS_sharpe',0):+.2f}  "
              f"2022={stats_.get('Y2022_sharpe',0):+.2f}  "
              f"OOSdd={stats_.get('OOS_maxdd',0):+.1%}  "
              f"maxW={max_w:.2%}")

    # ---- Persist ----
    daily.to_parquet(OUT / "bayes_returns.parquet")
    pd.DataFrame(rows).to_csv(OUT / "bayes_summary.csv", index=False)
    pd.DataFrame(log_rows).to_csv(OUT / "bayes_weights.csv", index=False)

    # ---- Headline comparison ----
    ew = next(r for r in rows if r["variant"] == "EW_TOP22")
    naive = next(r for r in rows if r["variant"] == "Sharpe_tilt_a10")
    print()
    print(f"EW_TOP22 baseline       FULL={ew['FULL_sharpe']:+.2f}  IS={ew['IS_sharpe']:+.2f}  "
          f"OOS={ew['OOS_sharpe']:+.2f}  2022={ew['Y2022_sharpe']:+.2f}")
    print(f"Naive Sharpe tilt a=1.0 FULL={naive['FULL_sharpe']:+.2f}  IS={naive['IS_sharpe']:+.2f}  "
          f"OOS={naive['OOS_sharpe']:+.2f}  2022={naive['Y2022_sharpe']:+.2f}")

    # ---- Constraint scan ----
    ew_sh = ew["OOS_sharpe"]
    ew_dd = ew["OOS_maxdd"]
    print(f"\nEW OOS Sharpe={ew_sh:+.3f}  OOS MaxDD={ew_dd:+.2%}")
    print(f"{'variant':<24} {'dSh':>7} {'dDD_rel':>10} {'pass':>6}")
    print("-" * 50)
    for r in rows:
        if r["variant"] == "EW_TOP22":
            continue
        dsh = r["OOS_sharpe"] - ew_sh
        dd_rel = (abs(r["OOS_maxdd"]) - abs(ew_dd)) / abs(ew_dd) if abs(ew_dd) > 1e-9 else 0
        passed = (dsh >= 0.10) and (dd_rel <= 0.10)
        print(f"{r['variant']:<24} {dsh:>+7.3f} {dd_rel:>+10.2%} {('YES' if passed else 'no'):>6}")

    print(f"\nSaved: {OUT/'bayes_returns.parquet'}")
    print(f"Saved: {OUT/'bayes_weights.csv'}")
    print(f"Saved: {OUT/'bayes_summary.csv'}")


if __name__ == "__main__":
    main()
