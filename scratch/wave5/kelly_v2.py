"""Wave 5 — Per-asset CONSERVATIVE Kelly position-sizing.

Prior Kelly attempt (half-Kelly with 30% cap) doubled OOS MaxDD vs EW.
This script tests 6 much more conservative variants and benchmarks them
against the equal-weight (EW) baseline of the TOP15 v11 sleeve set.

Constraint: beat EW OOS Sharpe (+2.82) by >=0.10 AND with <=10% relative
MaxDD increase.

Methodology:
  * Panel: scratch/quant/all_sleeve_returns_v11.parquet (21 sleeves).
  * Split: IS < 2024-01-01, OOS >= 2024-01-01.
  * Use TOP15 sleeves (same set v11 production uses).
  * Walk-forward EVERYTHING: every weight, Kelly fraction, vol, drawdown
    flag is computed from data strictly prior to the bar it acts on.
  * Monthly rebalance: weights are recomputed at month-end and held for
    the next month.
  * Apply regime gate (v11 GATES_HIGH) + fast decay tripwire as baseline
    (same baseline used by v11).
  * Vol-target the final portfolio to 10% annualised vol; max lev 15x;
    drawdown control (halve gross beyond 3% trailing 30d DD).

Outputs:
  kelly_v2_returns.parquet  — daily returns of every variant + EW baseline.
  kelly_v2_variants.csv     — summary stats per variant.
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
    fast_decay_tripwire, vol_target_overlay, drawdown_control,
    build_regime_mask, _bpy, GATES_HIGH,
)

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.10
MAX_LEV = 15.0
ANN = 365.25
SQRT_ANN = np.sqrt(ANN)

# v11 TOP15 sleeve set (same as production)
TOP15 = [
    "RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
    "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
    "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
    "CORR_REGIME", "SESSION_MOM", "MULTI_CONFIRM",
]

# Bucket assignments for the bucket-Kelly variant.
BUCKETS = {
    "calendar": ["EVE_XAU", "WED_BTC", "SESSION_MOM"],
    "MR":       ["D1REV_NAS", "D1REV_UK", "VOLFORECAST", "H4_SLEEVE"],
    "trend":    ["TREND_NEW", "XSMOM", "MULTI_CONFIRM"],
    "beta":     ["RISKPAR", "DEFEND"],
    "pairs":    ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def month_end_iterator(idx: pd.DatetimeIndex):
    """Yield (month_end_ts, position_in_idx) for every calendar month-end."""
    me = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="ME", tz="UTC")
    for ts in me:
        pos = idx.searchsorted(ts, side="right") - 1
        if pos <= 0:
            continue
        yield ts, pos


def kelly_fraction(window: pd.Series, max_lev: float = 5.0) -> float:
    """Single-sleeve Kelly f* = mu / sigma^2 (long-only floor at 0)."""
    mu = float(window.mean()) * ANN
    var = float(window.var(ddof=0)) * ANN
    if var <= 1e-9:
        return 0.0
    f = mu / var
    return float(max(0.0, min(f, max_lev)))


def vol_of_kelly_history(history: list[float], window: int = 6) -> float:
    """Trailing standard deviation of a sleeve's monthly Kelly fractions."""
    if len(history) < 2:
        return 0.0
    arr = np.array(history[-window:])
    return float(arr.std(ddof=0))


def apply_weights_to_panel(panel: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    """Multiply each daily sleeve return by its (forward) monthly weight.

    Weights are set on month-end bars only. We replace any all-zero row
    with NaN before ffill (so month-end weights propagate forward), then
    shift by 1 bar so a weight decided on month-end t is applied to the
    return of t+1 onwards.
    """
    w = weights_df.copy()
    zero_rows = (w.abs().sum(axis=1) < 1e-12)
    w[zero_rows] = np.nan
    w = w.reindex(panel.index).ffill().shift(1).fillna(0.0)
    return (panel * w).sum(axis=1)


def trailing_sharpe(window: pd.Series) -> float:
    mu = float(window.mean()) * ANN
    sd = float(window.std(ddof=0)) * SQRT_ANN
    return mu / sd if sd > 1e-9 else 0.0


def trailing_dd(window: pd.Series) -> float:
    eq = (1 + window).cumprod()
    return float((eq / eq.cummax() - 1).min())


# ---------------------------------------------------------------------------
# Variant solvers (each returns a DataFrame of monthly sleeve weights)
# ---------------------------------------------------------------------------
def variant_quarter_kelly_capped(panel: pd.DataFrame, sleeves: list[str],
                                  lookback: int = 252, frac: float = 0.25,
                                  cap: float = 0.15) -> pd.DataFrame:
    """Quarter Kelly with 15% per-sleeve cap. Floor 0%, renormalise to sum=1."""
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < lookback:
            continue
        window = panel.iloc[pos - lookback + 1: pos + 1][sleeves]
        f = np.array([kelly_fraction(window[s]) for s in sleeves])
        f = f * frac
        f = np.minimum(f, cap)
        f = np.maximum(f, 0.0)
        tot = f.sum()
        if tot < 1e-9:
            w = np.ones(len(sleeves)) / len(sleeves)
        else:
            w = f / tot
        # Apply at this month-end timestamp; apply_weights_to_panel ffills.
        weights.iloc[pos] = w
    return weights


def variant_vol_of_kelly(panel: pd.DataFrame, sleeves: list[str],
                          lookback: int = 252, frac: float = 0.25,
                          cap: float = 0.20, vol_window: int = 6) -> pd.DataFrame:
    """Kelly damped by the *vol of Kelly fractions*.

    For each sleeve, compute its monthly Kelly fraction history. Sleeves with
    higher trailing 6m std-of-Kelly get a multiplicative damping factor
    1/(1 + k*std). Stable winners keep their Kelly; volatile ones get halved.
    """
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    history: dict[str, list[float]] = {s: [] for s in sleeves}
    for me, pos in month_end_iterator(panel.index):
        if pos < lookback:
            continue
        window = panel.iloc[pos - lookback + 1: pos + 1][sleeves]
        raw_kelly = np.array([kelly_fraction(window[s]) for s in sleeves])
        # Update history with this month's raw Kelly fraction
        for s, k in zip(sleeves, raw_kelly):
            history[s].append(k)
        damping = np.ones(len(sleeves))
        for i, s in enumerate(sleeves):
            v = vol_of_kelly_history(history[s], window=vol_window)
            # Calibrate: typical vol-of-Kelly is in [0,1]. Use 1/(1+2v).
            damping[i] = 1.0 / (1.0 + 2.0 * v)
        damped = raw_kelly * damping * frac
        damped = np.minimum(damped, cap)
        damped = np.maximum(damped, 0.0)
        tot = damped.sum()
        if tot < 1e-9:
            w = np.ones(len(sleeves)) / len(sleeves)
        else:
            w = damped / tot
        weights.iloc[pos] = w
    return weights


def variant_bucket_kelly(panel: pd.DataFrame, sleeves: list[str],
                          lookback: int = 252, frac: float = 0.25,
                          cap: float = 0.35) -> pd.DataFrame:
    """Bucket-level Kelly: aggregate sleeves into buckets, Kelly across
    buckets, equal-weight inside bucket."""
    # Restrict bucket dict to sleeves present in TOP15
    buckets = {b: [s for s in members if s in sleeves]
               for b, members in BUCKETS.items()}
    buckets = {b: m for b, m in buckets.items() if m}
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < lookback:
            continue
        window = panel.iloc[pos - lookback + 1: pos + 1]
        bucket_streams = {b: window[m].mean(axis=1) for b, m in buckets.items()}
        f = np.array([kelly_fraction(stream) for stream in bucket_streams.values()])
        f = f * frac
        f = np.minimum(f, cap)
        f = np.maximum(f, 0.0)
        tot = f.sum()
        if tot < 1e-9:
            # Fallback: equal across buckets
            f = np.ones(len(buckets)) / len(buckets)
        else:
            f = f / tot
        bucket_weights = dict(zip(buckets.keys(), f))
        row = np.zeros(len(sleeves))
        for i, s in enumerate(sleeves):
            for b, members in buckets.items():
                if s in members:
                    row[i] = bucket_weights[b] / len(members)
                    break
        # Renormalise (sleeves outside any bucket remain at 0; missing edge-case)
        if row.sum() > 1e-9:
            row = row / row.sum()
        weights.iloc[pos] = row
    return weights


def variant_kelly_lockout(panel: pd.DataFrame, sleeves: list[str],
                           lookback: int = 252, frac: float = 0.25,
                           cap: float = 0.20, dd_trigger: float = -0.05,
                           lockout_months: int = 2) -> pd.DataFrame:
    """Standard quarter-Kelly weights, but if a sleeve's trailing 3-month DD
    falls below -5%, force its weight to 1/N for the next 2 months."""
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    lockout_remaining = {s: 0 for s in sleeves}
    for me, pos in month_end_iterator(panel.index):
        if pos < lookback:
            continue
        window = panel.iloc[pos - lookback + 1: pos + 1][sleeves]
        f = np.array([kelly_fraction(window[s]) for s in sleeves])
        f = f * frac
        f = np.minimum(f, cap)
        f = np.maximum(f, 0.0)
        tot = f.sum()
        if tot < 1e-9:
            kelly_w = np.ones(len(sleeves)) / len(sleeves)
        else:
            kelly_w = f / tot
        # Check trailing 3m DD per sleeve and update lockout state
        win3 = panel.iloc[max(0, pos - 63 + 1): pos + 1][sleeves]
        en_w = np.ones(len(sleeves)) / len(sleeves)
        final = kelly_w.copy()
        for i, s in enumerate(sleeves):
            dd = trailing_dd(win3[s])
            if dd < dd_trigger:
                lockout_remaining[s] = lockout_months
            if lockout_remaining[s] > 0:
                final[i] = en_w[i]
                lockout_remaining[s] -= 1
        # Renormalise
        if final.sum() > 1e-9:
            final = final / final.sum()
        weights.iloc[pos] = final
    return weights


def variant_edge_decay_penalty(panel: pd.DataFrame, sleeves: list[str],
                                 lookback: int = 252, frac: float = 0.25,
                                 cap: float = 0.20) -> pd.DataFrame:
    """Quarter-Kelly with a per-sleeve penalty proportional to the *gap*
    between trailing-3m Sharpe and trailing-12m Sharpe. Penalises regime
    changers (positive or negative). penalty = 1/(1 + |gap|)."""
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < lookback:
            continue
        w12 = panel.iloc[pos - lookback + 1: pos + 1][sleeves]
        w3 = panel.iloc[max(0, pos - 63 + 1): pos + 1][sleeves]
        f = np.array([kelly_fraction(w12[s]) for s in sleeves])
        f = f * frac
        # Compute Sharpe gap
        gap = np.array([abs(trailing_sharpe(w3[s]) - trailing_sharpe(w12[s])) for s in sleeves])
        penalty = 1.0 / (1.0 + gap)
        f = f * penalty
        f = np.minimum(f, cap)
        f = np.maximum(f, 0.0)
        tot = f.sum()
        if tot < 1e-9:
            w = np.ones(len(sleeves)) / len(sleeves)
        else:
            w = f / tot
        weights.iloc[pos] = w
    return weights


def variant_rp_blend(panel: pd.DataFrame, sleeves: list[str],
                      lookback: int = 60) -> pd.DataFrame:
    """50% EW + 50% inverse-vol risk parity (monthly rebalanced)."""
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    n = len(sleeves)
    for me, pos in month_end_iterator(panel.index):
        if pos < lookback:
            continue
        window = panel.iloc[pos - lookback + 1: pos + 1][sleeves]
        vols = window.std(ddof=0).values * SQRT_ANN
        safe = np.where(vols > 1e-9, vols, np.inf)
        inv = 1.0 / safe
        inv = np.where(np.isfinite(inv), inv, 0.0)
        if inv.sum() < 1e-9:
            rp = np.ones(n) / n
        else:
            rp = inv / inv.sum()
        ew = np.ones(n) / n
        w = 0.5 * ew + 0.5 * rp
        weights.iloc[pos] = w
    return weights


def variant_equal_weight(panel: pd.DataFrame, sleeves: list[str]) -> pd.DataFrame:
    """EW baseline as a monthly-rebalanced DataFrame of weights."""
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    n = len(sleeves)
    for me, pos in month_end_iterator(panel.index):
        weights.iloc[pos] = np.ones(n) / n
    return weights


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def evaluate(name: str, port: pd.Series) -> dict:
    """Apply vol-target + DD control overlays then compute IS/OOS stats."""
    vt, _ = vol_target_overlay(port, target_vol=TARGET_VOL, max_lev=MAX_LEV)
    dd_ctrl, _ = drawdown_control(vt)

    out = {"variant": name}
    for tag, mask in [("IS", dd_ctrl.index < SPLIT),
                      ("OOS", dd_ctrl.index >= SPLIT)]:
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
    return out, dd_ctrl


def concentration_stats(weights: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    """Average max-weight and average HHI over the OOS slice (active month-ends)."""
    sub = weights[mask]
    active = sub.loc[(sub.abs().sum(axis=1) > 1e-9)]
    if len(active) == 0:
        return 0.0, 0.0
    return float(active.max(axis=1).mean()), float((active ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    panel_all = pd.read_parquet(QUANT / "all_sleeve_returns_v11.parquet")
    panel_all.index = pd.to_datetime(panel_all.index, utc=True)

    # Restrict to TOP15 sleeves and apply v11 regime gates + fast decay tripwire.
    high_vol, _ = build_regime_mask(panel_all.index, 80)
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME"]:
        gates[s] = 1.5
    gates["VOLFORECAST"] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["SESSION_MOM"] = 1.0
    gates["MULTI_CONFIRM"] = 0.7

    g = panel_all.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    after_decay, _ = fast_decay_tripwire(g[TOP15], TOP15)
    panel = after_decay[TOP15].copy()

    # Build all variants
    variants_def = {
        "EW_baseline":        ("ew",   {}),
        "Quarter_Kelly_15cap": ("qk",   {"frac": 0.25, "cap": 0.15}),
        "Eighth_Kelly_10cap":  ("qk",   {"frac": 0.125, "cap": 0.10}),
        "VolOfKelly_damped":   ("vok",  {"frac": 0.25, "cap": 0.20, "vol_window": 6}),
        "Bucket_Kelly":        ("bk",   {"frac": 0.25, "cap": 0.35}),
        "Kelly_DD_lockout":    ("ko",   {"frac": 0.25, "cap": 0.20, "dd_trigger": -0.05}),
        "Kelly_EdgeDecay":     ("ed",   {"frac": 0.25, "cap": 0.20}),
        "RP_EW_blend":         ("rp",   {}),
    }

    # Additional sensitivity probes — same 6 ideas at lighter doses or
    # different parameterisations, plus EW-anchored blends. None of these
    # change the recommendation but they verify there isn't a sweet spot.
    sensitivity_def = {
        "QK_75EW_blend":      None,  # 75% EW + 25% Quarter_Kelly
        "Bucket_50EW_blend":  None,
        "VoK_lighter":        ("vok", {"frac": 0.125, "cap": 0.10, "vol_window": 6}),
        "Kelly_lockout_3pct": ("ko",  {"frac": 0.25, "cap": 0.20, "dd_trigger": -0.03}),
    }

    builder = {
        "ew":  variant_equal_weight,
        "qk":  variant_quarter_kelly_capped,
        "vok": variant_vol_of_kelly,
        "bk":  variant_bucket_kelly,
        "ko":  variant_kelly_lockout,
        "ed":  variant_edge_decay_penalty,
        "rp":  variant_rp_blend,
    }

    daily_returns = pd.DataFrame(index=panel.index)
    rows = []
    oos_mask = panel.index >= SPLIT
    weights_cache: dict[str, pd.DataFrame] = {}

    for name, (key, kwargs) in variants_def.items():
        if key == "ew":
            weights = variant_equal_weight(panel, TOP15)
        else:
            weights = builder[key](panel, TOP15, **kwargs)
        weights_cache[name] = weights
        port = apply_weights_to_panel(panel, weights)
        stats_, dd_ctrl_ret = evaluate(name, port)
        max_w, hhi = concentration_stats(weights, oos_mask)
        stats_["OOS_avg_maxweight"] = max_w
        stats_["OOS_avg_HHI"] = hhi
        rows.append(stats_)
        daily_returns[name] = dd_ctrl_ret
        print(f"{name:<24}  IS Sh={stats_.get('IS_sharpe',0):+.2f}  "
              f"OOS Sh={stats_.get('OOS_sharpe',0):+.2f}  "
              f"OOS DD={stats_.get('OOS_maxdd',0):+.1%}  "
              f"maxW={max_w:.2%}  HHI={hhi:.3f}")

    # ---- Sensitivity probes ----
    print("\n=== Sensitivity probes ===")
    ew_weights = weights_cache["EW_baseline"]
    for name, spec in sensitivity_def.items():
        if spec is None:
            if name == "QK_75EW_blend":
                base = weights_cache["Quarter_Kelly_15cap"]
                weights = 0.25 * base + 0.75 * ew_weights
            elif name == "Bucket_50EW_blend":
                base = weights_cache["Bucket_Kelly"]
                weights = 0.5 * base + 0.5 * ew_weights
            else:
                continue
        else:
            key, kwargs = spec
            weights = builder[key](panel, TOP15, **kwargs)
        weights_cache[name] = weights
        port = apply_weights_to_panel(panel, weights)
        stats_, dd_ctrl_ret = evaluate(name, port)
        max_w, hhi = concentration_stats(weights, oos_mask)
        stats_["OOS_avg_maxweight"] = max_w
        stats_["OOS_avg_HHI"] = hhi
        rows.append(stats_)
        daily_returns[name] = dd_ctrl_ret
        print(f"{name:<24}  IS Sh={stats_.get('IS_sharpe',0):+.2f}  "
              f"OOS Sh={stats_.get('OOS_sharpe',0):+.2f}  "
              f"OOS DD={stats_.get('OOS_maxdd',0):+.1%}  "
              f"maxW={max_w:.2%}  HHI={hhi:.3f}")

    # Persist outputs
    daily_returns.to_parquet(OUT / "kelly_v2_returns.parquet")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "kelly_v2_variants.csv", index=False)

    # ---- Constraint check ----
    ew = summary[summary["variant"] == "EW_baseline"].iloc[0]
    ew_sh = ew["OOS_sharpe"]
    ew_dd = ew["OOS_maxdd"]
    print(f"\nEW OOS Sharpe={ew_sh:+.3f}  OOS MaxDD={ew_dd:+.2%}")
    print(f"Target: OOS Sharpe >= {ew_sh+0.10:+.3f}  AND  OOS MaxDD >= {ew_dd*1.10:+.2%} (10% relative)")

    print("\n=== Constraint scan ===")
    print(f"{'variant':<24} {'dSh':>7} {'dDD_rel':>10} {'pass':>6}")
    print("-" * 50)
    for r in rows:
        if r["variant"] == "EW_baseline":
            continue
        dsh = r["OOS_sharpe"] - ew_sh
        # MaxDD is negative; "relative increase" = (|dd_var| - |dd_ew|) / |dd_ew|
        dd_rel = (abs(r["OOS_maxdd"]) - abs(ew_dd)) / abs(ew_dd) if abs(ew_dd) > 1e-9 else 0
        passed = (dsh >= 0.10) and (dd_rel <= 0.10)
        print(f"{r['variant']:<24} {dsh:>+7.3f} {dd_rel:>+10.2%} {('YES' if passed else 'no'):>6}")

    print(f"\nSaved: {OUT/'kelly_v2_returns.parquet'}")
    print(f"Saved: {OUT/'kelly_v2_variants.csv'}")


if __name__ == "__main__":
    main()
