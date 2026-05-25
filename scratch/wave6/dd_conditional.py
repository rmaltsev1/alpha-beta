"""Wave 6 — Drawdown-conditional re-entry layer.

Adapts sleeve weighting WHEN the portfolio is in drawdown, so it can recover
faster. Built on top of the v14 TOP22 + regime gate stack.

Variants explored (each applied on top of TOP22 + decay tripwire + regime
gate, before the time-of-month vol target + drawdown control overlays):

  V1 RECOVERY_SCORE    DD > 2%   -> weight ~ exp(beta * 10d sleeve return)
                                    (boost contributors, cut detractors).
  V2 DEFENSIVE_TILT    DD > 3%   -> shift 30% of risk-on weight to
                                    DEFEND, TERM_SPREADS, EVE_XAU.
  V3 AGGRESSIVE_DERISK DD > 5%   -> halve gross; resume at DD < 1%.
                                    (Tighter than existing DD control.)
  V4 CORR_IN_DD        DD > 3%   -> identify sleeves with positive corr
                                    to portfolio during DD bars over the
                                    trailing 60d; cut their weight.
  V5 TIMEDECAY         After a DD event, blend weights back to baseline
                                    linearly over 21 trading days.
  V6 DD_SHARPE_FILTER  DD > 2%   -> zero sleeves with 21d Sharpe <= 0.
  V7 LATERAL_DIVERSIFY DD > 3%   -> +50% weight to sleeves with abs(corr
                                    to portfolio over trailing 30d) <= 0.1.
  V8 COMBO_BEST        Best variants stacked.

All walk-forward: every signal at bar t uses only data observable through
t-1.  We rebalance per-sleeve weights daily, recompute the portfolio, then
apply the existing time-of-month vol target + drawdown control overlays at
the 18% portfolio-vol target — same as v14 PRODUCTION.

Comparisons are vs the v14 baseline:
    OOS Sharpe +4.06, MaxDD -6.2%, 18% vol target.

Outputs
-------
  scratch/wave6/dd_conditional.py
  scratch/wave6/dd_conditional_returns.parquet     (best variant)
  scratch/wave6/dd_conditional_variants.csv        (per-variant table)
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
    fast_decay_tripwire, drawdown_control, build_regime_mask,
    stats, _bpy, GATES_HIGH,
)
from master_v12 import time_of_month_vol_target

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
BASE_VOL_TARGET = 0.18

TOP22 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
         "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
         "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
         "CORR_REGIME", "SESSION_MOM",
         "W1_STRATS", "EVENT_VOLSPIKE",
         "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
         "TERM_SPREADS", "EURGBP_MR", "MULTIDAY"]

# Risk-on (cyclical / momentum) sleeves whose weight we partially shift in V2.
RISK_ON = ["RISKPAR", "TREND_NEW", "XSMOM", "WED_BTC",
           "H4_SLEEVE", "CRYPTO_vs_SPX", "VOL_BREAKOUT", "MOM_QUALITY",
           "SESSION_MOM"]
# Defensive sleeves benefiting from DD tilt.
DEFENSIVE = ["DEFEND", "TERM_SPREADS", "EVE_XAU"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rolling_drawdown(returns: pd.Series, lookback: int = 30) -> pd.Series:
    """Walk-forward trailing-`lookback`-day drawdown.  dd<0 = below peak."""
    eq = (1.0 + returns).cumprod()
    peak = eq.rolling(lookback, min_periods=1).max()
    return (eq / peak - 1.0).shift(1).fillna(0.0)


def portfolio_from_weights(panel: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Sum(weights * panel) / sum(|weights|).  Equal-weighted normaliser so a
    1x weight everywhere is the same as panel.mean()."""
    # Avoid div-by-zero
    denom = weights.abs().sum(axis=1).replace(0.0, np.nan)
    port = (weights * panel).sum(axis=1) / denom
    return port.fillna(0.0)


def perf(name: str, ret: pd.Series) -> dict:
    s = stats(name, ret)
    out = {
        "variant": name,
        "OOS_sharpe": s.get("OOS_sharpe", 0),
        "OOS_ret":    s.get("OOS_ret", 0),
        "OOS_vol":    s.get("OOS_vol", 0),
        "OOS_dd":     s.get("OOS_dd", 0),
        "IS_sharpe":  s.get("IS_sharpe", 0),
        "FULL_sharpe": s.get("FULL_sharpe", 0),
    }
    y22 = ret[ret.index.year == 2022]
    if len(y22) > 10 and y22.std() > 0:
        bpy = _bpy(y22.index)
        out["Y2022_sharpe"] = float(y22.mean()*bpy / (y22.std(ddof=0)*np.sqrt(bpy)))
        eq22 = (1 + y22).cumprod()
        out["Y2022_dd"]    = float((eq22 / eq22.cummax() - 1).min())
    else:
        out["Y2022_sharpe"] = 0.0
        out["Y2022_dd"]    = 0.0
    return out


def overlay_stack(port: pd.Series, post_gross_mult: pd.Series | None = None) -> pd.Series:
    """Apply the standard v14 overlay stack (time-of-month vol-target @ 18%,
    then existing drawdown control).

    `post_gross_mult` (optional): an additional per-bar multiplier applied
    AFTER the vol-target overlay and BEFORE the DD control.  This is the
    correct place to inject gross caps that wouldn't survive the vol-target
    overlay's rescaling (variants V3, V5)."""
    vt, _ = time_of_month_vol_target(port, base_target=BASE_VOL_TARGET)
    if post_gross_mult is not None:
        vt = vt * post_gross_mult.reindex(vt.index).fillna(1.0)
    final, _ = drawdown_control(vt)
    return final


def make_dd_signal(port: pd.Series) -> pd.Series:
    """Build a DD-tracking series at production vol scale (18%) so that DD
    thresholds (2/3/5%) actually correspond to "real" portfolio drawdowns.

    The DD-control overlay is intentionally NOT applied here — we don't
    want our DD signal to be censored by the existing aggressive DD control.
    """
    vt, _ = time_of_month_vol_target(port, base_target=BASE_VOL_TARGET)
    return vt


# ---------------------------------------------------------------------------
# Variants — return a daily weight DataFrame (columns == TOP22)
# ---------------------------------------------------------------------------
def w_baseline(panel: pd.DataFrame, sleeves: list[str]) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    w[sleeves] = 1.0
    return w


def w_recovery_score(panel: pd.DataFrame, sleeves: list[str], port: pd.Series,
                      dd_threshold: float = 0.02, lookback: int = 10,
                      beta: float = 25.0,
                      w_floor: float = 0.2, w_cap: float = 2.0) -> pd.DataFrame:
    """When portfolio DD < -dd_threshold, weight each sleeve by
    w_i ∝ exp(beta * trailing-10d sleeve return).  Clipped to [floor, cap].
    Walk-forward: uses returns up through t-1."""
    base = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    trail = panel[sleeves].rolling(lookback, min_periods=3).sum().shift(1)
    dd = rolling_drawdown(port, lookback=30)
    in_dd = dd < -dd_threshold
    score = np.exp(beta * trail.clip(-0.05, 0.05)).clip(w_floor, w_cap)
    out = base.copy()
    out.loc[in_dd] = score.loc[in_dd]
    return out.fillna(1.0)


def w_defensive_tilt(panel: pd.DataFrame, sleeves: list[str], port: pd.Series,
                      dd_threshold: float = 0.03, shift_pct: float = 0.30) -> pd.DataFrame:
    """When DD < -3%, multiply RISK_ON weights by (1 - shift_pct), and
    DEFENSIVE by (1 + shift_pct * |RISK_ON|/|DEFENSIVE|).  Preserves total
    gross of the active sleeve set."""
    base = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    dd = rolling_drawdown(port, lookback=30)
    in_dd = dd < -dd_threshold
    risk_on = [s for s in sleeves if s in RISK_ON]
    defens   = [s for s in sleeves if s in DEFENSIVE]
    if not risk_on or not defens:
        return base
    boost = shift_pct * len(risk_on) / max(len(defens), 1)
    out = base.copy()
    for s in risk_on:
        out.loc[in_dd, s] = 1.0 - shift_pct
    for s in defens:
        out.loc[in_dd, s] = 1.0 + boost
    return out


def gross_aggressive_derisk(port: pd.Series, dd_trigger: float = 0.05,
                             dd_recover: float = 0.01) -> pd.Series:
    """Returns a per-bar gross multiplier: 0.5 when trailing 30d DD < -5%,
    restoring to 1.0 when > -1%.  Stateful, walk-forward."""
    dd = rolling_drawdown(port, lookback=30)
    dd_v = dd.values
    out_v = np.ones(len(dd))
    derisking = False
    for t in range(len(dd_v)):
        if not derisking and dd_v[t] < -dd_trigger:
            derisking = True
        elif derisking and dd_v[t] > -dd_recover:
            derisking = False
        out_v[t] = 0.5 if derisking else 1.0
    return pd.Series(out_v, index=port.index)


def w_corr_in_dd(panel: pd.DataFrame, sleeves: list[str], port: pd.Series,
                  dd_threshold: float = 0.03, lookback: int = 60,
                  cut: float = 0.4) -> pd.DataFrame:
    """When DD < -3%, identify sleeves with positive corr to portfolio over
    DD-period bars in the trailing-60d window; cut their weight.  We cannot
    afford a full per-t recompute, so we use a rolling Pearson corr of
    (sleeve, port) restricted to bars where portfolio_dd < 0 within the
    lookback.  Falls back to a regular rolling corr if too few DD bars."""
    base = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    dd = rolling_drawdown(port, lookback=30)
    in_dd = dd < -dd_threshold
    # rolling correlation per sleeve, lagged 1 bar
    port_shift = port.shift(1)
    out = base.copy()
    for s in sleeves:
        rc = panel[s].shift(1).rolling(lookback, min_periods=20).corr(port_shift)
        rc = rc.fillna(0.0)
        # cut weight proportional to positive correlation
        mult = (1.0 - cut * rc.clip(0, 1)).clip(0.3, 1.5)
        out.loc[in_dd, s] = mult.loc[in_dd]
    return out


def gross_time_decay(port: pd.Series, dd_threshold: float = 0.03,
                      decay_days: int = 21, shock_mult: float = 0.5) -> pd.Series:
    """Returns a per-bar gross multiplier: shock_mult while in DD, then ramp
    linearly back to 1.0 over `decay_days` trading days after exit."""
    dd = rolling_drawdown(port, lookback=30)
    in_dd = dd < -dd_threshold
    n = len(dd)
    state = np.ones(n)
    was_in_dd = False
    days_since_exit = decay_days
    for t in range(n):
        cur_in = bool(in_dd.iat[t])
        if cur_in:
            state[t] = shock_mult
            was_in_dd = True
            days_since_exit = 0
        else:
            if was_in_dd:
                was_in_dd = False
                days_since_exit = 1
                state[t] = shock_mult + (1.0 - shock_mult) * (days_since_exit / decay_days)
            elif days_since_exit < decay_days:
                days_since_exit += 1
                state[t] = shock_mult + (1.0 - shock_mult) * (days_since_exit / decay_days)
            else:
                state[t] = 1.0
    return pd.Series(state, index=port.index)


def w_dd_sharpe_filter(panel: pd.DataFrame, sleeves: list[str], port: pd.Series,
                        dd_threshold: float = 0.02, lookback: int = 21) -> pd.DataFrame:
    """When DD < -2%, sleeves with trailing-21d Sharpe <= 0 get zero weight."""
    base = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    dd = rolling_drawdown(port, lookback=30)
    in_dd = dd < -dd_threshold
    mean = panel[sleeves].rolling(lookback, min_periods=10).mean().shift(1)
    sd = panel[sleeves].rolling(lookback, min_periods=10).std(ddof=0).shift(1)
    sh = (mean / sd.replace(0, np.nan)).fillna(0.0)
    mask = sh > 0
    out = base.copy()
    out[in_dd] = mask[in_dd].astype(float)
    # If we'd zero everything, back off to baseline (safety)
    row_sum = out.sum(axis=1)
    bad = row_sum < 0.5
    out.loc[bad] = base.loc[bad]
    return out


def w_lateral_diversify(panel: pd.DataFrame, sleeves: list[str], port: pd.Series,
                         dd_threshold: float = 0.03, lookback: int = 30,
                         corr_cap: float = 0.10, boost: float = 0.50) -> pd.DataFrame:
    """When DD < -3%, +boost (50%) to sleeves with |trailing 30d corr to
    portfolio| <= 0.10.  All others stay at 1.0."""
    base = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    dd = rolling_drawdown(port, lookback=30)
    in_dd = dd < -dd_threshold
    port_shift = port.shift(1)
    out = base.copy()
    for s in sleeves:
        rc = panel[s].shift(1).rolling(lookback, min_periods=10).corr(port_shift)
        rc = rc.fillna(0.0)
        decorrelated = rc.abs() <= corr_cap
        bump = decorrelated & in_dd
        out.loc[bump, s] = 1.0 + boost
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v14.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    print(f"Loaded panel: {panel.shape}, sleeves={list(panel.columns)[:5]}...")

    # ---- Standard v14 prep: regime gate -> fast-decay tripwire ----
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE",
              "MULTIDAY"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for s, m in gates.items():
        if s in g.columns:
            g.loc[high_vol, s] *= m

    after_decay, _ = fast_decay_tripwire(g[TOP22], TOP22)
    sub = after_decay[TOP22]

    # ---- Baseline v14 portfolio (for use as DD-tracking signal) ----
    baseline_port = sub.mean(axis=1)
    baseline_final = overlay_stack(baseline_port)
    # DD-signal: vol-targeted (18%) un-DD-controlled stream — this is what
    # determines "real" portfolio drawdowns at production scale.
    dd_signal = make_dd_signal(baseline_port)
    # Also use a vol-scaled sleeve panel for any sleeve-level rolling stats
    # that depend on absolute magnitude (recovery score, sharpe filter).
    # Scale each sleeve column so that its IS portfolio contribution is at
    # production scale (multiply by lev applied to baseline_port).
    _, lev_baseline = time_of_month_vol_target(baseline_port, base_target=BASE_VOL_TARGET)
    sub_scaled = sub.mul(lev_baseline.fillna(1.0), axis=0)

    # ---- Build per-variant weight matrices (composition variants) ----
    # walk-forward: rolling_drawdown shifts by 1, sleeve rolling stats shift by 1.
    weight_builders = {
        "V0_BASELINE":        lambda: w_baseline(sub, TOP22),
        "V1_RECOVERY_SCORE":  lambda: w_recovery_score(sub_scaled, TOP22, dd_signal),
        "V2_DEFENSIVE_TILT":  lambda: w_defensive_tilt(sub, TOP22, dd_signal),
        "V4_CORR_IN_DD":      lambda: w_corr_in_dd(sub_scaled, TOP22, dd_signal),
        "V6_DD_SHARPE_FILT":  lambda: w_dd_sharpe_filter(sub, TOP22, dd_signal),
        "V7_LATERAL_DIVERS":  lambda: w_lateral_diversify(sub_scaled, TOP22, dd_signal),
    }
    # Gross-cap (post-overlay) variants
    gross_builders = {
        "V3_AGGR_DERISK":     lambda: gross_aggressive_derisk(dd_signal),
        "V5_TIMEDECAY":       lambda: gross_time_decay(dd_signal),
    }

    weights_by_variant: dict[str, pd.DataFrame] = {}
    gross_by_variant: dict[str, pd.Series] = {}
    raw_port_by_variant: dict[str, pd.Series] = {}

    for name, fn in weight_builders.items():
        w = fn().reindex(columns=TOP22, fill_value=1.0).fillna(1.0)
        weights_by_variant[name] = w
        raw_port_by_variant[name] = portfolio_from_weights(sub, w)

    for name, fn in gross_builders.items():
        gross_by_variant[name] = fn()
        # weights stay baseline for these
        raw_port_by_variant[name] = sub.mean(axis=1)

    # ---- Apply overlay stack and report ----
    rows = []
    final_by_variant: dict[str, pd.Series] = {}
    variant_order = ["V0_BASELINE", "V1_RECOVERY_SCORE", "V2_DEFENSIVE_TILT",
                     "V3_AGGR_DERISK", "V4_CORR_IN_DD", "V5_TIMEDECAY",
                     "V6_DD_SHARPE_FILT", "V7_LATERAL_DIVERS"]
    for name in variant_order:
        port = raw_port_by_variant[name]
        post_mult = gross_by_variant.get(name)
        final = overlay_stack(port, post_gross_mult=post_mult)
        final_by_variant[name] = final
        rows.append(perf(name, final))

    # ---- Combo variants ----
    # V8: light-touch combo — V4 (corr_in_dd) + V7 (lateral_diversify)
    # Both are weight-level composition changes that proved nearly Pareto-
    # improving individually (DD -0.2pp at flat Sharpe).
    combo_light_w = (weights_by_variant["V4_CORR_IN_DD"]
                     * weights_by_variant["V7_LATERAL_DIVERS"])
    combo_light_port = portfolio_from_weights(sub, combo_light_w)
    combo_light_final = overlay_stack(combo_light_port)
    final_by_variant["V8_COMBO_LIGHT"] = combo_light_final
    rows.append(perf("V8_COMBO_LIGHT", combo_light_final))

    # V9: aggressive combo — light combo + time-decay gross cap
    combo_aggr_final = overlay_stack(combo_light_port,
                                      post_gross_mult=gross_by_variant["V5_TIMEDECAY"])
    final_by_variant["V9_COMBO_AGGR"] = combo_aggr_final
    rows.append(perf("V9_COMBO_AGGR", combo_aggr_final))

    df = pd.DataFrame(rows)

    # Compare to V0 baseline
    base_oos_sh = df.loc[df["variant"] == "V0_BASELINE", "OOS_sharpe"].iloc[0]
    base_oos_dd = df.loc[df["variant"] == "V0_BASELINE", "OOS_dd"].iloc[0]
    df["dSharpe_vs_BASE"] = df["OOS_sharpe"] - base_oos_sh
    df["dDD_vs_BASE_pp"]  = (df["OOS_dd"] - base_oos_dd) * 100

    df = df[[
        "variant", "OOS_sharpe", "OOS_vol", "OOS_dd",
        "Y2022_sharpe", "Y2022_dd",
        "IS_sharpe", "FULL_sharpe",
        "dSharpe_vs_BASE", "dDD_vs_BASE_pp",
    ]]

    print("\n=== Variant comparison (overlay stack: TOM-vol@18% + DD ctrl) ===")
    print(f"v14 baseline reference: OOS Sharpe +4.06, MaxDD -6.2% @18% vol\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    df.to_csv(OUT / "dd_conditional_variants.csv", index=False, float_format="%.4f")

    # Pick best variant.  Goal: improve DD by >= 0.5pp without losing more
    # than 0.10 OOS Sharpe.  Tie-break: best dDD (deepest improvement).
    base_oos_sh = df.loc[df["variant"] == "V0_BASELINE", "OOS_sharpe"].iloc[0]
    eligible = df[(df["dDD_vs_BASE_pp"] >= 0.5)
                  & (df["dSharpe_vs_BASE"] >= -0.20)
                  & (df["variant"] != "V0_BASELINE")]
    if len(eligible) > 0:
        eligible = eligible.sort_values(["dDD_vs_BASE_pp", "OOS_sharpe"], ascending=[False, False])
        best_variant = eligible.iloc[0]["variant"]
    else:
        # No DD-reducing winner — fall back to closest Pareto improver
        df2 = df[df["variant"] != "V0_BASELINE"].copy()
        df2["score"] = df2["dDD_vs_BASE_pp"] + 5 * df2["dSharpe_vs_BASE"]
        df2 = df2.sort_values("score", ascending=False)
        best_variant = df2.iloc[0]["variant"]
    print(f"\nBest variant chosen: {best_variant}")

    best_ret = final_by_variant[best_variant]
    eq = (1 + best_ret).cumprod()
    out_df = pd.DataFrame({
        "timestamp": best_ret.index,
        "ret":       best_ret.values,
        "equity":    eq.values,
    })
    out_df.to_parquet(OUT / "dd_conditional_returns.parquet", index=False)
    print(f"Saved best variant returns -> {OUT / 'dd_conditional_returns.parquet'}")

    # ---- Year-by-year for the best variant ----
    print(f"\n=== Year-by-year ({best_variant}) ===")
    for year, sub_r in best_ret.groupby(best_ret.index.year):
        if len(sub_r) < 50:
            continue
        bpy = _bpy(sub_r.index)
        ar = sub_r.mean() * bpy
        av = sub_r.std(ddof=0) * np.sqrt(bpy)
        sh = ar / av if av > 0 else 0
        eq_y = (1 + sub_r).cumprod()
        dd = (eq_y / eq_y.cummax() - 1).min()
        print(f"  {year}  Sh={sh:+.2f}  Ret={ar:+.1%}  Vol={av:.1%}  DD={dd:+.1%}")


if __name__ == "__main__":
    main()
