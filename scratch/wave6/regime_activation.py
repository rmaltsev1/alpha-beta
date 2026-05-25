"""Wave 6 — Regime-conditional sleeve activation/deactivation.

For each sleeve, compute IS Sharpe in each of 5 regimes (Crisis,
Calm-uptrend, Choppy, Vol-expansion, Vol-compression). "Favorable"
regimes have IS Sharpe >= 0.5.

Activation rules:
  A. Binary on/off (zero in unfavorable).
  B. Half-weight in unfavorable.
  C. 1.5x boost in favorable.

Compare against:
  - Equal-weight TOP22 raw
  - Equal-weight TOP22 + simple regime gate (current production)

IS <= 2024-01-01 ; OOS >= 2024-01-01.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

OUT = ROOT / "scratch" / "wave6"
QUANT = ROOT / "scratch" / "quant"

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

# Same TOP22 + GATES as master_v14
TOP22 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
         "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
         "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
         "CORR_REGIME", "SESSION_MOM",
         "W1_STRATS", "EVENT_VOLSPIKE",
         "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
         "TERM_SPREADS", "EURGBP_MR", "MULTIDAY"]

GATES_HIGH = {
    "RISKPAR": 0.5, "XSMOM": 0.5, "EVE_XAU": 0.5, "WED_BTC": 0.5,
    "D1REV_NAS": 1.5, "D1REV_UK": 1.5, "DEFEND": 1.5,
    "PAIRS_EXP": 1.5, "CRYPTO_vs_SPX": 1.5, "CORR_REGIME": 1.5,
    "STATARB_XS": 1.5, "MICROSTR_D1": 1.5, "EURGBP_MR": 1.5,
    "TERM_SPREADS": 1.5, "EVENT_VOLSPIKE": 1.5, "MULTIDAY": 1.5,
    "VOLFORECAST": 1.0, "W1_STRATS": 1.0, "SESSION_MOM": 1.0,
    "H4_SLEEVE": 1.5, "TREND_NEW": 0.5, "VOL_BREAKOUT": 1.2,
}


def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def sharpe(r):
    if len(r) < 2 or r.std() == 0:
        return 0.0
    bpy = _bpy(r.index)
    return float(r.mean() * bpy / (r.std(ddof=0) * np.sqrt(bpy)))


def stats(label, r):
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2:
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


# ---------------- Regime classification ----------------

def compute_spx_features(panel_index):
    """Compute SPX vol, 21d return, vol of 30d ago for regime classification.
    Returns DataFrame indexed by panel_index."""
    spx = pd.read_parquet(ROOT / "data" / "SPX500_USD" / "D1.parquet")
    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)
    spx = spx.sort_values("timestamp").reset_index(drop=True)
    spx["logret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["logret"].rolling(30).std()
    spx["ret21"] = spx["close"] / spx["close"].shift(21) - 1
    spx["rv30_lag30"] = spx["rv30"].shift(30)
    spx["vol_ratio"] = spx["rv30"] / spx["rv30_lag30"]

    df = pd.DataFrame({"timestamp": panel_index}).sort_values("timestamp")
    aligned = pd.merge_asof(
        df, spx[["timestamp", "rv30", "ret21", "rv30_lag30", "vol_ratio"]],
        on="timestamp", direction="backward",
    )
    aligned.index = aligned["timestamp"]
    return aligned[["rv30", "ret21", "rv30_lag30", "vol_ratio"]]


def classify_regimes(feat, is_mask):
    """Return DataFrame of booleans for each regime."""
    is_rv30 = feat.loc[is_mask, "rv30"].dropna()
    p80 = float(is_rv30.quantile(0.80))
    p40 = float(is_rv30.quantile(0.40))
    p60 = float(is_rv30.quantile(0.60))
    p20 = float(is_rv30.quantile(0.20))

    crisis = (feat["rv30"] > p80) & (feat["ret21"] < -0.03)
    calm_up = (feat["rv30"] < p40) & (feat["ret21"] > 0.02)
    choppy = (feat["rv30"] >= p40) & (feat["rv30"] <= p60) & (feat["ret21"].abs() < 0.01)
    vol_exp = feat["vol_ratio"] > 1.3
    vol_comp = feat["rv30"] < p20

    regimes = pd.DataFrame({
        "crisis": crisis.fillna(False),
        "calm_uptrend": calm_up.fillna(False),
        "choppy": choppy.fillna(False),
        "vol_expansion": vol_exp.fillna(False),
        "vol_compression": vol_comp.fillna(False),
    }, index=feat.index)
    cutoffs = {"p20": p20, "p40": p40, "p60": p60, "p80": p80}
    return regimes, cutoffs


# ---------------- Per-sleeve profiling ----------------

def per_sleeve_regime_profile(panel, regimes, is_mask):
    """For each sleeve x regime, compute IS Sharpe."""
    rows = []
    for s in panel.columns:
        r = panel[s]
        row = {"sleeve": s,
               "all_IS_sharpe": sharpe(r[is_mask])}
        for reg in regimes.columns:
            mask = is_mask & regimes[reg]
            n = int(mask.sum())
            sub = r[mask]
            sh = sharpe(sub) if n >= 20 else np.nan
            row[f"{reg}_sharpe"] = sh
            row[f"{reg}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def favorable_regimes_map(profile, sleeves, threshold=0.5):
    """For each sleeve, return set of favorable regime names (IS Sharpe >= threshold)."""
    out = {}
    regime_cols = ["crisis", "calm_uptrend", "choppy", "vol_expansion", "vol_compression"]
    for _, row in profile.iterrows():
        s = row["sleeve"]
        if s not in sleeves:
            continue
        favs = set()
        for reg in regime_cols:
            sh = row.get(f"{reg}_sharpe", np.nan)
            if pd.notna(sh) and sh >= threshold:
                favs.add(reg)
        out[s] = favs
    return out


# ---------------- Apply activation rules ----------------

def build_weight_matrix(panel, sleeves, regimes, fav_map, rule):
    """Build weight matrix of same shape as panel[sleeves].

    Rule A: 1.0 if any favorable regime active, else 0.0.
    Rule B: 1.0 if favorable, else 0.5.
    Rule C: 1.5 if favorable, else 1.0.
    """
    W = pd.DataFrame(1.0, index=panel.index, columns=sleeves)
    # is_fav[t,s] = True if at time t at least one of sleeve s's favorable regimes is active
    for s in sleeves:
        favs = fav_map.get(s, set())
        if not favs:
            # No favorable regime: treat all as unfavorable (default rule behavior)
            is_fav = pd.Series(False, index=panel.index)
        else:
            is_fav = regimes[list(favs)].any(axis=1)
        if rule == "A":
            W[s] = is_fav.astype(float)
        elif rule == "B":
            W[s] = np.where(is_fav, 1.0, 0.5)
        elif rule == "C":
            W[s] = np.where(is_fav, 1.5, 1.0)
        else:
            raise ValueError(rule)
    return W


def portfolio_returns(panel, sleeves, W):
    """Equal-weight average across sleeves, modulated by W (per-bar per-sleeve)."""
    contributions = panel[sleeves] * W
    # Equal weight: divide by number of sleeves (not by sum of weights, to keep
    # the deactivation effect — if 10/22 sleeves are off, allocation halves).
    return contributions.sum(axis=1) / len(sleeves)


def simple_regime_gate_returns(panel, sleeves, regimes_high_vol):
    """Equal-weight TOP22 with the master_v14 simple high-vol regime gate."""
    g = panel[sleeves].copy()
    for s, m in GATES_HIGH.items():
        if s in g.columns:
            g.loc[regimes_high_vol, s] *= m
    return g.mean(axis=1)


def main():
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v14.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    print(f"Panel: {panel.shape}, range {panel.index.min()} -> {panel.index.max()}")
    is_mask = panel.index < SPLIT
    oos_mask = panel.index >= SPLIT
    print(f"IS rows: {int(is_mask.sum())}  OOS rows: {int(oos_mask.sum())}")

    # Sleeves present
    sleeves = [s for s in TOP22 if s in panel.columns]
    print(f"Sleeves in TOP22: {len(sleeves)} / 22")

    # SPX features
    feat = compute_spx_features(panel.index)

    # Cross-asset avg correlation (computed from panel itself, not used for
    # regime def per the 5-regime spec, but exposed for diagnostics).
    rolling_corr = (
        panel[sleeves].rolling(63, min_periods=20)
        .corr().groupby(level=0).mean().mean(axis=1)
    )
    # Note: rolling_corr is informational only.

    # 30d SPX trend direction is captured by ret21 (close-to-close 21d return).

    # Regime classification (IS-fit cutoffs)
    regimes, cutoffs = classify_regimes(feat, is_mask)
    print(f"\nIS-fit cutoffs (SPX 30d rv): "
          f"p20={cutoffs['p20']:.5f}  p40={cutoffs['p40']:.5f}  "
          f"p60={cutoffs['p60']:.5f}  p80={cutoffs['p80']:.5f}")
    print(f"\nRegime occupancy:")
    for c in regimes.columns:
        n_is = int((regimes[c] & is_mask).sum())
        n_oos = int((regimes[c] & oos_mask).sum())
        print(f"  {c:<18}  IS={n_is:>5}  OOS={n_oos:>5}")

    # --- Per-sleeve profile (all 33 sleeves) ---
    profile = per_sleeve_regime_profile(panel, regimes, is_mask)
    profile.to_csv(OUT / "regime_profiles.csv", index=False)
    print(f"\nSaved regime_profiles.csv ({len(profile)} sleeves)")

    # --- Favorable regimes per sleeve ---
    fav_map = favorable_regimes_map(profile, sleeves, threshold=0.5)

    # Diagnostic: show favorable regimes for TOP22
    print(f"\nFavorable regimes (IS Sh >= 0.5) per TOP22 sleeve:")
    for s in sleeves:
        favs = fav_map.get(s, set())
        favs_str = ", ".join(sorted(favs)) if favs else "(none)"
        # Pull IS sharpes per regime for readability
        row = profile[profile["sleeve"] == s].iloc[0]
        per = {r: row.get(f"{r}_sharpe", np.nan) for r in regimes.columns}
        per_str = " ".join([f"{k[:3]}={(v if pd.notna(v) else 0):+.2f}" for k, v in per.items()])
        print(f"  {s:<16}  {per_str}   -> [{favs_str}]")

    # --- Build portfolio under each rule ---
    out_returns = {}
    rows = []

    # Baselines
    raw_ew = panel[sleeves].mean(axis=1)
    out_returns["ew_top22_raw"] = raw_ew
    rows.append(stats("EW TOP22 raw", raw_ew))

    # Simple regime gate (current production approach)
    # build high-vol mask from IS-fit p80
    high_vol = (feat["rv30"] > cutoffs["p80"]).fillna(False)
    high_vol.index = panel.index
    simple_gate = simple_regime_gate_returns(panel, sleeves, high_vol)
    out_returns["ew_top22_simple_regime_gate"] = simple_gate
    rows.append(stats("EW TOP22 + simple regime gate", simple_gate))

    for rule in ["A", "B", "C"]:
        W = build_weight_matrix(panel, sleeves, regimes, fav_map, rule)
        port = portfolio_returns(panel, sleeves, W)
        key = f"regime_act_rule_{rule}"
        out_returns[key] = port
        rows.append(stats(f"Regime-activation Rule {rule}", port))

    # --- Print summary table ---
    print(f"\n{'Variant':<38} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>6} {'OOS_DD':>8}")
    print("-" * 80)
    summary_rows = []
    for r_label, r_series in out_returns.items():
        s = stats(r_label, r_series)
        y22 = r_series[r_series.index.year == 2022]
        bpy22 = _bpy(y22.index) if len(y22) > 1 else 252
        sh22 = (y22.mean() * bpy22) / (y22.std(ddof=0) * np.sqrt(bpy22)) if y22.std() > 0 else 0
        print(f"{r_label:<38} {s.get('FULL_sharpe',0):>+6.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+6.2f} {sh22:>+6.2f} {s.get('OOS_dd',0):>+8.1%}")
        summary_rows.append({
            "variant": r_label,
            "FULL_sharpe": s.get("FULL_sharpe", 0),
            "IS_sharpe": s.get("IS_sharpe", 0),
            "OOS_sharpe": s.get("OOS_sharpe", 0),
            "Y2022_sharpe": sh22,
            "OOS_dd": s.get("OOS_dd", 0),
            "OOS_vol": s.get("OOS_vol", 0),
            "OOS_ret": s.get("OOS_ret", 0),
            "FULL_dd": s.get("FULL_dd", 0),
        })

    pd.DataFrame(summary_rows).to_csv(OUT / "regime_act_summary.csv", index=False)

    # Save returns
    out_df = pd.DataFrame(out_returns)
    out_df.to_parquet(OUT / "regime_act_returns.parquet")
    print(f"\nSaved regime_act_returns.parquet ({out_df.shape})")
    print(f"Saved regime_profiles.csv")
    print(f"Saved regime_act_summary.csv")

    # ---- Sleeve-level regime dependence rank ----
    print(f"\nMost regime-dependent sleeves (IS Sh max - min across regimes with n>=20):")
    rd_rows = []
    for _, row in profile.iterrows():
        s = row["sleeve"]
        if s not in sleeves:
            continue
        per = []
        for reg in regimes.columns:
            sh = row.get(f"{reg}_sharpe", np.nan)
            if pd.notna(sh):
                per.append(sh)
        if len(per) >= 2:
            rd_rows.append({"sleeve": s, "range": max(per) - min(per),
                            "max": max(per), "min": min(per),
                            "all_IS": row["all_IS_sharpe"]})
    rd_df = pd.DataFrame(rd_rows).sort_values("range", ascending=False)
    print(rd_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
