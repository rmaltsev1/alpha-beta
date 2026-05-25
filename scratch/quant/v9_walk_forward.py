"""Walk-forward validation of master_v9.

Each rebalance month, select sleeves using only trailing-12m Sharpe.
Compare static-TOP12 (hindsight) to WF_GATE_POS (production rule).
This is the honest test of v9.
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
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")


def yearly(r):
    rows = {}
    for year, sub in r.groupby(r.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60: continue
        bpy = len(sub) / span * 365.25
        rows[year] = sub.mean()*bpy / (sub.std(ddof=0)*np.sqrt(bpy)) if sub.std() > 0 else 0
    return pd.Series(rows)


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v9.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    TOP12 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX"]
    # All candidate sleeves for WF selection (broader pool)
    ALL_SLEEVES = list(panel.columns)
    print(f"All sleeves available: {ALL_SLEEVES}")

    gates = dict(GATES_HIGH)
    gates["PAIRS_EXP"] = 1.5
    gates["VOLFORECAST"] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["CRYPTO_vs_SPX"] = 1.5
    high_vol, _ = build_regime_mask(panel.index, 80)
    g = panel.copy()
    for sleeve, mult in gates.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult

    # Walk-forward sleeve selection — monthly rebalance, trailing 12m Sharpe
    # Pick all sleeves with trailing-12m Sharpe > 0.5 (WF_GATE_HIGH variant)
    bars = panel.index
    rebalance_dates = pd.date_range(bars[0].normalize(), bars[-1].normalize(),
                                     freq="ME", tz="UTC")
    weights_log = []
    portfolio_returns = []

    sleeve_selection = ALL_SLEEVES  # all 12+ sleeves
    target_n_min = 5  # at least 5 sleeves to avoid concentration

    for me in rebalance_dates:
        idx_pos = bars.searchsorted(me)
        if idx_pos < 252:  # need 252d history
            continue
        window = g.iloc[idx_pos - 252 : idx_pos]
        # Trailing 12m sharpes
        sharpes = {}
        for c in sleeve_selection:
            r = window[c]
            sh = r.mean()*365 / (r.std(ddof=0)*np.sqrt(365)) if r.std() > 0 else 0
            sharpes[c] = sh
        # Two variants: gate at 0.3 and top-N
        s_series = pd.Series(sharpes)
        # WF_GATE_POS: all positive trailing Sharpe
        gate_pos = s_series[s_series > 0]
        # WF_TOP8: top 8 by sharpe
        top8 = s_series.nlargest(8)
        # Use these on the next month's bars
        next_me = me + pd.offsets.MonthEnd(1)
        next_pos = bars.searchsorted(next_me)
        out_window = g.iloc[idx_pos:next_pos]
        if len(out_window) == 0:
            continue
        # Apply weights
        ret_gate_pos = out_window[list(gate_pos.index)].mean(axis=1) if len(gate_pos) > 0 else pd.Series(0, index=out_window.index)
        ret_top8 = out_window[list(top8.index)].mean(axis=1) if len(top8) > 0 else pd.Series(0, index=out_window.index)
        # Save log
        for sleeve, sh in s_series.items():
            weights_log.append({"month": me, "sleeve": sleeve, "trailing_sharpe": sh,
                                "in_gate_pos": sleeve in gate_pos.index,
                                "in_top8": sleeve in top8.index})
        portfolio_returns.append(pd.DataFrame({
            "WF_GATE_POS": ret_gate_pos,
            "WF_TOP8": ret_top8,
        }))

    wf_df = pd.concat(portfolio_returns)
    pd.DataFrame(weights_log).to_csv(OUT / "v9_wf_weights.csv", index=False)

    # Static benchmark
    static_top12 = g[TOP12].mean(axis=1)

    # Stats
    variants = {
        "STATIC_TOP12 (hindsight)": static_top12,
        "WF_GATE_POS (no look-ahead)": wf_df["WF_GATE_POS"],
        "WF_TOP8 (no look-ahead)": wf_df["WF_TOP8"],
    }

    print(f"\n{'Variant':<30} {'FULL':>6} {'IS':>6} {'OOS':>7} {'2022':>7} {'OOS_DD':>8}")
    for name, r in variants.items():
        s = stats(name, r)
        y22 = r[r.index.year == 2022]
        bpy = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy)/(y22.std(ddof=0)*np.sqrt(bpy)) if y22.std() > 0 else 0
        print(f"{name:<30} {s.get('FULL_sharpe',0):>+6.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+7.2f} {sh22:>+7.2f} {s.get('OOS_dd',0):>+8.1%}")

    print(f"\nYear-by-year Sharpe:")
    yr = {n: yearly(r) for n, r in variants.items()}
    print(pd.DataFrame(yr).round(2).T.to_string())

    # Most-selected sleeves
    print(f"\nMost-frequently-selected sleeves under WF_GATE_POS:")
    wlog = pd.DataFrame(weights_log)
    selection_pct = wlog.groupby("sleeve")["in_gate_pos"].mean().sort_values(ascending=False)
    for s, pct in selection_pct.items():
        print(f"  {s:<14} {pct:.1%}")


if __name__ == "__main__":
    main()
