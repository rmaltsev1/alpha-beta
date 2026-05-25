"""Walk-forward validation of v14 — honest production estimate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))
from master_v4 import build_regime_mask, GATES_HIGH, _bpy
from master_v12 import time_of_month_vol_target

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v14.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)

    TOP22 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
             "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
             "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
             "CORR_REGIME", "SESSION_MOM",
             "W1_STRATS", "EVENT_VOLSPIKE",
             "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
             "TERM_SPREADS", "EURGBP_MR", "MULTIDAY"]

    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE", "MULTIDAY"]:
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

    # Walk-forward: each month, select sleeves with trailing-12m Sharpe > 0
    bars = panel.index
    rebalance_dates = pd.date_range(bars[0].normalize(), bars[-1].normalize(), freq="ME", tz="UTC")
    portfolio_returns = []
    weights_log = []
    for me in rebalance_dates:
        idx_pos = bars.searchsorted(me)
        if idx_pos < 252: continue
        window = g.iloc[idx_pos - 252 : idx_pos]
        sharpes = {}
        for c in TOP22:
            r = window[c]
            sh = r.mean()*365 / (r.std(ddof=0)*np.sqrt(365)) if r.std() > 0 else 0
            sharpes[c] = sh
        s_series = pd.Series(sharpes)
        # WF_GATE_POS: include all with trailing-12m Sh > 0
        gate_pos = s_series[s_series > 0]
        next_me = me + pd.offsets.MonthEnd(1)
        next_pos = bars.searchsorted(next_me)
        out_window = g.iloc[idx_pos:next_pos]
        if len(out_window) == 0: continue
        if len(gate_pos) > 0:
            ret = out_window[list(gate_pos.index)].mean(axis=1)
        else:
            ret = pd.Series(0, index=out_window.index)
        portfolio_returns.append(ret)
        for s, sh in sharpes.items():
            weights_log.append({"month": me, "sleeve": s, "trailing_sh": sh,
                                "included": s in gate_pos.index})

    wf = pd.concat(portfolio_returns)
    pd.DataFrame(weights_log).to_csv(OUT / "v14_wf_weights.csv", index=False)

    static = g[TOP22].mean(axis=1)

    print(f"{'Variant':<26} {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>7} {'OOS_DD':>8}")
    for name, r in [("STATIC TOP22 (hindsight)", static),
                     ("WF_GATE_POS (no look-ahead)", wf)]:
        full_bpy = _bpy(r.index)
        full_sh = r.mean()*full_bpy/(r.std(ddof=0)*np.sqrt(full_bpy))
        is_part = r[r.index < SPLIT]
        oos_part = r[r.index >= SPLIT]
        is_bpy = _bpy(is_part.index) if len(is_part) > 0 else 252
        oos_bpy = _bpy(oos_part.index) if len(oos_part) > 0 else 252
        is_sh = is_part.mean()*is_bpy/(is_part.std(ddof=0)*np.sqrt(is_bpy)) if is_part.std() > 0 else 0
        oos_sh = oos_part.mean()*oos_bpy/(oos_part.std(ddof=0)*np.sqrt(oos_bpy)) if oos_part.std() > 0 else 0
        eq_oos = (1+oos_part).cumprod()
        oos_dd = (eq_oos/eq_oos.cummax()-1).min()
        y22 = r[r.index.year == 2022]
        bpy22 = _bpy(y22.index) if len(y22) > 0 else 252
        sh22 = (y22.mean()*bpy22)/(y22.std(ddof=0)*np.sqrt(bpy22)) if y22.std() > 0 else 0
        print(f"{name:<26} {full_sh:>+6.2f} {is_sh:>+6.2f} {oos_sh:>+6.2f} {sh22:>+7.2f} {oos_dd:>+8.1%}")

    # Vol-target the WF version
    vt, _ = time_of_month_vol_target(wf, base_target=0.18)
    from master_v4 import drawdown_control
    prod, _ = drawdown_control(vt)
    oos = prod[prod.index >= SPLIT]
    bpy = _bpy(oos.index)
    ar = oos.mean()*bpy
    av = oos.std(ddof=0)*np.sqrt(bpy)
    sh = ar/av if av > 0 else 0
    eq = (1+oos).cumprod()
    dd = (eq/eq.cummax()-1).min()
    print(f"\nWF_GATE_POS at 18% vol target (honest production):")
    print(f"  OOS Ret={ar:+.1%}  Vol={av:.1%}  Sh={sh:+.2f}  DD={dd:+.1%}  Mo={ar/12:+.2%}")


if __name__ == "__main__":
    main()
