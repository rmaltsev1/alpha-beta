"""Master v4 — production realism layer.

Adds three real institutional-grade overlays on top of v3:

  1. Portfolio-vol targeting overlay: scale gross so trailing-60d realized vol
     hits a fixed 10% target. Leverage capped at 5x. Levered version is what
     a real fund would run.

  2. Fast decay tripwire: per-sleeve, trailing-63d (~3-month) Sharpe checked
     monthly. If a sleeve's trailing Sharpe is < 0 for 2 consecutive monthly
     checks, that sleeve goes to cash for the next month. Re-enters when
     trailing Sharpe > 0.5 again. Catches dead sleeves in ~75d vs the 179d
     of the trailing-12m gate.

  3. Drawdown control: if portfolio's rolling 30-day drawdown exceeds 3%,
     halve total gross until DD recovers below 1%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

TOP8 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
        "D1REV_NAS", "WED_BTC", "DEFEND"]
GATES_HIGH = {
    "RISKPAR": 0.5, "TSMOM": 0.5, "XSMOM": 0.5, "EVE_XAU": 0.5, "WED_BTC": 0.5,
    "D1REV_NAS": 1.5, "D1REV_UK": 1.5, "DEFEND": 1.5,
}


def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2: continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


def build_regime_mask(idx, percentile=80):
    spx = get_candles("SPX500_USD", "D1").copy()
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["ret"].rolling(30).std()
    is_part = spx.loc[spx["timestamp"] < SPLIT, "rv30"].dropna()
    cutoff = float(is_part.quantile(percentile / 100))
    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)
    aligned = pd.merge_asof(
        pd.DataFrame({"timestamp": idx}).sort_values("timestamp"),
        spx[["timestamp", "rv30"]].sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    return pd.Series((aligned["rv30"] > cutoff).fillna(False).values, index=idx), cutoff


def apply_regime_gate(panel, high_vol):
    g = panel.copy()
    for sleeve, mult in GATES_HIGH.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult
    return g


def fast_decay_tripwire(panel, sleeves, lookback_days=63, gate_floor=-0.0,
                       reentry_threshold=0.5, consecutive_checks=2):
    """Per-sleeve gate based on trailing N-day Sharpe.

    Each calendar month-end, evaluate trailing 63-day Sharpe per sleeve.
    Maintain a per-sleeve "state": OFF if 2 consecutive checks below gate_floor.
    Re-enter when next check is above reentry_threshold.

    Returns: a DataFrame same shape as panel where dead sleeves are zeroed out.
    """
    out = panel.copy()
    sqrt_an = np.sqrt(365)
    # Per-sleeve checks at month-end timestamps
    month_ends = pd.date_range(panel.index[0].normalize(), panel.index[-1].normalize(),
                               freq="ME", tz="UTC")
    states = {s: "ON" for s in sleeves}
    neg_count = {s: 0 for s in sleeves}
    decay_log = []

    for me in month_ends:
        # Find the closest index <= me
        idx_pos = panel.index.searchsorted(me, side="right") - 1
        if idx_pos < lookback_days:
            continue
        window = panel.iloc[idx_pos - lookback_days + 1 : idx_pos + 1]
        for s in sleeves:
            r = window[s]
            ann_sh = r.mean() * 365 / (r.std(ddof=0) * sqrt_an) if r.std() > 0 else 0
            decay_log.append({"month_end": me, "sleeve": s, "trailing_sharpe": ann_sh,
                              "state_before": states[s]})
            if states[s] == "ON":
                if ann_sh < gate_floor:
                    neg_count[s] += 1
                    if neg_count[s] >= consecutive_checks:
                        states[s] = "OFF"
                else:
                    neg_count[s] = 0
            else:  # OFF
                if ann_sh > reentry_threshold:
                    states[s] = "ON"
                    neg_count[s] = 0
        # Apply state to the NEXT month's bars in `out`
        next_me = me + pd.offsets.MonthEnd(1)
        next_pos = panel.index.searchsorted(next_me, side="right")
        for s in sleeves:
            if states[s] == "OFF":
                out.iloc[idx_pos:next_pos, out.columns.get_loc(s)] = 0.0
    return out, pd.DataFrame(decay_log)


def vol_target_overlay(returns: pd.Series, target_vol: float = 0.10,
                       lookback: int = 60, max_lev: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """Multiply daily portfolio returns by a leverage that tries to hit a
    fixed target realized vol. Leverage at bar t uses ONLY data up through t-1.
    """
    rolling_vol = returns.rolling(lookback).std(ddof=0).shift(1) * np.sqrt(365.25)
    lev = (target_vol / rolling_vol).clip(upper=max_lev).fillna(1.0)
    lev = lev.where(rolling_vol > 0, 1.0)
    return returns * lev, lev


def drawdown_control(returns: pd.Series, dd_threshold: float = 0.03,
                     recovery_threshold: float = 0.01, dd_lookback: int = 30) -> tuple[pd.Series, pd.Series]:
    """If trailing 30-day drawdown > dd_threshold, halve gross until DD recovers
    below recovery_threshold. Walk-forward (uses only past data)."""
    eq = (1 + returns).cumprod()
    rolling_peak = eq.rolling(dd_lookback, min_periods=1).max()
    dd = eq / rolling_peak - 1
    dd_lag = dd.shift(1).fillna(0)
    mult = pd.Series(1.0, index=returns.index)
    in_control = False
    for i in range(1, len(returns)):
        prev_in = in_control
        if not in_control:
            if dd_lag.iloc[i] < -dd_threshold:
                in_control = True
        else:
            if dd_lag.iloc[i] > -recovery_threshold:
                in_control = False
        mult.iloc[i] = 0.5 if in_control else 1.0
    return returns * mult, mult


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v3.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    print(f"Panel sleeves: {list(panel.columns)}, bars: {len(panel)}")

    # Regime mask
    high_vol, cutoff = build_regime_mask(panel.index, 80)
    print(f"\nRegime cutoff SPX 30d-rv p80: {cutoff:.4f}, OOS high-vol bars: "
          f"{high_vol[panel.index >= SPLIT].sum()}/{(panel.index >= SPLIT).sum()}")

    # Apply regime gate
    gated = apply_regime_gate(panel, high_vol)

    # Build baseline portfolio (no overlays)
    baseline = gated[TOP8].mean(axis=1)

    # ---- Fast decay tripwire ----
    print("\n=== Applying fast decay tripwire (63d trailing Sharpe, 2 consec checks) ===")
    after_decay, decay_log = fast_decay_tripwire(gated, TOP8)
    decay_log.to_csv(OUT / "v4_decay_log.csv", index=False)
    tripwired = after_decay[TOP8].mean(axis=1)
    print(f"Bars where ≥1 sleeve was OFF (post-tripwire):")
    sleeves_off = (after_decay[TOP8] == 0).sum(axis=1)
    print(f"  Mean # sleeves off per bar: {sleeves_off.mean():.2f}")
    print(f"  Max # sleeves off:           {sleeves_off.max()}")

    # ---- Vol-target overlay (target 10% portfolio vol) ----
    print("\n=== Vol-target overlay (target 10%, max lev 5x) ===")
    vol_target_baseline, lev_baseline = vol_target_overlay(baseline, target_vol=0.10)
    vol_target_tripwired, lev_tripwired = vol_target_overlay(tripwired, target_vol=0.10)
    print(f"Average leverage applied (baseline):  {lev_baseline.mean():.2f}x  max={lev_baseline.max():.2f}x")
    print(f"Average leverage applied (tripwired): {lev_tripwired.mean():.2f}x  max={lev_tripwired.max():.2f}x")

    # ---- Drawdown control ----
    print("\n=== Drawdown control (halve if 30d DD > 3%, recover at 1%) ===")
    dd_baseline, dd_mult_b = drawdown_control(vol_target_baseline)
    dd_tripwired, dd_mult_t = drawdown_control(vol_target_tripwired)
    print(f"% of bars in DD-control state (baseline):  {(dd_mult_b < 1).mean():.1%}")
    print(f"% of bars in DD-control state (tripwired): {(dd_mult_t < 1).mean():.1%}")

    # ---- Compare all variants ----
    variants = {
        "v3 baseline (no overlays)":  baseline,
        "+ tripwire only":             tripwired,
        "+ tripwire + vol-target 10%": vol_target_tripwired,
        "+ tripwire + vol-target + DD control": dd_tripwired,
    }

    print(f"\n{'Variant':<42} {'FULL_Sh':>7} {'IS_Sh':>6} {'OOS_Sh':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
    rows = []
    for name, r in variants.items():
        s = stats(name, r)
        rows.append({**s})
        print(f"{name:<42} {s.get('FULL_sharpe',0):>+7.2f} {s.get('IS_sharpe',0):>+6.2f} "
              f"{s.get('OOS_sharpe',0):>+7.2f} {s.get('OOS_vol',0):>8.1%} {s.get('OOS_dd',0):>+7.1%}")
    pd.DataFrame(rows).to_csv(OUT / "master_v4_variants.csv", index=False)

    # Year-by-year
    print(f"\nYear-by-year:")
    yr = {}
    for name, r in variants.items():
        yr_row = {}
        for year, sub in r.groupby(r.index.year):
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            yr_row[year] = sh
        yr[name] = yr_row
    yr_df = pd.DataFrame(yr).T
    print(yr_df.round(2).to_string())
    yr_df.to_csv(OUT / "master_v4_yearly.csv")

    # Save the production v4 (with all overlays)
    prod = dd_tripwired
    eq = (1 + prod).cumprod()
    pd.DataFrame({"timestamp": prod.index, "ret": prod.values, "equity": eq.values}).to_parquet(
        OUT / "PRODUCTION_v4.parquet", index=False)
    print(f"\nProduction v4 (full overlays) saved to PRODUCTION_v4.parquet")
    print(f"  Total return: {(eq.iloc[-1]-1)*100:+.1f}%")
    print(f"  OOS slice:     {(1+prod[prod.index>=SPLIT]).cumprod().iloc[-1]*100-100:+.1f}%")


if __name__ == "__main__":
    main()
