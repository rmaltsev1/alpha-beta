"""Bucket-level optimal weighting (wave6).

The 22 production sleeves are grouped into 8 buckets. We test 6 weighting
schemes and compare them with sleeve-level equal-weight (the v14 baseline).

Schemes:
  1) EQ_BUCKET   : 1/8 to each bucket, equal-weight within bucket.
  2) RP_BUCKET   : inverse-vol at the bucket level, equal-weight within.
  3) SH12_BUCKET : bucket weights proportional to trailing-12m bucket Sharpe.
  4) SH22_BUCKET : bucket weights proportional to historical 2022 Sharpe.
  5) CAPPED_EW   : sleeve EW within bucket with a bucket cap (no bucket > 20%).
  6) SH_SHRINK   : bucket weights = alpha * trailing-12m bucket Sh weight
                   + (1 - alpha) * 1/8.

All schemes share the same v14 overlay stack:
  - high-vol regime gates (GATES_HIGH + v14 overrides)
  - fast decay tripwire (63d trailing Sharpe, 2 consec checks)
  - vol-target overlay (10% target, max 15x lev)
  - drawdown control (halve at 3% trailing-30d DD, recover at 1%)

The weighting decisions are walk-forward, refreshed monthly: at each month
end we compute bucket metrics on the trailing 12 months (the 2022 stress
weights use only data inside calendar-year 2022 that is also <= rebal date).
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
    fast_decay_tripwire,
    vol_target_overlay,
    drawdown_control,
    build_regime_mask,
    _bpy,
    GATES_HIGH,
)

OUT = ROOT / "scratch" / "wave6"
PANEL_PATH = ROOT / "scratch" / "quant" / "all_sleeve_returns_v14.parquet"
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
Y2022_START = pd.Timestamp("2022-01-01", tz="UTC")
Y2022_END = pd.Timestamp("2023-01-01", tz="UTC")
VOL_TARGET = 0.10
MAX_LEV = 15.0
LOOKBACK_BARS = 252
SQRT_AN = np.sqrt(365.25)

# Bucket assignment for the 22-sleeve panel.
BUCKETS = {
    "calendar":       ["EVE_XAU", "WED_BTC", "SESSION_MOM", "EVENT_VOLSPIKE"],
    "mean_reversion": ["D1REV_NAS", "D1REV_UK", "STATARB_XS"],
    "trend":          ["TSMOM", "TREND_NEW", "TREND_COND"],
    "xs_trend":       ["XSMOM", "CORR_REGIME"],
    "beta":           ["RISKPAR", "VOLFORECAST", "DEFEND"],
    "pairs":          ["PAIRS_EXP", "CRYPTO_vs_SPX", "EURGBP_MR"],
    "microstructure": ["MICROSTR_D1", "H4_SLEEVE", "MULTIDAY", "W1_STRATS"],
    "vol_crisis":     ["VOL_BREAKOUT", "TERM_SPREADS"],
}

BUCKET_CAP = 0.20  # scheme 5
SHRINK_ALPHA = 0.4  # scheme 6


def all_sleeves() -> list[str]:
    return [s for b in BUCKETS.values() for s in b]


def annualised_sharpe(r: pd.Series) -> float:
    if r.std(ddof=0) <= 0 or len(r) < 5:
        return 0.0
    return float(r.mean() * 365.25 / (r.std(ddof=0) * SQRT_AN))


def annualised_vol(r: pd.Series) -> float:
    if len(r) < 5:
        return np.nan
    return float(r.std(ddof=0) * SQRT_AN)


def equal_within_bucket_weights() -> dict[str, dict[str, float]]:
    """Static EW within each bucket."""
    out = {}
    for b, sleeves in BUCKETS.items():
        if not sleeves:
            continue
        w = 1.0 / len(sleeves)
        out[b] = {s: w for s in sleeves}
    return out


def softmax_positive(values: pd.Series) -> pd.Series:
    """Map values -> positive weights summing to 1.

    Negative values get 0; if all values are <= 0, fall back to equal weights.
    """
    pos = values.clip(lower=0)
    if pos.sum() <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return pos / pos.sum()


def bucket_return_series(panel: pd.DataFrame, bucket: str) -> pd.Series:
    """Equal-weight composite of the bucket's sleeves on the gated panel."""
    cols = [c for c in BUCKETS[bucket] if c in panel.columns]
    if not cols:
        return pd.Series(0.0, index=panel.index)
    return panel[cols].mean(axis=1)


def trailing_window(panel: pd.DataFrame, end_pos: int, lookback: int) -> pd.DataFrame:
    start = max(0, end_pos - lookback)
    return panel.iloc[start:end_pos]


def cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    """Iteratively cap weights at `cap` and redistribute excess to the rest."""
    w = weights.copy().astype(float)
    for _ in range(50):
        over = w[w > cap]
        if over.empty:
            break
        excess = (over - cap).sum()
        w.loc[over.index] = cap
        under = w[w < cap]
        if under.empty or under.sum() <= 0:
            break
        w.loc[under.index] += excess * under / under.sum()
    return w / w.sum()


def compute_bucket_weights(
    scheme: str,
    panel: pd.DataFrame,
    end_pos: int,
) -> pd.Series:
    """Return a Series indexed by bucket name summing to 1."""
    buckets = list(BUCKETS.keys())
    n = len(buckets)

    if scheme == "EQ_BUCKET":
        return pd.Series(1.0 / n, index=buckets)

    window = trailing_window(panel, end_pos, LOOKBACK_BARS)
    if len(window) < 30:
        return pd.Series(1.0 / n, index=buckets)

    if scheme == "RP_BUCKET":
        vols = {b: annualised_vol(bucket_return_series(window, b)) for b in buckets}
        inv = pd.Series({b: (1.0 / v) if v and v > 1e-8 else 0.0 for b, v in vols.items()})
        if inv.sum() <= 0:
            return pd.Series(1.0 / n, index=buckets)
        return inv / inv.sum()

    if scheme == "SH12_BUCKET":
        sh = pd.Series({b: annualised_sharpe(bucket_return_series(window, b))
                        for b in buckets})
        return softmax_positive(sh)

    if scheme == "SH22_BUCKET":
        # 2022 historical Sharpe weights. Only use 2022 data <= end_pos.
        end_ts = panel.index[end_pos - 1]
        upper = min(end_ts, Y2022_END - pd.Timedelta(seconds=1))
        if upper < Y2022_START:
            return pd.Series(1.0 / n, index=buckets)
        y22 = panel.loc[(panel.index >= Y2022_START) & (panel.index <= upper)]
        if len(y22) < 30:
            return pd.Series(1.0 / n, index=buckets)
        sh = pd.Series({b: annualised_sharpe(bucket_return_series(y22, b))
                        for b in buckets})
        return softmax_positive(sh)

    if scheme == "CAPPED_EW":
        # Bucket weight = bucket-Sharpe-tilt capped at BUCKET_CAP.
        sh = pd.Series({b: annualised_sharpe(bucket_return_series(window, b))
                        for b in buckets})
        raw = softmax_positive(sh)
        return cap_weights(raw, BUCKET_CAP)

    if scheme == "SH_SHRINK":
        sh = pd.Series({b: annualised_sharpe(bucket_return_series(window, b))
                        for b in buckets})
        tilt = softmax_positive(sh)
        eq = pd.Series(1.0 / n, index=buckets)
        return SHRINK_ALPHA * tilt + (1.0 - SHRINK_ALPHA) * eq

    raise ValueError(f"unknown scheme {scheme}")


def build_scheme_returns(
    scheme: str,
    panel: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply walk-forward bucket weighting on the gated panel; equal-weight within bucket.

    Returns: (daily portfolio return series, monthly bucket-weights log DataFrame).
    """
    bars = panel.index
    rebal = pd.date_range(bars[0].normalize(), bars[-1].normalize(), freq="ME", tz="UTC")
    ew_within = equal_within_bucket_weights()
    composite = pd.Series(0.0, index=bars)
    weight_log = []

    # Pre-compute bucket EW return series on the gated panel (constant within-bucket weights).
    bucket_series = {b: bucket_return_series(panel, b) for b in BUCKETS}

    cur_w = pd.Series(1.0 / len(BUCKETS), index=list(BUCKETS.keys()))
    last_pos = 0

    for me in rebal:
        idx_pos = bars.searchsorted(me, side="right")
        if idx_pos <= last_pos:
            continue
        # Apply current weights to the segment up to but not including this month-end
        seg = bars[last_pos:idx_pos]
        if len(seg) > 0:
            seg_ret = sum(cur_w[b] * bucket_series[b].loc[seg] for b in cur_w.index)
            composite.loc[seg] = seg_ret
        # Re-compute weights using data up to idx_pos (uses only past info)
        if idx_pos >= LOOKBACK_BARS:
            cur_w = compute_bucket_weights(scheme, panel, idx_pos)
        weight_log.append({"month_end": me, **{f"w_{b}": cur_w.get(b, 0.0) for b in BUCKETS}})
        last_pos = idx_pos

    # Tail segment after last rebalance
    if last_pos < len(bars):
        seg = bars[last_pos:]
        seg_ret = sum(cur_w[b] * bucket_series[b].loc[seg] for b in cur_w.index)
        composite.loc[seg] = seg_ret

    # Note: within-bucket is equal-weight (ew_within is implicit in bucket_series).
    _ = ew_within  # silence unused

    return composite, pd.DataFrame(weight_log)


def perf_block(name: str, r: pd.Series) -> dict:
    out = {"scheme": name}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS", r.index < SPLIT),
                      ("OOS", r.index >= SPLIT),
                      ("Y2022", (r.index >= Y2022_START) & (r.index < Y2022_END))]:
        sub = r[mask]
        if len(sub) < 5:
            for k in ("ret", "vol", "sharpe", "dd"):
                out[f"{tag}_{k}"] = np.nan
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean() * bpy)
        av = float(sub.std(ddof=0) * np.sqrt(bpy))
        sh = ar / av if av > 0 else 0.0
        eq = (1 + sub).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        out[f"{tag}_sharpe"] = sh
        out[f"{tag}_dd"] = dd
    return out


def main():
    panel = pd.read_parquet(PANEL_PATH)
    panel.index = pd.to_datetime(panel.index, utc=True)
    sleeves = [s for s in all_sleeves() if s in panel.columns]
    missing = [s for s in all_sleeves() if s not in panel.columns]
    if missing:
        print("WARNING: missing sleeves in panel:", missing)
    panel = panel[sleeves].copy()
    print(f"Universe: {len(sleeves)} sleeves across {len(BUCKETS)} buckets, "
          f"{len(panel)} bars {panel.index[0].date()} -> {panel.index[-1].date()}")

    # ---- v14 regime gate overrides (same as scratch/quant/v14_walk_forward.py) ----
    gates = dict(GATES_HIGH)
    for s in ["PAIRS_EXP", "CRYPTO_vs_SPX", "CORR_REGIME", "STATARB_XS",
              "MICROSTR_D1", "EURGBP_MR", "TERM_SPREADS", "EVENT_VOLSPIKE", "MULTIDAY"]:
        gates[s] = 1.5
    for s in ["VOLFORECAST", "W1_STRATS", "SESSION_MOM"]:
        gates[s] = 1.0
    gates["H4_SLEEVE"] = 1.5
    gates["TREND_NEW"] = 0.5
    gates["VOL_BREAKOUT"] = 1.2

    high_vol, cutoff = build_regime_mask(panel.index, 80)
    print(f"Regime cutoff (SPX 30d RV p80): {cutoff:.4f}; "
          f"high-vol bars: {high_vol.sum()}/{len(panel)}")
    gated = panel.copy()
    for s, m in gates.items():
        if s in gated.columns:
            gated.loc[high_vol, s] *= m

    # ---- Fast decay tripwire ----
    gated, decay_log = fast_decay_tripwire(gated, sleeves)
    print(f"Decay tripwire: mean # sleeves off/bar="
          f"{(gated[sleeves] == 0).sum(axis=1).mean():.2f}")

    schemes = ["EQ_BUCKET", "RP_BUCKET", "SH12_BUCKET", "SH22_BUCKET",
               "CAPPED_EW", "SH_SHRINK"]

    raw_returns: dict[str, pd.Series] = {}
    vol_returns: dict[str, pd.Series] = {}
    final_returns: dict[str, pd.Series] = {}
    weight_logs: dict[str, pd.DataFrame] = {}

    for sch in schemes:
        r_raw, wlog = build_scheme_returns(sch, gated)
        r_vt, _lev = vol_target_overlay(r_raw, target_vol=VOL_TARGET, max_lev=MAX_LEV)
        r_dd, _mlt = drawdown_control(r_vt)
        raw_returns[sch] = r_raw
        vol_returns[sch] = r_vt
        final_returns[sch] = r_dd
        weight_logs[sch] = wlog

    # ---- Sleeve-level equal-weight baseline (v14 walk-forward style, but full panel EW) ----
    sleeve_ew_raw = gated[sleeves].mean(axis=1)
    sleeve_ew_vt, _ = vol_target_overlay(sleeve_ew_raw, target_vol=VOL_TARGET, max_lev=MAX_LEV)
    sleeve_ew_final, _ = drawdown_control(sleeve_ew_vt)
    final_returns["SLEEVE_EW"] = sleeve_ew_final

    # ---- Build comparison ----
    rows = [perf_block(name, r) for name, r in final_returns.items()]
    comp = pd.DataFrame(rows).set_index("scheme")
    comp = comp[[
        "FULL_sharpe", "FULL_ret", "FULL_vol", "FULL_dd",
        "IS_sharpe", "IS_ret", "IS_vol", "IS_dd",
        "OOS_sharpe", "OOS_ret", "OOS_vol", "OOS_dd",
        "Y2022_sharpe", "Y2022_ret", "Y2022_dd",
    ]]
    comp.to_csv(OUT / "bucket_comparison.csv")

    # ---- Save daily returns for every scheme (incl. SLEEVE_EW) ----
    returns_df = pd.DataFrame({k: v for k, v in final_returns.items()})
    returns_df.index.name = "date"
    returns_df.to_parquet(OUT / "bucket_returns.parquet")

    # ---- Print headline table ----
    print("\n=== BUCKET WEIGHTING COMPARISON (OOS = post 2024-01-01) ===")
    print(f"{'Scheme':<14} "
          f"{'FULL_Sh':>7} {'IS_Sh':>7} {'OOS_Sh':>7} {'OOS_Ret':>8} "
          f"{'OOS_Vol':>8} {'OOS_DD':>8} {'2022_Sh':>8}")
    order = ["SLEEVE_EW"] + schemes
    for name in order:
        row = comp.loc[name]
        print(f"{name:<14} "
              f"{row['FULL_sharpe']:>+7.2f} {row['IS_sharpe']:>+7.2f} "
              f"{row['OOS_sharpe']:>+7.2f} {row['OOS_ret']:>+8.1%} "
              f"{row['OOS_vol']:>8.1%} {row['OOS_dd']:>+8.1%} "
              f"{row['Y2022_sharpe']:>+8.2f}")

    # ---- Best OOS scheme: log its monthly bucket weights ----
    best_oos = comp["OOS_sharpe"].drop("SLEEVE_EW").idxmax()
    print(f"\nBest OOS scheme: {best_oos} (OOS Sharpe={comp.loc[best_oos, 'OOS_sharpe']:+.2f})")
    weight_logs[best_oos].to_csv(OUT / f"bucket_weights_{best_oos}.csv", index=False)

    # ---- Average bucket weights across schemes (informational) ----
    print("\nAverage bucket weights (last 24 months) for each scheme:")
    for sch in schemes:
        wlog = weight_logs[sch]
        recent = wlog.tail(24).filter(like="w_").mean()
        print(f"  {sch:<14} " + " ".join(f"{b.replace('w_',''):>8}={v:.2f}"
                                          for b, v in recent.items()))


if __name__ == "__main__":
    main()
