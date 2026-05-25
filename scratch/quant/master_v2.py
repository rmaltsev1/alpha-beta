"""Master portfolio v2 — regime overlay + concentrated top-N.

Starts from master_portfolio.py's all_sleeve_returns.parquet (each sleeve already
rescaled to 5% IS vol). Adds two variants:

  * REGIME_OVERLAY: halve the beta-class sleeves (VOLMGD, RISKPAR) when SPX
    30d realized vol is above the IS 80th percentile. The other 9 sleeves are
    untouched. Targets the 2022 weakness specifically.

  * TOP_N: pick the top-N sleeves by IS Sharpe, equal-weighted, after de-duping
    highly correlated pairs (e.g. VOLMGD vs RISKPAR @ ρ=0.92 → keep RISKPAR).
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


def stats_for(label, r):
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2:
            continue
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        bpy = len(sub) / span * 365.25 if span > 0 else 252.0
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


def yearly(r):
    rows = {}
    for year, sub in r.groupby(r.index.year):
        span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
        if span < 60: continue
        bpy = len(sub) / span * 365.25
        sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
        rows[year] = sh
    return pd.Series(rows)


def main():
    df = pd.read_parquet(OUT / "all_sleeve_returns.parquet")
    df.index = pd.to_datetime(df.index, utc=True)

    # ---- (1) Regime mask: SPX 30d realized vol > IS 80th pctile ----
    spx = get_candles("SPX500_USD", "D1").copy()
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["rv30"] = spx["ret"].rolling(30).std()
    is_rv = spx.loc[spx["timestamp"] < SPLIT, "rv30"].dropna()
    cutoff = float(is_rv.quantile(0.80))
    spx_aligned = pd.DataFrame({"timestamp": spx["timestamp"], "rv30": spx["rv30"]})
    print(f"SPX 30d realized vol cutoff (IS p80): {cutoff:.4f}  (~{cutoff*np.sqrt(252)*100:.1f}% ann)")

    # Build per-day regime mask aligned to df.index
    spx_aligned["timestamp"] = pd.to_datetime(spx_aligned["timestamp"], utc=True)
    mask_df = pd.merge_asof(
        pd.DataFrame({"timestamp": df.index}).sort_values("timestamp"),
        spx_aligned.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    high_vol = mask_df["rv30"] > cutoff
    high_vol = high_vol.fillna(False).astype(bool)
    high_vol.index = df.index

    # ---- (2) Regime-overlay variant: halve the long-biased beta sleeves ----
    beta_sleeves = ["VOLMGD", "RISKPAR"]
    overlay = df.copy()
    for sleeve in beta_sleeves:
        if sleeve in overlay.columns:
            overlay.loc[high_vol, sleeve] *= 0.5
    regime_overlay = overlay.mean(axis=1)

    # ---- (3) Top-N by IS Sharpe, after de-duping the VOLMGD/RISKPAR pair ----
    is_part = df[df.index < SPLIT]
    bpy = 365.25
    is_sh = (is_part.mean() * bpy) / (is_part.std(ddof=0) * np.sqrt(bpy))
    # Drop VOLMGD because RISKPAR has higher IS Sharpe and they are 0.92 corr.
    candidates = [c for c in df.columns if c != "VOLMGD"]
    ranked = is_sh[candidates].sort_values(ascending=False)
    print(f"\nIS Sharpe ranking (after dropping VOLMGD dup of RISKPAR):")
    for n, s in ranked.items(): print(f"  {n:<14} {s:+.2f}")

    top5 = list(ranked.head(5).index)
    top7 = list(ranked.head(7).index)
    print(f"\nTOP5 = {top5}\nTOP7 = {top7}")

    top5_p = df[top5].mean(axis=1)
    top7_p = df[top7].mean(axis=1)

    # ---- Stats for all variants ----
    variants = {
        "EQUAL_WT (11)":  df.mean(axis=1),
        "REGIME_OVERLAY": regime_overlay,
        "TOP5_EW":        top5_p,
        "TOP7_EW":        top7_p,
    }

    print(f"\n{'Variant':<18} {'FULL_Sh':>7} {'IS_Sh':>6} {'OOS_Sh':>6} {'OOS_vol':>7} {'OOS_DD':>7}")
    print("-" * 60)
    saved_rows = []
    for vname, r in variants.items():
        s = stats_for(vname, r)
        saved_rows.append(s)
        print(f"{vname:<18} {s['FULL_sharpe']:>+7.2f} {s['IS_sharpe']:>+6.2f} "
              f"{s['OOS_sharpe']:>+6.2f} {s['OOS_vol']:>7.1%} {s['OOS_dd']:>+7.1%}")
    pd.DataFrame(saved_rows).to_csv(OUT / "master_v2_variants.csv", index=False)

    # Year-by-year for the best variant
    yr = pd.DataFrame({k: yearly(v) for k, v in variants.items()})
    print(f"\nYear-by-year Sharpe:")
    print(yr.round(2).to_string())
    yr.to_csv(OUT / "master_v2_yearly.csv")

    # Save the headline portfolio (regime overlay) as parquet
    eq = (1 + regime_overlay).cumprod()
    pd.DataFrame({"timestamp": regime_overlay.index, "ret": regime_overlay.values,
                  "equity": eq.values}).to_parquet(OUT / "master_v2_returns.parquet", index=False)


if __name__ == "__main__":
    main()
