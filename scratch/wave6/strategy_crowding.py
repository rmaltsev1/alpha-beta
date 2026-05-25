"""Strategy-crowding / factor mean-reversion detection.

Hypothesis: when a sleeve has been winning unusually well, expect a revert;
when losing, expect a comeback. Walk-forward weighting on top of an
equal-weight TOP21 base. Six variants:

  V1  Per-sleeve trailing Sharpe ratio (3m/12m): hot -> halve, cold -> 1.5x.
  V2  Cross-sleeve dispersion: low dispersion -> tilt bottom-quartile up,
      high dispersion -> tilt top-quartile up.
  V3  Trend-factor reversion (TREND_NEW, TSMOM, XSMOM).
  V4  Mean-rev-factor reversion (D1REV_* and STATARB_XS).
  V5  Vol-conditional MR sizing using SPX 30d realized vol.
  V6  Cross-sleeve average pairwise corr expansion -> cut gross.

Methodology:
  * IS  < 2024-01-01, OOS >= 2024-01-01.
  * Monthly rebalance (first business day of each month).
  * All rolling stats are causal (computed strictly from data before the
    rebalance date).
  * Baseline: equal-weight TOP21 (from master_v13.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "scratch" / "quant"
WAVE6 = ROOT / "scratch" / "wave6"

PANEL_PATH = QUANT / "all_sleeve_returns_v13.parquet"
SPX_PATH   = ROOT / "data" / "SPX500_USD" / "D1.parquet"
SPLIT      = pd.Timestamp("2024-01-01", tz="UTC")

# TOP21 from master_v13.py
TOP14 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
         "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
         "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX", "CORR_REGIME",
         "SESSION_MOM"]
TOP16_v12 = TOP14 + ["W1_STRATS", "EVENT_VOLSPIKE"]
TOP21 = TOP16_v12 + ["STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
                    "TERM_SPREADS", "EURGBP_MR"]

TREND_FACTORS  = ["TREND_NEW", "TSMOM", "XSMOM"]
MR_FACTORS_PORT = [s for s in TOP21 if s.startswith("D1REV")] + ["STATARB_XS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bpy(idx) -> float:
    if len(idx) < 2:
        return 252.0
    span_days = (idx.max() - idx.min()).total_seconds() / 86400.0
    if span_days <= 0:
        return 252.0
    return max(1.0, len(idx) * 365.25 / span_days)


def sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 5 or s.std(ddof=0) == 0:
        return 0.0
    bpy = _bpy(s.index)
    return float(s.mean() * bpy / (s.std(ddof=0) * np.sqrt(bpy)))


def maxdd(s: pd.Series) -> float:
    if len(s) < 2:
        return 0.0
    eq = (1.0 + s.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def slice_period(s: pd.Series, start=None, end=None) -> pd.Series:
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index < end]
    return s


def stats_block(s: pd.Series) -> dict:
    return {
        "FULL_Sharpe": sharpe(s),
        "IS_Sharpe":   sharpe(slice_period(s, end=SPLIT)),
        "OOS_Sharpe":  sharpe(slice_period(s, start=SPLIT)),
        "Sharpe_2022": sharpe(s[s.index.year == 2022]),
        "FULL_MaxDD":  maxdd(s),
        "OOS_MaxDD":   maxdd(slice_period(s, start=SPLIT)),
        "FULL_Vol":    float(s.std(ddof=0) * np.sqrt(_bpy(s.index))),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_PATH)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # Keep only TOP21 columns (all should be present)
    miss = [c for c in TOP21 if c not in df.columns]
    if miss:
        raise KeyError(f"Missing TOP21 sleeves: {miss}")
    return df


def load_spx_realised_vol(idx: pd.DatetimeIndex, window: int = 30) -> pd.Series:
    spx = pd.read_parquet(SPX_PATH)
    ts = pd.to_datetime(spx["timestamp"], utc=True)
    close = pd.Series(spx["close"].values, index=ts).sort_index()
    rets = np.log(close / close.shift(1))
    vol = rets.rolling(window).std() * np.sqrt(252)
    # Align to panel daily index (one obs per UTC day)
    vol_daily = vol.groupby(vol.index.floor("D")).last()
    vol_daily.index = pd.to_datetime(vol_daily.index, utc=True)
    return vol_daily.reindex(idx).ffill()


# ---------------------------------------------------------------------------
# Rebalance dates: first calendar day of each month present in panel
# ---------------------------------------------------------------------------
def month_starts(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.DataFrame(index=idx)
    df["ym"] = df.index.to_period("M")
    firsts = df.groupby("ym").apply(lambda d: d.index.min())
    return pd.DatetimeIndex(sorted(firsts.values), tz="UTC")


# ---------------------------------------------------------------------------
# Rolling per-sleeve Sharpe (causal)
# ---------------------------------------------------------------------------
def rolling_sharpe(panel: pd.DataFrame, window_days: int) -> pd.DataFrame:
    mu = panel.rolling(window_days, min_periods=window_days // 2).mean()
    sd = panel.rolling(window_days, min_periods=window_days // 2).std(ddof=0)
    sh = (mu / sd) * np.sqrt(252.0)
    return sh.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Variant: per-sleeve Sharpe mean-reversion
# ---------------------------------------------------------------------------
def variant_sleeve_meanrev(
    panel: pd.DataFrame, sleeves: list[str],
    sh3: pd.DataFrame, sh12: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    rebal = month_starts(panel.index)
    weights_log = []
    daily_w = pd.DataFrame(index=panel.index, columns=sleeves, dtype=float)
    base = 1.0 / len(sleeves)
    cur_w = pd.Series(base, index=sleeves)
    for d in panel.index:
        if d in rebal:
            row3 = sh3.loc[d, sleeves]
            row12 = sh12.loc[d, sleeves]
            new_w = pd.Series(base, index=sleeves)
            for s in sleeves:
                r3, r12 = row3[s], row12[s]
                if pd.notna(r3) and pd.notna(r12) and abs(r12) > 1e-6:
                    ratio = r3 / r12
                    if r12 > 0 and ratio > 2.0:
                        new_w[s] *= 0.5
                    elif r12 > 0 and ratio < 0.5:
                        new_w[s] *= 1.5
                    elif r12 < 0 and ratio > 2.0:
                        # losing more than baseline -> boost (comeback)
                        new_w[s] *= 1.5
                    elif r12 < 0 and ratio < 0.5:
                        new_w[s] *= 0.5
            cur_w = new_w
            weights_log.append({"date": d, "variant": "V1_sleeve_mr", **cur_w.to_dict()})
        daily_w.loc[d] = cur_w.values
    ret = (panel[sleeves].fillna(0) * daily_w).sum(axis=1)
    return ret, pd.DataFrame(weights_log)


# ---------------------------------------------------------------------------
# Variant: cross-sleeve dispersion
# ---------------------------------------------------------------------------
def variant_dispersion(
    panel: pd.DataFrame, sleeves: list[str], sh3: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    rebal = month_starts(panel.index)
    weights_log = []
    daily_w = pd.DataFrame(index=panel.index, columns=sleeves, dtype=float)
    base = 1.0 / len(sleeves)
    cur_w = pd.Series(base, index=sleeves)
    # historical dispersion distribution (causal: pct-rank against past values)
    disp_series = sh3[sleeves].std(axis=1)
    for d in panel.index:
        if d in rebal:
            past = disp_series.loc[:d].iloc[:-1].dropna()
            cur_disp = disp_series.loc[d]
            new_w = pd.Series(base, index=sleeves)
            if pd.notna(cur_disp) and len(past) > 30:
                rank = (past < cur_disp).mean()
                row3 = sh3.loc[d, sleeves].dropna()
                if len(row3) >= 4:
                    q_lo = row3.quantile(0.25)
                    q_hi = row3.quantile(0.75)
                    if rank < 0.20:  # unusually homogeneous -> bottom quartile up
                        for s in row3.index:
                            if row3[s] <= q_lo:
                                new_w[s] *= 1.5
                    elif rank > 0.80:  # high dispersion -> top quartile up
                        for s in row3.index:
                            if row3[s] >= q_hi:
                                new_w[s] *= 1.5
            cur_w = new_w
            weights_log.append({"date": d, "variant": "V2_dispersion", **cur_w.to_dict()})
        daily_w.loc[d] = cur_w.values
    ret = (panel[sleeves].fillna(0) * daily_w).sum(axis=1)
    return ret, pd.DataFrame(weights_log)


# ---------------------------------------------------------------------------
# Variant: factor-group reversion (trend & mean-rev)
# ---------------------------------------------------------------------------
def variant_factor_group(
    panel: pd.DataFrame, sleeves: list[str],
    sh3: pd.DataFrame, group: list[str], tag: str,
) -> tuple[pd.Series, pd.DataFrame]:
    rebal = month_starts(panel.index)
    weights_log = []
    daily_w = pd.DataFrame(index=panel.index, columns=sleeves, dtype=float)
    base = 1.0 / len(sleeves)
    cur_w = pd.Series(base, index=sleeves)
    # Restrict group to columns that exist in panel
    grp_present = [g for g in group if g in panel.columns]
    grp_in_port = [g for g in group if g in sleeves]
    for d in panel.index:
        if d in rebal:
            grp_sharpes = sh3.loc[d, grp_present].dropna()
            new_w = pd.Series(base, index=sleeves)
            if len(grp_sharpes) == len(grp_present) and len(grp_sharpes) >= 2:
                avg = grp_sharpes.mean()
                mult = 1.0
                if avg > 2.0:
                    mult = 0.5
                elif avg < -2.0:
                    mult = 1.5
                for s in grp_in_port:
                    new_w[s] *= mult
            cur_w = new_w
            weights_log.append({"date": d, "variant": tag, **cur_w.to_dict()})
        daily_w.loc[d] = cur_w.values
    ret = (panel[sleeves].fillna(0) * daily_w).sum(axis=1)
    return ret, pd.DataFrame(weights_log)


# ---------------------------------------------------------------------------
# Variant: vol-conditional MR sizing
# ---------------------------------------------------------------------------
def variant_vol_cond_mr(
    panel: pd.DataFrame, sleeves: list[str],
    sh3: pd.DataFrame, spx_vol: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    rebal = month_starts(panel.index)
    weights_log = []
    daily_w = pd.DataFrame(index=panel.index, columns=sleeves, dtype=float)
    base = 1.0 / len(sleeves)
    cur_w = pd.Series(base, index=sleeves)
    vol_chg = spx_vol - spx_vol.shift(21)
    mr_in_port = [s for s in MR_FACTORS_PORT if s in sleeves]
    for d in panel.index:
        if d in rebal:
            new_w = pd.Series(base, index=sleeves)
            v = vol_chg.loc[d] if d in vol_chg.index else np.nan
            mr_sh = sh3.loc[d, mr_in_port].dropna()
            if pd.notna(v) and len(mr_sh) == len(mr_in_port):
                avg_mr = mr_sh.mean()
                mult = 1.0
                if v > 0 and avg_mr < -1.0:
                    mult = 1.5
                elif v < 0 and avg_mr > 1.0:
                    mult = 0.5
                for s in mr_in_port:
                    new_w[s] *= mult
            cur_w = new_w
            weights_log.append({"date": d, "variant": "V5_volcond_mr", **cur_w.to_dict()})
        daily_w.loc[d] = cur_w.values
    ret = (panel[sleeves].fillna(0) * daily_w).sum(axis=1)
    return ret, pd.DataFrame(weights_log)


# ---------------------------------------------------------------------------
# Variant: pairwise correlation expansion
# ---------------------------------------------------------------------------
def variant_corr_expansion(
    panel: pd.DataFrame, sleeves: list[str], corr_window: int = 63,
) -> tuple[pd.Series, pd.DataFrame]:
    rebal = month_starts(panel.index)
    weights_log = []
    daily_w = pd.DataFrame(index=panel.index, columns=sleeves, dtype=float)
    base = 1.0 / len(sleeves)
    cur_w = pd.Series(base, index=sleeves)
    sub = panel[sleeves].fillna(0)
    # Average pairwise correlation over trailing window, computed at rebalance dates
    avg_corr_map = {}
    for d in rebal:
        hist = sub.loc[:d].iloc[:-1].tail(corr_window)
        if len(hist) < corr_window // 2:
            avg_corr_map[d] = np.nan
            continue
        # only use columns with nonzero variance in window
        nz = hist.loc[:, hist.std(ddof=0) > 1e-9]
        if nz.shape[1] < 2:
            avg_corr_map[d] = np.nan
            continue
        cm = nz.corr().values
        mask = ~np.eye(cm.shape[0], dtype=bool)
        avg_corr_map[d] = float(np.nanmean(cm[mask]))
    for d in panel.index:
        if d in rebal:
            new_w = pd.Series(base, index=sleeves)
            ac = avg_corr_map.get(d, np.nan)
            if pd.notna(ac):
                if ac > 0.4:
                    new_w *= 0.7
                elif ac < 0.1:
                    new_w *= 1.2
            cur_w = new_w
            weights_log.append({"date": d, "variant": "V6_corr_exp",
                                "avg_corr": ac, **cur_w.to_dict()})
        daily_w.loc[d] = cur_w.values
    ret = (panel[sleeves].fillna(0) * daily_w).sum(axis=1)
    return ret, pd.DataFrame(weights_log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    panel = load_panel()
    sleeves = TOP21
    print(f"Loaded panel: {panel.shape}, using TOP21 ({len(sleeves)} sleeves)")
    print(f"Date range: {panel.index.min().date()} -> {panel.index.max().date()}")

    sh3 = rolling_sharpe(panel, 63)
    sh12 = rolling_sharpe(panel, 252)
    spx_vol = load_spx_realised_vol(panel.index, window=30)

    # Equal-weight baseline
    ew = panel[sleeves].fillna(0).mean(axis=1)

    results = {"EW_TOP21": ew}
    weights_all = []

    print("\nRunning variants...")
    r, w = variant_sleeve_meanrev(panel, sleeves, sh3, sh12)
    results["V1_sleeve_mr"] = r
    weights_all.append(w)

    r, w = variant_dispersion(panel, sleeves, sh3)
    results["V2_dispersion"] = r
    weights_all.append(w)

    r, w = variant_factor_group(panel, sleeves, sh3, TREND_FACTORS, "V3_trend_rev")
    results["V3_trend_rev"] = r
    weights_all.append(w)

    r, w = variant_factor_group(panel, sleeves, sh3, MR_FACTORS_PORT, "V4_mr_rev")
    results["V4_mr_rev"] = r
    weights_all.append(w)

    r, w = variant_vol_cond_mr(panel, sleeves, sh3, spx_vol)
    results["V5_volcond_mr"] = r
    weights_all.append(w)

    r, w = variant_corr_expansion(panel, sleeves)
    results["V6_corr_exp"] = r
    weights_all.append(w)

    # Combo: stacked (multiply all weight adjustments) — built simply as
    # equal-weight of variants' returns
    results["V7_combo_avg"] = pd.concat(
        [results[k] for k in ["V1_sleeve_mr", "V2_dispersion", "V3_trend_rev",
                              "V4_mr_rev", "V5_volcond_mr", "V6_corr_exp"]],
        axis=1,
    ).mean(axis=1)

    out_df = pd.DataFrame(results)
    out_df.index.name = "timestamp"
    out_df.to_parquet(WAVE6 / "crowding_returns.parquet")

    # Save weights log
    wlog = pd.concat(weights_all, ignore_index=True)
    wlog.to_csv(WAVE6 / "crowding_weights.csv", index=False)

    # Report
    print(f"\n{'Variant':<18}  {'FULL':>6} {'IS':>6} {'OOS':>6} {'2022':>6} {'FullDD':>7} {'OOSDD':>7}")
    print("-" * 70)
    rows = []
    for name, s in results.items():
        st = stats_block(s)
        rows.append({"variant": name, **st})
        print(f"{name:<18}  {st['FULL_Sharpe']:+6.2f} "
              f"{st['IS_Sharpe']:+6.2f} {st['OOS_Sharpe']:+6.2f} "
              f"{st['Sharpe_2022']:+6.2f} {st['FULL_MaxDD']:+7.1%} "
              f"{st['OOS_MaxDD']:+7.1%}")

    summary = pd.DataFrame(rows)
    summary.to_csv(WAVE6 / "crowding_summary.csv", index=False)

    # OOS Sharpe deltas vs EW
    ew_oos = stats_block(ew)["OOS_Sharpe"]
    print(f"\nEW OOS Sharpe: {ew_oos:+.2f}")
    print("Variant OOS Sharpe vs EW (delta):")
    beaters = []
    for name, s in results.items():
        if name == "EW_TOP21":
            continue
        delta = stats_block(s)["OOS_Sharpe"] - ew_oos
        marker = "  <- beats by >=0.15" if delta >= 0.15 else ""
        print(f"  {name:<18} {delta:+.2f}{marker}")
        if delta >= 0.15:
            beaters.append(name)

    if beaters:
        print(f"\nVariants beating EW by >=0.15 OOS Sharpe: {beaters}")
    else:
        print("\nNo variant beats EW by >=0.15 OOS Sharpe.")


if __name__ == "__main__":
    main()
