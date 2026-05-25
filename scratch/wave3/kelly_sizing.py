"""Adaptive Kelly-criterion position-sizing layer (Wave 3).

Tilts each sleeve weight by trailing hit-rate / win-loss asymmetry (half-Kelly)
in addition to the standard regime gates. Compares against:
    - Equal-weight TOP12 (current production)
    - Inverse-vol weights
    - Sharpe-weighted (trailing 12m)
    - Half-Kelly + positive-3m-Sharpe filter (decay tripwire combo)

Walk-forward: rebalances monthly using only data prior to the rebalance date.
IS  : <  2024-01-01
OOS : >= 2024-01-01

Outputs
-------
kelly_returns.parquet  : per-sleeve returns variants + portfolio rolls
kelly_weights.csv      : per-rebalance weights for the half-Kelly variant
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import existing regime / overlay helpers from master_v4
QUANT = ROOT / "scratch" / "quant"
sys.path.insert(0, str(QUANT))
from master_v4 import (  # type: ignore
    GATES_HIGH,
    build_regime_mask,
    fast_decay_tripwire,
    vol_target_overlay,
    _bpy,
)

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

TOP12 = [
    "RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
    "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
    "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
]

# Extended gates from master_v9
GATES = dict(GATES_HIGH)
GATES["PAIRS_EXP"] = 1.5
GATES["VOLFORECAST"] = 1.0
GATES["H4_SLEEVE"] = 1.5
GATES["TREND_NEW"] = 0.5
GATES["CRYPTO_vs_SPX"] = 1.5

KELLY_LOOKBACK = 63        # ~3 months for hit-rate / W / L
SHARPE_LOOKBACK = 252      # ~12m trailing Sharpe
SHARPE_3M_LOOKBACK = 63    # ~3m trailing Sharpe filter
WEIGHT_CAP = 0.30          # max 30% per sleeve
TARGET_VOL = 0.10          # 10% annualized portfolio vol
ANN_FACTOR = 365           # daily bars, calendar-day basis


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def stats(label: str, r: pd.Series) -> dict:
    out = {"label": label}
    for tag, mask in [
        ("FULL", pd.Series(True, index=r.index)),
        ("IS",   r.index < SPLIT),
        ("OOS",  r.index >= SPLIT),
    ]:
        sub = r[mask]
        if len(sub) < 2:
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    # 2022 slice
    mask_22 = (r.index >= pd.Timestamp("2022-01-01", tz="UTC")) & (
        r.index < pd.Timestamp("2023-01-01", tz="UTC")
    )
    sub = r[mask_22]
    if len(sub) > 30:
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out["2022_sharpe"] = ar / av if av > 0 else 0.0
        out["2022_ret"] = ar
    return out


def normalize_weights(w: np.ndarray, cap: float = WEIGHT_CAP) -> np.ndarray:
    """Cap and renormalize. If all weights are zero, return zeros."""
    w = np.where(w > 0, w, 0.0)
    if w.sum() <= 0:
        return np.zeros_like(w)
    w = w / w.sum()
    # iterative cap: re-cap and renormalize until stable
    for _ in range(20):
        capped = np.minimum(w, cap)
        excess = w.sum() - capped.sum()
        if excess < 1e-9:
            w = capped
            break
        # redistribute to uncapped weights
        below = capped < cap - 1e-9
        if not below.any():
            w = capped
            break
        add = excess * (capped[below] / capped[below].sum()) if capped[below].sum() > 0 else 0
        w = capped.copy()
        w[below] = capped[below] + add
    if w.sum() > 0:
        w = w / w.sum()
    return w


# -----------------------------------------------------------------------------
# Per-sleeve sizing rules (walk-forward)
# -----------------------------------------------------------------------------

def kelly_weights_at(window: pd.DataFrame) -> dict:
    """Half-Kelly fractional weights from a trailing window.

    For each column:
        p   = fraction of positive return bars
        W   = mean of positive returns
        L   = mean magnitude of negative returns (positive number)
        b   = W / L
        q   = 1 - p
        f*  = (b*p - q) / b   (Kelly fraction; clipped at 0)
        half-Kelly = f* / 2
    """
    out = {}
    for col in window.columns:
        r = window[col].dropna().values
        if len(r) < 20:
            out[col] = 0.0
            continue
        pos = r[r > 0]
        neg = r[r < 0]
        if len(pos) == 0 or len(neg) == 0:
            out[col] = 0.0
            continue
        p = len(pos) / len(r)
        W = float(pos.mean())
        L = float(-neg.mean())  # magnitude
        if L <= 0 or W <= 0:
            out[col] = 0.0
            continue
        b = W / L
        q = 1.0 - p
        f_star = (b * p - q) / b
        out[col] = max(0.0, f_star) * 0.5  # half-Kelly
    return out


def inv_vol_weights_at(window: pd.DataFrame) -> dict:
    out = {}
    for col in window.columns:
        s = window[col].std(ddof=0)
        out[col] = (1.0 / s) if s > 1e-9 else 0.0
    return out


def sharpe_weights_at(window: pd.DataFrame) -> dict:
    out = {}
    for col in window.columns:
        r = window[col].dropna()
        if len(r) < 30:
            out[col] = 0.0
            continue
        mu = float(r.mean())
        sd = float(r.std(ddof=0))
        if sd <= 0:
            out[col] = 0.0
            continue
        sharpe = mu * ANN_FACTOR / (sd * np.sqrt(ANN_FACTOR))
        out[col] = max(0.0, sharpe)  # positive Sharpe only
    return out


def trailing_3m_sharpe(window: pd.DataFrame) -> dict:
    out = {}
    for col in window.columns:
        r = window[col].dropna()
        if len(r) < 20:
            out[col] = 0.0
            continue
        mu = float(r.mean())
        sd = float(r.std(ddof=0))
        if sd <= 0:
            out[col] = 0.0
            continue
        out[col] = mu * ANN_FACTOR / (sd * np.sqrt(ANN_FACTOR))
    return out


# -----------------------------------------------------------------------------
# Walk-forward driver
# -----------------------------------------------------------------------------

def build_weights_series(panel: pd.DataFrame, rule: str,
                          sleeves: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct walk-forward daily weight panel for one rule.

    Rebalances at each calendar month-end. Uses ONLY data with index < rebalance.
    Weights are applied starting from the NEXT bar after the rebalance.
    Equal-weight uses constant 1/N (no estimation).
    """
    weights = pd.DataFrame(0.0, index=panel.index, columns=sleeves)
    month_ends = pd.date_range(
        panel.index[0].normalize(), panel.index[-1].normalize(),
        freq="ME", tz="UTC"
    )

    rebal_records = []
    current_w = np.full(len(sleeves), 1.0 / len(sleeves))  # equal-weight start
    weights.iloc[0] = current_w

    for me in month_ends:
        idx_pos = panel.index.searchsorted(me, side="right") - 1
        if idx_pos < 0:
            continue
        # data strictly up through `me` (inclusive of me bar is fine for rebal-decision)
        cutoff_pos = idx_pos + 1
        history = panel.iloc[:cutoff_pos][sleeves]

        if rule == "equal":
            w_arr = np.full(len(sleeves), 1.0 / len(sleeves))

        elif rule == "inv_vol":
            window = history.iloc[-SHARPE_LOOKBACK:] if len(history) > SHARPE_LOOKBACK else history
            if len(window) < 30:
                w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
            else:
                w_dict = inv_vol_weights_at(window)
                w_arr = np.array([w_dict[s] for s in sleeves])
                w_arr = normalize_weights(w_arr)

        elif rule == "sharpe":
            window = history.iloc[-SHARPE_LOOKBACK:] if len(history) > SHARPE_LOOKBACK else history
            if len(window) < 60:
                w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
            else:
                w_dict = sharpe_weights_at(window)
                w_arr = np.array([w_dict[s] for s in sleeves])
                if w_arr.sum() <= 0:
                    w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
                else:
                    w_arr = normalize_weights(w_arr)

        elif rule == "kelly":
            window = history.iloc[-KELLY_LOOKBACK:] if len(history) > KELLY_LOOKBACK else history
            if len(window) < 30:
                w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
            else:
                w_dict = kelly_weights_at(window)
                w_arr = np.array([w_dict[s] for s in sleeves])
                if w_arr.sum() <= 0:
                    # No sleeves with positive Kelly => fall back to equal
                    w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
                else:
                    w_arr = normalize_weights(w_arr)

        elif rule == "kelly_filtered":
            # Half-Kelly only on sleeves with positive trailing-3m Sharpe
            window_k = history.iloc[-KELLY_LOOKBACK:] if len(history) > KELLY_LOOKBACK else history
            window_f = history.iloc[-SHARPE_3M_LOOKBACK:] if len(history) > SHARPE_3M_LOOKBACK else history
            if len(window_k) < 30 or len(window_f) < 20:
                w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
            else:
                w_k = kelly_weights_at(window_k)
                sh_3m = trailing_3m_sharpe(window_f)
                w_arr = np.array([
                    w_k[s] if sh_3m[s] > 0 else 0.0
                    for s in sleeves
                ])
                if w_arr.sum() <= 0:
                    w_arr = np.full(len(sleeves), 1.0 / len(sleeves))
                else:
                    w_arr = normalize_weights(w_arr)
        else:
            raise ValueError(rule)

        rebal_records.append({"month_end": me, **{s: w_arr[i] for i, s in enumerate(sleeves)}})

        # apply from next bar onward (walk-forward; rebalance trade fills overnight)
        next_pos = cutoff_pos
        end_pos = len(panel)
        # next month-end position
        next_me = me + pd.offsets.MonthEnd(1)
        next_me_pos = panel.index.searchsorted(next_me, side="right") - 1
        end_pos = min(next_me_pos + 1, len(panel))
        if end_pos <= next_pos:
            continue
        weights.iloc[next_pos:end_pos] = w_arr
        current_w = w_arr

    # forward-fill any trailing zeros after the last rebalance
    nonzero = (weights.abs().sum(axis=1) > 0)
    if nonzero.any():
        last_good = weights[nonzero].iloc[-1].values
        weights.loc[weights.index > weights[nonzero].index[-1]] = last_good

    return weights, pd.DataFrame(rebal_records).set_index("month_end")


def portfolio_returns(panel: pd.DataFrame, weights: pd.DataFrame,
                      sleeves: list[str]) -> pd.Series:
    """Weighted sum of sleeve returns. Weights are applied at the same bar
    (already lagged because the build_weights_series uses prior-bar info)."""
    return (panel[sleeves] * weights[sleeves]).sum(axis=1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    panel = pd.read_parquet(QUANT / "all_sleeve_returns_v9.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    print(f"Panel: {len(panel)} bars, {len(panel.columns)} sleeves")
    print(f"Date range: {panel.index.min()} -> {panel.index.max()}")
    print(f"IS/OOS split: {SPLIT}")

    # Apply regime gates (same as master_v9)
    high_vol, cutoff = build_regime_mask(panel.index, 80)
    print(f"Regime cutoff SPX 30d-rv p80: {cutoff:.4f}, "
          f"high-vol bars: {high_vol.sum()}/{len(high_vol)}")

    gated = panel.copy()
    for sleeve, mult in GATES.items():
        if sleeve in gated.columns:
            gated.loc[high_vol, sleeve] *= mult

    # Apply fast decay tripwire (state machine on per-sleeve 63d Sharpe)
    after_decay, _decay_log = fast_decay_tripwire(gated, TOP12)

    # ----------------------------------------------------------------------
    # Build weights for each variant
    # ----------------------------------------------------------------------
    print("\nBuilding walk-forward weights...")
    variants_w = {}
    variants_w["equal"], _ = build_weights_series(after_decay, "equal", TOP12)
    variants_w["inv_vol"], _ = build_weights_series(after_decay, "inv_vol", TOP12)
    variants_w["sharpe"], _ = build_weights_series(after_decay, "sharpe", TOP12)
    variants_w["kelly"], kelly_rebal = build_weights_series(after_decay, "kelly", TOP12)
    variants_w["kelly_filtered"], kelly_f_rebal = build_weights_series(
        after_decay, "kelly_filtered", TOP12
    )

    # Save Kelly rebalance weights
    kelly_rebal.to_csv(OUT / "kelly_weights.csv")
    print(f"Kelly rebal weights saved -> kelly_weights.csv ({len(kelly_rebal)} rebals)")

    # ----------------------------------------------------------------------
    # Build portfolio returns (gross), then 10% vol-target overlay
    # ----------------------------------------------------------------------
    raw_returns = {}
    final_returns = {}
    for name, w in variants_w.items():
        gross = portfolio_returns(after_decay, w, TOP12)
        levered, lev = vol_target_overlay(gross, target_vol=TARGET_VOL)
        raw_returns[name] = gross
        final_returns[name] = levered
        print(f"  {name:<18} avg lev={lev.mean():.2f}x max={lev.max():.2f}x")

    # ----------------------------------------------------------------------
    # Compare
    # ----------------------------------------------------------------------
    rows = []
    print(f"\n{'Variant':<20} {'FULL_Sh':>7} {'IS_Sh':>6} {'OOS_Sh':>7} "
          f"{'2022_Sh':>7} {'OOS_vol':>8} {'OOS_DD':>7}")
    for name, r in final_returns.items():
        s = stats(name, r)
        rows.append(s)
        print(f"{name:<20} {s.get('FULL_sharpe',0):>+7.2f} "
              f"{s.get('IS_sharpe',0):>+6.2f} {s.get('OOS_sharpe',0):>+7.2f} "
              f"{s.get('2022_sharpe',0):>+7.2f} "
              f"{s.get('OOS_vol',0):>8.1%} {s.get('OOS_dd',0):>+7.1%}")

    pd.DataFrame(rows).to_csv(OUT / "kelly_variants.csv", index=False)

    # Year-by-year
    print("\nYear-by-year Sharpe:")
    yr = {}
    for name, r in final_returns.items():
        yr_row = {}
        for year, sub in r.groupby(r.index.year):
            if len(sub) < 50:
                continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            yr_row[year] = sh
        yr[name] = yr_row
    yr_df = pd.DataFrame(yr).T
    print(yr_df.round(2).to_string())
    yr_df.to_csv(OUT / "kelly_yearly.csv")

    # ----------------------------------------------------------------------
    # Mean Kelly weight per sleeve (FULL & OOS) — which sleeves get the love?
    # ----------------------------------------------------------------------
    print("\nMean Kelly weights per sleeve:")
    kw = kelly_rebal[TOP12]
    is_kw = kw[kw.index < SPLIT].mean()
    oos_kw = kw[kw.index >= SPLIT].mean()
    full_kw = kw.mean()

    # Reference: per-sleeve IS Sharpe (no regime gates, raw panel returns)
    is_panel = panel[panel.index < SPLIT][TOP12]
    is_sleeve_sharpe = {}
    for s in TOP12:
        r = is_panel[s]
        if r.std() > 0:
            is_sleeve_sharpe[s] = float(r.mean() * ANN_FACTOR / (r.std(ddof=0) * np.sqrt(ANN_FACTOR)))
        else:
            is_sleeve_sharpe[s] = 0.0

    print(f"{'Sleeve':<16} {'IS_Sh':>7} {'IS_KellyW':>10} {'OOS_KellyW':>11} {'FULL_KellyW':>12}")
    for s in TOP12:
        print(f"{s:<16} {is_sleeve_sharpe[s]:>+7.2f} "
              f"{is_kw[s]:>10.1%} {oos_kw[s]:>11.1%} {full_kw[s]:>12.1%}")

    sleeve_summary = pd.DataFrame({
        "is_sharpe": is_sleeve_sharpe,
        "is_kelly_w": is_kw,
        "oos_kelly_w": oos_kw,
        "full_kelly_w": full_kw,
    })
    sleeve_summary.to_csv(OUT / "kelly_sleeve_summary.csv")

    # ----------------------------------------------------------------------
    # Save per-sleeve returns parquet (gross + vol-target portfolio returns)
    # ----------------------------------------------------------------------
    saved = pd.DataFrame(index=panel.index)
    for name, r in raw_returns.items():
        saved[f"{name}_gross"] = r
    for name, r in final_returns.items():
        saved[f"{name}_vt10"] = r
    saved.to_parquet(OUT / "kelly_returns.parquet")
    print(f"\nSaved portfolio returns -> kelly_returns.parquet ({saved.shape})")

    return rows


if __name__ == "__main__":
    main()
