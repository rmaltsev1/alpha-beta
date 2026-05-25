"""
Wave 6 (extended): Stop-loss optimization for the remaining sleeves.

Sleeves tested (all from v14):
    EVE_XAU, D1REV_UK, XSMOM, D1REV_NAS, WED_BTC, DEFEND,
    VOLFORECAST, H4_SLEEVE, CRYPTO_vs_SPX, CORR_REGIME, SESSION_MOM,
    W1_STRATS, EVENT_VOLSPIKE,
    STATARB_XS, MICROSTR_D1, VOL_BREAKOUT, TERM_SPREADS, EURGBP_MR, MULTIDAY

Stop families (per spec):
    1. Fixed 2% stop, 1-day cooldown
    2. Fixed 3% stop, 2-day cooldown
    3. Fixed 5% stop, 3-day cooldown
    4. 1.5x ATR(14) stop (D1 ATR from cumulative sleeve P&L)
    5. 2.0x ATR(14) stop
    6. Time-based 3-bar max hold
    7. Trailing 50% stop after 2% gain
    8. Equity-curve 5% trailing-21d drawdown halt

Episode model:
    Each contiguous block of non-zero daily P&L is a "trade episode". Within
    the episode we accumulate cum gain/loss and trigger the stop rule. After
    a stop fires we zero out the sleeve for N cooldown days. Next non-zero
    P&L day starts a new episode.

ATR is computed directly from the cumulative-P&L curve of each sleeve (not
from any underlying instrument), which is what the spec asks for: a true
sleeve-vol unit.

Selection:
    Best variant per sleeve must satisfy:
        is_sharpe_lift  >= -0.15
        oos_sharpe_lift >=  +0.10   (else fall back to baseline)
    among eligible variants, pick the one with the largest oos_sharpe_lift.

Outputs:
    scratch/wave6/stops_extended.py            <- this script
    scratch/wave6/stops_extended_matrix.csv    <- (sleeve, rule) x metrics
    scratch/wave6/stops_extended_returns.parquet  baseline + best variant per sleeve
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path("/Users/rinatmaltsev/Documents/Python Projects/alpha-beta/alpha-beta")
SLEEVE_FILE = ROOT / "scratch/quant/all_sleeve_returns_v14.parquet"
OUT_DIR = ROOT / "scratch/wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN = np.sqrt(252)
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

TARGET_SLEEVES = [
    "EVE_XAU", "D1REV_UK", "XSMOM", "D1REV_NAS", "WED_BTC", "DEFEND",
    "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX", "CORR_REGIME", "SESSION_MOM",
    "W1_STRATS", "EVENT_VOLSPIKE",
    "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT", "TERM_SPREADS",
    "EURGBP_MR", "MULTIDAY",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * ANN)


def max_dd(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    eq = (1.0 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def slice_metrics(r: pd.Series) -> dict:
    is_mask = r.index < SPLIT
    oos_mask = r.index >= SPLIT
    return {
        "sharpe_full": sharpe(r),
        "sharpe_is": sharpe(r[is_mask]),
        "sharpe_oos": sharpe(r[oos_mask]),
        "maxdd_full": max_dd(r),
        "maxdd_is": max_dd(r[is_mask]),
        "maxdd_oos": max_dd(r[oos_mask]),
    }


# ---------------------------------------------------------------------------
# ATR(14) computed from cumulative sleeve P&L
# ---------------------------------------------------------------------------
def sleeve_atr_from_cumpnl(returns: pd.Series, n: int = 14) -> pd.Series:
    """Synthesize OHLC from cumulative P&L (each bar's H/L is the rolling
    intra-bar extremes proxied by abs return)."""
    cum = returns.cumsum()
    close = cum
    prev_close = close.shift(1)
    # since we only have one observation per bar, treat |return| as TR proxy
    # and ATR(14) as 14d EWMA of |return|. This is the "D1 ATR computed from
    # cumulative sleeve P&L" the spec asks for.
    tr = returns.abs()
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr.bfill().fillna(returns.std())


# ---------------------------------------------------------------------------
# Episode state machine
# ---------------------------------------------------------------------------
def apply_stop(
    returns: pd.Series,
    *,
    stop_check: Callable[[dict, float, float, int, float], bool],
    cooldown: int = 1,
    extra: dict | None = None,
) -> pd.Series:
    """Walk daily returns. Within each contiguous non-zero episode call
    stop_check(state, ret, vol, day_in_trade, cum). If True, book today's
    P&L, then go flat for `cooldown` days.
    """
    extra = dict(extra) if extra else {}
    vols = extra.get("vol_series")
    out = returns.copy().astype(float)
    arr_val = out.values

    in_trade = False
    cum = 0.0
    peak = 0.0
    day_idx = 0
    cooldown_left = 0

    for i in range(len(arr_val)):
        r = float(arr_val[i])
        if cooldown_left > 0:
            out.iloc[i] = 0.0
            cooldown_left -= 1
            in_trade = False
            cum = peak = 0.0
            day_idx = 0
            continue

        if r == 0.0 and not in_trade:
            continue

        if not in_trade and r != 0.0:
            in_trade = True
            cum = 0.0
            peak = 0.0
            day_idx = 0

        if in_trade:
            cum += r
            peak = max(peak, cum)
            day_idx += 1
            vol_today = float(vols.iloc[i]) if vols is not None else np.nan
            triggered = stop_check(
                {"peak": peak, "day": day_idx, "cum": cum, "vol": vol_today, **extra},
                r, vol_today, day_idx, cum,
            )
            if triggered:
                cooldown_left = int(extra.get("cooldown_after", cooldown))
                in_trade = False
                cum = peak = 0.0
                day_idx = 0
            elif r == 0.0:
                in_trade = False
                cum = peak = 0.0
                day_idx = 0
    return out


def apply_equity_protection(
    returns: pd.Series,
    *,
    dd_thresh: float = -0.05,
    window: int = 21,
    halt_days: int = 5,
) -> pd.Series:
    """Sleeve circuit breaker: if trailing-`window` cumulative return drops
    below `dd_thresh`, force sleeve flat for `halt_days`."""
    out = returns.copy().astype(float)
    rolling = returns.rolling(window, min_periods=window).sum()
    halt = 0
    for i in range(len(out)):
        if halt > 0:
            out.iloc[i] = 0.0
            halt -= 1
            continue
        if i >= window and rolling.iloc[i - 1] < dd_thresh:
            halt = halt_days - 1
            out.iloc[i] = 0.0
    return out


# ---------------------------------------------------------------------------
# Stop-rule callable factories
# ---------------------------------------------------------------------------
def rule_fixed_pct(thr: float):
    def check(state, r, vol, day, cum):
        return cum <= -abs(thr)
    return check


def rule_atr(k: float):
    def check(state, r, vol, day, cum):
        if not np.isfinite(vol) or vol <= 0:
            return False
        return cum <= -k * vol
    return check


def rule_time(max_days: int):
    def check(state, r, vol, day, cum):
        return day >= max_days
    return check


def rule_trail(arm: float, frac: float):
    def check(state, r, vol, day, cum):
        peak = state["peak"]
        if peak < arm:
            return False
        return cum <= frac * peak
    return check


# ---------------------------------------------------------------------------
# Variants (the 8 rules from the spec)
# ---------------------------------------------------------------------------
def build_variants():
    return [
        ("fixed_2pct_cd1",         "fixed",  dict(thr=0.02), 1),
        ("fixed_3pct_cd2",         "fixed",  dict(thr=0.03), 2),
        ("fixed_5pct_cd3",         "fixed",  dict(thr=0.05), 3),
        ("atr_1.5x",               "atr",    dict(k=1.5),     1),
        ("atr_2.0x",               "atr",    dict(k=2.0),     1),
        ("time_3bar",              "time",   dict(max_days=3), 1),
        ("trail_arm2pct_frac50",   "trail",  dict(arm=0.02, frac=0.5), 1),
        ("eqprot_5pct_21d",        "eqprot", dict(dd_thresh=-0.05, window=21, halt_days=5), 0),
    ]


def apply_variant(returns: pd.Series, vol: pd.Series, family: str,
                  params: dict, cooldown: int) -> pd.Series:
    extra = {"vol_series": vol, "cooldown_after": cooldown}
    if family == "fixed":
        return apply_stop(returns, stop_check=rule_fixed_pct(params["thr"]),
                          extra=extra)
    if family == "atr":
        return apply_stop(returns, stop_check=rule_atr(params["k"]),
                          extra=extra)
    if family == "time":
        return apply_stop(returns, stop_check=rule_time(params["max_days"]),
                          extra=extra)
    if family == "trail":
        return apply_stop(returns,
                          stop_check=rule_trail(params["arm"], params["frac"]),
                          extra=extra)
    if family == "eqprot":
        return apply_equity_protection(returns, **params)
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    sleeves = pd.read_parquet(SLEEVE_FILE)
    sleeves.index = pd.to_datetime(sleeves.index, utc=True)

    variants = build_variants()
    rows = []
    best_payload: dict[str, tuple[str, pd.Series]] = {}

    for sleeve in TARGET_SLEEVES:
        if sleeve not in sleeves.columns:
            print(f"[{sleeve}] NOT IN PARQUET, skipping")
            continue
        r = sleeves[sleeve].astype(float)
        vol = sleeve_atr_from_cumpnl(r, n=14)
        base_m = slice_metrics(r)
        base_row = {
            "sleeve": sleeve, "family": "baseline", "variant": "baseline",
            "n_nonzero_days": int((r != 0).sum()),
            **base_m,
            "is_sharpe_lift": 0.0, "oos_sharpe_lift": 0.0,
            "maxdd_oos_improve": 0.0,
            "best_pick": False,
        }
        rows.append(base_row)

        best_oos_lift = -1e9
        best_name = "baseline"
        best_series = r

        for name, family, params, cd in variants:
            stopped = apply_variant(r, vol, family, params, cd)
            m = slice_metrics(stopped)
            row = {
                "sleeve": sleeve, "family": family, "variant": name,
                "n_nonzero_days": int((stopped != 0).sum()),
                **m,
                "is_sharpe_lift":  m["sharpe_is"]  - base_m["sharpe_is"],
                "oos_sharpe_lift": m["sharpe_oos"] - base_m["sharpe_oos"],
                "maxdd_oos_improve": base_m["maxdd_oos"] - m["maxdd_oos"],
                "best_pick": False,
            }
            rows.append(row)

            # Selection: IS not destroyed AND OOS lift >= +0.10
            if (row["is_sharpe_lift"] >= -0.15
                    and row["oos_sharpe_lift"] >= 0.10
                    and row["oos_sharpe_lift"] > best_oos_lift):
                best_oos_lift = row["oos_sharpe_lift"]
                best_name = name
                best_series = stopped

        best_payload[sleeve] = (best_name, best_series)
        print(f"[{sleeve}] baseline OOS Sharpe {base_m['sharpe_oos']:+.3f} | "
              f"best variant: {best_name} "
              f"({'lift '+format(best_oos_lift, '+.3f') if best_name != 'baseline' else 'no qualifier'})")

    df = pd.DataFrame(rows)

    # mark best_pick rows in matrix
    for sleeve, (best_name, _) in best_payload.items():
        mask = (df["sleeve"] == sleeve) & (df["variant"] == best_name)
        df.loc[mask, "best_pick"] = True

    col_order = ["sleeve", "family", "variant", "best_pick", "n_nonzero_days",
                 "sharpe_full", "sharpe_is", "sharpe_oos",
                 "is_sharpe_lift", "oos_sharpe_lift",
                 "maxdd_full", "maxdd_is", "maxdd_oos", "maxdd_oos_improve"]
    df = df[col_order]
    df.to_csv(OUT_DIR / "stops_extended_matrix.csv", index=False)

    # Build the best-returns parquet:
    # baseline + best variant returns per sleeve.
    out_cols = {}
    for sleeve in TARGET_SLEEVES:
        if sleeve not in sleeves.columns:
            continue
        out_cols[f"{sleeve}_baseline"] = sleeves[sleeve].astype(float)
        name, series = best_payload[sleeve]
        out_cols[f"{sleeve}_{name}"] = series.astype(float)
    pd.DataFrame(out_cols).to_parquet(OUT_DIR / "stops_extended_returns.parquet")

    print("\nWrote:")
    print(" ", OUT_DIR / "stops_extended_matrix.csv")
    print(" ", OUT_DIR / "stops_extended_returns.parquet")

    # --- aggregate: equal-weight blended Sharpe lift ------------------------
    baseline_panel = sleeves[TARGET_SLEEVES].astype(float)
    best_panel = pd.DataFrame(
        {s: best_payload[s][1] for s in TARGET_SLEEVES if s in baseline_panel.columns},
        index=baseline_panel.index,
    )

    def agg_metrics(panel):
        ew = panel.mean(axis=1)  # equal weight per day
        return slice_metrics(ew)

    agg_base = agg_metrics(baseline_panel)
    agg_best = agg_metrics(best_panel)
    print("\nAggregate (equal-weight) Sharpe:")
    print(f"  baseline OOS: {agg_base['sharpe_oos']:+.3f}  IS: {agg_base['sharpe_is']:+.3f}")
    print(f"  best-stop OOS: {agg_best['sharpe_oos']:+.3f}  IS: {agg_best['sharpe_is']:+.3f}")
    print(f"  delta OOS:    {agg_best['sharpe_oos'] - agg_base['sharpe_oos']:+.3f}")
    print(f"  delta IS:     {agg_best['sharpe_is']  - agg_base['sharpe_is']:+.3f}")

    return df


if __name__ == "__main__":
    df = main()
    best = df[df["best_pick"]]
    if len(best):
        print("\n=== Best variant per sleeve ===")
        print(best[["sleeve", "family", "variant",
                    "sharpe_is", "sharpe_oos",
                    "is_sharpe_lift", "oos_sharpe_lift",
                    "maxdd_oos_improve"]].to_string(index=False))
    else:
        print("\nNo sleeve produced a stop variant that cleared the +0.10/-0.15 bar.")
