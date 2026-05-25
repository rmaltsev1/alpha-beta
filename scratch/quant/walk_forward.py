"""Walk-forward sleeve-selection simulator.

At each monthly rebalance date t:
  * Compute trailing-12m Sharpe (and optionally trailing-3m) for each sleeve using
    only data strictly before t (no look-ahead).
  * Apply a selection rule to pick a sleeve subset.
  * Equal-weight the selected sleeves; hold until next rebalance.

Variants:
  WF_TOP7          top 7 by 12m Sharpe
  WF_TOP5          top 5 by 12m Sharpe
  WF_GATE_POS      every sleeve with 12m Sharpe > 0
  WF_GATE_HALF     every sleeve with 12m Sharpe > 0.5
  WF_RECENT_BIAS   top 5 by 0.6*Sharpe_12m + 0.4*Sharpe_3m
  STATIC_TOP7      always trades the hindsight-picked TOP7  (benchmark)

Outputs:
  scratch/quant/walkforward_variants.parquet      daily returns, wide
  scratch/quant/walkforward_breakdown.csv         per-variant headline stats
  scratch/quant/walkforward_selections.csv        which sleeves picked each month
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
START = pd.Timestamp("2021-01-01", tz="UTC")  # first rebalance date
END   = pd.Timestamp("2026-05-31", tz="UTC")

STATIC_TOP7 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK",
               "XSMOM", "D1REV_NAS", "WED_BTC"]


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _ann_factor(sub: pd.Series) -> float:
    span = (sub.index[-1] - sub.index[0]).total_seconds() / 86400
    return len(sub) / span * 365.25 if span > 0 else 365.25


def stats_for(label: str, r: pd.Series) -> dict:
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2:
            continue
        bpy = _ann_factor(sub)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        eq = (1 + sub).cumprod()
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


def yearly_sharpe(r: pd.Series) -> pd.Series:
    rows = {}
    for year, sub in r.groupby(r.index.year):
        if len(sub) < 30 or sub.std(ddof=0) == 0:
            continue
        bpy = _ann_factor(sub)
        ar = sub.mean() * bpy
        av = sub.std(ddof=0) * np.sqrt(bpy)
        rows[int(year)] = ar / av
    return pd.Series(rows)


def trailing_sharpe(returns: pd.DataFrame, end_excl: pd.Timestamp, days: int) -> pd.Series:
    """Trailing-`days` Sharpe per sleeve using returns strictly before `end_excl`."""
    start = end_excl - pd.Timedelta(days=days)
    window = returns.loc[(returns.index >= start) & (returns.index < end_excl)]
    if len(window) < 5:
        return pd.Series(0.0, index=returns.columns)
    bpy = _ann_factor(window)
    mu = window.mean() * bpy
    sd = window.std(ddof=0) * np.sqrt(bpy)
    sh = mu / sd.replace(0, np.nan)
    return sh.fillna(0.0)


# --------------------------------------------------------------------------- #
# Selection rules
# --------------------------------------------------------------------------- #
def select_top_n(sh12: pd.Series, sh3: pd.Series, n: int) -> list[str]:
    return list(sh12.sort_values(ascending=False).head(n).index)


def select_gate(sh12: pd.Series, sh3: pd.Series, threshold: float) -> list[str]:
    chosen = list(sh12[sh12 > threshold].sort_values(ascending=False).index)
    return chosen if chosen else list(sh12.sort_values(ascending=False).head(1).index)


def select_recent_bias(sh12: pd.Series, sh3: pd.Series, n: int = 5) -> list[str]:
    score = 0.6 * sh12.reindex(sh3.index).fillna(0) + 0.4 * sh3
    return list(score.sort_values(ascending=False).head(n).index)


SELECTION_RULES = {
    "WF_TOP7":        lambda s12, s3: select_top_n(s12, s3, 7),
    "WF_TOP5":        lambda s12, s3: select_top_n(s12, s3, 5),
    "WF_GATE_POS":    lambda s12, s3: select_gate(s12, s3, 0.0),
    "WF_GATE_HALF":   lambda s12, s3: select_gate(s12, s3, 0.5),
    "WF_RECENT_BIAS": lambda s12, s3: select_recent_bias(s12, s3, 5),
}


# --------------------------------------------------------------------------- #
# Walk-forward simulation
# --------------------------------------------------------------------------- #
def month_starts(index: pd.DatetimeIndex, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DatetimeIndex:
    """First trading-calendar day of each month in [start, end]."""
    sub = index[(index >= start) & (index <= end)]
    if len(sub) == 0:
        return sub
    s = sub.to_series()
    firsts = s.groupby([s.dt.year, s.dt.month]).first()
    return pd.DatetimeIndex(firsts.values, tz="UTC")


def walk_forward(returns: pd.DataFrame, rule_fn) -> tuple[pd.Series, pd.DataFrame]:
    """Run walk-forward for a given selection rule.

    Returns
    -------
    daily_returns : pd.Series   Portfolio daily returns (equal-weighted across selected sleeves).
    selections    : pd.DataFrame  index = rebalance date, columns = ['sleeves', 'n'].
    """
    rebals = month_starts(returns.index, START, END)
    sel_records = []
    seg_returns = []

    for i, t in enumerate(rebals):
        sh12 = trailing_sharpe(returns, t, 365)
        sh3  = trailing_sharpe(returns, t,  92)
        sleeves = rule_fn(sh12, sh3)
        if not sleeves:
            sleeves = [returns.columns[0]]  # safety net
        sel_records.append({"rebalance": t, "sleeves": ",".join(sleeves),
                            "n": len(sleeves)})

        # Hold from t (inclusive) until next rebalance (exclusive), or end of data.
        t_end = rebals[i + 1] if i + 1 < len(rebals) else returns.index.max() + pd.Timedelta(days=1)
        seg = returns.loc[(returns.index >= t) & (returns.index < t_end), sleeves]
        if len(seg):
            seg_returns.append(seg.mean(axis=1))

    daily = pd.concat(seg_returns).sort_index() if seg_returns else pd.Series(dtype=float)
    sel = pd.DataFrame(sel_records).set_index("rebalance")
    return daily, sel


def static_portfolio(returns: pd.DataFrame, sleeves: list[str]) -> pd.Series:
    sub = returns.loc[(returns.index >= START) & (returns.index <= END), sleeves]
    return sub.mean(axis=1)


# --------------------------------------------------------------------------- #
# Turnover + selection summaries
# --------------------------------------------------------------------------- #
def selection_turnover(sel: pd.DataFrame) -> float:
    """Fraction of rebalances where the set of held sleeves changes vs prior month."""
    sets = [frozenset(s.split(",")) for s in sel["sleeves"]]
    if len(sets) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(sets[:-1], sets[1:]) if a != b)
    return changes / (len(sets) - 1)


def top_picks(sel: pd.DataFrame, top: int = 3) -> list[tuple[str, int]]:
    cnt = Counter()
    for s in sel["sleeves"]:
        cnt.update(s.split(","))
    return cnt.most_common(top)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df = pd.read_parquet(OUT / "all_sleeve_returns.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    variants: dict[str, pd.Series] = {}
    selections: dict[str, pd.DataFrame] = {}

    for name, rule in SELECTION_RULES.items():
        daily, sel = walk_forward(df, rule)
        variants[name] = daily
        selections[name] = sel

    # Benchmark: static TOP7
    variants["STATIC_TOP7"] = static_portfolio(df, STATIC_TOP7)

    # Align all variants on the same index for cleaner output
    var_df = pd.DataFrame(variants).sort_index()

    # --- Save daily variant returns (wide) ---
    var_df.to_parquet(OUT / "walkforward_variants.parquet")

    # --- Save selections (one CSV, long format) ---
    rows = []
    for rule_name, sel in selections.items():
        for ts, row in sel.iterrows():
            rows.append({"rule": rule_name, "rebalance": ts,
                         "n": row["n"], "sleeves": row["sleeves"]})
    pd.DataFrame(rows).to_csv(OUT / "walkforward_selections.csv", index=False)

    # --- Per-variant breakdown ---
    headline_rows = []
    yearly_table = {}
    print(f"\n{'Variant':<16} {'FULL':>6} {'IS':>6} {'OOS':>6}  {'OOS_vol':>7} {'OOS_dd':>7}  "
          f"{'avgN':>4} {'turn':>5}  top picks")
    print("-" * 110)
    for name, r in var_df.items():
        s = stats_for(name, r)
        if name in selections:
            sel = selections[name]
            avg_n = float(sel["n"].mean())
            turn = selection_turnover(sel)
            tp = top_picks(sel, top=3)
            top_str = ", ".join(f"{k}({v})" for k, v in tp)
        else:
            avg_n = float(len(STATIC_TOP7))
            turn = 0.0
            top_str = ", ".join(STATIC_TOP7[:3])
        s.update({"avg_n": avg_n, "turnover": turn, "top_picks": top_str})
        headline_rows.append(s)
        yearly_table[name] = yearly_sharpe(r)
        print(f"{name:<16} {s.get('FULL_sharpe', 0):>+6.2f} {s.get('IS_sharpe', 0):>+6.2f} "
              f"{s.get('OOS_sharpe', 0):>+6.2f}  {s.get('OOS_vol', 0):>7.1%} {s.get('OOS_dd', 0):>+7.1%}  "
              f"{avg_n:>4.1f} {turn:>5.0%}  {top_str}")

    breakdown = pd.DataFrame(headline_rows)
    breakdown.to_csv(OUT / "walkforward_breakdown.csv", index=False)

    # --- Yearly Sharpe table ---
    yr_df = pd.DataFrame(yearly_table).round(2)
    print(f"\nYear-by-year Sharpe:")
    print(yr_df.to_string())

    # --- IS-overlap of WF_TOP7 vs STATIC_TOP7 selections ---
    static_set = set(STATIC_TOP7)
    overlap = []
    for ts, row in selections["WF_TOP7"].iterrows():
        s = set(row["sleeves"].split(","))
        overlap.append(len(s & static_set))
    print(f"\nWF_TOP7 vs STATIC_TOP7 overlap (n sleeves shared): "
          f"mean={np.mean(overlap):.2f} / 7, median={np.median(overlap):.0f}")


if __name__ == "__main__":
    main()
