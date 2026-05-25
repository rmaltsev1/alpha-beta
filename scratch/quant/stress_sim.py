"""Worst-case stress test: bootstrap 2022 returns into the OOS period and
also Monte-Carlo bootstrap arbitrary 12-month windows to see the distribution
of plausible outcomes.

Three tests:
  1. 2022 RE-INJECTION: replace the first 12 months of OOS with the actual
     2022 calendar year returns of each sleeve. Compute portfolio stats.
  2. MONTE-CARLO BOOTSTRAP: 1000 trials where we sample a contiguous 12-month
     block from the IS period and use it as a hypothetical year. Distribution
     of returns / Sharpes.
  3. DECAY DETECTION: simulate a sleeve dying mid-OOS (return stream zeroed
     after a point); show how the trailing-12m-Sharpe gate would respond.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "models"))

from alphabeta import get_candles
from alphabeta.backtest import backtest
import strategies_v2 as S2

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")

# Top8 production portfolio definition.
TOP8 = ["RISKPAR", "TSMOM", "EVE_XAU", "D1REV_UK", "XSMOM",
        "D1REV_NAS", "WED_BTC", "DEFEND"]
GATES_HIGH = {  # high-vol multiplier per sleeve
    "RISKPAR": 0.5, "TSMOM": 0.5, "XSMOM": 0.5, "EVE_XAU": 0.5, "WED_BTC": 0.5,
    "D1REV_NAS": 1.5, "D1REV_UK": 1.5, "DEFEND": 1.5,
}


def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    bpy = _bpy(r.index)
    ar = float(r.mean()) * bpy
    av = float(r.std(ddof=0)) * np.sqrt(bpy)
    eq = (1 + r).cumprod()
    out["sharpe"] = ar / av if av > 0 else 0.0
    out["ann_ret"] = ar
    out["ann_vol"] = av
    out["maxdd"] = float((eq / eq.cummax() - 1).min())
    out["total_ret"] = float(eq.iloc[-1] - 1)
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
    return pd.Series((aligned["rv30"] > cutoff).fillna(False).values, index=idx)


def apply_regime_gate(panel: pd.DataFrame, high_vol: pd.Series) -> pd.DataFrame:
    g = panel.copy()
    for sleeve, mult in GATES_HIGH.items():
        if sleeve in g.columns:
            g.loc[high_vol, sleeve] *= mult
    return g


def build_top8_portfolio(panel: pd.DataFrame) -> pd.Series:
    high_vol = build_regime_mask(panel.index, 80)
    gated = apply_regime_gate(panel, high_vol)
    return gated[TOP8].mean(axis=1)


def test_1_reinject_2022(panel: pd.DataFrame):
    """Replace first 12 months of OOS (2024-01 → 2024-12) with the 2022 returns
    of each sleeve. See how the portfolio handles a 'second 2022' in OOS.
    """
    print("=" * 70)
    print("TEST 1: re-inject 2022 returns into 2024 to simulate a 'second 2022'")
    print("=" * 70)

    # Slice 2022 returns per sleeve
    y2022 = panel[panel.index.year == 2022]
    if y2022.empty:
        print("(no 2022 data)")
        return

    # Replace the first len(y2022) bars of 2024 with 2022 data
    panel_stress = panel.copy()
    oos_start_idx = panel_stress.index.searchsorted(SPLIT)
    n = min(len(y2022), len(panel_stress) - oos_start_idx)
    replace_idx = panel_stress.index[oos_start_idx : oos_start_idx + n]
    src_idx = y2022.index[:n]
    for c in panel_stress.columns:
        panel_stress.loc[replace_idx, c] = y2022[c].iloc[:n].values

    # Build portfolio on the stressed panel
    portfolio = build_top8_portfolio(panel_stress)
    baseline = build_top8_portfolio(panel)

    print(f"\n               {'stressed':>10}  {'baseline':>10}")
    print(f"               {'(2024=2022)':>10}  {'(actual)':>10}")
    print("-" * 40)
    stressed_first_year = portfolio[(portfolio.index >= SPLIT) &
                                     (portfolio.index < SPLIT + pd.Timedelta(days=365))]
    base_first_year = baseline[(baseline.index >= SPLIT) &
                                (baseline.index < SPLIT + pd.Timedelta(days=365))]
    s_st = stats("stressed", stressed_first_year)
    s_bs = stats("baseline", base_first_year)
    for k in ["sharpe", "ann_ret", "ann_vol", "maxdd"]:
        v_st = s_st[k]
        v_bs = s_bs[k]
        fmt = "{:+.2%}" if k in ("ann_ret", "ann_vol", "maxdd") else "{:+.2f}"
        print(f"  {k:<10}    {fmt.format(v_st):>10}    {fmt.format(v_bs):>10}")

    # Full-OOS comparison
    print(f"\nFull-OOS portfolio comparison:")
    print(f"{'metric':<10} {'stressed':>10} {'baseline':>10}")
    p_st = portfolio[portfolio.index >= SPLIT]
    p_bs = baseline[baseline.index >= SPLIT]
    s_st_full = stats("OOS stressed", p_st)
    s_bs_full = stats("OOS baseline", p_bs)
    for k in ["sharpe", "ann_ret", "ann_vol", "maxdd"]:
        v_st = s_st_full[k]
        v_bs = s_bs_full[k]
        fmt = "{:+.2%}" if k in ("ann_ret", "ann_vol", "maxdd") else "{:+.2f}"
        print(f"  {k:<8} {fmt.format(v_st):>10} {fmt.format(v_bs):>10}")


def test_2_bootstrap(panel: pd.DataFrame, n_trials: int = 1000):
    """Monte-Carlo bootstrap: sample a random contiguous 12-month block from
    the IS period 1000 times and see the distribution of annual Sharpe."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Monte-Carlo bootstrap — {n_trials} trials of 12-month windows")
    print("=" * 70)

    is_panel = panel[panel.index < SPLIT]
    portfolio_is = build_top8_portfolio(is_panel)

    rng = np.random.default_rng(42)
    sharpes = []
    returns = []
    drawdowns = []
    n_bars_per_year = 365
    valid_starts = len(portfolio_is) - n_bars_per_year
    if valid_starts <= 0:
        print("(not enough IS data for a 12-month bootstrap)")
        return
    for _ in range(n_trials):
        start = rng.integers(0, valid_starts)
        window = portfolio_is.iloc[start : start + n_bars_per_year]
        bpy = _bpy(window.index)
        ar = float(window.mean()) * bpy
        av = float(window.std(ddof=0)) * np.sqrt(bpy)
        sh = ar / av if av > 0 else 0
        eq = (1 + window).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        sharpes.append(sh)
        returns.append(ar)
        drawdowns.append(dd)

    sharpes = np.array(sharpes); returns = np.array(returns); drawdowns = np.array(drawdowns)
    print(f"\nDistribution of 12-month-window outcomes:")
    print(f"  Sharpe:     mean={sharpes.mean():+.2f}  std={sharpes.std():.2f}  "
          f"5%={np.quantile(sharpes,0.05):+.2f}  median={np.median(sharpes):+.2f}  "
          f"95%={np.quantile(sharpes,0.95):+.2f}")
    print(f"  Ann ret:    mean={returns.mean():+.2%}  std={returns.std():.2%}  "
          f"5%={np.quantile(returns,0.05):+.2%}  95%={np.quantile(returns,0.95):+.2%}")
    print(f"  MaxDD:      mean={drawdowns.mean():+.2%}  worst={drawdowns.min():+.2%}  "
          f"5%={np.quantile(drawdowns,0.05):+.2%}")
    print(f"\nP(negative Sharpe) = {(sharpes < 0).mean():.1%}")
    print(f"P(Sharpe > 1.0)    = {(sharpes > 1.0).mean():.1%}")
    print(f"P(Sharpe > 2.0)    = {(sharpes > 2.0).mean():.1%}")

    pd.DataFrame({"sharpe": sharpes, "ann_ret": returns, "maxdd": drawdowns}).to_csv(
        OUT / "stress_bootstrap.csv", index=False)


def test_3_decay_detection(panel: pd.DataFrame):
    """Simulate a sleeve dying mid-OOS and show how a trailing-12m-Sharpe
    gate would catch it. Useful to size the operational tripwire."""
    print("\n" + "=" * 70)
    print("TEST 3: decay detection sensitivity — how fast does WF_GATE catch a dead sleeve?")
    print("=" * 70)

    # Pick a sleeve and zero it from 2024-06-01 onward
    target = "TSMOM"
    kill_date = pd.Timestamp("2024-06-01", tz="UTC")
    if target not in panel.columns:
        print(f"(sleeve {target} not in panel)")
        return
    stressed = panel.copy()
    stressed.loc[stressed.index >= kill_date, target] = 0.0

    # Compute trailing-252-day Sharpe each day, see how it evolves
    sharpe_baseline = panel[target].rolling(252).mean() * 365 / (
        panel[target].rolling(252).std(ddof=0) * np.sqrt(365))
    sharpe_stressed = stressed[target].rolling(252).mean() * 365 / (
        stressed[target].rolling(252).std(ddof=0) * np.sqrt(365))

    # Find when the trailing Sharpe crosses below 0
    after = sharpe_stressed[sharpe_stressed.index >= kill_date]
    crossed = after[after < 0]
    if len(crossed) > 0:
        first_neg = crossed.index[0]
        lag_days = (first_neg - kill_date).days
        print(f"  {target} killed on:        {kill_date.date()}")
        print(f"  First neg trailing 12m:   {first_neg.date()}  (lag: {lag_days} days)")
    else:
        print(f"  Trailing 12m never went negative within the OOS window (sleeve too profitable in IS)")

    # Find when sharpe drops by 50% from its pre-kill peak
    pre_peak = sharpe_baseline.loc[:kill_date].max()
    target_level = pre_peak * 0.5
    crossed_half = after[after < target_level]
    if len(crossed_half) > 0:
        first_half = crossed_half.index[0]
        lag_half = (first_half - kill_date).days
        print(f"  Pre-kill peak trailing12: {pre_peak:+.2f}")
        print(f"  Half-of-peak threshold:   {target_level:+.2f}")
        print(f"  First crossing:           {first_half.date()}  (lag: {lag_half} days)")

    out_df = pd.DataFrame({
        "timestamp": panel.index,
        "baseline_trailing_sharpe": sharpe_baseline.values,
        "stressed_trailing_sharpe": sharpe_stressed.values,
    })
    out_df.to_parquet(OUT / "decay_detection.parquet", index=False)


def main():
    panel = pd.read_parquet(OUT / "all_sleeve_returns_v3.parquet")
    panel.index = pd.to_datetime(panel.index, utc=True)
    test_1_reinject_2022(panel)
    test_2_bootstrap(panel)
    test_3_decay_detection(panel)


if __name__ == "__main__":
    main()
