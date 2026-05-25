"""Relative-strength asset selection (wave3).

Six finer-grained relative-strength / mean-reversion / dual-momentum
sub-sleeves. The premise is different from the existing XSMOM sleeve:
instead of one cross-sectional momentum signal at a fixed cadence, we
mix several adaptive-rebalance ranking schemes and let the survivor
filter prune what doesn't work.

Sub-sleeves built per the spec:
  1. adaptive_top1        — top-1 by 63d Sharpe, all 13 syms, weekly
  1b. adaptive_top3       — same but top-3 holdings (comparison)
  2. relstrength_basket   — long-strongest / short-weakest within each
                            basket by 63d return vs basket mean. Biweekly.
  3. continuous_rotation  — weights proportional to max(0, per-sym 63d
                            Sharpe), renormalized to sum 1. Weekly.
  4. meanrev_top1         — long the worst 21d performer (per basket),
                            5-day hold.
  5. dual_momentum        — abs-mom filter (252d > 0) AND rel-mom top-1
                            per basket. Cash when nothing qualifies.
  6. pair_momentum        — top-3 most stable correlated pairs; long the
                            stronger / short the weaker within each pair.

Methodology:
  - D1, 13 symbols, 2020-01-01..2026-05-24
  - IS < 2024-01-01, OOS >= 2024-01-01
  - Walk-forward all rankings (no peeking)
  - Vol-scale each sub-sleeve to 5% IS ann vol
  - Keep sub-sleeves with IS Sharpe >= 0.5 AND OOS Sharpe >= 0
  - Combine survivors equal-weight
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import cost_for

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
TRADING_DAYS = 252

BASKETS = {"crypto": CRYPTO, "fx": FOREX, "index": INDEX}
ALL_SYMS = CRYPTO + FOREX + INDEX


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_closes() -> pd.DataFrame:
    """Wide D1 close-price frame, UTC date index, NaN where a market is shut."""
    cols = {}
    for s in ALL_SYMS:
        df = get_candles(s, "D1")
        ser = df.set_index("timestamp")["close"].astype("float64")
        ser.index = pd.DatetimeIndex(ser.index, tz="UTC").floor("D")
        ser = ser[~ser.index.duplicated(keep="last")]
        cols[s] = ser
    out = pd.concat(cols, axis=1, sort=True)
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out.sort_index()


COST_VEC = np.array([cost_for(s) for s in ALL_SYMS], dtype="float64")


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def ann_stats(r: pd.Series, freq: int = TRADING_DAYS) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
    av = float(r.std(ddof=0) * np.sqrt(freq))
    ar = float(r.mean() * freq)
    sh = ar / av if av > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": sh, "ann_ret": ar, "ann_vol": av, "max_dd": dd}


def split_stats(r: pd.Series) -> dict:
    full = ann_stats(r)
    is_r = r[r.index < SPLIT]
    oos_r = r[r.index >= SPLIT]
    y22 = r[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
            (r.index < pd.Timestamp("2023-01-01", tz="UTC"))]
    y23 = r[(r.index >= pd.Timestamp("2023-01-01", tz="UTC")) &
            (r.index < pd.Timestamp("2024-01-01", tz="UTC"))]
    return {
        "full_sharpe": full["sharpe"],
        "full_ann_ret": full["ann_ret"],
        "full_vol": full["ann_vol"],
        "full_dd": full["max_dd"],
        "is_sharpe": ann_stats(is_r)["sharpe"],
        "is_vol": ann_stats(is_r)["ann_vol"],
        "oos_sharpe": ann_stats(oos_r)["sharpe"],
        "oos_vol": ann_stats(oos_r)["ann_vol"],
        "y2022_sharpe": ann_stats(y22)["sharpe"],
        "y2023_sharpe": ann_stats(y23)["sharpe"],
    }


# -----------------------------------------------------------------------------
# Rebalance scheduling helpers
# -----------------------------------------------------------------------------
def rebalance_hold(weights: pd.DataFrame, every: int) -> pd.DataFrame:
    """Hold weights from each rebalance day for `every` calendar days.

    weights is a wide frame indexed by date with per-symbol target weights
    computed using data available at the start of that day (i.e. already
    causally lagged by the builder). We sample weights every `every` rows
    and ffill in between.
    """
    held = weights.copy()
    keep = np.zeros(len(held), dtype=bool)
    keep[::every] = True
    held[~keep] = np.nan
    held = held.ffill().fillna(0.0)
    return held


def rebalance_weekly_monday(weights: pd.DataFrame) -> pd.DataFrame:
    """Rebalance only on Mondays — ffill between."""
    held = weights.copy()
    is_mon = held.index.weekday == 0
    held[~is_mon] = np.nan
    held = held.ffill().fillna(0.0)
    return held


# -----------------------------------------------------------------------------
# Returns + cost from a wide weights frame
# -----------------------------------------------------------------------------
def returns_and_costs(closes: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Net return stream. weights[t] is the position held DURING bar t — we
    shift by 1 here so today's weight reflects yesterday's close info.
    Costs charged on |Δw| per symbol per bar.
    """
    # Make sure column order matches COST_VEC
    weights = weights.reindex(columns=ALL_SYMS).fillna(0.0)
    closes = closes.reindex(columns=ALL_SYMS)
    ret = closes.pct_change().fillna(0.0)

    pos = weights.shift(1).fillna(0.0)
    gross = (pos * ret).sum(axis=1)
    dpos = pos.diff().abs().fillna(pos.abs())
    cost_stream = (dpos.values * COST_VEC).sum(axis=1)
    cost_stream = pd.Series(cost_stream, index=closes.index)
    return gross - cost_stream


def vol_scale_to_is(ret: pd.Series, target: float = TARGET_VOL) -> tuple[pd.Series, float]:
    """Scale entire return series by a constant chosen so IS ann vol == target."""
    is_r = ret[ret.index < SPLIT].dropna()
    if len(is_r) < 60:
        return ret * 0.0, 0.0
    iv = float(is_r.std(ddof=0) * np.sqrt(TRADING_DAYS))
    if iv < 1e-6:
        return ret * 0.0, 0.0
    scale = float(np.clip(target / iv, 0.0, 10.0))
    return ret * scale, scale


# -----------------------------------------------------------------------------
# Sub-sleeve 1: adaptive top-K by trailing 63d Sharpe over all 13 symbols.
# -----------------------------------------------------------------------------
def build_adaptive_topk(closes: pd.DataFrame, k: int, lookback: int = 63) -> pd.DataFrame:
    """For each day, rank all symbols by trailing `lookback`-day Sharpe and
    long the top-k equally weighted. Long-only. Rebalanced weekly.
    """
    log_ret = np.log(closes / closes.shift(1))
    roll_mean = log_ret.rolling(lookback, min_periods=lookback).mean()
    roll_std = log_ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)
    # decision uses info up to t-1
    sharpe_lag = sharpe.shift(1)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    rank = sharpe_lag.rank(axis=1, method="first", ascending=False)
    # top-k symbols where rank in [1..k]
    mask = (rank >= 1) & (rank <= k)
    weights[mask] = 1.0 / k
    # rows where not enough valid data -> all zero
    valid = sharpe_lag.notna().sum(axis=1) >= k
    weights[~valid] = 0.0
    held = rebalance_weekly_monday(weights)
    return held


# -----------------------------------------------------------------------------
# Sub-sleeve 2: relative strength vs basket mean. Per-basket long/short.
# -----------------------------------------------------------------------------
def build_relstrength_basket(closes: pd.DataFrame, lookback: int = 63,
                              hold_days: int = 10) -> pd.DataFrame:
    """For each basket, compute each symbol's trailing `lookback`-day log
    return minus the basket-mean. Long the strongest, short the weakest.
    Rebalance roughly biweekly (every `hold_days` calendar days).
    """
    log_close = np.log(closes)
    ret_lb = (log_close - log_close.shift(lookback)).shift(1)  # causal

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for name, syms in BASKETS.items():
        sub = ret_lb[syms]
        basket_mean = sub.mean(axis=1)
        rel = sub.sub(basket_mean, axis=0)
        # find strongest & weakest per day
        for t in rel.index:
            row = rel.loc[t].dropna()
            if len(row) < 2:
                continue
            strongest = row.idxmax()
            weakest = row.idxmin()
            weights.at[t, strongest] = 0.5  # 50% long
            weights.at[t, weakest] = -0.5   # 50% short
    held = rebalance_hold(weights, every=hold_days)
    return held


# -----------------------------------------------------------------------------
# Sub-sleeve 3: continuous rotation. Weights ∝ max(0, per-sym Sharpe).
# -----------------------------------------------------------------------------
def build_continuous_rotation(closes: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    log_ret = np.log(closes / closes.shift(1))
    roll_mean = log_ret.rolling(lookback, min_periods=lookback).mean()
    roll_std = log_ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)
    score = sharpe.shift(1).clip(lower=0.0).fillna(0.0)  # causal, drop negatives
    row_sum = score.sum(axis=1).replace(0.0, np.nan)
    weights = score.div(row_sum, axis=0).fillna(0.0)
    # weekly rebalance — weights only change on Mondays
    held = rebalance_weekly_monday(weights)
    return held


# -----------------------------------------------------------------------------
# Sub-sleeve 4: mean-reversion top-1 per basket. Hold for 5 days.
# -----------------------------------------------------------------------------
def build_meanrev_top1(closes: pd.DataFrame, lookback: int = 21,
                        hold_days: int = 5) -> pd.DataFrame:
    """Per basket, each rebalance window long the worst trailing-21d performer.
    5-day hold.
    """
    log_close = np.log(closes)
    ret_lb = (log_close - log_close.shift(lookback)).shift(1)  # causal

    # weight per basket is 1/3 (3 baskets); inside basket 1.0 on worst
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    n_baskets = len(BASKETS)
    basket_w = 1.0 / n_baskets
    for name, syms in BASKETS.items():
        sub = ret_lb[syms]
        for t in sub.index:
            row = sub.loc[t].dropna()
            if len(row) < 2:
                continue
            worst = row.idxmin()
            weights.at[t, worst] = basket_w
    held = rebalance_hold(weights, every=hold_days)
    return held


# -----------------------------------------------------------------------------
# Sub-sleeve 5: dual momentum. Abs-mom + rel-mom per basket. Cash if none.
# -----------------------------------------------------------------------------
def build_dual_momentum(closes: pd.DataFrame, abs_lb: int = 252, rel_lb: int = 63) -> pd.DataFrame:
    log_close = np.log(closes)
    abs_ret = (log_close - log_close.shift(abs_lb)).shift(1)  # causal
    rel_ret = (log_close - log_close.shift(rel_lb)).shift(1)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    basket_w = 1.0 / len(BASKETS)
    for name, syms in BASKETS.items():
        sub_abs = abs_ret[syms]
        sub_rel = rel_ret[syms]
        qualified = sub_abs > 0
        # mask rel return where not qualified
        rel_masked = sub_rel.where(qualified, np.nan)
        for t in rel_masked.index:
            row = rel_masked.loc[t].dropna()
            if len(row) == 0:
                continue  # cash
            best = row.idxmax()
            weights.at[t, best] = basket_w
    # rebalance monthly (every 21 trading days)
    held = rebalance_hold(weights, every=21)
    return held


# -----------------------------------------------------------------------------
# Sub-sleeve 6: pair momentum. Find top-3 most stable correlated pairs,
# long stronger / short weaker.
# -----------------------------------------------------------------------------
def build_pair_momentum(closes: pd.DataFrame, mom_lb: int = 63,
                         corr_lb: int = 126, hold_days: int = 10) -> pd.DataFrame:
    """Pair-momentum sleeve. Walk-forward:
      1. compute rolling 126-day correlation between every pair on log-returns
      2. measure stability = mean(rolling corr) - 2 * std(rolling corr) over a
         recent window. High = consistently correlated.
      3. Pick top-3 stable pairs. For each pair, long the stronger asset
         (higher 63d return) / short the weaker.
      4. Rebalance every 10 days.

    We pick the top-3 stable pairs ONCE based on the IS window so the OOS
    test is honest; that's a hyperparameter chosen on training data only.
    """
    log_ret = np.log(closes / closes.shift(1)).dropna(how="all")
    is_log_ret = log_ret[log_ret.index < SPLIT]

    # stability over IS
    pair_scores = []
    syms = closes.columns.tolist()
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            joint = is_log_ret[[a, b]].dropna()
            if len(joint) < corr_lb * 2:
                continue
            rc = joint[a].rolling(corr_lb, min_periods=corr_lb).corr(joint[b]).dropna()
            if len(rc) < 60:
                continue
            mean_c = rc.mean()
            std_c = rc.std(ddof=0)
            # require positive correlation (so "stronger/weaker" is meaningful)
            score = mean_c - 2 * std_c
            if mean_c < 0.2:
                continue
            pair_scores.append((a, b, mean_c, std_c, score))

    pair_scores.sort(key=lambda x: -x[4])
    top_pairs = pair_scores[:3]
    print(f"    top-3 pairs (chosen on IS only):")
    for a, b, m, s, sc in top_pairs:
        print(f"      {a:<10} <-> {b:<10}  mean_corr={m:+.2f}  std={s:.2f}  score={sc:+.2f}")

    if not top_pairs:
        return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    log_close = np.log(closes)
    mom = (log_close - log_close.shift(mom_lb)).shift(1)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    # weight: 1/(3 pairs); each leg = 0.5 of pair = 1/6 of book
    pair_w = 1.0 / len(top_pairs)
    leg_w = pair_w * 0.5
    for a, b, *_ in top_pairs:
        # at each t, long the higher-mom, short the lower-mom
        sign_a = np.sign(mom[a] - mom[b])
        weights[a] = weights[a] + sign_a.fillna(0.0) * leg_w
        weights[b] = weights[b] - sign_a.fillna(0.0) * leg_w
    held = rebalance_hold(weights, every=hold_days)
    return held


# -----------------------------------------------------------------------------
# Run one sub-sleeve and produce a daily return Series
# -----------------------------------------------------------------------------
def run_sleeve(name: str, closes: pd.DataFrame, builder, **kwargs) -> pd.Series:
    w = builder(closes, **kwargs)
    raw = returns_and_costs(closes, w)
    scaled, scale = vol_scale_to_is(raw)
    return scaled


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main():
    print("Wave3: relative-strength asset selection")
    print(f"  Symbols: {len(ALL_SYMS)}")
    print(f"  Split:   {SPLIT.date()}")
    print(f"  Target:  {TARGET_VOL:.1%} ann vol per sub-sleeve (IS-fit)")
    print()

    closes = load_closes()
    print(f"  Loaded closes: {closes.shape}")
    print(f"  Date range: {closes.index[0].date()} -> {closes.index[-1].date()}")
    print()

    sleeves_specs = [
        ("adaptive_top1",        build_adaptive_topk,     {"k": 1, "lookback": 63}),
        ("adaptive_top3",        build_adaptive_topk,     {"k": 3, "lookback": 63}),
        ("relstrength_basket",   build_relstrength_basket,{"lookback": 63, "hold_days": 10}),
        ("continuous_rotation",  build_continuous_rotation,{"lookback": 63}),
        ("meanrev_top1",         build_meanrev_top1,      {"lookback": 21, "hold_days": 5}),
        ("dual_momentum",        build_dual_momentum,     {"abs_lb": 252, "rel_lb": 63}),
        ("pair_momentum",        build_pair_momentum,     {"mom_lb": 63, "corr_lb": 126, "hold_days": 10}),
    ]

    sleeve_returns: dict[str, pd.Series] = {}
    rows = []
    for name, builder, kw in sleeves_specs:
        print(f"  building {name} ...")
        r = run_sleeve(name, closes, builder, **kw)
        r.name = name
        sleeve_returns[name] = r
        s = split_stats(r)
        rows.append({"sleeve": name, **s})
        print(f"    IS={s['is_sharpe']:+.2f}  OOS={s['oos_sharpe']:+.2f}  "
              f"Y22={s['y2022_sharpe']:+.2f}  DD={s['full_dd']:+.1%}  "
              f"FULL={s['full_sharpe']:+.2f}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "rel_strength_breakdown.csv", index=False)
    print(f"\nWrote {OUT / 'rel_strength_breakdown.csv'}")

    # Survivor filter: IS >= 0.5 AND OOS >= 0
    keep_mask = (breakdown["is_sharpe"] >= 0.5) & (breakdown["oos_sharpe"] >= 0)
    survivors = breakdown[keep_mask].copy()
    print(f"\nSurvivors (IS>=0.5 AND OOS>=0): {len(survivors)} / {len(breakdown)}")
    if len(survivors):
        for _, row in survivors.iterrows():
            print(f"  KEEP {row['sleeve']:<22}  IS={row['is_sharpe']:+.2f}  "
                  f"OOS={row['oos_sharpe']:+.2f}  FULL={row['full_sharpe']:+.2f}")

    # Combine survivors equal-weight
    if len(survivors):
        keys = survivors["sleeve"].tolist()
        aligned = pd.concat([sleeve_returns[k] for k in keys], axis=1).fillna(0.0)
        combined = aligned.mean(axis=1)
    else:
        # No survivors -> empty sleeve
        combined = pd.Series(0.0, index=closes.index)

    combined.name = "ret"
    combined.index.name = "timestamp"

    # Persist sleeve returns parquet (one combined daily series, master-combiner shape)
    df_out = combined.reset_index()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], utc=True)
    df_out.to_parquet(OUT / "rel_strength_returns.parquet", index=False)
    print(f"Wrote {OUT / 'rel_strength_returns.parquet'}  ({len(df_out)} rows)")

    # Headline
    print("\nHEADLINE (combined survivors)")
    print("-" * 60)
    s = split_stats(combined)
    print(f"  FULL Sharpe: {s['full_sharpe']:+.2f}  Vol={s['full_vol']:.1%}  DD={s['full_dd']:+.1%}")
    print(f"  IS   Sharpe: {s['is_sharpe']:+.2f}  Vol={s['is_vol']:.1%}")
    print(f"  OOS  Sharpe: {s['oos_sharpe']:+.2f}  Vol={s['oos_vol']:.1%}")
    print(f"  2022 Sharpe: {s['y2022_sharpe']:+.2f}")
    print(f"  2023 Sharpe: {s['y2023_sharpe']:+.2f}")

    # Yearly
    print("\nYearly Sharpes (combined):")
    for y, grp in combined.groupby(combined.index.year):
        ys = ann_stats(grp)
        print(f"  {y}: Sharpe={ys['sharpe']:+.2f}  Ret={ys['ann_ret']:+.1%}  Vol={ys['ann_vol']:.1%}")


if __name__ == "__main__":
    main()
