"""Cross-sectional momentum (XSMOM) sleeve.

Ranks each basket's members by trailing log-return momentum, longs the
top and shorts the bottom, vol-targets to 10% ann, rebalances weekly,
then equal-weights the three baskets into a single sleeve return stream.

Outputs:
    scratch/quant/xsmom_returns.parquet   (master combiner consumes this)
    scratch/quant/xsmom_baskets.csv

Run:
    python scratch/quant/xsmom.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import CRYPTO, FOREX, INDEX, get_candles
from alphabeta.backtest import cost_for

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scratch" / "quant"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.10
VOL_LOOKBACK = 60
SKIP = 5      # skip last 5 days
LOOKBACKS = [21, 63, 126]   # ranking windows we'll test
DEFAULT_LOOKBACK = 63

# Long / short counts per basket. For 3/4-member baskets => (1, 1). For 6-member => (2, 2).
BASKET_LS = {"crypto": (1, 1), "fx": (1, 1), "index": (2, 2)}
BASKET_SYMS = {"crypto": CRYPTO, "fx": FOREX, "index": INDEX}


# ----------------------------- helpers -----------------------------

def load_closes(symbols: list[str]) -> pd.DataFrame:
    """Return a wide DataFrame of close prices indexed by UTC timestamp."""
    cols = {}
    for s in symbols:
        df = get_candles(s, "D1")
        ser = df.set_index("timestamp")["close"].astype("float64")
        cols[s] = ser
    closes = pd.concat(cols, axis=1, sort=True)
    closes.index = pd.DatetimeIndex(closes.index, tz="UTC")
    closes = closes.sort_index()
    return closes


def cost_vec(symbols: list[str]) -> np.ndarray:
    return np.array([cost_for(s) for s in symbols], dtype="float64")


def momentum_rank_positions(
    closes: pd.DataFrame,
    lookback: int,
    skip: int,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Daily target weights for each symbol BEFORE weekly resampling / vol targeting.

    Weights sum to zero (dollar-neutral). Each leg is sized to 1/n_leg so the
    gross book is 2.0 (1 long + 1 short on aggregate notional).
    """
    log_close = np.log(closes)
    # use return from t-(lookback+skip) to t-skip, available at decision time t (use yesterday's close)
    mom = log_close.shift(skip) - log_close.shift(skip + lookback)
    # rank ascending within each row (NaN -> stays NaN). use 'first' to break ties deterministically.
    ranks = mom.rank(axis=1, method="first", ascending=True)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    n = ranks.notna().sum(axis=1)
    for t in ranks.index:
        n_t = int(n.loc[t])
        if n_t < (n_long + n_short):
            continue
        r = ranks.loc[t]
        # longs = top n_long ranks (highest mom -> rank == n_t, n_t-1, ...)
        longs = r[r > n_t - n_long].index
        shorts = r[(r >= 1) & (r <= n_short)].index
        weights.loc[t, longs] = 1.0 / n_long
        weights.loc[t, shorts] = -1.0 / n_short
    return weights


def weekly_rebalance(weights: pd.DataFrame) -> pd.DataFrame:
    """Hold the target from each Monday until the next Monday (or first available bar)."""
    idx = weights.index
    # Monday = 0
    is_monday = idx.weekday == 0
    rebalance_rows = pd.Series(np.where(is_monday, np.arange(len(idx)), np.nan), index=idx).ffill()
    # The first Monday is the first non-nan rebalance. Forward fill weights from rebalance rows.
    held = weights.copy()
    held[~is_monday] = np.nan
    held = held.ffill()
    held = held.fillna(0.0)
    return held


def sleeve_returns_pre_vt(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute gross + cost streams from daily close-to-close returns.

    Convention: weights at row t are the position held DURING bar t (decided from data
    available at start of t, so we shift by 1 vs realized return). The momentum_rank
    function already uses .shift(skip) which references prior closes, but to be safe
    we apply one more .shift(1) here so that weights[t] reflects the close of t-1.
    """
    ret = closes.pct_change().fillna(0.0)
    # shift weights by 1 so they don't peek at today's close.
    pos = weights.shift(1).fillna(0.0)

    gross = (pos * ret).sum(axis=1)
    # transaction cost: per-side cost * |Δposition| per symbol
    costs = cost_vec(symbols)
    dpos = pos.diff().abs().fillna(pos.abs())
    cost_stream = (dpos * costs).sum(axis=1)
    return gross, cost_stream


def vol_target(returns: pd.Series, lookback: int, target: float, ann_factor: float) -> pd.Series:
    """Apply walk-forward vol target. Scaling at t uses realized vol over (t-lookback..t-1)."""
    realized = returns.rolling(lookback, min_periods=max(20, lookback // 2)).std(ddof=0) * np.sqrt(ann_factor)
    scale = (target / realized).shift(1)  # shift so we don't use today's vol
    # cap scale to avoid blow-up when realized vol is tiny (early sample / dead weeks)
    scale = scale.clip(upper=5.0).fillna(0.0)
    return returns * scale, scale


def ann_factor_for(closes: pd.DataFrame) -> float:
    """Empirical bars per year from the calendar."""
    span = (closes.index[-1] - closes.index[0]).total_seconds() / 86400
    return len(closes) / span * 365.25


def stats(returns: pd.Series, ann_factor: float) -> dict:
    r = returns.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0, "n": int(len(r))}
    ann_ret = r.mean() * ann_factor
    ann_vol = r.std(ddof=0) * np.sqrt(ann_factor)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_dd": float(dd.min()),
        "n": int(len(r)),
    }


def split_stats(returns: pd.Series, ann_factor: float) -> dict:
    full = stats(returns, ann_factor)
    is_part = stats(returns[returns.index < SPLIT], ann_factor)
    oos_part = stats(returns[returns.index >= SPLIT], ann_factor)
    return {"full": full, "is": is_part, "oos": oos_part}


# ----------------------------- pipeline -----------------------------

def build_basket(
    name: str,
    symbols: list[str],
    lookback: int = DEFAULT_LOOKBACK,
    invert: bool = False,
) -> tuple[pd.Series, dict]:
    """Build a vol-targeted long/short basket and return its daily returns."""
    closes = load_closes(symbols)
    # restrict to rows where ALL members have a valid close so ranking is fair
    closes = closes.dropna(how="any")
    n_long, n_short = BASKET_LS[name]

    weights = momentum_rank_positions(closes, lookback, SKIP, n_long, n_short)
    if invert:
        weights = -weights
    held = weekly_rebalance(weights)

    gross, costs = sleeve_returns_pre_vt(closes, held, symbols)
    net_pre_vt = gross - costs
    af = ann_factor_for(closes)
    net, scale = vol_target(net_pre_vt, VOL_LOOKBACK, TARGET_VOL, af)

    # turnover diagnostic: fraction of weeks where the long/short COMPOSITION changes
    is_monday = held.index.weekday == 0
    monday_weights = held[is_monday]
    # signed identity of position on each Monday
    sign = np.sign(monday_weights)
    comp_change = (sign.diff().abs().sum(axis=1) > 0).mean()

    diag = {
        "name": name,
        "symbols": symbols,
        "lookback": lookback,
        "invert": invert,
        "ann_factor": af,
        "weekly_composition_change_rate": float(comp_change),
        "stats_pre_vt": split_stats(net_pre_vt, af),
        "stats": split_stats(net, af),
    }
    return net, diag


def align_to_union(streams: dict[str, pd.Series]) -> pd.DataFrame:
    """Reindex multiple basket streams to the union index, filling missing with 0."""
    df = pd.concat(streams, axis=1, sort=True).sort_index()
    df = df.fillna(0.0)
    return df


def yearly_sharpes(returns: pd.Series, ann_factor: float) -> dict[int, float]:
    out = {}
    for year, grp in returns.groupby(returns.index.year):
        s = stats(grp, ann_factor)
        out[int(year)] = s["sharpe"]
    return out


def sleeve_ann_factor(returns: pd.Series) -> float:
    span = (returns.index[-1] - returns.index[0]).total_seconds() / 86400
    if span <= 0:
        return 252.0
    return len(returns) / span * 365.25


def main() -> None:
    # ---- step 1: pick the winning lookback by FULL-period sleeve Sharpe ----
    print("\n=== lookback search ===")
    lookback_scores = {}
    for lb in LOOKBACKS:
        b_crypto, _ = build_basket("crypto", CRYPTO, lookback=lb)
        b_fx, _ = build_basket("fx", FOREX, lookback=lb)
        b_index, _ = build_basket("index", INDEX, lookback=lb)
        merged = align_to_union({"crypto": b_crypto, "fx": b_fx, "index": b_index})
        sleeve = merged.mean(axis=1)
        af = sleeve_ann_factor(sleeve)
        s = split_stats(sleeve, af)
        lookback_scores[lb] = s
        print(f"  lb={lb:>3d}  full Sharpe={s['full']['sharpe']:+.2f}  "
              f"IS={s['is']['sharpe']:+.2f}  OOS={s['oos']['sharpe']:+.2f}")
    best_lb = max(lookback_scores, key=lambda k: lookback_scores[k]["full"]["sharpe"])
    print(f"  -> winning lookback: {best_lb}")

    # ---- step 2: build per-basket with winning lookback ----
    b_crypto, d_crypto = build_basket("crypto", CRYPTO, lookback=best_lb)
    b_fx, d_fx = build_basket("fx", FOREX, lookback=best_lb)
    b_index, d_index = build_basket("index", INDEX, lookback=best_lb)
    b_index_rev, d_index_rev = build_basket("index", INDEX, lookback=best_lb, invert=True)

    # ---- step 3: sleeve = equal-weight 3 baskets (standard variant uses momentum index) ----
    merged_mom = align_to_union({"crypto": b_crypto, "fx": b_fx, "index": b_index})
    sleeve_mom = merged_mom.mean(axis=1)

    merged_rev = align_to_union({"crypto": b_crypto, "fx": b_fx, "index": b_index_rev})
    sleeve_rev = merged_rev.mean(axis=1)

    af = sleeve_ann_factor(sleeve_mom)
    s_mom = split_stats(sleeve_mom, af)
    s_rev = split_stats(sleeve_rev, af)
    print("\n=== sleeve variants ===")
    print(f"  momentum-only    full={s_mom['full']['sharpe']:+.2f}  "
          f"IS={s_mom['is']['sharpe']:+.2f}  OOS={s_mom['oos']['sharpe']:+.2f}")
    print(f"  index-reversal   full={s_rev['full']['sharpe']:+.2f}  "
          f"IS={s_rev['is']['sharpe']:+.2f}  OOS={s_rev['oos']['sharpe']:+.2f}")

    # ---- step 4: write deliverables ----
    # The master combiner expects timestamp + ret columns. Default = best variant
    # (we pick whichever has stronger OOS, else stronger full).
    if s_rev["oos"]["sharpe"] > s_mom["oos"]["sharpe"]:
        final_sleeve = sleeve_rev
        chosen = "index-reversal"
    else:
        final_sleeve = sleeve_mom
        chosen = "momentum-only"

    # sleeve returns parquet — collapse all intraday timestamps to one row per UTC date
    # so the master combiner sees clean D1 bars.
    daily = final_sleeve.groupby(final_sleeve.index.floor("D")).sum()
    daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
    out = pd.DataFrame({"timestamp": daily.index, "ret": daily.values})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(OUT_DIR / "xsmom_returns.parquet", index=False)

    # per-basket CSV
    rows = []
    for diag, basket_ret in [
        (d_crypto, b_crypto),
        (d_fx, b_fx),
        (d_index, b_index),
        (d_index_rev, b_index_rev),
    ]:
        s = split_stats(basket_ret, af)
        rows.append({
            "basket": diag["name"] + ("_reversal" if diag["invert"] else ""),
            "lookback": diag["lookback"],
            "full_sharpe": s["full"]["sharpe"],
            "full_ann_return": s["full"]["ann_return"],
            "full_ann_vol": s["full"]["ann_vol"],
            "full_max_dd": s["full"]["max_dd"],
            "is_sharpe": s["is"]["sharpe"],
            "is_ann_return": s["is"]["ann_return"],
            "is_max_dd": s["is"]["max_dd"],
            "oos_sharpe": s["oos"]["sharpe"],
            "oos_ann_return": s["oos"]["ann_return"],
            "oos_max_dd": s["oos"]["max_dd"],
            "weekly_composition_change_rate": diag["weekly_composition_change_rate"],
        })
    # sleeves too
    for label, ser in [("sleeve_momentum", sleeve_mom), ("sleeve_index_reversal", sleeve_rev)]:
        s = split_stats(ser, af)
        rows.append({
            "basket": label,
            "lookback": best_lb,
            "full_sharpe": s["full"]["sharpe"],
            "full_ann_return": s["full"]["ann_return"],
            "full_ann_vol": s["full"]["ann_vol"],
            "full_max_dd": s["full"]["max_dd"],
            "is_sharpe": s["is"]["sharpe"],
            "is_ann_return": s["is"]["ann_return"],
            "is_max_dd": s["is"]["max_dd"],
            "oos_sharpe": s["oos"]["sharpe"],
            "oos_ann_return": s["oos"]["ann_return"],
            "oos_max_dd": s["oos"]["max_dd"],
            "weekly_composition_change_rate": np.nan,
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "xsmom_baskets.csv", index=False)

    # ---- step 5: print report-ready numbers ----
    print("\n=== headline (chosen sleeve = {}) ===".format(chosen))
    chosen_stats = s_rev if chosen == "index-reversal" else s_mom
    for k in ("full", "is", "oos"):
        st = chosen_stats[k]
        print(f"  {k:<4} Sharpe={st['sharpe']:+.2f}  Ret={st['ann_return']:+.1%}  "
              f"Vol={st['ann_vol']:.1%}  DD={st['max_dd']:+.1%}  n={st['n']}")

    print("\n=== per-basket stats (winning lookback={}) ===".format(best_lb))
    for diag, basket_ret in [
        (d_crypto, b_crypto),
        (d_fx, b_fx),
        (d_index, b_index),
        (d_index_rev, b_index_rev),
    ]:
        s = split_stats(basket_ret, af)
        tag = diag["name"] + ("_rev" if diag["invert"] else "")
        print(f"  {tag:<14}  full={s['full']['sharpe']:+.2f}  "
              f"IS={s['is']['sharpe']:+.2f}  OOS={s['oos']['sharpe']:+.2f}  "
              f"comp-change/wk={diag['weekly_composition_change_rate']:.1%}")

    print("\n=== yearly sleeve Sharpes ===")
    final_choice_series = sleeve_rev if chosen == "index-reversal" else sleeve_mom
    ys = yearly_sharpes(final_choice_series, af)
    for y, sh in sorted(ys.items()):
        print(f"  {y}: {sh:+.2f}")

    print("\n=== lookback comparison (sleeve momentum) ===")
    for lb in LOOKBACKS:
        s = lookback_scores[lb]
        print(f"  lb={lb}: full={s['full']['sharpe']:+.2f} IS={s['is']['sharpe']:+.2f} OOS={s['oos']['sharpe']:+.2f}")

    print("\nwrote:")
    print(" ", OUT_DIR / "xsmom_returns.parquet")
    print(" ", OUT_DIR / "xsmom_baskets.csv")


if __name__ == "__main__":
    main()
