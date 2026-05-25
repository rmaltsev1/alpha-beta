"""Asset-class rotation strategy.

Three baskets: crypto / forex(safe-haven) / indices. Each month, rank baskets
by trailing 63d Sharpe of their equal-weight basket return series. Allocate
to top-K (K=1, 2, or 3). Within the chosen basket(s), equal-weight constituents.
Vol-target portfolio to 10% ann (walk-forward 60d realized).

Also test: risk-off override. If indices basket trailing 21d return < -5%,
force allocation to FX (safe-haven).

Compares variants vs:
  - Equal-weight buy-and-hold across all 13
  - Crypto-only buy-and-hold
  - Indices-only buy-and-hold

Outputs:
  scratch/quant/rotation_returns.parquet   (best variant)
  scratch/quant/rotation_comparison.csv

Run:
  python scratch/quant/rotation.py
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
SHARPE_LOOKBACK = 63
RISK_OFF_LOOKBACK = 21
RISK_OFF_THRESHOLD = -0.05
ANN = 252.0

BASKETS = {"crypto": CRYPTO, "fx": FOREX, "index": INDEX}
BASKET_NAMES = list(BASKETS.keys())


# ----------------------------- helpers -----------------------------

def load_closes(symbols: list[str]) -> pd.DataFrame:
    """Load D1 closes, normalizing each timestamp to its UTC date so crypto (00:00)
    and OANDA (22:00) line up on the same daily index."""
    cols = {}
    for s in symbols:
        df = get_candles(s, "D1")
        ser = df.set_index("timestamp")["close"].astype("float64")
        # collapse each bar to its UTC date — keep last (latest) close in case of dup dates
        ser.index = pd.DatetimeIndex(ser.index, tz="UTC").floor("D")
        ser = ser[~ser.index.duplicated(keep="last")]
        cols[s] = ser
    out = pd.concat(cols, axis=1, sort=True)
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out.sort_index()


def basket_returns(closes: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """Equal-weight basket daily simple return. Members re-normalized when some are NaN."""
    sub = closes[symbols]
    rets = sub.pct_change()
    # equal-weight across the members that exist on each day
    w = sub.notna().astype("float64")
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    br = (rets * w).sum(axis=1, skipna=True)
    # Mask rows where no member exists
    br[w.sum(axis=1) == 0] = np.nan
    return br


def rolling_sharpe_log(ret: pd.Series, lookback: int) -> pd.Series:
    """Trailing Sharpe using log returns: mean / std * sqrt(252).

    Computes on the basket's NATIVE trading-day series (NaNs dropped) and
    reindexes back to the full calendar — so an FX basket isn't penalized
    for being NaN on crypto weekends. Within a window, only the most recent
    `lookback` *trading* observations are used.
    """
    native = ret.dropna()
    log_ret = np.log1p(native)
    mean = log_ret.rolling(lookback, min_periods=lookback).mean()
    std = log_ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = (mean / std) * np.sqrt(ANN)
    # Reindex back to the full calendar and forward-fill so a rebalance day
    # falling on a non-trading-day for that basket still gets a value.
    return sharpe.reindex(ret.index).ffill()


def first_trading_day_each_month(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First available bar in each calendar month (UTC)."""
    keys = [(t.year, t.month) for t in idx]
    df = pd.DataFrame({"ts": idx, "ym": keys}, index=range(len(idx)))
    firsts = df.groupby("ym")["ts"].min()
    out = pd.DatetimeIndex(pd.to_datetime(firsts.values, utc=True))
    return out.sort_values()


# ----------------------------- core -----------------------------

def build_rotation_weights(
    closes_per_basket: dict[str, pd.DataFrame],
    basket_rets: pd.DataFrame,
    top_k: int,
    risk_off: bool,
    all_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (asset_weights_daily, basket_alloc_daily).

    asset_weights_daily: columns = all symbols, daily target weights.
    basket_alloc_daily: columns = basket names, daily basket allocations (sum to 1
        on active days, 0 before first valid rebalance).

    Decision at rebalance date t uses Sharpe computed from data through t-1.
    """
    # rolling Sharpe per basket (uses returns up through and including t).
    # To avoid lookahead, decision at rebalance t uses Sharpe through t-1 (shift 1).
    sharpe = pd.DataFrame({b: rolling_sharpe_log(basket_rets[b], SHARPE_LOOKBACK)
                           for b in BASKET_NAMES})
    sharpe_dec = sharpe.shift(1)

    # 21-day return for indices, also shifted to avoid lookahead.
    idx_native = basket_rets["index"].dropna()
    idx_21d_log = np.log1p(idx_native).rolling(RISK_OFF_LOOKBACK, min_periods=RISK_OFF_LOOKBACK).sum()
    idx_21d_simple = np.expm1(idx_21d_log).reindex(basket_rets.index).ffill().shift(1)

    rebal_dates = first_trading_day_each_month(all_dates)

    # all assets
    all_syms = sum([list(BASKETS[b]) for b in BASKET_NAMES], [])
    asset_w = pd.DataFrame(0.0, index=all_dates, columns=all_syms)
    basket_alloc = pd.DataFrame(0.0, index=all_dates, columns=BASKET_NAMES)

    current_w = pd.Series(0.0, index=all_syms)
    current_alloc = pd.Series(0.0, index=BASKET_NAMES)
    triggered_risk_off_dates: list[pd.Timestamp] = []

    rebal_set = set(rebal_dates)
    for t in all_dates:
        if t in rebal_set:
            sh = sharpe_dec.loc[t] if t in sharpe_dec.index else None
            if sh is None or sh.isna().any():
                # not enough data yet; hold prior allocation (0 if none yet)
                pass
            else:
                # Default top-K ranking
                ranked = sh.sort_values(ascending=False)
                chosen = list(ranked.index[:top_k])
                if risk_off:
                    rv = idx_21d_simple.loc[t] if t in idx_21d_simple.index else np.nan
                    if pd.notna(rv) and rv < RISK_OFF_THRESHOLD:
                        chosen = ["fx"]
                        triggered_risk_off_dates.append(t)
                new_alloc = pd.Series(0.0, index=BASKET_NAMES)
                for b in chosen:
                    new_alloc[b] = 1.0 / len(chosen)
                # within-basket equal-weight
                new_w = pd.Series(0.0, index=all_syms)
                for b, a in new_alloc.items():
                    if a == 0:
                        continue
                    syms = BASKETS[b]
                    new_w[syms] = a / len(syms)
                current_w = new_w
                current_alloc = new_alloc
        asset_w.loc[t] = current_w.values
        basket_alloc.loc[t] = current_alloc.values

    # Attach metadata as attribute
    asset_w.attrs["risk_off_triggers"] = triggered_risk_off_dates
    return asset_w, basket_alloc


def gross_and_costs(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Apply weights (decided at start of bar t) to bar t's return; cost on Δweight."""
    ret = closes[symbols].pct_change().fillna(0.0)
    # weights at row t come from decision made using data <= t-1 (we already shifted Sharpe).
    # But the asset weights we built include the rebalance day t itself — apply them to bar t+1
    # to be safe (i.e., shift weights by 1 day).
    pos = weights[symbols].shift(1).fillna(0.0)
    gross = (pos * ret).sum(axis=1)
    costs = np.array([cost_for(s) for s in symbols], dtype="float64")
    dpos = pos.diff().abs().fillna(pos.abs())
    cost_stream = (dpos * costs).sum(axis=1)
    return gross, cost_stream


def vol_target(net: pd.Series, lookback: int, target: float) -> pd.Series:
    realized = net.rolling(lookback, min_periods=max(20, lookback // 2)).std(ddof=0) * np.sqrt(ANN)
    scale = (target / realized).shift(1)
    scale = scale.clip(upper=5.0).fillna(0.0)
    return net * scale


def stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0, "n": int(len(r))}
    ann_ret = r.mean() * ANN
    ann_vol = r.std(ddof=0) * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_dd": float(dd.min()),
        "n": int(len(r)),
    }


def split_stats(returns: pd.Series) -> dict:
    out = {"full": stats(returns), "is": stats(returns[returns.index < SPLIT]),
           "oos": stats(returns[returns.index >= SPLIT])}
    y2022 = returns[(returns.index >= pd.Timestamp("2022-01-01", tz="UTC"))
                    & (returns.index < pd.Timestamp("2023-01-01", tz="UTC"))]
    out["2022"] = stats(y2022)
    return out


def yearly_sharpes(returns: pd.Series) -> dict[int, dict]:
    out = {}
    for year, grp in returns.groupby(returns.index.year):
        out[int(year)] = stats(grp)
    return out


# ----------------------------- pipeline -----------------------------

def build_variant(
    closes_all: pd.DataFrame,
    basket_rets: pd.DataFrame,
    top_k: int,
    risk_off: bool,
) -> tuple[pd.Series, pd.DataFrame, list]:
    all_syms = sum([list(BASKETS[b]) for b in BASKET_NAMES], [])
    asset_w, basket_alloc = build_rotation_weights(
        closes_per_basket={b: closes_all[BASKETS[b]] for b in BASKET_NAMES},
        basket_rets=basket_rets,
        top_k=top_k,
        risk_off=risk_off,
        all_dates=closes_all.index,
    )
    gross, cost = gross_and_costs(closes_all, asset_w, all_syms)
    net_pre_vt = gross - cost
    net = vol_target(net_pre_vt, VOL_LOOKBACK, TARGET_VOL)
    return net, basket_alloc, asset_w.attrs.get("risk_off_triggers", [])


def baseline_buyhold(closes_all: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """Equal-weight buy-and-hold daily-rebalanced. No vol targeting (raw beta)."""
    rets = closes_all[symbols].pct_change()
    w = closes_all[symbols].notna().astype("float64")
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    br = (rets * w).sum(axis=1, skipna=True)
    return br.fillna(0.0)


def baseline_buyhold_vt(closes_all: pd.DataFrame, symbols: list[str]) -> pd.Series:
    """Same baseline but vol-targeted to 10% (apples-to-apples vs rotation)."""
    raw = baseline_buyhold(closes_all, symbols)
    return vol_target(raw, VOL_LOOKBACK, TARGET_VOL)


def main() -> None:
    all_syms = sum([list(BASKETS[b]) for b in BASKET_NAMES], [])
    closes_all = load_closes(all_syms)
    # Restrict to rows where at least one basket has a return on the prior day.
    closes_all = closes_all.sort_index()

    # basket equal-weight returns
    basket_rets = pd.DataFrame({b: basket_returns(closes_all, BASKETS[b]) for b in BASKET_NAMES})

    print(f"date range: {closes_all.index.min().date()} -> {closes_all.index.max().date()}  ({len(closes_all)} bars)")

    # ---- variants ----
    variants = {}
    triggers_by_variant = {}
    allocs_by_variant = {}
    for tag, top_k, risk_off in [
        ("top1", 1, False),
        ("top2", 2, False),
        ("top3", 3, False),
        ("top1_riskoff", 1, True),
        ("top2_riskoff", 2, True),
    ]:
        net, basket_alloc, triggers = build_variant(closes_all, basket_rets, top_k, risk_off)
        variants[tag] = net
        triggers_by_variant[tag] = triggers
        allocs_by_variant[tag] = basket_alloc

    # ---- baselines ----
    baselines = {
        "bh_all13": baseline_buyhold(closes_all, all_syms),
        "bh_all13_vt10": baseline_buyhold_vt(closes_all, all_syms),
        "bh_crypto": baseline_buyhold(closes_all, CRYPTO),
        "bh_crypto_vt10": baseline_buyhold_vt(closes_all, CRYPTO),
        "bh_indices": baseline_buyhold(closes_all, INDEX),
        "bh_indices_vt10": baseline_buyhold_vt(closes_all, INDEX),
    }

    # ---- compute stats ----
    rows = []
    print("\n=== headline ===")
    print(f"{'variant':<22} {'full':>7} {'IS':>7} {'OOS':>7} {'2022':>7}  {'fullRet':>8} {'fullVol':>8} {'fullDD':>8}")
    for name, ser in {**variants, **baselines}.items():
        s = split_stats(ser)
        rows.append({
            "variant": name,
            "full_sharpe": s["full"]["sharpe"], "full_ann_return": s["full"]["ann_return"],
            "full_ann_vol": s["full"]["ann_vol"], "full_max_dd": s["full"]["max_dd"],
            "is_sharpe": s["is"]["sharpe"], "is_ann_return": s["is"]["ann_return"],
            "is_max_dd": s["is"]["max_dd"],
            "oos_sharpe": s["oos"]["sharpe"], "oos_ann_return": s["oos"]["ann_return"],
            "oos_max_dd": s["oos"]["max_dd"],
            "2022_sharpe": s["2022"]["sharpe"], "2022_ann_return": s["2022"]["ann_return"],
            "2022_max_dd": s["2022"]["max_dd"],
            "n": s["full"]["n"],
        })
        print(f"  {name:<20} {s['full']['sharpe']:>+6.2f} {s['is']['sharpe']:>+6.2f} "
              f"{s['oos']['sharpe']:>+6.2f} {s['2022']['sharpe']:>+6.2f}  "
              f"{s['full']['ann_return']:>+7.1%} {s['full']['ann_vol']:>7.1%} {s['full']['max_dd']:>+7.1%}")

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(OUT_DIR / "rotation_comparison.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'rotation_comparison.csv'}")

    # ---- yearly per variant ----
    print("\n=== yearly Sharpe (variants only) ===")
    years = sorted({int(y) for y in closes_all.index.year.unique()})
    header = "  " + "year  ".rjust(8) + " " + " ".join(f"{v:>14}" for v in variants)
    print(header)
    yearly_rows = []
    for y in years:
        cells = [f"{y}"]
        for v, ser in variants.items():
            grp = ser[(ser.index.year == y)]
            s = stats(grp)
            cells.append(f"{s['sharpe']:+.2f}")
            yearly_rows.append({"year": y, "variant": v,
                                "sharpe": s["sharpe"], "ann_return": s["ann_return"],
                                "ann_vol": s["ann_vol"], "max_dd": s["max_dd"]})
        print("  " + cells[0].rjust(6) + "  " + "  ".join(c.rjust(13) for c in cells[1:]))
    pd.DataFrame(yearly_rows).to_csv(OUT_DIR / "rotation_yearly.csv", index=False)

    # ---- basket selection frequency for each variant ----
    print("\n=== basket selection frequency (fraction of rebalances where basket received >0 alloc) ===")
    sel_rows = []
    rebal_dates = first_trading_day_each_month(closes_all.index)
    for v, alloc in allocs_by_variant.items():
        sub = alloc.loc[rebal_dates]
        # Drop rebal dates before the strategy was active
        active = (sub.sum(axis=1) > 0)
        sub = sub[active]
        freq = (sub > 0).mean()
        n_riskoff = len(triggers_by_variant[v])
        print(f"  {v:<18}  crypto={freq.get('crypto', 0):.0%}  fx={freq.get('fx', 0):.0%}  index={freq.get('index', 0):.0%}"
              f"  n_active_rebal={int(active.sum())}  n_riskoff_triggers={n_riskoff}")
        sel_rows.append({
            "variant": v,
            "freq_crypto": float(freq.get('crypto', 0)),
            "freq_fx": float(freq.get('fx', 0)),
            "freq_index": float(freq.get('index', 0)),
            "n_active_rebalances": int(active.sum()),
            "n_riskoff_triggers": int(n_riskoff),
        })
    pd.DataFrame(sel_rows).to_csv(OUT_DIR / "rotation_basket_freq.csv", index=False)

    # ---- 2022 risk-off detail ----
    print("\n=== risk-off triggers (year-by-year, top1_riskoff) ===")
    triggers = triggers_by_variant.get("top1_riskoff", [])
    by_year = {}
    for t in triggers:
        by_year.setdefault(t.year, []).append(t)
    for y in sorted(by_year):
        print(f"  {y}: {len(by_year[y])} triggers (first={by_year[y][0].date()}, last={by_year[y][-1].date()})")

    # ---- pick best variant: prefer OOS Sharpe, tiebreak full ----
    best_var = max(variants, key=lambda k: (split_stats(variants[k])["oos"]["sharpe"],
                                            split_stats(variants[k])["full"]["sharpe"]))
    print(f"\n=== best variant by OOS Sharpe: {best_var} ===")
    best = variants[best_var]
    s = split_stats(best)
    print(f"  full={s['full']['sharpe']:+.2f}  IS={s['is']['sharpe']:+.2f}  "
          f"OOS={s['oos']['sharpe']:+.2f}  2022={s['2022']['sharpe']:+.2f}")

    # ---- write sleeve parquet ----
    daily = best.dropna().copy()
    # Ensure UTC tz-aware index
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    out = pd.DataFrame({"timestamp": daily.index, "ret": daily.values})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(OUT_DIR / "rotation_returns.parquet", index=False)
    print(f"\nwrote {OUT_DIR / 'rotation_returns.parquet'} ({len(out)} rows; best variant = {best_var})")

    # ---- alpha vs baseline: regress rotation on bh_all13_vt10 ----
    base = baselines["bh_all13_vt10"]
    df = pd.concat({"rot": best, "base": base}, axis=1).dropna()
    cov = df["rot"].cov(df["base"])
    var = df["base"].var()
    beta = cov / var if var > 0 else 0.0
    alpha_daily = df["rot"].mean() - beta * df["base"].mean()
    alpha_ann = alpha_daily * ANN
    corr = df["rot"].corr(df["base"])
    print(f"\n=== rotation vs equal-weight (vt10) baseline ===")
    print(f"  beta={beta:+.2f}  alpha_ann={alpha_ann:+.1%}  corr={corr:+.2f}")


if __name__ == "__main__":
    main()
