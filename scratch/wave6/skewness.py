"""Skewness-premium strategies (lottery effect).

The "lottery effect" says assets with positive expected skew (long-tail right)
tend to underperform on a risk-adjusted basis: investors overpay for the right
tail. Symmetric prediction — names with negative skew (insurance-like) earn a
premium. We build 5 sub-sleeves around this idea and combine survivors.

Methodology:
    - IS: < 2024-01-01.  OOS: >= 2024-01-01.
    - Walk-forward: all skew estimates are trailing rolling windows; any
      threshold tuning uses IS data only via `is_quantile`.
    - Each survivor vol-scaled to 5% ann vol on IS.
    - Filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.
    - Combine survivors equal-weight.

Outputs:
    scratch/wave6/skewness.py                  (this file)
    scratch/wave6/skewness_returns.parquet     (combined sleeve returns)
    scratch/wave6/skewness_breakdown.csv       (per-sub-sleeve table)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import CRYPTO, FOREX, INDEX, ALL_SYMBOLS, get_candles
from alphabeta.backtest import cost_for


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scratch" / "wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05    # 5% per-sub-sleeve annualised IS vol

INDICES_EQ = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP", "DE30_EUR", "JP225_USD"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_d1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "D1").sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    df["ret"] = df["close"].pct_change()
    return df


def perf_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return dict(sharpe=np.nan, ann_return=np.nan, ann_vol=np.nan, max_dd=np.nan, n=0)
    ann_ret = r.mean() * bpy
    ann_vol = r.std(ddof=0) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(sharpe=float(sharpe), ann_return=float(ann_ret),
                ann_vol=float(ann_vol), max_dd=float(dd), n=int(len(r)))


def split_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_mask = idx < SPLIT
    return {
        "FULL": perf_stats(r, bpy),
        "IS":   perf_stats(r[is_mask], bpy),
        "OOS":  perf_stats(r[~is_mask], bpy),
        "Y2022": perf_stats(r[(idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
                              (idx < pd.Timestamp("2023-01-01", tz="UTC"))], bpy),
        "Y2024_25": perf_stats(r[idx >= pd.Timestamp("2024-01-01", tz="UTC")], bpy),
    }


def to_daily(s: pd.Series) -> pd.Series:
    """Collapse a per-symbol return series to calendar day."""
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).floor("1D")
    s = s.groupby(level=0).sum()
    return s


def vol_scale(ret: pd.Series, target: float = TARGET_SUB_VOL,
              bpy: float = 252.0) -> tuple[pd.Series, float]:
    """Scale a return stream so IS realised ann vol == target. Single scalar."""
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_r = r[idx < SPLIT]
    iv = is_r.std(ddof=0) * np.sqrt(bpy)
    if iv <= 0 or not np.isfinite(iv):
        return ret * 0.0, 0.0
    k = target / iv
    return ret * k, float(k)


def position_to_returns(df: pd.DataFrame, pos: pd.Series, symbol: str) -> pd.Series:
    """Apply position to price series with per-side cost. Indexed by timestamp."""
    pos = pos.astype("float64").fillna(0.0)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos * bar_ret
    cps = cost_for(symbol)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = gross - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


def is_quantile(series: pd.Series, q: float, timestamps: pd.Series) -> float:
    """Quantile computed only on in-sample data (timestamps < SPLIT)."""
    is_mask = timestamps < SPLIT
    is_vals = series[is_mask].dropna()
    if len(is_vals) == 0:
        return np.nan
    return float(is_vals.quantile(q))


def shifted_skew(log_ret: pd.Series, window: int) -> pd.Series:
    """Trailing rolling skew, shifted by 1 to avoid look-ahead."""
    return log_ret.rolling(window, min_periods=max(20, window // 2)).skew().shift(1)


def shifted_vol(log_ret: pd.Series, window: int) -> pd.Series:
    return log_ret.rolling(window, min_periods=max(10, window // 2)).std(ddof=1).shift(1)


# ---------------------------------------------------------------------------
# Sub-sleeve 1: Skewness short tercile (long/short, cross-sectional)
# Daily rebalance: for each date, rank assets by trailing 60d skew. Short the
# top tercile (high positive skew = lottery names), long the bottom tercile
# (negative skew = insurance-like). Equal-weight per leg, dollar-neutral.
# ---------------------------------------------------------------------------
def sleeve1_skew_tercile(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = list(data.keys())
    # Build a daily panel of trailing 60d skew per symbol, on a common calendar.
    panels = {}
    rets = {}
    for sym in syms:
        df = data[sym]
        sk = shifted_skew(df["log_ret"], 60)
        sk.index = df["timestamp"]
        rets[sym] = df.set_index("timestamp")["close"].pct_change()
        panels[sym] = sk
    skew_df = pd.concat(panels, axis=1).sort_index()
    skew_df.index = pd.DatetimeIndex(skew_df.index).floor("1D")
    skew_df = skew_df.groupby(level=0).last()  # daily
    ret_df = pd.concat(rets, axis=1).sort_index()
    ret_df.index = pd.DatetimeIndex(ret_df.index).floor("1D")
    ret_df = ret_df.groupby(level=0).sum()  # daily

    # For each date, rank skew; bottom tercile = +1, top tercile = -1.
    # n=13 symbols means tercile size ~ 4 each.
    pos_df = pd.DataFrame(0.0, index=skew_df.index, columns=skew_df.columns)
    for ts, row in skew_df.iterrows():
        vals = row.dropna()
        if len(vals) < 6:
            continue
        k = max(2, len(vals) // 3)
        ranks = vals.rank(method="first")
        # bottom-k => long
        bot = ranks.nsmallest(k).index
        top = ranks.nlargest(k).index
        # normalize so each leg has gross 1.0
        for s in bot:
            pos_df.loc[ts, s] = 1.0 / len(bot)
        for s in top:
            pos_df.loc[ts, s] = -1.0 / len(top)

    # Apply per-asset cost on |Δpos|. Net return = pos * ret - cost*|Δpos|.
    aligned = ret_df.reindex(pos_df.index).fillna(0.0)
    gross = (pos_df * aligned).sum(axis=1)
    dpos = pos_df.diff().abs()
    dpos.iloc[0] = pos_df.iloc[0].abs()
    cost_vec = pd.Series({s: cost_for(s) for s in pos_df.columns})
    cost = (dpos * cost_vec).sum(axis=1)
    net = gross - cost
    return net


# ---------------------------------------------------------------------------
# Sub-sleeve 2: Realized-skew flip predictor
# Per-asset: compute trailing 21d skew. When it flips from negative to
# strongly positive (>+0.5 SD jump vs trailing skew distribution), short for
# 5 days. The intuition: a fresh right-tail squeeze tends to mean-revert.
# Uses IS distribution of skew changes for normalization.
# ---------------------------------------------------------------------------
def sleeve2_skew_flip(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = list(data.keys())
    streams = []
    for sym in syms:
        df = data[sym]
        sk21 = shifted_skew(df["log_ret"], 21)
        # change in skew vs 5d prior
        dsk = sk21 - sk21.shift(5)
        # SD of dsk on IS
        dsk_is = dsk[df["timestamp"] < SPLIT]
        sd = dsk_is.std(ddof=0)
        if not (sd > 0) or not np.isfinite(sd):
            continue

        # trigger: previous skew negative AND change > 0.5 SD AND new skew positive
        trig = (sk21.shift(5) < 0) & (dsk > 0.5 * sd) & (sk21 > 0)
        # short for 5 bars
        n = len(df)
        pos = np.zeros(n)
        hold = 0
        trig_arr = trig.values
        for t in range(n):
            if hold > 0:
                pos[t] = -1.0
                hold -= 1
                continue
            if bool(trig_arr[t]):
                pos[t] = -1.0
                hold = 4  # plus this bar = 5 total
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    if not streams:
        return pd.Series(dtype=float)
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 3: Skew-vol interaction ("calm-but-lottery-like")
# Per-asset: when trailing 60d skew > +1.0 AND trailing 30d realized vol is
# below the IS median, the asset is in a "calm but right-skewed" setup —
# classic pre-event lottery. Short until skew falls back to <0.5.
# ---------------------------------------------------------------------------
def sleeve3_skew_vol(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = list(data.keys())
    streams = []
    for sym in syms:
        df = data[sym]
        sk60 = shifted_skew(df["log_ret"], 60)
        rv30 = shifted_vol(df["log_ret"], 30)
        p50 = is_quantile(rv30, 0.50, df["timestamp"])
        if not np.isfinite(p50):
            continue

        # state machine: short when (skew>1 AND vol<p50). Exit when skew<0.5.
        sk_v = sk60.values
        rv_v = rv30.values
        n = len(df)
        pos = np.zeros(n)
        state = 0  # 0 flat, -1 short
        for t in range(n):
            sv = sk_v[t]; rvv = rv_v[t]
            if not np.isfinite(sv) or not np.isfinite(rvv):
                pos[t] = state
                continue
            if state == 0:
                if sv > 1.0 and rvv < p50:
                    state = -1
            else:  # state == -1
                if sv < 0.5:
                    state = 0
            pos[t] = float(state)
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    if not streams:
        return pd.Series(dtype=float)
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 4: Cross-sectional skew rank, monthly rebalance
# Each month-end, rank all 13 by trailing 60d skew. Long the 3 most negative,
# short the 3 most positive. Equal-weight basket, held for 21 trading days.
# ---------------------------------------------------------------------------
def sleeve4_monthly_skew_rank(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = list(data.keys())
    panels = {}
    rets = {}
    for sym in syms:
        df = data[sym]
        sk = shifted_skew(df["log_ret"], 60)
        sk.index = df["timestamp"]
        panels[sym] = sk
        rets[sym] = df.set_index("timestamp")["close"].pct_change()
    skew_df = pd.concat(panels, axis=1).sort_index()
    skew_df.index = pd.DatetimeIndex(skew_df.index).floor("1D")
    skew_df = skew_df.groupby(level=0).last()
    ret_df = pd.concat(rets, axis=1).sort_index()
    ret_df.index = pd.DatetimeIndex(ret_df.index).floor("1D")
    ret_df = ret_df.groupby(level=0).sum()

    # Build position frame; rebalance on the first business day of each month.
    # Hold position constant within the month.
    pos_df = pd.DataFrame(0.0, index=skew_df.index, columns=skew_df.columns)
    current_pos = pd.Series(0.0, index=skew_df.columns)
    last_month = None
    for ts in skew_df.index:
        m = (ts.year, ts.month)
        if m != last_month:
            row = skew_df.loc[ts].dropna()
            if len(row) >= 6:
                bot3 = row.nsmallest(3).index
                top3 = row.nlargest(3).index
                current_pos = pd.Series(0.0, index=skew_df.columns)
                for s in bot3:
                    current_pos[s] = 1.0 / 3
                for s in top3:
                    current_pos[s] = -1.0 / 3
            last_month = m
        pos_df.loc[ts] = current_pos.values

    aligned = ret_df.reindex(pos_df.index).fillna(0.0)
    gross = (pos_df * aligned).sum(axis=1)
    dpos = pos_df.diff().abs()
    dpos.iloc[0] = pos_df.iloc[0].abs()
    cost_vec = pd.Series({s: cost_for(s) for s in pos_df.columns})
    cost = (dpos * cost_vec).sum(axis=1)
    return gross - cost


# ---------------------------------------------------------------------------
# Sub-sleeve 5: SPX/NAS skew divergence
# When SPX skew > +0.5 (right-tailed = market-implied bullish) and NAS skew
# < -0.5 (left-tailed = bearish on tech), the two are diverging. Trade:
# long SPX / short NAS until skews reconverge (|spx_skew - nas_skew| < 0.4).
# Also trade the mirror (short SPX / long NAS) when the inequality flips.
# ---------------------------------------------------------------------------
def sleeve5_spx_nas_skew_div(data: dict[str, pd.DataFrame]) -> pd.Series:
    a = data["SPX500_USD"].set_index("timestamp")
    b = data["NAS100_USD"].set_index("timestamp")
    common_idx = a.index.intersection(b.index)
    a = a.loc[common_idx].sort_index()
    b = b.loc[common_idx].sort_index()
    sk_a = a["log_ret"].rolling(60, min_periods=30).skew().shift(1)
    sk_b = b["log_ret"].rolling(60, min_periods=30).skew().shift(1)
    spread = sk_a - sk_b
    n = len(common_idx)
    pos_a = np.zeros(n)
    pos_b = np.zeros(n)
    state = 0
    sk_a_v = sk_a.values
    sk_b_v = sk_b.values
    sp_v = spread.values
    for t in range(n):
        sa = sk_a_v[t]; sb = sk_b_v[t]; sp = sp_v[t]
        if not (np.isfinite(sa) and np.isfinite(sb) and np.isfinite(sp)):
            pos_a[t] = state; pos_b[t] = -state
            continue
        if state == 0:
            if sa > 0.5 and sb < -0.5:
                state = 1   # long spx / short nas
            elif sa < -0.5 and sb > 0.5:
                state = -1  # short spx / long nas
        elif state == 1:
            if abs(sp) < 0.4:
                state = 0
        elif state == -1:
            if abs(sp) < 0.4:
                state = 0
        pos_a[t] = float(state)
        pos_b[t] = float(-state)
    pos_a = pd.Series(pos_a, index=common_idx)
    pos_b = pd.Series(pos_b, index=common_idx)

    ra = a["close"].pct_change().fillna(0.0)
    rb = b["close"].pct_change().fillna(0.0)
    cpa = cost_for("SPX500_USD"); cpb = cost_for("NAS100_USD")
    da = pos_a.diff().abs(); da.iloc[0] = abs(pos_a.iloc[0])
    db = pos_b.diff().abs(); db.iloc[0] = abs(pos_b.iloc[0])
    net = (pos_a * ra - da * cpa) + (pos_b * rb - db * cpb)
    net = net * 0.5  # 0.5 weight per leg
    net.name = "spx_nas_div"
    return to_daily(net)


# ---------------------------------------------------------------------------
# diagnostics: per-symbol persistent skew on IS
# ---------------------------------------------------------------------------
def diagnose_persistent_skew(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in data.items():
        # IS only
        is_lr = df.loc[df["timestamp"] < SPLIT, "log_ret"].dropna()
        oos_lr = df.loc[df["timestamp"] >= SPLIT, "log_ret"].dropna()
        sk60 = df["log_ret"].rolling(60, min_periods=40).skew()
        sk60_is = sk60[df["timestamp"] < SPLIT].dropna()
        rows.append({
            "symbol": sym,
            "n_is": len(is_lr),
            "is_full_skew": float(is_lr.skew()),
            "oos_full_skew": float(oos_lr.skew()) if len(oos_lr) else np.nan,
            "is_mean_60d_skew": float(sk60_is.mean()),
            "is_median_60d_skew": float(sk60_is.median()),
            "is_pct_pos_skew": float((sk60_is > 0).mean()),
            "is_pct_neg_skew": float((sk60_is < 0).mean()),
        })
    return pd.DataFrame(rows).set_index("symbol")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading D1 data (all 13 symbols) ===")
    data = {s: load_d1(s) for s in ALL_SYMBOLS}
    for s, df in data.items():
        print(f"  {s}: {len(df)} rows, {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    print("\n=== Persistent skew diagnostics (IS) ===")
    persist = diagnose_persistent_skew(data)
    persist_sorted = persist.sort_values("is_mean_60d_skew")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print(persist_sorted)
    persist_sorted.to_csv(OUT_DIR / "skewness_persistent.csv", float_format="%.4f")

    builders = {
        "S1_skew_tercile":       lambda: sleeve1_skew_tercile(data),
        "S2_skew_flip":          lambda: sleeve2_skew_flip(data),
        "S3_skew_vol":           lambda: sleeve3_skew_vol(data),
        "S4_monthly_skew_rank":  lambda: sleeve4_monthly_skew_rank(data),
        "S5_spx_nas_div":        lambda: sleeve5_spx_nas_skew_div(data),
    }

    print("\n=== Building sub-sleeves ===")
    raw_streams: dict[str, pd.Series] = {}
    for name, fn in builders.items():
        s = fn()
        if len(s) == 0:
            print(f"  {name}: EMPTY")
            continue
        s = s.sort_index()
        s.index = pd.DatetimeIndex(s.index, tz="UTC") if s.index.tz is None else pd.DatetimeIndex(s.index)
        raw_streams[name] = s
        print(f"  built {name}: {len(s)} bars, "
              f"first={s.index.min().date()}, last={s.index.max().date()}")

    scaled_streams: dict[str, pd.Series] = {}
    rows = []
    for name, s in raw_streams.items():
        scaled, k = vol_scale(s, target=TARGET_SUB_VOL, bpy=252.0)
        scaled_streams[name] = scaled
        stats = split_stats(scaled, bpy=252.0)
        rows.append({
            "sleeve": name,
            "scale_factor": k,
            "FULL_sharpe":  stats["FULL"]["sharpe"],
            "FULL_ret":     stats["FULL"]["ann_return"],
            "FULL_vol":     stats["FULL"]["ann_vol"],
            "FULL_dd":      stats["FULL"]["max_dd"],
            "IS_sharpe":    stats["IS"]["sharpe"],
            "IS_ret":       stats["IS"]["ann_return"],
            "IS_dd":        stats["IS"]["max_dd"],
            "OOS_sharpe":   stats["OOS"]["sharpe"],
            "OOS_ret":      stats["OOS"]["ann_return"],
            "OOS_dd":       stats["OOS"]["max_dd"],
            "Y2022_sharpe": stats["Y2022"]["sharpe"],
            "Y2022_ret":    stats["Y2022"]["ann_return"],
            "Y2024_25_sharpe": stats["Y2024_25"]["sharpe"],
            "Y2024_25_ret":    stats["Y2024_25"]["ann_return"],
        })
    breakdown = pd.DataFrame(rows).set_index("sleeve")

    # Survivors: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.
    is_ok = breakdown["IS_sharpe"] >= 0.4
    oos_ok = breakdown["OOS_sharpe"] >= 0.0
    survivors = breakdown[is_ok & oos_ok].index.tolist()
    breakdown["survivor"] = breakdown.index.isin(survivors)

    print("\n=== Per sub-sleeve breakdown ===")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print(breakdown)

    if survivors:
        panel = pd.concat({k: scaled_streams[k] for k in survivors},
                          axis=1, sort=True).fillna(0.0)
        combined = panel.mean(axis=1)
    else:
        all_idx = pd.DatetimeIndex(sorted(set(
            ts for s in scaled_streams.values() for ts in s.index
        )), tz="UTC")
        combined = pd.Series(0.0, index=all_idx)

    combined = combined.sort_index()
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC") if combined.index.tz is None else pd.DatetimeIndex(combined.index)

    combined_stats = split_stats(combined, bpy=252.0)
    print(f"\n=== Survivors: {survivors}")
    print("=== Combined sleeve stats ===")
    for k, v in combined_stats.items():
        print(f"  {k:>9} : sharpe={v['sharpe']:+.3f} ret={v['ann_return']:+.3%} "
              f"vol={v['ann_vol']:.3%} dd={v['max_dd']:+.3%} n={v['n']}")

    combined_row = {
        "scale_factor": 1.0,
        "FULL_sharpe":  combined_stats["FULL"]["sharpe"],
        "FULL_ret":     combined_stats["FULL"]["ann_return"],
        "FULL_vol":     combined_stats["FULL"]["ann_vol"],
        "FULL_dd":      combined_stats["FULL"]["max_dd"],
        "IS_sharpe":    combined_stats["IS"]["sharpe"],
        "IS_ret":       combined_stats["IS"]["ann_return"],
        "IS_dd":        combined_stats["IS"]["max_dd"],
        "OOS_sharpe":   combined_stats["OOS"]["sharpe"],
        "OOS_ret":      combined_stats["OOS"]["ann_return"],
        "OOS_dd":       combined_stats["OOS"]["max_dd"],
        "Y2022_sharpe": combined_stats["Y2022"]["sharpe"],
        "Y2022_ret":    combined_stats["Y2022"]["ann_return"],
        "Y2024_25_sharpe": combined_stats["Y2024_25"]["sharpe"],
        "Y2024_25_ret":    combined_stats["Y2024_25"]["ann_return"],
        "survivor": True,
    }
    breakdown.loc["COMBINED"] = combined_row
    breakdown.to_csv(OUT_DIR / "skewness_breakdown.csv", float_format="%.4f")

    out = combined.rename("ret").to_frame()
    out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
    out = out.reset_index()
    out.to_parquet(OUT_DIR / "skewness_returns.parquet", index=False)
    print("\nSaved:")
    print(f"  {OUT_DIR / 'skewness_returns.parquet'}")
    print(f"  {OUT_DIR / 'skewness_breakdown.csv'}")
    print(f"  {OUT_DIR / 'skewness_persistent.csv'}")


if __name__ == "__main__":
    main()
