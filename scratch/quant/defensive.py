"""Defensive / long-volatility sleeve.

Goal: build a sub-sleeve combination that profits in equity sell-offs and
specifically flips 2022 from break-even to positive.

We construct 6 candidate sub-sleeves, walk-forward the thresholds (use
in-sample quantiles only, then apply OOS), vol-scale each survivor to 5%
annualised vol on IS data, then equal-weight the survivors that have
positive IS Sharpe AND positive 2022 return.

Survivors are the *defensive* sleeves: those that did their job in 2022.

Outputs:
    scratch/quant/defensive.py                (this file)
    scratch/quant/defensive_returns.parquet   (D1 combined sleeve returns)
    scratch/quant/defensive_breakdown.csv     (per-sub-sleeve table)

All sub-sleeves use only data observable at the open of bar t (we shift
signals by 1 bar). All thresholds use in-sample (<2024-01-01) quantiles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import ALL_SYMBOLS, CRYPTO, FOREX, INDEX, get_candles
from alphabeta.backtest import cost_for


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scratch" / "quant"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05    # 5% per-sub-sleeve annualised


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ann_factor(symbol: str) -> float:
    return 365.0 if symbol in CRYPTO else 252.0


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
        "IS": perf_stats(r[is_mask], bpy),
        "OOS": perf_stats(r[~is_mask], bpy),
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


# ---------------------------------------------------------------------------
# Sub-sleeve 1: Vol-breakout short on equities (SPX + NAS + US30)
# ---------------------------------------------------------------------------
def sleeve1_vol_breakout_short(data: dict[str, pd.DataFrame]) -> pd.Series:
    """Short equities when 30d realised vol > IS p80 of rolling-vol; exit when
    realised vol falls back below its IS median.

    For each of SPX / NAS / US30 build a 0/-1 position. Equal-weight the three
    per-symbol return streams, then this gets vol-scaled later.
    """
    syms = ["SPX500_USD", "NAS100_USD", "US30_USD"]
    streams = []
    for sym in syms:
        df = data[sym]
        rv30 = df["log_ret"].rolling(30).std(ddof=1) * np.sqrt(252.0)
        # IS p80 and median computed from rolling vol values whose bar < SPLIT
        p80 = is_quantile(rv30, 0.80, df["timestamp"])
        med = is_quantile(rv30, 0.50, df["timestamp"])
        # State machine: short on cross-above p80, exit on cross-below median.
        rv30s = rv30.shift(1).values  # value known at open of bar t
        pos = np.zeros(len(df))
        short = False
        for t in range(len(df)):
            v = rv30s[t]
            if not np.isfinite(v):
                pos[t] = 0.0
                continue
            if not short and v > p80:
                short = True
            elif short and v < med:
                short = False
            pos[t] = -1.0 if short else 0.0
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 2: Safe-haven long basket (USD_JPY + XAU_USD) gated by stress
# ---------------------------------------------------------------------------
def sleeve2_safe_haven(data: dict[str, pd.DataFrame]) -> pd.Series:
    """Long USD_JPY + long XAU_USD, each vol-targeted to 10% ann, but only when
    equity-stress flag fires: SPX 20d return < -5% OR SPX 30d realised vol >
    IS p80.
    """
    spx = data["SPX500_USD"].copy()
    spx_idx = spx.set_index("timestamp")
    spx20 = spx_idx["close"].pct_change(20)
    rv30 = spx_idx["log_ret"].rolling(30).std(ddof=1) * np.sqrt(252.0)

    # IS p80 of SPX rv30 (per timestamp)
    is_mask = spx_idx.index < SPLIT
    p80 = float(rv30[is_mask].dropna().quantile(0.80))

    # Stress flag: SPX 20d return < -0.05  OR  rv30 > p80
    stress = ((spx20 < -0.05) | (rv30 > p80))
    # Shift by 1 — known at open of bar t
    stress = stress.shift(1).fillna(False)
    stress.name = "stress"

    syms = ["USD_JPY", "XAU_USD"]
    target_per_leg = 0.10
    streams = []
    for sym in syms:
        df = data[sym]
        rv = df["log_ret"].rolling(30).std(ddof=1).shift(1) * np.sqrt(252.0)
        size = (target_per_leg / rv.replace(0.0, np.nan)).clip(lower=0.0, upper=2.5).fillna(0.0)
        # apply stress gate
        df_ts = df.set_index("timestamp")
        size.index = df["timestamp"]
        gate = stress.reindex(size.index).fillna(False).astype(float)
        pos_ts = (size.values * gate.values)
        pos = pd.Series(pos_ts, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 3: Crypto short on vol shock (BTC mean-revert short)
# ---------------------------------------------------------------------------
def sleeve3_btc_vol_short(data: dict[str, pd.DataFrame]) -> pd.Series:
    """Short BTC for 5 days when BTC 30d realised vol > IS p80 AND BTC 5d
    return < -10%.
    """
    df = data["BTCUSDT"]
    rv30 = df["log_ret"].rolling(30).std(ddof=1) * np.sqrt(365.0)
    r5 = df["close"].pct_change(5)
    p80 = is_quantile(rv30, 0.80, df["timestamp"])

    rv30s = rv30.shift(1).values
    r5s = r5.shift(1).values

    n = len(df)
    pos = np.zeros(n)
    bars_held = 0
    short = False
    for t in range(n):
        if short:
            pos[t] = -1.0
            bars_held += 1
            if bars_held >= 5:
                short = False
                bars_held = 0
            continue
        v = rv30s[t]
        r = r5s[t]
        if (np.isfinite(v) and np.isfinite(r) and v > p80 and r < -0.10):
            short = True
            bars_held = 1
            pos[t] = -1.0
    pos = pd.Series(pos, index=df.index)
    ret = position_to_returns(df, pos, "BTCUSDT")
    return to_daily(ret)


# ---------------------------------------------------------------------------
# Sub-sleeve 4: Asymmetric mean-reversion LONG on big down-days
# ---------------------------------------------------------------------------
def sleeve4_asym_mr_long(data: dict[str, pd.DataFrame]) -> pd.Series:
    """For equity indices, go LONG only on big down-days (prior bar log_ret <
    -2 * std60). Hold for up to 3 bars. Captures the bounce without taking
    the long side of up-days (where the un-asymmetric strategy added zero).

    Equal-weight across NAS / UK / DE (the indices the v3 D1REV strategies
    worked on per the prompt).
    """
    syms = ["NAS100_USD", "UK100_GBP", "DE30_EUR"]
    streams = []
    for sym in syms:
        df = data[sym]
        std60 = df["log_ret"].rolling(60).std(ddof=1)
        threshold = -2.0 * std60
        prior_ret = df["log_ret"].shift(1)
        prior_thr = threshold.shift(2)  # threshold known at open of t
        trig = prior_ret < prior_thr

        n = len(df)
        pos = np.zeros(n)
        hold = 0
        for t in range(n):
            if hold > 0:
                pos[t] = 1.0
                hold -= 1
                continue
            if bool(trig.iloc[t]):
                pos[t] = 1.0
                hold = 2  # already in for bar t, hold 2 more
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Sub-sleeve 5: Vol-of-vol short equities (rate-of-change of vol)
# ---------------------------------------------------------------------------
def sleeve5_volvol_short(data: dict[str, pd.DataFrame]) -> pd.Series:
    """Compute SPX 30d realised vol; then take rolling-60d std of that vol
    series; when this vol-of-vol exceeds its IS p80, short SPX for 5 days.
    """
    df = data["SPX500_USD"]
    rv30 = df["log_ret"].rolling(30).std(ddof=1) * np.sqrt(252.0)
    vov = rv30.rolling(60).std(ddof=1)
    p80 = is_quantile(vov, 0.80, df["timestamp"])

    vov_s = vov.shift(1).values
    n = len(df)
    pos = np.zeros(n)
    short = False
    hold = 0
    for t in range(n):
        if short:
            pos[t] = -1.0
            hold += 1
            if hold >= 5:
                short = False
                hold = 0
            continue
        v = vov_s[t]
        if np.isfinite(v) and v > p80:
            short = True
            hold = 1
            pos[t] = -1.0
    pos = pd.Series(pos, index=df.index)
    ret = position_to_returns(df, pos, "SPX500_USD")
    return to_daily(ret)


# ---------------------------------------------------------------------------
# Sub-sleeve 6: Carry-trade reversal (long JPY when USD_JPY tanks)
# ---------------------------------------------------------------------------
def sleeve6_carry_reversal(data: dict[str, pd.DataFrame]) -> pd.Series:
    """When USD_JPY 5d log return < -1.5%, short USD_JPY for 5 bars (i.e. long
    JPY). The carry unwind on risk-off.
    """
    df = data["USD_JPY"]
    r5 = (np.log(df["close"]) - np.log(df["close"].shift(5)))
    trig = (r5.shift(1) < -0.015)

    n = len(df)
    pos = np.zeros(n)
    short = False
    hold = 0
    for t in range(n):
        if short:
            pos[t] = -1.0
            hold += 1
            if hold >= 5:
                short = False
                hold = 0
            continue
        if bool(trig.iloc[t]):
            short = True
            hold = 1
            pos[t] = -1.0
    pos = pd.Series(pos, index=df.index)
    ret = position_to_returns(df, pos, "USD_JPY")
    return to_daily(ret)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading D1 data for 13 symbols ===")
    needed = sorted(set([
        "SPX500_USD", "NAS100_USD", "US30_USD",
        "UK100_GBP", "DE30_EUR", "JP225_USD",
        "USD_JPY", "XAU_USD", "EUR_USD", "GBP_USD",
        "BTCUSDT", "ETHUSDT", "SOLUSDT",
    ]))
    data = {s: load_d1(s) for s in needed}

    builders = {
        "S1_vol_breakout_short_eq":   sleeve1_vol_breakout_short,
        "S2_safe_haven_basket":       sleeve2_safe_haven,
        "S3_btc_vol_shock_short":     sleeve3_btc_vol_short,
        "S4_asym_mr_long_eq":         sleeve4_asym_mr_long,
        "S5_volvol_short_spx":        sleeve5_volvol_short,
        "S6_carry_reversal_jpy":      sleeve6_carry_reversal,
    }

    print("=== Building sub-sleeves ===")
    raw_streams: dict[str, pd.Series] = {}
    for name, fn in builders.items():
        s = fn(data)
        s = s.sort_index()
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
        raw_streams[name] = s
        print(f"  built {name}: {len(s)} bars, "
              f"first={s.index.min().date()}, last={s.index.max().date()}")

    # Vol-scale each to 5% ann on IS, then re-evaluate stats.
    scaled_streams: dict[str, pd.Series] = {}
    rows = []
    for name, s in raw_streams.items():
        scaled, k = vol_scale(s, target=TARGET_SUB_VOL, bpy=252.0)
        scaled_streams[name] = scaled
        stats = split_stats(scaled, bpy=252.0)
        rows.append({
            "sleeve": name,
            "scale_factor": k,
            "FULL_sharpe": stats["FULL"]["sharpe"],
            "FULL_ret":    stats["FULL"]["ann_return"],
            "FULL_vol":    stats["FULL"]["ann_vol"],
            "FULL_dd":     stats["FULL"]["max_dd"],
            "IS_sharpe":   stats["IS"]["sharpe"],
            "IS_ret":      stats["IS"]["ann_return"],
            "IS_dd":       stats["IS"]["max_dd"],
            "OOS_sharpe":  stats["OOS"]["sharpe"],
            "OOS_ret":     stats["OOS"]["ann_return"],
            "OOS_dd":      stats["OOS"]["max_dd"],
            "Y2022_sharpe": stats["Y2022"]["sharpe"],
            "Y2022_ret":    stats["Y2022"]["ann_return"],
            "Y2024_25_sharpe": stats["Y2024_25"]["sharpe"],
            "Y2024_25_ret":    stats["Y2024_25"]["ann_return"],
        })
    breakdown = pd.DataFrame(rows).set_index("sleeve")

    # Survivors: positive IS Sharpe AND positive 2022 return.
    is_pos = breakdown["IS_sharpe"] > 0
    y22_pos = breakdown["Y2022_ret"] > 0
    survivors = breakdown[is_pos & y22_pos].index.tolist()
    breakdown["survivor"] = breakdown.index.isin(survivors)

    print("\n=== Per sub-sleeve breakdown ===")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print(breakdown)

    # Combine survivors equal-weight on union calendar.
    if survivors:
        panel = pd.concat({k: scaled_streams[k] for k in survivors},
                          axis=1, sort=True).fillna(0.0)
        combined = panel.mean(axis=1)
    else:
        # No survivors: emit zero stream of unioned dates.
        all_idx = pd.DatetimeIndex(sorted(set(
            ts for s in scaled_streams.values() for ts in s.index
        )), tz="UTC")
        combined = pd.Series(0.0, index=all_idx)

    combined = combined.sort_index()
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC")

    combined_stats = split_stats(combined, bpy=252.0)
    print("\n=== Survivors:", survivors)
    print("=== Combined sleeve stats ===")
    for k, v in combined_stats.items():
        print(f"  {k:>9} : sharpe={v['sharpe']:+.3f} ret={v['ann_return']:+.3%} "
              f"vol={v['ann_vol']:.3%} dd={v['max_dd']:+.3%} n={v['n']}")

    # Append a 'COMBINED' row to breakdown for convenience.
    combined_row = {
        "scale_factor": 1.0,
        "FULL_sharpe": combined_stats["FULL"]["sharpe"],
        "FULL_ret":    combined_stats["FULL"]["ann_return"],
        "FULL_vol":    combined_stats["FULL"]["ann_vol"],
        "FULL_dd":     combined_stats["FULL"]["max_dd"],
        "IS_sharpe":   combined_stats["IS"]["sharpe"],
        "IS_ret":      combined_stats["IS"]["ann_return"],
        "IS_dd":       combined_stats["IS"]["max_dd"],
        "OOS_sharpe":  combined_stats["OOS"]["sharpe"],
        "OOS_ret":     combined_stats["OOS"]["ann_return"],
        "OOS_dd":      combined_stats["OOS"]["max_dd"],
        "Y2022_sharpe": combined_stats["Y2022"]["sharpe"],
        "Y2022_ret":    combined_stats["Y2022"]["ann_return"],
        "Y2024_25_sharpe": combined_stats["Y2024_25"]["sharpe"],
        "Y2024_25_ret":    combined_stats["Y2024_25"]["ann_return"],
        "survivor": True,
    }
    breakdown.loc["COMBINED"] = combined_row
    breakdown.to_csv(OUT_DIR / "defensive_breakdown.csv", float_format="%.4f")

    # Save combined returns
    out = combined.rename("ret").to_frame()
    out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
    out = out.reset_index()
    out.to_parquet(OUT_DIR / "defensive_returns.parquet", index=False)
    print("\nSaved:")
    print(f"  {OUT_DIR / 'defensive_returns.parquet'}")
    print(f"  {OUT_DIR / 'defensive_breakdown.csv'}")


if __name__ == "__main__":
    main()
