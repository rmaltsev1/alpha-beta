"""Vol-managed sleeve: long-only + dip-buying, vol-targeted per symbol.

Sub-sleeve A: long-only with daily rebalanced position = target_vol / realized_vol,
              capped [0, 2.5], target 10% ann vol per symbol, then aggregated.
Sub-sleeve B: dip-buying — enter long when prior bar's log return < -2 std (60d roll),
              hold up to 5 bars or until cumret > +1% / < -2%, then vol-managed.
Final sleeve = equal-weight of A and B.

Outputs:
    scratch/quant/volmgmt_returns.parquet
    scratch/quant/volmgmt_subA_returns.parquet
    scratch/quant/volmgmt_subB_returns.parquet
    scratch/quant/volmgmt_breakdown.csv

Walk-forward: every rolling stat is shifted by one bar before being used as a
signal so there is no look-ahead.
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

# ---------- config ----------
TARGET_ANN_VOL = 0.10            # 10% per-symbol vol target
LEVERAGE_CAP = 2.5
VOL_WINDOW_A = 30                # 30d realized-vol window for sub-sleeve A
VOL_WINDOW_B = 60                # 60d std for the dip threshold
DIP_SIGMA = -2.0                 # entry threshold (in sigmas)
HOLD_BARS = 5
TAKE_PROFIT = 0.01
STOP_LOSS = -0.02


def ann_factor(symbol: str) -> float:
    """D1 bars per year — crypto 365, others 252."""
    return 365.0 if symbol in CRYPTO else 252.0


def load_d1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "D1").sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    return df


# ---------- sub-sleeve A: vol-managed long-only ----------
def sub_A_position(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Daily-rebalanced long position = target_vol / realized_vol, capped.

    Walk-forward: we use the rolling std computed on returns up to t-1
    (.shift(1)) before deciding position for bar t.
    """
    bpy = ann_factor(symbol)
    # daily realized vol over the past VOL_WINDOW_A bars (excluding current bar)
    rv_d = df["log_ret"].rolling(VOL_WINDOW_A).std(ddof=1).shift(1)
    rv_ann = rv_d * np.sqrt(bpy)
    # avoid zero / NaN: until we have a vol estimate, sit flat
    raw = TARGET_ANN_VOL / rv_ann.replace(0.0, np.nan)
    pos = raw.clip(lower=0.0, upper=LEVERAGE_CAP).fillna(0.0)
    return pos


def sub_A_returns(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Net per-bar return for sub-sleeve A on this symbol."""
    pos = sub_A_position(df, symbol)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos * bar_ret
    cps = cost_for(symbol)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = gross - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


# ---------- sub-sleeve B: vol-managed dip-buying ----------
def sub_B_position(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Dip-buying with state-machine exit + vol-managed sizing.

    Trigger at bar t when log_ret[t-1] < -2 * std60d (computed up to t-2),
    enter long at bar t. Hold up to HOLD_BARS bars or until cumret >= +1%
    or <= -2% measured from entry close.

    We compute a base 0/1 (long-only) position with the state-machine, then
    multiply by the same vol-target multiplier as sub-sleeve A.
    """
    bpy = ann_factor(symbol)
    log_ret = df["log_ret"].values  # length N; log_ret[0] is NaN

    # std of log_ret over previous 60 bars, evaluated at t-1 (so the
    # log_ret[t-1] threshold uses std computed from bars [t-61, t-2]).
    std60 = df["log_ret"].rolling(VOL_WINDOW_B).std(ddof=1).shift(2).values
    # log return on the *prior* bar (t-1), as known at the open of bar t
    prior_ret = df["log_ret"].shift(1).values

    rv_d = df["log_ret"].rolling(VOL_WINDOW_A).std(ddof=1).shift(1).values
    rv_ann = rv_d * np.sqrt(bpy)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.where(rv_ann > 0, TARGET_ANN_VOL / rv_ann, 0.0)
    size = np.clip(size, 0.0, LEVERAGE_CAP)
    size = np.nan_to_num(size, nan=0.0)

    close = df["close"].values
    n = len(df)
    pos = np.zeros(n, dtype="float64")

    # state-machine
    in_trade = False
    entry_close = np.nan
    bars_held = 0
    entry_idx = -1

    for t in range(n):
        # Check entry trigger first (so we can act on bar t)
        if not in_trade:
            sig_ok = (
                np.isfinite(prior_ret[t])
                and np.isfinite(std60[t])
                and std60[t] > 0
                and prior_ret[t] < DIP_SIGMA * std60[t]
            )
            if sig_ok and size[t] > 0:
                in_trade = True
                entry_close = close[t - 1] if t >= 1 else close[t]
                bars_held = 0
                entry_idx = t
                pos[t] = size[t]
                bars_held += 1
                continue
            else:
                pos[t] = 0.0
                continue

        # Currently in a trade: position held during bar t is sized at entry.
        # Exit check happens at end-of-bar.
        pos[t] = size[entry_idx]
        bars_held += 1
        cur_cumret = close[t] / entry_close - 1.0 if entry_close > 0 else 0.0
        if (
            bars_held >= HOLD_BARS
            or cur_cumret >= TAKE_PROFIT
            or cur_cumret <= STOP_LOSS
        ):
            in_trade = False
            entry_close = np.nan
            bars_held = 0
            entry_idx = -1

    s = pd.Series(pos, index=df.index)
    return s


def sub_B_returns(df: pd.DataFrame, symbol: str) -> pd.Series:
    pos = sub_B_position(df, symbol)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos * bar_ret
    cps = cost_for(symbol)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = gross - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


# ---------- no-vol-mgmt baseline for sub-sleeve A ----------
def sub_A_baseline_returns(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Naive buy-and-hold (position=1), to compare vol-managed Sharpe lift."""
    bar_ret = df["close"].pct_change().fillna(0.0)
    cps = cost_for(symbol)
    # buy at first bar, hold to end: one cost event on bar 0.
    pos = pd.Series(1.0, index=df.index)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = bar_ret - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


# ---------- aggregation ----------
def _to_daily(s: pd.Series) -> pd.Series:
    """Collapse a per-symbol return series (timestamped at the broker close)
    onto calendar day. Crypto closes at 00:00 UTC of day d+1, FX/index at
    21:00 / 22:00 UTC of day d. Bucket by calendar date of the closing bar.
    """
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).floor("1D")
    # if multiple bars land on the same date (shouldn't, but be safe), sum
    return s.groupby(level=0).sum()


def equal_weight(streams: dict[str, pd.Series]) -> pd.Series:
    """Daily equal-weight across symbols. Each per-symbol return stream is
    first collapsed to calendar-day, then averaged 1/N across all 13 columns
    (missing days count as zero — that symbol contributed nothing that day).
    """
    daily = {s: _to_daily(r) for s, r in streams.items()}
    df = pd.concat(daily, axis=1, sort=True)
    df = df.sort_index()
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df.fillna(0.0).mean(axis=1)


def class_weighted(streams: dict[str, pd.Series]) -> pd.Series:
    """1/3 weight to each asset class, equal-weight within class."""
    classes = {"crypto": CRYPTO, "fx": FOREX, "index": INDEX}
    pieces = []
    for cname, syms in classes.items():
        sub = {s: streams[s] for s in syms if s in streams}
        if not sub:
            continue
        pieces.append(equal_weight(sub).rename(cname))
    df = pd.concat(pieces, axis=1, sort=True).fillna(0.0)
    return df.mean(axis=1)


# ---------- stats ----------
def perf_stats(ret: pd.Series, bpy: float = 365.0) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return dict(sharpe=np.nan, ann_return=np.nan, ann_vol=np.nan, max_dd=np.nan, n=0)
    ann_ret = r.mean() * bpy
    ann_vol = r.std(ddof=0) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(
        sharpe=float(sharpe),
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        max_dd=float(dd),
        n=int(len(r)),
    )


def yearly_breakdown(ret: pd.Series, bpy: float = 365.0) -> pd.DataFrame:
    r = ret.dropna()
    if len(r) == 0:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(r.index)
    rows = []
    for yr in sorted(set(idx.year)):
        rr = r[idx.year == yr]
        rows.append(dict(year=yr, **perf_stats(rr, bpy)))
    return pd.DataFrame(rows).set_index("year")


def split_stats(ret: pd.Series, bpy: float = 365.0) -> dict:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_mask = idx < SPLIT
    return {
        "FULL": perf_stats(r, bpy),
        "IS": perf_stats(r[is_mask], bpy),
        "OOS": perf_stats(r[~is_mask], bpy),
    }


# ---------- main ----------
def main() -> None:
    print("=== Loading D1 data for 13 symbols ===")
    data = {s: load_d1(s) for s in ALL_SYMBOLS}

    # ---- per-symbol sub-A / sub-B / baseline ----
    A_streams: dict[str, pd.Series] = {}
    B_streams: dict[str, pd.Series] = {}
    BL_streams: dict[str, pd.Series] = {}  # no-vol-mgmt baseline
    breakdown_rows = []

    for sym in ALL_SYMBOLS:
        df = data[sym]
        retA = sub_A_returns(df, sym)
        retB = sub_B_returns(df, sym)
        retBL = sub_A_baseline_returns(df, sym)

        A_streams[sym] = retA
        B_streams[sym] = retB
        BL_streams[sym] = retBL

        sA = split_stats(retA)
        sB = split_stats(retB)
        sBL = split_stats(retBL)
        row = {"symbol": sym}
        for k in ("FULL", "IS", "OOS"):
            row[f"A_{k}_sharpe"] = sA[k]["sharpe"]
            row[f"A_{k}_ret"] = sA[k]["ann_return"]
            row[f"A_{k}_vol"] = sA[k]["ann_vol"]
            row[f"A_{k}_dd"] = sA[k]["max_dd"]
            row[f"B_{k}_sharpe"] = sB[k]["sharpe"]
            row[f"B_{k}_ret"] = sB[k]["ann_return"]
            row[f"B_{k}_vol"] = sB[k]["ann_vol"]
            row[f"B_{k}_dd"] = sB[k]["max_dd"]
            row[f"BL_{k}_sharpe"] = sBL[k]["sharpe"]
            row[f"BL_{k}_ret"] = sBL[k]["ann_return"]
            row[f"BL_{k}_vol"] = sBL[k]["ann_vol"]
            row[f"BL_{k}_dd"] = sBL[k]["max_dd"]
        breakdown_rows.append(row)

    breakdown = pd.DataFrame(breakdown_rows).set_index("symbol")
    breakdown.to_csv(OUT_DIR / "volmgmt_breakdown.csv", float_format="%.4f")

    # ---- aggregate to sleeves ----
    A_eq = equal_weight(A_streams)
    B_eq = equal_weight(B_streams)
    BL_eq = equal_weight(BL_streams)

    A_cls = class_weighted(A_streams)
    BL_cls = class_weighted(BL_streams)

    # Final combined sleeve = 50/50 A and B (equal-weight aggregation versions)
    combined = (A_eq.add(B_eq, fill_value=0.0)) / 2.0
    # Make sure combined is reindexed
    combined = combined.sort_index()

    # ---- save parquets ----
    def _save(s: pd.Series, path: Path) -> None:
        out = s.rename("ret").to_frame()
        out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
        out = out.reset_index()
        out.to_parquet(path, index=False)

    _save(A_eq, OUT_DIR / "volmgmt_subA_returns.parquet")
    _save(B_eq, OUT_DIR / "volmgmt_subB_returns.parquet")
    _save(combined, OUT_DIR / "volmgmt_returns.parquet")

    # ---- print report ----
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    print("\n=== Headline (FULL / IS / OOS) ===")
    rows = []
    for name, ret in [
        ("subA_eq", A_eq),
        ("subA_class", A_cls),
        ("subB_eq", B_eq),
        ("baseline_BH_eq", BL_eq),
        ("baseline_BH_class", BL_cls),
        ("combined_AB", combined),
    ]:
        s = split_stats(ret)
        rows.append({
            "sleeve": name,
            "FULL_sharpe": s["FULL"]["sharpe"], "FULL_ret": s["FULL"]["ann_return"],
            "FULL_vol": s["FULL"]["ann_vol"], "FULL_dd": s["FULL"]["max_dd"],
            "IS_sharpe": s["IS"]["sharpe"], "IS_ret": s["IS"]["ann_return"], "IS_dd": s["IS"]["max_dd"],
            "OOS_sharpe": s["OOS"]["sharpe"], "OOS_ret": s["OOS"]["ann_return"], "OOS_dd": s["OOS"]["max_dd"],
        })
    headline = pd.DataFrame(rows).set_index("sleeve")
    print(headline)
    headline.to_csv(OUT_DIR / "volmgmt_headline.csv", float_format="%.4f")

    print("\n=== Year-by-year: combined sleeve ===")
    yc = yearly_breakdown(combined)
    print(yc)
    yc.to_csv(OUT_DIR / "volmgmt_yearly_combined.csv", float_format="%.4f")

    print("\n=== Year-by-year: sub-A equal-weight ===")
    yA = yearly_breakdown(A_eq)
    print(yA)
    yA.to_csv(OUT_DIR / "volmgmt_yearly_subA.csv", float_format="%.4f")

    print("\n=== Year-by-year: sub-B equal-weight ===")
    yB = yearly_breakdown(B_eq)
    print(yB)
    yB.to_csv(OUT_DIR / "volmgmt_yearly_subB.csv", float_format="%.4f")

    print("\n=== Year-by-year: naive buy-and-hold equal-weight ===")
    yBL = yearly_breakdown(BL_eq)
    print(yBL)
    yBL.to_csv(OUT_DIR / "volmgmt_yearly_baseline.csv", float_format="%.4f")

    print("\n=== Per-symbol sub-A Sharpe lift vs naive (FULL) ===")
    lift = (breakdown["A_FULL_sharpe"] - breakdown["BL_FULL_sharpe"]).sort_values(ascending=False)
    print(lift)

    print("\nFiles written to:", OUT_DIR)


if __name__ == "__main__":
    main()
