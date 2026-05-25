"""Post-loss / drawdown-recovery strategies — wave 6.

Eight sub-sleeves that systematically buy after losses at different scales:

  S1 3D_DOWN_BUY          per-symbol: 3 consecutive D1 down days -> long 2 days.
  S2 DD5PCT_BOUNCE        US-index + crypto: -5% off rolling-20d high -> long
                           up to 5 days OR until +3% off intra-trade low.
  S3 CAPITULATION_BUY     crypto only: D1 ret < -3 SD AND vol > 1.5x avg ->
                           long 3 days.
  S4 LOW_TOUCH_20D        per-symbol: touch of 20d low -> long 3 days.
  S5 RISK_OFF_RECOVERY    multi-asset: SPX -10% off 60d high AND realised
                           vol peaked-and-dropping -> long equity basket 5 d.
  S6 WEEKLY_LOSERS        cross-sectional: each week, long the bottom-3
                           symbols by trailing 5d return, for 5 days.
  S7 STREAK_BREAK         per-symbol: 5+ consecutive down bars -> long 3 d.
                           (Spec also asks short after 5+ up bars; included
                           symmetrically.)
  S8 VOL_CLIMAX           per-symbol: 5d realised vol above IS p95 AND price
                           at low of the lookback period -> long 5 days.

Methodology
-----------
- IS  < 2024-01-01 ; OOS >= 2024-01-01.
- All thresholds and z-score baselines are walk-forward (use only data
  available *before* bar t — i.e. shift(1) and expanding statistics, with
  IS quantiles for sleeves where stated).
- Each sub-sleeve is vol-scaled to a 5% IS realised vol (single scalar).
- Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
- Survivors combined equal-weight on the union calendar.

Outputs
-------
  scratch/wave6/dd_recovery.py
  scratch/wave6/dd_recovery_returns.parquet
  scratch/wave6/dd_recovery_breakdown.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from alphabeta import CRYPTO, INDEX, ALL_SYMBOLS, get_candles  # noqa: E402
from alphabeta.backtest import cost_for  # noqa: E402

OUT_DIR = REPO / "scratch" / "wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05

US_INDICES = ["SPX500_USD", "NAS100_USD", "US30_USD"]
EQUITY_BASKET = ["SPX500_USD", "NAS100_USD", "US30_USD",
                 "UK100_GBP", "DE30_EUR", "JP225_USD"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_d1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "D1").sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"]).diff()
    return df


def perf_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return dict(sharpe=np.nan, ann_return=np.nan, ann_vol=np.nan,
                    max_dd=np.nan, n=0)
    ann_ret = r.mean() * bpy
    ann_vol = r.std(ddof=0) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min() if len(eq) > 0 else 0.0
    return dict(sharpe=float(sharpe), ann_return=float(ann_ret),
                ann_vol=float(ann_vol), max_dd=float(dd), n=int(len(r)))


def split_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_mask = idx < SPLIT
    return {
        "FULL":     perf_stats(r, bpy),
        "IS":       perf_stats(r[is_mask], bpy),
        "OOS":      perf_stats(r[~is_mask], bpy),
        "Y2022":    perf_stats(r[(idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                  (idx < pd.Timestamp("2023-01-01", tz="UTC"))], bpy),
        "Y2024_25": perf_stats(r[idx >= pd.Timestamp("2024-01-01", tz="UTC")], bpy),
    }


def to_daily(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).floor("1D")
    s = s.groupby(level=0).sum()
    return s


def vol_scale(ret: pd.Series, target: float = TARGET_SUB_VOL,
              bpy: float = 252.0) -> tuple[pd.Series, float]:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_r = r[idx < SPLIT]
    iv = is_r.std(ddof=0) * np.sqrt(bpy)
    if iv <= 0 or not np.isfinite(iv):
        return ret * 0.0, 0.0
    k = target / iv
    return ret * k, float(k)


def position_to_returns(df: pd.DataFrame, pos: pd.Series, symbol: str) -> pd.Series:
    pos = pos.astype("float64").fillna(0.0)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos * bar_ret
    cps = cost_for(symbol)
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    net = gross - dpos * cps
    net.index = df["timestamp"]
    net.name = symbol
    return net


def hold_signal(trigger: pd.Series, hold_bars: int, direction: float = 1.0,
                latest_wins: bool = True) -> pd.Series:
    """Build a position series: each True in `trigger` opens a `hold_bars`-bar
    position in `direction`. trigger[t] is assumed already shifted so that it
    represents a signal observable at the *start* of bar t.

    If a new trigger fires inside an existing hold, the hold is reset
    (latest_wins=True) or ignored (False).
    """
    n = len(trigger)
    pos = np.zeros(n)
    remaining = 0
    trig = trigger.fillna(False).astype(bool).values
    for t in range(n):
        if trig[t] and (latest_wins or remaining == 0):
            remaining = hold_bars
        if remaining > 0:
            pos[t] = direction
            remaining -= 1
    return pd.Series(pos, index=trigger.index)


# ---------------------------------------------------------------------------
# S1: 3 consecutive D1 down days -> long 2 days (per symbol).
# ---------------------------------------------------------------------------
def sleeve1_three_down(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        r = df["ret"].fillna(0.0)
        down = (r < 0).astype(int)
        # 3 consecutive down ending on day t-1 (so we can long bar t)
        three_dn = (down.rolling(3).sum() == 3).shift(1).fillna(False)
        pos = hold_signal(three_dn, hold_bars=2, direction=+1.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S2: -5% off 20d rolling high -> long up to 5 days OR exit when
# intra-trade close rises 3% off the in-trade low. US indices + crypto only.
# ---------------------------------------------------------------------------
def sleeve2_dd5_bounce(data: dict[str, pd.DataFrame]) -> pd.Series:
    syms = US_INDICES + CRYPTO
    streams = []
    for sym in syms:
        df = data[sym]
        close = df["close"]
        roll_hi = close.rolling(20, min_periods=10).max()
        # trigger uses info up to bar t-1
        dd = (close / roll_hi - 1.0).shift(1)
        trigger = (dd <= -0.05).fillna(False).values
        bar_ret_arr = close.pct_change().fillna(0.0).values
        close_arr = close.values
        n = len(df)
        pos = np.zeros(n)
        in_trade = False
        remaining = 0
        trade_low_close = np.nan  # tracked on closes since entry
        for t in range(n):
            if not in_trade and trigger[t]:
                in_trade = True
                remaining = 5
                # reset trade low to entry-bar previous close
                trade_low_close = close_arr[t - 1] if t > 0 else close_arr[t]
            if in_trade:
                pos[t] = 1.0
                # update trade low with this bar's close
                if close_arr[t] < trade_low_close:
                    trade_low_close = close_arr[t]
                remaining -= 1
                bounce = close_arr[t] / trade_low_close - 1.0
                if bounce >= 0.03 or remaining <= 0:
                    in_trade = False
                    remaining = 0
                    trade_low_close = np.nan
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S3: Capitulation: ret < -3 SD AND vol > 1.5x avg -> long 3 days. Crypto only.
# SD and avg-volume baselines are *expanding* (walk-forward) means / std,
# computed from data prior to bar t.
# ---------------------------------------------------------------------------
def sleeve3_capitulation(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    min_lb = 60
    for sym in CRYPTO:
        df = data[sym]
        r = df["ret"]
        v = df["volume"].astype("float64")
        # Expanding mean / std using only past data (shift by 1 then expanding).
        rm = r.shift(1).expanding(min_periods=min_lb).mean()
        rs = r.shift(1).expanding(min_periods=min_lb).std(ddof=0)
        vm = v.shift(1).expanding(min_periods=min_lb).mean()
        # ret-z computed on the realised return at bar t-1 (signal known at
        # start of bar t).
        z = (r.shift(1) - rm) / rs
        vrat = v.shift(1) / vm
        trig = ((z < -3.0) & (vrat > 1.5)).fillna(False)
        pos = hold_signal(trig, hold_bars=3, direction=+1.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S4: Touch of 20d low -> long 3 days. Per symbol.
# "Touch" = low[t-1] equals 20d rolling low computed over bars ending at t-1.
# ---------------------------------------------------------------------------
def sleeve4_low_touch(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        low = df["low"]
        roll_low_20 = low.rolling(20, min_periods=10).min()
        touched = (low <= roll_low_20).shift(1).fillna(False)
        pos = hold_signal(touched, hold_bars=3, direction=+1.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S5: Multi-asset risk-off recovery.
# Trigger (known at start of bar t):
#   - SPX_close[t-1] / SPX_high(60)[t-1] - 1 <= -0.10
#   - SPX 20d realised vol has *peaked then dropped*:
#       rv20[t-1] < rv20[t-6]   AND   max(rv20[t-15..t-6]) > rv20[t-1]
#     (vol turned down off a recent peak).
# Action: long equal-weight equity basket for 5 days.
# Trigger reset if still in trade (latest wins).
# ---------------------------------------------------------------------------
def sleeve5_risk_off(data: dict[str, pd.DataFrame]) -> pd.Series:
    spx = data["SPX500_USD"]
    close = spx["close"]
    log_ret = spx["log_ret"]
    roll_hi_60 = close.rolling(60, min_periods=30).max()
    dd = close / roll_hi_60 - 1.0
    rv20 = log_ret.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252.0)
    # SPX time axis (the basket trades on this calendar)
    ts = spx["timestamp"]
    # Evaluate trigger at start of bar t using data up to t-1.
    dd_s = dd.shift(1)
    rv = rv20.shift(1)
    rv_prev = rv20.shift(6)
    rv_peak_win = rv20.shift(6).rolling(10, min_periods=5).max()
    cond_dd = (dd_s <= -0.10)
    cond_vol_down = (rv < rv_prev) & (rv_peak_win > rv)
    trigger = (cond_dd & cond_vol_down).fillna(False).values

    n = len(spx)
    in_trade = np.zeros(n, dtype=bool)
    remaining = 0
    for t in range(n):
        if trigger[t]:
            remaining = 5
        if remaining > 0:
            in_trade[t] = True
            remaining -= 1
    # For each basket member: position = +1 on dates that overlap a trade,
    # else 0. Trades are dated by SPX calendar; we map each member's bar
    # timestamps to SPX-day floors and check membership.
    spx_trade_days = set(
        pd.DatetimeIndex(ts[in_trade]).tz_convert("UTC").floor("1D")
    )
    streams = []
    for sym in EQUITY_BASKET:
        df = data[sym]
        days = pd.DatetimeIndex(df["timestamp"]).tz_convert("UTC").floor("1D")
        pos = pd.Series([1.0 if d in spx_trade_days else 0.0 for d in days],
                        index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S6: Weekly losers comeback.
# Each ISO week, on the first available bar of the week (>= Monday), long
# the symbols with the worst trailing 5d return (bottom 3 of the 13). Hold 5
# bars on each chosen symbol's own calendar.
# ---------------------------------------------------------------------------
def sleeve6_weekly_losers(data: dict[str, pd.DataFrame]) -> pd.Series:
    # Build a panel of 5d trailing returns indexed by day (floor).
    syms = list(data.keys())
    closes = {}
    for sym in syms:
        df = data[sym]
        s = pd.Series(df["close"].values,
                      index=pd.DatetimeIndex(df["timestamp"]).floor("1D"))
        s = s[~s.index.duplicated(keep="last")]
        closes[sym] = s
    panel = pd.concat(closes, axis=1).sort_index()
    # 5d trailing return based on prior close
    trail5 = panel.pct_change(5).shift(1)
    # ISO week id
    iso = pd.DataFrame({
        "date": panel.index,
        "yw": panel.index.to_series().dt.strftime("%G-%V").values,
    })
    iso = iso.set_index("date")
    first_of_week = iso.groupby("yw").head(1).index  # first calendar date per ISO week
    # Selections per first-of-week date
    selections: dict[pd.Timestamp, list[str]] = {}
    for d in first_of_week:
        row = trail5.loc[d]
        if row.notna().sum() < 5:
            continue
        bot3 = row.nsmallest(3).index.tolist()
        selections[d] = bot3

    # Map each first-of-week date forward for up to 5 calendar days.
    # For each symbol, build a holding mask on its own calendar.
    streams = []
    for sym in syms:
        df = data[sym]
        days = pd.DatetimeIndex(df["timestamp"]).floor("1D")
        active = pd.Series(0.0, index=df.index)
        for d, picks in selections.items():
            if sym not in picks:
                continue
            # window = [d, d + 5 calendar days)
            end = d + pd.Timedelta(days=5)
            mask = (days >= d) & (days < end)
            active[mask] = 1.0
        ret = position_to_returns(df, active, sym)
        streams.append(to_daily(ret).rename(sym))
    aligned = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return aligned.mean(axis=1)


# ---------------------------------------------------------------------------
# S7: Streak-break trades.
# After 5+ consecutive down bars -> long for 3 bars.
# After 5+ consecutive up   bars -> short for 3 bars.
# Per symbol.
# ---------------------------------------------------------------------------
def sleeve7_streak_break(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        r = df["ret"].fillna(0.0)
        up = (r > 0).astype(int)
        dn = (r < 0).astype(int)
        run_up = up.rolling(5).sum()
        run_dn = dn.rolling(5).sum()
        # trigger on close of t-1 -> position starts at t
        long_trig = (run_dn == 5).shift(1).fillna(False)
        short_trig = (run_up == 5).shift(1).fillna(False)
        long_pos = hold_signal(long_trig, hold_bars=3, direction=+1.0)
        short_pos = hold_signal(short_trig, hold_bars=3, direction=-1.0)
        # combine (allow superposition; both rarely overlap)
        pos = (long_pos + short_pos).clip(-1.0, 1.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S8: Volatility climax.
# 5d realised vol > IS p95 of 5d vol AND price at low of the lookback (5d).
# Long 5 days. Per symbol.
# ---------------------------------------------------------------------------
def sleeve8_vol_climax(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym, df in data.items():
        log_ret = df["log_ret"]
        rv5 = log_ret.rolling(5, min_periods=5).std(ddof=0) * np.sqrt(252.0)
        # IS p95
        is_mask = df["timestamp"] < SPLIT
        thresh = float(rv5[is_mask].dropna().quantile(0.95))
        # at-low: close <= 5d rolling min of close (using last 5 bars including t-1)
        close = df["close"]
        roll_min_5 = close.rolling(5, min_periods=5).min()
        at_low = (close <= roll_min_5)
        trig = ((rv5 > thresh) & at_low).shift(1).fillna(False)
        pos = hold_signal(trig, hold_bars=5, direction=+1.0)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# Per-symbol diagnostics (used to identify "cleanest bottom-buying edge").
# ---------------------------------------------------------------------------
def per_symbol_three_down(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    out = {}
    for sym, df in data.items():
        r = df["ret"].fillna(0.0)
        down = (r < 0).astype(int)
        three_dn = (down.rolling(3).sum() == 3).shift(1).fillna(False)
        pos = hold_signal(three_dn, hold_bars=2, direction=+1.0)
        ret = position_to_returns(df, pos, sym)
        out[sym] = to_daily(ret)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading D1 data for 13 symbols ===")
    data = {s: load_d1(s) for s in ALL_SYMBOLS}
    for s in ALL_SYMBOLS:
        df = data[s]
        print(f"  {s:14} bars={len(df):4d}  "
              f"{df['timestamp'].iloc[0].date()}..{df['timestamp'].iloc[-1].date()}")

    builders = {
        "S1_3D_DOWN_BUY":      lambda: sleeve1_three_down(data),
        "S2_DD5PCT_BOUNCE":    lambda: sleeve2_dd5_bounce(data),
        "S3_CAPITULATION":     lambda: sleeve3_capitulation(data),
        "S4_LOW_TOUCH_20D":    lambda: sleeve4_low_touch(data),
        "S5_RISK_OFF_RECOV":   lambda: sleeve5_risk_off(data),
        "S6_WEEKLY_LOSERS":    lambda: sleeve6_weekly_losers(data),
        "S7_STREAK_BREAK":     lambda: sleeve7_streak_break(data),
        "S8_VOL_CLIMAX":       lambda: sleeve8_vol_climax(data),
    }

    print("\n=== Building sub-sleeves ===")
    raw_streams: dict[str, pd.Series] = {}
    for name, fn in builders.items():
        s = fn().sort_index()
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
        raw_streams[name] = s
        print(f"  built {name:<22} n={len(s)}  "
              f"first={s.index.min().date()}  last={s.index.max().date()}  "
              f"sum_ret={s.sum():+.4f}")

    print("\n=== Vol-scale + stats ===")
    scaled_streams: dict[str, pd.Series] = {}
    rows = []
    for name, s in raw_streams.items():
        scaled, k = vol_scale(s, target=TARGET_SUB_VOL, bpy=252.0)
        scaled_streams[name] = scaled
        st = split_stats(scaled, bpy=252.0)
        rows.append({
            "sleeve": name,
            "scale_factor": k,
            "FULL_sharpe":   st["FULL"]["sharpe"],
            "FULL_ret":      st["FULL"]["ann_return"],
            "FULL_vol":      st["FULL"]["ann_vol"],
            "FULL_dd":       st["FULL"]["max_dd"],
            "IS_sharpe":     st["IS"]["sharpe"],
            "IS_ret":        st["IS"]["ann_return"],
            "IS_dd":         st["IS"]["max_dd"],
            "OOS_sharpe":    st["OOS"]["sharpe"],
            "OOS_ret":       st["OOS"]["ann_return"],
            "OOS_dd":        st["OOS"]["max_dd"],
            "Y2022_sharpe":  st["Y2022"]["sharpe"],
            "Y2022_ret":     st["Y2022"]["ann_return"],
            "Y2024_25_sharpe": st["Y2024_25"]["sharpe"],
            "Y2024_25_ret":    st["Y2024_25"]["ann_return"],
        })
    breakdown = pd.DataFrame(rows).set_index("sleeve")

    # Survivors
    is_ok = breakdown["IS_sharpe"] >= 0.5
    oos_ok = breakdown["OOS_sharpe"] >= 0.0
    survivors = breakdown[is_ok & oos_ok].index.tolist()
    breakdown["survivor"] = breakdown.index.isin(survivors)

    print("\n=== Per-sleeve breakdown ===")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print(breakdown)

    # Combined survivors equal-weight on union calendar
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
    combined.index = pd.DatetimeIndex(combined.index, tz="UTC")
    combined_stats = split_stats(combined, bpy=252.0)

    print("\n=== Survivors:", survivors)
    print("=== Combined dd-recovery sleeve stats ===")
    for k, v in combined_stats.items():
        print(f"  {k:>9} : sharpe={v['sharpe']:+.3f} ret={v['ann_return']:+.3%} "
              f"vol={v['ann_vol']:.3%} dd={v['max_dd']:+.3%} n={v['n']}")

    # COMBINED row
    breakdown.loc["COMBINED"] = {
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
    breakdown.to_csv(OUT_DIR / "dd_recovery_breakdown.csv", float_format="%.4f")

    out = combined.rename("ret").to_frame()
    out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
    out = out.reset_index()
    out.to_parquet(OUT_DIR / "dd_recovery_returns.parquet", index=False)

    # ---- per-symbol bottom-buying diagnostic ----
    print("\n=== Per-symbol S1 (3-down-day) Sharpe — bottom-buying edge ===")
    per_sym = per_symbol_three_down(data)
    sym_rows = []
    for sym, s in per_sym.items():
        st = split_stats(s, bpy=(365.0 if sym in CRYPTO else 252.0))
        sym_rows.append({
            "symbol": sym,
            "IS_sharpe":  st["IS"]["sharpe"],
            "OOS_sharpe": st["OOS"]["sharpe"],
            "Y2022_sharpe": st["Y2022"]["sharpe"],
            "FULL_sharpe": st["FULL"]["sharpe"],
            "FULL_ret":    st["FULL"]["ann_return"],
        })
    sym_df = pd.DataFrame(sym_rows).set_index("symbol")
    sym_df = sym_df.sort_values("FULL_sharpe", ascending=False)
    print(sym_df)

    print("\nSaved:")
    print(f"  {OUT_DIR / 'dd_recovery_returns.parquet'}")
    print(f"  {OUT_DIR / 'dd_recovery_breakdown.csv'}")


if __name__ == "__main__":
    main()
