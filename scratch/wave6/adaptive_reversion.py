"""Wave-6: Reversal-of-reversal + adaptive mean-reversion sleeves.

A vanilla D1 reversion strategy (D1REV) fades the prior day's move. It is
profitable in oscillating regimes (2020-21, 2023, parts of 2024) but bleeds
in persistent trends (2022, 2025 momentum runs). The sub-sleeves here try
to harvest the reversion edge while limiting damage when the regime breaks.

Vanilla baseline (matches scratch/models/strategies_v2.d1_reversion with
threshold_bps=50):
    log_ret = log(close/close.shift(1))
    pos[t] = -1 if log_ret[t-1] > +50bp
           = +1 if log_ret[t-1] < -50bp
           =  0 otherwise
    (i.e. signal shifted by 1; held for 1 bar)

Sub-sleeves built:
    S1  Self-adapting D1 reversion (flip to momentum when trailing-21d PnL
        of vanilla reversion is negative).
    S2  Reversal-of-reversal: when a vanilla D1REV trade just LOST, take a
        2-bar continuation trade in the direction price kept moving.
    S3a Multi-period D1 reversion, 3-bar hold.
    S3b Multi-period D1 reversion, 5-bar hold.
    S4  Drawdown-adaptive reversion: vanilla D1REV but pos size halved when
        trailing-21d DD < -3%; restored when trailing-21d return turns
        positive.
    S5  Two-sided reversion gated by 20d trend (fade up only in downtrend,
        fade down only in uptrend).
    S6  Momentum-confirmed exit: vanilla D1REV with early exit when the
        position is losing AND 5d trend confirms move against us.
    S7  Volatility-conditional reversion: trade only when 30d realised vol
        is in the IS middle 60% range.

Plus a vanilla baseline (D1REV_BASE) for direct comparison.

Methodology:
    IS  < 2024-01-01;  OOS >= 2024-01-01.
    All adaptive thresholds/quantiles set on IS data only.
    Each survivor is vol-scaled so IS realised vol == 5% ann (single scalar).
    Survival filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
    Survivors combined equal-weight on the union calendar.

Outputs:
    scratch/wave6/adaptive_reversion.py             (this file)
    scratch/wave6/adaptive_reversion_returns.parquet (combined survivor returns)
    scratch/wave6/adaptive_reversion_breakdown.csv  (per sub-sleeve table)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import CRYPTO, get_candles
from alphabeta.backtest import cost_for


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scratch" / "wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05

# The 6 equity-index universe used for the D1REV sleeves.
INDICES_EQ = ["SPX500_USD", "NAS100_USD", "US30_USD",
              "UK100_GBP", "DE30_EUR", "JP225_USD"]

REV_THRESH_BPS = 50.0  # match D1REV_NAS/UK/SPX in master_portfolio.py


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
        return dict(sharpe=np.nan, ann_return=np.nan, ann_vol=np.nan,
                    max_dd=np.nan, n=0)
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
        "FULL":  perf_stats(r, bpy),
        "IS":    perf_stats(r[is_mask], bpy),
        "OOS":   perf_stats(r[~is_mask], bpy),
        "Y2022": perf_stats(r[(idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
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


# ---------------------------------------------------------------------------
# vanilla D1 reversion signal (matches strategies_v2.d1_reversion)
# Returns the *position* series aligned to df rows.
# ---------------------------------------------------------------------------
def vanilla_d1rev_signal(df: pd.DataFrame, threshold_bps: float = REV_THRESH_BPS
                          ) -> pd.Series:
    ret = np.log(df["close"] / df["close"].shift(1))
    thresh = threshold_bps / 10_000.0
    sig = pd.Series(0.0, index=df.index)
    sig[ret > thresh] = -1.0
    sig[ret < -thresh] = +1.0
    return sig.shift(1).fillna(0.0)


# ---------------------------------------------------------------------------
# Baseline: vanilla D1REV across 6 indices, equal-weight (cross-sectional
# average of per-symbol returns). This is the apples-to-apples comparator.
# ---------------------------------------------------------------------------
def baseline_vanilla(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        pos = vanilla_d1rev_signal(df)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S1: Self-adapting D1 reversion.
# Track rolling-21d PnL of vanilla D1REV per symbol. If <= 0 at start of bar
# t, FLIP the sign of the signal (turn the reversion into momentum). Else
# keep the reversion sign. Per-symbol decision; combined equal-weight.
# ---------------------------------------------------------------------------
def sleeve1_self_adapting(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        pos_rev = vanilla_d1rev_signal(df)
        # Compute per-bar pnl of the reversion signal *gross* (cost is small
        # relative to the regime detector; we want a clean P&L signal).
        bar_ret = df["close"].pct_change().fillna(0.0)
        gross_rev = pos_rev * bar_ret
        # Trailing 21d sum of reversion PnL — using info up to bar t-1.
        trail = gross_rev.rolling(21, min_periods=10).sum().shift(1)
        # Default to reversion until enough history.
        sign = np.where(trail.fillna(1.0) >= 0.0, 1.0, -1.0)
        adapted = pos_rev * pd.Series(sign, index=df.index)
        ret = position_to_returns(df, adapted, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S2: Reversal-of-reversal.
# When the vanilla D1REV trade at bar t-1 LOST (sign(pos_rev[t-1])*ret[t-1] <0,
# i.e. price kept moving in the original direction), expect continuation: go
# in the direction that price was actually moving for 2 bars.
# Direction = sign of log_ret on the trigger day (the day prior to the failed
# fade — that's the original "big move" day).
# ---------------------------------------------------------------------------
def sleeve2_revrev(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        log_ret = np.log(df["close"] / df["close"].shift(1))
        # vanilla rev signal as a fresh series (1 day-ahead-shifted positions).
        thresh = REV_THRESH_BPS / 10_000.0
        big_up = (log_ret > thresh).astype(float)
        big_dn = (log_ret < -thresh).astype(float)
        # raw reversion direction *for next day*: -1 after up, +1 after down
        # (i.e. pos_rev[t] = raw_dir[t-1])
        raw_dir = (-big_up + big_dn)            # at the day of the move
        # next-day reversion pnl gross (without costs):
        next_ret = df["close"].pct_change().shift(-1)  # ret of bar t+1
        rev_pnl = raw_dir * next_ret                   # pnl realised at t+1
        # rev failed on bar t  =>  trigger continuation for bars t+2 and t+3.
        failed = (rev_pnl < 0) & (raw_dir != 0)
        cont_dir = -raw_dir                            # opposite of fade = direction of move
        n = len(df)
        pos = np.zeros(n)
        hold = 0
        cur = 0.0
        for t in range(n):
            if hold > 0:
                pos[t] = cur
                hold -= 1
                continue
            # at start of bar t we know failed[t-2] (since failed depends on
            # ret at t-1). Use shift-by-2.
            if t >= 2 and bool(failed.iloc[t - 2]):
                cur = float(cont_dir.iloc[t - 2])
                pos[t] = cur
                hold = 1  # one more bar after this -> 2 bars total
            else:
                pos[t] = 0.0
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S3: Multi-period reversion.
# Standard D1 fade signal, but instead of 1-bar hold, hold for `hold_bars`.
# If a new fade signal fires while already in a position, override (latest
# signal wins). At the end of the hold, exit to flat.
# ---------------------------------------------------------------------------
def _multi_period_reversion(data: dict[str, pd.DataFrame], hold_bars: int) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        log_ret = np.log(df["close"] / df["close"].shift(1))
        thresh = REV_THRESH_BPS / 10_000.0
        raw_dir = pd.Series(0.0, index=df.index)
        raw_dir[log_ret > thresh] = -1.0
        raw_dir[log_ret < -thresh] = +1.0
        # raw_dir at day t triggers fade for days t+1..t+hold_bars
        n = len(df)
        pos = np.zeros(n)
        remaining = 0
        cur = 0.0
        # iterate over each bar t representing the *current* day. The trigger
        # observed at end of day t-1 starts a fade at day t.
        trig = raw_dir.shift(1).fillna(0.0).values
        for t in range(n):
            if trig[t] != 0.0:
                # new signal: reset the fade to fresh hold
                cur = float(trig[t])
                remaining = hold_bars
            if remaining > 0:
                pos[t] = cur
                remaining -= 1
            else:
                pos[t] = 0.0
                cur = 0.0
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


def sleeve3a_mp_3d(data): return _multi_period_reversion(data, hold_bars=3)
def sleeve3b_mp_5d(data): return _multi_period_reversion(data, hold_bars=5)


# ---------------------------------------------------------------------------
# S4: Drawdown-adaptive reversion.
# Vanilla D1REV. Pos size starts at 1.0. If trailing-21d DD of the *strategy*
# is worse than -3%, halve to 0.5. Re-double to 1.0 once trailing-21d return
# is positive again.
# ---------------------------------------------------------------------------
def sleeve4_dd_adaptive(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        pos_rev = vanilla_d1rev_signal(df)
        bar_ret = df["close"].pct_change().fillna(0.0)
        gross_rev = pos_rev * bar_ret
        n = len(df)
        # walk forward: at each bar compute trailing-21d DD and 21d return
        # using info up to t-1.
        eq = (1.0 + gross_rev).cumprod()
        win = 21
        # rolling max over the last 21 bars of equity
        roll_max = eq.rolling(win, min_periods=5).max()
        roll_dd = eq / roll_max - 1.0
        roll_ret = gross_rev.rolling(win, min_periods=5).sum()
        # use info known at open of bar t  => shift by 1
        dd_s = roll_dd.shift(1).fillna(0.0)
        ret_s = roll_ret.shift(1).fillna(0.0)
        size = np.ones(n)
        cur_size = 1.0
        for t in range(n):
            if cur_size == 1.0 and dd_s.iloc[t] < -0.03:
                cur_size = 0.5
            elif cur_size == 0.5 and ret_s.iloc[t] > 0:
                cur_size = 1.0
            size[t] = cur_size
        adapted = pos_rev * pd.Series(size, index=df.index)
        ret = position_to_returns(df, adapted, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S5: Two-sided reversion gated by 20d trend.
#   - Fade big UP days  only when 20d trend (close/close.shift(20)-1) < 0
#   - Fade big DOWN days only when 20d trend > 0
# ---------------------------------------------------------------------------
def sleeve5_two_sided(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        log_ret = np.log(df["close"] / df["close"].shift(1))
        thresh = REV_THRESH_BPS / 10_000.0
        trend20 = df["close"].pct_change(20)
        # both signals known at end of bar t-1 -> shift(1)
        big_up   = (log_ret > thresh).shift(1).fillna(False).astype(bool)
        big_dn   = (log_ret < -thresh).shift(1).fillna(False).astype(bool)
        td_s     = trend20.shift(1).fillna(0.0)
        pos = pd.Series(0.0, index=df.index)
        pos[(big_up) & (td_s < 0)] = -1.0   # fade up in downtrend
        pos[(big_dn) & (td_s > 0)] = +1.0   # fade dn in uptrend
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S6: Momentum-confirmed reversion exit.
# Standard D1REV. While in a 1-bar fade trade we cannot really exit (it's a
# 1-bar hold). So generalise to a 3-bar fade and cut early if both:
#     - intra-trade PnL is negative AND
#     - sign of 5d trend matches the *fade-against* direction.
# That is: we faded an up day (pos=-1); we'd cut if cum PnL < 0 and 5d trend
# > 0.  Symmetric on the other side.
# ---------------------------------------------------------------------------
def sleeve6_momentum_exit(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        log_ret = np.log(df["close"] / df["close"].shift(1))
        thresh = REV_THRESH_BPS / 10_000.0
        trend5 = df["close"].pct_change(5)
        bar_ret = df["close"].pct_change().fillna(0.0).values
        raw_dir = pd.Series(0.0, index=df.index)
        raw_dir[log_ret > thresh] = -1.0
        raw_dir[log_ret < -thresh] = +1.0
        trig = raw_dir.shift(1).fillna(0.0).values
        td5 = trend5.shift(1).fillna(0.0).values
        n = len(df)
        pos = np.zeros(n)
        cur = 0.0
        remaining = 0
        cum_pnl = 0.0
        max_hold = 3
        for t in range(n):
            # check new signal
            if trig[t] != 0.0:
                cur = float(trig[t])
                remaining = max_hold
                cum_pnl = 0.0
            if remaining > 0:
                pos[t] = cur
                cum_pnl += cur * bar_ret[t]
                remaining -= 1
                # momentum-confirmed early exit at end of bar t
                # (i.e. flatten starting bar t+1) — only after at least 1 bar held
                if remaining > 0 and cum_pnl < 0:
                    # cur=-1 (fading up) and td5>0 -> trend confirms against us
                    # cur=+1 (fading dn) and td5<0 -> trend confirms against us
                    if (cur < 0 and td5[t] > 0) or (cur > 0 and td5[t] < 0):
                        remaining = 0
                        cur = 0.0
            else:
                pos[t] = 0.0
                cur = 0.0
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# S7: Volatility-conditional reversion.
# Vanilla D1REV but only when 30d realised vol is in IS middle 60% (p20..p80).
# Quantiles set on IS data only.
# ---------------------------------------------------------------------------
def sleeve7_vol_conditional(data: dict[str, pd.DataFrame]) -> pd.Series:
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        pos_rev = vanilla_d1rev_signal(df)
        rv30 = df["log_ret"].rolling(30).std(ddof=1) * np.sqrt(252.0)
        # IS quantiles
        is_mask = df["timestamp"] < SPLIT
        is_vals = rv30[is_mask].dropna()
        p20 = float(is_vals.quantile(0.20))
        p80 = float(is_vals.quantile(0.80))
        gate = ((rv30 >= p20) & (rv30 <= p80)).shift(1).fillna(False).astype(float)
        adapted = pos_rev * gate
        ret = position_to_returns(df, adapted, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading D1 data for 6 indices ===")
    data = {s: load_d1(s) for s in INDICES_EQ}
    for s in INDICES_EQ:
        df = data[s]
        print(f"  {s}: {len(df)} bars  {df['timestamp'].iloc[0].date()}"
              f" .. {df['timestamp'].iloc[-1].date()}")

    builders = {
        "D1REV_BASE":        lambda: baseline_vanilla(data),
        "S1_self_adapting":  lambda: sleeve1_self_adapting(data),
        "S2_revrev":         lambda: sleeve2_revrev(data),
        "S3a_mp_3d":         lambda: sleeve3a_mp_3d(data),
        "S3b_mp_5d":         lambda: sleeve3b_mp_5d(data),
        "S4_dd_adaptive":    lambda: sleeve4_dd_adaptive(data),
        "S5_two_sided":      lambda: sleeve5_two_sided(data),
        "S6_momentum_exit":  lambda: sleeve6_momentum_exit(data),
        "S7_vol_cond":       lambda: sleeve7_vol_conditional(data),
    }
    # Eligible for combination (exclude baseline).
    ADAPTIVE_NAMES = [k for k in builders if k != "D1REV_BASE"]

    print("\n=== Building sub-sleeves ===")
    raw_streams: dict[str, pd.Series] = {}
    for name, fn in builders.items():
        s = fn().sort_index()
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
        raw_streams[name] = s
        print(f"  built {name:<20} n={len(s)}  first={s.index.min().date()}"
              f" last={s.index.max().date()}")

    print("\n=== Vol-scale + stats ===")
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

    # Survivors (excluding baseline)
    eligible = breakdown.loc[ADAPTIVE_NAMES]
    is_ok = eligible["IS_sharpe"] >= 0.5
    oos_ok = eligible["OOS_sharpe"] >= 0.0
    survivors = eligible[is_ok & oos_ok].index.tolist()
    breakdown["survivor"] = breakdown.index.isin(survivors)

    print("\n=== Per-sub-sleeve breakdown ===")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print(breakdown)

    # Combine survivors equal-weight on union calendar.
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
    base_stats = split_stats(scaled_streams["D1REV_BASE"], bpy=252.0)

    print("\n=== Survivors:", survivors)
    print("=== Combined sleeve stats ===")
    for k, v in combined_stats.items():
        print(f"  {k:>9} : sharpe={v['sharpe']:+.3f} ret={v['ann_return']:+.3%} "
              f"vol={v['ann_vol']:.3%} dd={v['max_dd']:+.3%} n={v['n']}")
    print("=== Vanilla D1REV_BASE (5%-vol-scaled) stats ===")
    for k, v in base_stats.items():
        print(f"  {k:>9} : sharpe={v['sharpe']:+.3f} ret={v['ann_return']:+.3%} "
              f"vol={v['ann_vol']:.3%} dd={v['max_dd']:+.3%} n={v['n']}")

    # Append COMBINED row.
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
    breakdown.to_csv(OUT_DIR / "adaptive_reversion_breakdown.csv",
                     float_format="%.4f")

    out = combined.rename("ret").to_frame()
    out.index = pd.DatetimeIndex(out.index, tz="UTC").rename("timestamp")
    out = out.reset_index()
    out.to_parquet(OUT_DIR / "adaptive_reversion_returns.parquet", index=False)

    print("\nSaved:")
    print(f"  {OUT_DIR / 'adaptive_reversion_returns.parquet'}")
    print(f"  {OUT_DIR / 'adaptive_reversion_breakdown.csv'}")


if __name__ == "__main__":
    main()
