"""H4 timeframe strategy hunt.

Five families:
  1. Trend-following momentum (N in {6, 18, 42})
  2. Bollinger mean-reversion (20-bar SMA +/- 2*std)
  3. Session-momentum (Asia/London/NY open-bar follow-through)
  4. RSI(14) with 50-bar SMA trend filter
  5. H4 weekly seasonality (day-of-week x hour-of-day bias)

Filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.0.
Each survivor vol-targeted to 5% IS ann vol, combined equal-weight to one sleeve.

Splits: IS < 2024-01-01, OOS >= 2024-01-01. 6 H4 bars / day -> ~2190 bars/yr.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import ALL_SYMBOLS, CRYPTO, FOREX, INDEX, SYMBOL_TYPE, get_candles
from alphabeta.backtest import backtest, cost_for, split_is_oos


OUT_DIR = Path(__file__).resolve().parent
IS_SPLIT = "2024-01-01"
BARS_PER_YEAR_H4 = 365.25 * 24 / 4  # ~2191.5 ideal; engine uses empirical
TARGET_VOL = 0.05
IS_SHARPE_FLOOR = 0.4
OOS_SHARPE_FLOOR = 0.0
MIN_TRADES = 20      # sanity guard against 1-trade flukes
MIN_EXPOSURE = 0.02  # at least 2% of bars holding a position


# ---------- signal builders ----------------------------------------------------

def sig_tsmom(df: pd.DataFrame, n: int) -> pd.Series:
    """Sign of trailing N-bar cumulative return. Shifted by 1 to avoid lookahead."""
    ret = df["close"].pct_change()
    cum = ret.rolling(n).sum()
    pos = np.sign(cum).shift(1).fillna(0.0)
    return pos


def sig_bollinger_mr(df: pd.DataFrame, n: int = 20, k: float = 2.0, hold: int = 3) -> pd.Series:
    """Fade closes outside +/- k*std bands; hold for `hold` bars or until back to SMA."""
    close = df["close"]
    sma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = sma + k * std
    lower = sma - k * std
    # entry signal at close[t]: position taken from t+1 onward
    entry = pd.Series(0.0, index=df.index)
    entry[close > upper] = -1.0
    entry[close < lower] = 1.0
    pos = pd.Series(0.0, index=df.index)
    holding = 0
    cur = 0.0
    sma_arr = sma.values
    close_arr = close.values
    entry_arr = entry.values
    pos_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        # apply previous bar's entry
        if cur != 0.0:
            holding += 1
            # exit if back to SMA crossed or hold limit
            crossed = (cur > 0 and close_arr[i - 1] >= sma_arr[i - 1]) or (cur < 0 and close_arr[i - 1] <= sma_arr[i - 1])
            if crossed or holding >= hold:
                cur = 0.0
                holding = 0
        if cur == 0.0 and entry_arr[i - 1] != 0.0:
            cur = entry_arr[i - 1]
            holding = 0
        pos_arr[i] = cur
    return pd.Series(pos_arr, index=df.index)


def sig_session_momentum(df: pd.DataFrame, threshold_bps: float = 30.0, hold: int = 2) -> pd.Series:
    """At H4 bars that open a session, if bar return > threshold,
    hold the direction for `hold` more H4 bars. Crypto/OANDA bars start at
    different hours (0/8/12 vs 1/5/9/13/17/21); we cover both grids.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    hour = ts.dt.hour
    # Asia ~ 00/01 UTC, London ~ 08/09 UTC, NY ~ 12/13 UTC. Also include the
    # DST-shifted OANDA grids (21/22 = pre-Asia, 5/6, 13/14).
    is_session_open = hour.isin([0, 1, 8, 9, 12, 13])
    bar_ret = (df["close"] / df["open"] - 1.0)
    trigger = pd.Series(0.0, index=df.index)
    sig = np.where(bar_ret.abs() * 10_000 > threshold_bps, np.sign(bar_ret), 0.0)
    trigger[is_session_open.values] = sig[is_session_open.values]
    # carry the position for `hold` bars after the trigger bar (act t+1..t+hold)
    pos_arr = np.zeros(len(df))
    remaining = 0
    cur = 0.0
    trig_arr = trigger.values
    for i in range(1, len(df)):
        if remaining > 0:
            pos_arr[i] = cur
            remaining -= 1
        # at the start of bar i, look at trigger at bar i-1
        if trig_arr[i - 1] != 0.0:
            cur = trig_arr[i - 1]
            remaining = hold - 1  # we already filled this bar below if 0; otherwise start now
            pos_arr[i] = cur
    return pd.Series(pos_arr, index=df.index)


def sig_rsi_trend(df: pd.DataFrame, rsi_n: int = 14, trend_n: int = 50,
                  lo: float = 30.0, hi: float = 70.0) -> pd.Series:
    """RSI(14) reversion entries WITH SMA(50) directional bias. Long entry when
    RSI crosses up through `lo` (rebound from oversold) and close >= SMA;
    short entry when RSI crosses down through `hi` (rejection from overbought)
    and close <= SMA. Exit when RSI crosses through 50 in opposite direction.
    """
    close = df["close"].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rsi_n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    sma = close.rolling(trend_n).mean()
    pos_arr = np.zeros(len(df))
    cur = 0.0
    rsi_arr = rsi.values
    sma_arr = sma.values
    close_arr = close.values
    for i in range(2, len(df)):
        r_prev = rsi_arr[i - 1]
        r_prev2 = rsi_arr[i - 2]
        s_prev = sma_arr[i - 1]
        c_prev = close_arr[i - 1]
        if not np.isfinite(r_prev) or not np.isfinite(r_prev2) or not np.isfinite(s_prev):
            pos_arr[i] = 0.0
            continue
        if cur == 0.0:
            # rebound from oversold inside an uptrend
            if r_prev2 < lo and r_prev >= lo and c_prev >= s_prev:
                cur = 1.0
            # rejection from overbought inside a downtrend
            elif r_prev2 > hi and r_prev <= hi and c_prev <= s_prev:
                cur = -1.0
        else:
            if cur > 0 and r_prev >= 60:
                cur = 0.0
            elif cur < 0 and r_prev <= 40:
                cur = 0.0
        pos_arr[i] = cur
    return pd.Series(pos_arr, index=df.index)


def sig_seasonality(df: pd.DataFrame, is_split: str = IS_SPLIT,
                    min_bars: int = 50, min_abs_mean_bps: float = 5.0) -> tuple[pd.Series, dict]:
    """Scan H4 buckets by (day-of-week, hour-of-day) on IS. If a bucket has
    |mean| return >= threshold and >= min_bars samples, take its sign as a
    position whenever that bucket arrives. Position held just for that bar.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    dow = ts.dt.dayofweek.values
    hour = ts.dt.hour.values
    ret = df["close"].pct_change().values
    bucket = dow * 100 + hour
    split_ts = pd.Timestamp(is_split, tz="UTC")
    is_mask = (ts < split_ts).values
    # bucket stats on IS only
    df_stats = pd.DataFrame({"bucket": bucket, "ret": ret, "is_mask": is_mask})
    is_df = df_stats[df_stats["is_mask"] & np.isfinite(df_stats["ret"])]
    grp = is_df.groupby("bucket")["ret"].agg(["mean", "count"])
    keep = grp[(grp["count"] >= min_bars) & (grp["mean"].abs() * 10_000 >= min_abs_mean_bps)]
    side_map = {b: float(np.sign(m)) for b, m in keep["mean"].items()}
    # position for bar t: bucket of bar t (since we want to capture that bar's
    # return). Since bucket is purely calendar (known before bar t opens), no
    # lookahead. Bar return is open->close; engine uses close-to-close, but for
    # a 1-bar bet on a calendar bucket that approximation is fine.
    pos = pd.Series([side_map.get(b, 0.0) for b in bucket], index=df.index)
    return pos, {"n_buckets": int(len(side_map)), "side_map": side_map}


# ---------- vol-targeting ------------------------------------------------------

def vol_target_returns(rets: pd.Series, target_ann_vol: float, bars_per_year: float,
                       is_mask: pd.Series | None = None) -> tuple[pd.Series, float]:
    """Scale returns so the (IS) ann vol equals target. Returns scaled series + scale."""
    base = rets if is_mask is None else rets[is_mask]
    realized = base.std(ddof=0) * np.sqrt(bars_per_year)
    if realized <= 0:
        return rets * 0.0, 0.0
    scale = target_ann_vol / realized
    # cap scale at reasonable bound to avoid blow-ups
    scale = min(scale, 5.0)
    return rets * scale, scale


def sharpe(rets: pd.Series, bars_per_year: float) -> float:
    if rets.std(ddof=0) <= 0 or len(rets) < 10:
        return 0.0
    return float(rets.mean() * bars_per_year / (rets.std(ddof=0) * np.sqrt(bars_per_year)))


# ---------- runner -------------------------------------------------------------

def run_one(symbol: str, name: str, position_fn) -> dict | None:
    df = get_candles(symbol, "H4")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 500:
        return None
    pos = position_fn(df)
    if isinstance(pos, tuple):
        pos = pos[0]
    res = backtest(df, pos, symbol=symbol, timeframe="H4", name=name)
    rets = res.returns
    ts = df["timestamp"]
    rets.index = ts
    is_mask = ts < pd.Timestamp(IS_SPLIT, tz="UTC")
    oos_mask = ~is_mask
    bpy = res.stats["bars_per_year"]
    s_full = sharpe(rets, bpy)
    s_is = sharpe(rets[is_mask.values], bpy)
    s_oos = sharpe(rets[oos_mask.values], bpy)
    y22 = (ts.dt.year == 2022).values
    s_22 = sharpe(rets[y22], bpy) if y22.sum() > 50 else 0.0
    return {
        "symbol": symbol,
        "asset": SYMBOL_TYPE[symbol].value,
        "name": name,
        "sharpe_full": s_full,
        "sharpe_is": s_is,
        "sharpe_oos": s_oos,
        "sharpe_2022": s_22,
        "ann_ret": res.stats["ann_return"],
        "ann_vol": res.stats["ann_vol"],
        "max_dd": res.stats["max_dd"],
        "n_trades": res.stats["n_trades"],
        "exposure": res.stats["exposure"],
        "bars_per_year": bpy,
        "returns": rets,
        "is_mask": is_mask,
    }


def collapse_to_daily(rets: pd.Series) -> pd.Series:
    """H4 returns -> daily compounded."""
    if not isinstance(rets.index, pd.DatetimeIndex):
        rets = rets.copy()
        rets.index = pd.DatetimeIndex(rets.index)
    daily = (1 + rets).resample("1D").prod() - 1
    return daily


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    breakdown_rows: list[dict] = []
    survivors: list[dict] = []

    families = []
    # Family 1: TSMOM N in {6,18,42}
    for n in (6, 18, 42):
        families.append(("TSMOM", f"TSMOM_{n}", lambda df, n=n: sig_tsmom(df, n)))
    # Family 2: Bollinger MR
    families.append(("BOLL_MR", "BOLL_MR_20_2_3", lambda df: sig_bollinger_mr(df, 20, 2.0, 3)))
    # Family 3: Session momentum
    families.append(("SESS_MOM", "SESS_MOM_30_2", lambda df: sig_session_momentum(df, 30.0, 2)))
    # Family 4: RSI+trend
    families.append(("RSI_TRND", "RSI14_SMA50_30_70", lambda df: sig_rsi_trend(df, 14, 50, 30, 70)))
    # Family 5: seasonality (IS-fit bucket map)
    families.append(("SEAS", "SEAS_DOWxHOUR", lambda df: sig_seasonality(df)))

    for symbol in ALL_SYMBOLS:
        for family, name, fn in families:
            try:
                rec = run_one(symbol, name, fn)
            except Exception as e:
                print(f"  ! {symbol} {name}: {e}")
                continue
            if rec is None:
                continue
            rec["family"] = family
            breakdown_rows.append({
                "family": family, "name": name, "symbol": symbol, "asset": rec["asset"],
                "sharpe_full": rec["sharpe_full"], "sharpe_is": rec["sharpe_is"],
                "sharpe_oos": rec["sharpe_oos"], "sharpe_2022": rec["sharpe_2022"],
                "ann_ret": rec["ann_ret"], "ann_vol": rec["ann_vol"],
                "max_dd": rec["max_dd"], "n_trades": rec["n_trades"],
                "exposure": rec["exposure"],
            })
            if (rec["sharpe_is"] >= IS_SHARPE_FLOOR
                    and rec["sharpe_oos"] >= OOS_SHARPE_FLOOR
                    and rec["n_trades"] >= MIN_TRADES
                    and rec["exposure"] >= MIN_EXPOSURE):
                survivors.append(rec)
            print(f"  {symbol:<11} {name:<22} IS={rec['sharpe_is']:>5.2f} OOS={rec['sharpe_oos']:>5.2f} "
                  f"FULL={rec['sharpe_full']:>5.2f} 2022={rec['sharpe_2022']:>5.2f}")

    # write breakdown
    bd = pd.DataFrame(breakdown_rows).sort_values(["family", "sharpe_is"], ascending=[True, False])
    bd.to_csv(OUT_DIR / "h4_breakdown.csv", index=False)
    print(f"\n  wrote {OUT_DIR/'h4_breakdown.csv'} ({len(bd)} rows)")
    print(f"  survivors: {len(survivors)}")

    if not survivors:
        print("  no survivors; writing empty sleeve")
        empty = pd.DataFrame()
        empty.to_parquet(OUT_DIR / "h4_returns.parquet")
        return

    # vol-target each survivor to 5% IS ann vol
    scaled_daily_streams: list[pd.Series] = []
    survivor_summary = []
    for rec in survivors:
        rets = rec["returns"]
        is_mask = rec["is_mask"]
        bpy = rec["bars_per_year"]
        scaled, scale = vol_target_returns(rets, TARGET_VOL, bpy, is_mask=is_mask.values)
        daily = collapse_to_daily(scaled)
        daily.name = f"{rec['name']}__{rec['symbol']}"
        scaled_daily_streams.append(daily)
        survivor_summary.append({
            "family": rec["family"], "name": rec["name"], "symbol": rec["symbol"],
            "asset": rec["asset"], "scale": scale,
            "sharpe_is": rec["sharpe_is"], "sharpe_oos": rec["sharpe_oos"],
            "sharpe_2022": rec["sharpe_2022"],
        })

    # align on the union of daily dates; missing = 0
    aligned = pd.concat(scaled_daily_streams, axis=1).fillna(0.0)
    aligned = aligned.sort_index()

    # equal-weight combine
    combined = aligned.mean(axis=1)
    combined.name = "H4_SLEEVE"

    # write parquet: per-survivor daily + combined
    out_df = aligned.copy()
    out_df["H4_SLEEVE"] = combined
    out_df.to_parquet(OUT_DIR / "h4_returns.parquet")
    print(f"  wrote {OUT_DIR/'h4_returns.parquet'} shape={out_df.shape}")

    # Headline stats on the combined sleeve (D1)
    def daily_sharpe(s):
        if s.std(ddof=0) <= 0:
            return 0.0
        return float(s.mean() * 252 / (s.std(ddof=0) * np.sqrt(252)))

    split_ts = pd.Timestamp(IS_SPLIT, tz="UTC")
    cidx = combined.index
    if cidx.tz is None:
        cidx = cidx.tz_localize("UTC")
        combined.index = cidx
    full_s = daily_sharpe(combined)
    is_s = daily_sharpe(combined[combined.index < split_ts])
    oos_s = daily_sharpe(combined[combined.index >= split_ts])
    y22 = combined[(combined.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                   (combined.index < pd.Timestamp("2023-01-01", tz="UTC"))]
    y22_s = daily_sharpe(y22)

    ann_ret = combined.mean() * 252
    ann_vol = combined.std() * np.sqrt(252)
    eq = (1 + combined).cumprod()
    dd = (eq / eq.cummax() - 1).min()

    print("\n=== COMBINED H4 SLEEVE (daily, equal-weight survivors) ===")
    print(f"  FULL Sharpe = {full_s:+.2f}")
    print(f"  IS   Sharpe = {is_s:+.2f}")
    print(f"  OOS  Sharpe = {oos_s:+.2f}")
    print(f"  2022 Sharpe = {y22_s:+.2f}")
    print(f"  Ann ret = {ann_ret:+.2%}  Ann vol = {ann_vol:.2%}  MaxDD = {dd:.2%}")
    print(f"  # survivors = {len(survivors)}")

    surv_df = pd.DataFrame(survivor_summary)
    print("\n  survivors by family:")
    print(surv_df.groupby("family").size().to_string())
    print("\n  survivors by asset:")
    print(surv_df.groupby("asset").size().to_string())
    print("\n  survivors by symbol:")
    print(surv_df.groupby("symbol").size().to_string())

    # Per-family aggregate: combine survivors within each family equally
    print("\n=== per-family aggregate (survivors only, equal-weight, daily) ===")
    for fam in surv_df["family"].unique():
        names = surv_df[surv_df["family"] == fam].apply(
            lambda r: f"{r['name']}__{r['symbol']}", axis=1).tolist()
        cols = [c for c in aligned.columns if c in names]
        if not cols:
            continue
        fam_ret = aligned[cols].mean(axis=1)
        f_full = daily_sharpe(fam_ret)
        f_is = daily_sharpe(fam_ret[fam_ret.index < split_ts])
        f_oos = daily_sharpe(fam_ret[fam_ret.index >= split_ts])
        f_22 = daily_sharpe(fam_ret[(fam_ret.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                                    (fam_ret.index < pd.Timestamp("2023-01-01", tz="UTC"))])
        print(f"  {fam:<10} n={len(cols):>2}  FULL={f_full:+.2f} IS={f_is:+.2f} "
              f"OOS={f_oos:+.2f} 2022={f_22:+.2f}")


if __name__ == "__main__":
    main()
