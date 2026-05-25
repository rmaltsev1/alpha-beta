"""V2 backtest run — concentrated, vol-targeted sleeves + portfolio combine.

Vol scaling: each sleeve's scale factor is computed on IS data only (pre-2024)
so OOS performance is honest. Same scale applied to IS, OOS, and FULL slices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alphabeta import get_candles
from alphabeta.backtest import backtest, split_is_oos

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies_v2 as S2

OUT = Path(__file__).resolve().parent
SPLIT_DATE = "2024-01-01"
TARGET_VOL = 0.10  # 10% annualized vol per sleeve


def _bars_per_year(df: pd.DataFrame) -> float:
    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    return len(df) / span * 365.25 if span > 0 else 252.0


def scale_factor(df_is: pd.DataFrame, pos_is: pd.Series, target_vol: float) -> float:
    """Multiplier that brings IS strategy realized vol to target_vol."""
    ret = np.log(df_is["close"] / df_is["close"].shift(1)).fillna(0)
    raw = pos_is.values * ret.values
    bpy = _bars_per_year(df_is)
    ann_vol = float(np.std(raw, ddof=0) * np.sqrt(bpy))
    return (target_vol / ann_vol) if ann_vol > 1e-9 else 0.0


def run_sleeve(name, symbol, timeframe, build_position):
    df = get_candles(symbol, timeframe)
    if len(df) < 200:
        return None, None

    is_df, oos_df = split_is_oos(df, split=SPLIT_DATE)
    # 1) build raw position on each slice independently (calendar-only signals
    #    so the IS pos == positions at IS-indices of full pos — same thing)
    is_pos  = pd.Series(build_position(is_df).values,  index=is_df.index)  if len(is_df) else pd.Series(dtype=float)
    oos_pos = pd.Series(build_position(oos_df).values, index=oos_df.index) if len(oos_df) else pd.Series(dtype=float)
    full_pos = pd.Series(build_position(df).values, index=df.index)

    # 2) compute scale on IS only
    scale = scale_factor(is_df, is_pos, TARGET_VOL) if len(is_df) else 0.0

    rows = []
    full_returns_ts = None
    for label, frame, pos_slice in [
        ("FULL", df,     full_pos * scale),
        ("IS",   is_df,  is_pos   * scale),
        ("OOS",  oos_df, oos_pos  * scale),
    ]:
        if len(frame) < 30:
            continue
        res = backtest(frame, pos_slice, symbol=symbol, timeframe=timeframe, name=name)
        s = res.stats
        rows.append({
            "name": name, "symbol": symbol, "tf": timeframe, "period": label,
            "scale": scale, "exposure": s["exposure"],
            "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
            "sharpe": s["sharpe"], "max_dd": s["max_dd"],
            "n_trades": s["n_trades"], "hit_rate": s["hit_rate"],
        })
        if label == "FULL":
            idx = pd.to_datetime(frame["timestamp"].values, utc=True)
            full_returns_ts = pd.Series(res.returns.values, index=idx)
    return rows, full_returns_ts


def main():
    sleeves = [
        ("MON23_SPX",     "SPX500_USD", "H1",  S2.mon_23_long),
        ("MON23_NAS",     "NAS100_USD", "H1",  S2.mon_23_long),
        ("MON23_US30",    "US30_USD",   "H1",  S2.mon_23_long),
        ("EVE_EUR",       "EUR_USD",    "H1",  S2.evening_long),
        ("EVE_GBP",       "GBP_USD",    "H1",  S2.evening_long),
        ("EVE_XAU",       "XAU_USD",    "H1",  S2.evening_long),
        ("EVE_JPY_SHORT", "USD_JPY",    "H1",  S2.evening_short),
        ("D1REV_NAS",     "NAS100_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_UK",      "UK100_GBP",  "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_SPX",     "SPX500_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("WED_BTC",       "BTCUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_ETH",       "ETHUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_SOL",       "SOLUSDT",    "D1",  S2.crypto_wed_long),
        ("JPY_WED23",     "USD_JPY",    "H1",  S2.jpy_wed_23_short),
    ]

    all_rows = []
    streams: dict[str, pd.Series] = {}
    for name, sym, tf, fn in sleeves:
        rows, ts_ret = run_sleeve(name, sym, tf, fn)
        if rows:
            all_rows.extend(rows)
        if ts_ret is not None:
            streams[name] = ts_ret

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT / "results_v2.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    cols = ["name", "symbol", "tf", "period", "scale", "exposure",
            "ann_return", "ann_vol", "sharpe", "max_dd", "n_trades", "hit_rate"]
    print(out[cols].to_string(index=False,
        formatters={
            "scale":      lambda x: f"{x:>6.2f}",
            "exposure":   lambda x: f"{x:.1%}",
            "ann_return": lambda x: f"{x:+.1%}",
            "ann_vol":    lambda x: f"{x:.1%}",
            "sharpe":     lambda x: f"{x:+.2f}",
            "max_dd":     lambda x: f"{x:+.1%}",
            "hit_rate":   lambda x: f"{x:.1%}",
        }))

    # --- Portfolio: equal weight across sleeves ---
    df_streams = pd.concat(streams, axis=1).fillna(0.0)
    portfolio = df_streams.mean(axis=1)
    eq = (1 + portfolio).cumprod()

    portfolio_df = pd.DataFrame({"timestamp": df_streams.index, "ret": portfolio.values, "equity": eq.values})
    portfolio_df.to_parquet(OUT / "portfolio_v2.parquet", index=False)

    split = pd.Timestamp(SPLIT_DATE, tz="UTC")
    print()
    for label, mask in [
        ("FULL", pd.Series(True, index=portfolio.index)),
        ("IS",   portfolio.index < split),
        ("OOS",  portfolio.index >= split),
    ]:
        r = portfolio[mask]
        span = (r.index[-1] - r.index[0]).total_seconds() / 86400
        bpy = len(r) / span * 365.25 if span > 0 else 252.0
        ann_ret = float(r.mean()) * bpy
        ann_vol = float(r.std(ddof=0)) * np.sqrt(bpy)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        eq_l = (1 + r).cumprod()
        dd = eq_l / eq_l.cummax() - 1
        print(f"PORTFOLIO {label:<4}  ann_ret={ann_ret:+6.1%}  vol={ann_vol:5.1%}  "
              f"Sharpe={sharpe:+5.2f}  MaxDD={dd.min():+6.1%}  bars={len(r)}")

    print("\nSleeve return correlations (weekly resampled, FULL):")
    weekly = df_streams.resample("1W").sum()
    corr = weekly.corr()
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
