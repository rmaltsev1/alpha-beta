"""V3 — only survivors of OOS testing.

V2 lessons:
  * MON23 indices effect doesn't survive 2024+. Drop.
  * EVE_EUR / EVE_GBP / EVE_JPY block sleeves: DST gaps fragment the position
    series — 2× the round trips, 2× the cost — and the signals were weaker
    than XAU anyway. Drop.
  * JPY_WED23 collapsed OOS. Drop.

Surviving signals (all positive IS *and* OOS):
  - EVE_XAU      OOS Sharpe +2.21 (gold has the strongest 21–23 UTC drift)
  - WED_BTC/ETH/SOL  OOS +0.88..+1.11
  - D1REV_NAS    OOS +0.75
  - D1REV_UK     OOS +0.40
  - D1REV_SPX    OOS +0.39  (kept; small but positive)

Target vol lowered to 5%/sleeve to control turnover-cost drag.
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
TARGET_VOL = 0.05


def _bars_per_year(df):
    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
    return len(df) / span * 365.25 if span > 0 else 252.0


def scale_factor(df_is, pos_is, target_vol):
    ret = np.log(df_is["close"] / df_is["close"].shift(1)).fillna(0)
    raw = pos_is.values * ret.values
    ann_vol = float(np.std(raw, ddof=0) * np.sqrt(_bars_per_year(df_is)))
    return (target_vol / ann_vol) if ann_vol > 1e-9 else 0.0


def run_sleeve(name, symbol, timeframe, build_position, target_vol=TARGET_VOL):
    df = get_candles(symbol, timeframe)
    is_df, oos_df = split_is_oos(df, split=SPLIT_DATE)
    is_pos  = pd.Series(build_position(is_df).values,  index=is_df.index)
    oos_pos = pd.Series(build_position(oos_df).values, index=oos_df.index) if len(oos_df) else pd.Series(dtype=float)
    full_pos = pd.Series(build_position(df).values, index=df.index)
    scale = scale_factor(is_df, is_pos, target_vol)

    rows = []
    full_ret_ts = None
    for label, frame, pos in [
        ("FULL", df,     full_pos * scale),
        ("IS",   is_df,  is_pos   * scale),
        ("OOS",  oos_df, oos_pos  * scale),
    ]:
        if len(frame) < 30:
            continue
        res = backtest(frame, pos, symbol=symbol, timeframe=timeframe, name=name)
        s = res.stats
        rows.append({"name": name, "symbol": symbol, "tf": timeframe, "period": label,
                     "scale": scale, "exposure": s["exposure"],
                     "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                     "sharpe": s["sharpe"], "max_dd": s["max_dd"],
                     "n_trades": s["n_trades"], "hit_rate": s["hit_rate"]})
        if label == "FULL":
            idx = pd.to_datetime(frame["timestamp"].values, utc=True)
            full_ret_ts = pd.Series(res.returns.values, index=idx)
    return rows, full_ret_ts


def main():
    # Use only sleeves that survived v2 OOS tests.
    sleeves = [
        ("EVE_XAU",   "XAU_USD",    "H1",  S2.evening_long),                                  # XAU 21-23
        ("WED_BTC",   "BTCUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_ETH",   "ETHUSDT",    "D1",  S2.crypto_wed_long),
        ("WED_SOL",   "SOLUSDT",    "D1",  S2.crypto_wed_long),
        ("D1REV_NAS", "NAS100_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_UK",  "UK100_GBP",  "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
        ("D1REV_SPX", "SPX500_USD", "D1",  lambda df: S2.d1_reversion(df, threshold_bps=50)),
    ]

    all_rows = []
    streams = {}
    for name, sym, tf, fn in sleeves:
        rows, ts = run_sleeve(name, sym, tf, fn)
        all_rows.extend(rows or [])
        if ts is not None:
            streams[name] = ts

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT / "results_v3.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_rows", None)
    cols = ["name","symbol","tf","period","scale","exposure",
            "ann_return","ann_vol","sharpe","max_dd","n_trades","hit_rate"]
    print(out[cols].to_string(index=False, formatters={
        "scale": lambda x: f"{x:>6.2f}", "exposure": lambda x: f"{x:.1%}",
        "ann_return": lambda x: f"{x:+.1%}", "ann_vol": lambda x: f"{x:.1%}",
        "sharpe": lambda x: f"{x:+.2f}", "max_dd": lambda x: f"{x:+.1%}",
        "hit_rate": lambda x: f"{x:.1%}",
    }))

    # Portfolio (equal weight + risk-parity weight)
    df_streams = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    ew = df_streams.mean(axis=1)
    # Risk-parity weights: inverse of IS vol per sleeve, normalized.
    split = pd.Timestamp(SPLIT_DATE, tz="UTC")
    is_streams = df_streams.loc[df_streams.index < split]
    bpy_per = {c: 365.25 * len(is_streams[c].loc[is_streams[c] != 0]) /
               max((is_streams.index[-1] - is_streams.index[0]).total_seconds() / 86400, 1)
               for c in is_streams.columns}
    is_vol = is_streams.std(ddof=0) * np.sqrt(_bars_per_year(get_candles("BTCUSDT", "D1")))
    inv_vol = 1.0 / is_vol.replace(0, np.nan)
    rp_w = (inv_vol / inv_vol.sum()).fillna(0.0)
    rp = (df_streams * rp_w.values).sum(axis=1)

    pd.DataFrame({
        "timestamp": df_streams.index, "ew_ret": ew.values, "rp_ret": rp.values,
        "ew_equity": (1 + ew).cumprod().values, "rp_equity": (1 + rp).cumprod().values,
    }).to_parquet(OUT / "portfolio_v3.parquet", index=False)

    print(f"\nrisk-parity weights:")
    for k, v in rp_w.items():
        print(f"  {k:<12} {v:6.1%}")

    def stats(name, r):
        is_mask = r.index < split
        for label, mask in [("FULL", pd.Series(True, index=r.index)), ("IS", is_mask), ("OOS", ~is_mask)]:
            s = r[mask]
            if len(s) < 2:
                continue
            span = (s.index[-1] - s.index[0]).total_seconds() / 86400
            bpy = len(s) / span * 365.25
            ar = float(s.mean()) * bpy
            av = float(s.std(ddof=0)) * np.sqrt(bpy)
            sh = ar / av if av > 0 else 0
            eq = (1 + s).cumprod()
            dd = eq / eq.cummax() - 1
            print(f"  {name:<22} {label:<4}  ann_ret={ar:+6.1%}  vol={av:5.1%}  Sharpe={sh:+5.2f}  MaxDD={dd.min():+6.1%}")

    print()
    stats("PORTFOLIO equal-wt", ew)
    print()
    stats("PORTFOLIO risk-par", rp)

    # Pairwise correlations
    print("\nWeekly sleeve correlations (FULL):")
    weekly = df_streams.resample("1W").sum()
    print(weekly.corr().round(2).to_string())


if __name__ == "__main__":
    main()
