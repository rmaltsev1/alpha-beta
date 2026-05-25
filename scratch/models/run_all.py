"""Run every candidate strategy IS (≤2024-01-01) and OOS (≥2024-01-01),
save per-strategy stats + equity curves, print a comparison table.

Run from repo root with:
    PYTHONPATH=. python scratch/models/run_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alphabeta import get_candles
from alphabeta.backtest import backtest, split_is_oos

# Local import — relies on cwd being repo root or PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies as S


OUT = Path(__file__).resolve().parent
SPLIT_DATE = "2024-01-01"


def run_one(name, symbol, timeframe, build_position, *, costs_override=None):
    df = get_candles(symbol, timeframe)
    if len(df) < 200:
        return None

    is_df, oos_df = split_is_oos(df, split=SPLIT_DATE)
    rows = []
    for label, frame in [("FULL", df), ("IS", is_df), ("OOS", oos_df)]:
        if len(frame) < 30:
            continue
        pos = build_position(frame)
        if not isinstance(pos, pd.Series):
            pos = pd.Series(pos, index=frame.index)
        res = backtest(
            frame, pos,
            symbol=symbol, timeframe=timeframe, name=name,
            cost_per_side=costs_override,
        )
        s = res.stats
        rows.append({
            "name": name, "symbol": symbol, "tf": timeframe, "period": label,
            "first": s["first"], "last": s["last"], "n_bars": s["n_bars"],
            "exposure": s["exposure"], "n_trades": s["n_trades"],
            "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
            "sharpe": s["sharpe"], "max_dd": s["max_dd"],
            "hit_rate": s["hit_rate"], "profit_factor": s["profit_factor"],
        })
        # Save equity curve for FULL only.
        if label == "FULL":
            eq = pd.DataFrame({
                "timestamp": frame["timestamp"].values,
                "equity": res.equity.values,
                "position": res.position.values,
                "ret": res.returns.values,
            })
            tag = f"{name}_{symbol}_{timeframe}".replace(" ", "_").replace("/", "-")
            eq.to_parquet(OUT / f"equity_{tag}.parquet", index=False)
    return rows


def main():
    rows = []

    # ---- Strategy 1: XAU 23:00 UTC long (1-hour hold) ----
    rows += run_one("xau_23h_long", "XAU_USD", "H1",
                    lambda df: S.hour_of_day(df, {23: +1})) or []

    # ---- Strategy 1b: same idea but on every metal-ish symbol ----
    for sym in ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY"]:
        rows += run_one(f"hod_23_long", sym, "H1",
                        lambda df: S.hour_of_day(df, {23: +1})) or []

    # ---- Strategy 2: EUR_USD London pattern (+08 / -11) ----
    rows += run_one("eur_london_08_11", "EUR_USD", "H1",
                    lambda df: S.eur_london_pattern(df)) or []
    # Same template applied across FX (cheap to look)
    for sym in ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]:
        rows += run_one(f"london_08_11", sym, "H1",
                        lambda df: S.eur_london_pattern(df)) or []

    # ---- Strategy 3: Monday NY session long on US indices ----
    for sym in ["SPX500_USD", "NAS100_USD", "US30_USD"]:
        rows += run_one("monday_ny_long", sym, "H1",
                        lambda df: S.monday_session(df, ny_only=True)) or []

    # ---- Strategy 4: NY session long, every weekday, US indices ----
    for sym in ["SPX500_USD", "NAS100_USD", "US30_USD"]:
        rows += run_one("ny_session_long", sym, "H1",
                        lambda df: S.ny_session_long(df)) or []

    # ---- Strategy 5: D1 mean-reversion on equities + large-cap crypto ----
    for sym in ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR", "UK100_GBP",
                "BTCUSDT", "ETHUSDT"]:
        rows += run_one("d1_reversion", sym, "D1",
                        lambda df: S.daily_reversion(df, threshold_bps=0)) or []

    # ---- Strategy 5b: D1 mean-reversion only on large moves (50 bps) ----
    for sym in ["SPX500_USD", "NAS100_USD", "US30_USD", "BTCUSDT", "ETHUSDT"]:
        rows += run_one("d1_reversion_50bps", sym, "D1",
                        lambda df: S.daily_reversion(df, threshold_bps=50)) or []

    # ---- Strategy 6: Crypto midweek (Wed) D1 long ----
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        rows += run_one("crypto_wed_long", sym, "D1",
                        lambda df: S.crypto_midweek(df)) or []

    # ---- Strategy 7: JP225 Asia-open continuation ----
    rows += run_one("jp_asia_open_cont", "JP225_USD", "H1",
                    lambda df: S.asia_open_breakout(df)) or []

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "results.csv", index=False)

    # Print nicely
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)
    cols = ["name", "symbol", "tf", "period", "n_bars", "exposure",
            "n_trades", "ann_return", "ann_vol", "sharpe", "max_dd", "hit_rate"]
    print(out[cols].to_string(index=False,
        formatters={
            "exposure":    lambda x: f"{x:.1%}",
            "ann_return":  lambda x: f"{x:+.1%}",
            "ann_vol":     lambda x: f"{x:.1%}",
            "sharpe":      lambda x: f"{x:+.2f}",
            "max_dd":      lambda x: f"{x:+.1%}",
            "hit_rate":    lambda x: f"{x:.1%}",
        }))


if __name__ == "__main__":
    main()
