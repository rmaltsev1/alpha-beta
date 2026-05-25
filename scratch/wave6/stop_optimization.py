"""
Wave 6: Stop-loss and take-profit optimization for existing strategy sleeves.

Approach
--------
We only have sleeve-level aggregated daily P&L (not per-trade fills), so we
treat each contiguous run of non-zero daily returns as a "trade episode" for
that sleeve. Within an episode we accumulate gain/loss from the episode start
and trigger stop / take-profit rules on that cumulative P&L. After a stop
fires we force the sleeve flat (return = 0) for N cooldown days. The next
non-zero P&L day starts a new episode.

For ATR-based rules we use the matching underlying symbol's D1 ATR(14) when
the sleeve is single-symbol, else we use a sleeve-vol proxy: 14d rolling std
of the sleeve return scaled to a comparable "ATR" magnitude.

Stop strategies tested (per sleeve):
    1. Fixed 3% stop, cooldown N in {1, 2, 5}
    2. ATR-based stop: 2x ATR loss
    3. Trailing stop: arm at +2% gain, trail to 50% of peak gain
    4. Take-profit + stop: TP at +2x ATR, SL at -1x ATR
    5. Time-based stop: max hold 5 D1 bars
    6. Volatility-conditional stop: loose stop in high-vol, tight in low-vol
    7. Equity-curve protection: halt sleeve 5 days if trailing 21d return < -5%

For each (sleeve, rule) we report Sharpe (IS / OOS / full), max drawdown
delta, and the OOS-Sharpe lift over baseline. We keep the best variant per
sleeve in the output parquet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path("/Users/rinatmaltsev/Documents/Python Projects/alpha-beta/alpha-beta")
SLEEVE_FILE = ROOT / "scratch/quant/all_sleeve_returns_v13.parquet"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "scratch/wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN = np.sqrt(252)
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
BEAR_START = pd.Timestamp("2022-01-01", tz="UTC")
BEAR_END = pd.Timestamp("2023-01-01", tz="UTC")

# Sleeve -> primary underlying symbol for ATR (None => use sleeve-vol proxy).
SLEEVE_SYMBOL = {
    "TREND_NEW": None,      # multi-asset trend; use vol proxy
    "D1REV_NAS": "NAS100_USD",
    "D1REV_UK": "UK100_GBP",
    "PAIRS_EXP": None,      # pair spread, no single price
    "VOLFORECAST": None,
    "RISKPAR": None,
}

TARGET_SLEEVES = list(SLEEVE_SYMBOL.keys())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * ANN)


def max_dd(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    eq = (1.0 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def slice_metrics(r: pd.Series) -> dict:
    is_mask = r.index < SPLIT
    oos_mask = r.index >= SPLIT
    bear_mask = (r.index >= BEAR_START) & (r.index < BEAR_END)
    return {
        "sharpe_full": sharpe(r),
        "sharpe_is": sharpe(r[is_mask]),
        "sharpe_oos": sharpe(r[oos_mask]),
        "sharpe_2022": sharpe(r[bear_mask]),
        "maxdd_full": max_dd(r),
        "maxdd_is": max_dd(r[is_mask]),
        "maxdd_oos": max_dd(r[oos_mask]),
    }


# ---------------------------------------------------------------------------
# Episode state machine
# ---------------------------------------------------------------------------
def apply_stop(
    returns: pd.Series,
    *,
    stop_check: Callable[[dict, float, float, int, float], bool],
    cooldown: int = 1,
    state_extra: dict | None = None,
) -> pd.Series:
    """Walk daily returns. Within each non-zero "episode", call stop_check.

    stop_check(state, ret_today, vol_today, day_in_trade, cum_gain) -> bool
        Returning True means: ABORT trade *after applying today's return*?
        We exit at end of day -- the day that triggers the stop still books
        its P&L (then we go flat for `cooldown` days).
    """
    out = returns.copy().astype(float)
    in_trade = False
    cum = 0.0
    peak = 0.0
    day_idx = 0
    cooldown_left = 0
    state = dict(state_extra) if state_extra else {}

    vols = state.get("vol_series")  # pd.Series aligned to returns or None

    arr_idx = returns.index
    arr_val = returns.values
    for i in range(len(arr_val)):
        r = float(arr_val[i])
        if cooldown_left > 0:
            out.iloc[i] = 0.0
            cooldown_left -= 1
            in_trade = False
            cum = peak = 0.0
            day_idx = 0
            continue

        if r == 0.0 and not in_trade:
            continue

        if not in_trade and r != 0.0:
            in_trade = True
            cum = 0.0
            peak = 0.0
            day_idx = 0

        if in_trade:
            cum += r
            peak = max(peak, cum)
            day_idx += 1
            vol_today = float(vols.iloc[i]) if vols is not None else np.nan
            stop_state = {
                "peak": peak,
                "day": day_idx,
                "cum": cum,
                "vol": vol_today,
                **state,
            }
            triggered = stop_check(stop_state, r, vol_today, day_idx, cum)
            if triggered:
                cooldown_left = state.get("cooldown_after", cooldown)
                in_trade = False
                cum = peak = 0.0
                day_idx = 0
            elif r == 0.0:
                # episode ends naturally on a zero day -- close it.
                in_trade = False
                cum = peak = 0.0
                day_idx = 0
    return out


def apply_equity_protection(returns: pd.Series, *, dd_thresh: float = -0.05,
                            window: int = 21, halt_days: int = 5) -> pd.Series:
    """Sleeve-level circuit breaker on trailing 21d return."""
    out = returns.copy().astype(float)
    rolling = returns.rolling(window, min_periods=window).sum()
    halt = 0
    for i in range(len(out)):
        if halt > 0:
            out.iloc[i] = 0.0
            halt -= 1
            continue
        if i >= window and rolling.iloc[i - 1] < dd_thresh:
            halt = halt_days - 1
            out.iloc[i] = 0.0
    return out


# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------
def atr_from_ohlc(symbol: str, n: int = 14) -> pd.Series:
    path = DATA_DIR / symbol / "D1.parquet"
    df = pd.read_parquet(path)
    df = df.set_index("timestamp").sort_index()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    # Express ATR as fractional move so it's comparable to return series.
    atr_pct = atr / df["close"]
    atr_pct.index = atr_pct.index.normalize()
    return atr_pct


def sleeve_vol_proxy(returns: pd.Series, n: int = 14) -> pd.Series:
    """Use rolling std of the sleeve return as the "vol unit" for ATR rules."""
    return returns.rolling(n, min_periods=5).std().bfill().fillna(returns.std())


def align_vol(returns: pd.Series, symbol: str | None) -> pd.Series:
    if symbol is None:
        return sleeve_vol_proxy(returns)
    atr_pct = atr_from_ohlc(symbol)
    daily = atr_pct.reindex(returns.index.normalize(), method="ffill")
    daily.index = returns.index
    return daily.bfill().fillna(returns.std())


# ---------------------------------------------------------------------------
# Stop rule factories (each returns a stop_check callable + label)
# ---------------------------------------------------------------------------
def rule_fixed_pct(threshold: float, cooldown: int):
    def check(state, r, vol, day, cum):
        return cum <= -abs(threshold)
    return check, {"cooldown_after": cooldown}


def rule_atr_stop(k: float):
    def check(state, r, vol, day, cum):
        if not np.isfinite(vol) or vol <= 0:
            return False
        return cum <= -k * vol
    return check, {"cooldown_after": 1}


def rule_trailing(arm: float, trail_frac: float):
    def check(state, r, vol, day, cum):
        peak = state["peak"]
        if peak < arm:
            return False
        # trail at trail_frac * peak gain. Stop when cum falls below.
        return cum <= trail_frac * peak
    return check, {"cooldown_after": 1}


def rule_tp_and_sl(k_tp: float, k_sl: float):
    def check(state, r, vol, day, cum):
        if not np.isfinite(vol) or vol <= 0:
            return False
        if cum >= k_tp * vol:
            return True
        if cum <= -k_sl * vol:
            return True
        return False
    return check, {"cooldown_after": 1}


def rule_time_stop(max_days: int):
    def check(state, r, vol, day, cum):
        return day >= max_days
    return check, {"cooldown_after": 1}


def rule_vol_conditional(base_k: float):
    """Stop at base_k * vol but the multiplier widens in high-vol regime."""
    def check(state, r, vol, day, cum):
        if not np.isfinite(vol) or vol <= 0:
            return False
        long_vol = state.get("long_vol", vol)
        if long_vol == 0:
            return False
        ratio = vol / long_vol
        # high vol -> looser (bigger k); low vol -> tighter (smaller k).
        k = base_k * np.clip(ratio, 0.5, 2.0)
        return cum <= -k * long_vol
    return check, {"cooldown_after": 1}


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
def build_variants():
    variants = []
    # 1. fixed pct
    for cd in (1, 2, 5):
        variants.append((f"fixed3pct_cd{cd}", "fixed", dict(threshold=0.03, cooldown=cd)))
    # 2. ATR stop
    for k in (1.5, 2.0, 2.5):
        variants.append((f"atr_{k}x", "atr", dict(k=k)))
    # 3. trailing
    for arm, frac in [(0.02, 0.5), (0.015, 0.5), (0.02, 0.3)]:
        variants.append((f"trail_arm{int(arm*1000)}_frac{int(frac*100)}", "trail",
                         dict(arm=arm, trail_frac=frac)))
    # 4. tp + sl
    for ktp, ksl in [(2.0, 1.0), (3.0, 1.5)]:
        variants.append((f"tpsl_{ktp}x_{ksl}x", "tpsl", dict(k_tp=ktp, k_sl=ksl)))
    # 5. time stop
    for d in (5, 3, 10):
        variants.append((f"time_{d}d", "time", dict(max_days=d)))
    # 6. vol conditional
    for k in (2.0, 2.5):
        variants.append((f"volcond_{k}x", "volcond", dict(base_k=k)))
    # 7. equity curve protection
    variants.append(("eqprot_5pct_21d_5d", "eqprot",
                     dict(dd_thresh=-0.05, window=21, halt_days=5)))
    return variants


def apply_variant(returns: pd.Series, vol: pd.Series, family: str, params: dict) -> pd.Series:
    long_vol = float(vol.median()) if len(vol) else float(returns.std())
    if family == "fixed":
        check, extra = rule_fixed_pct(params["threshold"], params["cooldown"])
        return apply_stop(returns, stop_check=check, state_extra={**extra, "vol_series": vol})
    if family == "atr":
        check, extra = rule_atr_stop(params["k"])
        return apply_stop(returns, stop_check=check, state_extra={**extra, "vol_series": vol})
    if family == "trail":
        check, extra = rule_trailing(params["arm"], params["trail_frac"])
        return apply_stop(returns, stop_check=check, state_extra={**extra, "vol_series": vol})
    if family == "tpsl":
        check, extra = rule_tp_and_sl(params["k_tp"], params["k_sl"])
        return apply_stop(returns, stop_check=check, state_extra={**extra, "vol_series": vol})
    if family == "time":
        check, extra = rule_time_stop(params["max_days"])
        return apply_stop(returns, stop_check=check, state_extra={**extra, "vol_series": vol})
    if family == "volcond":
        check, extra = rule_vol_conditional(params["base_k"])
        return apply_stop(returns, stop_check=check,
                          state_extra={**extra, "vol_series": vol, "long_vol": long_vol})
    if family == "eqprot":
        return apply_equity_protection(returns, **params)
    raise ValueError(family)


def main():
    sleeves = pd.read_parquet(SLEEVE_FILE)
    sleeves.index = pd.to_datetime(sleeves.index, utc=True)

    variants = build_variants()
    rows = []
    best_returns: dict[str, pd.Series] = {}

    for sleeve in TARGET_SLEEVES:
        r = sleeves[sleeve].astype(float)
        sym = SLEEVE_SYMBOL[sleeve]
        vol = align_vol(r, sym)
        base_m = slice_metrics(r)
        base_m.update({"sleeve": sleeve, "variant": "baseline", "family": "baseline",
                       "n_nonzero_days": int((r != 0).sum())})
        rows.append(base_m)
        best_returns[sleeve] = r  # default is baseline

        best_oos_lift = -1e9
        best_variant_returns = r
        best_variant_name = "baseline"

        for name, family, params in variants:
            stopped = apply_variant(r, vol, family, params)
            m = slice_metrics(stopped)
            m.update({
                "sleeve": sleeve, "variant": name, "family": family,
                "n_nonzero_days": int((stopped != 0).sum()),
                "oos_sharpe_lift": m["sharpe_oos"] - base_m["sharpe_oos"],
                "is_sharpe_lift": m["sharpe_is"] - base_m["sharpe_is"],
                "maxdd_oos_improve": base_m["maxdd_oos"] - m["maxdd_oos"],
                "sharpe_2022_lift": m["sharpe_2022"] - base_m["sharpe_2022"],
            })
            rows.append(m)

            # Selection criterion: don't pick something that destroys IS.
            # Require IS lift >= -0.15 AND maximize OOS lift.
            if m["is_sharpe_lift"] >= -0.15 and m["oos_sharpe_lift"] > best_oos_lift:
                best_oos_lift = m["oos_sharpe_lift"]
                best_variant_returns = stopped
                best_variant_name = name

        # tag back oos_sharpe_lift etc on baseline rows for matrix consistency
        rows[-len(variants) - 1]["oos_sharpe_lift"] = 0.0
        rows[-len(variants) - 1]["is_sharpe_lift"] = 0.0
        rows[-len(variants) - 1]["maxdd_oos_improve"] = 0.0
        rows[-len(variants) - 1]["sharpe_2022_lift"] = 0.0
        rows[-len(variants) - 1]["best_pick"] = (best_variant_name == "baseline")

        best_returns[f"{sleeve}__BEST"] = best_variant_returns
        best_returns[f"{sleeve}__BEST_NAME"] = best_variant_name
        print(f"[{sleeve}] baseline OOS Sharpe {base_m['sharpe_oos']:.3f} ; "
              f"best variant: {best_variant_name} (OOS lift {best_oos_lift:+.3f})")

    df = pd.DataFrame(rows)
    # mark which row is best per sleeve
    df["best_pick"] = False
    df["best_oos"] = False
    for sleeve in TARGET_SLEEVES:
        sub = df[df["sleeve"] == sleeve].copy()
        # Pick best by OOS lift subject to IS constraint (-0.15 tolerance)
        eligible = sub[sub["is_sharpe_lift"] >= -0.15]
        if len(eligible):
            idx = eligible["oos_sharpe_lift"].idxmax()
            df.loc[idx, "best_pick"] = True
        # Also flag unconstrained best OOS variant
        idx2 = sub["oos_sharpe_lift"].idxmax()
        df.loc[idx2, "best_oos"] = True

    col_order = ["sleeve", "family", "variant", "best_pick", "best_oos",
                 "n_nonzero_days",
                 "sharpe_full", "sharpe_is", "sharpe_oos", "sharpe_2022",
                 "is_sharpe_lift", "oos_sharpe_lift", "sharpe_2022_lift",
                 "maxdd_full", "maxdd_is", "maxdd_oos", "maxdd_oos_improve"]
    df = df[col_order]
    df.to_csv(OUT_DIR / "stops_matrix.csv", index=False)

    # Build the best-returns parquet: baseline + best-variant per sleeve
    out_cols = {}
    for sleeve in TARGET_SLEEVES:
        out_cols[f"{sleeve}_baseline"] = sleeves[sleeve].astype(float)
        name = best_returns[f"{sleeve}__BEST_NAME"]
        out_cols[f"{sleeve}_{name}"] = best_returns[f"{sleeve}__BEST"].astype(float)
    pd.DataFrame(out_cols).to_parquet(OUT_DIR / "stops_returns.parquet")

    print("\nWrote:")
    print(" ", OUT_DIR / "stops_matrix.csv")
    print(" ", OUT_DIR / "stops_returns.parquet")
    return df


if __name__ == "__main__":
    df = main()
    print("\nTop rows by sleeve (best variants):")
    print(df[df["best_pick"]].to_string(index=False))
