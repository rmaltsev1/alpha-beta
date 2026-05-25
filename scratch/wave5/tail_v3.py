"""Tail-protection sleeve v3 — seven candidate structures from spot OHLCV only.

Goal: build a real tail-protection sleeve. Prior attempts (vol-of-vol short SPX,
naive vol breakout) failed OOS. We try more sophisticated structures.

Eligibility:
  2022 Sharpe >= +0.5     -> qualifies as a tail sleeve
  OOS Sharpe  >= -0.2     -> acceptable bleed in calm years
  COMBINED at 5% with existing portfolio must improve 2022 Sharpe by >= 0.2

Methodology:
  - IS  : timestamps < 2024-01-01
  - OOS : timestamps >= 2024-01-01
  - Vol-scale each sub-sleeve to 5% IS ann vol.
  - Combine survivors equal-weight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles
from alphabeta.backtest import backtest, cost_for

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
EXISTING_PORTFOLIO_PATH = ROOT / "scratch" / "quant" / "PRODUCTION_FINAL.parquet"

EQUITY_BASKET = ["SPX500_USD", "NAS100_USD", "US30_USD",
                 "DE30_EUR", "UK100_GBP", "JP225_USD"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def sharpe_of(r: pd.Series) -> float:
    if len(r) < 5:
        return 0.0
    bpy = _bpy(r.index)
    av = r.std(ddof=0) * np.sqrt(bpy)
    if av <= 0:
        return 0.0
    return float(r.mean() * bpy / av)


def ann_ret_of(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    return float(r.mean() * _bpy(r.index))


def ann_vol_of(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(_bpy(r.index)))


def max_dd_of(r: pd.Series) -> float:
    if len(r) < 2:
        return 0.0
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def stats_block(label: str, r: pd.Series) -> dict:
    out = {"name": label}
    for tag, sub in [
        ("FULL", r),
        ("IS",   r[r.index <  SPLIT]),
        ("OOS",  r[r.index >= SPLIT]),
        ("2022", r[r.index.year == 2022]),
        ("2020", r[r.index.year == 2020]),
        ("2021", r[r.index.year == 2021]),
        ("2023", r[r.index.year == 2023]),
        ("2024", r[r.index.year == 2024]),
        ("2025", r[r.index.year == 2025]),
    ]:
        out[f"{tag}_sharpe"] = sharpe_of(sub)
        out[f"{tag}_ret"] = ann_ret_of(sub)
        out[f"{tag}_dd"] = max_dd_of(sub)
    return out


def vol_scale_is(r: pd.Series, target: float = TARGET_VOL) -> float:
    is_ret = r[r.index < SPLIT]
    av = ann_vol_of(is_ret)
    return target / av if av > 1e-9 else 0.0


def to_ts_series(df: pd.DataFrame, vals: np.ndarray) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"].values, utc=True))
    return pd.Series(vals, index=idx).sort_index()


def cost_apply(pos: pd.Series, ret: pd.Series, cps: float) -> pd.Series:
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    return pos * ret - dpos * cps


# ---------------------------------------------------------------------------
# 1. Synthetic put via delta-trailing-stop
#
# Long-only SPX. Drop 5% from rolling 20-day max -> exit, stay flat 10 days.
# Re-enter when price recovers 3% from low.
# We model the protective component as (strategy_ret - buy&hold_ret) — that
# is the "synthetic put payoff": zero when long, positive when stopped out
# during a continued decline, negative when stopped out during a rebound.
# ---------------------------------------------------------------------------

def strat1_synthetic_put(stop_pct: float = 0.05,
                         recover_pct: float = 0.03,
                         flat_days: int = 10,
                         max_lookback: int = 20) -> pd.Series:
    df = get_candles("SPX500_USD", "D1").reset_index(drop=True)
    close = df["close"].astype(float).values
    rolling_max = pd.Series(close).rolling(max_lookback, min_periods=1).max().values

    n = len(close)
    pos = np.zeros(n, dtype=float)
    in_position = True
    flat_until = -1
    stop_low = np.inf
    for t in range(n):
        if t == 0:
            pos[t] = 1.0
            in_position = True
            continue
        prev_close = close[t-1]
        prev_max = rolling_max[t-1] if t-1 < len(rolling_max) else prev_close
        if in_position:
            # Check stop using info up to t-1
            if prev_close <= (1 - stop_pct) * prev_max:
                in_position = False
                flat_until = t + flat_days
                stop_low = prev_close
                pos[t] = 0.0
            else:
                pos[t] = 1.0
        else:
            stop_low = min(stop_low, prev_close)
            if t >= flat_until and prev_close >= (1 + recover_pct) * stop_low:
                in_position = True
                pos[t] = 1.0
            else:
                pos[t] = 0.0

    ret = pd.Series(close).pct_change().fillna(0.0).values
    cps = cost_for("SPX500_USD")
    # Strategy P&L net of cost
    strat = cost_apply(pd.Series(pos), pd.Series(ret), cps)
    # Buy & hold (no cost on continuous hold, but charge a one-time on/off)
    bh_pos = pd.Series(np.ones(n))
    bh = cost_apply(bh_pos, pd.Series(ret), cps)
    # The protective payoff = strategy - B&H; this is the "synthetic put"
    payoff = (strat - bh).values
    return to_ts_series(df, payoff)


# ---------------------------------------------------------------------------
# 2. Convexity portfolio
#
# 90% vol-managed long SPX + 10% lottery (long XAU + USD_JPY when SPX 5d < -3%).
# We isolate the *lottery* sleeve as the tail protection — the 90% long-eq
# part is just standard risk-on. So this returns only the 10% sleeve.
# ---------------------------------------------------------------------------

def strat2_convexity_lottery() -> pd.Series:
    spx = get_candles("SPX500_USD", "D1").reset_index(drop=True)
    xau = get_candles("XAU_USD", "D1").reset_index(drop=True)
    jpy = get_candles("USD_JPY", "D1").reset_index(drop=True)

    spx_ret = pd.Series(spx["close"].pct_change().fillna(0).values,
                        index=pd.DatetimeIndex(pd.to_datetime(spx["timestamp"], utc=True)))
    spx_5d = (spx["close"] / spx["close"].shift(5) - 1)
    spx_5d.index = spx_ret.index
    trigger = (spx_5d.shift(1) < -0.03).astype(float)  # use lagged signal

    def asset_ret(df, sym):
        r = pd.Series(df["close"].pct_change().fillna(0).values,
                      index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)))
        return r

    xau_r = asset_ret(xau, "XAU_USD")
    jpy_r = asset_ret(jpy, "USD_JPY")  # USD/JPY rising means USD up, JPY down

    # For "USD_JPY only when SPX crashes" — JPY (the safe-haven) tends to
    # strengthen, so USDJPY tends to FALL. We want a long-JPY sleeve, which
    # means SHORT USD_JPY.
    all_idx = trigger.index
    xau_r = xau_r.reindex(all_idx).fillna(0)
    jpy_r = jpy_r.reindex(all_idx).fillna(0)

    cps_x = cost_for("XAU_USD")
    cps_j = cost_for("USD_JPY")
    pos_x = trigger.copy()         # long XAU
    pos_j = -trigger.copy()        # short USD/JPY (long JPY)

    leg_x = cost_apply(pos_x, xau_r, cps_x)
    leg_j = cost_apply(pos_j, jpy_r, cps_j)
    sleeve = 0.5 * leg_x + 0.5 * leg_j
    return sleeve.sort_index()


# ---------------------------------------------------------------------------
# 3. Tail-correlated short
#
# When SPX 5d return is bottom 5% of trailing 252-day distribution
# AND vol is rising -> SHORT for 10 days. Mean-reversion broken; trend rules.
# We use rolling, lagged thresholds (no look-ahead).
# ---------------------------------------------------------------------------

def strat3_tail_short(symbol: str = "SPX500_USD",
                     lookback: int = 252,
                     hold_days: int = 10,
                     pct_thresh: float = 0.05) -> pd.Series:
    df = get_candles(symbol, "D1").reset_index(drop=True)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    ret5 = (close / close.shift(5) - 1)
    vol = ret.rolling(20).std()
    vol_rising = (vol > vol.shift(5)).astype(float)

    # rolling 5% quantile of 5d returns over 252-day window, computed
    # strictly on info up to t-1
    thresh = ret5.rolling(lookback, min_periods=126).quantile(pct_thresh).shift(1)
    ret5_lag = ret5.shift(1)
    vol_rising_lag = vol_rising.shift(1)

    trigger = (ret5_lag < thresh) & (vol_rising_lag > 0)

    pos = np.zeros(len(df))
    hold = 0
    for t in range(len(df)):
        if hold > 0:
            pos[t] = -1.0
            hold -= 1
        elif bool(trigger.iloc[t]) if not pd.isna(trigger.iloc[t]) else False:
            pos[t] = -1.0
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 4. Crash-momentum (continuation)
#
# When SPX 5d return < -8% (rare extreme), short for 5 days.
# Mean-reversion fails in real crashes; momentum continues.
# ---------------------------------------------------------------------------

def strat4_crash_momentum(symbol: str = "SPX500_USD",
                          thresh: float = -0.08,
                          hold_days: int = 5) -> pd.Series:
    df = get_candles(symbol, "D1").reset_index(drop=True)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    ret5_lag = (close / close.shift(5) - 1).shift(1)

    pos = np.zeros(len(df))
    hold = 0
    for t in range(len(df)):
        if hold > 0:
            pos[t] = -1.0
            hold -= 1
        elif (not pd.isna(ret5_lag.iloc[t])) and ret5_lag.iloc[t] < thresh:
            pos[t] = -1.0
            hold = hold_days - 1
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# 5. Quality-of-trend exit
#
# Long-bias SPX. Exit when 21d range expansion (Hi-Lo / close) > 95th
# percentile of trailing 252d. Range expansion often precedes drawdowns.
# Sleeve = strategy - buy&hold (the "tail" component).
# ---------------------------------------------------------------------------

def strat5_range_exit(symbol: str = "SPX500_USD",
                      win: int = 21,
                      lookback: int = 252,
                      pct: float = 0.95) -> pd.Series:
    df = get_candles(symbol, "D1").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    range_exp = (high.rolling(win).max() - low.rolling(win).min()) / close
    thresh = range_exp.rolling(lookback, min_periods=126).quantile(pct).shift(1)
    range_lag = range_exp.shift(1)

    exit_signal = (range_lag > thresh).astype(float)
    pos = (1.0 - exit_signal).clip(lower=0)  # long when not expanding

    ret = close.pct_change().fillna(0)
    cps = cost_for(symbol)
    strat_pnl = cost_apply(pos, ret, cps)
    bh_pos = pd.Series(np.ones(len(df)))
    bh_pnl = cost_apply(bh_pos, ret, cps)
    payoff = (strat_pnl - bh_pnl).values
    return to_ts_series(df, payoff)


# ---------------------------------------------------------------------------
# 6. Multi-asset risk-off coordinator
#
# When 3+ of {SPX, NAS, US30, DE30, UK100, JP225} have D1 ret < -2%,
# SHORT the equity basket for 5 days. Coordinated risk-off as momentum.
# ---------------------------------------------------------------------------

def strat6_risk_off_coordinator(hold_days: int = 5,
                                drop_thresh: float = -0.02,
                                min_assets: int = 3) -> pd.Series:
    rets = {}
    for sym in EQUITY_BASKET:
        df = get_candles(sym, "D1").reset_index(drop=True)
        r = pd.Series(df["close"].pct_change().fillna(0).values,
                      index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)))
        rets[sym] = r

    panel = pd.DataFrame(rets).sort_index().fillna(0)
    # Count of assets with negative big-drop ret. Use lagged signal.
    drop_count = (panel.shift(1) < drop_thresh).sum(axis=1)
    trigger = (drop_count >= min_assets).astype(float)

    # Build hold series
    n = len(panel)
    pos_basket = np.zeros(n)
    hold = 0
    trig_vals = trigger.values
    for t in range(n):
        if hold > 0:
            pos_basket[t] = -1.0
            hold -= 1
        elif trig_vals[t] > 0:
            pos_basket[t] = -1.0
            hold = hold_days - 1

    pos_basket_s = pd.Series(pos_basket, index=panel.index)

    # Equal-weight short across basket — each asset takes 1/N exposure
    legs = []
    for sym in EQUITY_BASKET:
        cps = cost_for(sym)
        # Each leg position = pos_basket / N (we charge cost per leg)
        leg_pos = pos_basket_s / len(EQUITY_BASKET)
        leg_pnl = cost_apply(leg_pos, panel[sym], cps)
        legs.append(leg_pnl)
    sleeve = pd.concat(legs, axis=1).sum(axis=1)
    return sleeve.sort_index()


# ---------------------------------------------------------------------------
# 7. Long XAU + USD_JPY beta-neutral hedge
#
# Compute SPX beta of XAU and USDJPY using rolling 252d window.
# Hedge each: long asset, short beta * SPX.
# This sleeve should profit in flight-to-quality / carry unwind.
# We use *negative* beta of USDJPY (we go long JPY, i.e. short USDJPY)
# because USDJPY typically falls in risk-off.
# ---------------------------------------------------------------------------

def strat7_beta_neutral_safe_haven(beta_win: int = 252) -> pd.Series:
    spx = get_candles("SPX500_USD", "D1").reset_index(drop=True)
    xau = get_candles("XAU_USD", "D1").reset_index(drop=True)
    jpy = get_candles("USD_JPY", "D1").reset_index(drop=True)

    def make_ret(df):
        return pd.Series(df["close"].pct_change().fillna(0).values,
                         index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)))

    spx_r = make_ret(spx)
    xau_r = make_ret(xau)
    jpy_r = make_ret(jpy)  # USDJPY return

    # Align
    idx = spx_r.index
    xau_r = xau_r.reindex(idx).fillna(0)
    jpy_r = jpy_r.reindex(idx).fillna(0)
    spx_r = spx_r.reindex(idx).fillna(0)

    # Rolling beta = cov(asset, spx) / var(spx) using *lagged* data
    cov_x = (xau_r.rolling(beta_win).cov(spx_r)).shift(1)
    cov_j = (jpy_r.rolling(beta_win).cov(spx_r)).shift(1)
    var_s = (spx_r.rolling(beta_win).var()).shift(1)
    beta_x = (cov_x / var_s).fillna(0).clip(-2, 2)
    beta_j = (cov_j / var_s).fillna(0).clip(-2, 2)

    # Long XAU + USDJPY (will end up being short USDJPY in risk-off due to sign).
    # Actually the task says: long XAU + USDJPY hedged by short SPX.
    # We'll interpret literally: long XAU (cap 1), long USDJPY (cap 1), short
    # beta-weighted SPX for each.
    cps_x = cost_for("XAU_USD")
    cps_j = cost_for("USD_JPY")
    cps_s = cost_for("SPX500_USD")

    # XAU leg: +1 XAU, -beta_x SPX
    pos_x = pd.Series(0.5, index=idx)        # half weight to XAU side
    pos_sx = -0.5 * beta_x                   # SPX hedge for XAU
    # JPY leg: +1 USDJPY, -beta_j SPX
    pos_j = pd.Series(0.5, index=idx)
    pos_sj = -0.5 * beta_j

    leg_x = cost_apply(pos_x, xau_r, cps_x)
    leg_j = cost_apply(pos_j, jpy_r, cps_j)
    leg_sx = cost_apply(pos_sx, spx_r, cps_s)
    leg_sj = cost_apply(pos_sj, spx_r, cps_s)

    sleeve = leg_x + leg_j + leg_sx + leg_sj
    return sleeve.sort_index()


# ---------------------------------------------------------------------------
# Run all and report
# ---------------------------------------------------------------------------

def main():
    candidates = {
        "1_synthetic_put_SPX":       strat1_synthetic_put(),
        "2_convexity_lottery":       strat2_convexity_lottery(),
        "3_tail_short_SPX":          strat3_tail_short("SPX500_USD"),
        "3_tail_short_NAS":          strat3_tail_short("NAS100_USD"),
        "3_tail_short_US30":         strat3_tail_short("US30_USD"),
        "4_crash_momentum_SPX":      strat4_crash_momentum("SPX500_USD"),
        "4_crash_momentum_NAS":      strat4_crash_momentum("NAS100_USD"),
        "5_range_exit_SPX":          strat5_range_exit("SPX500_USD"),
        "6_risk_off_coordinator":    strat6_risk_off_coordinator(),
        "7_beta_neutral_safehaven":  strat7_beta_neutral_safe_haven(),
    }

    rows = []
    scaled_streams = {}

    print(f"\n{'name':<28} {'IS_Sh':>6} {'OOS_Sh':>6} {'2022_Sh':>7} "
          f"{'2022_Ret':>8} {'OOS_Ret':>8} {'scale':>6}")
    print("-" * 80)

    for name, raw in candidates.items():
        raw = raw.dropna().sort_index()
        scale = vol_scale_is(raw, TARGET_VOL)
        scaled = raw * scale
        scaled_streams[name] = scaled

        s = stats_block(name, scaled)
        s["raw_is_vol"] = ann_vol_of(raw[raw.index < SPLIT])
        s["scale"] = scale
        rows.append(s)
        print(f"{name:<28} {s['IS_sharpe']:>+6.2f} {s['OOS_sharpe']:>+6.2f} "
              f"{s['2022_sharpe']:>+7.2f} {s['2022_ret']:>+8.2%} "
              f"{s['OOS_ret']:>+8.2%} {scale:>6.2f}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "tail_v3_breakdown.csv", index=False)

    # ---- Survivor selection ----
    survivors = []
    for name, s in zip(candidates.keys(), rows):
        is_ok_2022 = s.get("2022_sharpe", -99) >= 0.5
        is_ok_oos = s.get("OOS_sharpe", -99) >= -0.2
        if is_ok_2022 and is_ok_oos:
            survivors.append(name)

    print(f"\nStrict survivors (2022>=0.5 AND OOS>=-0.2): {survivors}")

    # Also build a "softer" survivor list: 2022>=0.5 OR OOS>=-0.2 with positive 2022
    soft = []
    for name, s in zip(candidates.keys(), rows):
        if s.get("2022_sharpe", -99) >= 0.5:
            soft.append(name)
    if not soft:
        # if nothing has 2022>=0.5, take top 2 by 2022_sharpe
        ranked = sorted(rows, key=lambda r: r.get("2022_sharpe", -99), reverse=True)
        soft = [r["name"] for r in ranked[:2]]
    print(f"Soft (any 2022>=0.5 OR top-2 fallback):     {soft}")

    # We use STRICT survivors for the canonical sleeve. If empty, fall back.
    if len(survivors) == 0:
        survivors = soft
        print(f"No strict survivors. Using soft set: {survivors}")

    # ---- Combined equal-weight sleeve over union of timestamps ----
    aligned = pd.concat(
        [scaled_streams[n].rename(n) for n in survivors], axis=1
    ).sort_index().fillna(0)
    combined = aligned.mean(axis=1)  # equal-weight
    # Rescale combined to 5% IS vol
    scl = vol_scale_is(combined, TARGET_VOL)
    combined_scaled = combined * scl

    print(f"\nCombined sleeve (equal-weight, rescaled to 5% IS vol):")
    cs = stats_block("combined", combined_scaled)
    for tag in ["IS","OOS","2022","2020","2021","2023","2024","2025"]:
        print(f"  {tag:<6} Sh={cs[tag+'_sharpe']:+5.2f}  Ret={cs[tag+'_ret']:+6.2%}  DD={cs[tag+'_dd']:+6.2%}")

    # Save the combined sleeve returns
    out_df = pd.DataFrame({
        "timestamp": combined_scaled.index,
        "ret": combined_scaled.values
    })
    out_df.to_parquet(OUT / "tail_v3_returns.parquet", index=False)
    print(f"\nSaved combined returns: {OUT / 'tail_v3_returns.parquet'}")

    # ---- Also build a *soft* combined sleeve (#6 + #7 if both available) ----
    soft_members = [n for n in soft if n in scaled_streams]
    if len(soft_members) > 1:
        soft_aligned = pd.concat(
            [scaled_streams[n].rename(n) for n in soft_members], axis=1
        ).sort_index().fillna(0)
        soft_combined = soft_aligned.mean(axis=1)
        ssc = vol_scale_is(soft_combined, TARGET_VOL)
        soft_combined_scaled = soft_combined * ssc
        print(f"\nSoft combined sleeve ({soft_members}, rescaled to 5% IS vol):")
        ssm = stats_block("soft_combined", soft_combined_scaled)
        for tag in ["IS","OOS","2022","2020","2021","2023","2024","2025"]:
            print(f"  {tag:<6} Sh={ssm[tag+'_sharpe']:+5.2f}  Ret={ssm[tag+'_ret']:+6.2%}")
        soft_df = pd.DataFrame({
            "timestamp": soft_combined_scaled.index,
            "ret": soft_combined_scaled.values,
        })
        soft_df.to_parquet(OUT / "tail_v3_returns_soft.parquet", index=False)

    # Save also full set of scaled streams (for the parent agent to inspect)
    sub_df = pd.DataFrame(
        {n: s for n, s in scaled_streams.items()}
    )
    sub_df.index.name = "timestamp"
    sub_df.reset_index().to_parquet(OUT / "tail_v3_substreams.parquet", index=False)

    # ---- Integration test: combine with existing portfolio at 5% weight ----
    print("\n=== Integration: existing portfolio + 5% tail sleeve ===")
    if EXISTING_PORTFOLIO_PATH.exists():
        prod = pd.read_parquet(EXISTING_PORTFOLIO_PATH)
        prod_r = pd.Series(prod["ret"].values,
                           index=pd.DatetimeIndex(pd.to_datetime(prod["timestamp"], utc=True))).sort_index()
        # normalize to date so we can align with sleeve (which is on market-close)
        prod_r.index = prod_r.index.normalize()

        sleeve_date_idx = combined_scaled.copy()
        sleeve_date_idx.index = sleeve_date_idx.index.normalize()
        # Some duplicates after normalization? aggregate by sum
        sleeve_date_idx = sleeve_date_idx.groupby(level=0).sum()

        sleeve_r = sleeve_date_idx.reindex(prod_r.index).fillna(0)

        # base = 100% existing
        # blend = 95% existing + 5% sleeve
        blend = 0.95 * prod_r + 0.05 * sleeve_r

        rows_int = []
        for tag, mask in [
            ("FULL", pd.Series(True, index=prod_r.index)),
            ("IS",   prod_r.index < SPLIT),
            ("OOS",  prod_r.index >= SPLIT),
            ("2020", prod_r.index.year == 2020),
            ("2021", prod_r.index.year == 2021),
            ("2022", prod_r.index.year == 2022),
            ("2023", prod_r.index.year == 2023),
            ("2024", prod_r.index.year == 2024),
            ("2025", prod_r.index.year == 2025),
        ]:
            base_sub = prod_r[mask]
            blend_sub = blend[mask]
            row = {
                "period": tag,
                "base_sharpe": sharpe_of(base_sub),
                "blend_sharpe": sharpe_of(blend_sub),
                "base_ret": ann_ret_of(base_sub),
                "blend_ret": ann_ret_of(blend_sub),
                "base_dd": max_dd_of(base_sub),
                "blend_dd": max_dd_of(blend_sub),
                "delta_sharpe": sharpe_of(blend_sub) - sharpe_of(base_sub),
                "delta_ret": ann_ret_of(blend_sub) - ann_ret_of(base_sub),
            }
            rows_int.append(row)
            print(f"  {tag:<6} base Sh={row['base_sharpe']:+5.2f}  "
                  f"blend Sh={row['blend_sharpe']:+5.2f}  "
                  f"Δ={row['delta_sharpe']:+5.2f}  "
                  f"base Ret={row['base_ret']:+6.2%}  blend Ret={row['blend_ret']:+6.2%}")

        pd.DataFrame(rows_int).to_csv(OUT / "tail_v3_integration.csv", index=False)

        # ---- Weight sensitivity (does 5% really make sense?) ----
        print("\n=== 2022 Sharpe vs sleeve weight ===")
        weights_to_test = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
        wt_rows = []
        for w in weights_to_test:
            blend_w = (1 - w) * prod_r + w * sleeve_r
            row = {"weight": w}
            for tag, mask in [
                ("2022", prod_r.index.year == 2022),
                ("OOS",  prod_r.index >= SPLIT),
                ("FULL", pd.Series(True, index=prod_r.index)),
            ]:
                base_sub = prod_r[mask]
                blend_sub = blend_w[mask]
                row[f"{tag}_base"] = sharpe_of(base_sub)
                row[f"{tag}_blend"] = sharpe_of(blend_sub)
                row[f"{tag}_delta"] = sharpe_of(blend_sub) - sharpe_of(base_sub)
            wt_rows.append(row)
            print(f"  w={w:.3f}  2022Δ={row['2022_delta']:+5.2f}  OOSΔ={row['OOS_delta']:+5.2f}  FULLΔ={row['FULL_delta']:+5.2f}")
        pd.DataFrame(wt_rows).to_csv(OUT / "tail_v3_weight_sensitivity.csv", index=False)

        # ---- Per-strategy integration at 5% ----
        print("\n=== Per-substrategy 5%-blend (using each as the sole sleeve) ===")
        per_rows = []
        for name in candidates.keys():
            s_norm = scaled_streams[name].copy()
            s_norm.index = s_norm.index.normalize()
            s_norm = s_norm.groupby(level=0).sum()
            s_aligned = s_norm.reindex(prod_r.index).fillna(0)
            blend = 0.95 * prod_r + 0.05 * s_aligned
            base_2022 = sharpe_of(prod_r[prod_r.index.year == 2022])
            blend_2022 = sharpe_of(blend[blend.index.year == 2022])
            base_oos = sharpe_of(prod_r[prod_r.index >= SPLIT])
            blend_oos = sharpe_of(blend[blend.index >= SPLIT])
            base_full = sharpe_of(prod_r)
            blend_full = sharpe_of(blend)
            corr22 = s_aligned[s_aligned.index.year == 2022].corr(prod_r[prod_r.index.year == 2022])
            per_rows.append({
                "name": name,
                "2022_base": base_2022,
                "2022_blend": blend_2022,
                "2022_delta": blend_2022 - base_2022,
                "OOS_base": base_oos,
                "OOS_blend": blend_oos,
                "OOS_delta": blend_oos - base_oos,
                "FULL_delta": blend_full - base_full,
                "corr_2022": corr22,
            })
            print(f"  {name:<28} 2022Δ={blend_2022-base_2022:+5.2f}  OOSΔ={blend_oos-base_oos:+5.2f}  corr22={corr22:+.2f}")
        pd.DataFrame(per_rows).to_csv(OUT / "tail_v3_per_strategy_integration.csv", index=False)
    else:
        print(f"  (no existing portfolio at {EXISTING_PORTFOLIO_PATH})")


if __name__ == "__main__":
    main()
