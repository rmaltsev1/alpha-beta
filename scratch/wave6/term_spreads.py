"""Term-structure-aware vol strategies + multi-leg basket spreads — wave 6.

Two families:

  TERM-STRUCTURE VOL (per-asset):
    T1. Short/Long RV ratio (5d/30d) — mean reversion / vol expansion
    T2. Vol cone — all horizons (5/15/30/60/90d) at 75th+ pctile -> short eq;
                   all at 25th- -> long eq
    T3. Vol-of-vol regime spread — switch between trend & mean-reversion
        depending on how noisy 30d RV itself is.

  MULTI-LEG SPREADS (cross-asset baskets):
    M4. Risk-on basket  vs  safe-haven basket
    M5. US equity basket  vs  non-US equity basket
    M6. Gold  vs  USD-pair-inverted basket
    M7. Crypto basket  vs  equity basket, gated on falling crypto/SPX corr
    M8. BTC/ETH Asian-session vs US-session (H1)

Methodology:
  IS  < 2024-01-01;  OOS >= 2024-01-01.
  Walk-forward all signals.  Vol-scale each sleeve to 5% IS ann vol.
  Survivor filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.
  Combine survivors equal-weight, then rescale to 5% IS vol.

Outputs:
  scratch/wave6/term_spreads_returns.parquet
  scratch/wave6/term_spreads_breakdown.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles
from alphabeta.backtest import cost_for
from alphabeta.symbols import CRYPTO, FOREX, INDEX

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05

EQUITY_BASKET = ["SPX500_USD", "NAS100_USD", "US30_USD",
                 "DE30_EUR", "UK100_GBP", "JP225_USD"]
US_EQUITY    = ["SPX500_USD", "NAS100_USD", "US30_USD"]
NONUS_EQUITY = ["DE30_EUR", "UK100_GBP", "JP225_USD"]
CRYPTO_MAJ   = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
RISK_ON      = ["BTCUSDT", "ETHUSDT", "SOLUSDT",
                "SPX500_USD", "NAS100_USD", "US30_USD"]
SAFE_HAVEN   = ["XAU_USD", "USD_JPY"]


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
        ("2020", r[r.index.year == 2020]),
        ("2021", r[r.index.year == 2021]),
        ("2022", r[r.index.year == 2022]),
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


def _load_d1(symbol: str) -> pd.DataFrame:
    return get_candles(symbol, "D1").reset_index(drop=True)


def _load_h1(symbol: str) -> pd.DataFrame:
    return get_candles(symbol, "H1").reset_index(drop=True)


def _close_index(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        df["close"].astype(float).values,
        index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)),
    )


def _ret_index(df: pd.DataFrame) -> pd.Series:
    c = _close_index(df)
    return c.pct_change().fillna(0)


# ---------------------------------------------------------------------------
# T1. Short-term vs long-term realized-vol ratio
#   ratio = 5d RV / 30d RV (lagged).
#     ratio > 1.5  -> short-term stress, expect mean reversion       -> LONG
#     ratio < 0.7  -> compression, expect vol expansion              -> SHORT (smaller)
#   Vol-scaled small position: size = clip(0.2 / pred_vol, 0, 1).
# ---------------------------------------------------------------------------

def t1_st_lt_vol_ratio(symbol: str,
                       short_win: int = 5,
                       long_win: int = 30,
                       high_thresh: float = 1.5,
                       low_thresh: float = 0.7,
                       ewma_span: int = 10,
                       hold_days: int = 5) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < long_win + 60:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    rv_s = ret.rolling(short_win).std() * np.sqrt(252)
    rv_l = ret.rolling(long_win).std() * np.sqrt(252)
    ratio = (rv_s / rv_l).shift(1)

    pred_vol = (ret.ewm(span=ewma_span, adjust=False).std() *
                np.sqrt(252)).shift(1)
    target = 0.20
    size = (target / pred_vol.replace(0, np.nan)).clip(0, 1.0)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    rvals = ratio.values
    size_vals = size.fillna(0).values
    for t in range(n):
        if hold > 0:
            pos[t] = sign * size_vals[t]
            hold -= 1
            continue
        v = rvals[t]
        if np.isnan(v) or size_vals[t] <= 0:
            pos[t] = 0.0
            continue
        if v > high_thresh:
            sign = 1.0
            pos[t] = sign * size_vals[t]
            hold = hold_days - 1
        elif v < low_thresh:
            sign = -0.5
            pos[t] = sign * size_vals[t]
            hold = hold_days - 1
        else:
            pos[t] = 0.0
    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# T2. Vol cone — multi-horizon RV regime
#   Compute RV at 5/15/30/60/90d horizons. For each, take rolling 252d pctile
#   rank of its own history (walk-forward). If ALL 5 ranks >= 0.75 -> SHORT.
#   If ALL 5 ranks <= 0.25 -> LONG. Otherwise flat.
# ---------------------------------------------------------------------------

def t2_vol_cone(symbol: str,
                horizons=(5, 15, 30, 60, 90),
                rank_win: int = 252,
                rank_min: int = 126,
                hold_days: int = 5) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < max(horizons) + rank_win + 30:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)

    ranks = []
    for h in horizons:
        rv = ret.rolling(h).std() * np.sqrt(252)
        rk = rv.rolling(rank_win, min_periods=rank_min).rank(pct=True)
        ranks.append(rk.shift(1))
    rk_df = pd.concat(ranks, axis=1)
    rk_df.columns = [f"r{h}" for h in horizons]

    all_high = (rk_df >= 0.75).all(axis=1)
    all_low  = (rk_df <= 0.25).all(axis=1)

    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    hi = all_high.values
    lo = all_low.values
    for t in range(n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
            continue
        if hi[t]:
            sign = -1.0
            pos[t] = sign
            hold = hold_days - 1
        elif lo[t]:
            sign = 1.0
            pos[t] = sign
            hold = hold_days - 1
        else:
            pos[t] = 0.0
    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# T3. Vol-of-vol regime spread — switch between trend and mean-reversion
#   vov = std of 30d RV over last 30 days (walk-forward).
#   If vov > rolling 70th pctile (regime-uncertain) -> mean-revert signal:
#       pos = -sign(5d return)
#   else (stable regime) -> trend signal:
#       pos = sign(63d return)
# ---------------------------------------------------------------------------

def t3_vov_regime(symbol: str,
                  rv_win: int = 30,
                  vov_win: int = 30,
                  pct_win: int = 252,
                  pct_min: int = 126,
                  high_pct: float = 0.70,
                  mr_lookback: int = 5,
                  trend_lookback: int = 63,
                  hold_days: int = 5) -> pd.Series:
    df = _load_d1(symbol)
    if len(df) < pct_win + rv_win + 60:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0)
    rv = ret.rolling(rv_win).std() * np.sqrt(252)
    vov = rv.rolling(vov_win).std()
    vov_thresh = vov.rolling(pct_win, min_periods=pct_min).quantile(high_pct).shift(1)
    vov_lag = vov.shift(1)

    mr_sig = -np.sign((close / close.shift(mr_lookback) - 1).shift(1))
    tr_sig =  np.sign((close / close.shift(trend_lookback) - 1).shift(1))

    use_mr = (vov_lag > vov_thresh).fillna(False)
    raw_pos = np.where(use_mr, mr_sig, tr_sig)
    pos_series = pd.Series(raw_pos, index=close.index).fillna(0)

    # Hold for hold_days to cut churn
    n = len(df)
    pos = np.zeros(n)
    hold = 0
    sign = 0.0
    rps = pos_series.values
    for t in range(n):
        if hold > 0:
            pos[t] = sign
            hold -= 1
            continue
        s = rps[t]
        if s != 0:
            sign = float(s)
            pos[t] = sign
            hold = hold_days - 1
        else:
            pos[t] = 0.0
    cps = cost_for(symbol)
    sleeve = cost_apply(pd.Series(pos, index=close.index), ret, cps)
    return to_ts_series(df, sleeve.values)


# ---------------------------------------------------------------------------
# Multi-leg helpers
# ---------------------------------------------------------------------------

def _basket_returns(symbols: list[str]) -> pd.DataFrame:
    """Daily returns table (cols=symbols, index=date). Inner-joined on date."""
    series = {}
    for s in symbols:
        try:
            df = _load_d1(s)
        except FileNotFoundError:
            continue
        series[s] = _ret_index(df)
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1).sort_index()
    # Use forward-fill of zero for missing trading days (asset closed) — but
    # that creates fake returns. Better: only score signals on the union of
    # all dates and treat NaN as 0 for that asset on that day.
    return out.fillna(0)


def _equal_weight_basket(symbols: list[str]) -> pd.Series:
    rets = _basket_returns(symbols)
    if rets.empty:
        return pd.Series(dtype=float)
    return rets.mean(axis=1)


def _signed_basket_pnl(long_syms: list[str], short_syms: list[str],
                       sign_series: pd.Series) -> pd.Series:
    """Daily pnl of a long-short basket controlled by sign_series.
    sign_series: +1 -> long long_syms / short short_syms; -1 -> reverse; 0 -> flat.
    Each leg equal-weight, cost charged per-symbol on |dpos|.
    """
    long_rets  = _basket_returns(long_syms)
    short_rets = _basket_returns(short_syms)
    if long_rets.empty or short_rets.empty:
        return pd.Series(dtype=float)
    # Align all on the union of dates
    all_idx = long_rets.index.union(short_rets.index)
    long_rets  = long_rets.reindex(all_idx).fillna(0)
    short_rets = short_rets.reindex(all_idx).fillna(0)
    sig = sign_series.reindex(all_idx).fillna(0)

    nL, nS = long_rets.shape[1], short_rets.shape[1]
    # Per-symbol positions
    long_pos  = pd.DataFrame(np.tile(sig.values[:, None], nL) / nL,
                             index=all_idx, columns=long_rets.columns)
    short_pos = pd.DataFrame(np.tile(-sig.values[:, None], nS) / nS,
                             index=all_idx, columns=short_rets.columns)

    # Costs per symbol
    def _cost(pos_df: pd.DataFrame) -> pd.Series:
        c = pd.Series(0.0, index=pos_df.index)
        for col in pos_df.columns:
            cps = cost_for(col)
            d = pos_df[col].diff().fillna(pos_df[col].iloc[0]).abs()
            c = c + d * cps
        return c

    gross = (long_pos.values * long_rets.values).sum(axis=1) + \
            (short_pos.values * short_rets.values).sum(axis=1)
    cost  = _cost(long_pos) + _cost(short_pos)
    net = pd.Series(gross, index=all_idx) - cost
    return net.sort_index()


# ---------------------------------------------------------------------------
# M4. Risk-on basket vs safe-haven basket
#   risk-on regime:  SPX 21d return > 0  AND  SPX 21d RV < rolling 60th pctile
#   risk-off regime: SPX 21d return < 0  AND  SPX 21d RV > rolling 60th pctile
#   sign = +1 in risk-on, -1 in risk-off (long-short basket reverses).
# ---------------------------------------------------------------------------

def m4_risk_on_off(rv_win: int = 21,
                   ret_win: int = 21,
                   pct_win: int = 252,
                   pct_min: int = 126,
                   vol_pct: float = 0.60,
                   hold_days: int = 5) -> pd.Series:
    spx = _load_d1("SPX500_USD")
    spx_close = _close_index(spx)
    spx_ret = spx_close.pct_change().fillna(0)
    mom = (spx_close / spx_close.shift(ret_win) - 1).shift(1)
    rv = (spx_ret.rolling(rv_win).std() * np.sqrt(252))
    rv_q = rv.rolling(pct_win, min_periods=pct_min).quantile(vol_pct).shift(1)
    rv_lag = rv.shift(1)

    risk_on  = (mom > 0) & (rv_lag < rv_q)
    risk_off = (mom < 0) & (rv_lag > rv_q)

    sig = pd.Series(0.0, index=spx_close.index)
    sig[risk_on]  = 1.0
    sig[risk_off] = -1.0

    # Hold-min: cut churn
    sig_h = sig.copy()
    cur = 0.0
    hold = 0
    out = np.zeros(len(sig_h))
    vals = sig_h.values
    for t in range(len(out)):
        if hold > 0:
            out[t] = cur
            hold -= 1
            continue
        if vals[t] != 0:
            cur = vals[t]
            out[t] = cur
            hold = hold_days - 1
        else:
            out[t] = 0.0
    sig_h = pd.Series(out, index=sig.index)
    return _signed_basket_pnl(RISK_ON, SAFE_HAVEN, sig_h)


# ---------------------------------------------------------------------------
# M5. US equity vs non-US equity
#   sign = sign( trailing 63d US-basket - trailing 63d nonUS-basket ), lagged
# ---------------------------------------------------------------------------

def m5_us_vs_nonus(lookback: int = 63,
                   hold_days: int = 5) -> pd.Series:
    us_rets = _basket_returns(US_EQUITY)
    nu_rets = _basket_returns(NONUS_EQUITY)
    if us_rets.empty or nu_rets.empty:
        return pd.Series(dtype=float)
    us_mean = us_rets.mean(axis=1)
    nu_mean = nu_rets.mean(axis=1)
    us_cum = (1 + us_mean).rolling(lookback).apply(lambda x: x.prod() - 1)
    nu_cum = (1 + nu_mean).rolling(lookback).apply(lambda x: x.prod() - 1)
    diff = (us_cum - nu_cum).shift(1)
    raw = np.sign(diff).fillna(0)

    n = len(raw)
    out = np.zeros(n)
    cur = 0.0
    hold = 0
    vals = raw.values
    for t in range(n):
        if hold > 0:
            out[t] = cur
            hold -= 1
            continue
        if vals[t] != 0:
            cur = float(vals[t])
            out[t] = cur
            hold = hold_days - 1
        else:
            out[t] = 0.0
    sig = pd.Series(out, index=raw.index)
    return _signed_basket_pnl(US_EQUITY, NONUS_EQUITY, sig)


# ---------------------------------------------------------------------------
# M6. Gold vs USD-pair-inverted basket
#   Sign turns on when XAU has positive 21d momentum AND DXY-proxy
#   (negative of average(EUR_USD, GBP_USD)) has negative 21d momentum
#   (USD weakening). Otherwise flat. The "USD basket" used as short leg is
#   actually EUR_USD + GBP_USD (long EUR/GBP = short USD), so to express
#   "long XAU when USD weakens" the short leg below is *USD-strength*.
#   Implementation: long [XAU_USD] short [EUR_USD, GBP_USD]_inverted means
#   long XAU and SHORT (long_EUR_USD + long_GBP_USD)_inverted = long XAU and
#   LONG EUR/GBP. We implement directly as long [XAU] + long [EUR, GBP].
# ---------------------------------------------------------------------------

def m6_gold_vs_usd(lookback: int = 21,
                   hold_days: int = 5) -> pd.Series:
    xau = _load_d1("XAU_USD")
    eur = _load_d1("EUR_USD")
    gbp = _load_d1("GBP_USD")
    xau_close = _close_index(xau)
    xau_mom = (xau_close / xau_close.shift(lookback) - 1).shift(1)

    eur_close = _close_index(eur)
    gbp_close = _close_index(gbp)
    # USD strength proxy: avg of -EUR mom and -GBP mom
    eur_mom = (eur_close / eur_close.shift(lookback) - 1).shift(1)
    gbp_mom = (gbp_close / gbp_close.shift(lookback) - 1).shift(1)
    usd_strength = -((eur_mom + gbp_mom) / 2.0)

    # Align
    idx = xau_mom.index.union(usd_strength.index)
    xau_mom_a = xau_mom.reindex(idx).ffill()
    usd_str_a = usd_strength.reindex(idx).ffill()

    on = (xau_mom_a > 0) & (usd_str_a < 0)
    sig_raw = on.astype(float)  # 0/1 only — one-sided trade

    # Hold
    n = len(sig_raw)
    out = np.zeros(n)
    cur = 0.0
    hold = 0
    vals = sig_raw.values
    for t in range(n):
        if hold > 0:
            out[t] = cur
            hold -= 1
            continue
        if vals[t] > 0:
            cur = 1.0
            out[t] = cur
            hold = hold_days - 1
        else:
            cur = 0.0
            out[t] = 0.0
    sig = pd.Series(out, index=sig_raw.index)
    # Long XAU; long EUR & GBP (i.e., short USD via FX)
    # In our basket P&L helper, the "long" leg gets +sign and short leg -sign.
    # We want long XAU, long EUR_USD, long GBP_USD all gated by sig=+1 only.
    # We construct it as two parallel long-only baskets, weighted half-half,
    # using `_signed_basket_pnl(long=[XAU], short=[...])` with sign flipped on
    # the short leg by inverting the symbol set: instead, just hand-build.
    xau_pos = sig.reindex(xau_close.index).fillna(0)
    eur_pos = sig.reindex(eur_close.index).fillna(0)
    gbp_pos = sig.reindex(gbp_close.index).fillna(0)

    xau_ret = xau_close.pct_change().fillna(0)
    eur_ret = eur_close.pct_change().fillna(0)
    gbp_ret = gbp_close.pct_change().fillna(0)

    cps_xau, cps_eur, cps_gbp = cost_for("XAU_USD"), cost_for("EUR_USD"), cost_for("GBP_USD")

    def _leg(pos: pd.Series, ret: pd.Series, cps: float, weight: float) -> pd.Series:
        p = pos * weight
        dpos = p.diff().fillna(p.iloc[0]).abs()
        return p * ret - dpos * cps

    # 0.5 weight on XAU, 0.25 each on EUR and GBP -> total notional = 1
    leg1 = _leg(xau_pos, xau_ret, cps_xau, 0.5)
    leg2 = _leg(eur_pos, eur_ret, cps_eur, 0.25)
    leg3 = _leg(gbp_pos, gbp_ret, cps_gbp, 0.25)
    all_idx = leg1.index.union(leg2.index).union(leg3.index)
    pnl = (leg1.reindex(all_idx).fillna(0) +
           leg2.reindex(all_idx).fillna(0) +
           leg3.reindex(all_idx).fillna(0))
    return pnl.sort_index()


# ---------------------------------------------------------------------------
# M7. Crypto basket vs equity basket — gated on falling 30d corr(BTC, SPX)
#   When rolling 30d corr(BTC, SPX) is BELOW its trailing 252d 30th pctile,
#   the two are decoupling. Take direction from crypto's own 21d momentum:
#       long crypto / short equity if crypto 21d > 0
#       short crypto / long equity if crypto 21d < 0
# ---------------------------------------------------------------------------

def m7_crypto_vs_equity(corr_win: int = 30,
                        pct_win: int = 252,
                        pct_min: int = 126,
                        corr_pct: float = 0.30,
                        mom_win: int = 21,
                        hold_days: int = 5) -> pd.Series:
    btc = _load_d1("BTCUSDT")
    spx = _load_d1("SPX500_USD")
    btc_ret = _ret_index(btc)
    spx_ret = _ret_index(spx)
    idx = btc_ret.index.union(spx_ret.index)
    br = btc_ret.reindex(idx).fillna(0)
    sr = spx_ret.reindex(idx).fillna(0)
    rolling_corr = br.rolling(corr_win).corr(sr)
    corr_q = rolling_corr.rolling(pct_win, min_periods=pct_min).quantile(corr_pct).shift(1)
    corr_lag = rolling_corr.shift(1)
    decoupling = (corr_lag < corr_q)

    # Crypto basket momentum
    crypto_rets = _basket_returns(CRYPTO_MAJ).reindex(idx).fillna(0)
    crypto_eq = (1 + crypto_rets.mean(axis=1)).cumprod()
    crypto_mom = (crypto_eq / crypto_eq.shift(mom_win) - 1).shift(1)

    sig = pd.Series(0.0, index=idx)
    sig[decoupling & (crypto_mom > 0)] =  1.0
    sig[decoupling & (crypto_mom < 0)] = -1.0

    # Hold
    n = len(sig)
    out = np.zeros(n)
    cur = 0.0
    hold = 0
    vals = sig.values
    for t in range(n):
        if hold > 0:
            out[t] = cur
            hold -= 1
            continue
        if vals[t] != 0:
            cur = float(vals[t])
            out[t] = cur
            hold = hold_days - 1
        else:
            out[t] = 0.0
    sig_h = pd.Series(out, index=sig.index)
    return _signed_basket_pnl(CRYPTO_MAJ, EQUITY_BASKET, sig_h)


# ---------------------------------------------------------------------------
# M8. BTC/ETH Asian-session vs US-session (H1)
#   For each H1 bar in 00..07 UTC (Asian): long sign.
#   For each H1 bar in 12..19 UTC (US):    short sign.
#   Direction comes from the trailing-90d mean return of the SAME session
#   slice (walk-forward, lagged). I.e., if Asian session historically pays
#   positive in last 90 days -> go long during Asian session; else short.
#   Same for US session.
#
#   Reported per symbol; H1 returns. Combined into daily by summing within day.
# ---------------------------------------------------------------------------

def m8_session_btc_eth(symbol: str,
                       asia_hours=range(0, 8),
                       us_hours=range(12, 20),
                       lookback_days: int = 90) -> pd.Series:
    df = _load_h1(symbol)
    if len(df) < 24 * 100:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    ret = close.pct_change().fillna(0).values
    hour = ts.dt.hour.values
    in_asia = np.isin(hour, list(asia_hours))
    in_us   = np.isin(hour, list(us_hours))

    n = len(df)
    pos = np.zeros(n)

    # Walk-forward bias: re-evaluate daily. The bias for today comes from
    # mean session ret over [t-24*lookback_days, t-1] of that session's hours.
    win = 24 * lookback_days
    # Pre-compute session masks
    asia_r = np.where(in_asia, ret, 0.0)
    us_r   = np.where(in_us,   ret, 0.0)

    asia_r_s = pd.Series(asia_r)
    us_r_s = pd.Series(us_r)
    asia_bias = asia_r_s.rolling(win, min_periods=win // 2).mean().shift(1).fillna(0).values
    us_bias   = us_r_s.rolling(win, min_periods=win // 2).mean().shift(1).fillna(0).values

    for t in range(n):
        if in_asia[t]:
            pos[t] = np.sign(asia_bias[t])
        elif in_us[t]:
            pos[t] = np.sign(us_bias[t])
        else:
            pos[t] = 0.0

    cps = cost_for(symbol)
    pos_s = pd.Series(pos, index=close.index)
    dpos = pos_s.diff().fillna(pos_s.iloc[0]).abs()
    sleeve = pos_s * pd.Series(ret, index=close.index) - dpos * cps

    # Convert to daily by summing within UTC day
    s = pd.Series(sleeve.values,
                  index=pd.DatetimeIndex(ts.values))
    daily = s.resample("1D").sum()
    daily.index = daily.index.tz_convert("UTC") if daily.index.tz is not None else daily.index.tz_localize("UTC")
    return daily.dropna()


# ---------------------------------------------------------------------------
# Build / score / combine
# ---------------------------------------------------------------------------

PER_ASSET_VOL = (EQUITY_BASKET +
                 ["BTCUSDT", "ETHUSDT", "SOLUSDT"] +
                 ["XAU_USD", "USD_JPY", "EUR_USD", "GBP_USD"])


def family_of(name: str) -> str:
    return name.split("__")[0]


def asset_of(name: str) -> str:
    parts = name.split("__")
    return parts[-1] if len(parts) > 1 else "basket"


def build_candidates() -> dict[str, pd.Series]:
    cands: dict[str, pd.Series] = {}
    # term-structure (per asset)
    for sym in PER_ASSET_VOL:
        try:
            cands[f"T1_st_lt_ratio__{sym}"]  = t1_st_lt_vol_ratio(sym)
            cands[f"T2_vol_cone__{sym}"]     = t2_vol_cone(sym)
            cands[f"T3_vov_regime__{sym}"]   = t3_vov_regime(sym)
        except FileNotFoundError:
            continue
    # multi-leg (basket)
    cands["M4_risk_on_off__basket"]     = m4_risk_on_off()
    cands["M5_us_vs_nonus__basket"]     = m5_us_vs_nonus()
    cands["M6_gold_vs_usd__basket"]     = m6_gold_vs_usd()
    cands["M7_crypto_vs_eq__basket"]    = m7_crypto_vs_equity()
    # sessions: BTC + ETH
    cands["M8_session__BTCUSDT"]        = m8_session_btc_eth("BTCUSDT")
    cands["M8_session__ETHUSDT"]        = m8_session_btc_eth("ETHUSDT")
    return cands


def main():
    candidates = build_candidates()
    print(f"\nBuilt {len(candidates)} candidate sleeves.")

    rows = []
    scaled_streams = {}
    print(f"\n{'name':<36} {'IS_Sh':>6} {'OOS_Sh':>6} {'2022_Sh':>7} "
          f"{'2022_Ret':>8} {'OOS_Ret':>8} {'scale':>6}")
    print("-" * 96)
    for name, raw in candidates.items():
        raw = raw.dropna().sort_index()
        if len(raw) < 10:
            continue
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        scale = vol_scale_is(raw, TARGET_VOL)
        if scale == 0.0 or not np.isfinite(scale):
            continue
        scaled = raw * scale
        scaled_streams[name] = scaled
        s = stats_block(name, scaled)
        s["raw_is_vol"] = ann_vol_of(raw[raw.index < SPLIT])
        s["scale"] = float(scale)
        s["family"] = family_of(name)
        s["asset"]  = asset_of(name)
        rows.append(s)
        print(f"{name:<36} {s['IS_sharpe']:>+6.2f} {s['OOS_sharpe']:>+6.2f} "
              f"{s['2022_sharpe']:>+7.2f} {s['2022_ret']:>+8.2%} "
              f"{s['OOS_ret']:>+8.2%} {scale:>6.2f}")

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "term_spreads_breakdown.csv", index=False)

    survivors = [s["name"] for s in rows
                 if s["IS_sharpe"] >= 0.4 and s["OOS_sharpe"] >= 0.0]
    print(f"\nSurvivors (IS >= 0.4 AND OOS >= 0):  ({len(survivors)})")
    for n in survivors:
        print(f"   {n}")

    if survivors:
        aligned = pd.concat(
            [scaled_streams[n].rename(n) for n in survivors], axis=1
        ).sort_index().fillna(0)
        combined = aligned.mean(axis=1)
    else:
        ranked = sorted(rows, key=lambda r: r["IS_sharpe"], reverse=True)[:3]
        top = [r["name"] for r in ranked]
        print(f"\nNo survivors — falling back to top-3 IS: {top}")
        aligned = pd.concat(
            [scaled_streams[n].rename(n) for n in top], axis=1
        ).sort_index().fillna(0)
        combined = aligned.mean(axis=1)
        survivors = top

    scl = vol_scale_is(combined, TARGET_VOL)
    combined_scaled = combined * scl

    print(f"\nCombined sleeve (equal-weight, rescaled to 5% IS vol):")
    cs = stats_block("combined", combined_scaled)
    for tag in ["IS","OOS","2020","2021","2022","2023","2024","2025"]:
        print(f"  {tag:<6} Sh={cs[tag+'_sharpe']:+5.2f}  "
              f"Ret={cs[tag+'_ret']:+6.2%}  DD={cs[tag+'_dd']:+6.2%}")

    out_df = pd.DataFrame({
        "timestamp": combined_scaled.index,
        "ret": combined_scaled.values,
    })
    out_df.to_parquet(OUT / "term_spreads_returns.parquet", index=False)
    print(f"\nSaved combined returns: {OUT / 'term_spreads_returns.parquet'}")
    print(f"Saved breakdown:        {OUT / 'term_spreads_breakdown.csv'}")

    # Family / asset-class summary
    bk = pd.DataFrame(rows)
    print("\n=== Family summary (mean IS / OOS / 2022 Sharpe) ===")
    fam = bk.groupby("family").agg(
        n=("name", "count"),
        IS_sharpe=("IS_sharpe", "mean"),
        OOS_sharpe=("OOS_sharpe", "mean"),
        s2022=("2022_sharpe", "mean"),
    )
    print(fam.to_string())

    # Term vs spread bucket summary
    bk["bucket"] = bk["family"].map(lambda x: "term" if x.startswith("T") else "spread")
    print("\n=== Bucket summary (term-structure vs multi-leg) ===")
    bb = bk.groupby("bucket").agg(
        n=("name", "count"),
        IS_sharpe=("IS_sharpe", "mean"),
        OOS_sharpe=("OOS_sharpe", "mean"),
        s2022=("2022_sharpe", "mean"),
        OOS_ret=("OOS_ret", "mean"),
    )
    print(bb.to_string())


if __name__ == "__main__":
    main()
