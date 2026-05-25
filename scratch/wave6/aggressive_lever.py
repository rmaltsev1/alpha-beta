"""
Aggressive leverage scenarios on the v15 portfolio base return stream.

Goal: push monthly mean return to +6-8 %/month while keeping MaxDD <= 12 %.

Input: scratch/quant/PRODUCTION_FINAL_v15.parquet (overlay-applied final v15
returns).  We treat that stream as the "unleveraged" base alpha and apply
dynamic leverage on top of it.

All leverage rules are walk-forward (use only data up to t-1).
Maximum gross leverage allowed: 25x.
IS  : dates <  2024-01-01 (used for percentile calibration only)
OOS : dates >= 2024-01-01

Outputs:
    scratch/wave6/aggressive_lever_variants.csv
    scratch/wave6/aggressive_lever_returns.parquet
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO = Path('/Users/rinatmaltsev/Documents/Python Projects/alpha-beta/alpha-beta')
SRC_PARQUET = REPO / 'scratch' / 'quant' / 'PRODUCTION_FINAL_v15.parquet'
OUT_DIR = REPO / 'scratch' / 'wave6'
OUT_VARIANTS = OUT_DIR / 'aggressive_lever_variants.csv'
OUT_RETURNS = OUT_DIR / 'aggressive_lever_returns.parquet'

IS_END = pd.Timestamp('2024-01-01', tz='UTC')
ANN = 365.0  # daily-stream convention used elsewhere in repo
MAX_LEV = 25.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_base() -> pd.Series:
    p = pd.read_parquet(SRC_PARQUET)
    p['timestamp'] = pd.to_datetime(p['timestamp'], utc=True)
    s = p.set_index('timestamp')['ret'].astype(float).sort_index()
    s = s[~s.index.duplicated(keep='first')]
    return s


def annualised(r: pd.Series) -> dict:
    if r.std() == 0 or r.empty:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(
        ann_ret=float(r.mean() * ANN),
        ann_vol=float(r.std() * np.sqrt(ANN)),
        sharpe=float(r.mean() / r.std() * np.sqrt(ANN)),
        maxdd=float(dd),
    )


def monthly_stats(r: pd.Series) -> dict:
    if r.empty:
        return dict(m_mean=np.nan, m_p5=np.nan, m_p50=np.nan)
    m = (1 + r).resample('ME').prod() - 1
    return dict(
        m_mean=float(m.mean()),
        m_p5=float(m.quantile(0.05)),
        m_p50=float(m.median()),
    )


def stress_2022(r: pd.Series) -> dict:
    sub = r[(r.index >= '2022-01-01') & (r.index < '2023-01-01')]
    if sub.empty:
        return dict(stress2022_ret=np.nan, stress2022_dd=np.nan)
    eq = (1 + sub).cumprod()
    return dict(
        stress2022_ret=float(eq.iloc[-1] - 1),
        stress2022_dd=float((eq / eq.cummax() - 1).min()),
    )


def lever_stats(lev: pd.Series) -> dict:
    return dict(lev_mean=float(lev.mean()), lev_max=float(lev.max()))


def summarise(name: str, r_lev: pd.Series, lev: pd.Series) -> dict:
    out = {'variant': name}
    oos = r_lev[r_lev.index >= IS_END]
    out.update({f'oos_{k}': v for k, v in annualised(oos).items()})
    out.update({f'oos_{k}': v for k, v in monthly_stats(oos).items()})
    out.update(stress_2022(r_lev))
    out.update(lever_stats(lev))
    # Full-sample for reference
    full = annualised(r_lev)
    out['full_ann_ret'] = full['ann_ret']
    out['full_maxdd'] = full['maxdd']
    out['full_sharpe'] = full['sharpe']
    return out


# ---------------------------------------------------------------------------
# Walk-forward primitives
# ---------------------------------------------------------------------------
WARMUP_DAYS = 90  # need this much history before any leverage is applied


def trailing_realised_vol(r: pd.Series, win: int) -> pd.Series:
    """Vol of r over the last `win` days, lagged 1 day (use up to t-1).

    Returns NaN before `WARMUP_DAYS` days of history have accumulated, which
    forces leverage to 0 (or stay flat at 1) during the warm-up window.
    """
    rv = r.rolling(win, min_periods=max(5, win // 2)).std().shift(1) * np.sqrt(ANN)
    # Force NaN until WARMUP_DAYS days of data
    rv.iloc[:WARMUP_DAYS] = np.nan
    return rv


def trailing_drawdown_pct(r: pd.Series) -> pd.Series:
    """Current drawdown from running peak of the *unleveraged* base, lagged 1d.

    We use the base equity here so that the leverage rule itself does not
    create a feedback loop on its own past leverage.  This is conservative.
    """
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return dd.shift(1)


def cap_lev(lev: pd.Series) -> pd.Series:
    return lev.clip(lower=0.0, upper=MAX_LEV)


def apply_lev(base: pd.Series, lev: pd.Series) -> pd.Series:
    lev = cap_lev(lev.reindex(base.index).fillna(0.0))
    return base * lev


def base_vol_target_lev(base: pd.Series, target_vol: float, win: int = 30) -> pd.Series:
    rv = trailing_realised_vol(base, win)
    lev = (target_vol / rv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cap_lev(lev)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def variant_1_stepped_dd(base: pd.Series) -> pd.Series:
    """Stepped vol-target with hard DD stop.

    Target 25% vol normally.  DD>5% -> 12%, DD>8% -> 8%.  When DD recovers
    below 2% restore to 25%.  We update the target each day using the lagged
    base drawdown.
    """
    dd = trailing_drawdown_pct(base)
    # Build target with hysteresis-ish bands using only lagged DD signal.
    target = pd.Series(0.25, index=base.index)
    target[dd < -0.05] = 0.12
    target[dd < -0.08] = 0.08
    # "When DD recovers below 2%, restore" -> already 25% in default branch
    # because dd > -0.02 means dd >= -0.02, default 25% applies.
    rv = trailing_realised_vol(base, 30)
    lev = (target / rv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cap_lev(lev)


def variant_2_sharpe_ramp(base: pd.Series) -> pd.Series:
    """Vol-target ramps based on trailing-3m Sharpe.

    Sharpe > +4  -> target 30%; Sharpe < +1 -> target 10%; linear in between.
    """
    win = 63  # ~3 months trading days, but data is daily 365-cal => use 90
    win = 90
    mu = base.rolling(win, min_periods=30).mean()
    sd = base.rolling(win, min_periods=30).std()
    sh = (mu / sd * np.sqrt(ANN)).shift(1)
    target = ((sh - 1.0) / (4.0 - 1.0)).clip(0.0, 1.0)
    target = 0.10 + target * (0.30 - 0.10)
    target = target.fillna(0.10)  # default low until enough data
    rv = trailing_realised_vol(base, 30)
    lev = (target / rv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cap_lev(lev)


def variant_3_equity_slope(base: pd.Series) -> pd.Series:
    """Equity-curve-aware leverage.

    eq slope (30d) > 0 AND DD < 1% -> lever to 27% vol target.  When slope
    flattens (slope <= 0) drop to 18%.
    """
    eq = (1 + base).cumprod()
    slope = (eq - eq.shift(30)) / 30
    slope = slope.shift(1)
    dd = trailing_drawdown_pct(base)
    target = pd.Series(0.18, index=base.index)
    mask = (slope > 0) & (dd > -0.01)
    target[mask] = 0.27
    rv = trailing_realised_vol(base, 30)
    lev = (target / rv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cap_lev(lev)


def variant_4_floor_ceiling(base: pd.Series) -> pd.Series:
    """Hard floor + soft ceiling.

    Floor: target 25% vol always.  Ceiling: if 5d realised portfolio vol >
    30% (post-leverage), halve gross.
    """
    rv30 = trailing_realised_vol(base, 30)
    lev_base = (0.25 / rv30).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lev_base = cap_lev(lev_base)
    # Iteratively realise then halve when post-lev 5d vol > 30%.
    r_lev = base * lev_base
    rv5_post = r_lev.rolling(5, min_periods=3).std().shift(1) * np.sqrt(ANN)
    halve = (rv5_post > 0.30).fillna(False)
    lev = lev_base.copy()
    lev[halve] = lev_base[halve] * 0.5
    return cap_lev(lev)


def variant_5_spx_regime(base: pd.Series) -> pd.Series:
    """Step-up leverage in low-vol SPX regimes.

    SPX 30d RV < IS 25th pctile -> target 30%; > IS 75th pctile -> 12%; else
    22% (linear ramp between 25th and 75th to keep continuous).
    """
    spx = pd.read_parquet(REPO / 'data' / 'SPX500_USD' / 'D1.parquet')
    spx['timestamp'] = pd.to_datetime(spx['timestamp'], utc=True)
    spx = spx.set_index('timestamp')['close'].sort_index()
    spx_ret = np.log(spx).diff()
    spx_rv = spx_ret.rolling(30, min_periods=15).std().shift(1) * np.sqrt(252)
    spx_rv = spx_rv.reindex(base.index, method='ffill')
    # Calibrate percentiles on IS slice only.
    is_rv = spx_rv[spx_rv.index < IS_END].dropna()
    q25 = is_rv.quantile(0.25)
    q75 = is_rv.quantile(0.75)
    # Map rv -> target: low rv -> 0.30, high rv -> 0.12, linear in between.
    target = ((spx_rv - q25) / (q75 - q25)).clip(0.0, 1.0)
    target = 0.30 - target * (0.30 - 0.12)
    target = target.fillna(0.22)
    rv = trailing_realised_vol(base, 30)
    lev = (target / rv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cap_lev(lev)


def variant_6_volofvol_filter(base: pd.Series) -> pd.Series:
    """22% vol target with re-targeting only when 30d vol changes >= 10%."""
    rv = trailing_realised_vol(base, 30)
    lev_raw = (0.22 / rv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lev_raw = cap_lev(lev_raw)
    out = lev_raw.copy()
    last = np.nan
    last_rv = np.nan
    vals = []
    for ts, x in zip(rv.index, rv.values):
        if np.isnan(x) or x == 0:
            vals.append(0.0 if np.isnan(last) else last)
            continue
        if np.isnan(last):
            new = min(0.22 / x, MAX_LEV)
            last = new
            last_rv = x
            vals.append(new)
            continue
        change = abs(x - last_rv) / last_rv if last_rv else np.inf
        if change >= 0.10:
            new = min(0.22 / x, MAX_LEV)
            last = new
            last_rv = x
        vals.append(last)
    out = pd.Series(vals, index=rv.index)
    return cap_lev(out)


def variant_7_asym_bands(base: pd.Series) -> pd.Series:
    """Asymmetric DD bands.

    Target 25% vol baseline.  Upside: tolerate up to +5% deviation (i.e. lever
    can drift up to 1.05x of target leverage without re-flexing).  Downside:
    1% tolerance (cut quickly).  Mechanically: compute desired lev from rv
    each day; current lev only adjusts down immediately if desired < 0.99 x
    current, and adjusts up only if desired > 1.05 x current.
    """
    rv = trailing_realised_vol(base, 30)
    desired = (0.25 / rv).replace([np.inf, -np.inf], np.nan)
    out = []
    cur = np.nan
    for x in desired.values:
        if np.isnan(x):
            out.append(0.0 if np.isnan(cur) else cur)
            continue
        if np.isnan(cur):
            cur = min(x, MAX_LEV)
        else:
            if x < cur * 0.99:  # cut fast on downside
                cur = min(x, MAX_LEV)
            elif x > cur * 1.05:  # only step up after >5% improvement
                cur = min(x, MAX_LEV)
        out.append(cur)
    return cap_lev(pd.Series(out, index=desired.index))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()

    variants = {
        'V1_stepped_DD': variant_1_stepped_dd,
        'V2_sharpe_ramp': variant_2_sharpe_ramp,
        'V3_equity_slope': variant_3_equity_slope,
        'V4_floor_ceiling': variant_4_floor_ceiling,
        'V5_spx_regime': variant_5_spx_regime,
        'V6_volofvol_filter': variant_6_volofvol_filter,
        'V7_asym_bands': variant_7_asym_bands,
    }

    rows = []
    returns_out = pd.DataFrame(index=base.index)
    returns_out['base_v15'] = base
    lev_out = pd.DataFrame(index=base.index)

    for name, fn in variants.items():
        lev = fn(base).reindex(base.index)
        # Warm-up: leverage = 1 (unleveraged base) until we have enough history
        lev.iloc[:WARMUP_DAYS] = 1.0
        lev = cap_lev(lev.fillna(1.0))
        r_lev = base * lev
        returns_out[name] = r_lev
        lev_out[name] = lev
        rows.append(summarise(name, r_lev, lev))

    # Also include the unleveraged baseline (lev=1) for reference.
    rows.append(summarise('V0_base_v15_lev1', base, pd.Series(1.0, index=base.index)))
    returns_out['V0_base_v15_lev1'] = base

    df = pd.DataFrame(rows)
    # Sort by OOS monthly mean (descending)
    df = df.sort_values('oos_m_mean', ascending=False).reset_index(drop=True)

    # Pretty columns
    cols = [
        'variant',
        'oos_ann_ret', 'oos_ann_vol', 'oos_sharpe', 'oos_maxdd',
        'oos_m_mean', 'oos_m_p5', 'oos_m_p50',
        'stress2022_ret', 'stress2022_dd',
        'lev_mean', 'lev_max',
        'full_ann_ret', 'full_maxdd', 'full_sharpe',
    ]
    df = df[cols]
    df.to_csv(OUT_VARIANTS, index=False, float_format='%.6f')

    returns_out.to_parquet(OUT_RETURNS)

    print('Saved:', OUT_VARIANTS)
    print('Saved:', OUT_RETURNS)
    print()
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
