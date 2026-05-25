"""Agent 1: return distribution + volatility structure across 13 symbols.

Run from repo root with the venv active:
    python scratch/agent1_vol.py

Outputs:
    scratch/agent1_distribution_D1.csv
    scratch/agent1_distribution_H1.csv
    scratch/agent1_tails_D1.csv
    scratch/agent1_tails_H1.csv
    scratch/agent1_clustering_H1.csv
    scratch/agent1_regime_D1.csv
    scratch/agent1_hurst_D1.csv
    scratch/agent1_summary.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import ALL_SYMBOLS, CRYPTO, get_candles


SCRATCH = Path(__file__).resolve().parent
OUT_DIR = SCRATCH

# Annualization factors.
# - Crypto: 365 days/yr; D1 has 365 bars/yr, H1 has 365*24 bars/yr.
# - Forex/Index: 252 trading days/yr; D1 has 252 bars/yr, H1 has 252*24 bars/yr.
def ann_factors(symbol: str) -> tuple[float, float]:
    """Return (bars_per_year_D1, bars_per_year_H1)."""
    if symbol in CRYPTO:
        return 365.0, 365.0 * 24.0
    return 252.0, 252.0 * 24.0


def log_returns(close: pd.Series) -> pd.Series:
    r = np.log(close).diff().dropna()
    # guard against zero/negative prices producing inf
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def distribution_stats(r: pd.Series, bars_per_year: float) -> dict:
    n = len(r)
    mean = r.mean()
    std = r.std(ddof=1)
    skew = r.skew()
    # pandas .kurt() already returns excess (Fisher) kurtosis
    ex_kurt = r.kurt()
    ann_mean = mean * bars_per_year
    ann_vol = std * math.sqrt(bars_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
    return dict(
        n=n,
        ann_mean=ann_mean,
        ann_vol=ann_vol,
        skew=skew,
        ex_kurt=ex_kurt,
        sharpe=sharpe,
    )


def tail_stats(r: pd.Series) -> dict:
    absr = r.abs()
    q99 = absr.quantile(0.99)
    med = absr.median()
    ratio = q99 / med if med > 0 else np.nan
    std = r.std(ddof=1)
    n_5sig = int((absr > 5 * std).sum()) if std > 0 else 0
    drawup = r.max()
    drawdown = r.min()
    return dict(
        q99_over_med=ratio,
        n_5sigma=n_5sig,
        max_drawup=drawup,
        max_drawdown=drawdown,
    )


def abs_return_acf(r: pd.Series, lags: list[int]) -> dict:
    absr = r.abs().reset_index(drop=True)
    out = {}
    for L in lags:
        if len(absr) <= L + 1:
            out[f"acf_abs_lag{L}"] = np.nan
            continue
        a = absr.iloc[L:].reset_index(drop=True)
        b = absr.iloc[:-L].reset_index(drop=True)
        out[f"acf_abs_lag{L}"] = float(a.corr(b))
    return out


def rolling_realized_vol_ar1(r_d1: pd.Series, window: int = 30) -> dict:
    """AR(1) coefficient of a `window`-day rolling realized vol on D1 returns."""
    rv = r_d1.rolling(window).std(ddof=1).dropna()
    if len(rv) < window + 5:
        return dict(ar1_rv30=np.nan, rv30_mean=np.nan, rv30_std=np.nan)
    x = rv.shift(1).dropna()
    y = rv.loc[x.index]
    # simple OLS slope = cov/var
    xv = x.values
    yv = y.values
    xc = xv - xv.mean()
    yc = yv - yv.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom > 0 else np.nan
    return dict(
        ar1_rv30=slope,
        rv30_mean=float(rv.mean()),
        rv30_std=float(rv.std(ddof=1)),
    )


def hurst_rs(series: pd.Series) -> float:
    """Hurst exponent via classic R/S analysis on cumulative deviations.

    series: a 1D pandas Series (we pass |log returns| for the tail/vol-clustering
    interpretation requested).
    """
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < 100:
        return np.nan
    # window sizes spaced log-uniformly
    min_w = 16
    max_w = n // 4
    if max_w <= min_w:
        return np.nan
    sizes = np.unique(np.logspace(np.log10(min_w), np.log10(max_w), num=20).astype(int))
    sizes = sizes[sizes >= min_w]
    rs_vals = []
    use_sizes = []
    for w in sizes:
        # split into non-overlapping windows of size w
        n_chunks = n // w
        if n_chunks < 2:
            continue
        chunks = x[: n_chunks * w].reshape(n_chunks, w)
        rs_chunk = []
        for c in chunks:
            mean = c.mean()
            dev = c - mean
            cum = np.cumsum(dev)
            R = cum.max() - cum.min()
            S = c.std(ddof=1)
            if S > 0 and R > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
            use_sizes.append(w)
    if len(use_sizes) < 4:
        return np.nan
    log_w = np.log(use_sizes)
    log_rs = np.log(rs_vals)
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(slope)


def analyse_symbol(symbol: str) -> dict:
    bpy_d1, bpy_h1 = ann_factors(symbol)
    d1 = get_candles(symbol, "D1").sort_values("timestamp").reset_index(drop=True)
    h1 = get_candles(symbol, "H1").sort_values("timestamp").reset_index(drop=True)
    r_d1 = log_returns(d1["close"])
    r_h1 = log_returns(h1["close"])

    out = {"symbol": symbol}

    # 1) Distribution stats
    ds_d1 = distribution_stats(r_d1, bpy_d1)
    ds_h1 = distribution_stats(r_h1, bpy_h1)
    out.update({f"D1_{k}": v for k, v in ds_d1.items()})
    out.update({f"H1_{k}": v for k, v in ds_h1.items()})

    # 2) Tails
    t_d1 = tail_stats(r_d1)
    t_h1 = tail_stats(r_h1)
    out.update({f"D1_{k}": v for k, v in t_d1.items()})
    out.update({f"H1_{k}": v for k, v in t_h1.items()})

    # 3) Vol-clustering: ACF of |r| on H1 at lags 1, 5, 24, 168
    acf_h1 = abs_return_acf(r_h1, [1, 5, 24, 168])
    out.update({f"H1_{k}": v for k, v in acf_h1.items()})

    # 4) Vol regime persistence: AR(1) of 30-day rolling vol on D1
    rv = rolling_realized_vol_ar1(r_d1, window=30)
    out.update({f"D1_{k}": v for k, v in rv.items()})

    # 5) Hurst exponent of |r| on D1
    H = hurst_rs(r_d1.abs())
    out["D1_hurst_abs_RS"] = H

    return out


def main() -> None:
    rows = []
    for sym in ALL_SYMBOLS:
        print(f"[{sym}] computing…", flush=True)
        rows.append(analyse_symbol(sym))
    df = pd.DataFrame(rows).set_index("symbol")

    # Headline summary
    summary_cols = [
        "D1_n", "D1_ann_mean", "D1_ann_vol", "D1_skew", "D1_ex_kurt", "D1_sharpe",
        "H1_n", "H1_ann_vol", "H1_ex_kurt", "H1_sharpe",
        "D1_q99_over_med", "D1_n_5sigma", "D1_max_drawup", "D1_max_drawdown",
        "H1_q99_over_med", "H1_n_5sigma",
        "H1_acf_abs_lag1", "H1_acf_abs_lag5", "H1_acf_abs_lag24", "H1_acf_abs_lag168",
        "D1_ar1_rv30", "D1_hurst_abs_RS",
    ]
    df[summary_cols].to_csv(OUT_DIR / "agent1_summary.csv")

    # Sub-tables
    dist_d1_cols = ["D1_n", "D1_ann_mean", "D1_ann_vol", "D1_skew", "D1_ex_kurt", "D1_sharpe"]
    dist_h1_cols = ["H1_n", "H1_ann_mean", "H1_ann_vol", "H1_skew", "H1_ex_kurt", "H1_sharpe"]
    tail_d1_cols = ["D1_q99_over_med", "D1_n_5sigma", "D1_max_drawup", "D1_max_drawdown"]
    tail_h1_cols = ["H1_q99_over_med", "H1_n_5sigma", "H1_max_drawup", "H1_max_drawdown"]
    cluster_cols = ["H1_acf_abs_lag1", "H1_acf_abs_lag5", "H1_acf_abs_lag24", "H1_acf_abs_lag168"]
    regime_cols = ["D1_ar1_rv30", "D1_rv30_mean", "D1_rv30_std"]
    hurst_cols = ["D1_hurst_abs_RS"]

    df[dist_d1_cols].to_csv(OUT_DIR / "agent1_distribution_D1.csv")
    df[dist_h1_cols].to_csv(OUT_DIR / "agent1_distribution_H1.csv")
    df[tail_d1_cols].to_csv(OUT_DIR / "agent1_tails_D1.csv")
    df[tail_h1_cols].to_csv(OUT_DIR / "agent1_tails_H1.csv")
    df[cluster_cols].to_csv(OUT_DIR / "agent1_clustering_H1.csv")
    df[regime_cols].to_csv(OUT_DIR / "agent1_regime_D1.csv")
    df[hurst_cols].to_csv(OUT_DIR / "agent1_hurst_D1.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print("\n=== summary ===")
    print(df[summary_cols])


if __name__ == "__main__":
    main()
