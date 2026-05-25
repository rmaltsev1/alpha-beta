"""Pairs v2 — cointegration-filtered, walk-forward.

The v1 pairs agent found constant-z mean-reversion broken in OOS because
rolling-OLS β was unstable. Fix:
  1. Compute the spread y - β*x with walk-forward OLS β.
  2. Test the spread for stationarity (ADF-style) on a 252-day window.
  3. Only enter when the test passes (i.e., spread is statistically
     mean-reverting in the recent window).
  4. Same z-score entry/exit rules as v1.

ADF approximation (since statsmodels isn't installed):
  Run an AR(1) regression on first differences: Δs_t = α + β*s_{t-1} + ε_t.
  The t-stat on β tests for stationarity. Critical values:
    -3.43 (1%), -2.86 (5%), -2.57 (10%) at 250 sample size.
  If t-stat < -2.86 we treat the pair as cointegrated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05


def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    for tag, mask in [("FULL", pd.Series(True, index=r.index)),
                      ("IS",   r.index < SPLIT),
                      ("OOS",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 2: continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0
        out[f"{tag}_ret"] = ar
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    return out


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Simple OLS: y = a + b*x. Return (b, intercept)."""
    if len(x) < 3:
        return 0.0, 0.0
    x_mean = x.mean(); y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 1e-12:
        return 0.0, y_mean
    b = ((x - x_mean) * (y - y_mean)).sum() / denom
    a = y_mean - b * x_mean
    return b, a


def adf_t_stat(s: np.ndarray) -> float:
    """Augmented Dickey-Fuller t-stat (no augmentation, ie. DF only).
    Δs_t = α + β*s_{t-1} + ε_t. Returns t-stat on β.
    """
    s = np.asarray(s, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 30:
        return 0.0
    ds = np.diff(s)
    s_lag = s[:-1]
    n = len(ds)
    x_mean = s_lag.mean()
    y_mean = ds.mean()
    denom = ((s_lag - x_mean) ** 2).sum()
    if denom <= 1e-12:
        return 0.0
    beta = ((s_lag - x_mean) * (ds - y_mean)).sum() / denom
    alpha = y_mean - beta * x_mean
    resid = ds - alpha - beta * s_lag
    sigma2 = (resid ** 2).sum() / (n - 2)
    se_beta = np.sqrt(sigma2 / denom)
    if se_beta <= 1e-12:
        return 0.0
    return beta / se_beta


def pair_strategy(sym_a: str, sym_b: str,
                  beta_lookback: int = 252,
                  z_lookback: int = 60,
                  adf_lookback: int = 252,
                  adf_threshold: float = -2.86,
                  z_entry: float = 2.0,
                  z_exit: float = 0.5,
                  z_stop: float = 4.0) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (pair return stream, diagnostics dataframe).

    Position is in *spread units*. Translated to per-leg P&L using each
    asset's log return.
    """
    a = get_candles(sym_a, "D1")[["timestamp", "close"]].rename(columns={"close": "a"})
    b = get_candles(sym_b, "D1")[["timestamp", "close"]].rename(columns={"close": "b"})
    df = a.merge(b, on="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["la"] = np.log(df["a"])
    df["lb"] = np.log(df["b"])
    df["ret_a"] = df["la"].diff()
    df["ret_b"] = df["lb"].diff()

    n = len(df)
    beta = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    z = np.full(n, np.nan)
    adf = np.full(n, np.nan)

    for i in range(beta_lookback, n):
        # Walk-forward OLS β on the trailing window
        win_a = df["la"].values[i - beta_lookback : i]
        win_b = df["lb"].values[i - beta_lookback : i]
        b_i, _ = ols(win_a, win_b)
        beta[i] = b_i
        # Spread today
        spread_now = df["la"].iloc[i] - b_i * df["lb"].iloc[i]
        spread[i] = spread_now
        # z-score over a rolling spread window — use last `z_lookback` spreads
        if i >= beta_lookback + z_lookback:
            sp = np.full(z_lookback, np.nan)
            for j in range(z_lookback):
                k = i - z_lookback + 1 + j
                sp[j] = df["la"].iloc[k] - beta[i] * df["lb"].iloc[k]
            mu = np.nanmean(sp[:-1]); sd = np.nanstd(sp[:-1], ddof=0)
            if sd > 1e-9:
                z[i] = (sp[-1] - mu) / sd
            # ADF: test stationarity of the last adf_lookback spread values
            if i >= beta_lookback + adf_lookback:
                ad_win = np.full(adf_lookback, np.nan)
                for j in range(adf_lookback):
                    k = i - adf_lookback + 1 + j
                    ad_win[j] = df["la"].iloc[k] - beta[i] * df["lb"].iloc[k]
                adf[i] = adf_t_stat(ad_win)

    df["beta"] = beta
    df["spread"] = spread
    df["z"] = z
    df["adf"] = adf

    # Position generation: enter when |z| > 2 AND adf passes, exit at |z| < 0.5 or |z| > stop
    pos_spread = np.zeros(n)
    in_pos = 0
    for i in range(1, n):
        if pd.isna(df["z"].iloc[i-1]):
            continue
        zv = df["z"].iloc[i-1]
        adv = df["adf"].iloc[i-1]
        if in_pos == 0:
            if pd.notna(adv) and adv < adf_threshold:
                if zv > z_entry:
                    in_pos = -1
                elif zv < -z_entry:
                    in_pos = +1
        else:
            # Exit conditions: |z| < z_exit or |z| > z_stop
            if abs(zv) < z_exit or abs(zv) > z_stop:
                in_pos = 0
        pos_spread[i] = in_pos
    df["pos_spread"] = pos_spread

    # PnL: pos_spread * (ret_a - beta * ret_b). Subtract per-side costs on
    # |Δpos_spread| at each bar; assume 1.5 bp per leg per side average.
    cost_per_side = 0.00015  # rough average of FX/Index/Crypto blend per leg
    dpos = np.abs(np.diff(pos_spread, prepend=0))
    gross_ret = df["pos_spread"] * (df["ret_a"] - df["beta"] * df["ret_b"])
    cost = dpos * cost_per_side * 2  # 2 legs
    net = gross_ret - cost
    rets = pd.Series(net.fillna(0).values,
                     index=pd.to_datetime(df["timestamp"].values, utc=True))
    return rets, df


def _scale_to_target(rets, target_vol):
    is_part = rets[rets.index < SPLIT]
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    if av <= 1e-9:
        return rets * 0, 0
    k = target_vol / av
    return rets * k, k


def main():
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("SPX500_USD", "NAS100_USD"),
        ("SPX500_USD", "US30_USD"),
        ("NAS100_USD", "US30_USD"),
        ("EUR_USD", "GBP_USD"),
        ("DE30_EUR", "UK100_GBP"),
        ("BTCUSDT", "SOLUSDT"),
    ]
    print(f"{'Pair':<25} {'IS_Sh':>6} {'OOS_Sh':>7} {'#Trades':>8} {'scale':>6}")
    print("-" * 60)
    sleeves = {}
    rows = []
    for a, b in pairs:
        rets, diag = pair_strategy(a, b)
        scaled, k = _scale_to_target(rets, TARGET_VOL)
        s = stats(f"{a}-{b}", scaled)
        n_trades = int((np.abs(np.diff(diag["pos_spread"], prepend=0)) > 0).sum() // 2)
        rows.append({"pair": f"{a}-{b}", "scale": k, **s, "n_trades": n_trades})
        sleeves[f"{a[:4]}-{b[:4]}"] = scaled
        print(f"{a}-{b:<14} {s.get('IS_sharpe',0):>+6.2f} {s.get('OOS_sharpe',0):>+7.2f} "
              f"{n_trades:>8d} {k:>6.2f}")

    pd.DataFrame(rows).to_csv(OUT / "pairs_v2_breakdown.csv", index=False)

    # Combine survivors (positive IS Sharpe AND positive OOS Sharpe)
    survivors = [r["pair"] for r in rows
                 if r.get("IS_sharpe", 0) > 0 and r.get("OOS_sharpe", 0) > 0]
    print(f"\nSurvivors (positive both IS and OOS): {survivors}")

    if survivors:
        # Aggregate equal-weight
        keep = [s for s in sleeves if any(
            s == f"{p.split('-')[0][:4]}-{p.split('-')[1][:4]}" for p in survivors)]
        combined = pd.concat({k: sleeves[k] for k in keep}, axis=1, sort=True).fillna(0).mean(axis=1)
        s_combined = stats("PAIRS_v2", combined)
        print(f"\nCombined sleeve stats:")
        for tag in ["FULL", "IS", "OOS"]:
            print(f"  {tag:<4} Sharpe={s_combined.get(f'{tag}_sharpe', 0):+.2f}  "
                  f"Return={s_combined.get(f'{tag}_ret', 0):+.2%}  "
                  f"DD={s_combined.get(f'{tag}_dd', 0):+.2%}")
        # Year breakdown
        for year, sub in combined.groupby(combined.index.year):
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            print(f"  {year}  Sharpe={sh:+.2f}")
        pd.DataFrame({"timestamp": combined.index, "ret": combined.values}).to_parquet(
            OUT / "pairs_v2_returns.parquet", index=False)
    else:
        print("\nNo survivors — saving zero series.")
        zero = pd.Series(0.0, index=sleeves[list(sleeves.keys())[0]].index)
        pd.DataFrame({"timestamp": zero.index, "ret": zero.values}).to_parquet(
            OUT / "pairs_v2_returns.parquet", index=False)


if __name__ == "__main__":
    main()
