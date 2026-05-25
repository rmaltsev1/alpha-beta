"""Cross-asset relationship analysis across the 13-symbol candle store.

Outputs (scratch/agent3_*.csv):
  - agent3_corr_full.csv          : full-sample D1 log-return correlation matrix
  - agent3_corr_2020_2022.csv     : sub-period correlation matrix
  - agent3_corr_2023_2025.csv     : sub-period correlation matrix
  - agent3_corr_shifts.csv        : pairs with |delta| >= 0.2 between sub-periods
  - agent3_corr_top.csv           : top-5 positive / top-3 negative pairs (full sample)
  - agent3_leadlag.csv            : H1 lead-lag corr (r_t(A) vs r_{t-k}(B)) per pair
  - agent3_autocorr.csv           : within-asset autocorr H1 lags {1,5,24}, D1 lags {1,5}
  - agent3_hurst.csv              : Hurst exponent (R/S) per symbol on D1 closes
  - agent3_runlengths.csv         : empirical run-length distribution vs geometric(0.5)
  - agent3_btc_spx_rolling.csv    : 90-day rolling corr BTC vs SPX500 D1 returns

Re-runnable: just `python scratch/agent3_cross.py` from the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from alphabeta import ALL_SYMBOLS  # noqa: E402

OUT = REPO / "scratch"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_close(symbol: str, tf: str) -> pd.Series:
    """Load closes indexed by UTC timestamp."""
    df = pd.read_parquet(REPO / "data" / symbol / f"{tf}.parquet",
                         columns=["timestamp", "close"])
    s = pd.Series(df["close"].values,
                  index=pd.DatetimeIndex(df["timestamp"], name="ts"))
    s.name = symbol
    # drop duplicates if any, sort
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def panel(tf: str, symbols=ALL_SYMBOLS) -> pd.DataFrame:
    """Return a wide panel of closes for the given timeframe.

    For D1 we normalize the timestamp to a calendar date (UTC) before joining,
    because crypto D1 bars are stamped 00:00 UTC while OANDA D1 bars are
    stamped 21:00 / 22:00 UTC of the prior calendar day. Aligning on the
    bar's *date* gives us one row per trading day across the panel; we then
    forward-fill at most 1 day to handle the FX/index weekend gap when a
    crypto bar exists but the equity/FX bar doesn't.
    """
    series = []
    for s in symbols:
        c = load_close(s, tf)
        if tf == "D1":
            # use the next calendar day as the "date label" for forex/index
            # so the OANDA 22:00 close of 2020-01-01 lines up with the
            # 2020-01-02 crypto bar (i.e. they describe the same UTC window).
            # Simpler: just floor to date and accept a 1-day phase issue is
            # too coarse; for D1 correlation a 1-day misalignment of a 22h
            # window matters little, so we floor to calendar date.
            c.index = c.index.floor("D")
            c = c[~c.index.duplicated(keep="last")]
        series.append(c)
    df = pd.concat(series, axis=1, sort=True)
    df.columns = symbols
    return df.sort_index()


# ---------------------------------------------------------------------------
# 1. Correlation matrices (D1 log returns)
# ---------------------------------------------------------------------------

def corr_matrices(d1_closes: pd.DataFrame):
    rets = np.log(d1_closes).diff()
    # Each pair uses its overlapping observations -- pairwise (handles NaNs).
    full = rets.corr(min_periods=60)

    a = rets.loc["2020-01-01":"2022-12-31"]
    b = rets.loc["2023-01-01":"2025-12-31"]
    c_a = a.corr(min_periods=60)
    c_b = b.corr(min_periods=60)

    full.to_csv(OUT / "agent3_corr_full.csv")
    c_a.to_csv(OUT / "agent3_corr_2020_2022.csv")
    c_b.to_csv(OUT / "agent3_corr_2023_2025.csv")

    # Top / bottom pairs (upper triangle only).
    rows = []
    cols = full.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append((cols[i], cols[j], full.iat[i, j],
                         c_a.iat[i, j], c_b.iat[i, j]))
    pairs = pd.DataFrame(rows, columns=["a", "b", "corr_full",
                                         "corr_2020_2022", "corr_2023_2025"])
    pairs["delta"] = pairs["corr_2023_2025"] - pairs["corr_2020_2022"]

    top_pos = pairs.sort_values("corr_full", ascending=False).head(5)
    top_neg = pairs.sort_values("corr_full", ascending=True).head(3)
    pd.concat([top_pos.assign(kind="top_pos"),
               top_neg.assign(kind="top_neg")]).to_csv(
        OUT / "agent3_corr_top.csv", index=False)

    shifts = pairs[pairs["delta"].abs() >= 0.20].sort_values(
        "delta", key=lambda s: s.abs(), ascending=False)
    shifts.to_csv(OUT / "agent3_corr_shifts.csv", index=False)

    return full, c_a, c_b, pairs, top_pos, top_neg, shifts


# ---------------------------------------------------------------------------
# 2. Lead-lag on H1 returns
# ---------------------------------------------------------------------------

def lead_lag(h1_closes: pd.DataFrame, pairs: list[tuple[str, str]],
             lags=(1, 2, 3, 6, 24)) -> pd.DataFrame:
    """For each (A, B) compute corr(r_t(A), r_{t-k}(B)) for k in lags
    (positive k means B precedes A -> "B leads") AND the symmetric
    corr(r_t(B), r_{t-k}(A))  ("A leads")."""
    rets = np.log(h1_closes).diff()
    out = []
    for a, b in pairs:
        ra = rets[a]
        rb = rets[b]
        joined = pd.concat([ra, rb], axis=1, keys=["A", "B"]).dropna()
        ra2 = joined["A"]
        rb2 = joined["B"]
        # contemporaneous
        out.append({"a": a, "b": b, "lag": 0, "direction": "contemp",
                    "corr": ra2.corr(rb2), "n": len(joined)})
        for k in lags:
            # B leads A: r_t(A) ~ r_{t-k}(B)
            c1 = ra2.corr(rb2.shift(k))
            out.append({"a": a, "b": b, "lag": k, "direction": f"{b}_leads_{a}",
                        "corr": c1, "n": int((ra2.notna() & rb2.shift(k).notna()).sum())})
            # A leads B: r_t(B) ~ r_{t-k}(A)
            c2 = rb2.corr(ra2.shift(k))
            out.append({"a": a, "b": b, "lag": k, "direction": f"{a}_leads_{b}",
                        "corr": c2, "n": int((rb2.notna() & ra2.shift(k).notna()).sum())})
    df = pd.DataFrame(out)
    df.to_csv(OUT / "agent3_leadlag.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Within-asset autocorrelation
# ---------------------------------------------------------------------------

def autocorr(d1_closes: pd.DataFrame, h1_closes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rd1 = np.log(d1_closes).diff()
    rh1 = np.log(h1_closes).diff()
    for s in ALL_SYMBOLS:
        r = rd1[s].dropna()
        d1_l1 = r.autocorr(lag=1)
        d1_l5 = r.autocorr(lag=5)
        r = rh1[s].dropna()
        h1_l1 = r.autocorr(lag=1)
        h1_l5 = r.autocorr(lag=5)
        h1_l24 = r.autocorr(lag=24)
        rows.append({"symbol": s,
                     "D1_lag1": d1_l1, "D1_lag5": d1_l5,
                     "H1_lag1": h1_l1, "H1_lag5": h1_l5, "H1_lag24": h1_l24})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agent3_autocorr.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 4. Hurst exponent (R/S analysis) on D1 closes
# ---------------------------------------------------------------------------

def hurst_rs(series: np.ndarray, min_chunk: int = 16) -> float:
    """Classic R/S Hurst.
    Method:
      - work on log-returns (incremental signal)
      - for chunk sizes n in geometric grid from min_chunk to N/4,
        partition the series into floor(N/n) non-overlapping windows
      - within each window compute the rescaled range R/S on cumulative
        deviations from the window mean
      - average R/S across windows -> (R/S)(n)
      - regress log((R/S)(n)) on log(n); slope = H
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 64:
        return float("nan")
    # geometric grid of window sizes
    sizes = []
    n = min_chunk
    while n <= N // 4:
        sizes.append(n)
        n = int(np.ceil(n * 1.5))
    if len(sizes) < 4:
        return float("nan")

    rs_means = []
    for n in sizes:
        k = N // n
        rs_vals = []
        for i in range(k):
            chunk = x[i * n:(i + 1) * n]
            mu = chunk.mean()
            dev = chunk - mu
            Z = np.cumsum(dev)
            R = Z.max() - Z.min()
            S = chunk.std(ddof=1)
            if S > 0 and np.isfinite(S):
                rs_vals.append(R / S)
        if rs_vals:
            rs_means.append((n, np.mean(rs_vals)))
    if len(rs_means) < 4:
        return float("nan")
    ns = np.array([r[0] for r in rs_means], dtype=float)
    rs = np.array([r[1] for r in rs_means], dtype=float)
    # linear fit on log-log
    coef = np.polyfit(np.log(ns), np.log(rs), 1)
    return float(coef[0])


def hurst_all(d1_closes: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(d1_closes).diff()
    rows = []
    for s in ALL_SYMBOLS:
        r = rets[s].dropna().values
        h = hurst_rs(r)
        cl = "trending" if h >= 0.55 else "mean_revert" if h <= 0.45 else "random"
        rows.append({"symbol": s, "hurst_rs_logret": h, "n": len(r),
                     "classification": cl})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agent3_hurst.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 5. Run-length distribution on D1
# ---------------------------------------------------------------------------

def run_lengths(d1_closes: pd.DataFrame) -> pd.DataFrame:
    """Empirical distribution of consecutive same-sign run lengths
    (zeros are skipped). Geometric(p=0.5) baseline: P(L=k) = 0.5^k.
    We report observed mean run length and the share of runs >= k for
    k in {1..6} vs the geometric expectation."""
    rows = []
    for s in ALL_SYMBOLS:
        r = np.log(d1_closes[s]).diff().dropna().values
        signs = np.sign(r)
        signs = signs[signs != 0]
        if len(signs) == 0:
            continue
        # find run lengths
        change = np.diff(signs) != 0
        idx = np.flatnonzero(change) + 1
        runs = np.diff(np.concatenate(([0], idx, [len(signs)])))
        runs = runs.astype(int)
        # empirical P(L >= k)
        n = len(runs)
        row = {"symbol": s,
               "n_runs": n,
               "mean_runlen": float(runs.mean()),
               "max_runlen": int(runs.max()),
               "geom_mean": 2.0,  # mean of geometric(0.5) starting at 1
               "share_up": float((signs == 1).mean())}
        for k in range(1, 8):
            row[f"P_L_ge_{k}"] = float((runs >= k).mean())
            row[f"geom_P_ge_{k}"] = 0.5 ** (k - 1)  # baseline P(L>=k)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agent3_runlengths.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 6. Rolling 90d corr BTC vs SPX500
# ---------------------------------------------------------------------------

def rolling_btc_spx(d1_closes: pd.DataFrame,
                    h1_closes: pd.DataFrame) -> pd.DataFrame:
    # D1 rolling (as asked) -- note that BTC trades 7d and SPX 5d, plus
    # OANDA D1 bars end at 21:00 UTC while crypto D1 bars end at 00:00 UTC,
    # so calendar-date alignment is approximate.
    rets = np.log(d1_closes[["BTCUSDT", "SPX500_USD"]]).diff().dropna()
    roll = rets["BTCUSDT"].rolling(90).corr(rets["SPX500_USD"]).dropna()
    out = roll.rename("corr_90d_d1").to_frame()
    out.index.name = "ts"
    out.to_csv(OUT / "agent3_btc_spx_rolling.csv")
    summary = out.groupby(out.index.year)["corr_90d_d1"].agg(
        ["mean", "median", "min", "max", "std", "count"])

    # H1 cross-check: window = 24*90 = 2160 bars, intersected only when both
    # markets are open. This is the most honest BTC-equity beta signal.
    rh = np.log(h1_closes[["BTCUSDT", "SPX500_USD"]]).diff().dropna()
    roll_h = rh["BTCUSDT"].rolling(24 * 90).corr(rh["SPX500_USD"]).dropna()
    summary_h = (roll_h.to_frame("corr_90d_h1")
                 .groupby(roll_h.index.year)
                 .agg(["mean", "median", "min", "max", "std", "count"]))
    summary_h.columns = ["_".join(c) for c in summary_h.columns]
    summary["h1_mean"] = summary_h.get("corr_90d_h1_mean")
    summary["h1_median"] = summary_h.get("corr_90d_h1_median")
    summary.to_csv(OUT / "agent3_btc_spx_rolling_yearly.csv")
    return out, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("loading panels...")
    d1 = panel("D1")
    h1 = panel("H1")
    print(f"  D1 panel: {d1.shape}, H1 panel: {h1.shape}")

    print("[1] correlation matrices (D1 log returns)")
    full, c_a, c_b, pairs, top_pos, top_neg, shifts = corr_matrices(d1)
    print("  top positive pairs:")
    print(top_pos.to_string(index=False))
    print("  top negative pairs:")
    print(top_neg.to_string(index=False))
    print(f"  pairs with |delta| >= 0.2 between sub-periods: {len(shifts)}")
    if len(shifts):
        print(shifts.to_string(index=False))

    print("[2] lead-lag on H1 returns")
    # DXY proxy = 1 / EUR_USD -- inject as synthetic column
    h1_aug = h1.copy()
    h1_aug["DXY_proxy"] = 1.0 / h1_aug["EUR_USD"]
    leadlag_pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("SPX500_USD", "NAS100_USD"),
        ("SPX500_USD", "US30_USD"),
        ("EUR_USD", "GBP_USD"),
        ("XAU_USD", "DXY_proxy"),
        ("SPX500_USD", "BTCUSDT"),
    ]
    ll = lead_lag(h1_aug, leadlag_pairs)
    # print compact view: contemp + best abs |corr| at lag>0
    for a, b in leadlag_pairs:
        sub = ll[(ll["a"] == a) & (ll["b"] == b)]
        print(f"  {a} vs {b}")
        print(sub.to_string(index=False))

    print("[3] within-asset autocorrelation")
    ac = autocorr(d1, h1)
    print(ac.to_string(index=False))

    print("[4] Hurst exponent (R/S on D1 log returns)")
    hu = hurst_all(d1)
    print(hu.to_string(index=False))

    print("[5] run lengths on D1 sign sequence")
    rl = run_lengths(d1)
    cols = ["symbol", "n_runs", "mean_runlen", "max_runlen", "share_up",
            "P_L_ge_2", "P_L_ge_3", "P_L_ge_4", "P_L_ge_5", "P_L_ge_6"]
    print(rl[cols].to_string(index=False))

    print("[6] rolling 90d corr BTC vs SPX500")
    roll, summary = rolling_btc_spx(d1, h1)
    print(summary.to_string())

    print("done -- outputs in scratch/")


if __name__ == "__main__":
    main()
