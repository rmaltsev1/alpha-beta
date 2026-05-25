"""D1 pairs mean-reversion sleeve.

For each pair (A, B):

  spread_t        = log(A_t) - beta_t * log(B_t)
  beta_t          = OLS slope from log(B[t-252..t-1]) -> log(A[t-252..t-1])
  z_t             = (spread_t - mean60(spread)[t]) / std60(spread)[t]
  signal          = -1 when z crosses +2.0 outward, +1 when z crosses -2.0 outward
                    flat when |z| crosses 0.5 inward, or |z| > 4.0 stop
  position on leg A: signal * w_A;  on leg B: -signal * beta * w_B
  w_X             = clip(target_vol / rolling_60d_vol(log_ret_X), 0, 5)

All beta / z / vol use only past data (shift(1)) so the strategy is
walk-forward at every bar.

Sleeve = equal-weight average of pair returns (the per-pair return is the
sum of net leg returns after per-side costs).

Outputs:
  scratch/quant/pairs_returns.parquet  (timestamp, ret)
  scratch/quant/pairs_breakdown.csv    (per-pair IS / OOS stats)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from alphabeta import get_candles
from alphabeta.backtest import cost_for


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BETA_WINDOW = 252       # days for rolling beta regression
Z_WINDOW    = 60        # days for spread z-score
VOL_WINDOW  = 60        # days for per-leg vol target
TARGET_VOL  = 0.05      # 5% annualised per leg
Z_ENTRY     = 2.0
Z_EXIT      = 0.5
Z_STOP      = 4.0
ANN         = 252.0     # D1 bars per year
SPLIT       = pd.Timestamp("2024-01-01", tz="UTC")

PAIRS = [
    ("BTCUSDT", "ETHUSDT"),
    ("SPX500_USD", "NAS100_USD"),
    ("SPX500_USD", "US30_USD"),
    ("EUR_USD", "GBP_USD"),
    ("XAU_USD", "EUR_USD_INV"),     # synthetic DXY = 1 / EUR_USD
]


OUT_DIR = Path(__file__).resolve().parent
RETURNS_PATH = OUT_DIR / "pairs_returns.parquet"
BREAKDOWN_PATH = OUT_DIR / "pairs_breakdown.csv"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_close(symbol: str) -> pd.Series:
    """Daily close indexed by calendar date (UTC), so crypto (00:00) and
    FX/index (22:00 prior day) can be aligned by trading date."""
    if symbol == "EUR_USD_INV":
        df = get_candles("EUR_USD", "D1")
        s = 1.0 / df["close"].astype("float64")
    else:
        df = get_candles(symbol, "D1")
        s = df["close"].astype("float64")
    # FX bars are stamped 22:00 prev-day; round to that bar's calendar date.
    # Crypto is 00:00 of the day. Normalise everything to date.
    date = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
    # FX bar stamped 2020-01-01 22:00 -> that's the close of 2020-01-01.
    # so normalise() already gives us the right date.
    s = pd.Series(s.values, index=date.values, name=symbol)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def rolling_ols_beta(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
    """Rolling OLS slope of y on x over `win` bars. Uses *past* data only:
    beta at index t is fit on x[t-win .. t-1] vs y[t-win .. t-1]."""
    # shift by 1 so bar t uses bars up to and including t-1
    y_p = y.shift(1)
    x_p = x.shift(1)
    mx = x_p.rolling(win).mean()
    my = y_p.rolling(win).mean()
    cov = (x_p * y_p).rolling(win).mean() - mx * my
    var = (x_p * x_p).rolling(win).mean() - mx * mx
    beta = cov / var
    return beta


def rolling_z(spread: pd.Series, win: int) -> pd.Series:
    """z-score using mean/std over the previous `win` bars (no current bar)."""
    sp = spread.shift(1)
    mu = sp.rolling(win).mean()
    sd = sp.rolling(win).std(ddof=0)
    return (spread - mu) / sd


def signal_from_z(z: pd.Series) -> pd.Series:
    """Mean-reversion signal:
        +1 when z is below -Z_ENTRY  (spread too cheap, expect to rise)
        -1 when z is above +Z_ENTRY  (spread too rich, expect to fall)
         0 otherwise (after |z|<Z_EXIT, or |z|>Z_STOP stop)

    Trades only on a *crossing* of an entry band (going outward) — once
    in a position, we hold until |z|<Z_EXIT or |z|>Z_STOP.
    """
    z = z.values
    n = len(z)
    sig = np.zeros(n, dtype="float64")
    pos = 0.0
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            sig[i] = pos
            continue
        if pos == 0.0:
            if zi <= -Z_ENTRY:
                pos = 1.0
            elif zi >= Z_ENTRY:
                pos = -1.0
        else:
            if abs(zi) < Z_EXIT or abs(zi) > Z_STOP:
                pos = 0.0
        sig[i] = pos
    return pd.Series(sig, index=z if False else None)


def vol_scale(log_ret: pd.Series, win: int, target: float) -> pd.Series:
    """Scale so each leg targets `target` annualised vol. Uses lagged vol
    estimated on the previous `win` bars (no look-ahead)."""
    realised = log_ret.rolling(win).std(ddof=0).shift(1) * np.sqrt(ANN)
    scale = (target / realised).clip(upper=5.0).fillna(0.0)
    return scale


# ----------------------------------------------------------------------------
# Per-pair engine
# ----------------------------------------------------------------------------
def run_pair(a: str, b: str) -> dict:
    sa = load_close(a)
    sb = load_close(b)
    df = pd.concat({"A": sa, "B": sb}, axis=1).dropna()
    df = df.sort_index()

    log_a = np.log(df["A"])
    log_b = np.log(df["B"])

    # Walk-forward beta and z
    beta = rolling_ols_beta(log_a, log_b, BETA_WINDOW)
    spread = log_a - beta * log_b
    z = rolling_z(spread, Z_WINDOW)

    sig_vals = signal_from_z(z)
    signal = pd.Series(sig_vals.values, index=df.index)

    # Bar returns (close-to-close pct change)
    ret_a = df["A"].pct_change().fillna(0.0)
    ret_b = df["B"].pct_change().fillna(0.0)
    logret_a = np.log(df["A"]).diff().fillna(0.0)
    logret_b = np.log(df["B"]).diff().fillna(0.0)

    # Vol-target each leg
    w_a = vol_scale(logret_a, VOL_WINDOW, TARGET_VOL)
    w_b = vol_scale(logret_b, VOL_WINDOW, TARGET_VOL)

    # Positions (held during bar t -> derived from signal at start of t,
    # which is built from data through t-1). Signal is already lagged-safe
    # via the shifted rolling stats.
    pos_a = signal * w_a
    pos_b = -signal * beta * w_b

    # Costs
    cost_a = cost_for(a) if a != "EUR_USD_INV" else cost_for("EUR_USD")
    cost_b = cost_for(b) if b != "EUR_USD_INV" else cost_for("EUR_USD")
    dpa = pos_a.diff().fillna(pos_a.iloc[0]).abs()
    dpb = pos_b.diff().fillna(pos_b.iloc[0]).abs()

    pnl_a = pos_a.shift(1).fillna(0.0) * ret_a - dpa * cost_a
    pnl_b = pos_b.shift(1).fillna(0.0) * ret_b - dpb * cost_b
    pair_ret = (pnl_a + pnl_b).fillna(0.0)

    # Build trade ledger for hold-time / win-rate
    trades = []
    in_pos = False
    entry_i = None
    entry_pnl_cum = None
    cum = pair_ret.cumsum()
    sig_vals = signal.values
    for i in range(len(signal)):
        s = sig_vals[i]
        if not in_pos and s != 0:
            in_pos = True
            entry_i = i
            entry_pnl_cum = cum.iloc[i - 1] if i > 0 else 0.0
        elif in_pos and s == 0:
            exit_pnl_cum = cum.iloc[i - 1] if i > 0 else 0.0
            trades.append({
                "entry_idx": entry_i,
                "exit_idx": i - 1,
                "entry_date": df.index[entry_i],
                "exit_date": df.index[i - 1],
                "hold_days": (i - 1) - entry_i + 1,
                "pnl": exit_pnl_cum - entry_pnl_cum,
            })
            in_pos = False
            entry_i = None
    if in_pos and entry_i is not None:
        exit_pnl_cum = cum.iloc[-1]
        trades.append({
            "entry_idx": entry_i,
            "exit_idx": len(signal) - 1,
            "entry_date": df.index[entry_i],
            "exit_date": df.index[-1],
            "hold_days": len(signal) - entry_i,
            "pnl": exit_pnl_cum - entry_pnl_cum,
        })
    trades_df = pd.DataFrame(trades)

    # Stats helpers
    def stats(ret: pd.Series) -> dict:
        if len(ret) == 0 or ret.std(ddof=0) == 0:
            return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
        ann_ret = ret.mean() * ANN
        ann_vol = ret.std(ddof=0) * np.sqrt(ANN)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        eq = (1.0 + ret).cumprod()
        dd = (eq / eq.cummax() - 1.0).min()
        return {"sharpe": float(sharpe), "ann_ret": float(ann_ret),
                "ann_vol": float(ann_vol), "max_dd": float(dd)}

    # IS / OOS split by index date
    idx = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index
    if df.index.tz is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
        pair_ret.index = df.index
        signal.index = df.index
        beta.index = df.index
        z.index = df.index
    is_mask = df.index < SPLIT
    oos_mask = df.index >= SPLIT
    full_stats = stats(pair_ret)
    is_stats = stats(pair_ret[is_mask])
    oos_stats = stats(pair_ret[oos_mask])

    # Trade-level stats split by entry date
    def trade_stats(td: pd.DataFrame) -> dict:
        if td.empty:
            return {"n": 0, "avg_hold": 0.0, "hit_rate": 0.0}
        return {
            "n": int(len(td)),
            "avg_hold": float(td["hold_days"].mean()),
            "hit_rate": float((td["pnl"] > 0).mean()),
        }
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=["entry_date", "exit_date", "hold_days", "pnl"])
    else:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"], utc=True)
    is_trades = trades_df[trades_df["entry_date"] < SPLIT] if not trades_df.empty else trades_df
    oos_trades = trades_df[trades_df["entry_date"] >= SPLIT] if not trades_df.empty else trades_df

    return {
        "name": f"{a}-{b}",
        "a": a, "b": b,
        "ret": pair_ret,
        "signal": signal,
        "beta": beta,
        "z": z,
        "spread": spread,
        "full": full_stats,
        "is": is_stats,
        "oos": oos_stats,
        "trades": trades_df,
        "is_trades": trade_stats(is_trades),
        "oos_trades": trade_stats(oos_trades),
        "all_trades": trade_stats(trades_df),
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for a, b in PAIRS:
        print(f"running {a} vs {b} ...")
        r = run_pair(a, b)
        print(
            f"  FULL Sh={r['full']['sharpe']:+.2f} ret={r['full']['ann_ret']:+.1%} "
            f"vol={r['full']['ann_vol']:.1%} dd={r['full']['max_dd']:+.1%} "
            f"trades={r['all_trades']['n']}  hold={r['all_trades']['avg_hold']:.1f}d  "
            f"hit={r['all_trades']['hit_rate']:.0%}"
        )
        print(
            f"   IS  Sh={r['is']['sharpe']:+.2f} ret={r['is']['ann_ret']:+.1%}   "
            f"OOS Sh={r['oos']['sharpe']:+.2f} ret={r['oos']['ann_ret']:+.1%}"
        )
        results.append(r)

    # ----------- Aggregate sleeve -----------
    # Align all pair-return series on union of dates
    aligned = pd.concat({r["name"]: r["ret"] for r in results}, axis=1).sort_index()
    aligned = aligned.fillna(0.0)
    # Equal weight across the 5 pairs
    sleeve = aligned.mean(axis=1)
    sleeve.name = "ret"

    # Save returns parquet
    out = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(RETURNS_PATH, index=False)
    print(f"wrote {RETURNS_PATH}")

    # Sleeve stats
    def sleeve_stats(ret: pd.Series) -> dict:
        if len(ret) == 0:
            return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
        ann_ret = ret.mean() * ANN
        ann_vol = ret.std(ddof=0) * np.sqrt(ANN)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        eq = (1.0 + ret).cumprod()
        dd = (eq / eq.cummax() - 1.0).min()
        return {"sharpe": float(sharpe), "ann_ret": float(ann_ret),
                "ann_vol": float(ann_vol), "max_dd": float(dd),
                "tot_ret": float(eq.iloc[-1] - 1.0)}

    full = sleeve_stats(sleeve)
    is_s = sleeve_stats(sleeve[sleeve.index < SPLIT])
    oos_s = sleeve_stats(sleeve[sleeve.index >= SPLIT])
    print()
    print("=" * 70)
    print("SLEEVE (equal-weight 5 pairs):")
    print(f"  FULL Sharpe={full['sharpe']:+.2f}  ret={full['ann_ret']:+.1%}  "
          f"vol={full['ann_vol']:.1%}  DD={full['max_dd']:+.1%}  "
          f"tot={full['tot_ret']:+.1%}")
    print(f"   IS  Sharpe={is_s['sharpe']:+.2f}  ret={is_s['ann_ret']:+.1%}  "
          f"DD={is_s['max_dd']:+.1%}")
    print(f"   OOS Sharpe={oos_s['sharpe']:+.2f}  ret={oos_s['ann_ret']:+.1%}  "
          f"DD={oos_s['max_dd']:+.1%}")

    # ----------- Per-pair breakdown CSV -----------
    rows = []
    for r in results:
        rows.append({
            "pair": r["name"],
            "sharpe_full": r["full"]["sharpe"],
            "sharpe_is": r["is"]["sharpe"],
            "sharpe_oos": r["oos"]["sharpe"],
            "ann_ret_full": r["full"]["ann_ret"],
            "ann_ret_is": r["is"]["ann_ret"],
            "ann_ret_oos": r["oos"]["ann_ret"],
            "max_dd_full": r["full"]["max_dd"],
            "n_trades_full": r["all_trades"]["n"],
            "n_trades_is": r["is_trades"]["n"],
            "n_trades_oos": r["oos_trades"]["n"],
            "avg_hold_days": r["all_trades"]["avg_hold"],
            "hit_rate_full": r["all_trades"]["hit_rate"],
            "hit_rate_is": r["is_trades"]["hit_rate"],
            "hit_rate_oos": r["oos_trades"]["hit_rate"],
        })
    rows.append({
        "pair": "SLEEVE",
        "sharpe_full": full["sharpe"],
        "sharpe_is": is_s["sharpe"],
        "sharpe_oos": oos_s["sharpe"],
        "ann_ret_full": full["ann_ret"],
        "ann_ret_is": is_s["ann_ret"],
        "ann_ret_oos": oos_s["ann_ret"],
        "max_dd_full": full["max_dd"],
        "n_trades_full": sum(r["all_trades"]["n"] for r in results),
        "n_trades_is": sum(r["is_trades"]["n"] for r in results),
        "n_trades_oos": sum(r["oos_trades"]["n"] for r in results),
        "avg_hold_days": float(np.mean([r["all_trades"]["avg_hold"] for r in results if r["all_trades"]["n"] > 0])) if any(r["all_trades"]["n"] > 0 for r in results) else 0.0,
        "hit_rate_full": float(np.mean([r["all_trades"]["hit_rate"] for r in results if r["all_trades"]["n"] > 0])) if any(r["all_trades"]["n"] > 0 for r in results) else 0.0,
        "hit_rate_is": float(np.mean([r["is_trades"]["hit_rate"] for r in results if r["is_trades"]["n"] > 0])) if any(r["is_trades"]["n"] > 0 for r in results) else 0.0,
        "hit_rate_oos": float(np.mean([r["oos_trades"]["hit_rate"] for r in results if r["oos_trades"]["n"] > 0])) if any(r["oos_trades"]["n"] > 0 for r in results) else 0.0,
    })
    bd = pd.DataFrame(rows)
    bd.to_csv(BREAKDOWN_PATH, index=False)
    print(f"wrote {BREAKDOWN_PATH}")

    # ----------- Regime diagnostics: BTC-ETH beta path -----------
    print()
    print("BTC-ETH regime check:")
    btc_eth = next(r for r in results if r["name"] == "BTCUSDT-ETHUSDT")
    beta = btc_eth["beta"].dropna()
    if len(beta):
        for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            mask = beta.index.year == yr
            sub = beta[mask]
            if len(sub):
                print(f"  {yr}: beta avg={sub.mean():+.2f} std={sub.std():.2f} "
                      f"min={sub.min():+.2f} max={sub.max():+.2f}")
    z = btc_eth["z"].dropna()
    if len(z):
        for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            mask = z.index.year == yr
            sub = z[mask]
            if len(sub):
                print(f"  {yr}: z avg={sub.mean():+.2f} std={sub.std():.2f}  "
                      f"|z|>2 frac={(sub.abs() > 2).mean():.0%}")


if __name__ == "__main__":
    main()
