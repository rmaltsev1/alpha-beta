"""Carry / structural-drift sleeve — wave 3.

Builds five long-bias sleeves that try to harvest the equity risk premium,
carry-trade premium, and crypto secular drift, all under walk-forward
vol-management:

  1. Vol-managed equity beta: 1/3 SPX + 1/3 NAS + 1/3 US30, sized to 10% vol.
  2. Carry USD_JPY long, vol-managed, gated by "not stressed" filter.
  3. Crypto secular long: 1/2 BTC + 1/2 ETH, vol-managed to 15%/asset,
     filtered by trailing 252d > 0.
  4. Multi-asset drift risk-parity: 1/3 equity basket + 1/3 carry FX +
     1/3 crypto basket, inverse-vol within and across, 10% portfolio vol.
  5. All-weather: 40/20/20/20 of equity / carry / crypto / XAU long,
     monthly-rebalanced, vol-targeted to 10%.

Methodology
-----------
- Returns are computed at the native bar via the alphabeta.backtest engine
  (cost per side handled by the engine using the symbol's asset class).
- Vol management is walk-forward: realized vol uses only the trailing window
  visible at the *start* of each bar (we shift by 1).
- Sub-sleeves are scaled to 5% IS annualized vol (IS-only scaling factor
  applied uniformly, including OOS, to keep the test honest).
- Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0. Survivors combined equal-weight.

Outputs
-------
  scratch/wave3/carry_drift.py                  (this file, re-runnable)
  scratch/wave3/carry_drift_returns.parquet     (timestamp UTC, per-sleeve cols)
  scratch/wave3/carry_drift_breakdown.csv       (per-sleeve stats)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles, ALL_SYMBOLS  # noqa: E402
from alphabeta.backtest import backtest, cost_for  # noqa: E402

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
SUB_TARGET_VOL = 0.05      # each surviving sub-sleeve scaled to 5% IS ann vol
EQ_TARGET = 0.10           # equity sleeve internal vol target
JPY_TARGET = 0.10          # carry sleeve internal vol target
CRYPTO_TARGET = 0.15       # crypto sleeve internal vol target (per asset)
DRIFT_TARGET = 0.10        # risk-parity drift target
AW_TARGET = 0.10           # all-weather target
VOL_WIN = 30               # realized-vol window (days)
LEV_LO, LEV_HI = 0.3, 2.0  # vol-management leverage cap
RET_LB = 252               # filter trailing-return lookback

EQUITY_SYMS = ["SPX500_USD", "NAS100_USD", "US30_USD"]
CRYPTO_SYMS = ["BTCUSDT", "ETHUSDT"]


# --- helpers ---------------------------------------------------------------

def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    out = {"label": label}
    r = pd.Series(r).dropna()
    if len(r) < 5:
        return out
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    windows = [
        ("FULL", np.ones(len(idx), dtype=bool)),
        ("IS", np.asarray(idx < SPLIT)),
        ("OOS", np.asarray(idx >= SPLIT)),
        ("Y2022", np.asarray(
            (idx >= pd.Timestamp("2022-01-01", tz="UTC"))
            & (idx < pd.Timestamp("2023-01-01", tz="UTC"))
        )),
    ]
    for tag, mask in windows:
        sub = r[mask]
        if len(sub) < 5:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            out[f"{tag}_vol"] = 0.0
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar / av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
    return out


def scale_to_is_vol(rets: pd.Series, target: float = SUB_TARGET_VOL) -> float:
    is_r = rets[rets.index < SPLIT].dropna()
    if len(is_r) < 30:
        return 0.0
    bpy = _bpy(is_r.index)
    av = float(is_r.std(ddof=0) * np.sqrt(bpy))
    return target / av if av > 1e-9 else 0.0


def _utc_normalize(df):
    """Return df with a normalized UTC date index from its `timestamp` col."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out.index = ts.dt.normalize()
    out.index.name = "date"
    return out


def vol_manage(pos_base: pd.Series, df: pd.DataFrame, target_vol: float,
               win: int = VOL_WIN, cap_lo: float = LEV_LO,
               cap_hi: float = LEV_HI) -> pd.Series:
    """Multiply `pos_base` by walk-forward leverage = target / realized_vol.

    Realized vol is computed from yesterday's close-to-close returns over
    `win` bars (shifted by 1, so observable at bar start).
    """
    close = df["close"].astype("float64")
    bar_ret = close.pct_change()
    bpy = _bpy(pd.to_datetime(df["timestamp"], utc=True))
    realized = bar_ret.rolling(win).std() * np.sqrt(bpy)
    realized = realized.shift(1)  # info only from prior bar
    lev = (target_vol / realized).clip(lower=cap_lo, upper=cap_hi)
    lev = lev.fillna(cap_lo)
    return (pos_base.values * lev.values).clip(-cap_hi, cap_hi)


def run_position(df: pd.DataFrame, pos: np.ndarray | pd.Series, *,
                 name: str, symbol: str):
    """Backtest a position array against a df and return (rets_native, res)."""
    p = pd.Series(np.asarray(pos), index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, p, symbol=symbol, timeframe="D1", name=name)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    return rets, res


def combine_streams(streams: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight combine per-symbol return streams onto a UTC date index."""
    if not streams:
        return pd.Series(dtype="float64")
    aligned = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    if aligned.index.tz is None:
        aligned.index = aligned.index.tz_localize("UTC")
    return aligned.mean(axis=1)


def realized_vol_panel(df: pd.DataFrame, win: int = VOL_WIN) -> pd.Series:
    """Daily realized vol panel keyed by UTC date (for filter checks)."""
    close = df["close"].astype("float64")
    bar_ret = close.pct_change()
    bpy = _bpy(pd.to_datetime(df["timestamp"], utc=True))
    rv = bar_ret.rolling(win).std() * np.sqrt(bpy)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    out = pd.Series(rv.values, index=pd.DatetimeIndex(ts))
    return out.groupby(out.index).last()


# --- data load -------------------------------------------------------------

print("Loading D1 data...")
NEEDED = list(set(EQUITY_SYMS + CRYPTO_SYMS + ["USD_JPY", "XAU_USD"]))
DATA = {s: get_candles(s, "D1") for s in NEEDED}
for s, df in DATA.items():
    print(f"  {s:12} {len(df):5}  "
          f"{df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


# ---------------------------------------------------------------------------
# Sleeve 1 — Vol-managed equity beta
# ---------------------------------------------------------------------------

def sleeve_equity_beta():
    print("\n=== Sleeve 1: Vol-managed equity beta ===")
    streams = {}
    for sym in EQUITY_SYMS:
        df = DATA[sym]
        base = pd.Series(1.0, index=df.index)        # always long
        pos = vol_manage(base, df, target_vol=EQ_TARGET)
        rets, _ = run_position(df, pos, name=f"EQVOL_{sym}", symbol=sym)
        streams[sym] = rets
        s = stats(sym, rets)
        print(f"  {sym:<12} IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  "
              f"Y2022_Sh={s.get('Y2022_sharpe',0):+.2f}")
    return combine_streams(streams)


# ---------------------------------------------------------------------------
# Sleeve 2 — Carry USD_JPY long
# ---------------------------------------------------------------------------

def sleeve_carry_jpy():
    print("\n=== Sleeve 2: Carry USD_JPY long ===")
    sym = "USD_JPY"
    df = DATA[sym]

    close = df["close"].astype("float64")
    ret_252 = close.pct_change(RET_LB).shift(1)               # walk-forward
    bar_ret = close.pct_change()
    bpy = _bpy(pd.to_datetime(df["timestamp"], utc=True))
    rv30 = (bar_ret.rolling(VOL_WIN).std() * np.sqrt(bpy)).shift(1)

    # IS realized-vol 80th percentile — walk-forward (only past-IS info)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    is_mask = ts < SPLIT
    is_rv = rv30[is_mask].dropna()
    p80 = float(np.quantile(is_rv, 0.80)) if len(is_rv) > 30 else np.inf
    print(f"  IS 30d realized-vol 80th pct = {p80:.4f}")

    filt = (ret_252 > 0) & (rv30 < p80)
    filt = filt.fillna(False)

    base = pd.Series(np.where(filt.values, 1.0, 0.0), index=df.index)
    pos = vol_manage(base, df, target_vol=JPY_TARGET)
    rets, _ = run_position(df, pos, name="CARRY_USDJPY", symbol=sym)
    s = stats(sym, rets)
    print(f"  USD_JPY      IS_Sh={s.get('IS_sharpe',0):+.2f}  "
          f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  "
          f"Y2022_Sh={s.get('Y2022_sharpe',0):+.2f}  "
          f"exposure={float(filt.mean()):.2%}")
    return rets


# ---------------------------------------------------------------------------
# Sleeve 3 — Crypto secular long
# ---------------------------------------------------------------------------

def sleeve_crypto_drift():
    print("\n=== Sleeve 3: Crypto secular long ===")
    streams = {}
    for sym in CRYPTO_SYMS:
        df = DATA[sym]
        close = df["close"].astype("float64")
        ret_252 = close.pct_change(RET_LB).shift(1)
        filt = (ret_252 > 0).fillna(False)
        base = pd.Series(np.where(filt.values, 1.0, 0.0), index=df.index)
        pos = vol_manage(base, df, target_vol=CRYPTO_TARGET)
        rets, _ = run_position(df, pos, name=f"CRYPTODRIFT_{sym}", symbol=sym)
        streams[sym] = rets
        s = stats(sym, rets)
        print(f"  {sym:<8}     IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  "
              f"Y2022_Sh={s.get('Y2022_sharpe',0):+.2f}  "
              f"exposure={float(filt.mean()):.2%}")
    return combine_streams(streams)


# ---------------------------------------------------------------------------
# Sleeve 4 — Multi-asset risk-parity drift
# ---------------------------------------------------------------------------

def sleeve_riskparity_drift():
    print("\n=== Sleeve 4: Multi-asset risk-parity drift ===")
    # Per-symbol native vol-managed long position, but at TARGET=10% so each
    # sub-stream is comparable. Then equal-weight across baskets, then within
    # each basket inverse-vol weight.
    sub_streams = {}
    sub_rv = {}                          # IS-realized vol of each sub stream

    # equity basket
    eq_streams = {}
    for sym in EQUITY_SYMS:
        df = DATA[sym]
        base = pd.Series(1.0, index=df.index)
        pos = vol_manage(base, df, target_vol=DRIFT_TARGET)
        rets, _ = run_position(df, pos, name=f"DRIFT_EQ_{sym}", symbol=sym)
        eq_streams[sym] = rets

    # crypto basket
    cr_streams = {}
    for sym in CRYPTO_SYMS:
        df = DATA[sym]
        base = pd.Series(1.0, index=df.index)
        pos = vol_manage(base, df, target_vol=DRIFT_TARGET)
        rets, _ = run_position(df, pos, name=f"DRIFT_CR_{sym}", symbol=sym)
        cr_streams[sym] = rets

    # carry-FX basket = USD_JPY long with same gate as sleeve 2 (no extra logic
    # for "basket"; the carry sleeve has only one ticker in this universe).
    sym = "USD_JPY"
    df = DATA[sym]
    close = df["close"].astype("float64")
    ret_252 = close.pct_change(RET_LB).shift(1)
    bar_ret = close.pct_change()
    bpy = _bpy(pd.to_datetime(df["timestamp"], utc=True))
    rv30 = (bar_ret.rolling(VOL_WIN).std() * np.sqrt(bpy)).shift(1)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    is_rv = rv30[ts < SPLIT].dropna()
    p80 = float(np.quantile(is_rv, 0.80)) if len(is_rv) > 30 else np.inf
    filt = ((ret_252 > 0) & (rv30 < p80)).fillna(False)
    base = pd.Series(np.where(filt.values, 1.0, 0.0), index=df.index)
    pos = vol_manage(base, df, target_vol=DRIFT_TARGET)
    carry_rets, _ = run_position(df, pos, name="DRIFT_CARRY_USDJPY", symbol=sym)

    # Inverse-vol weights within each basket using IS realized vol
    def inv_vol_combine(streams: dict[str, pd.Series]) -> pd.Series:
        is_vols = {}
        for k, r in streams.items():
            r_is = r[r.index < SPLIT].dropna()
            bpy_ = _bpy(r_is.index) if len(r_is) > 5 else 252
            v = r_is.std(ddof=0) * np.sqrt(bpy_) if len(r_is) > 5 else 0
            is_vols[k] = float(v) if v > 1e-9 else 1.0
        inv = {k: 1.0 / v for k, v in is_vols.items()}
        w = {k: x / sum(inv.values()) for k, x in inv.items()}
        df_ = pd.concat(streams, axis=1, sort=True).fillna(0.0)
        return (df_ * pd.Series(w)).sum(axis=1)

    eq_basket = inv_vol_combine(eq_streams)
    cr_basket = inv_vol_combine(cr_streams)

    # Equal-weight across baskets
    panel = pd.concat({
        "equity": eq_basket,
        "carry": carry_rets,
        "crypto": cr_basket,
    }, axis=1, sort=True).fillna(0.0)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    # cross-basket inverse-vol weighting on IS
    is_panel = panel[panel.index < SPLIT]
    bpy_p = _bpy(is_panel.index)
    vols = is_panel.std(ddof=0) * np.sqrt(bpy_p)
    inv = 1.0 / vols.replace(0, np.nan).fillna(1.0)
    w = (inv / inv.sum()).values
    print(f"  basket weights: equity={w[0]:.2f}, carry={w[1]:.2f}, crypto={w[2]:.2f}")
    combined = (panel.values * w).sum(axis=1)
    rets = pd.Series(combined, index=panel.index)

    # Portfolio-level vol overlay to 10% (walk-forward, using past 30d)
    rolling_vol = rets.rolling(VOL_WIN).std() * np.sqrt(252)
    lev = (DRIFT_TARGET / rolling_vol.shift(1)).clip(lower=LEV_LO, upper=LEV_HI)
    lev = lev.fillna(LEV_LO)
    rets_managed = rets * lev

    s = stats("RP_DRIFT", rets_managed)
    print(f"  RP_DRIFT     IS_Sh={s.get('IS_sharpe',0):+.2f}  "
          f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  "
          f"Y2022_Sh={s.get('Y2022_sharpe',0):+.2f}")
    return rets_managed


# ---------------------------------------------------------------------------
# Sleeve 5 — All-weather
# ---------------------------------------------------------------------------

def sleeve_all_weather(eq_rets: pd.Series, carry_rets: pd.Series,
                       crypto_rets: pd.Series):
    print("\n=== Sleeve 5: All-weather (40/20/20/20) ===")
    # XAU long, vol-managed at 10% (defensive sleeve)
    sym = "XAU_USD"
    df = DATA[sym]
    base = pd.Series(1.0, index=df.index)
    pos = vol_manage(base, df, target_vol=AW_TARGET)
    xau_rets, _ = run_position(df, pos, name="AW_XAU", symbol=sym)

    panel = pd.concat({
        "eq": eq_rets,
        "carry": carry_rets,
        "crypto": crypto_rets,
        "xau": xau_rets,
    }, axis=1, sort=True).fillna(0.0)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")

    # 40 / 20 / 20 / 20 weights; rebalanced monthly (compute monthly aggregate,
    # then re-distribute? Easier: weights are fixed, and "monthly rebalanced"
    # in this implementation means the dollar-weights reset at month start. For
    # daily return-stream blending of already-vol-managed sub-sleeves, fixed
    # weights are the right approximation — there is no compounding drift to
    # correct since each daily return is already vol-scaled).
    w = np.array([0.40, 0.20, 0.20, 0.20])
    combined = (panel.values * w).sum(axis=1)
    rets = pd.Series(combined, index=panel.index)

    # Portfolio-level vol overlay to 10%
    rv = rets.rolling(VOL_WIN).std() * np.sqrt(252)
    lev = (AW_TARGET / rv.shift(1)).clip(lower=LEV_LO, upper=LEV_HI)
    lev = lev.fillna(LEV_LO)
    rets_managed = rets * lev

    s = stats("ALL_WEATHER", rets_managed)
    print(f"  AW           IS_Sh={s.get('IS_sharpe',0):+.2f}  "
          f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  "
          f"Y2022_Sh={s.get('Y2022_sharpe',0):+.2f}")
    return rets_managed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    eq = sleeve_equity_beta()
    carry = sleeve_carry_jpy()
    crypto = sleeve_crypto_drift()
    drift = sleeve_riskparity_drift()
    aw = sleeve_all_weather(eq, carry, crypto)

    raw_sleeves = {
        "EQUITY_BETA": eq,
        "CARRY_USDJPY": carry,
        "CRYPTO_DRIFT": crypto,
        "RP_DRIFT": drift,
        "ALL_WEATHER": aw,
    }

    # Vol-scale each sub-sleeve to 5% IS ann vol (uniformly applied)
    scaled = {}
    breakdown_rows = []
    for name, r in raw_sleeves.items():
        scale = scale_to_is_vol(r, SUB_TARGET_VOL)
        sc = r * scale
        scaled[name] = sc
        s = stats(name, sc)
        s["scale"] = scale
        s["sleeve"] = name
        breakdown_rows.append(s)
        print(f"  scale[{name:<13}] = {scale:.3f}")

    breakdown = pd.DataFrame(breakdown_rows)
    cols = ["sleeve"] + [c for c in breakdown.columns
                        if c not in ("sleeve", "label")]
    breakdown = breakdown[cols]
    breakdown.to_csv(OUT / "carry_drift_breakdown.csv", index=False)

    # Survivor filter
    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.5)
                          & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) ===")
    if not survivors.empty:
        print(survivors[["sleeve", "IS_sharpe", "OOS_sharpe",
                         "Y2022_sharpe", "FULL_sharpe",
                         "FULL_ret"]].to_string(index=False))
    else:
        print("(no sleeves passed both gates)")

    surv_names = survivors["sleeve"].tolist()
    panel_streams = {n: scaled[n] for n in surv_names}

    # Save per-sleeve returns parquet — full panel of scaled sleeve streams
    panel_df = pd.concat(scaled, axis=1, sort=True).fillna(0.0)
    if panel_df.index.tz is None:
        panel_df.index = panel_df.index.tz_localize("UTC")
    # add a 'survivors_mean' column = equal-weight blend of survivors (or NaN
    # if no survivors)
    if surv_names:
        panel_df["survivors_mean"] = panel_df[surv_names].mean(axis=1)
    else:
        panel_df["survivors_mean"] = 0.0
    out_df = panel_df.reset_index().rename(columns={"index": "timestamp",
                                                    "date": "timestamp"})
    out_df.to_parquet(OUT / "carry_drift_returns.parquet", index=False)

    # Combined survivor sleeve stats
    if surv_names:
        combined = panel_df[surv_names].mean(axis=1)
    else:
        combined = panel_df["survivors_mean"]
    sc = stats("CARRY_DRIFT_COMBINED", combined)
    print("\n=== Combined survivor sleeve (equal-weight) ===")
    for tag in ["FULL", "IS", "OOS", "Y2022"]:
        sh = sc.get(f"{tag}_sharpe", 0)
        rt = sc.get(f"{tag}_ret", 0)
        vv = sc.get(f"{tag}_vol", 0)
        print(f"  {tag:<6}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  AnnVol={vv:.2%}")

    # Yearly Sharpe of combined
    print("\n=== Combined yearly Sharpe ===")
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 20:
            continue
        bpy = _bpy(sub.index)
        std = sub.std(ddof=0)
        sh = (sub.mean() * bpy) / (std * np.sqrt(bpy)) if std > 0 else 0.0
        rt = sub.mean() * bpy
        print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")

    # Correlation with existing RISKPAR sleeve
    try:
        rp_path = ROOT / "scratch" / "quant" / "risk_parity_returns.parquet"
        rp = pd.read_parquet(rp_path)
        rp_idx = pd.to_datetime(rp["timestamp"], utc=True).dt.normalize()
        rp_s = pd.Series(rp["ret"].values, index=pd.DatetimeIndex(rp_idx))
        joined = pd.concat({"this": combined, "RISKPAR": rp_s},
                           axis=1, sort=True).dropna()
        print(f"\n=== Correlation vs existing RISKPAR (n={len(joined)}) ===")
        for tag, mask in [
            ("FULL", np.ones(len(joined), dtype=bool)),
            ("IS",   np.asarray(joined.index < SPLIT)),
            ("OOS",  np.asarray(joined.index >= SPLIT)),
            ("Y2022", np.asarray(
                (joined.index >= pd.Timestamp("2022-01-01", tz="UTC"))
                & (joined.index < pd.Timestamp("2023-01-01", tz="UTC")))),
        ]:
            sub = joined[mask]
            if len(sub) < 20:
                continue
            cp = float(sub["this"].corr(sub["RISKPAR"]))
            print(f"  {tag:<6}  pearson={cp:+.3f}  n={len(sub)}")
    except FileNotFoundError:
        print("\n(no existing RISKPAR parquet to compare)")


if __name__ == "__main__":
    main()
