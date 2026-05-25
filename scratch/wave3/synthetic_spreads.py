"""Synthetic-basket pair trades (wave3).

Builds 5 candidate "spread" trades, each pitting a multi-asset *synthetic
basket* against a single asset (or against another synthetic basket). Unlike
classic single-symbol pairs (pairs_v2.py), here at least one leg is a
weighted log-price aggregate.

Synthetic baskets (log-price space):
  DXY proxy        = -0.58 log(EUR_USD) - 0.12 log(GBP_USD) + 0.14 log(USD_JPY)
  EU equity        = mean(log(UK100_GBP), log(DE30_EUR))
  US equity        = mean(log(SPX500_USD), log(NAS100_USD), log(US30_USD))
  Crypto mcap      = 0.55 log(BTCUSDT) + 0.30 log(ETHUSDT) + 0.15 log(SOLUSDT)
  Defensive (XAU+JPY) = 0.5 log(XAU_USD) + 0.5 log(USD_JPY)

Trades:
  1. XAU_USD vs DXY proxy
  2. US equity vs EU equity
  3. Crypto mcap vs SPX500_USD (with risk-on correlation filter)
  4. JP225_USD vs USD_JPY
  5. Defensive (XAU+JPY) vs US equity, gated by VIX-proxy (30d SPX realized vol top decile)

Methodology mirrors pairs_v2:
  - Walk-forward OLS β (252d), spread = ya - β * yb
  - 60d rolling z-score
  - DF (Dickey-Fuller) t-stat on 252d spread window; threshold -2.86
  - Entry |z| > 2.0, exit |z| < 0.5, stop |z| > 4.0
  - IS < 2024-01-01, OOS >= 2024-01-01
  - Survivor filter: IS Sharpe >= 0.3 AND OOS Sharpe >= 0
  - Vol-scale survivors to 5% IS ann vol; combine equal-weight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles, SYMBOL_TYPE
from alphabeta.backtest import DEFAULT_COSTS_BPS

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
TRADING_DAYS = 252

# Per-side cost lookup (fraction)
COST_FRAC = {sym: DEFAULT_COSTS_BPS[atype] / 10_000.0
             for sym, atype in SYMBOL_TYPE.items()}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return TRADING_DAYS
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else TRADING_DAYS


def split_stats(r: pd.Series) -> dict:
    """Return IS / OOS / FULL / 2022 sharpe + dd / vol."""
    r = r.dropna()
    out = {}
    for tag, mask in [("full", pd.Series(True, index=r.index)),
                      ("is",   r.index <  SPLIT),
                      ("oos",  r.index >= SPLIT)]:
        sub = r[mask]
        if len(sub) < 5:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            out[f"{tag}_vol"] = 0.0
            out[f"{tag}_dd"] = 0.0
            continue
        bpy = _bpy(sub.index)
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        ar = float(sub.mean()) * bpy
        out[f"{tag}_sharpe"] = ar / av if av > 1e-12 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
        eq = (1 + sub).cumprod()
        out[f"{tag}_dd"] = float((eq / eq.cummax() - 1).min())
    y22 = r[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
            (r.index <  pd.Timestamp("2023-01-01", tz="UTC"))]
    if len(y22) > 5:
        bpy = _bpy(y22.index)
        av = float(y22.std(ddof=0)) * np.sqrt(bpy)
        ar = float(y22.mean()) * bpy
        out["y2022_sharpe"] = ar / av if av > 1e-12 else 0.0
    else:
        out["y2022_sharpe"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------
def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
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
    """DF (no augmentation) t-stat. Critical at 5% ≈ -2.86 for n≈250."""
    s = np.asarray(s, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 30:
        return 0.0
    ds = np.diff(s)
    s_lag = s[:-1]
    n = len(ds)
    xm = s_lag.mean(); ym = ds.mean()
    denom = ((s_lag - xm) ** 2).sum()
    if denom <= 1e-12:
        return 0.0
    beta = ((s_lag - xm) * (ds - ym)).sum() / denom
    alpha = ym - beta * xm
    resid = ds - alpha - beta * s_lag
    sigma2 = (resid ** 2).sum() / max(n - 2, 1)
    se = np.sqrt(sigma2 / denom)
    if se <= 1e-12:
        return 0.0
    return beta / se


# ---------------------------------------------------------------------------
# Load all D1 closes onto a common timestamp grid (log space)
# ---------------------------------------------------------------------------
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD",
           "SPX500_USD", "NAS100_USD", "US30_USD",
           "UK100_GBP", "DE30_EUR", "JP225_USD",
           "BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _load_frame(symbols: list[str] | None = None) -> pd.DataFrame:
    """Wide-format log-close frame on common UTC daily index (inner join over
    the supplied subset). When `symbols` is None, returns the full join across
    all 13 symbols (which is bottlenecked on SOLUSDT, ~Aug 2020 →).
    """
    if symbols is None:
        symbols = SYMBOLS
    closes = {}
    for sym in symbols:
        df = get_candles(sym, "D1")[["timestamp", "close"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
        closes[sym] = np.log(df["close"].astype(float))
    out = pd.concat(closes, axis=1, sort=True).sort_index()
    # Inner-join across the requested symbols (drop weekends where FX/Index nan).
    out = out.dropna(how="any")
    return out


# ---------------------------------------------------------------------------
# Build synthetic baskets (in log-price space)
# ---------------------------------------------------------------------------
def build_baskets(L: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of synthetic log-price 'levels' aligned to L.index."""
    B = pd.DataFrame(index=L.index)
    # DXY proxy: positive when USD strong. Note JPY weight is +0.14 because
    # USD_JPY moves up when USD strengthens.
    B["DXY_proxy"]  = (-0.58 * L["EUR_USD"]
                       - 0.12 * L["GBP_USD"]
                       + 0.14 * L["USD_JPY"])
    B["EU_equity"]  = 0.5 * (L["UK100_GBP"] + L["DE30_EUR"])
    B["US_equity"]  = (L["SPX500_USD"] + L["NAS100_USD"] + L["US30_USD"]) / 3.0
    B["CRYPTO_mcap"] = (0.55 * L["BTCUSDT"]
                        + 0.30 * L["ETHUSDT"]
                        + 0.15 * L["SOLUSDT"])
    B["DEFEND"] = 0.5 * (L["XAU_USD"] + L["USD_JPY"])
    return B


def build_baskets_for(L: pd.DataFrame) -> pd.DataFrame:
    """Build only the baskets whose constituents are present in L."""
    B = pd.DataFrame(index=L.index)
    cols = set(L.columns)
    if {"EUR_USD", "GBP_USD", "USD_JPY"} <= cols:
        B["DXY_proxy"] = (-0.58 * L["EUR_USD"]
                          - 0.12 * L["GBP_USD"]
                          + 0.14 * L["USD_JPY"])
    if {"UK100_GBP", "DE30_EUR"} <= cols:
        B["EU_equity"] = 0.5 * (L["UK100_GBP"] + L["DE30_EUR"])
    if {"SPX500_USD", "NAS100_USD", "US30_USD"} <= cols:
        B["US_equity"] = (L["SPX500_USD"] + L["NAS100_USD"] + L["US30_USD"]) / 3.0
    if {"BTCUSDT", "ETHUSDT", "SOLUSDT"} <= cols:
        B["CRYPTO_mcap"] = (0.55 * L["BTCUSDT"]
                            + 0.30 * L["ETHUSDT"]
                            + 0.15 * L["SOLUSDT"])
    if {"XAU_USD", "USD_JPY"} <= cols:
        B["DEFEND"] = 0.5 * (L["XAU_USD"] + L["USD_JPY"])
    return B


# ---------------------------------------------------------------------------
# Cost lookup for legs (handles single-asset OR synthetic baskets)
# ---------------------------------------------------------------------------
LEG_WEIGHTS = {
    # leg name -> dict of {symbol: |weight|}, used for per-side cost calc.
    # The cost charged on a unit |Δpos| in spread units = sum_i |w_i| * cost_i.
    "DXY_proxy":   {"EUR_USD": 0.58, "GBP_USD": 0.12, "USD_JPY": 0.14},
    "EU_equity":   {"UK100_GBP": 0.5, "DE30_EUR": 0.5},
    "US_equity":   {"SPX500_USD": 1/3, "NAS100_USD": 1/3, "US30_USD": 1/3},
    "CRYPTO_mcap": {"BTCUSDT": 0.55, "ETHUSDT": 0.30, "SOLUSDT": 0.15},
    "DEFEND":      {"XAU_USD": 0.5, "USD_JPY": 0.5},
    # Single-leg passthroughs:
    "XAU_USD":     {"XAU_USD": 1.0},
    "SPX500_USD":  {"SPX500_USD": 1.0},
    "USD_JPY":     {"USD_JPY": 1.0},
    "JP225_USD":   {"JP225_USD": 1.0},
}


def leg_cost(leg: str) -> float:
    """Per-unit-notional cost-per-side for a leg's basket."""
    return sum(abs(w) * COST_FRAC[s] for s, w in LEG_WEIGHTS[leg].items())


# ---------------------------------------------------------------------------
# Core spread-trade engine (walk-forward β, z, ADF; pos rules)
# ---------------------------------------------------------------------------
def run_spread(
    name: str,
    a: pd.Series,       # log-level series for leg A (target)
    b: pd.Series,       # log-level series for leg B (hedge)
    leg_a: str,
    leg_b: str,
    *,
    beta_lookback: int = 252,
    z_lookback: int = 60,
    adf_lookback: int = 252,
    adf_threshold: float = -2.86,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    extra_gate: pd.Series | None = None,   # optional bool series; only enter when True
    extra_gate_pos_only: int | None = None,  # if set, only allow positions of this sign while gated
) -> tuple[pd.Series, pd.DataFrame]:
    """Generic spread trade. Returns (daily-return-series, diag-df).
    Daily returns are in spread units, net of leg costs on |Δpos|.
    """
    df = pd.DataFrame({"la": a, "lb": b}).dropna().copy()
    df["ret_a"] = df["la"].diff()
    df["ret_b"] = df["lb"].diff()
    n = len(df)
    la = df["la"].values
    lb = df["lb"].values

    beta = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    z = np.full(n, np.nan)
    adf = np.full(n, np.nan)

    for i in range(beta_lookback, n):
        # Walk-forward OLS β on trailing window: la = α + β * lb + ε
        win_a = la[i - beta_lookback:i]
        win_b = lb[i - beta_lookback:i]
        b_i, _ = ols(win_a, win_b)
        beta[i] = b_i
        spread[i] = la[i] - b_i * lb[i]
        if i >= beta_lookback + z_lookback:
            # Spread series over last z_lookback days (using current β)
            ks = np.arange(i - z_lookback + 1, i + 1)
            sp = la[ks] - b_i * lb[ks]
            mu = sp[:-1].mean()
            sd = sp[:-1].std(ddof=0)
            if sd > 1e-9:
                z[i] = (sp[-1] - mu) / sd
            if i >= beta_lookback + adf_lookback:
                ks2 = np.arange(i - adf_lookback + 1, i + 1)
                ad_win = la[ks2] - b_i * lb[ks2]
                adf[i] = adf_t_stat(ad_win)

    df["beta"] = beta
    df["spread"] = spread
    df["z"] = z
    df["adf"] = adf

    # Extra gate (optional) reindexed to df.index
    if extra_gate is not None:
        eg = extra_gate.reindex(df.index)
    else:
        eg = pd.Series(True, index=df.index)
    df["gate"] = eg.astype(object).where(eg.notna(), True).astype(bool)

    # Position in spread units. Use t-1 signals for t position (no look-ahead).
    pos_spread = np.zeros(n)
    in_pos = 0
    for i in range(1, n):
        zv = z[i - 1]
        adv = adf[i - 1]
        gate_ok = bool(df["gate"].iloc[i - 1])
        if pd.isna(zv):
            pos_spread[i] = in_pos
            continue
        if in_pos == 0:
            if pd.notna(adv) and adv < adf_threshold and gate_ok:
                if zv > z_entry:
                    in_pos = -1
                elif zv < -z_entry:
                    in_pos = +1
            if extra_gate_pos_only is not None and in_pos != 0 and in_pos != extra_gate_pos_only:
                # Constrained-direction sleeve (e.g. VIX-spike: only long defensives)
                in_pos = 0
        else:
            if abs(zv) < z_exit or abs(zv) > z_stop:
                in_pos = 0
        pos_spread[i] = in_pos

    df["pos"] = pos_spread

    # PnL in spread units: pos_t * (ret_a_t - β_t-1 * ret_b_t). Use lagged β
    # to keep things causal (β is computed at t-1 close).
    beta_lag = pd.Series(beta, index=df.index).shift(1)
    gross = df["pos"] * (df["ret_a"] - beta_lag * df["ret_b"])
    # Cost: |Δpos| * (leg_a_cost + |β| * leg_b_cost) per side, both legs trade together.
    dpos = df["pos"].diff().abs().fillna(df["pos"].iloc[0] if pd.notna(df["pos"].iloc[0]) else 0.0)
    ca = leg_cost(leg_a)
    cb = leg_cost(leg_b)
    cost = dpos * (ca + beta_lag.abs().fillna(0.0) * cb)
    net = (gross - cost).fillna(0.0)
    net.index = pd.to_datetime(df.index, utc=True)
    return net, df


# ---------------------------------------------------------------------------
# VIX proxy helper for Trade 5
# ---------------------------------------------------------------------------
def vix_proxy(spx_log: pd.Series, window: int = 30) -> pd.Series:
    """30d realized vol of SPX log returns, annualized."""
    r = spx_log.diff()
    return r.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def vix_top_decile_gate(vix: pd.Series, lookback: int = 252, q: float = 0.9) -> pd.Series:
    """Bool series: True when VIX > rolling `q` percentile.

    Default q=0.9 = top decile. Lower q (e.g. 0.75) = top quartile, broader.
    """
    pct = vix.rolling(lookback, min_periods=60).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )
    return pct >= q


# ---------------------------------------------------------------------------
# Recent-corr gate for Trade 3
# ---------------------------------------------------------------------------
def recent_corr_high(a: pd.Series, b: pd.Series, window: int = 90, thr: float = 0.3) -> pd.Series:
    """Rolling-window corr of log-returns; True when corr > thr (risk-on regime)."""
    ra = a.diff()
    rb = b.diff()
    c = ra.rolling(window, min_periods=window).corr(rb)
    return c > thr


# ---------------------------------------------------------------------------
# Vol-scale to TARGET_VOL using IS subset
# ---------------------------------------------------------------------------
def vol_scale(rets: pd.Series, target: float = TARGET_VOL) -> tuple[pd.Series, float]:
    is_part = rets[rets.index < SPLIT]
    bpy = _bpy(is_part.index) if len(is_part) > 1 else TRADING_DAYS
    av = float(is_part.std(ddof=0)) * np.sqrt(bpy)
    if av <= 1e-9:
        return rets * 0.0, 0.0
    k = target / av
    k = float(np.clip(k, 0.0, 20.0))
    return rets * k, k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Wave3: synthetic-basket pair trades")
    print(f"  Split:   {SPLIT.date()}")
    print(f"  Target:  {TARGET_VOL:.1%} ann vol")
    print()
    # Per-trade: load only the symbols each trade needs so non-crypto trades
    # get the full Jan-2020+ history (SOL only starts Aug-2020). The crypto
    # trade necessarily uses the shorter SOL-bottlenecked grid.
    # NOTE on gating: the per-candidate text describes *motivation* (e.g. "when
    # crypto/SPX corr is high, fade crypto when z<-1"). We use the unified
    # methodology rules (|z|>2, ADF) for entry. We TEST gating as a refinement
    # only on candidates whose definition explicitly requires it (DEFEND must
    # be tied to VIX-stress regime; without that gate, it's just XAU+JPY
    # long-vol).  An exploratory corr-gate variant for CRYPTO is reported in
    # the breakdown but is *worse* than the ungated version and is not used
    # for survival selection.
    trades = [
        # name, leg_a (target), leg_b (hedge), required-syms, gate-builder, pos_only
        ("XAU_vs_DXY",      "XAU_USD",     "DXY_proxy",
            ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY"], None, None),
        ("US_vs_EU_equity", "US_equity",   "EU_equity",
            ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP", "DE30_EUR"], None, None),
        ("CRYPTO_vs_SPX",   "CRYPTO_mcap", "SPX500_USD",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SPX500_USD"], None, None),
        ("CRYPTO_vs_SPX_corrgate", "CRYPTO_mcap", "SPX500_USD",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SPX500_USD"], "corr_gate", None),
        ("JP225_vs_USDJPY", "JP225_USD",   "USD_JPY",
            ["JP225_USD", "USD_JPY"], None, None),
        ("DEFEND_vs_USeq",  "DEFEND",      "US_equity",
            ["XAU_USD", "USD_JPY", "SPX500_USD", "NAS100_USD", "US30_USD"], "vix_gate", +1),
    ]

    rows = []
    sleeve_raw: dict[str, pd.Series] = {}     # pre-scale
    sleeve_scaled: dict[str, pd.Series] = {}  # post-scale (5% IS ann vol)

    for name, la_name, lb_name, req_syms, gate_kind, pos_only in trades:
        L = _load_frame(req_syms)
        B = build_baskets_for(L)
        if name == "XAU_vs_DXY":
            print(f"  [load] {name}: grid {L.shape}  ({L.index.min().date()} → {L.index.max().date()})")
        a = L[la_name] if la_name in L.columns else B[la_name]
        b = L[lb_name] if lb_name in L.columns else B[lb_name]
        gate = None
        if gate_kind == "corr_gate":
            # Daily-bar crypto/SPX corr is dampened by weekend trading mismatch
            # (crypto trades weekends, SPX doesn't). On common days corr maxes
            # ~0.28; we use a percentile-based gate ("recent risk-on") on a
            # walk-forward rolling 252d window — top tercile (>0.67).
            ra = B["CRYPTO_mcap"].diff()
            rb = L["SPX500_USD"].diff()
            c90 = ra.rolling(90, min_periods=90).corr(rb)
            gate = c90.rolling(252, min_periods=60).apply(
                lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
            ) >= 0.67
        elif gate_kind == "vix_gate":
            # VIX top-quartile (q=0.75) — top-decile is too rare to overlap
            # with |z|>2 entries and produces ~0 trades.
            vix = vix_proxy(L["SPX500_USD"], window=30)
            gate = vix_top_decile_gate(vix, lookback=252, q=0.75)
        rets, diag = run_spread(
            name, a, b, leg_a=la_name, leg_b=lb_name,
            extra_gate=gate, extra_gate_pos_only=pos_only,
        )
        sleeve_raw[name] = rets
        scaled, k = vol_scale(rets, TARGET_VOL)
        sleeve_scaled[name] = scaled
        st = split_stats(scaled)
        n_trades = int((diag["pos"].diff().abs() > 0).sum() // 2)
        # Quick stationarity diagnostic on the latest in-sample window
        spread_is = diag.loc[diag.index < SPLIT, "spread"].dropna()
        if len(spread_is) >= 252:
            adf_is = adf_t_stat(spread_is.values[-252:])
        else:
            adf_is = 0.0
        rows.append({
            "name": name,
            "leg_a": la_name,
            "leg_b": lb_name,
            "scale": round(k, 3),
            "is_sharpe":  round(st["is_sharpe"], 3),
            "oos_sharpe": round(st["oos_sharpe"], 3),
            "full_sharpe": round(st["full_sharpe"], 3),
            "y2022_sharpe": round(st["y2022_sharpe"], 3),
            "is_vol":  round(st["is_vol"], 4),
            "oos_vol": round(st["oos_vol"], 4),
            "full_dd": round(st["full_dd"], 4),
            "n_trades": n_trades,
            "adf_t_is_lastwin": round(adf_is, 3),
        })
        print(f"  {name:<20}  IS={st['is_sharpe']:+.2f}  OOS={st['oos_sharpe']:+.2f}  "
              f"Y22={st['y2022_sharpe']:+.2f}  DD={st['full_dd']:+.1%}  "
              f"#tr={n_trades:>3d}  adf_IS={adf_is:+.2f}")

    bd = pd.DataFrame(rows)
    bd.to_csv(OUT / "synthetic_breakdown.csv", index=False)
    print(f"\nWrote {OUT / 'synthetic_breakdown.csv'}")

    # Survivors: IS >= 0.3 AND OOS >= 0
    surv_mask = (bd["is_sharpe"] >= 0.3) & (bd["oos_sharpe"] >= 0)
    survivors = bd[surv_mask]["name"].tolist()
    print(f"\nSurvivors ({len(survivors)}): {survivors}")

    if survivors:
        aligned = pd.concat({k: sleeve_scaled[k] for k in survivors},
                            axis=1).sort_index().fillna(0.0)
        combined = aligned.mean(axis=1)
    else:
        # Empty series with proper dtype
        any_key = next(iter(sleeve_scaled.keys()))
        combined = pd.Series(0.0, index=sleeve_scaled[any_key].index)

    combined.name = "ret"
    combined.index.name = "timestamp"
    df_out = combined.reset_index()
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], utc=True)
    df_out.to_parquet(OUT / "synthetic_returns.parquet", index=False)
    print(f"Wrote {OUT / 'synthetic_returns.parquet'}  ({len(df_out)} rows)")

    # Headline
    print("\nHEADLINE (combined survivors, equal-weight)")
    print("-" * 60)
    s = split_stats(combined)
    print(f"  FULL Sharpe: {s['full_sharpe']:+.2f}  Vol={s['full_vol']:.1%}  DD={s['full_dd']:+.1%}")
    print(f"  IS   Sharpe: {s['is_sharpe']:+.2f}  Vol={s['is_vol']:.1%}")
    print(f"  OOS  Sharpe: {s['oos_sharpe']:+.2f}  Vol={s['oos_vol']:.1%}")
    print(f"  2022 Sharpe: {s['y2022_sharpe']:+.2f}")

    # Per-year breakdown for the combined sleeve
    print("\nPer-year combined sleeve Sharpe:")
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 30:
            continue
        bpy = _bpy(sub.index)
        av = sub.std(ddof=0) * np.sqrt(bpy)
        sh = (sub.mean() * bpy / av) if av > 1e-12 else 0.0
        print(f"  {year}  Sharpe={sh:+.2f}  ret={sub.mean()*bpy:+.2%}  n={len(sub)}")


if __name__ == "__main__":
    main()
