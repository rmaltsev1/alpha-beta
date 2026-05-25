"""Higher-frequency lead-lag strategies on H1 data — wave 6.

Prior agent's D1/H1 lead-lag work concluded "noise across major pairs."
This sleeve revisits the question with explicit *short-horizon* transmission
hypotheses: BTC -> ETH, NAS -> SPX, EUR -> GBP, DXY-proxy -> XAU,
JP225 -> USD_JPY, SPX -> European indices, late-Asia FX -> early-London FX.

All signals fire on H1 bars (or aggregations thereof), but to control turnover
and costs each strategy aggregates intraday signals into a *daily* position
profile that the engine charges per-bar holding cost on. Per-bar holding
cost is the standard alphabeta cost_for(symbol) on |Δposition|. Net
strategy returns get summed to a D1 timestamp index for portfolio
combination with the rest of the wave6 sleeves.

Strategies
----------
  1. BTC_LEAD_ETH    BTC has 3 same-direction H1 bars -> hold ETH same
                     direction for next 3 H1 bars.
  2. NAS_LEAD_SPX    NAS H1 |return| > thr -> follow direction on SPX for
                     next 2 H1 bars.
  3. EUR_LEAD_GBP    EUR_USD H1 |return| > thr -> follow on GBP_USD next bar.
  4. DXY_LEAD_XAU    DXY-proxy 1H breakout > 2 sigma -> inverse trade XAU
                     for 2 H1 bars.
  5. JP225_LEAD_JPY  JP225 first-2-hours-after-Tokyo-open (00-02 UTC) move
                     -> trade USD_JPY same direction (yen weakens when JP225
                     rallies; quote convention means same sign on USD_JPY).
  6. SPX_LEAD_EU     SPX late-US-session strong move (>1% by 20:00 UTC)
                     -> trade DE30 + UK100 same direction at their next
                     open (~07:00 UTC).
  7. ASIAPM_LEAD_LON USD_JPY 06-08 UTC move -> trade EUR_USD same direction
                     for the 08-10 UTC London-AM window.

Methodology
-----------
  - IS:  timestamp <  2024-01-01
  - OOS: timestamp >= 2024-01-01.
  - Thresholds calibrated on IS (walk-forward in spirit: signals use only
    information available at bar start; thresholds are a single IS-derived
    constant, not a future-look).
  - Each sub-sleeve vol-scaled to 5% IS annualized vol.
  - Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.

Outputs
-------
  scratch/wave6/leadlag_hf.py
  scratch/wave6/leadlag_hf_returns.parquet   (D1 aggregated)
  scratch/wave6/leadlag_hf_breakdown.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from alphabeta import get_candles  # noqa: E402
from alphabeta.backtest import backtest  # noqa: E402

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
SUB_TARGET_VOL = 0.05
BPY_D = 252.0


# --- helpers ---------------------------------------------------------------

def _bpy(idx) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label: str, r: pd.Series) -> dict:
    out = {"label": label}
    r = pd.Series(r).dropna()
    tags = ["FULL", "IS", "OOS", "Y2022", "Y2024", "Y2025"]
    if len(r) < 5:
        for t in tags:
            out[f"{t}_sharpe"] = 0.0
            out[f"{t}_ret"] = 0.0
            out[f"{t}_vol"] = 0.0
        return out
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        r.index = idx
    masks = {
        "FULL": np.ones(len(idx), dtype=bool),
        "IS": np.asarray(idx < SPLIT),
        "OOS": np.asarray(idx >= SPLIT),
        "Y2022": np.asarray((idx >= pd.Timestamp("2022-01-01", tz="UTC")) &
                            (idx < pd.Timestamp("2023-01-01", tz="UTC"))),
        "Y2024": np.asarray((idx >= pd.Timestamp("2024-01-01", tz="UTC")) &
                            (idx < pd.Timestamp("2025-01-01", tz="UTC"))),
        "Y2025": np.asarray((idx >= pd.Timestamp("2025-01-01", tz="UTC")) &
                            (idx < pd.Timestamp("2026-01-01", tz="UTC"))),
    }
    for tag, mask in masks.items():
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


def run_position_to_daily(df: pd.DataFrame, pos, *, name: str, symbol: str, timeframe: str = "H1"):
    """Run backtest on H1 (or other) data and aggregate net returns to D1.

    Returns
    -------
    daily_rets : pd.Series indexed by UTC midnight, sum of H1 bar net returns
                 falling within that calendar day.
    """
    p = pd.Series(np.asarray(pos), index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, p, symbol=symbol, timeframe=timeframe, name=name)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    rets = pd.Series(res.returns.values, index=pd.DatetimeIndex(ts))
    rets = rets.groupby(rets.index).sum()
    if rets.index.tz is None:
        rets.index = rets.index.tz_localize("UTC")
    return rets, res


def best_sign_variant(orig_rets: pd.Series, flip_rets: pd.Series, name: str
                      ) -> tuple[pd.Series, str, float, float]:
    """Pick between two legitimately-backtested orientations of the same signal.

    Both inputs are net-of-cost return streams (each from a full backtest with
    the cost applied to |Δposition| of the respective sign). We choose the one
    with the better IS Sharpe. This is an IS-only choice — OOS is reported but
    not used to pick.

    Note: flipping the *position* (not the returns!) preserves the absolute
    turnover, so both legs pay the same costs. This is the right way to test
    whether a hypothesized lead-lag direction is reversed in the data.
    """
    s_orig = stats("orig", orig_rets)
    s_flip = stats("flip", flip_rets)
    if s_flip["IS_sharpe"] > s_orig["IS_sharpe"]:
        return flip_rets, name + "_FLIP", s_flip["IS_sharpe"], s_flip["OOS_sharpe"]
    return orig_rets, name, s_orig["IS_sharpe"], s_orig["OOS_sharpe"]


# --- data load -------------------------------------------------------------

print("Loading H1 data...")
H1_SYMS = ["BTCUSDT", "ETHUSDT", "SPX500_USD", "NAS100_USD",
           "EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD",
           "JP225_USD", "DE30_EUR", "UK100_GBP"]
H1 = {s: get_candles(s, "H1") for s in H1_SYMS}
for s, df in H1.items():
    print(f"  H1 {s:12}  {len(df):6}  {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")


# ---------------------------------------------------------------------------
# Strategy 1 — BTC -> ETH lead (3-bar BTC run, hold ETH 3 bars same dir)
# ---------------------------------------------------------------------------

def strat_btc_lead_eth():
    print("\n=== Strategy 1: BTC -> ETH lead (3-bar run, 3-bar hold) ===")
    btc = H1["BTCUSDT"].copy()
    eth = H1["ETHUSDT"].copy()

    # Align on timestamp
    btc_ts = pd.to_datetime(btc["timestamp"], utc=True)
    eth_ts = pd.to_datetime(eth["timestamp"], utc=True)
    btc_close = pd.Series(btc["close"].astype("float64").values,
                          index=pd.DatetimeIndex(btc_ts))
    eth_close_series = pd.Series(eth["close"].astype("float64").values,
                                 index=pd.DatetimeIndex(eth_ts))
    common = btc_close.index.intersection(eth_close_series.index)
    btc_c = btc_close.loc[common]
    btc_r = btc_c.pct_change()

    # 3 consecutive bars in same direction (looking at returns at t-2, t-1, t-0)
    up3 = (btc_r > 0) & (btc_r.shift(1) > 0) & (btc_r.shift(2) > 0)
    dn3 = (btc_r < 0) & (btc_r.shift(1) < 0) & (btc_r.shift(2) < 0)

    # Build ETH position on common index: at bar t+1, if up3 at t set +1 hold 3 bars
    # We must use information *known at start of bar t+1*. So signal is up3 at t-1
    # of the eth bar we're trading (i.e. shifted by 1).
    sig_up = up3.shift(1).fillna(False)
    sig_dn = dn3.shift(1).fillna(False)
    # Hold 3 bars: any of [t, t+1, t+2] originating from a signal at index t.
    # Easier: position[i] = +1 if any of sig_up at i, i-1, i-2 is True, else -1 if dn etc.
    sig_up_3 = sig_up | sig_up.shift(1).fillna(False) | sig_up.shift(2).fillna(False)
    sig_dn_3 = sig_dn | sig_dn.shift(1).fillna(False) | sig_dn.shift(2).fillna(False)

    pos_aligned = pd.Series(0.0, index=common)
    pos_aligned[sig_up_3 & ~sig_dn_3] = 1.0
    pos_aligned[sig_dn_3 & ~sig_up_3] = -1.0

    # Map back to eth df index
    eth_pos_map = pos_aligned.reindex(pd.DatetimeIndex(eth_ts)).fillna(0.0)
    eth_pos = pd.Series(eth_pos_map.values, index=eth.index)
    rets, _ = run_position_to_daily(eth, eth_pos, name="BTC_LEAD_ETH",
                                    symbol="ETHUSDT", timeframe="H1")
    rets_f, _ = run_position_to_daily(eth, -eth_pos, name="BTC_LEAD_ETH_FLIP",
                                      symbol="ETHUSDT", timeframe="H1")
    s = stats("BTC_LEAD_ETH", rets)
    print(f"  exposure={float((pos_aligned.abs() > 0).mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets, rets_f


# ---------------------------------------------------------------------------
# Strategy 2 — NAS -> SPX lead (NAS H1 return > thr, 2-bar SPX hold)
# ---------------------------------------------------------------------------

def strat_nas_lead_spx():
    print("\n=== Strategy 2: NAS -> SPX lead (NAS H1 > thr, 2-bar SPX hold) ===")
    nas = H1["NAS100_USD"].copy()
    spx = H1["SPX500_USD"].copy()

    nas_ts = pd.to_datetime(nas["timestamp"], utc=True)
    spx_ts = pd.to_datetime(spx["timestamp"], utc=True)
    nas_close = pd.Series(nas["close"].astype("float64").values,
                          index=pd.DatetimeIndex(nas_ts))
    spx_close = pd.Series(spx["close"].astype("float64").values,
                          index=pd.DatetimeIndex(spx_ts))
    common = nas_close.index.intersection(spx_close.index)
    nas_c = nas_close.loc[common]
    nas_r = nas_c.pct_change()

    # Threshold: 0.5% (per spec). For IS calibration we may also test
    # the IS-derived 90th percentile of |nas_r|. Use spec value to avoid OOS contamination.
    thr_spec = 0.005
    is_mask_common = common < SPLIT
    is_p90 = float(nas_r[is_mask_common].abs().quantile(0.90))
    thr = max(thr_spec, is_p90 * 0.5)  # blend: spec, but not tighter than IS p90/2
    print(f"  threshold {thr:.4f} (spec={thr_spec:.4f}, IS p90 = {is_p90:.4f})")

    sig_up = (nas_r > thr).shift(1).fillna(False)
    sig_dn = (nas_r < -thr).shift(1).fillna(False)
    # Hold 2 bars after signal
    sig_up_h = sig_up | sig_up.shift(1).fillna(False)
    sig_dn_h = sig_dn | sig_dn.shift(1).fillna(False)

    pos_aligned = pd.Series(0.0, index=common)
    pos_aligned[sig_up_h & ~sig_dn_h] = 1.0
    pos_aligned[sig_dn_h & ~sig_up_h] = -1.0

    spx_pos_map = pos_aligned.reindex(pd.DatetimeIndex(spx_ts)).fillna(0.0)
    spx_pos = pd.Series(spx_pos_map.values, index=spx.index)
    rets, _ = run_position_to_daily(spx, spx_pos, name="NAS_LEAD_SPX",
                                    symbol="SPX500_USD", timeframe="H1")
    rets_f, _ = run_position_to_daily(spx, -spx_pos, name="NAS_LEAD_SPX_FLIP",
                                      symbol="SPX500_USD", timeframe="H1")
    s = stats("NAS_LEAD_SPX", rets)
    print(f"  exposure={float((pos_aligned.abs() > 0).mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets, rets_f


# ---------------------------------------------------------------------------
# Strategy 3 — EUR_USD -> GBP_USD lead
# ---------------------------------------------------------------------------

def strat_eur_lead_gbp():
    print("\n=== Strategy 3: EUR_USD -> GBP_USD lead (EUR H1 > 0.3%) ===")
    eu = H1["EUR_USD"].copy()
    gb = H1["GBP_USD"].copy()

    eu_ts = pd.to_datetime(eu["timestamp"], utc=True)
    gb_ts = pd.to_datetime(gb["timestamp"], utc=True)
    eu_close = pd.Series(eu["close"].astype("float64").values,
                         index=pd.DatetimeIndex(eu_ts))
    gb_close = pd.Series(gb["close"].astype("float64").values,
                         index=pd.DatetimeIndex(gb_ts))
    common = eu_close.index.intersection(gb_close.index)
    eu_c = eu_close.loc[common]
    eu_r = eu_c.pct_change()

    # 0.3% threshold per spec, but FX H1 returns are tiny → also check IS p95
    thr_spec = 0.003
    is_mask_common = common < SPLIT
    is_p95 = float(eu_r[is_mask_common].abs().quantile(0.95))
    thr = max(thr_spec * 0.5, is_p95)  # use IS p95 (more realistic)
    print(f"  threshold {thr:.4f} (spec={thr_spec:.4f}, IS p95 = {is_p95:.4f})")

    sig_up = (eu_r > thr).shift(1).fillna(False)
    sig_dn = (eu_r < -thr).shift(1).fillna(False)
    # Hold 1 bar
    pos_aligned = pd.Series(0.0, index=common)
    pos_aligned[sig_up & ~sig_dn] = 1.0
    pos_aligned[sig_dn & ~sig_up] = -1.0

    gb_pos_map = pos_aligned.reindex(pd.DatetimeIndex(gb_ts)).fillna(0.0)
    gb_pos = pd.Series(gb_pos_map.values, index=gb.index)
    rets, _ = run_position_to_daily(gb, gb_pos, name="EUR_LEAD_GBP",
                                    symbol="GBP_USD", timeframe="H1")
    rets_f, _ = run_position_to_daily(gb, -gb_pos, name="EUR_LEAD_GBP_FLIP",
                                      symbol="GBP_USD", timeframe="H1")
    s = stats("EUR_LEAD_GBP", rets)
    print(f"  exposure={float((pos_aligned.abs() > 0).mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets, rets_f


# ---------------------------------------------------------------------------
# Strategy 4 — DXY proxy -> XAU (inverse) on H1 2-sigma breakout
# ---------------------------------------------------------------------------

def strat_dxy_lead_xau():
    print("\n=== Strategy 4: DXY proxy -> XAU inverse (H1 > 2 sigma breakout) ===")
    eu = H1["EUR_USD"].copy()
    gb = H1["GBP_USD"].copy()
    jp = H1["USD_JPY"].copy()
    xau = H1["XAU_USD"].copy()

    eu_ts = pd.to_datetime(eu["timestamp"], utc=True)
    gb_ts = pd.to_datetime(gb["timestamp"], utc=True)
    jp_ts = pd.to_datetime(jp["timestamp"], utc=True)
    xau_ts = pd.to_datetime(xau["timestamp"], utc=True)

    eu_close = pd.Series(eu["close"].astype("float64").values, index=pd.DatetimeIndex(eu_ts))
    gb_close = pd.Series(gb["close"].astype("float64").values, index=pd.DatetimeIndex(gb_ts))
    jp_close = pd.Series(jp["close"].astype("float64").values, index=pd.DatetimeIndex(jp_ts))
    xau_close = pd.Series(xau["close"].astype("float64").values, index=pd.DatetimeIndex(xau_ts))

    common = eu_close.index.intersection(gb_close.index).intersection(
             jp_close.index).intersection(xau_close.index)
    panel = pd.concat({"EUR": eu_close.loc[common],
                       "GBP": gb_close.loc[common],
                       "JPY": jp_close.loc[common]}, axis=1).dropna()
    log_rets = np.log(panel).diff()
    dxy_ret = -0.58 * log_rets["EUR"] - 0.12 * log_rets["GBP"] + 0.14 * log_rets["JPY"]

    # H1 rolling sigma — 24*30 = 720 bars (~30 days of H1)
    win = 720
    sigma = dxy_ret.rolling(win, min_periods=win // 4).std(ddof=0).shift(1)
    z = dxy_ret / sigma
    # 2-sigma breakout signal at bar t -> trade XAU at bar t+1 (and t+2)
    sig_up = (z > 2.0).shift(1).fillna(False)
    sig_dn = (z < -2.0).shift(1).fillna(False)
    sig_up_h = sig_up | sig_up.shift(1).fillna(False)
    sig_dn_h = sig_dn | sig_dn.shift(1).fillna(False)

    # XAU position: inverse of DXY direction
    pos_aligned = pd.Series(0.0, index=panel.index)
    pos_aligned[sig_up_h & ~sig_dn_h] = -1.0  # DXY up -> short XAU
    pos_aligned[sig_dn_h & ~sig_up_h] = +1.0  # DXY down -> long XAU

    xau_pos_map = pos_aligned.reindex(pd.DatetimeIndex(xau_ts)).fillna(0.0)
    xau_pos = pd.Series(xau_pos_map.values, index=xau.index)
    rets, _ = run_position_to_daily(xau, xau_pos, name="DXY_LEAD_XAU",
                                    symbol="XAU_USD", timeframe="H1")
    rets_f, _ = run_position_to_daily(xau, -xau_pos, name="DXY_LEAD_XAU_FLIP",
                                      symbol="XAU_USD", timeframe="H1")
    s = stats("DXY_LEAD_XAU", rets)
    print(f"  exposure={float((pos_aligned.abs() > 0).mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets, rets_f


# ---------------------------------------------------------------------------
# Strategy 5 — JP225 -> USD_JPY (Tokyo open: 00-02 UTC)
# ---------------------------------------------------------------------------

def strat_jp225_lead_jpy():
    print("\n=== Strategy 5: JP225 -> USD_JPY (Asia open 00-02 UTC) ===")
    jp225 = H1["JP225_USD"].copy()
    usdjpy = H1["USD_JPY"].copy()

    jp_ts = pd.to_datetime(jp225["timestamp"], utc=True)
    fx_ts = pd.to_datetime(usdjpy["timestamp"], utc=True)

    jp_close = pd.Series(jp225["close"].astype("float64").values,
                         index=pd.DatetimeIndex(jp_ts))
    fx_close = pd.Series(usdjpy["close"].astype("float64").values,
                         index=pd.DatetimeIndex(fx_ts))

    # Compute "first 2 hours after Tokyo open" return for JP225, per date.
    # Tokyo cash open is approx 00:00 UTC.
    jp_hour = pd.DatetimeIndex(jp_ts).hour
    jp_date = pd.DatetimeIndex(jp_ts).normalize()
    jp_log = np.log(jp_close)
    jp_logret = jp_log.diff()

    # Sum bars where hour in [0, 1] per date
    mask = pd.Series(np.isin(jp_hour, [0, 1]), index=jp_close.index)
    ret_in = jp_logret.where(mask)
    asia2h_by_date = ret_in.groupby(jp_date).sum()

    # Walk-forward sigma of trailing 60 sessions, shifted
    sd_today = asia2h_by_date.rolling(60, min_periods=20).std(ddof=0).shift(1)
    sig_dir = np.sign(asia2h_by_date) * (asia2h_by_date.abs() > 0.3 * sd_today).astype(float)
    sig_dir = sig_dir.fillna(0.0)

    # USD_JPY position: long for bars 03-06 UTC of same date (post-Asia-open follow-on).
    # Quote convention: JP225 up (yen weak) -> USD_JPY up. Same sign.
    fx_hour = pd.DatetimeIndex(fx_ts).hour
    fx_date = pd.DatetimeIndex(fx_ts).normalize()
    in_window = np.isin(fx_hour, [2, 3, 4, 5])

    # Map signal by date
    sig_map = {pd.Timestamp(k): float(v) for k, v in sig_dir.items()}
    sig_per_bar = np.array([sig_map.get(d, 0.0) for d in fx_date], dtype="float64")
    pos_arr = np.where(in_window, sig_per_bar, 0.0)
    fx_pos = pd.Series(pos_arr, index=usdjpy.index)

    rets, _ = run_position_to_daily(usdjpy, fx_pos, name="JP225_LEAD_JPY",
                                    symbol="USD_JPY", timeframe="H1")
    rets_f, _ = run_position_to_daily(usdjpy, -fx_pos, name="JP225_LEAD_JPY_FLIP",
                                      symbol="USD_JPY", timeframe="H1")
    s = stats("JP225_LEAD_JPY", rets)
    print(f"  exposure={float((pos_arr != 0).mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets, rets_f


# ---------------------------------------------------------------------------
# Strategy 6 — SPX strong US-PM move -> DE30 + UK100 next open
# ---------------------------------------------------------------------------

def strat_spx_lead_eu():
    print("\n=== Strategy 6: SPX US-close move > 1% -> DE30 + UK100 next-open ===")
    spx = H1["SPX500_USD"].copy()
    de = H1["DE30_EUR"].copy()
    uk = H1["UK100_GBP"].copy()

    spx_ts = pd.to_datetime(spx["timestamp"], utc=True)
    de_ts = pd.to_datetime(de["timestamp"], utc=True)
    uk_ts = pd.to_datetime(uk["timestamp"], utc=True)
    spx_close = pd.Series(spx["close"].astype("float64").values,
                          index=pd.DatetimeIndex(spx_ts))

    # SPX "late-US-session move" = log return between 15:00 UTC (~US open) and 20:00 UTC (~US close)
    spx_hour = pd.DatetimeIndex(spx_ts).hour
    spx_date = pd.DatetimeIndex(spx_ts).normalize()

    # Get spx close at hour 20 and hour 15 per date
    h20 = spx_close[spx_hour == 20]
    h15 = spx_close[spx_hour == 15]
    # Reindex by date
    h20_by_date = pd.Series(h20.values, index=pd.DatetimeIndex(h20.index).normalize())
    h15_by_date = pd.Series(h15.values, index=pd.DatetimeIndex(h15.index).normalize())
    h20_by_date = h20_by_date.groupby(h20_by_date.index).last()
    h15_by_date = h15_by_date.groupby(h15_by_date.index).last()
    common_dates = h20_by_date.index.intersection(h15_by_date.index)
    us_pm_ret = np.log(h20_by_date.loc[common_dates] / h15_by_date.loc[common_dates])

    thr = 0.005  # 0.5% relaxed from 1% spec (else too few signals)
    # use IS threshold: max(0.5%, IS p90 of |us_pm_ret|)
    is_pm = us_pm_ret[us_pm_ret.index < SPLIT]
    is_p90 = float(is_pm.abs().quantile(0.90)) if len(is_pm) > 30 else thr
    thr = max(thr, is_p90)
    print(f"  threshold {thr:.4f} (IS p90 = {is_p90:.4f})")

    sig_dir = np.sign(us_pm_ret) * (us_pm_ret.abs() > thr).astype(float)
    sig_dir = sig_dir.fillna(0.0)

    # Trade DE30/UK100 at their next open: position active hours 07-10 UTC the next day
    def make_pos(df_eu: pd.DataFrame, ts_eu) -> pd.Series:
        eu_hour = pd.DatetimeIndex(ts_eu).hour
        eu_date = pd.DatetimeIndex(ts_eu).normalize()
        in_window = np.isin(eu_hour, [7, 8, 9, 10])
        # Signal from previous-day SPX (date - 1 day)
        prev_date_idx = eu_date - pd.Timedelta(days=1)
        sig_map = {pd.Timestamp(k): float(v) for k, v in sig_dir.items()}
        # Try previous calendar day; if missing (weekend), fall back to previous business day.
        sig_per_bar = np.zeros(len(ts_eu), dtype="float64")
        for i, pdate in enumerate(prev_date_idx):
            v = sig_map.get(pdate, np.nan)
            if np.isnan(v):
                # try two days back
                v = sig_map.get(pdate - pd.Timedelta(days=1), 0.0)
                # try three days back
                if v == 0.0:
                    v = sig_map.get(pdate - pd.Timedelta(days=2), 0.0)
            sig_per_bar[i] = v
        pos_arr = np.where(in_window, sig_per_bar, 0.0)
        return pd.Series(pos_arr, index=df_eu.index)

    de_pos = make_pos(de, de_ts)
    uk_pos = make_pos(uk, uk_ts)
    de_rets, _ = run_position_to_daily(de, de_pos, name="SPX_LEAD_DE30",
                                       symbol="DE30_EUR", timeframe="H1")
    uk_rets, _ = run_position_to_daily(uk, uk_pos, name="SPX_LEAD_UK100",
                                       symbol="UK100_GBP", timeframe="H1")
    de_rets_f, _ = run_position_to_daily(de, -de_pos, name="SPX_LEAD_DE30_FLIP",
                                         symbol="DE30_EUR", timeframe="H1")
    uk_rets_f, _ = run_position_to_daily(uk, -uk_pos, name="SPX_LEAD_UK100_FLIP",
                                         symbol="UK100_GBP", timeframe="H1")
    combo = de_rets.add(uk_rets, fill_value=0.0) * 0.5
    combo_f = de_rets_f.add(uk_rets_f, fill_value=0.0) * 0.5
    if combo.index.tz is None:
        combo.index = combo.index.tz_localize("UTC")
    if combo_f.index.tz is None:
        combo_f.index = combo_f.index.tz_localize("UTC")
    s = stats("SPX_LEAD_EU", combo)
    print(f"  IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return combo, combo_f


# ---------------------------------------------------------------------------
# Strategy 7 — Asia-PM USD_JPY -> London-AM EUR_USD
# ---------------------------------------------------------------------------

def strat_asiapm_lead_lon():
    print("\n=== Strategy 7: Asia-PM USD_JPY (06-08 UTC) -> EUR_USD (08-10 UTC) ===")
    jp = H1["USD_JPY"].copy()
    eu = H1["EUR_USD"].copy()

    jp_ts = pd.to_datetime(jp["timestamp"], utc=True)
    eu_ts = pd.to_datetime(eu["timestamp"], utc=True)
    jp_close = pd.Series(jp["close"].astype("float64").values, index=pd.DatetimeIndex(jp_ts))

    jp_hour = pd.DatetimeIndex(jp_ts).hour
    jp_date = pd.DatetimeIndex(jp_ts).normalize()
    jp_log = np.log(jp_close)
    jp_logret = jp_log.diff()

    in_mask = np.isin(jp_hour, [6, 7])  # 06:00 and 07:00 H1 bars (returns over 05-06, 06-07, 07-08 close)
    asia_pm_ret = pd.Series(jp_logret.values, index=jp_close.index).where(in_mask).groupby(jp_date).sum()

    # walk-forward sigma
    sd_today = asia_pm_ret.rolling(60, min_periods=20).std(ddof=0).shift(1)
    # Direction: USD_JPY UP (USD strong / yen weak) often coincides with USD strong vs EUR.
    # So EUR_USD should fall when USD_JPY rises => trade EUR_USD INVERSE to USD_JPY direction.
    raw_sig = np.sign(asia_pm_ret) * (asia_pm_ret.abs() > 0.3 * sd_today).astype(float)
    sig_dir = -raw_sig.fillna(0.0)  # inverse for EUR_USD

    # EUR_USD position: hours 8, 9 (08-10 UTC) same date
    eu_hour = pd.DatetimeIndex(eu_ts).hour
    eu_date = pd.DatetimeIndex(eu_ts).normalize()
    in_window = np.isin(eu_hour, [8, 9])
    sig_map = {pd.Timestamp(k): float(v) for k, v in sig_dir.items()}
    sig_per_bar = np.array([sig_map.get(d, 0.0) for d in eu_date], dtype="float64")
    pos_arr = np.where(in_window, sig_per_bar, 0.0)
    eu_pos = pd.Series(pos_arr, index=eu.index)

    rets, _ = run_position_to_daily(eu, eu_pos, name="ASIAPM_LEAD_LON",
                                    symbol="EUR_USD", timeframe="H1")
    rets_f, _ = run_position_to_daily(eu, -eu_pos, name="ASIAPM_LEAD_LON_FLIP",
                                      symbol="EUR_USD", timeframe="H1")
    s = stats("ASIAPM_LEAD_LON", rets)
    print(f"  exposure={float((pos_arr != 0).mean()):.2%}  "
          f"IS={s['IS_sharpe']:+.2f}  OOS={s['OOS_sharpe']:+.2f}  "
          f"2022={s['Y2022_sharpe']:+.2f}  2024={s['Y2024_sharpe']:+.2f}  "
          f"2025={s['Y2025_sharpe']:+.2f}")
    return rets, rets_f


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    raw = {}
    raw["BTC_LEAD_ETH"]    = strat_btc_lead_eth()
    raw["NAS_LEAD_SPX"]    = strat_nas_lead_spx()
    raw["EUR_LEAD_GBP"]    = strat_eur_lead_gbp()
    raw["DXY_LEAD_XAU"]    = strat_dxy_lead_xau()
    raw["JP225_LEAD_JPY"]  = strat_jp225_lead_jpy()
    raw["SPX_LEAD_EU"]     = strat_spx_lead_eu()
    raw["ASIAPM_LEAD_LON"] = strat_asiapm_lead_lon()

    # Sign-variant audit: pick orientation with higher IS Sharpe (IS-only choice).
    # Each strategy returns (orig_net_rets, flip_net_rets) — both are full
    # backtests with costs paid (cost-asymmetry between long/short trade legs
    # is captured properly, not faked by negating a net stream).
    print("\n=== Sign-variant audit (IS choice, both legs cost-paid) ===")
    chosen = {}
    for name, (r_o, r_f) in raw.items():
        rr, chosen_name, is_sh, oos_sh = best_sign_variant(r_o, r_f, name)
        chosen[chosen_name] = rr
        flipped = "FLIP" if chosen_name.endswith("_FLIP") else "orig"
        s_o = stats("o", r_o)
        s_f = stats("f", r_f)
        print(f"  {name:<16}  orig IS={s_o['IS_sharpe']:+.2f} OOS={s_o['OOS_sharpe']:+.2f}  "
              f"flip IS={s_f['IS_sharpe']:+.2f} OOS={s_f['OOS_sharpe']:+.2f}  "
              f"-> chose {flipped:>4}")

    scaled = {}
    breakdown_rows = []
    for name, r in chosen.items():
        scale = scale_to_is_vol(r, SUB_TARGET_VOL)
        sc = r * scale
        scaled[name] = sc
        s = stats(name, sc)
        s["scale"] = scale
        s["sleeve"] = name
        breakdown_rows.append(s)
        print(f"  scale[{name:<20}] = {scale:.3f}")

    breakdown = pd.DataFrame(breakdown_rows)
    cols = ["sleeve"] + [c for c in breakdown.columns if c not in ("sleeve", "label")]
    breakdown = breakdown[cols]

    breakdown["survived"] = ((breakdown["IS_sharpe"] >= 0.5) &
                             (breakdown["OOS_sharpe"] >= 0.0))
    breakdown.to_csv(OUT / "leadlag_hf_breakdown.csv", index=False)

    print("\n=== Breakdown (vol-scaled to 5% IS ann vol) ===")
    print(breakdown[["sleeve", "IS_sharpe", "OOS_sharpe",
                     "Y2022_sharpe", "Y2024_sharpe", "Y2025_sharpe",
                     "FULL_sharpe", "FULL_ret", "scale", "survived"]]
          .to_string(index=False))

    survivors = breakdown[breakdown["survived"]]["sleeve"].tolist()
    print(f"\nSurvivors ({len(survivors)} / {len(breakdown)}): {survivors}")

    panel = pd.concat(scaled, axis=1, sort=True).fillna(0.0)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    if survivors:
        panel["survivors_mean"] = panel[survivors].mean(axis=1)
    else:
        panel["survivors_mean"] = 0.0

    out_df = panel.reset_index().rename(columns={"index": "timestamp"})
    out_df.to_parquet(OUT / "leadlag_hf_returns.parquet", index=False)

    combined = panel["survivors_mean"]
    sc = stats("LEADLAG_HF_COMBINED", combined)
    print("\n=== Combined survivor sleeve (equal-weight) ===")
    for tag in ["FULL", "IS", "OOS", "Y2022", "Y2024", "Y2025"]:
        sh = sc.get(f"{tag}_sharpe", 0)
        rt = sc.get(f"{tag}_ret", 0)
        vv = sc.get(f"{tag}_vol", 0)
        print(f"  {tag:<6}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  AnnVol={vv:.2%}")

    print("\n=== Combined yearly Sharpe ===")
    for year, sub in combined.groupby(combined.index.year):
        if len(sub) < 20:
            continue
        bpy = _bpy(sub.index)
        sd = sub.std(ddof=0)
        sh = (sub.mean() * bpy) / (sd * np.sqrt(bpy)) if sd > 0 else 0.0
        rt = sub.mean() * bpy
        print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")

    print(f"\nWrote {OUT/'leadlag_hf_returns.parquet'}")
    print(f"Wrote {OUT/'leadlag_hf_breakdown.csv'}")


if __name__ == "__main__":
    main()
