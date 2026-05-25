"""Classical technical-indicator ensembles at D1 (wave 6).

Question: do textbook TA indicators have any real edge at D1 when applied
with proper construction (walk-forward, vol-target, IS-only filter, OOS gate)?
Or are they folklore that only "works" with hindsight?

Eight indicators / regimes:
  1. MACD_TREND  -- MACD(12,26,9) cross + zero-line filter. Long-only. Per symbol.
  2. BOLL_REV    -- 20-day SMA +/- 2sigma. Fade 2-bar pops + dips. Per symbol.
                    Restricted to mean-reverting regime (close vs 20d MA slope > 0).
  3. STOCH_ADX   -- Stoch(14) < 20 AND ADX(14) < 25 -> long (oversold w/ no trend).
                    Stoch > 80 AND ADX < 25 -> short. Per symbol.
  4. TRIPLE_MA   -- EMA(8) > EMA(21) > EMA(50): long. Strict inverse: short. Per symbol.
  5. AROON_TREND -- Aroon-up > 80 AND Aroon-down < 20: long. Indices only.
  6. OBV_MOM     -- OBV slope (20d) confirms 20d price momentum. Crypto only.
  7. CCI_REV     -- CCI(20). Long < -100, short > +100. Per symbol.
  8. WILLR_REV   -- Williams %R(14). Long < -80, short > -20. Per symbol.

Methodology:
  - IS  <  2024-01-01,  OOS >= 2024-01-01.
  - Walk-forward (final .shift(1) before backtest()).
  - Per-(strategy,symbol) sub-sleeve vol-scaled to 5% IS annualized vol.
  - Survival filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.
  - Survivors combined equal-weight within strategy; strategies equal-weight.

Outputs:
  scratch/wave6/classical_indicators.py
  scratch/wave6/classical_returns.parquet      -- D1-aligned (UTC), timestamp + ret
  scratch/wave6/classical_breakdown.csv        -- per-(strategy,symbol) IS/OOS/2022 stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO, FOREX, INDEX, SYMBOL_TYPE
from alphabeta.backtest import backtest


# -- knobs --------------------------------------------------------------------
SPLIT          = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL     = 0.05
MIN_IS_SHRP    = 0.5
MIN_OOS_SHRP   = 0.0
D1_PER_YEAR    = 365.25
OUT_DIR        = Path(__file__).resolve().parent

STRATS = [
    "MACD_TREND", "BOLL_REV", "STOCH_ADX", "TRIPLE_MA",
    "AROON_TREND", "OBV_MOM", "CCI_REV", "WILLR_REV",
]


# -- helpers ------------------------------------------------------------------
def _stats(r: pd.Series, freq: float) -> dict:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0,
                "n": int(len(r))}
    mu = r.mean() * freq
    sd = r.std(ddof=0) * np.sqrt(freq)
    sh = mu / sd if sd > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"sharpe": float(sh), "ann_return": float(mu), "ann_vol": float(sd),
            "max_dd": float(dd), "n": int(len(r))}


def _utc(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _is_oos_year(r: pd.Series, freq: float):
    is_r  = r.loc[r.index <  SPLIT]
    oos_r = r.loc[r.index >= SPLIT]
    y22   = r.loc[(r.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                  (r.index <  pd.Timestamp("2023-01-01", tz="UTC"))]
    return (_stats(r, freq), _stats(is_r, freq), _stats(oos_r, freq), _stats(y22, freq))


def _vol_scale_is(ret: pd.Series, freq: float, target: float = TARGET_VOL) -> float:
    is_r = ret.loc[ret.index < SPLIT].dropna()
    if is_r.empty:
        return 0.0
    sd = is_r.std(ddof=0) * np.sqrt(freq)
    if sd <= 0:
        return 0.0
    return float(target / sd)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _load_d1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "D1").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


# -- indicator computations ---------------------------------------------------
def _macd(close: pd.Series, fast=12, slow=26, sig=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd  = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False, min_periods=sig).mean()
    return macd, signal


def _bollinger(close: pd.Series, win=20, k=2.0):
    ma = close.rolling(win, min_periods=win).mean()
    sd = close.rolling(win, min_periods=win).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, win=14) -> pd.Series:
    hh = high.rolling(win, min_periods=win).max()
    ll = low.rolling(win, min_periods=win).min()
    rng = (hh - ll).replace(0.0, np.nan)
    k = 100.0 * (close - ll) / rng
    return k


def _true_range(high, low, close) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    return tr


def _adx(high, low, close, win=14) -> pd.Series:
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = _true_range(high, low, close)
    # Wilder smoothing approximated via EMA with alpha = 1/win.
    atr = tr.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean()
    pdi = 100.0 * plus_dm.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean() / atr
    mdi = 100.0 * minus_dm.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean() / atr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean()
    return adx


def _aroon(high: pd.Series, low: pd.Series, win=14):
    # Aroon-up: 100 * (win - bars since highest high) / win, over window of (win+1).
    rng = win + 1
    def _ago_high(s):
        return float(rng - 1 - np.argmax(s[::-1]))
    def _ago_low(s):
        return float(rng - 1 - np.argmin(s[::-1]))
    bars_since_hi = high.rolling(rng, min_periods=rng).apply(_ago_high, raw=True)
    bars_since_lo = low.rolling(rng, min_periods=rng).apply(_ago_low, raw=True)
    aroon_up = 100.0 * (win - bars_since_hi) / win
    aroon_dn = 100.0 * (win - bars_since_lo) / win
    return aroon_up, aroon_dn


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * volume.fillna(0.0)).cumsum()
    return obv


def _cci(high, low, close, win=20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(win, min_periods=win).mean()
    md  = tp.rolling(win, min_periods=win).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * md.replace(0.0, np.nan))
    return cci


def _willr(high, low, close, win=14) -> pd.Series:
    hh = high.rolling(win, min_periods=win).max()
    ll = low.rolling(win, min_periods=win).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return -100.0 * (hh - close) / rng


# -- per-symbol position builders (each returns *already-shifted* positions) --
def pos_macd_trend(df: pd.DataFrame) -> pd.Series:
    """Long when MACD crosses above signal AND MACD > 0. Long-only.

    State: enter long on cross-up while macd>0; exit when macd < signal OR macd < 0.
    """
    c = df["close"].astype("float64")
    macd, sig = _macd(c)
    long_on  = (macd > sig) & (macd > 0)
    pos = pd.Series(0.0, index=df.index)
    cur = 0.0
    long_arr = long_on.values
    for i in range(len(c)):
        if np.isnan(macd.iloc[i]) or np.isnan(sig.iloc[i]):
            pos.iloc[i] = 0.0
            continue
        if long_arr[i]:
            cur = 1.0
        else:
            cur = 0.0
        pos.iloc[i] = cur
    return pos.shift(1).fillna(0.0)


def pos_boll_rev(df: pd.DataFrame, win=20, k=2.0, hold=2) -> pd.Series:
    """Fade band touches: short 2 bars on close>upper, long 2 bars on close<lower.

    Mean-reversion regime filter: require slope of 20d MA over last 20 bars to be
    near zero (no trending environment). Specifically |MA_t / MA_{t-20} - 1| < 0.05.
    """
    c = df["close"].astype("float64")
    ma, upper, lower = _bollinger(c, win, k)
    # regime: |20d return of MA| < 5%
    ma_chg = (ma / ma.shift(win) - 1.0).abs()
    flat = (ma_chg < 0.05).fillna(False)

    raw = pd.Series(0.0, index=df.index)
    raw[(c > upper) & flat] = -1.0
    raw[(c < lower) & flat] =  1.0
    # hold for 'hold' bars (carry the signal forward)
    pos = pd.Series(0.0, index=df.index)
    cnt = 0
    direction = 0.0
    raw_arr = raw.values
    for i in range(len(c)):
        if cnt > 0:
            pos.iloc[i] = direction
            cnt -= 1
        if raw_arr[i] != 0.0 and cnt == 0:
            direction = float(raw_arr[i])
            cnt = hold
    return pos.shift(1).fillna(0.0)


def pos_stoch_adx(df: pd.DataFrame, win=14, adx_thr=25.0,
                  os_lvl=20.0, ob_lvl=80.0) -> pd.Series:
    """Long when Stoch<20 AND ADX<25; short when Stoch>80 AND ADX<25 (range fade)."""
    h, l, c = df["high"].astype("float64"), df["low"].astype("float64"), df["close"].astype("float64")
    k = _stoch(h, l, c, win)
    a = _adx(h, l, c, win)
    pos = pd.Series(0.0, index=df.index)
    pos[(k < os_lvl) & (a < adx_thr)] =  1.0
    pos[(k > ob_lvl) & (a < adx_thr)] = -1.0
    return pos.shift(1).fillna(0.0)


def pos_triple_ma(df: pd.DataFrame, f=8, m=21, s=50) -> pd.Series:
    """EMA(8) > EMA(21) > EMA(50) -> long; strict inverse -> short; else flat."""
    c = df["close"].astype("float64")
    e1 = _ema(c, f)
    e2 = _ema(c, m)
    e3 = _ema(c, s)
    pos = pd.Series(0.0, index=df.index)
    pos[(e1 > e2) & (e2 > e3)] =  1.0
    pos[(e1 < e2) & (e2 < e3)] = -1.0
    return pos.shift(1).fillna(0.0)


def pos_aroon(df: pd.DataFrame, win=14) -> pd.Series:
    """Aroon-up > 80 AND Aroon-down < 20 -> long. Symmetric short. Indices only."""
    h, l = df["high"].astype("float64"), df["low"].astype("float64")
    au, ad = _aroon(h, l, win)
    pos = pd.Series(0.0, index=df.index)
    pos[(au > 80) & (ad < 20)] =  1.0
    pos[(ad > 80) & (au < 20)] = -1.0
    return pos.shift(1).fillna(0.0)


def pos_obv_mom(df: pd.DataFrame, win=20) -> pd.Series:
    """OBV up-slope confirms price up-slope -> long; both down -> short.

    Slope here = sign of (X_t - X_{t-win}). Crypto only.
    """
    c = df["close"].astype("float64")
    v = df["volume"].astype("float64")
    obv = _obv(c, v)
    p_mom = np.sign(c - c.shift(win))
    o_mom = np.sign(obv - obv.shift(win))
    pos = pd.Series(0.0, index=df.index)
    pos[(p_mom > 0) & (o_mom > 0)] =  1.0
    pos[(p_mom < 0) & (o_mom < 0)] = -1.0
    return pos.shift(1).fillna(0.0)


def pos_cci_rev(df: pd.DataFrame, win=20, lo=-100.0, hi=100.0, hold=3) -> pd.Series:
    """CCI(20) < -100 -> long (oversold); > +100 -> short (overbought). Hold up to 'hold' bars."""
    h, l, c = df["high"].astype("float64"), df["low"].astype("float64"), df["close"].astype("float64")
    cci = _cci(h, l, c, win)
    raw = pd.Series(0.0, index=df.index)
    raw[cci < lo] =  1.0
    raw[cci > hi] = -1.0
    # carry forward 'hold' bars
    pos = pd.Series(0.0, index=df.index)
    cnt = 0
    direction = 0.0
    arr = raw.values
    for i in range(len(c)):
        if cnt > 0:
            pos.iloc[i] = direction
            cnt -= 1
        if arr[i] != 0.0 and cnt == 0:
            direction = float(arr[i])
            cnt = hold
    return pos.shift(1).fillna(0.0)


def pos_willr_rev(df: pd.DataFrame, win=14, lo=-80.0, hi=-20.0, hold=3) -> pd.Series:
    """Williams %R < -80 (oversold) -> long. > -20 (overbought) -> short. Hold 3."""
    h, l, c = df["high"].astype("float64"), df["low"].astype("float64"), df["close"].astype("float64")
    w = _willr(h, l, c, win)
    raw = pd.Series(0.0, index=df.index)
    raw[w < lo] =  1.0
    raw[w > hi] = -1.0
    pos = pd.Series(0.0, index=df.index)
    cnt = 0
    direction = 0.0
    arr = raw.values
    for i in range(len(c)):
        if cnt > 0:
            pos.iloc[i] = direction
            cnt -= 1
        if arr[i] != 0.0 and cnt == 0:
            direction = float(arr[i])
            cnt = hold
    return pos.shift(1).fillna(0.0)


# Universe by strategy
UNIVERSE = {
    "MACD_TREND":  ALL_SYMBOLS,
    "BOLL_REV":    ALL_SYMBOLS,
    "STOCH_ADX":   ALL_SYMBOLS,
    "TRIPLE_MA":   ALL_SYMBOLS,
    "AROON_TREND": INDEX,
    "OBV_MOM":     CRYPTO,
    "CCI_REV":     ALL_SYMBOLS,
    "WILLR_REV":   ALL_SYMBOLS,
}

POS_FN = {
    "MACD_TREND":  pos_macd_trend,
    "BOLL_REV":    pos_boll_rev,
    "STOCH_ADX":   pos_stoch_adx,
    "TRIPLE_MA":   pos_triple_ma,
    "AROON_TREND": pos_aroon,
    "OBV_MOM":     pos_obv_mom,
    "CCI_REV":     pos_cci_rev,
    "WILLR_REV":   pos_willr_rev,
}


# -- driver -------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Classical indicators @ D1, {len(ALL_SYMBOLS)} symbols, split={SPLIT.date()}")

    # Pre-load all D1 frames
    d1_frames = {s: _load_d1(s) for s in ALL_SYMBOLS}

    # Build D1 union calendar
    last_ts  = max(pd.to_datetime(df["timestamp"].iloc[-1], utc=True) for df in d1_frames.values())
    first_ts = min(pd.to_datetime(df["timestamp"].iloc[0], utc=True) for df in d1_frames.values())
    d1_union = pd.date_range(first_ts.normalize(), last_ts.normalize(), freq="D", tz="UTC")

    survival_rows = []
    per_strat_streams: dict[str, list[pd.Series]] = {sname: [] for sname in STRATS}

    for sname in STRATS:
        for s in UNIVERSE[sname]:
            df = d1_frames[s]
            ts_d1 = _utc(df["timestamp"])
            pos = POS_FN[sname](df)
            pos = pos.fillna(0.0).clip(-3.0, 3.0)
            pos.index = df.index

            res = backtest(df, pos, symbol=s, timeframe="D1",
                           name=f"d1cls_{sname}_{s}")
            ret = res.returns.copy()
            ret.index = ts_d1
            ret.name = f"{sname}_{s}"

            full, is_, oos, y22 = _is_oos_year(ret, D1_PER_YEAR)
            survive = (is_["sharpe"] >= MIN_IS_SHRP) and (oos["sharpe"] >= MIN_OOS_SHRP)
            scale = _vol_scale_is(ret, D1_PER_YEAR) if survive else 0.0
            scaled = (ret * scale) if scale > 0 else ret * 0.0

            survival_rows.append({
                "strategy":         sname,
                "symbol":           s,
                "asset_class":      SYMBOL_TYPE[s].value,
                "is_sharpe":        is_["sharpe"],
                "oos_sharpe":       oos["sharpe"],
                "full_sharpe":      full["sharpe"],
                "yr2022_sharpe":    y22["sharpe"],
                "is_ann_return":    is_["ann_return"],
                "oos_ann_return":   oos["ann_return"],
                "full_ann_return":  full["ann_return"],
                "full_max_dd":      full["max_dd"],
                "is_n":             is_["n"],
                "oos_n":            oos["n"],
                "n_trades":         int(res.stats["n_trades"]),
                "exposure":         float(res.stats["exposure"]),
                "survived":         bool(survive),
                "vol_scale":        float(scale),
            })
            if survive:
                per_strat_streams[sname].append(scaled)

    bd = pd.DataFrame(survival_rows)
    bd.to_csv(OUT_DIR / "classical_breakdown.csv", index=False)

    # Helper to align a tz-aware return series onto the d1_union calendar
    def _align(r: pd.Series) -> pd.Series:
        out = pd.Series(0.0, index=d1_union)
        idx = pd.DatetimeIndex(r.index).normalize()
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        s = pd.Series(r.values, index=idx)
        s = s.groupby(s.index).sum()
        common = out.index.intersection(s.index)
        out.loc[common] = s.loc[common].values
        return out

    strat_d1: dict[str, pd.Series] = {}
    n_survivors: dict[str, int] = {}
    for sname, lst in per_strat_streams.items():
        n_survivors[sname] = len(lst)
        if not lst:
            strat_d1[sname] = pd.Series(0.0, index=d1_union)
            continue
        d1_streams = [_align(r) for r in lst]
        mat = pd.concat(d1_streams, axis=1).fillna(0.0)
        strat_d1[sname] = mat.mean(axis=1)

    active = [sn for sn in STRATS if n_survivors[sn] > 0]

    if active:
        sleeve_raw = pd.concat([strat_d1[sn].rename(sn) for sn in active], axis=1).mean(axis=1)
    else:
        sleeve_raw = pd.Series(0.0, index=d1_union)

    sleeve_scale = _vol_scale_is(sleeve_raw, D1_PER_YEAR, TARGET_VOL)
    sleeve = (sleeve_raw * sleeve_scale).rename("ret")

    out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(OUT_DIR / "classical_returns.parquet", index=False)

    # ---- Reporting ----------------------------------------------------------
    def block(label: str, r: pd.Series) -> None:
        st = _stats(r, D1_PER_YEAR)
        print(f"  {label:<6} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print(f"=== CLASSICAL INDICATORS SLEEVE (D1, vol-scaled to {TARGET_VOL:.0%} IS) ===")
    print(f"Sleeve IS vol scale: {sleeve_scale:.3f}")
    print(f"Active strategies (>=1 survivor): {active}")
    block("FULL", sleeve)
    block("IS",   sleeve.loc[sleeve.index <  SPLIT])
    block("OOS",  sleeve.loc[sleeve.index >= SPLIT])
    block("2022",
          sleeve.loc[(sleeve.index >= pd.Timestamp("2022-01-01", tz="UTC")) &
                     (sleeve.index <  pd.Timestamp("2023-01-01", tz="UTC"))])

    print("\n--- Per-strategy stats (post symbol-EW, pre sleeve-scale) ---")
    for sname in STRATS:
        n_surv = n_survivors[sname]
        n_total = len(UNIVERSE[sname])
        if n_surv == 0:
            print(f"  {sname:<12}  0/{n_total} survivors")
            continue
        r = strat_d1[sname]
        full, is_, oos, y22 = _is_oos_year(r, D1_PER_YEAR)
        print(f"  {sname:<12}  surv={n_surv:>2d}/{n_total:<2d}  "
              f"FULL Sh={full['sharpe']:+5.2f}  IS Sh={is_['sharpe']:+5.2f}  "
              f"OOS Sh={oos['sharpe']:+5.2f}  2022 Sh={y22['sharpe']:+5.2f}")

    print("\n--- Survivor counts by (strategy, asset_class) ---")
    if not bd[bd["survived"]].empty:
        piv = (bd[bd["survived"]]
               .groupby(["strategy", "asset_class"]).size().unstack(fill_value=0))
        print(piv.to_string())
    else:
        print("  none")

    print("\n--- Sleeve sharpes by year ---")
    yrs = sleeve.groupby(sleeve.index.year)
    for yr, sub in yrs:
        st = _stats(sub, D1_PER_YEAR)
        print(f"  {yr}  Sh={st['sharpe']:+5.2f}  Ret={st['ann_return']:+7.2%}  "
              f"Vol={st['ann_vol']:6.2%}  DD={st['max_dd']:+7.2%}  n={st['n']}")

    # ---- Correlation vs TREND_NEW, TSMOM, D1REV --------------------------
    print("\n--- Correlation vs TREND_NEW / TSMOM / D1REV* ---")
    base = OUT_DIR.parent.parent
    asr_path = base / "scratch" / "quant" / "all_sleeve_returns_v12.parquet"
    if asr_path.exists():
        try:
            asr = pd.read_parquet(asr_path)
            if not isinstance(asr.index, pd.DatetimeIndex):
                # try to set index from first col
                idx = pd.to_datetime(asr.index, utc=True)
            else:
                idx = pd.to_datetime(asr.index, utc=True)
            asr.index = idx
            cols_of_interest = ["TREND_NEW", "TSMOM", "D1REV_NAS", "D1REV_SPX", "D1REV_UK"]
            joined = pd.concat([sleeve.rename("classical"),
                                asr[cols_of_interest]], axis=1)
            joined = joined.fillna(0.0)
            for col in cols_of_interest:
                mask = (joined["classical"] != 0) & (joined[col] != 0)
                if mask.sum() < 10:
                    corr = float("nan")
                else:
                    corr = joined.loc[mask, "classical"].corr(joined.loc[mask, col])
                print(f"  {col:<14}  corr={corr:+.3f}  overlap_n={int(mask.sum())}")
            # Composite "D1REV" = simple sum of the three D1REV columns
            d1rev_total = asr[["D1REV_NAS", "D1REV_SPX", "D1REV_UK"]].sum(axis=1)
            j2 = pd.concat([sleeve.rename("classical"),
                            d1rev_total.rename("D1REV_ALL")], axis=1).fillna(0.0)
            mask = (j2["classical"] != 0) & (j2["D1REV_ALL"] != 0)
            corr = j2.loc[mask, "classical"].corr(j2.loc[mask, "D1REV_ALL"]) if mask.sum() >= 10 else float("nan")
            print(f"  {'D1REV_ALL':<14}  corr={corr:+.3f}  overlap_n={int(mask.sum())}")
        except Exception as exc:
            print(f"  (error: {exc})")
    else:
        print(f"  {asr_path} not found")

    print(f"\nWrote {OUT_DIR / 'classical_returns.parquet'}")
    print(f"Wrote {OUT_DIR / 'classical_breakdown.csv'}")


if __name__ == "__main__":
    main()
