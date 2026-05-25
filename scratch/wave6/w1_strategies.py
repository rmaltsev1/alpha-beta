"""W1 (weekly) long-horizon trend signals (wave 6).

The only timeframe we hadn't touched. ~302-334 W1 bars per symbol over 2020-2026,
so noise is lower but sample size is small (~232 IS bars / ~100 OOS bars).

Seven strategies:
  1. MOM12     -- sign(trailing 12-week return). Per symbol.
  2. DONCH     -- Donchian breakout: long at 12w high, exit at 12w low (no short).
  3. ZREV      -- z-score of weekly return vs 52w std; fade |z|>2 for 1 week.
  4. RSI_MA    -- 4-week RSI > 60 AND price > 26w MA; long-only defensive.
  5. XSMOM     -- cross-sectional: long top 3, short bottom 3 by 12w return.
  6. EXHAUST   -- 8 consecutive +ve (or -ve) W1 returns -> fade for 2 weeks.
  7. CALENDAR  -- first-week-of-month / last-week-of-quarter / year-end effects on W1.

Methodology:
  - IS  <  2024-01-01,  OOS >= 2024-01-01.
  - All signals walk-forward (final shift(1) before feeding to backtest).
  - Per-(strategy, symbol) stream vol-scaled to 5% IS annualized vol.
  - Survival filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.
  - Survivors combined equal-weight per strategy; equal-weight across strategies.
  - Sleeve scaled to 5% IS ann vol; collapsed to D1 calendar (zeros on non-W1 days).

Outputs:
  scratch/wave6/w1_strategies.py
  scratch/wave6/w1_returns.parquet      -- D1-aligned (UTC), timestamp + ret
  scratch/wave6/w1_breakdown.csv        -- per-(strategy,symbol) IS/OOS/2022 stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, SYMBOL_TYPE
from alphabeta.backtest import backtest


# -- knobs --------------------------------------------------------------------
SPLIT          = pd.Timestamp("2024-01-01", tz="UTC")
MOM_WIN        = 12       # weeks
DONCH_LONG     = 12       # weeks (entry channel)
DONCH_EXIT     = 12       # weeks (exit channel = 12w low, per spec). 26 also tested below.
DONCH_SHORT    = 26       # secondary lookback for the down-channel option (not used as exit here)
ZWIN           = 52       # weeks
Z_THRESH       = 2.0
RSI_WIN        = 4        # "monthly" RSI on weekly bars
RSI_HIGH       = 60.0
MA_WIN         = 26       # weeks
XSMOM_TOP_N    = 3
EXHAUST_RUN    = 8
EXHAUST_HOLD   = 2        # weeks
TARGET_VOL     = 0.05
MIN_IS_SHRP    = 0.4
MIN_OOS_SHRP   = 0.0
W1_PER_YEAR    = 52.1775  # avg weeks/year for annualization
D1_PER_YEAR    = 365.25
OUT_DIR        = Path(__file__).resolve().parent

STRATS = ["MOM12", "DONCH", "ZREV", "RSI_MA", "XSMOM", "EXHAUST", "CALENDAR"]


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


def _rsi(close: pd.Series, win: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.rolling(win, min_periods=win).mean()
    roll_dn = dn.rolling(win, min_periods=win).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _load_w1(symbol: str) -> pd.DataFrame:
    df = get_candles(symbol, "W1").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


# -- per-symbol position builders (returned BEFORE shift; we shift at backtest time)
def pos_mom12(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype("float64")
    r12 = np.log(c / c.shift(MOM_WIN))
    pos = np.sign(r12).fillna(0.0)
    return pos.shift(1).fillna(0.0)


def pos_donch(df: pd.DataFrame) -> pd.Series:
    """Long-only Donchian: enter long when close >= 12w high; exit when close <= 12w low.
    Stateful: position persists until exit triggered.
    """
    c = df["close"].astype("float64")
    # Use prior-bar channel only (rolling on shift(1) close so signal is observable)
    hi = c.shift(1).rolling(DONCH_LONG, min_periods=DONCH_LONG).max()
    lo = c.shift(1).rolling(DONCH_EXIT, min_periods=DONCH_EXIT).min()
    pos = np.zeros(len(c))
    cur = 0.0
    for i in range(len(c)):
        if np.isnan(hi.iloc[i]) or np.isnan(lo.iloc[i]):
            pos[i] = 0.0
            continue
        if cur <= 0 and c.iloc[i] >= hi.iloc[i]:
            cur = 1.0
        elif cur > 0 and c.iloc[i] <= lo.iloc[i]:
            cur = 0.0
        pos[i] = cur
    # The signal above uses current-bar close, which is forward-looking on bar i.
    # Shift by 1 to make it observable at start of bar i+1.
    s = pd.Series(pos, index=df.index)
    return s.shift(1).fillna(0.0)


def pos_zrev(df: pd.DataFrame) -> pd.Series:
    """Fade |z| > 2 for one bar (one W1 bar)."""
    c = df["close"].astype("float64")
    r = c.pct_change()
    sd = r.rolling(ZWIN, min_periods=ZWIN).std(ddof=0)
    z = (r / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # We fade the move: short after a >+2 spike, long after a <-2 spike.
    raw = pd.Series(0.0, index=df.index)
    raw[z >  Z_THRESH] = -1.0
    raw[z < -Z_THRESH] =  1.0
    # The signal is computed from bar t close -> shift by 1 to act on bar t+1.
    # That gives the "fade for 1 week" hold automatically (single non-zero bar).
    return raw.shift(1).fillna(0.0)


def pos_rsi_ma(df: pd.DataFrame) -> pd.Series:
    """Long when 4w RSI > 60 AND price > 26w MA; flat otherwise (long-only)."""
    c = df["close"].astype("float64")
    rsi = _rsi(c, RSI_WIN)
    ma  = c.rolling(MA_WIN, min_periods=MA_WIN).mean()
    long_sig = (rsi > RSI_HIGH) & (c > ma)
    pos = pd.Series(0.0, index=df.index)
    pos[long_sig] = 1.0
    return pos.shift(1).fillna(0.0)


def pos_exhaust(df: pd.DataFrame) -> pd.Series:
    """8 consecutive +ve W1 returns -> fade for 2 weeks (short).
    8 consecutive -ve -> fade for 2 weeks (long).
    """
    c = df["close"].astype("float64")
    r = c.pct_change().fillna(0.0)
    sign = np.sign(r)
    pos = np.zeros(len(c))
    # Compute count of consecutive same-sign at each bar
    run = np.zeros(len(c))
    for i in range(len(c)):
        if i == 0:
            run[i] = 0
        elif sign.iloc[i] != 0 and sign.iloc[i] == sign.iloc[i - 1]:
            run[i] = run[i - 1] + 1
        else:
            run[i] = 1 if sign.iloc[i] != 0 else 0
    # When run >= 8, fade for next 2 bars
    hold = 0
    direction = 0.0
    for i in range(len(c)):
        if hold > 0:
            pos[i] = direction
            hold -= 1
        if run[i] >= EXHAUST_RUN and hold == 0:
            direction = -float(sign.iloc[i])
            hold = EXHAUST_HOLD
            # don't take a position on the same bar (signal is from close of i)
    s = pd.Series(pos, index=df.index)
    # Shift to make sure we act on bar t+1, not t.
    return s.shift(1).fillna(0.0)


def pos_calendar(df: pd.DataFrame) -> pd.Series:
    """First-week-of-month: long.
    Last-week-of-quarter: long.
    Year-end (last W1 in December): long.

    These are the "well-known" positive seasonality effects on weekly bars.
    Signal is computed deterministically from the bar's own timestamp (no data leakage).
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    pos = np.zeros(len(df))
    # We want the position for bar t to be observable at start of bar t.
    # Each W1 bar timestamp is its open; treat it as the *week containing* that date.
    for i in range(len(df)):
        t = ts.iloc[i]
        flag = False
        # first week of month: bar opens on day-of-month <= 7
        if t.day <= 7:
            flag = True
        # last week of quarter: bar opens within the last 8 days of Mar/Jun/Sep/Dec
        eom_quarters = {3, 6, 9, 12}
        if t.month in eom_quarters:
            # last day of the month
            next_month = (t.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
            last_day = (next_month - pd.Timedelta(days=1)).day
            if (last_day - t.day) < 8:
                flag = True
        # year-end (December last week): handled above; also if month==12 and day >= 25
        if t.month == 12 and t.day >= 25:
            flag = True
        pos[i] = 1.0 if flag else 0.0
    # No shift: the rule depends only on the bar's open timestamp, which is known
    # at the start of the bar. (Confirmed walk-forward.)
    return pd.Series(pos, index=df.index)


# -- cross-sectional momentum (built across all symbols) ----------------------
def build_xsmom(symbols: list[str]) -> dict[str, pd.Series]:
    """For each W1 bar, rank symbols by trailing 12w log-return; long top N, short bottom N.

    We align all symbols to a common W1 timestamp union (most symbols share the same
    weekly grid; SOL starts later). Returns a dict[symbol] -> pos series indexed by
    that symbol's own row index (compatible with backtest()).
    """
    closes = {}
    for s in symbols:
        df = _load_w1(s)
        closes[s] = pd.Series(df["close"].astype("float64").values,
                              index=_utc(df["timestamp"]))
    aligned = pd.concat(closes, axis=1).sort_index()
    # 12-week log return per symbol
    lr12 = np.log(aligned / aligned.shift(MOM_WIN))
    # Rank cross-sectionally each week (ignore NaN)
    rank = lr12.rank(axis=1, method="average")
    n_per_row = lr12.notna().sum(axis=1)
    # Long top N, short bottom N
    out = {}
    for s in symbols:
        r = rank[s]
        n = n_per_row
        long_mask = (n >= 2 * XSMOM_TOP_N) & (r > (n - XSMOM_TOP_N))
        short_mask = (n >= 2 * XSMOM_TOP_N) & (r <= XSMOM_TOP_N)
        pos = pd.Series(0.0, index=aligned.index)
        pos[long_mask] = 1.0
        pos[short_mask] = -1.0
        # Shift by 1 (walk-forward), then align to this symbol's own row index
        pos = pos.shift(1).fillna(0.0)
        out[s] = pos
    return out, aligned.index  # aligned.index is union of W1 timestamps


# -- driver -------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"W1 strategies, {len(ALL_SYMBOLS)} symbols, split={SPLIT.date()}")

    # Pre-load all W1 frames once
    w1_frames = {s: _load_w1(s) for s in ALL_SYMBOLS}

    # XSMOM positions are computed on the cross-section
    xsmom_pos_map, _xs_union = build_xsmom(ALL_SYMBOLS)

    # For each (strategy, symbol), build position, run backtest, score, vol-scale
    survival_rows = []
    per_strat_streams = {sname: [] for sname in STRATS}

    # Union daily index for the final D1-aligned parquet
    # We'll build it from full date range 2020-01-01 .. last observed timestamp + 1 day.
    last_ts = max(pd.to_datetime(df["timestamp"].iloc[-1], utc=True) for df in w1_frames.values())
    first_ts = min(pd.to_datetime(df["timestamp"].iloc[0], utc=True) for df in w1_frames.values())
    d1_union = pd.date_range(first_ts.normalize(), last_ts.normalize(), freq="D", tz="UTC")

    for s in ALL_SYMBOLS:
        df = w1_frames[s]
        ts_w1 = _utc(df["timestamp"])

        for sname in STRATS:
            if sname == "MOM12":
                pos = pos_mom12(df)
            elif sname == "DONCH":
                pos = pos_donch(df)
            elif sname == "ZREV":
                pos = pos_zrev(df)
            elif sname == "RSI_MA":
                pos = pos_rsi_ma(df)
            elif sname == "EXHAUST":
                pos = pos_exhaust(df)
            elif sname == "CALENDAR":
                pos = pos_calendar(df)
            elif sname == "XSMOM":
                # xsmom is indexed by the W1 union; reindex onto this symbol's ts_w1
                xpos = xsmom_pos_map[s]
                # asof-merge onto this symbol's own W1 timestamps
                aligned = pd.merge_asof(
                    pd.DataFrame({"ts": ts_w1}).sort_values("ts"),
                    pd.DataFrame({"ts": xpos.index, "p": xpos.values}).sort_values("ts"),
                    on="ts", direction="backward",
                )
                pos = pd.Series(aligned["p"].fillna(0.0).values, index=df.index)
            else:
                raise ValueError(sname)

            pos = pos.fillna(0.0).clip(-3.0, 3.0)
            pos.index = df.index

            res = backtest(df, pos, symbol=s, timeframe="W1",
                           name=f"w1_{sname}_{s}")
            ret = res.returns.copy()
            ret.index = ts_w1  # tz-aware
            ret.name = f"{sname}_{s}"

            full, is_, oos, y22 = _is_oos_year(ret, W1_PER_YEAR)
            survive = (is_["sharpe"] >= MIN_IS_SHRP) and (oos["sharpe"] >= MIN_OOS_SHRP)
            scale = _vol_scale_is(ret, W1_PER_YEAR) if survive else 0.0
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
    bd.to_csv(OUT_DIR / "w1_breakdown.csv", index=False)

    # ---- Combine survivors equal-weight within each strategy ----------------
    def _to_d1(w1_ret: pd.Series) -> pd.Series:
        """Take a W1-indexed return series and place each value on its W1 timestamp
        within the D1 calendar; zeros elsewhere.
        """
        out = pd.Series(0.0, index=d1_union)
        # snap each timestamp to its calendar day
        idx = pd.DatetimeIndex(w1_ret.index).normalize()
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        s = pd.Series(w1_ret.values, index=idx)
        # Aggregate dupes (shouldn't happen for a single symbol but be safe)
        s = s.groupby(s.index).sum()
        common = out.index.intersection(s.index)
        out.loc[common] = s.loc[common].values
        return out

    strat_d1 = {}
    n_survivors = {}
    for sname, lst in per_strat_streams.items():
        n_survivors[sname] = len(lst)
        if not lst:
            strat_d1[sname] = pd.Series(0.0, index=d1_union)
            continue
        d1_streams = [_to_d1(r) for r in lst]
        mat = pd.concat(d1_streams, axis=1).fillna(0.0)
        strat_d1[sname] = mat.mean(axis=1)

    active = [sn for sn in STRATS if n_survivors[sn] > 0]

    # ---- Sleeve = equal-weight across active strategies ---------------------
    if active:
        sleeve_raw = pd.concat([strat_d1[sn].rename(sn) for sn in active], axis=1).mean(axis=1)
    else:
        sleeve_raw = pd.Series(0.0, index=d1_union)

    sleeve_scale = _vol_scale_is(sleeve_raw, D1_PER_YEAR, TARGET_VOL)
    sleeve = (sleeve_raw * sleeve_scale).rename("ret")

    # ---- Save sleeve parquet (D1, UTC, timestamp + ret) ---------------------
    out_df = pd.DataFrame({"timestamp": sleeve.index, "ret": sleeve.values})
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], utc=True)
    out_df.to_parquet(OUT_DIR / "w1_returns.parquet", index=False)

    # ---- Reporting ----------------------------------------------------------
    def block(label: str, r: pd.Series) -> None:
        st = _stats(r, D1_PER_YEAR)
        print(f"  {label:<6} Sharpe={st['sharpe']:+5.2f}  "
              f"Ret={st['ann_return']:+7.2%}  Vol={st['ann_vol']:6.2%}  "
              f"DD={st['max_dd']:+7.2%}  n={st['n']}")

    print()
    print(f"=== W1 SLEEVE (D1-aligned, vol-scaled to {TARGET_VOL:.0%} IS) ===")
    print(f"Sleeve IS vol scale factor: {sleeve_scale:.3f}")
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
        if n_surv == 0:
            print(f"  {sname:<10}  0 survivors")
            continue
        r = strat_d1[sname]
        full, is_, oos, y22 = _is_oos_year(r, D1_PER_YEAR)
        print(f"  {sname:<10}  survivors={n_surv:>2d}  "
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

    # ---- D1 sleeve overlap check vs existing wave5/wave6 sleeves -----------
    print("\n--- Correlation vs existing D1 sleeves ---")
    candidates = [
        "scratch/wave5/multi_tf_returns.parquet",
        "scratch/wave5/asymmetric_returns.parquet",
        "scratch/wave5/tail_v3_returns.parquet",
        "scratch/wave5/regime_voltarget_returns.parquet",
        "scratch/wave5/adaptive_weighting_returns.parquet",
        "scratch/wave5/kelly_v2_returns.parquet",
    ]
    base = OUT_DIR.parent.parent
    for c in candidates:
        p = base / c
        if not p.exists():
            continue
        try:
            oth = pd.read_parquet(p)
            if "timestamp" not in oth.columns or "ret" not in oth.columns:
                # Some parquets store a wide table or have a different schema.
                print(f"  {c.split('/')[-1]:<38}  (schema mismatch, skipping)")
                continue
            oth["timestamp"] = pd.to_datetime(oth["timestamp"], utc=True)
            oth = oth.set_index("timestamp")["ret"]
            joined = pd.concat([sleeve.rename("w1"), oth.rename("oth")], axis=1).fillna(0.0)
            mask = (joined["w1"] != 0) & (joined["oth"] != 0)
            if mask.sum() < 10:
                corr = float("nan")
            else:
                corr = joined.loc[mask, "w1"].corr(joined.loc[mask, "oth"])
            print(f"  {c.split('/')[-1]:<38}  corr={corr:+.3f}  overlap_n={int(mask.sum())}")
        except Exception as exc:
            print(f"  {c.split('/')[-1]:<38}  (error: {exc})")

    print(f"\nWrote {OUT_DIR / 'w1_returns.parquet'}")
    print(f"Wrote {OUT_DIR / 'w1_breakdown.csv'}")


if __name__ == "__main__":
    main()
