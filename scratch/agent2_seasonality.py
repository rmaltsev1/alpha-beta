"""Seasonality / calendar / session analysis for all 13 symbols.

Outputs CSVs to scratch/agent2_*.csv. Re-runnable.

Sections:
  1. Day-of-week  (D1 log returns)
  2. Hour-of-day  (H1 log returns)
  3. Session stats Asia/London/NY (H1)
  4. Session-open bar range vs other hours
  5. End-of-month vs start vs middle (D1)
  6. Crypto weekend vol + Mon-open gap
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python scratch/agent2_seasonality.py` without setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alphabeta import get_candles, ALL_SYMBOLS, CRYPTO, FOREX, INDEX  # noqa: E402

OUT = Path(__file__).parent
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def log_ret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def session_date(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Return the calendar-date the D1 bar *represents*.

    OANDA's D1 candles start at 22:00 UTC the previous calendar day (NY 5pm
    close convention). So a bar timestamped 2020-01-05 22:00 UTC is actually
    *Monday's* session. We push forward by 2h to land on the session date.
    Crypto bars start at 00:00 UTC and need no shift.
    """
    if symbol in CRYPTO:
        return df["timestamp"]
    # OANDA opens between 21:00 and 22:00 UTC depending on DST. +4h is enough
    # to push *both* into the next calendar day cleanly.
    return df["timestamp"] + pd.Timedelta(hours=4)


# ---------------------------------------------------------------------------
# 1. Day-of-week effect (D1 log returns)
# ---------------------------------------------------------------------------
def dow_table() -> pd.DataFrame:
    rows = []
    for sym in ALL_SYMBOLS:
        df = get_candles(sym, "D1").copy()
        df["r"] = log_ret(df["close"])
        df = df.dropna(subset=["r"])
        df["dow"] = session_date(df, sym).dt.dayofweek  # 0=Mon ... 6=Sun
        n_total = len(df)
        for dow in range(7):
            sub = df.loc[df["dow"] == dow, "r"]
            if sub.empty:
                continue
            mean = sub.mean()
            std = sub.std(ddof=1)
            n = len(sub)
            se = std / math.sqrt(n) if n > 1 else np.nan
            t = mean / se if se and se > 0 else np.nan
            # ~95% two-sided ~ |t|>1.96; flag at 99% (|t|>2.58) too
            sig95 = bool(abs(t) > 1.96) if not math.isnan(t) else False
            sig99 = bool(abs(t) > 2.58) if not math.isnan(t) else False
            rows.append(
                dict(
                    symbol=sym,
                    weekday=WEEKDAY_NAMES[dow],
                    n=n,
                    mean_bps=mean * 1e4,
                    std_bps=std * 1e4,
                    t=t,
                    sig95=sig95,
                    sig99=sig99,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Hour-of-day (H1 log returns)
# ---------------------------------------------------------------------------
def hod_table() -> pd.DataFrame:
    rows = []
    for sym in ALL_SYMBOLS:
        df = get_candles(sym, "H1").copy()
        df["r"] = log_ret(df["close"])
        df = df.dropna(subset=["r"])
        df["hour"] = df["timestamp"].dt.hour
        for hr in range(24):
            sub = df.loc[df["hour"] == hr, "r"]
            if sub.empty:
                continue
            mean = sub.mean()
            std = sub.std(ddof=1)
            n = len(sub)
            se = std / math.sqrt(n) if n > 1 else np.nan
            t = mean / se if se and se > 0 else np.nan
            rows.append(
                dict(
                    symbol=sym,
                    hour=hr,
                    n=n,
                    mean_bps=mean * 1e4,
                    std_bps=std * 1e4,
                    t=t,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Session statistics (H1, UTC)
# ---------------------------------------------------------------------------
SESSIONS = {
    "Asia":    (0, 7),    # 00:00 <= h < 07:00
    "London":  (7, 15),
    "NY":      (12, 20),
}


def session_table() -> pd.DataFrame:
    rows = []
    for sym in ALL_SYMBOLS:
        df = get_candles(sym, "H1").copy()
        df["r"] = log_ret(df["close"])
        df = df.dropna(subset=["r"]).copy()
        df["hour"] = df["timestamp"].dt.hour
        df["date"] = df["timestamp"].dt.date

        # total absolute daily move (sum of |r| per UTC day) — denominator
        daily_abs = df.groupby("date")["r"].apply(lambda s: s.abs().sum())
        total_abs = daily_abs.sum()

        for name, (h0, h1) in SESSIONS.items():
            mask = (df["hour"] >= h0) & (df["hour"] < h1)
            sub = df.loc[mask]
            if sub.empty:
                continue
            cum_ret = sub["r"].sum()
            avg_r = sub["r"].mean()
            avg_vol = sub["r"].std(ddof=1)
            session_abs = sub["r"].abs().sum()
            pct_of_move = session_abs / total_abs if total_abs > 0 else np.nan
            rows.append(
                dict(
                    symbol=sym,
                    session=name,
                    bars=len(sub),
                    cum_log_ret=cum_ret,
                    cum_pct=(math.exp(cum_ret) - 1) * 100,
                    avg_bar_ret_bps=avg_r * 1e4,
                    avg_bar_vol_bps=avg_vol * 1e4,
                    pct_of_daily_abs_move=pct_of_move * 100,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Session-open bar range vs other hours (indices + forex)
# ---------------------------------------------------------------------------
# Open hours: Asia=00, London=07, NY=12
OPEN_HOURS = {"Asia": 0, "London": 7, "NY": 12}


def session_open_range() -> pd.DataFrame:
    rows = []
    syms = FOREX + INDEX
    for sym in syms:
        df = get_candles(sym, "H1").copy()
        df["hour"] = df["timestamp"].dt.hour
        # Relative range so symbols are comparable
        df["rel_range"] = (df["high"] - df["low"]) / df["close"]
        avg_other = df["rel_range"].mean()
        for sname, oh in OPEN_HOURS.items():
            open_bar = df.loc[df["hour"] == oh, "rel_range"]
            # "adjacent" = hour-1 mod 24 and hour+1
            adj = df.loc[df["hour"].isin([(oh - 1) % 24, (oh + 1) % 24]), "rel_range"]
            # "other" = everything except this open hour
            other = df.loc[df["hour"] != oh, "rel_range"]
            if open_bar.empty:
                continue
            rows.append(
                dict(
                    symbol=sym,
                    session=sname,
                    open_hour=oh,
                    open_avg_range_bps=open_bar.mean() * 1e4,
                    adj_avg_range_bps=adj.mean() * 1e4,
                    other_avg_range_bps=other.mean() * 1e4,
                    ratio_open_vs_adj=open_bar.mean() / adj.mean() if adj.mean() > 0 else np.nan,
                    ratio_open_vs_other=open_bar.mean() / other.mean() if other.mean() > 0 else np.nan,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. End-of-month / start-of-month (D1)
# ---------------------------------------------------------------------------
def turn_of_month() -> pd.DataFrame:
    rows = []
    for sym in ALL_SYMBOLS:
        df = get_candles(sym, "D1").copy()
        df["r"] = log_ret(df["close"])
        df = df.dropna(subset=["r"]).copy()
        sd = session_date(df, sym)
        df["ym"] = sd.dt.tz_localize(None).dt.to_period("M")

        # Per month rank from start (0..n-1) and from end (0..n-1)
        df["rank_fwd"] = df.groupby("ym").cumcount()
        df["rank_back"] = df.groupby("ym").cumcount(ascending=False)

        last3_mask = df["rank_back"] < 3
        first3_mask = df["rank_fwd"] < 3
        # "middle" = neither first 3 nor last 3
        middle_mask = ~(last3_mask | first3_mask)

        def stat(mask, label):
            sub = df.loc[mask, "r"]
            if sub.empty:
                return None
            n = len(sub)
            mean = sub.mean()
            std = sub.std(ddof=1)
            se = std / math.sqrt(n) if n > 1 else np.nan
            t = mean / se if se and se > 0 else np.nan
            return dict(
                symbol=sym,
                bucket=label,
                n=n,
                mean_bps=mean * 1e4,
                std_bps=std * 1e4,
                t=t,
            )

        for mask, label in [
            (last3_mask, "last3"),
            (first3_mask, "first3"),
            (middle_mask, "middle"),
        ]:
            r = stat(mask, label)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Crypto weekend behavior
# ---------------------------------------------------------------------------
def crypto_weekend() -> pd.DataFrame:
    rows = []
    for sym in CRYPTO:
        h1 = get_candles(sym, "H1").copy()
        h1["r"] = log_ret(h1["close"])
        h1 = h1.dropna(subset=["r"]).copy()
        h1["dow"] = h1["timestamp"].dt.dayofweek  # 0=Mon..6=Sun
        weekend = h1.loc[h1["dow"].isin([5, 6]), "r"]
        weekday = h1.loc[~h1["dow"].isin([5, 6]), "r"]

        # "Monday Asia open" = Monday 00:00 UTC bar — return is log(close/open)
        h1["intra_r"] = np.log(h1["close"] / h1["open"])
        mon_asia = h1.loc[(h1["dow"] == 0) & (h1["timestamp"].dt.hour == 0), "intra_r"]

        # gap-like effect: log(Mon00 open / Fri last close)
        d1 = get_candles(sym, "D1").copy()
        d1["dow"] = d1["timestamp"].dt.dayofweek
        # Friday daily close
        fri = d1.loc[d1["dow"] == 4, ["timestamp", "close"]].rename(
            columns={"timestamp": "fri_date", "close": "fri_close"}
        )
        fri["fri_date"] = fri["fri_date"].dt.date
        # Monday Asia open hourly bar open
        mon = h1.loc[(h1["dow"] == 0) & (h1["timestamp"].dt.hour == 0),
                     ["timestamp", "open"]].rename(columns={"open": "mon_open"})
        mon["mon_date"] = mon["timestamp"].dt.date
        # Pair by matching Friday + 3 calendar days = Monday
        mon["fri_date"] = mon["timestamp"].dt.normalize() - pd.Timedelta(days=3)
        mon["fri_date"] = mon["fri_date"].dt.date
        merged = mon.merge(fri, on="fri_date", how="inner")
        merged["gap"] = np.log(merged["mon_open"] / merged["fri_close"])

        rows.append(
            dict(
                symbol=sym,
                weekend_vol_bps=weekend.std(ddof=1) * 1e4,
                weekday_vol_bps=weekday.std(ddof=1) * 1e4,
                weekend_vs_weekday_vol=weekend.std(ddof=1) / weekday.std(ddof=1),
                weekend_mean_bps=weekend.mean() * 1e4,
                weekday_mean_bps=weekday.mean() * 1e4,
                mon_asia_open_bar_mean_bps=mon_asia.mean() * 1e4,
                mon_asia_open_bar_std_bps=mon_asia.std(ddof=1) * 1e4,
                fri_to_mon_gap_mean_bps=merged["gap"].mean() * 1e4,
                fri_to_mon_gap_std_bps=merged["gap"].std(ddof=1) * 1e4,
                n_gaps=len(merged),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    out = OUT
    print(">> day-of-week")
    dow = dow_table()
    dow.to_csv(out / "agent2_dow.csv", index=False)
    print(">> hour-of-day")
    hod = hod_table()
    hod.to_csv(out / "agent2_hod.csv", index=False)
    print(">> sessions")
    sess = session_table()
    sess.to_csv(out / "agent2_sessions.csv", index=False)
    print(">> session-open ranges")
    so = session_open_range()
    so.to_csv(out / "agent2_session_open_range.csv", index=False)
    print(">> turn-of-month")
    tom = turn_of_month()
    tom.to_csv(out / "agent2_turn_of_month.csv", index=False)
    print(">> crypto weekend")
    cw = crypto_weekend()
    cw.to_csv(out / "agent2_crypto_weekend.csv", index=False)
    print("done")
    return dow, hod, sess, so, tom, cw


if __name__ == "__main__":
    main()
