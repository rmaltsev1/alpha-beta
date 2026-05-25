"""Crypto-specific alpha hunt — wave 3.

Tests seven crypto-native ideas across BTC/ETH/SOL on D1 (and H1 where
tactical entry matters). Each sub-sleeve is:
  - signal computed walk-forward (no look-ahead beyond canonical conv:
    position[t] depends on data observable at start of bar t — we shift)
  - vol-scaled to 5% IS annualized vol
  - backtested with 5bp/side crypto cost via alphabeta.backtest

Survivor filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0.

Outputs:
  - scratch/wave3/crypto_alpha_returns.parquet  (D1 sleeve returns, UTC)
  - scratch/wave3/crypto_alpha_breakdown.csv    (per sub-sleeve stats)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from alphabeta import get_candles
from alphabeta.backtest import backtest, cost_for

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
CRYPTOS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ----- utilities ----------------------------------------------------------

def _bpy(idx):
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / 86400
    return len(idx) / span * 365.25 if span > 0 else 252.0


def stats(label, r):
    """Compute Sharpe/return for FULL / IS / OOS / 2022 windows."""
    out = {"label": label}
    r = r.dropna()
    if len(r) < 5:
        return out
    windows = [
        ("FULL", pd.Series(True, index=r.index)),
        ("IS",   r.index < SPLIT),
        ("OOS",  r.index >= SPLIT),
        ("Y2022", (r.index >= pd.Timestamp("2022-01-01", tz="UTC"))
                   & (r.index < pd.Timestamp("2023-01-01", tz="UTC"))),
    ]
    for tag, mask in windows:
        sub = r[mask]
        if len(sub) < 5:
            out[f"{tag}_sharpe"] = 0.0
            out[f"{tag}_ret"] = 0.0
            continue
        bpy = _bpy(sub.index)
        ar = float(sub.mean()) * bpy
        av = float(sub.std(ddof=0)) * np.sqrt(bpy)
        out[f"{tag}_sharpe"] = ar/av if av > 0 else 0.0
        out[f"{tag}_ret"] = ar
        out[f"{tag}_vol"] = av
    return out


def scale_to_is_vol(rets: pd.Series, target: float = TARGET_VOL) -> float:
    """Vol-scale factor from IS subset of a return series."""
    is_r = rets[rets.index < SPLIT].dropna()
    if len(is_r) < 30:
        return 0.0
    bpy = _bpy(is_r.index)
    av = float(is_r.std(ddof=0) * np.sqrt(bpy))
    return target / av if av > 1e-9 else 0.0


def run_signal(df: pd.DataFrame, signal: pd.Series, name: str, symbol: str,
               timeframe: str = "D1"):
    """Run a position series through backtest; return (timestamped returns, scale)."""
    pos = pd.Series(signal.values, index=df.index, dtype="float64").fillna(0.0)
    res = backtest(df, pos, symbol=symbol, timeframe=timeframe, name=name)
    idx = pd.to_datetime(df["timestamp"].values, utc=True)
    rets_unscaled = pd.Series(res.returns.values, index=idx)
    scale = scale_to_is_vol(rets_unscaled, TARGET_VOL)
    rets_scaled = rets_unscaled * scale
    return rets_scaled, scale, res


def to_daily(rets: pd.Series) -> pd.Series:
    """Resample arbitrary-frequency returns to UTC daily by summing within each day."""
    if rets.empty:
        return rets
    idx = pd.DatetimeIndex(rets.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    rets = pd.Series(rets.values, index=idx)
    return rets.resample("1D").sum()


# ----- load data ----------------------------------------------------------

print("Loading data...")
DATA_D1 = {s: get_candles(s, "D1") for s in CRYPTOS}
DATA_H1 = {s: get_candles(s, "H1") for s in CRYPTOS}
for s, df in DATA_D1.items():
    print(f"  {s} D1: {len(df)} rows  {df.timestamp.iloc[0].date()} -> {df.timestamp.iloc[-1].date()}")


def ts_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(df["timestamp"].values, utc=True)


# ----- IDEA 1: BTC dominance trend ---------------------------------------
# Long BTC when ETH/BTC and SOL/BTC ratios are in 10-day downtrend (BTC dominant).
# When ratio uptrends, alts are running — long ETH/SOL.

def idea1_dominance_trend():
    print("\n=== IDEA 1: BTC dominance trend ===")
    btc = DATA_D1["BTCUSDT"].copy()
    eth = DATA_D1["ETHUSDT"].copy()
    sol = DATA_D1["SOLUSDT"].copy()

    btc_i = btc.set_index("timestamp")["close"]
    eth_i = eth.set_index("timestamp")["close"]
    sol_i = sol.set_index("timestamp")["close"]

    common = btc_i.index.intersection(eth_i.index).intersection(sol_i.index)
    btc_a = btc_i.reindex(common)
    eth_a = eth_i.reindex(common)
    sol_a = sol_i.reindex(common)

    eth_btc = eth_a / btc_a
    sol_btc = sol_a / btc_a

    LOOK = 10
    eth_btc_ret = eth_btc.pct_change(LOOK)
    sol_btc_ret = sol_btc.pct_change(LOOK)

    # dominance score: average of -ratio_return (higher = BTC dominant)
    dom_score = -(eth_btc_ret + sol_btc_ret) / 2.0
    dom_score = dom_score.shift(1)  # only act on yesterday's info

    sleeves = {}

    # Sub-sleeve: long BTC when dominant (dom_score > 0), else flat
    for sym, ratio_ret in [("BTCUSDT", dom_score),
                            ("ETHUSDT", -dom_score),
                            ("SOLUSDT", -dom_score)]:
        df_sym = DATA_D1[sym].copy()
        ratio_aligned = ratio_ret.reindex(df_sym["timestamp"]).values
        sig = np.where(ratio_aligned > 0, 1.0, np.where(ratio_aligned < 0, 0.0, 0.0))
        # Convert to walk-forward long/flat (long when in this coin's favor)
        sig = pd.Series(sig, index=df_sym.index).fillna(0.0)
        rets, scale, _ = run_signal(df_sym, sig, f"DOM_{sym}", sym)
        s = stats(f"dom_{sym}", rets)
        s["scale"] = scale
        sleeves[f"dom_{sym}"] = (rets, s)
        print(f"  dom_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- IDEA 2: ETH/BTC mean-reversion at extremes ------------------------
# z-score of log(ratio) on 60-day window; fade |z|>2.

def idea2_ratio_meanrev():
    print("\n=== IDEA 2: Ratio mean-reversion ===")
    pairs = [("ETHUSDT", "BTCUSDT"), ("SOLUSDT", "BTCUSDT"), ("SOLUSDT", "ETHUSDT")]
    WIN = 60
    Z_THRESH = 2.0
    HOLD = 5  # bars

    sleeves = {}
    for num_sym, den_sym in pairs:
        num = DATA_D1[num_sym].set_index("timestamp")["close"]
        den = DATA_D1[den_sym].set_index("timestamp")["close"]
        common = num.index.intersection(den.index)
        ratio = np.log(num.reindex(common) / den.reindex(common))
        m = ratio.rolling(WIN).mean()
        sd = ratio.rolling(WIN).std()
        z = (ratio - m) / sd
        z_shift = z.shift(1)

        # When z>2 → ratio over-extended UP → short num + long den (mean reversion)
        # When z<-2 → over-extended DOWN → long num + short den
        # Implement as a pair: position on num = -sign(z) when |z|>thresh, else 0
        # We'll express this as separate positions on the two legs.
        raw_num_sig = pd.Series(0.0, index=z_shift.index)
        raw_num_sig[z_shift > Z_THRESH] = -1.0
        raw_num_sig[z_shift < -Z_THRESH] = +1.0
        # Hold for HOLD bars after entry — replicate by rolling forward-fill of nonzero
        # Build "holding" by carrying forward last nonzero for up to HOLD bars
        held = raw_num_sig.copy()
        last_sig = 0.0
        last_idx = -HOLD
        out_sig = []
        for i, v in enumerate(raw_num_sig.values):
            if v != 0:
                last_sig = v
                last_idx = i
                out_sig.append(v)
            elif i - last_idx < HOLD:
                out_sig.append(last_sig)
            else:
                out_sig.append(0.0)
        held = pd.Series(out_sig, index=raw_num_sig.index)
        den_sig = -held

        for sym_role, sig_ts in [(num_sym, held), (den_sym, den_sig)]:
            df_sym = DATA_D1[sym_role].copy()
            sig_aligned = sig_ts.reindex(df_sym["timestamp"]).fillna(0.0)
            sig = pd.Series(sig_aligned.values, index=df_sym.index)
            tag = f"mr_{num_sym[:3]}_{den_sym[:3]}_{sym_role[:3]}"
            rets, scale, _ = run_signal(df_sym, sig, tag, sym_role)
            s = stats(tag, rets)
            s["scale"] = scale
            sleeves[tag] = (rets, s)
            print(f"  {tag:<22}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
                  f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- IDEA 3: Cross-coin lead-lag (BTC prior 3d → SOL next day) ---------

def idea3_leadlag():
    print("\n=== IDEA 3: BTC -> SOL lead-lag (3d) ===")
    btc = DATA_D1["BTCUSDT"].set_index("timestamp")["close"]
    sol = DATA_D1["SOLUSDT"].set_index("timestamp")["close"]
    eth = DATA_D1["ETHUSDT"].set_index("timestamp")["close"]

    sleeves = {}
    for src_sym, tgt_sym, src_px, tgt_px in [
        ("BTCUSDT", "SOLUSDT", btc, sol),
        ("BTCUSDT", "ETHUSDT", btc, eth),
        ("ETHUSDT", "SOLUSDT", eth, sol),
    ]:
        common = src_px.index.intersection(tgt_px.index)
        src_a = src_px.reindex(common)
        tgt_a = tgt_px.reindex(common)
        src_3d = src_a.pct_change(3).shift(1)  # observable at start of next bar
        # Position on target = sign(src_3d), held 1 bar
        df_sym = DATA_D1[tgt_sym].copy()
        sig_aligned = src_3d.reindex(df_sym["timestamp"]).fillna(0.0)
        sig = np.sign(sig_aligned.values)
        sig = pd.Series(sig, index=df_sym.index).fillna(0.0)
        tag = f"ll_{src_sym[:3]}_to_{tgt_sym[:3]}"
        rets, scale, _ = run_signal(df_sym, sig, tag, tgt_sym)
        s = stats(tag, rets)
        s["scale"] = scale
        sleeves[tag] = (rets, s)
        print(f"  {tag:<22}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- IDEA 4: Weekend gap fill on H1 -------------------------------------
# Fri close -> Mon Asia open often gaps. Short gap (mean-reversion) on Mon
# at hours 0-7 UTC, then flat.

def idea4_weekend_gap():
    print("\n=== IDEA 4: Weekend gap fill (H1) ===")
    sleeves = {}
    for sym in CRYPTOS:
        df = DATA_H1[sym].copy()
        df["dow"] = df["timestamp"].dt.dayofweek  # Mon=0
        df["hour"] = df["timestamp"].dt.hour

        # For each Monday hour 0 bar, compute gap = (Mon00 close - Fri close) / Fri close
        # We need previous Friday's last close. Easier: take close at Sat 00 UTC
        # (Crypto: candles trade through weekend, but liquidity dips. "Gap" approximate
        # as: 48-hour return into Mon 00 UTC).

        # Compute return from previous Fri 22:00 UTC to current bar
        close = df["close"]
        # 48h lookback: ~48 hours = 48 bars on H1
        ret_48 = close.pct_change(48)  # ~Fri 00 -> Sun 00
        # We define gap as ret_24 from Sun 00 -> Mon 00 (small) but the canonical
        # weekend gap effect: position taken at Mon 00 UTC = -sign(ret over Sat+Sun)
        # Use the 48h return ending at this bar.
        df["gap48"] = ret_48.shift(1)  # info at start of bar t

        # Signal: only on Mon, hours 0-7 inclusive: short the gap direction
        # i.e. position = -sign(gap48)
        mask = (df["dow"] == 0) & (df["hour"].between(0, 7))
        sig = pd.Series(0.0, index=df.index)
        sig[mask] = -np.sign(df["gap48"][mask].fillna(0.0))
        sig = sig.fillna(0.0)

        rets, scale, _ = run_signal(df, sig, f"WKD_{sym}", sym, timeframe="H1")
        # Convert H1 returns to daily for downstream aggregation
        daily = to_daily(rets)
        s = stats(f"wkd_{sym}", daily)
        s["scale"] = scale
        sleeves[f"wkd_{sym}"] = (daily, s)
        print(f"  wkd_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- IDEA 5: Volume-price divergence: new 20d high on low volume -------

def idea5_vol_price_div():
    print("\n=== IDEA 5: Volume-price divergence ===")
    sleeves = {}
    for sym in CRYPTOS:
        df = DATA_D1[sym].copy()
        close = df["close"]
        vol = df["volume"]
        high20 = close.rolling(20).max()
        vol20 = vol.rolling(20).mean()

        # New 20d high AND volume below 20d avg → bearish divergence → short 3 days
        new_high = (close >= high20) & (vol < vol20)
        new_high_shift = new_high.shift(1).fillna(False)
        # Hold short for 3 days
        sig = pd.Series(0.0, index=df.index)
        last_signal_idx = -10
        for i, v in enumerate(new_high_shift.values):
            if v:
                last_signal_idx = i
                sig.iloc[i] = -1.0
            elif i - last_signal_idx < 3:
                sig.iloc[i] = -1.0
        rets, scale, _ = run_signal(df, sig, f"VPD_{sym}", sym)
        s = stats(f"vpd_{sym}", rets)
        s["scale"] = scale
        sleeves[f"vpd_{sym}"] = (rets, s)
        print(f"  vpd_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- IDEA 6: High-vol short / blow-off top -----------------------------

def idea6_blowoff_top():
    print("\n=== IDEA 6: Blow-off top short ===")
    sleeves = {}
    for sym in CRYPTOS:
        df = DATA_D1[sym].copy()
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        # 30d realized vol
        rv30 = log_ret.rolling(30).std() * np.sqrt(365)
        # 30d return
        ret30 = close.pct_change(30)
        # IS calibration window for top decile
        is_mask = df["timestamp"] < SPLIT
        rv_is = rv30[is_mask].dropna()
        if len(rv_is) < 50:
            continue
        rv_p90 = float(np.quantile(rv_is, 0.90))

        # Condition observable yesterday → trigger today
        trigger = (rv30.shift(1) >= rv_p90) & (ret30.shift(1) >= 0.20)
        # Hold short for 5 days
        sig = pd.Series(0.0, index=df.index)
        last_idx = -10
        for i, v in enumerate(trigger.fillna(False).values):
            if v:
                last_idx = i
                sig.iloc[i] = -1.0
            elif i - last_idx < 5:
                sig.iloc[i] = -1.0
        rets, scale, _ = run_signal(df, sig, f"BOT_{sym}", sym)
        s = stats(f"bot_{sym}", rets)
        s["scale"] = scale
        sleeves[f"bot_{sym}"] = (rets, s)
        print(f"  bot_{sym:<10}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}  rv_p90={rv_p90:.2%}")
    return sleeves


# ----- IDEA 7: Funding-rate proxy: BTC up >10%, ETH lagging --------------
# When BTC has rallied >10% in 7d but ETH <5%, expect snap-back: short BTC / long ETH 3d.
# Also try reverse: ETH up, BTC lagging → long BTC / short ETH.

def idea7_funding_proxy():
    print("\n=== IDEA 7: BTC vs ETH momentum snap-back ===")
    btc = DATA_D1["BTCUSDT"].set_index("timestamp")["close"]
    eth = DATA_D1["ETHUSDT"].set_index("timestamp")["close"]
    common = btc.index.intersection(eth.index)
    btc_a = btc.reindex(common)
    eth_a = eth.reindex(common)

    btc7 = btc_a.pct_change(7)
    eth7 = eth_a.pct_change(7)

    # BTC overshooting: short BTC / long ETH
    overshoot_btc = (btc7 > 0.10) & (eth7 < 0.05)
    # ETH overshooting: long BTC / short ETH
    overshoot_eth = (eth7 > 0.10) & (btc7 < 0.05)

    # Hold 3 days, position observable next bar
    overshoot_btc_s = overshoot_btc.shift(1).fillna(False)
    overshoot_eth_s = overshoot_eth.shift(1).fillna(False)

    HOLD = 3
    sig_btc = pd.Series(0.0, index=common)
    sig_eth = pd.Series(0.0, index=common)
    last_b = -10; last_e = -10
    for i in range(len(common)):
        if overshoot_btc_s.iloc[i]:
            last_b = i
        if overshoot_eth_s.iloc[i]:
            last_e = i
        btc_pos = 0.0; eth_pos = 0.0
        if i - last_b < HOLD:
            btc_pos += -1.0; eth_pos += +1.0
        if i - last_e < HOLD:
            btc_pos += +1.0; eth_pos += -1.0
        # cap at +-1
        sig_btc.iloc[i] = max(-1.0, min(1.0, btc_pos))
        sig_eth.iloc[i] = max(-1.0, min(1.0, eth_pos))

    sleeves = {}
    for sym, sig_ts in [("BTCUSDT", sig_btc), ("ETHUSDT", sig_eth)]:
        df_sym = DATA_D1[sym].copy()
        sig_aligned = sig_ts.reindex(df_sym["timestamp"]).fillna(0.0)
        sig = pd.Series(sig_aligned.values, index=df_sym.index)
        tag = f"fnd_{sym[:3]}"
        rets, scale, _ = run_signal(df_sym, sig, tag, sym)
        s = stats(tag, rets)
        s["scale"] = scale
        sleeves[tag] = (rets, s)
        print(f"  {tag:<22}  IS_Sh={s.get('IS_sharpe',0):+.2f}  "
              f"OOS_Sh={s.get('OOS_sharpe',0):+.2f}  Y2022={s.get('Y2022_sharpe',0):+.2f}")
    return sleeves


# ----- MAIN ---------------------------------------------------------------

def main():
    all_sleeves = {}
    all_sleeves.update(idea1_dominance_trend())
    all_sleeves.update(idea2_ratio_meanrev())
    all_sleeves.update(idea3_leadlag())
    all_sleeves.update(idea4_weekend_gap())
    all_sleeves.update(idea5_vol_price_div())
    all_sleeves.update(idea6_blowoff_top())
    all_sleeves.update(idea7_funding_proxy())

    # Build breakdown
    rows = []
    for name, (rets, s) in all_sleeves.items():
        rows.append({"sleeve": name, **s})
    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(OUT / "crypto_alpha_breakdown.csv", index=False)

    # Filter: IS Sharpe >= 0.5 AND OOS Sharpe >= 0
    survivors = breakdown[(breakdown["IS_sharpe"] >= 0.5) & (breakdown["OOS_sharpe"] >= 0)]
    print(f"\n=== Survivors ({len(survivors)} / {len(breakdown)}) ===")
    if not survivors.empty:
        print(survivors[["sleeve", "IS_sharpe", "OOS_sharpe", "Y2022_sharpe",
                         "FULL_sharpe", "FULL_ret"]].to_string(index=False))

    # Daily-aligned survivor return panel
    survivor_names = survivors["sleeve"].tolist()
    panel = {}
    for name in survivor_names:
        rets = all_sleeves[name][0]
        # ensure daily
        idx = pd.DatetimeIndex(rets.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        # Re-normalize index to date
        s = pd.Series(rets.values, index=idx)
        # All D1 sleeves are already daily; H1 weekend was converted to_daily.
        s.index = s.index.normalize()
        panel[name] = s

    if panel:
        ret_df = pd.concat(panel, axis=1, sort=True).fillna(0.0)
        ret_df.index = ret_df.index.tz_convert("UTC") if ret_df.index.tz else ret_df.index.tz_localize("UTC")
        # Save with timestamp column
        out_df = ret_df.reset_index().rename(columns={"index": "timestamp"})
        out_df.to_parquet(OUT / "crypto_alpha_returns.parquet", index=False)

        # Combined sleeve
        combined = ret_df.mean(axis=1)
        s_c = stats("CRYPTO_ALPHA_COMBINED", combined)
        print(f"\n=== Combined survivor sleeve (equal-weight) ===")
        for tag in ["FULL", "IS", "OOS", "Y2022"]:
            sh = s_c.get(f"{tag}_sharpe", 0)
            rt = s_c.get(f"{tag}_ret", 0)
            print(f"  {tag:<6}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}")

        # Year-by-year
        print("\n=== Combined yearly Sharpe ===")
        for year, sub in combined.groupby(combined.index.year):
            if len(sub) < 20:
                continue
            bpy = _bpy(sub.index)
            std = sub.std(ddof=0)
            sh = (sub.mean() * bpy) / (std * np.sqrt(bpy)) if std > 0 else 0.0
            rt = sub.mean() * bpy
            print(f"  {year}  Sharpe={sh:+.2f}  AnnRet={rt:+.2%}  Bars={len(sub)}")
    else:
        # Still write empty file with timestamp column for downstream
        pd.DataFrame({"timestamp": pd.DatetimeIndex([], tz="UTC")}).to_parquet(
            OUT / "crypto_alpha_returns.parquet", index=False)
        print("\nNo survivors.")


if __name__ == "__main__":
    main()
