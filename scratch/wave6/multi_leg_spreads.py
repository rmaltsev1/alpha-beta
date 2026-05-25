"""Multi-leg (3- and 4-leg) spread combinations.

Goal: extend the cointegrated 2-leg pairs from pairs_v2.py to higher-order
linear combinations. Each sleeve:
  1. Constructs a log-price linear combination y_t = w'·log(p_t) using either
     fixed prescribed weights (basket sleeves) or walk-forward OLS β on a
     hedged residual (β-hedged sleeves).
  2. Computes a 60d z-score on the residual.
  3. Gates entries with the same DF t-stat ADF approximation as pairs_v2
     (252d window, |t| < -2.86 means stationary residual).
  4. Trades mean-reversion: |z| > 2 enter, |z| < 0.5 exit, |z| > 4 stop.
  5. PnL is computed from the per-leg log returns using the same weights.

Construction list (8 sleeves):
  1. EUR-GBP-USD triangle: log(EUR_USD) + log(GBP_USD) - 2*log(EUR_GBP_synth)
     using EUR_GBP_synth = EUR_USD / GBP_USD. (Triangle is balanced exactly.)
  2. Equity region triangle: SPX vs (DE30, UK100) — walk-forward 2-leg OLS β.
  3. Risk-on basket: long (BTC+ETH+SPX+NAS)/4, short (XAU+1/USD_JPY)/2.
  4. DXY proxy: long basket of (1/EUR_USD, USD_JPY, 1/GBP_USD), short EUR_USD.
  5. Crypto cap-weighted: 0.55*BTC + 0.3*ETH + 0.15*SOL vs short SPX.
  6. DE30 beta-hedged: long DE30, short walk-forward β-fitted SPX hedge.
  7. Safe-haven triangle: long (XAU + USD_JPY)/2, short (SPX + NAS)/2.
  8. Yen triangle: USD_JPY hedged vs EUR_USD and synthetic EUR_JPY.

IS ≤ 2024-01-01, OOS ≥ 2024-01-01. Each surviving sleeve is vol-scaled to 5%
IS annualised vol. Filter: IS Sharpe ≥ 0.3 AND OOS ≥ 0. Survivors are
combined equal-weight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

from alphabeta import get_candles
from pairs_v2 import adf_t_stat, ols, _bpy, stats

OUT = Path(__file__).resolve().parent
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_VOL = 0.05
COST_PER_LEG = 0.00015  # ~1.5 bp per leg per round-trip side


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_symbols(symbols: list[str]) -> pd.DataFrame:
    """Return aligned wide log-price df indexed by date, columns = symbols."""
    parts = {}
    for s in symbols:
        c = get_candles(s, "D1")[["timestamp", "close"]].copy()
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True).dt.normalize()
        c = c.drop_duplicates("timestamp").set_index("timestamp")["close"]
        parts[s] = c
    df = pd.concat(parts, axis=1).dropna(how="any")
    return df  # raw close prices


# ---------------------------------------------------------------------------
# Generic spread engine
# ---------------------------------------------------------------------------
def trade_spread(spread: pd.Series,
                 leg_returns: pd.DataFrame,
                 weights_today: pd.DataFrame,
                 z_lookback: int = 60,
                 adf_lookback: int = 252,
                 adf_threshold: float = -2.86,
                 z_entry: float = 2.0,
                 z_exit: float = 0.5,
                 z_stop: float = 4.0) -> tuple[pd.Series, dict]:
    """Generic mean-reversion trader on a residual spread series.

    Parameters
    ----------
    spread : Series of residual values (already constructed).
    leg_returns : DataFrame of per-leg log returns (rows = dates, cols = leg id).
    weights_today : DataFrame same shape as leg_returns; weight applied to each
        leg's return at each date. Sign convention: spread = sum(w_i * log p_i),
        so PnL of a long-spread position uses the same weights on returns.
    """
    spread = spread.copy()
    n = len(spread)
    idx = spread.index
    z = pd.Series(np.nan, index=idx)
    adf = pd.Series(np.nan, index=idx)

    sp_vals = spread.values
    for i in range(z_lookback + adf_lookback, n):
        win_z = sp_vals[i - z_lookback : i]  # excludes today's value for ex-ante z
        mu = np.nanmean(win_z)
        sd = np.nanstd(win_z, ddof=0)
        if sd > 1e-9:
            z.iloc[i] = (sp_vals[i] - mu) / sd
        win_adf = sp_vals[i - adf_lookback : i + 1]
        adf.iloc[i] = adf_t_stat(win_adf)

    pos = pd.Series(0.0, index=idx)
    in_pos = 0
    for i in range(1, n):
        zv = z.iloc[i - 1]
        adv = adf.iloc[i - 1]
        if pd.isna(zv):
            pos.iloc[i] = in_pos
            continue
        if in_pos == 0:
            if pd.notna(adv) and adv < adf_threshold:
                if zv > z_entry:
                    in_pos = -1
                elif zv < -z_entry:
                    in_pos = 1
        else:
            if abs(zv) < z_exit or abs(zv) > z_stop:
                in_pos = 0
        pos.iloc[i] = in_pos

    # PnL: long-spread position * sum_i (w_i * r_i)
    leg_pnl = (leg_returns * weights_today).sum(axis=1)
    gross = pos * leg_pnl
    # Costs: |Δ(pos * w_i)| * cost per leg, summed
    pos_times_w = weights_today.mul(pos, axis=0)
    dpos_legs = pos_times_w.diff().abs().fillna(0)
    cost = dpos_legs.sum(axis=1) * COST_PER_LEG
    net = (gross - cost).fillna(0)

    diagnostics = {
        "pos": pos,
        "z": z,
        "adf": adf,
        "spread": spread,
        "n_trades": int((pos.diff().abs() > 0).sum() // 2),
    }
    return net, diagnostics


# ---------------------------------------------------------------------------
# Sleeve constructors — each returns (net_returns, n_trades)
# ---------------------------------------------------------------------------
def sleeve_eur_gbp_usd_triangle():
    """log(EUR_USD) + log(GBP_USD) - 2*log(EUR_GBP).

    EUR_GBP_synth = EUR_USD / GBP_USD, so the triangle algebraically yields:
       spread = log(EUR_USD) + log(GBP_USD) - 2*[log(EUR_USD)-log(GBP_USD)]
              = -log(EUR_USD) + 3*log(GBP_USD)
    That is a balanced linear combination; it tests EUR-vs-GBP relative valuation
    against the cross.
    """
    px = load_symbols(["EUR_USD", "GBP_USD"])
    lp = np.log(px)
    # Use explicit triangle definition (with EUR_GBP synth = EUR_USD/GBP_USD)
    leg_log_eurgbp = lp["EUR_USD"] - lp["GBP_USD"]
    spread = lp["EUR_USD"] + lp["GBP_USD"] - 2 * leg_log_eurgbp
    # That reduces to -lp[EUR_USD] + 3*lp[GBP_USD]
    leg_returns = lp.diff()
    w = pd.DataFrame({"EUR_USD": -1.0, "GBP_USD": 3.0},
                     index=lp.index)
    return trade_spread(spread, leg_returns, w)


def sleeve_equity_region_triangle():
    """Walk-forward 2-leg OLS: log(SPX) - β1*log(DE30) - β2*log(UK100).

    Long SPX, short β-fitted DE30+UK100 hedge. ADF gate on residual.
    """
    px = load_symbols(["SPX500_USD", "DE30_EUR", "UK100_GBP"])
    lp = np.log(px)
    n = len(lp)
    beta1 = np.full(n, np.nan)
    beta2 = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    LB = 252
    for i in range(LB, n):
        y = lp["SPX500_USD"].values[i - LB : i]
        x1 = lp["DE30_EUR"].values[i - LB : i]
        x2 = lp["UK100_GBP"].values[i - LB : i]
        # Solve OLS: y = a + b1*x1 + b2*x2
        X = np.column_stack([np.ones(LB), x1, x2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        _, b1, b2 = coef
        beta1[i] = b1
        beta2[i] = b2
        spread[i] = (lp["SPX500_USD"].iloc[i]
                     - b1 * lp["DE30_EUR"].iloc[i]
                     - b2 * lp["UK100_GBP"].iloc[i])
    sp = pd.Series(spread, index=lp.index)
    leg_returns = lp.diff()
    w = pd.DataFrame({
        "SPX500_USD": np.where(np.isnan(beta1), 0.0, 1.0),
        "DE30_EUR":   -beta1,
        "UK100_GBP":  -beta2,
    }, index=lp.index)
    return trade_spread(sp, leg_returns, w)


def sleeve_risk_on_off_basket():
    """Long (BTC+ETH+SPX+NAS)/4, short (XAU + 1/USD_JPY)/2.

    1/USD_JPY is equivalent to JPY/USD; log(1/USD_JPY) = -log(USD_JPY).
    So the basket equals:
       0.25*[lBTC+lETH+lSPX+lNAS] - 0.5*lXAU + 0.5*lUSD_JPY
    """
    px = load_symbols(["BTCUSDT", "ETHUSDT", "SPX500_USD", "NAS100_USD",
                       "XAU_USD", "USD_JPY"])
    lp = np.log(px)
    w_map = {
        "BTCUSDT": 0.25, "ETHUSDT": 0.25,
        "SPX500_USD": 0.25, "NAS100_USD": 0.25,
        "XAU_USD": -0.5, "USD_JPY": 0.5,  # 0.5*log(USD/JPY) opposes 0.5*log(1/USD_JPY) so flip
    }
    # Wait: basket says SHORT 1/USD_JPY. Spread = long - short = long - 1/USDJPY
    # log(1/USD_JPY) = -log(USD_JPY). Short means subtract: -[-log(USD_JPY)] = +log(USD_JPY).
    # So weight on USD_JPY should be +0.5 (consistent).
    spread = sum(w_map[s] * lp[s] for s in w_map)
    leg_returns = lp.diff()
    w = pd.DataFrame({s: w_map[s] for s in w_map}, index=lp.index)
    return trade_spread(spread, leg_returns, w)


def sleeve_dxy_proxy():
    """DXY proxy basket vs EUR_USD.

    Construct synthetic DXY = -log(EUR_USD)*0.5 + log(USD_JPY)*0.3 - log(GBP_USD)*0.2.
    Then spread = DXY_synth - (-log(EUR_USD)) = DXY_synth + log(EUR_USD).

    Net weights collapse to: 0.5*log(EUR_USD) + 0.3*log(USD_JPY) - 0.2*log(GBP_USD).
    Tests overshoot in dollar strength vs EUR-only proxy.
    """
    px = load_symbols(["EUR_USD", "USD_JPY", "GBP_USD"])
    lp = np.log(px)
    w_map = {"EUR_USD": 0.5, "USD_JPY": 0.3, "GBP_USD": -0.2}
    spread = sum(w_map[s] * lp[s] for s in w_map)
    leg_returns = lp.diff()
    w = pd.DataFrame({s: w_map[s] for s in w_map}, index=lp.index)
    return trade_spread(spread, leg_returns, w)


def sleeve_crypto_capwt_vs_spx():
    """0.55*BTC + 0.30*ETH + 0.15*SOL vs short SPX (β-hedged).

    Walk-forward OLS β of the crypto basket vs SPX, then trade residual.
    """
    px = load_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT", "SPX500_USD"])
    lp = np.log(px)
    basket = 0.55 * lp["BTCUSDT"] + 0.30 * lp["ETHUSDT"] + 0.15 * lp["SOLUSDT"]
    n = len(lp)
    beta = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    LB = 252
    bvals = basket.values
    svals = lp["SPX500_USD"].values
    for i in range(LB, n):
        b, _ = ols(bvals[i - LB : i], svals[i - LB : i])
        beta[i] = b
        spread[i] = bvals[i] - b * svals[i]
    sp = pd.Series(spread, index=lp.index)
    leg_returns = lp.diff()
    w = pd.DataFrame({
        "BTCUSDT": np.where(np.isnan(beta), 0.0, 0.55),
        "ETHUSDT": np.where(np.isnan(beta), 0.0, 0.30),
        "SOLUSDT": np.where(np.isnan(beta), 0.0, 0.15),
        "SPX500_USD": -beta,
    }, index=lp.index)
    return trade_spread(sp, leg_returns, w)


def sleeve_de30_beta_hedged():
    """Long DE30, short walk-forward β·SPX hedge — DE30 idiosyncratic residual."""
    px = load_symbols(["DE30_EUR", "SPX500_USD"])
    lp = np.log(px)
    n = len(lp)
    beta = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    LB = 252
    dvals = lp["DE30_EUR"].values
    svals = lp["SPX500_USD"].values
    for i in range(LB, n):
        b, _ = ols(dvals[i - LB : i], svals[i - LB : i])
        # Force β toward task spec hint of 0.5 — but use OLS for proper hedge.
        # Cap at reasonable bounds.
        b = float(np.clip(b, 0.1, 2.0))
        beta[i] = b
        spread[i] = dvals[i] - b * svals[i]
    sp = pd.Series(spread, index=lp.index)
    leg_returns = lp.diff()
    w = pd.DataFrame({
        "DE30_EUR": np.where(np.isnan(beta), 0.0, 1.0),
        "SPX500_USD": -beta,
    }, index=lp.index)
    return trade_spread(sp, leg_returns, w)


def sleeve_safe_haven_triangle():
    """Long (XAU + USD_JPY)/2, short (SPX + NAS)/2.

    Tests safe-haven flow patterns: gold + dollar against equity risk basket.
    """
    px = load_symbols(["XAU_USD", "USD_JPY", "SPX500_USD", "NAS100_USD"])
    lp = np.log(px)
    w_map = {"XAU_USD": 0.5, "USD_JPY": 0.5, "SPX500_USD": -0.5, "NAS100_USD": -0.5}
    spread = sum(w_map[s] * lp[s] for s in w_map)
    leg_returns = lp.diff()
    w = pd.DataFrame({s: w_map[s] for s in w_map}, index=lp.index)
    return trade_spread(spread, leg_returns, w)


def sleeve_yen_triangle():
    """Yen triangle: long USD_JPY, short EUR_USD, short synthetic EUR_JPY hedge.

    EUR_JPY_synth = EUR_USD * USD_JPY, so log(EUR_JPY_synth) = log(EUR_USD) + log(USD_JPY).
    Triangle:
       spread = log(USD_JPY) - log(EUR_USD) - 0.5*log(EUR_JPY_synth)
              = log(USD_JPY) - log(EUR_USD) - 0.5*[log(EUR_USD)+log(USD_JPY)]
              = 0.5*log(USD_JPY) - 1.5*log(EUR_USD)
    Captures yen-specific moves not explained by EUR cross.
    """
    px = load_symbols(["USD_JPY", "EUR_USD"])
    lp = np.log(px)
    w_map = {"USD_JPY": 0.5, "EUR_USD": -1.5}
    spread = sum(w_map[s] * lp[s] for s in w_map)
    leg_returns = lp.diff()
    w = pd.DataFrame({s: w_map[s] for s in w_map}, index=lp.index)
    return trade_spread(spread, leg_returns, w)


# ---------------------------------------------------------------------------
# Volatility scaling + reporting
# ---------------------------------------------------------------------------
def scale_to_target(rets: pd.Series, target_vol: float = TARGET_VOL):
    is_part = rets[rets.index < SPLIT]
    if len(is_part) < 30:
        return rets * 0.0, 0.0
    av = float(is_part.std(ddof=0)) * np.sqrt(365.25)
    if av <= 1e-9:
        return rets * 0.0, 0.0
    k = target_vol / av
    return rets * k, k


def year_sharpe(rets: pd.Series, year: int) -> float:
    sub = rets[rets.index.year == year]
    if len(sub) < 10:
        return 0.0
    bpy = _bpy(sub.index)
    ar = float(sub.mean()) * bpy
    av = float(sub.std(ddof=0)) * np.sqrt(bpy)
    return ar / av if av > 0 else 0.0


def main():
    sleeves = [
        ("EUR_GBP_USD_TRI", sleeve_eur_gbp_usd_triangle),
        ("EQ_REGION_TRI",   sleeve_equity_region_triangle),
        ("RISK_ON_OFF",     sleeve_risk_on_off_basket),
        ("DXY_PROXY",       sleeve_dxy_proxy),
        ("CRYPTO_CAPWT",    sleeve_crypto_capwt_vs_spx),
        ("DE30_HEDGED",     sleeve_de30_beta_hedged),
        ("SAFE_HAVEN_TRI",  sleeve_safe_haven_triangle),
        ("YEN_TRI",         sleeve_yen_triangle),
    ]

    print(f"{'Sleeve':<18} {'IS_Sh':>6} {'OOS_Sh':>7} {'2022_Sh':>8} "
          f"{'#Trd':>5} {'scale':>6} {'survive':>8}")
    print("-" * 75)

    rows = []
    streams = {}
    for name, fn in sleeves:
        try:
            rets, diag = fn()
            rets = rets.dropna()
            scaled, k = scale_to_target(rets, TARGET_VOL)
            s = stats(name, scaled)
            n_trades = diag["n_trades"]
            sh22 = year_sharpe(scaled, 2022)
            is_sh = s.get("IS_sharpe", 0)
            oos_sh = s.get("OOS_sharpe", 0)
            survive = (is_sh >= 0.3) and (oos_sh >= 0)
            rows.append({
                "sleeve": name, "scale": k, "n_trades": n_trades,
                "2022_sharpe": sh22,
                "IS_sharpe": is_sh, "OOS_sharpe": oos_sh,
                "FULL_sharpe": s.get("FULL_sharpe", 0),
                "IS_ret": s.get("IS_ret", 0), "OOS_ret": s.get("OOS_ret", 0),
                "FULL_dd": s.get("FULL_dd", 0),
                "survive": bool(survive),
            })
            streams[name] = scaled
            mark = " *" if survive else ""
            print(f"{name:<18} {is_sh:>+6.2f} {oos_sh:>+7.2f} "
                  f"{sh22:>+8.2f} {n_trades:>5d} {k:>6.2f} {str(survive):>8}{mark}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{name}: ERROR {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multi_leg_breakdown.csv", index=False)

    survivors = [r["sleeve"] for r in rows if r["survive"]]
    print(f"\nSurvivors (IS≥0.3, OOS≥0): {len(survivors)} of {len(sleeves)} — {survivors}")

    if survivors:
        combined = pd.concat({k: streams[k] for k in survivors}, axis=1,
                             sort=True).fillna(0).mean(axis=1)
        s_combined = stats("MULTI_LEG", combined)
        print(f"\nCombined MULTI_LEG sleeve ({len(survivors)} survivors, equal-weight):")
        for tag in ["FULL", "IS", "OOS"]:
            print(f"  {tag:<4} Sharpe={s_combined.get(f'{tag}_sharpe', 0):+.2f}  "
                  f"Return={s_combined.get(f'{tag}_ret', 0):+.2%}  "
                  f"DD={s_combined.get(f'{tag}_dd', 0):+.2%}")
        for year in sorted(set(combined.index.year)):
            sub = combined[combined.index.year == year]
            if len(sub) < 50: continue
            bpy = _bpy(sub.index)
            sh = sub.mean() * bpy / (sub.std(ddof=0) * np.sqrt(bpy)) if sub.std() > 0 else 0
            print(f"  {year}  Sharpe={sh:+.2f}  Ret={sub.sum():+.2%}")

        out_df = pd.DataFrame({"timestamp": combined.index, "ret": combined.values})
        out_df.to_parquet(OUT / "multi_leg_returns.parquet", index=False)
        print(f"\nSaved combined returns to {OUT / 'multi_leg_returns.parquet'}")

        # Correlations vs reference sleeves
        ref_files = {
            "PAIRS_EXP": ROOT / "scratch" / "quant" / "pairs_expanded_returns.parquet",
            "CRYPTO_vs_SPX_v1": ROOT / "scratch" / "wave3" if (ROOT/"scratch/wave3").exists() else None,
        }
        # PAIRS_EXP
        pe_path = ROOT / "scratch" / "quant" / "pairs_expanded_returns.parquet"
        if pe_path.exists():
            pe = pd.read_parquet(pe_path)
            pe["timestamp"] = pd.to_datetime(pe["timestamp"], utc=True)
            pe = pe.set_index("timestamp")["ret"]
            joined = pd.concat([combined.rename("ML"), pe.rename("PE")], axis=1).dropna()
            if len(joined) > 30:
                corr = joined.corr().iloc[0, 1]
                print(f"\nCorrelation MULTI_LEG vs PAIRS_EXP: {corr:+.3f}")
                joined_oos = joined[joined.index >= SPLIT]
                if len(joined_oos) > 10:
                    corr_oos = joined_oos.corr().iloc[0, 1]
                    print(f"Correlation MULTI_LEG vs PAIRS_EXP (OOS): {corr_oos:+.3f}")
    else:
        print("\nNo survivors — saving zero series for combined.")
        any_stream = streams[list(streams.keys())[0]] if streams else None
        if any_stream is not None:
            zero = pd.Series(0.0, index=any_stream.index)
            out_df = pd.DataFrame({"timestamp": zero.index, "ret": zero.values})
            out_df.to_parquet(OUT / "multi_leg_returns.parquet", index=False)


if __name__ == "__main__":
    main()
