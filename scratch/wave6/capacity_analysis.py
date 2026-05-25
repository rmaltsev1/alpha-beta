"""Capacity / Scalability analysis for the v14 production portfolio.

We estimate the AUM ceiling beyond which slippage erodes the OOS edge of
each sleeve, then aggregate to a portfolio capacity.

Approach
--------
1. Per-sleeve turnover (% capital traded / year) is approximated from the
   number of position changes in 2024 OOS. We use a *direction-flip* proxy
   on the sign of daily PnL as a stand-in for trade events (we don't have
   raw positions stored in the v14 panel). To translate into notional
   turnover we then multiply by the vol-target leverage required to push a
   5%-vol building block up to the 18% portfolio target (the v14 production
   leverage).

2. ADV per symbol = mean(close * volume) over the last 252 trading days.
   - Crypto volume is real base-asset units --> notional = price * volume.
   - OANDA FX/index 'volume' is tick count; we proxy:
       index  -> 1,000 USD per tick
       FX     -> 100,000 USD per tick (single contract)
   (These are admitted proxies, see report.)

3. Capacity per sleeve = AUM at which daily traded notional > 1% of the
   aggregate ADV of the symbols the sleeve trades.

4. Strategy capacity = the binding-constraint aggregation of sleeve
   capacities given v14 equal-weight TOP22 portfolio (sleeve gets 1/N of
   AUM).

5. Slippage scenarios from $1M to $10B with bps/trade impact growing as
   sqrt(AUM / capacity).

6. Bottleneck identification: sleeves whose capacity is below the
   portfolio median are flagged and we report how much Sharpe is lost when
   they are dropped at the $100M and $1B levels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "quant"))

OUT = Path(__file__).resolve().parent
QUANT = ROOT / "scratch" / "quant"
DATA = ROOT / "data"
SPLIT_OOS = pd.Timestamp("2024-01-01", tz="UTC")

# ---- v14 production composition (mirror master_v14.py) ---------------
TOP22 = ["RISKPAR", "TREND_NEW", "EVE_XAU", "D1REV_UK", "XSMOM",
         "D1REV_NAS", "WED_BTC", "DEFEND", "PAIRS_EXP",
         "VOLFORECAST", "H4_SLEEVE", "CRYPTO_vs_SPX",
         "CORR_REGIME", "SESSION_MOM",
         "W1_STRATS", "EVENT_VOLSPIKE",
         "STATARB_XS", "MICROSTR_D1", "VOL_BREAKOUT",
         "TERM_SPREADS", "EURGBP_MR", "MULTIDAY"]

PORTFOLIO_VOL_TARGET = 0.18    # v14 production target
SLEEVE_VOL = 0.05              # IS calibration target per sleeve
PORTFOLIO_LEVERAGE = PORTFOLIO_VOL_TARGET / SLEEVE_VOL  # ~3.6x

# ---- Sleeve -> symbols mapping (from sleeve scripts) -----------------
CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FOREX  = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
INDEX  = ["SPX500_USD", "NAS100_USD", "US30_USD",
          "UK100_GBP", "DE30_EUR", "JP225_USD"]
ALL_SYMBOLS = CRYPTO + FOREX + INDEX

SLEEVE_SYMBOLS: dict[str, list[str]] = {
    # Calendar / single-asset
    "EVE_XAU":        ["XAU_USD"],
    "WED_BTC":        ["BTCUSDT"],
    "WED_ETH":        ["ETHUSDT"],
    "WED_SOL":        ["SOLUSDT"],
    "D1REV_NAS":      ["NAS100_USD"],
    "D1REV_UK":       ["UK100_GBP"],
    "D1REV_SPX":      ["SPX500_USD"],
    "EURGBP_MR":      ["EUR_USD", "GBP_USD"],
    "SESSION_MOM":    ["JP225_USD"],
    # Cross-asset risk-parity / momentum
    "RISKPAR":        ALL_SYMBOLS,
    "TSMOM":          ALL_SYMBOLS,
    "XSMOM":          ALL_SYMBOLS,
    "VOLMGD":         ALL_SYMBOLS,
    "DEFEND":         ["XAU_USD", "USD_JPY"],
    "VOLFORECAST":    ALL_SYMBOLS,
    "H4_SLEEVE":      ALL_SYMBOLS,
    "TREND_NEW":      ALL_SYMBOLS,
    "TREND_COND":     ALL_SYMBOLS,
    # Crypto-only
    "CRYPTO_DOM":     CRYPTO,
    "CRYPTO_vs_SPX":  CRYPTO + ["SPX500_USD"],
    # Multi-asset cross / pairs
    "PAIRS_EXP":      ALL_SYMBOLS,
    "STATARB_XS":     ALL_SYMBOLS,
    "CORR_REGIME":    ALL_SYMBOLS,
    "MULTI_CONFIRM":  ALL_SYMBOLS,
    "MOM_QUALITY":    ALL_SYMBOLS,
    "VRP_PROXY":      ALL_SYMBOLS,
    "TAIL_SAFEHAVEN": ["XAU_USD", "USD_JPY"],
    "W1_STRATS":      ALL_SYMBOLS,
    "EVENT_VOLSPIKE": ALL_SYMBOLS,
    "MICROSTR_D1":    ALL_SYMBOLS,
    "VOL_BREAKOUT":   ALL_SYMBOLS,
    "TERM_SPREADS":   FOREX + INDEX,
    "MULTIDAY":       ALL_SYMBOLS,
}

ASSET_CLASS: dict[str, str] = (
    {s: "crypto" for s in CRYPTO}
    | {s: "fx" for s in FOREX}
    | {s: "index" for s in INDEX}
)

# Volume -> notional conversion (USD per "unit" reported by data feed)
TICK_NOTIONAL = {  # USD per OANDA tick (rough order-of-magnitude proxies)
    "fx":     100_000.0,
    "index":  1_000.0,
}


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(QUANT / "all_sleeve_returns_v14.parquet")
    p.index = pd.to_datetime(p.index, utc=True)
    return p


def compute_adv() -> pd.DataFrame:
    """Notional ADV (USD) by symbol over the trailing 252 sessions."""
    rows = []
    for sym in ALL_SYMBOLS:
        path = DATA / sym / "D1.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").tail(252)
        cls = ASSET_CLASS[sym]
        if cls == "crypto":
            # volume is base-asset units -> USD notional
            notional_per_day = df["close"] * df["volume"]
        else:
            mult = TICK_NOTIONAL[cls]
            notional_per_day = df["volume"] * mult
        adv = float(notional_per_day.mean())
        rows.append({"symbol": sym, "asset_class": cls,
                     "avg_price": float(df["close"].mean()),
                     "avg_volume": float(df["volume"].mean()),
                     "notional_ADV_usd": adv})
    return pd.DataFrame(rows).set_index("symbol")


def sleeve_turnover(panel: pd.DataFrame) -> pd.DataFrame:
    """Approximate per-sleeve annual turnover from 2024 OOS PnL.

    We have no stored positions in the v14 panel, only daily sleeve
    returns. We count *direction-flips* in the daily PnL sign as a proxy
    for round-trips: each flip closes the previous position and opens a
    new one in the opposite direction, which costs 2x notional in
    impact. Active days without a flip are assumed to be holds (no
    additional turnover).

    Round-trips per year = 2 * flips / year.
    """
    oos = panel[panel.index >= SPLIT_OOS]
    n_days = len(oos)
    years = n_days / 365.25
    rows = []
    for col in panel.columns:
        s = oos[col]
        active_days = int((s.abs() > 1e-9).sum())
        sign = np.sign(s.where(s.abs() > 1e-9, 0))
        flips = int((sign.replace(0, np.nan).ffill().diff().abs() > 0).sum())
        # 2x notional per flip (close old + open new)
        trades_per_year = 2.0 * flips / years if years > 0 else 0.0
        rows.append({
            "sleeve":          col,
            "active_days_2024": active_days,
            "flips_2024":       flips,
            "trades_per_year":  trades_per_year,
        })
    return pd.DataFrame(rows).set_index("sleeve")


def sleeve_capacity(turn: pd.DataFrame, adv: pd.DataFrame,
                    n_sleeves: int, threshold: float = 0.01) -> pd.DataFrame:
    """Per-sleeve AUM ceiling under the 1%-of-ADV rule.

    Logic: at AUM A,
       sleeve gross notional = A * (1/n_sleeves) * portfolio_leverage
       per-name notional      = sleeve_notional / n_names (sleeve diversifies)
       round-trips per name  = trades_per_year / n_names           (worst case
                                                                    full
                                                                    rotation
                                                                    is shared)
       daily traded per name = per-name notional * trades_per_year / 252
                              / n_names

    For each name we require daily-traded <= threshold * ADV_name.
    The binding name is the one with the tightest ratio (typically the
    smallest ADV). We report that.
    """
    rows = []
    sleeve_share = 1.0 / n_sleeves
    for sleeve, row in turn.iterrows():
        syms = [s for s in SLEEVE_SYMBOLS.get(sleeve, []) if s in adv.index]
        if not syms:
            rows.append({"sleeve": sleeve, "capacity_usd": np.nan,
                         "binding_symbol": None,
                         "binding_adv_usd": np.nan,
                         "n_names": 0,
                         "sleeve_share": sleeve_share})
            continue
        sleeve_advs = adv.loc[syms, "notional_ADV_usd"]
        n_names = len(syms)
        binding_sym = sleeve_advs.idxmin()
        binding_adv = float(sleeve_advs.min())
        daily_trades = row["trades_per_year"] / 252.0
        if daily_trades <= 0:
            capacity = np.inf
        else:
            # AUM * sleeve_share * lev / n_names * daily_trades <= thr*ADV_min
            capacity = (threshold * binding_adv * n_names) / (
                sleeve_share * PORTFOLIO_LEVERAGE * daily_trades
            )
        rows.append({
            "sleeve": sleeve,
            "capacity_usd": capacity,
            "binding_symbol": binding_sym,
            "binding_adv_usd": binding_adv,
            "n_names": n_names,
            "sleeve_share": sleeve_share,
        })
    return pd.DataFrame(rows).set_index("sleeve")


def slippage_scenarios(panel: pd.DataFrame, capacities: pd.DataFrame,
                       turn: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio OOS Sharpe / return at various AUM with linear
    market-impact slippage that scales as sqrt(AUM / capacity).

    Per spec, anchor slippage at these levels:
        $1M  -> 0 bp/trade extra
        $10M -> 0.5 bp/trade extra
        $100M-> 2 bp/trade extra
        $1B  -> 5 bp/trade extra
        $10B -> 15 bp/trade extra
    These anchors imply the impact constant for a sleeve at its capacity
    threshold. We then apply sleeve-specific slippage = anchor *
    sqrt(sleeve_use), where sleeve_use = AUM / sleeve_capacity, so that a
    sleeve at 100% utilization in the $1B/$10B regime gets hit hardest.
    """
    anchors = {  # bps per trade *applied to a sleeve at capacity*
        1e6:  0.0,
        1e7:  0.5,
        1e8:  2.0,
        1e9:  5.0,
        1e10: 15.0,
    }
    oos = panel[panel.index >= SPLIT_OOS].copy()
    rows = []
    for aum, base_bp in anchors.items():
        # Build slippage-adjusted sleeve panel
        adj = oos[TOP22].copy()
        for sleeve in TOP22:
            cap = capacities.loc[sleeve, "capacity_usd"] if sleeve in capacities.index else np.nan
            trades_y = turn.loc[sleeve, "trades_per_year"] if sleeve in turn.index else 0.0
            if not np.isfinite(cap) or cap <= 0:
                continue
            util = aum / cap                          # capacity utilization
            sleeve_bp = base_bp * np.sqrt(max(util, 0.0))   # sqrt-impact
            # Slippage drag per year = bp/trade * trades/year, /1e4 -> ret
            annual_drag = sleeve_bp * trades_y / 1e4
            daily_drag = annual_drag / 365.25
            adj[sleeve] = adj[sleeve] - daily_drag
        # Equal-weight TOP22 and apply portfolio leverage
        port = adj.mean(axis=1) * PORTFOLIO_LEVERAGE
        ar = float(port.mean()) * 365.25
        av = float(port.std(ddof=0)) * np.sqrt(365.25)
        sh = ar / av if av > 0 else 0.0
        eq = (1 + port).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        rows.append({
            "AUM_usd": aum,
            "bp_per_trade_anchor": base_bp,
            "OOS_return": ar,
            "OOS_vol":   av,
            "OOS_sharpe": sh,
            "OOS_max_dd": dd,
        })
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    adv = compute_adv()
    turn = sleeve_turnover(panel)

    n_top = len(TOP22)
    cap = sleeve_capacity(turn, adv, n_sleeves=n_top, threshold=0.01)

    # ----- Per-sleeve capacity table -----
    out = turn.join(cap, how="outer")
    out["asset_classes"] = [",".join(sorted({
        ASSET_CLASS.get(s, "?") for s in SLEEVE_SYMBOLS.get(sl, [])
    })) for sl in out.index]
    out["in_top22"] = out.index.isin(TOP22)
    out = out.sort_values("capacity_usd")
    out.to_csv(OUT / "capacity_per_sleeve.csv", float_format="%.6g")
    print("Saved per-sleeve capacity:", OUT / "capacity_per_sleeve.csv")

    # ----- Portfolio scenarios -----
    sc = slippage_scenarios(panel, cap, turn)
    base_sh = float(sc.loc[sc["AUM_usd"] == 1e6, "OOS_sharpe"].iloc[0])
    sc["pct_of_baseline_sharpe"] = sc["OOS_sharpe"] / base_sh
    # Find AUM where Sharpe drops by 25%
    sc["AUM_M"] = sc["AUM_usd"] / 1e6
    sc.to_csv(OUT / "capacity_scenarios.csv", index=False, float_format="%.6g")
    print("Saved portfolio scenarios:", OUT / "capacity_scenarios.csv")

    print()
    print("=== Per-sleeve capacity (TOP22 only) ===")
    in_top = out[out["in_top22"]].copy()
    in_top["capacity_M"] = in_top["capacity_usd"] / 1e6
    cols = ["asset_classes", "trades_per_year", "binding_symbol",
            "binding_adv_usd", "capacity_M"]
    print(in_top[cols].to_string(float_format=lambda x: f"{x:,.2f}"))

    print()
    print(f"=== Portfolio capacity scenarios (baseline OOS Sh = {base_sh:.2f}) ===")
    print(sc[["AUM_M", "bp_per_trade_anchor", "OOS_sharpe",
              "pct_of_baseline_sharpe", "OOS_return", "OOS_max_dd"]
            ].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Sharpe-25%-degradation AUM (linear interp on log10(AUM))
    target = 0.75 * base_sh
    s = sc.sort_values("AUM_usd")
    aum_break = np.nan
    for i in range(len(s) - 1):
        sh1, sh2 = s["OOS_sharpe"].iloc[i], s["OOS_sharpe"].iloc[i + 1]
        a1, a2 = s["AUM_usd"].iloc[i], s["AUM_usd"].iloc[i + 1]
        if sh1 >= target >= sh2:
            la = (np.log10(a1) + (target - sh1) / (sh2 - sh1) *
                  (np.log10(a2) - np.log10(a1)))
            aum_break = 10 ** la
            break
    print()
    if np.isfinite(aum_break):
        print(f"Sharpe -25% AUM ≈ ${aum_break/1e6:,.1f}M")
    else:
        print("Sharpe never degrades by 25% in the tested AUM range.")

    # Bottleneck table
    print()
    print("=== Capacity bottlenecks (TOP22 sorted by capacity) ===")
    bot = in_top.sort_values("capacity_usd").head(10)
    print(bot[["asset_classes", "binding_symbol", "trades_per_year",
               "capacity_M"]].to_string(float_format=lambda x: f"{x:,.2f}"))

    # ----- Scaling recommendations: what if we drop low-capacity sleeves? --
    print()
    print("=== Scaling alternatives: drop sleeves below threshold X ===")
    drop_rows = []
    for aum in [1e8, 1e9]:
        for keep_cap in [10e6, 50e6, 100e6, 250e6, 1e9]:
            kept = [s for s in TOP22
                    if (s in cap.index)
                    and np.isfinite(cap.loc[s, "capacity_usd"])
                    and cap.loc[s, "capacity_usd"] >= keep_cap]
            if not kept:
                continue
            sub = panel[panel.index >= SPLIT_OOS][kept].copy()
            # Apply slippage at this AUM using the same scaling
            base_bp_table = {1e6: 0.0, 1e7: 0.5, 1e8: 2.0,
                             1e9: 5.0, 1e10: 15.0}
            base_bp = base_bp_table.get(aum, 0)
            for sleeve in kept:
                c = cap.loc[sleeve, "capacity_usd"]
                ty = turn.loc[sleeve, "trades_per_year"]
                util = aum / c if c > 0 else np.inf
                bp = base_bp * np.sqrt(max(util, 0))
                annual_drag = bp * ty / 1e4
                daily_drag = annual_drag / 365.25
                sub[sleeve] = sub[sleeve] - daily_drag
            port = sub.mean(axis=1) * PORTFOLIO_LEVERAGE
            ar = float(port.mean()) * 365.25
            av = float(port.std(ddof=0)) * np.sqrt(365.25)
            sh = ar / av if av > 0 else 0
            drop_rows.append({
                "AUM_M": aum / 1e6,
                "keep_cap_M": keep_cap / 1e6,
                "n_kept": len(kept),
                "OOS_sharpe": sh,
                "OOS_return": ar,
            })
    drop_df = pd.DataFrame(drop_rows)
    print(drop_df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    drop_df.to_csv(OUT / "capacity_scaling_alternatives.csv",
                   index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
