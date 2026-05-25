"""Risk-parity (ERC, equal risk contribution) long-only sleeve, D1.

Methodology
-----------
1. For each symbol, compute D1 log returns.
2. Align all 13 symbols on a common (date-normalized) calendar. Missing days
   (weekends / holidays / pre-listing) get NaN; the return panel is filled
   with 0 only where the symbol traded but the *other* symbols didn't move.
3. Each Monday close, using a rolling 90-day window of past log returns
   strictly *before* the rebalance bar, estimate Sigma_t (sample covariance).
4. Compute inverse-vol weights:   w_i = (1/sigma_i) / sum_j (1/sigma_j),
   then refine with 1 Newton step on the full ERC condition
   (sigma_i * (Sigma w)_i constant across i), long-only.
5. Hold those weights through the week.
6. Portfolio-level vol overlay: scale total notional by
   leverage_t = TARGET_VOL / realized_30d_vol(portfolio_unscaled),
   capped at LEV_CAP.

For comparison:
  - Equal-weight 1/13 buy-and-hold (rebalanced never).
  - 60/40 SPX500_USD / BTCUSDT, rebalanced weekly.

Outputs
-------
  scratch/quant/risk_parity.py                  (this file, re-runnable)
  scratch/quant/risk_parity_returns.parquet     (timestamp UTC, ret)
  scratch/quant/risk_parity_comparison.csv      (3 rows x IS/OOS stats)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import get_candles, ALL_SYMBOLS, SYMBOL_TYPE
from alphabeta.backtest import cost_for


# -- knobs --------------------------------------------------------------------
COV_WIN = 90                # rolling window (bars) for covariance estimation
VOL_OVERLAY_WIN = 30        # backward window for portfolio realized-vol overlay
TARGET_VOL = 0.10           # 10% ann at portfolio level
LEV_CAP = 3.0               # cap on overlay leverage (safety)
REBAL_DOW = 0               # Monday = 0  (date.weekday())
SPLIT = "2024-01-01"
BARS_PER_YEAR = 365.25
NEWTON_ITERS = 1            # ERC refinement iterations
OUT_DIR = Path(__file__).resolve().parent


# -- panel construction -------------------------------------------------------
def build_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (log_ret_panel, simple_ret_panel) indexed by UTC date.

    Both panels have one column per symbol. Cells are NaN where the symbol
    did not trade that calendar day.
    """
    log_cols = {}
    simp_cols = {}
    for s in ALL_SYMBOLS:
        df = get_candles(s, "D1").copy()
        # Normalize the per-asset native timestamp to its calendar date in UTC
        # so we can align FX (close ~21:00) with crypto (00:00).
        df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
        df = df.drop_duplicates(subset="date", keep="last").set_index("date")
        close = df["close"].astype("float64")
        log_ret = np.log(close / close.shift(1))
        simp_ret = close.pct_change()
        log_cols[s] = log_ret
        simp_cols[s] = simp_ret

    log_panel = pd.concat(log_cols, axis=1).sort_index()
    simp_panel = pd.concat(simp_cols, axis=1).sort_index()
    # Tz-aware UTC midnight index.
    log_panel.index = pd.DatetimeIndex(log_panel.index, name="date")
    simp_panel.index = pd.DatetimeIndex(simp_panel.index, name="date")
    return log_panel, simp_panel


# -- weight solvers -----------------------------------------------------------
def inv_vol_weights(sigma_diag: np.ndarray) -> np.ndarray:
    inv = np.where(sigma_diag > 0, 1.0 / sigma_diag, 0.0)
    tot = inv.sum()
    return inv / tot if tot > 0 else np.full_like(inv, 1.0 / len(inv))


def erc_weights(cov: np.ndarray, n_iters: int = NEWTON_ITERS) -> np.ndarray:
    """Long-only ERC weights starting from inverse-vol heuristic.

    The ERC condition is  w_i * (Sigma w)_i  equal across i (each asset
    contributes equally to portfolio variance). We iterate the classic
    multiplicative fixed-point update from Maillard, Roncalli, Teiletche (2010):

        w_i  <-  w_i * sqrt( target / (w_i * (Sigma w)_i) )
        w    <-  w / sum(w)

    which is monotone toward the ERC solution under PSD Sigma. After n_iters,
    we renormalize. Long-only is enforced by clipping at 0.
    """
    n = cov.shape[0]
    sig = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    w = inv_vol_weights(sig)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.full(n, 1.0 / n)
    for _ in range(n_iters):
        Sw = cov @ w
        contrib = w * Sw
        target = contrib.mean()
        # Avoid div-by-zero
        ratio = np.where(contrib > 1e-18, target / contrib, 1.0)
        w = w * np.sqrt(ratio)
        w = np.clip(w, 0.0, None)
        tot = w.sum()
        if tot <= 0:
            return np.full(n, 1.0 / n)
        w = w / tot
    return w


# -- weight schedule construction (walk-forward) ------------------------------
def compute_rp_weights(log_panel: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward risk-parity weights, rebalanced each Monday.

    Weight at row t uses log returns strictly before t (i.e. through t-1).
    Symbols that don't have COV_WIN observations yet get 0.
    """
    dates = log_panel.index
    syms = log_panel.columns.tolist()
    n = len(syms)

    # For symbols with NaN in the lookback window (didn't trade that day),
    # treat the missing return as 0 for covariance estimation.
    filled = log_panel.fillna(0.0)
    avail = log_panel.notna()  # True if the symbol traded that calendar day.

    weights = pd.DataFrame(0.0, index=dates, columns=syms)
    # Track most-recent weights so we hold between rebalances.
    last_w = np.zeros(n)

    for i, dt in enumerate(dates):
        if i < COV_WIN + 1:
            weights.iloc[i] = last_w
            continue
        # Rebalance on Monday only; otherwise hold.
        if dt.weekday() != REBAL_DOW:
            weights.iloc[i] = last_w
            continue

        window = filled.iloc[i - COV_WIN : i].to_numpy()  # uses rows up to t-1
        # Only include symbols that have any nonzero observation in window.
        active = (np.abs(window).sum(axis=0) > 0)
        # Also require the symbol to be available at rebalance bar t.
        active = active & avail.iloc[i].to_numpy()

        if active.sum() < 2:
            weights.iloc[i] = last_w
            continue

        w_full = np.zeros(n)
        sub = window[:, active]
        # Sample covariance (biased / N is fine — drift is negligible vs. 90 obs)
        cov = np.cov(sub, rowvar=False, ddof=0)
        # Symmetrize + tiny ridge for numerical stability
        cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(cov.shape[0])
        w_active = erc_weights(cov, n_iters=NEWTON_ITERS)
        w_full[active] = w_active
        last_w = w_full
        weights.iloc[i] = last_w

    return weights


# -- portfolio return given a weight schedule ---------------------------------
def apply_costs_and_returns(
    weights: pd.DataFrame,
    simp_panel: pd.DataFrame,
    cps_vec: np.ndarray,
) -> pd.Series:
    """Combine target weights and bar simple returns into a portfolio stream.

    Cost = sum_i cps_i * |dw_i| each bar (charged on weight changes).
    Missing simple-return cells (symbol not trading) -> 0 contribution.
    """
    simp = simp_panel.reindex(weights.index).fillna(0.0).to_numpy()
    w = weights.to_numpy()
    # Use previous bar's weights to earn current bar's return: shift by 1.
    w_eff = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    gross = (w_eff * simp).sum(axis=1)
    dw = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1]))))
    cost = dw @ cps_vec
    net = gross - cost
    return pd.Series(net, index=weights.index, name="ret")


def vol_overlay(returns: pd.Series, target_vol: float, win: int, cap: float) -> pd.Series:
    """Scale a return series so trailing realized vol matches target.

    leverage[t] uses returns strictly *before* t (shift by 1).
    """
    realized = returns.rolling(win, min_periods=win).std(ddof=0) * np.sqrt(BARS_PER_YEAR)
    lev = (target_vol / realized).clip(upper=cap).shift(1)
    lev = lev.fillna(0.0)
    return returns * lev, lev


# -- baselines ----------------------------------------------------------------
def equal_weight_returns(simp_panel: pd.DataFrame, cps_vec: np.ndarray) -> pd.Series:
    """1/13 buy-and-hold. Initialize at first row where >= 2 symbols available;
    we keep weights constant -> no rebalance costs after initial allocation.
    Missing-symbol returns are 0 -> drift is naturally absorbed.
    """
    n = simp_panel.shape[1]
    w0 = np.full(n, 1.0 / n)
    simp = simp_panel.fillna(0.0).to_numpy()
    # Effective bar weights: zero on bar 0 (we just initialized), w0 thereafter.
    w_eff = np.vstack([np.zeros((1, n)), np.tile(w0, (simp.shape[0] - 1, 1))])
    gross = (w_eff * simp).sum(axis=1)
    # One-time turnover at first bar.
    cost = np.zeros(simp.shape[0])
    cost[0] = (np.abs(w0) * cps_vec).sum()
    net = gross - cost
    return pd.Series(net, index=simp_panel.index, name="ret")


def sixty_forty_returns(
    simp_panel: pd.DataFrame, cps_vec: np.ndarray, sym_cps: dict[str, float]
) -> pd.Series:
    """60% SPX500_USD / 40% BTCUSDT, weekly Monday rebalance."""
    syms = simp_panel.columns.tolist()
    target = np.zeros(len(syms))
    target[syms.index("SPX500_USD")] = 0.60
    target[syms.index("BTCUSDT")] = 0.40

    dates = simp_panel.index
    weights = pd.DataFrame(0.0, index=dates, columns=syms)
    last_w = np.zeros(len(syms))
    for i, dt in enumerate(dates):
        if i == 0 or dt.weekday() == REBAL_DOW:
            last_w = target.copy()
        weights.iloc[i] = last_w

    return apply_costs_and_returns(weights, simp_panel, cps_vec)


# -- stats --------------------------------------------------------------------
def sleeve_stats(returns: pd.Series, freq: float = BARS_PER_YEAR) -> dict:
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_dd": 0.0}
    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "max_dd": float(dd),
    }


def split_stats(ret: pd.Series) -> dict:
    split_ts = pd.Timestamp(SPLIT, tz="UTC")
    is_r = ret[ret.index < split_ts]
    oos_r = ret[ret.index >= split_ts]
    full = sleeve_stats(ret)
    is_s = sleeve_stats(is_r)
    oos_s = sleeve_stats(oos_r)
    return {
        "full_sharpe": full["sharpe"],     "full_ret": full["ann_return"],
        "full_vol":   full["ann_vol"],     "full_dd":  full["max_dd"],
        "is_sharpe":  is_s["sharpe"],      "is_ret":   is_s["ann_return"],
        "is_vol":     is_s["ann_vol"],     "is_dd":    is_s["max_dd"],
        "oos_sharpe": oos_s["sharpe"],     "oos_ret":  oos_s["ann_return"],
        "oos_vol":    oos_s["ann_vol"],    "oos_dd":   oos_s["max_dd"],
    }


# -- driver -------------------------------------------------------------------
def main() -> None:
    log_panel, simp_panel = build_panels()
    syms = list(simp_panel.columns)
    cps_vec = np.array([cost_for(s) for s in syms], dtype="float64")

    # ---- risk-parity weights -------------------------------------------------
    rp_weights = compute_rp_weights(log_panel)
    rp_unscaled = apply_costs_and_returns(rp_weights, simp_panel, cps_vec)
    rp_scaled, lev = vol_overlay(rp_unscaled, TARGET_VOL, VOL_OVERLAY_WIN, LEV_CAP)
    rp_scaled.name = "ret"

    # ---- baselines -----------------------------------------------------------
    ew_ret = equal_weight_returns(simp_panel, cps_vec)
    sf_ret = sixty_forty_returns(simp_panel, cps_vec, {s: c for s, c in zip(syms, cps_vec)})

    # ---- save parquet -------------------------------------------------------
    out_parquet = OUT_DIR / "risk_parity_returns.parquet"
    out_df = pd.DataFrame({
        "timestamp": pd.to_datetime(rp_scaled.index, utc=True),
        "ret": rp_scaled.values,
    })
    out_df.to_parquet(out_parquet, index=False)

    # ---- comparison CSV -----------------------------------------------------
    rows = []
    for name, r in [("risk_parity", rp_scaled), ("equal_weight", ew_ret), ("60_40", sf_ret)]:
        d = split_stats(r)
        d["sleeve"] = name
        rows.append(d)
    comp = pd.DataFrame(rows).set_index("sleeve")
    comp = comp[
        [
            "full_sharpe", "full_ret", "full_vol", "full_dd",
            "is_sharpe",   "is_ret",   "is_vol",   "is_dd",
            "oos_sharpe",  "oos_ret",  "oos_vol",  "oos_dd",
        ]
    ]
    comp.to_csv(OUT_DIR / "risk_parity_comparison.csv")

    # ---- report -------------------------------------------------------------
    def block(label: str, r: pd.Series) -> None:
        st = sleeve_stats(r)
        print(f"  {label:<14} Sharpe={st['sharpe']:+5.2f} "
              f"Ret={st['ann_return']:+7.2%} Vol={st['ann_vol']:6.2%} "
              f"DD={st['max_dd']:+7.2%} n={len(r)}")

    print("=== RISK-PARITY SLEEVE ===")
    split_ts = pd.Timestamp(SPLIT, tz="UTC")
    for name, r in [("RP FULL", rp_scaled), ("RP IS", rp_scaled[rp_scaled.index < split_ts]),
                     ("RP OOS", rp_scaled[rp_scaled.index >= split_ts])]:
        block(name, r)

    print("\n=== BASELINES ===")
    for name, r in [("EW FULL", ew_ret), ("EW IS", ew_ret[ew_ret.index < split_ts]),
                     ("EW OOS", ew_ret[ew_ret.index >= split_ts]),
                     ("60/40 FULL", sf_ret), ("60/40 IS", sf_ret[sf_ret.index < split_ts]),
                     ("60/40 OOS", sf_ret[sf_ret.index >= split_ts])]:
        block(name, r)

    # Year-by-year
    print("\n=== YEAR-BY-YEAR SHARPES ===")
    print(f"{'year':<6}{'risk_parity':>13}{'equal_weight':>14}{'60_40':>10}")
    years = sorted({d.year for d in rp_scaled.index})
    for y in years:
        rp_y = rp_scaled[rp_scaled.index.year == y]
        ew_y = ew_ret[ew_ret.index.year == y]
        sf_y = sf_ret[sf_ret.index.year == y]
        a = sleeve_stats(rp_y)["sharpe"]
        b = sleeve_stats(ew_y)["sharpe"]
        c = sleeve_stats(sf_y)["sharpe"]
        print(f"{y:<6}{a:>13.2f}{b:>14.2f}{c:>10.2f}")

    # Yearly returns (helps for the 2022 question)
    print("\n=== YEAR-BY-YEAR RETURNS ===")
    print(f"{'year':<6}{'risk_parity':>13}{'equal_weight':>14}{'60_40':>10}")
    for y in years:
        rp_y = rp_scaled[rp_scaled.index.year == y]
        ew_y = ew_ret[ew_ret.index.year == y]
        sf_y = sf_ret[sf_ret.index.year == y]
        a = sleeve_stats(rp_y)["ann_return"]
        b = sleeve_stats(ew_y)["ann_return"]
        c = sleeve_stats(sf_y)["ann_return"]
        print(f"{y:<6}{a:>+12.2%}{b:>+13.2%}{c:>+9.2%}")

    print("\n=== AVERAGE RP WEIGHTS (across rebalanced bars, weight != 0) ===")
    # Average over bars where rp_weights has any nonzero (i.e. solver active)
    active_mask = (rp_weights.abs().sum(axis=1) > 0)
    avg_w = rp_weights[active_mask].mean(axis=0)
    grouped = []
    for s in syms:
        grouped.append((s, SYMBOL_TYPE[s].value, float(avg_w[s])))
    grouped.sort(key=lambda kv: -kv[2])
    for s, ac, w in grouped:
        print(f"  {s:<12s} [{ac:<6s}]  {w:6.2%}")
    print("  -- by class --")
    for ac in ("forex", "index", "crypto"):
        tot = sum(w for _, a, w in grouped if a == ac)
        print(f"  {ac:<6s}  {tot:6.2%}")

    # Average leverage applied by overlay
    print(f"\nMean overlay leverage = {lev[lev > 0].mean():.2f}, "
          f"median = {lev[lev > 0].median():.2f}")

    print(f"\nWrote {out_parquet}")
    print(f"Wrote {OUT_DIR / 'risk_parity_comparison.csv'}")


if __name__ == "__main__":
    main()
