"""Wave-6: 2-state Hidden Markov Regime model (numpy-only EM).

Fit a 2-state Gaussian HMM to SPX D1 log returns. State 0 = "bull" (higher
mean, lower variance); state 1 = "bear" (lower mean, higher variance).
EM via forward-backward (Baum-Welch). Online filter via Viterbi-style
forward pass for production-time state probabilities.

Walk-forward: refit every 6 months on a trailing 252-day window. Within
each refit window, the most recent fit's parameters generate online state
probabilities for the following 6 months (causal: filter only).

Strategies (vol-scaled to 5% IS ann vol):
  HMM_BULL_TSMOM    bull-only TSMOM across 13 symbols
  HMM_BEAR_REV      bear-only D1 reversion on equity indices
  HMM_SLEEVE_MIX    bull -> trend tilt, bear -> reversion tilt
  HMM_REGIME_FLIP   tactical short SPX 5 days on bull->bear (long on bear->bull)
  HMM_VOLTGT        vol target = 12% in bull, 8% in bear (on SPX TSMOM)

Methodology:
  IS  < 2024-01-01;  OOS >= 2024-01-01.
  Filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0.
  HMM is refit walk-forward every 6 months on the trailing 252 D1 bars of
  SPX. Out-of-window state probabilities come from the online filter
  applied with the most recent fit's parameters.

Outputs:
  scratch/wave6/hmm_regime.py
  scratch/wave6/hmm_returns.parquet     -- per-strategy daily returns
  scratch/wave6/hmm_states.parquet      -- bull/bear filtered probabilities
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphabeta import (
    ALL_SYMBOLS, CRYPTO, FOREX, INDEX, get_candles, SYMBOL_TYPE,
)
from alphabeta.backtest import cost_for


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scratch" / "wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TARGET_SUB_VOL = 0.05

# HMM knobs
HMM_WIN = 252            # bars per refit window
HMM_REFIT_EVERY = 126    # bars between refits (~6 months)
HMM_MAX_ITER = 100       # EM iterations
HMM_TOL = 1e-5

# Strategy knobs
LOOKBACKS_TSMOM = [21, 63, 126, 252]
VOL_WIN = 60
REV_THRESH_BPS = 50.0

INDICES_EQ = ["SPX500_USD", "NAS100_USD", "US30_USD",
              "UK100_GBP", "DE30_EUR", "JP225_USD"]


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------
def ann_factor(symbol: str) -> float:
    return 365.0 if symbol in CRYPTO else 252.0


def perf_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return dict(sharpe=0.0, ann_return=0.0, ann_vol=0.0, max_dd=0.0,
                    n=int(len(r)))
    ann_ret = r.mean() * bpy
    ann_vol = r.std(ddof=0) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return dict(sharpe=float(sharpe), ann_return=float(ann_ret),
                ann_vol=float(ann_vol), max_dd=dd, n=int(len(r)))


def split_stats(ret: pd.Series, bpy: float = 252.0) -> dict:
    r = ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    is_mask = idx < SPLIT
    y22 = (idx >= pd.Timestamp("2022-01-01", tz="UTC")) & \
          (idx < pd.Timestamp("2023-01-01", tz="UTC"))
    return {
        "FULL":  perf_stats(r, bpy),
        "IS":    perf_stats(r[is_mask], bpy),
        "OOS":   perf_stats(r[~is_mask], bpy),
        "Y2022": perf_stats(r[y22], bpy),
    }


def vol_scale_is(ret: pd.Series, target: float = TARGET_SUB_VOL,
                 bpy: float = 252.0) -> tuple[pd.Series, float]:
    r = ret.dropna()
    if len(r) == 0:
        return ret * 0.0, 0.0
    idx = pd.DatetimeIndex(r.index)
    is_r = r[idx < SPLIT]
    iv = is_r.std(ddof=0) * np.sqrt(bpy)
    if iv <= 0 or not np.isfinite(iv):
        return ret * 0.0, 0.0
    k = target / iv
    return ret * k, float(k)


def to_daily(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index).floor("1D")
    s = s.groupby(level=0).sum()
    return s


# ---------------------------------------------------------------------------
# HMM (numpy only)
# ---------------------------------------------------------------------------
def _gauss_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """log N(x | mu, var). x: (T,). Returns (T,)."""
    var = max(float(var), 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


def _forward_backward(x: np.ndarray, A: np.ndarray, mu: np.ndarray,
                      var: np.ndarray, pi: np.ndarray):
    """Run the forward-backward in log space for a 2-state Gaussian HMM.

    Returns
        gamma : (T, 2) posterior p(state=k | x_{1:T}, theta)
        xi_sum: (2, 2) sum_t p(s_t=i, s_{t+1}=j | x_{1:T})
        ll    : float scalar log-likelihood
    """
    T = len(x)
    K = 2
    # emission log-prob (T, K)
    logB = np.zeros((T, K))
    for k in range(K):
        logB[:, k] = _gauss_logpdf(x, mu[k], var[k])
    logA = np.log(np.maximum(A, 1e-300))
    logpi = np.log(np.maximum(pi, 1e-300))

    # forward in log space
    log_alpha = np.zeros((T, K))
    log_alpha[0] = logpi + logB[0]
    for t in range(1, T):
        # log_alpha[t, j] = logB[t, j] + logsumexp_i(log_alpha[t-1, i] + logA[i, j])
        m = np.max(log_alpha[t - 1])
        for j in range(K):
            log_alpha[t, j] = logB[t, j] + m + np.log(
                np.sum(np.exp(log_alpha[t - 1] - m) * A[:, j]) + 1e-300
            )

    # backward
    log_beta = np.zeros((T, K))
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j(logA[i, j] + logB[t+1, j] + log_beta[t+1, j])
        v = logA + (logB[t + 1] + log_beta[t + 1])[None, :]  # (K, K)
        m = np.max(v, axis=1)
        log_beta[t] = m + np.log(np.sum(np.exp(v - m[:, None]), axis=1) + 1e-300)

    # log-likelihood
    m = np.max(log_alpha[-1])
    ll = m + np.log(np.sum(np.exp(log_alpha[-1] - m)))

    # gamma
    log_gamma = log_alpha + log_beta - ll
    gamma = np.exp(log_gamma)
    # numerical normalization
    gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    # xi summed over t
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        log_xi_t = (log_alpha[t][:, None] + logA
                    + logB[t + 1][None, :] + log_beta[t + 1][None, :] - ll)
        xi_t = np.exp(log_xi_t)
        s = xi_t.sum()
        if s > 0:
            xi_t = xi_t / s
        xi_sum += xi_t
    return gamma, xi_sum, float(ll)


def fit_hmm(x: np.ndarray, max_iter: int = HMM_MAX_ITER,
            tol: float = HMM_TOL, seed: int = 0):
    """Fit a 2-state Gaussian HMM via EM (Baum-Welch).

    Initializes state 0 with above-median returns, state 1 below.
    Returns dict with A, mu, var, pi, ll_history.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype="float64")
    T = len(x)
    K = 2
    # init means by median split: bull = upper half, bear = lower half
    med = np.median(x)
    up = x[x >= med]
    dn = x[x < med]
    mu = np.array([up.mean() if len(up) else 0.0,
                   dn.mean() if len(dn) else 0.0])
    var = np.array([up.var() if len(up) > 1 else x.var() + 1e-8,
                    dn.var() if len(dn) > 1 else x.var() + 1e-8])
    var = np.maximum(var, 1e-10)
    A = np.array([[0.95, 0.05],
                  [0.05, 0.95]])
    pi = np.array([0.6, 0.4])

    ll_hist = []
    prev_ll = -np.inf
    for it in range(max_iter):
        gamma, xi_sum, ll = _forward_backward(x, A, mu, var, pi)
        ll_hist.append(ll)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        A_new = xi_sum / np.maximum(row_sums, 1e-300)
        # mu / var
        w = gamma  # (T, K)
        Nk = w.sum(axis=0)
        mu_new = (w * x[:, None]).sum(axis=0) / np.maximum(Nk, 1e-300)
        var_new = (w * (x[:, None] - mu_new[None, :]) ** 2).sum(axis=0) \
                   / np.maximum(Nk, 1e-300)
        var_new = np.maximum(var_new, 1e-10)

        A = A_new
        mu = mu_new
        var = var_new

        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll

    # Canonicalize: bull (state 0) = higher mean.
    if mu[1] > mu[0]:
        mu = mu[[1, 0]]
        var = var[[1, 0]]
        A = A[np.ix_([1, 0], [1, 0])]
        pi = pi[[1, 0]]
        # gamma columns will be swapped when we re-run forward; not needed here
    return dict(A=A, mu=mu, var=var, pi=pi, ll=ll_hist)


def online_filter(x: np.ndarray, model: dict) -> np.ndarray:
    """Forward-only filtered probabilities p(s_t | x_{1:t}).
    Returns (T, 2) array. Pure numpy, log space.
    """
    A = model["A"]; mu = model["mu"]; var = model["var"]; pi = model["pi"]
    T = len(x); K = 2
    logA = np.log(np.maximum(A, 1e-300))
    logpi = np.log(np.maximum(pi, 1e-300))
    logB = np.zeros((T, K))
    for k in range(K):
        logB[:, k] = _gauss_logpdf(x, mu[k], var[k])
    log_alpha = np.zeros((T, K))
    log_alpha[0] = logpi + logB[0]
    # normalize each step (filtered probs)
    m = np.max(log_alpha[0])
    log_norm = m + np.log(np.sum(np.exp(log_alpha[0] - m)))
    log_alpha[0] -= log_norm
    for t in range(1, T):
        m = np.max(log_alpha[t - 1])
        for j in range(K):
            log_alpha[t, j] = logB[t, j] + m + np.log(
                np.sum(np.exp(log_alpha[t - 1] - m) * A[:, j]) + 1e-300
            )
        m = np.max(log_alpha[t])
        log_norm = m + np.log(np.sum(np.exp(log_alpha[t] - m)))
        log_alpha[t] -= log_norm
    return np.exp(log_alpha)


# ---------------------------------------------------------------------------
# Walk-forward HMM on SPX D1 log returns
# ---------------------------------------------------------------------------
def walk_forward_hmm(spx_df: pd.DataFrame) -> pd.DataFrame:
    """Run walk-forward HMM on SPX D1 and return DataFrame with:
       timestamp, logret, p_bull (filtered), p_bear, state (argmax).
    """
    df = spx_df.copy()
    df["logret"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["logret"]).reset_index(drop=True)
    x = df["logret"].to_numpy()
    T = len(x)

    p_bull = np.full(T, np.nan)
    p_bear = np.full(T, np.nan)

    # We need an initial fit before producing any filtered probs.
    # Fit on first HMM_WIN bars; for indices [HMM_WIN .. HMM_WIN+HMM_REFIT_EVERY)
    # use online filter seeded from that fit, applied incrementally.
    # Refit each HMM_REFIT_EVERY bars on the trailing HMM_WIN bars.
    if T < HMM_WIN + 1:
        return pd.DataFrame({
            "timestamp": df["timestamp"], "logret": x,
            "p_bull": p_bull, "p_bear": p_bear,
        })

    # Anchor refit boundaries: first refit at t = HMM_WIN, then at HMM_WIN+REFIT_EVERY, ...
    refit_anchors = list(range(HMM_WIN, T, HMM_REFIT_EVERY))
    # ensure last anchor doesn't overshoot
    if refit_anchors[-1] >= T:
        refit_anchors = refit_anchors[:-1]

    model = None
    for i, anchor in enumerate(refit_anchors):
        train = x[anchor - HMM_WIN: anchor]
        model = fit_hmm(train, seed=i)
        # Filter the *training window itself* to produce IS-style filtered probs
        # for those bars too — but only fill those that haven't been filled yet
        # (we don't overwrite already-filtered probs from earlier walk-forward
        # steps; first pass we DO fill the entire trailing window).
        g_train = online_filter(train, model)
        if i == 0:
            # backfill the training window
            p_bull[anchor - HMM_WIN: anchor] = g_train[:, 0]
            p_bear[anchor - HMM_WIN: anchor] = g_train[:, 1]

        # forward filter from this anchor up to next anchor (or end of series)
        nxt = refit_anchors[i + 1] if i + 1 < len(refit_anchors) else T
        # Seed the online filter at `anchor` with the last training-window state
        # posterior (provides continuity instead of resetting to pi).
        last_state = g_train[-1]  # (2,)
        # Build a chain forward: alpha[anchor-1] = last_state; then advance.
        # Implement an incremental step in log-space.
        logA = np.log(np.maximum(model["A"], 1e-300))
        log_alpha = np.log(np.maximum(last_state, 1e-300))
        for t in range(anchor, nxt):
            # emission
            logB_t = np.array([
                _gauss_logpdf(np.array([x[t]]), model["mu"][0], model["var"][0])[0],
                _gauss_logpdf(np.array([x[t]]), model["mu"][1], model["var"][1])[0],
            ])
            m = np.max(log_alpha)
            new = np.zeros(2)
            for j in range(2):
                new[j] = logB_t[j] + m + np.log(
                    np.sum(np.exp(log_alpha - m) * model["A"][:, j]) + 1e-300
                )
            log_alpha = new
            # normalize
            mn = np.max(log_alpha)
            ln = mn + np.log(np.sum(np.exp(log_alpha - mn)))
            log_alpha -= ln
            g = np.exp(log_alpha)
            p_bull[t] = g[0]
            p_bear[t] = g[1]

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "logret": x,
        "p_bull": p_bull,
        "p_bear": p_bear,
    })
    out["state"] = np.where(out["p_bull"] >= 0.5, 0, 1)  # 0=bull, 1=bear
    out["state"] = out["state"].where(out["p_bull"].notna(), other=np.nan)
    return out


# ---------------------------------------------------------------------------
# Helpers to align HMM state to each symbol's calendar (causal, backward-fill).
# ---------------------------------------------------------------------------
def align_state_to_symbol(state_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Left-join state probabilities to df's timestamps with backward search."""
    left = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"], utc=True)})
    right = state_df.copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    left = left.sort_values("timestamp").reset_index(drop=True)
    right = right.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(left, right[["timestamp", "p_bull", "p_bear", "state"]],
                           on="timestamp", direction="backward")
    return merged  # columns: timestamp, p_bull, p_bear, state


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------
def position_to_returns(df: pd.DataFrame, pos: pd.Series, symbol: str) -> pd.Series:
    pos = pos.astype("float64").fillna(0.0)
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = pos.values * bar_ret.values
    cps = cost_for(symbol)
    dpos = np.zeros_like(pos.values)
    dpos[0] = abs(pos.values[0])
    dpos[1:] = np.abs(np.diff(pos.values))
    net = gross - dpos * cps
    s = pd.Series(net, index=pd.to_datetime(df["timestamp"], utc=True), name=symbol)
    return s


def tsmom_position(df: pd.DataFrame, lookback: int, target_vol: float = 0.10,
                   bpy: float = 252.0) -> pd.Series:
    close = df["close"].astype("float64").to_numpy()
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    s = pd.Series(log_ret, index=df.index)
    trail = s.rolling(lookback, min_periods=lookback).sum()
    raw = np.sign(trail)
    rv = s.rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=0) * np.sqrt(bpy)
    scale = (target_vol / rv).where(rv > 0)
    pos = (raw * scale).clip(-2.0, 2.0).fillna(0.0)
    return pos.shift(1).fillna(0.0)


def d1rev_position(df: pd.DataFrame, threshold_bps: float = REV_THRESH_BPS) -> pd.Series:
    ret = np.log(df["close"] / df["close"].shift(1))
    thresh = threshold_bps / 10_000.0
    sig = pd.Series(0.0, index=df.index)
    sig[ret > thresh] = -1.0
    sig[ret < -thresh] = +1.0
    return sig.shift(1).fillna(0.0)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def strat_bull_tsmom(data: dict, state_df: pd.DataFrame) -> pd.Series:
    """Bull-only TSMOM across all 13 symbols, equal-weight per-symbol.
    Trade TSMOM signal only when filtered bull prob >= 0.5 (state=bull).
    """
    streams = []
    for sym in ALL_SYMBOLS:
        df = data[sym]
        bpy = ann_factor(sym)
        # build avg TSMOM position
        pos_avg = pd.Series(0.0, index=df.index)
        for L in LOOKBACKS_TSMOM:
            pos_avg = pos_avg + tsmom_position(df, L, bpy=bpy)
        pos_avg = pos_avg / len(LOOKBACKS_TSMOM)
        # state filter
        st = align_state_to_symbol(state_df, df)
        bull = (st["state"].fillna(0).values == 0).astype(float)
        # only trade in bull; otherwise flat
        pos = pos_avg.values * bull
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


def strat_bear_reversion(data: dict, state_df: pd.DataFrame) -> pd.Series:
    """Bear-only D1 reversion across equity indices."""
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        pos_raw = d1rev_position(df)
        st = align_state_to_symbol(state_df, df)
        bear = (st["state"].fillna(1).values == 1).astype(float)
        pos = pd.Series(pos_raw.values * bear, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


def strat_sleeve_mix(data: dict, state_df: pd.DataFrame) -> pd.Series:
    """Bull state: weight=1.0 trend, 0.3 reversion. Bear: 0.3 trend, 1.0 reversion."""
    streams = []
    for sym in INDICES_EQ:
        df = data[sym]
        bpy = ann_factor(sym)
        pos_trend = pd.Series(0.0, index=df.index)
        for L in LOOKBACKS_TSMOM:
            pos_trend = pos_trend + tsmom_position(df, L, bpy=bpy)
        pos_trend = pos_trend / len(LOOKBACKS_TSMOM)
        pos_rev = d1rev_position(df)
        st = align_state_to_symbol(state_df, df)
        p_bull = st["p_bull"].fillna(0.5).values
        # Smooth mix: bull weight on trend = p_bull, on rev = (1-p_bull)*0.3 + p_bull*0.0
        w_trend = 1.0 * p_bull + 0.3 * (1.0 - p_bull)
        w_rev   = 0.3 * p_bull + 1.0 * (1.0 - p_bull)
        pos = pos_trend.values * w_trend + pos_rev.values * w_rev
        pos = pd.Series(pos, index=df.index)
        ret = position_to_returns(df, pos, sym)
        streams.append(to_daily(ret).rename(sym))
    panel = pd.concat(streams, axis=1, sort=True).fillna(0.0)
    return panel.mean(axis=1)


def strat_regime_flip(spx_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.Series:
    """Tactical SPX short on bull->bear flip, long on bear->bull flip. Hold 5 days."""
    df = spx_df.copy()
    st = align_state_to_symbol(state_df, df)
    state = st["state"].values
    n = len(df)
    pos = np.zeros(n)
    hold = 0
    cur = 0.0
    for t in range(1, n):
        if hold > 0:
            pos[t] = cur
            hold -= 1
            continue
        s_prev = state[t - 1]
        # earlier transition
        s_pp = state[t - 2] if t >= 2 else s_prev
        if np.isnan(s_prev) or np.isnan(s_pp):
            continue
        if s_pp == 0 and s_prev == 1:  # bull -> bear: short for 5 days
            cur = -1.0
            pos[t] = cur
            hold = 4  # 5 days total
        elif s_pp == 1 and s_prev == 0:  # bear -> bull: long for 5 days
            cur = +1.0
            pos[t] = cur
            hold = 4
    pos = pd.Series(pos, index=df.index)
    ret = position_to_returns(df, pos, "SPX500_USD")
    return to_daily(ret)


def strat_voltgt(spx_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.Series:
    """SPX TSMOM with state-dependent vol target: bull=12%, bear=8%."""
    df = spx_df.copy()
    # average TSMOM signal (sign only — vol scaling applied per state below)
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    sigs = []
    for L in LOOKBACKS_TSMOM:
        trail = log_ret.rolling(L, min_periods=L).sum()
        sigs.append(np.sign(trail))
    raw = pd.concat(sigs, axis=1).mean(axis=1)
    raw = raw.shift(1).fillna(0.0)

    rv = log_ret.rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=0) * np.sqrt(252.0)
    st = align_state_to_symbol(state_df, df)
    p_bull = st["p_bull"].fillna(0.5).values
    target = 0.12 * p_bull + 0.08 * (1.0 - p_bull)
    scale = pd.Series(target, index=df.index) / rv
    scale = scale.where(rv > 0).fillna(0.0).clip(-2.0, 2.0)
    pos = (raw.values * scale.values)
    pos = pd.Series(pos, index=df.index)
    ret = position_to_returns(df, pos, "SPX500_USD")
    return to_daily(ret)


# ---------------------------------------------------------------------------
# Diagnostics: existing simple regime gate (SPX vol > IS p80)
# ---------------------------------------------------------------------------
def simple_vol_gate_states(spx_df: pd.DataFrame) -> pd.DataFrame:
    """Classify state by SPX 30d realized vol > IS p80 = bear, else bull."""
    df = spx_df.copy()
    df["logret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv30"] = df["logret"].rolling(30).std()
    df = df.dropna(subset=["logret"]).reset_index(drop=True)
    is_rv = df.loc[df["timestamp"] < SPLIT, "rv30"].dropna()
    p80 = float(is_rv.quantile(0.80))
    df["state_simple"] = np.where(df["rv30"] > p80, 1, 0)  # 1 = bear/high-vol
    return df[["timestamp", "rv30", "state_simple"]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("Loading SPX D1...")
    spx = get_candles("SPX500_USD", "D1")
    spx = spx.sort_values("timestamp").reset_index(drop=True)

    print(f"SPX D1: {len(spx)} bars from {spx['timestamp'].min()} -> {spx['timestamp'].max()}")

    print("Running walk-forward HMM fit...")
    state_df = walk_forward_hmm(spx)
    n_total = len(state_df)
    n_filled = int(state_df["p_bull"].notna().sum())
    print(f"State series: {n_filled} / {n_total} bars filled")
    # Diagnostic: how often is HMM in bear state?
    bear_share = float((state_df["state"] == 1).sum()) / n_filled
    print(f"Bear-state share (filtered): {bear_share:.2%}")

    # 2022 vs full
    s22 = state_df[(state_df["timestamp"] >= pd.Timestamp("2022-01-01", tz="UTC")) &
                   (state_df["timestamp"] <  pd.Timestamp("2023-01-01", tz="UTC"))]
    bear22 = float((s22["state"] == 1).sum()) / max(int(s22["state"].notna().sum()), 1)
    print(f"2022 bear-state share: {bear22:.2%}")

    # Save state probabilities
    state_out = state_df[["timestamp", "logret", "p_bull", "p_bear", "state"]].copy()
    state_out["timestamp"] = pd.to_datetime(state_out["timestamp"], utc=True)
    state_out.to_parquet(OUT_DIR / "hmm_states.parquet", index=False)
    print(f"Wrote hmm_states.parquet")

    # Also produce simple-gate states for comparison
    simple = simple_vol_gate_states(spx)
    simple_bear22 = float(
        (simple.loc[(simple["timestamp"] >= pd.Timestamp("2022-01-01", tz="UTC")) &
                    (simple["timestamp"] <  pd.Timestamp("2023-01-01", tz="UTC")),
                    "state_simple"] == 1).mean()
    )
    print(f"2022 simple-vol-gate (>p80) bear share: {simple_bear22:.2%}")

    # Load all 13 symbols
    print("Loading 13 symbols D1...")
    data = {s: get_candles(s, "D1").sort_values("timestamp").reset_index(drop=True)
            for s in ALL_SYMBOLS}

    # Run all 5 strategies (raw, pre-scaling)
    raw_returns = {}
    print("Strategy 1: HMM_BULL_TSMOM ...")
    raw_returns["HMM_BULL_TSMOM"] = strat_bull_tsmom(data, state_df)
    print("Strategy 2: HMM_BEAR_REV ...")
    raw_returns["HMM_BEAR_REV"] = strat_bear_reversion(data, state_df)
    print("Strategy 3: HMM_SLEEVE_MIX ...")
    raw_returns["HMM_SLEEVE_MIX"] = strat_sleeve_mix(data, state_df)
    print("Strategy 4: HMM_REGIME_FLIP ...")
    raw_returns["HMM_REGIME_FLIP"] = strat_regime_flip(spx, state_df)
    print("Strategy 5: HMM_VOLTGT ...")
    raw_returns["HMM_VOLTGT"] = strat_voltgt(spx, state_df)

    # Vol-scale each to 5% IS ann vol
    scaled = {}
    scalers = {}
    for name, r in raw_returns.items():
        r2, k = vol_scale_is(r)
        scaled[name] = r2
        scalers[name] = k

    # Per-strategy table (FULL / IS / OOS / 2022)
    rows = []
    for name, r in scaled.items():
        st = split_stats(r, bpy=252.0)
        rows.append({
            "strategy": name,
            "scaler": scalers[name],
            "FULL_sharpe": st["FULL"]["sharpe"],
            "IS_sharpe": st["IS"]["sharpe"],
            "OOS_sharpe": st["OOS"]["sharpe"],
            "Y2022_sharpe": st["Y2022"]["sharpe"],
            "FULL_ret": st["FULL"]["ann_return"],
            "OOS_ret": st["OOS"]["ann_return"],
            "FULL_dd": st["FULL"]["max_dd"],
            "OOS_dd": st["OOS"]["max_dd"],
            "FULL_vol": st["FULL"]["ann_vol"],
            "OOS_vol": st["OOS"]["ann_vol"],
            "n_FULL": st["FULL"]["n"],
            "n_OOS": st["OOS"]["n"],
        })
    table = pd.DataFrame(rows)

    # Filter: IS Sharpe >= 0.4 AND OOS Sharpe >= 0
    table["survives"] = (table["IS_sharpe"] >= 0.4) & (table["OOS_sharpe"] >= 0)

    print("\n=== HMM Strategy Table (vol-scaled to 5% IS) ===")
    print(table.to_string(index=False,
                          formatters={c: "{:+.2f}".format
                                      for c in ["FULL_sharpe", "IS_sharpe",
                                                "OOS_sharpe", "Y2022_sharpe"]}))

    # Save returns parquet
    out_df = pd.DataFrame(scaled)
    # Standardize index name
    out_df.index = pd.DatetimeIndex(out_df.index, name="timestamp")
    out_df.to_parquet(OUT_DIR / "hmm_returns.parquet")
    print(f"\nWrote hmm_returns.parquet ({out_df.shape})")

    # Combined survivors equal-weight
    survs = table.loc[table["survives"], "strategy"].tolist()
    if survs:
        combo = out_df[survs].mean(axis=1)
        st_combo = split_stats(combo, bpy=252.0)
        print(f"\nEqual-weight survivors: {survs}")
        print(f"  FULL Sh={st_combo['FULL']['sharpe']:+.2f}  "
              f"IS Sh={st_combo['IS']['sharpe']:+.2f}  "
              f"OOS Sh={st_combo['OOS']['sharpe']:+.2f}  "
              f"2022 Sh={st_combo['Y2022']['sharpe']:+.2f}")
    else:
        print("\nNo survivors at IS>=0.4 AND OOS>=0 filter.")

    # Save the summary table CSV
    table.to_csv(OUT_DIR / "hmm_summary.csv", index=False)

    # Comparison vs simple regime gate (SPX vol > IS p80)
    # Use HMM_BEAR_REV as the apples-to-apples comparator: rebuild the same
    # bear-only reversion strategy but with the simple-gate state.
    simple_state = simple[["timestamp", "state_simple"]].rename(columns={
        "state_simple": "state"
    })
    simple_state["p_bull"] = (simple_state["state"] == 0).astype(float)
    simple_state["p_bear"] = 1.0 - simple_state["p_bull"]

    # Bear-only D1 reversion with simple gate
    print("\nComparing HMM bear-rev vs simple-vol-gate bear-rev ...")
    rev_simple = strat_bear_reversion(data, simple_state)
    rev_simple, _ = vol_scale_is(rev_simple)
    rev_hmm = scaled["HMM_BEAR_REV"]
    s_hmm = split_stats(rev_hmm)
    s_simple = split_stats(rev_simple)
    print(f"HMM    bear-rev:  IS Sh={s_hmm['IS']['sharpe']:+.2f}  "
          f"OOS Sh={s_hmm['OOS']['sharpe']:+.2f}  "
          f"2022 Sh={s_hmm['Y2022']['sharpe']:+.2f}")
    print(f"Simple bear-rev:  IS Sh={s_simple['IS']['sharpe']:+.2f}  "
          f"OOS Sh={s_simple['OOS']['sharpe']:+.2f}  "
          f"2022 Sh={s_simple['Y2022']['sharpe']:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
