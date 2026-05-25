# Strategy

What the alpha-beta portfolio is actually doing, and why it works. This is the strategy / methodology / research-findings document. For code structure see [ARCHITECTURE.md](ARCHITECTURE.md); for performance numbers see [`scratch/quant/FINAL_REPORT_v15.md`](scratch/quant/FINAL_REPORT_v15.md).

## Thesis in one paragraph

**No single edge in this universe is large enough to live on.** What does work is assembling 23 small, real, near-uncorrelated edges, vol-scaling each to a common 5 % annualized vol, gating them by market regime, stop-lossing the volatile ones, and levering the ensemble to the target vol budget. The portfolio's monthly return is dominated by the drift of a Sharpe-4 process rather than by the noise of any individual signal.

## The 23 sleeves

Listed roughly in order of OOS Sharpe contribution.

### 1. HMM_BULL_TSMOM (OOS Sharpe +2.41)

A 2-state Gaussian HMM is fit walk-forward on SPX D1 log returns (252-bar trailing window, EM-fit, refit every 6 months). State 0 = "bull" (higher mean, lower vol); state 1 = "bear" (lower mean, higher vol). When the online filtered `P(bull)` > 0.6, run a 21-day momentum strategy across all 13 symbols. When `P(bull)` < 0.4, sit flat.

**Why it works:** the HMM classifies 2022 as 91.9 % bear (vs the simple "SPX vol > 80th percentile" gate's 43.4 %). Cleaner regime signal → trend strategies stop fighting the chop.

### 2. EVE_XAU (OOS +2.15)

Long XAU/USD during the 21:00, 22:00, and 23:00 UTC hours, weekdays. Discovered by a per-hour t-statistic scan: gold consistently drifts +1.75 / +2.32 / +2.47 bps respectively in those three hours, with t-stats up to **+5.0** (the largest single-bar effect in the dataset).

**Why it works:** physical-gold market hand-off — Asian buyers pre-positioning before Wellington open at 22:00 UTC plus systematic CB demand. The effect is consistent across Mon–Fri.

### 3. VOLFORECAST (OOS +1.95)

For each of the 13 symbols, position size = `+1 × (target_vol / EWMA-forecast vol)`, clipped to [0, 3]. When forecast vol spikes, cut leverage; when calm, lever up. Moreira-Muir managed-volatility strategy.

**Why it works:** all 13 symbols have AR(1) > 0.985 on 30-day realized vol — last month's vol is the single best predictor of next-week vol. **All 13 symbols are individually positive OOS.**

### 4. RISKPAR (OOS +1.71, with stops +1.98)

Classic risk-parity: weights inversely proportional to each symbol's 90-day realized vol; portfolio-level scaled to 10 % annualized vol. With ATR stops applied: gains +0.27 Sharpe.

Average weights: FX 62 % (low vol), Indices 33 %, Crypto 5 % (high vol → squeezed).

### 5. STATARB_XS (OOS +1.62)

Cross-sectional Bollinger-z mean-reversion. Each day, rank the 13 symbols by `(close − 20d MA) / 20d std`. Long the 3 most-negative-z, short the 3 most-positive. Equal-weight basket, rebalance daily.

**Why it works:** correlation 0.05 with the existing D1 mean-reversion sleeves — genuinely orthogonal time horizon (20 days vs 1 day).

### 6. CORR_REGIME (OOS +1.24, 2022 +1.66)

8 sub-strategies built around the average pairwise correlation across all 13 symbols (60d rolling). When avg corr is in IS bottom-quartile → risk-on (long crypto basket). When in top-quartile → risk-off (defensive). The cleanest sub-strategy: long BTC / long ETH when 60d avg corr < IS 25th percentile AND BTC trailing 30d return > 0.

### 7. W1_STRATS (OOS +1.48)

Weekly-bar strategies — the only timeframe nobody had tested in earlier waves. 30 of 91 per-strategy survivors. Donchian breakout (long when close > 12-week high) is the standout. Slow signals, low turnover, cost-friendly.

**Why it works:** correlation 0.02–0.13 with all D1 sleeves. Captures macro-scale drift that daily signals can't see.

### 8. TREND_NEW (OOS +1.08, with stops +1.49)

7-family trend ensemble: trailing-stop on indices, mom-reversal at extremes, vol-regime-gated TSMOM, cross-asset confirmation, triple-screen, MACD, Donchian. After IS Sharpe ≥ 0.4 filtering: 26 of 97 survivors. Replaces the original vanilla TSMOM.

### 9. MICROSTR_D1 (OOS +1.36, 2022 +1.58)

5 microstructure patterns at D1: gap-fill on US indices (close-to-open gaps > 0.5 % tend to fade), range-expansion fade, pivot-equilibrium reversion (close > 1.01× pivot → fade), gap-and-go (small positive lift), Doji-reversal. Indices-heavy because OHLC patterns are cleanest there.

### 10. XSMOM (OOS +1.24)

Cross-sectional momentum within each basket: rank by trailing 63-day return skipping last 5 days. Long the basket's top, short the bottom. Critically: **the indices basket is inverted** (long the loser, short the winner) — D1 mean-reversion is real on indices, classic momentum is noise.

### 11. WED_BTC (OOS +1.06)

BTC long on Wednesday D1 bars. Discovered as +43.7 bps mean return with t = 2.31 on Wednesdays in the per-day-of-week scan. Same effect smaller on ETH and SOL.

**Why it works:** unclear. Possibly Asia mid-week funding rolls + US mid-week risk decisions. Robust across BTC/ETH/SOL.

### 12. VOL_BREAKOUT (OOS +1.10, 2022 +1.21)

6 vol-breakout strategies: vol-regime-switched momentum, compression-then-expansion, Bollinger squeeze, NR7 breakout, inside-day breakout. SOL is the cleanest substrate (passes 2 independent strategy families). Indices add the 2022 alpha.

### 13. EURGBP_MR (OOS +1.04)

Mean-reversion on the synthetic EUR/GBP cross (constructed as EUR_USD / GBP_USD). Only single-pair FX strategy that survived filtering — same-currency-bloc spread mean-reverts cleanly even when each leg trends.

### 14. TERM_SPREADS (OOS +0.97, 2022 +1.12)

4-strategy mini-sleeve: vol-of-vol regime switching (SPX/NAS/ETH) + crypto-vs-equity correlation-gated long. Captures dispersion regime trades.

### 15. EVENT_VOLSPIKE (OOS +0.66, with stops +0.82)

When any symbol's D1 |return| > 3σ on day t, the next-day expected return is consistent and sign-predictable. FX and metals tend to fade; equity indices continue. Per-symbol sign learned on IS only.

### 16. PAIRS_EXP (OOS +0.62, with stops +1.01, 2022 +2.05)

11 cointegrated cross-asset pairs (out of 78 candidate combinations of the 13 symbols). Walk-forward OLS β + ADF stationarity test. Only trade z-score extremes when ADF passes. Survivors include unusual pairs like GBP_USD vs US30_USD (Sharpe +1.06) and EUR_USD vs DE30_EUR.

**Why it works:** these aren't classic "two highly correlated names" pairs — they're cross-asset combinations where the spread is statistically stationary even though the underlyings trend.

### 17. D1REV_NAS, D1REV_UK (OOS +0.72, +0.40)

Fade yesterday's D1 close on indices, with a 50 bp gate (don't trade noise). NAS and UK survived; SPX is marginal.

### 18. CRYPTO_vs_SPX (OOS +0.17, with stops +0.30, 2022 +0.75)

Single market-neutral pair: crypto basket (mcap-weighted BTC + ETH + SOL) vs SPX. Long the cheap side when ADF passes. Modest standalone Sharpe but uncorrelated with everything else.

### 19. DEFEND (OOS +0.59, with stops +0.82)

The defensive sleeve. Long USD_JPY + XAU when SPX is in equity stress (20d return < −5 % or 30d vol > IS p80). Stress-gated safe-haven trade. Solves 2022 — without it, that year goes from +2.5 % to +1.5 % at portfolio level.

### 20. SESSION_MOM (OOS +0.88)

JP225 Asia-open momentum continuation. The 00:00 UTC Tokyo cash-open bar carries momentum 4–6 H1 bars into the European session. The only "session momentum" strategy that survived (Asia-overreaction fade, NY-open reversal, gap-continuation all failed).

### 21. MULTIDAY (OOS +0.64, 2022 +2.80)

2-strategy mini-sleeve: 5-bar momentum acceleration + 2-bar absorption pattern. Captures swing-trade patterns at D1.

### 22. WED_ETH, WED_SOL (not in TOP15 but in extended panel)

Crypto Wednesday-long effect for ETH and SOL. Already covered by WED_BTC dynamic; smaller weights.

### 23. HMM_SLEEVE_MIX, HMM_BEAR_REV (extended panel)

The other two HMM-state-conditional sleeves. Smaller weights — HMM_BULL_TSMOM is the dominant contributor.

## The 5 production overlays

The sleeves are the alpha. The overlays are the risk management.

### 1. Per-sleeve regime gates

When SPX 30-day realized vol > IS 80th percentile (~24 % annualized), apply per-sleeve multipliers:

- **Halved (0.5×):** TREND_NEW, RISKPAR, VOLFORECAST, EVE_XAU, WED_BTC, HMM_BULL_TSMOM — directional / beta sleeves get whipsawed in high-vol
- **Amplified (1.5×):** D1REV_NAS, D1REV_UK, PAIRS_EXP, DEFEND, MICROSTR_D1, STATARB_XS, CORR_REGIME, EVENT_VOLSPIKE, MULTIDAY, TERM_SPREADS, EURGBP_MR, CRYPTO_vs_SPX — mean-reversion + crisis sleeves make most of their money in high-vol regimes

This is the single most important overlay. Without it, 2022 is −0.23 Sharpe. With it, +0.52. With *all* overlays + HMM, +4.21.

### 2. Fast decay tripwire

Each calendar month-end, compute trailing-63-day Sharpe per sleeve. If a sleeve's trailing-63d Sharpe is < 0 for two consecutive monthly checks, set its weight to zero. Re-enter when trailing-63d Sharpe > 0.5.

Catches dead sleeves in ~75 days. The previous trailing-12-month version took ~179 days — too slow for production.

### 3. Time-of-month vol-target overlay

Base target: 18 % annualized portfolio vol. Modulated by day-of-month:
- Days 1–10: 20 % (turn-of-month captures the structural drift)
- Days 11–20: 16 %
- Days 21+: 18 %

Achieved via scaling factor `target_vol / trailing_60d_realized_vol`, capped at 15× leverage.

A Pareto improvement over static 18 %: +0.09 OOS Sharpe, similar realized vol, slightly smaller MaxDD.

### 4. Drawdown control

When the portfolio's 30-day rolling drawdown exceeds 3 %, halve total gross exposure. When DD recovers to < 1 %, restore full gross. State-machine, not a stop.

Active ~8 % of bars in production.

### 5. Per-sleeve stop-losses

ATR-based or time-based stops applied to 6 specific sleeves where they add ≥ +0.10 OOS Sharpe:

| Sleeve | Stop rule | OOS Sharpe lift |
|---|---|---|
| TREND_NEW | TP 3× ATR / SL 1.5× ATR | +0.34 |
| PAIRS_EXP | 2× ATR stop | +0.32 |
| RISKPAR | TP 3× ATR / SL 1.5× ATR | +0.23 |
| DEFEND | 3-bar time stop | +0.23 |
| EVENT_VOLSPIKE | 3-bar time stop | +0.16 |
| CRYPTO_vs_SPX | 1.5× ATR stop | +0.13 |

The other 17 sleeves' trade durations are too short or P&L too granular for stops to fire usefully.

## Why this works — the meta-learnings

After 50+ hypotheses tested across 11 research waves, the patterns are clear.

### What's robust
1. **Diversification dominates sizing.** Equal-weight beats every tested sizing scheme (Kelly, Bayesian shrinkage, ML, Sharpe-tilt, bucket-RP) once sleeves are vol-scaled.
2. **Cointegration gates rescue pairs.** Naive z-score on rolling-OLS β fails OOS because β drifts. ADF stationarity test on the residual filters out non-stationary windows.
3. **Per-sleeve regime gates beat portfolio-level gates.** D1REV *makes money* in high-vol while RISKPAR loses — they need different multipliers.
4. **Walk-forward sleeve selection costs ~0.5 OOS Sharpe.** Acceptable; static-hindsight gives +4.13, honest walk-forward gives ~+3.4.
5. **D1 dominates intraday at retail costs.** Every M15/H1 strategy tested died on costs. ETH M15 momentum had gross OOS Sharpe +1.41 but net **−1.40** after 5 bp/side × high turnover.
6. **HMM identifies regimes 2× better than threshold gates.** 91.9 % of 2022 classified as bear vs 43.4 % for SPX-vol-percentile gate.
7. **The Weekly timeframe is the unused gem.** Correlation 0.02–0.13 with everything else — genuine cross-timescale diversification.

### What's fragile
1. **Crypto calendar effects are non-stationary.** Both hour-of-week and day-of-month "discoveries" flipped OOS post-2024. Same regime-shift signature as the Monday-23 indices effect (real 2020–23, dead 2024+).
2. **Capacity is real.** UK100 (FTSE) on OANDA limits 15 of 23 sleeves. The strategy is solid at $1–20M; degrades at $100M; collapses to a 4-sleeve macro overlay at $1B unless routing to LIFFE futures.
3. **The HMM signal degrades in calm bull markets.** It correctly nails 2022 but in 2024–25 (sustained bull) the regime is "always bull" and the overlay does nothing useful.

## Rejected hypotheses (the negative findings)

Every one of these was tested, instrumented, and rejected with hard data. Documenting failures is as important as documenting wins — these are paths future research should not re-attempt without new information.

| Hypothesis | Why it failed |
|---|---|
| ML linear meta-signal | Indistinguishable from equal-weight |
| ML non-linear (numpy gradient boost on 18 features) | Same — residual signal below noise floor |
| Half-Kelly position sizing | Tied with EW on Sharpe, doubled MaxDD |
| Conservative Kelly v2 (6 variants) | No variant cleared the +0.10 Sharpe / ≤10 % DD constraint |
| Bayesian shrinkage weighting | Optimal shrinkage parameter → infinity = equal-weight |
| Bucket-level weighting (6 schemes) | Best variant +0.06 OOS Sharpe (noise) |
| Session reversion (Asia/London/NY opens) | All 16 variants failed — sessions are MOMENTUM events not reversion |
| Session momentum (the flip) | Only JP225 Asia survived (1 of 17) — not a general effect |
| Relative-strength rotation | OOS +0.40, 2022 −1.54 — overlaps with XSMOM and worse |
| Tail vol-of-vol short SPX | Made +1.50 in 2022 but bled −1.19 OOS — net cost of insurance too high without options |
| Trend-confirmed vol short | Same problem |
| Carry / structural drift | 0.83 correlation with RISKPAR — duplicate beta |
| VWAP strategies (55 variants) | 1 survivor, sleeve OOS +0.06 — VWAP doesn't carry signal on this universe |
| Multi-symbol confirmation | 0.60 correlation with TREND_NEW — duplicate trend |
| FX volume fade | All 24 variants negative — OANDA "volume" (tick count) carries no signal in either direction |
| Crypto hour-of-week | OOS −4.55 — pure data-mining |
| Crypto turn-of-month (a priori hypothesis) | IS +2.13 → OOS −1.32 — calendar effects flipped post-2024 |
| Intraday strategies (24 variants on M15/H1) | All fail cost gate at retail spreads |
| Skewness premium | No lottery-like names in this universe |
| Adaptive D1 reversion | Doesn't beat vanilla |
| DD recovery sleeves | 2022 −0.63 — persistent down-trends hurt mechanical "buy the dip" |
| Anomaly detection (7 variants) | 0 survivors |
| H1 lead-lag (BTC-ETH, NAS-SPX, EUR-GBP, SPX-EU, etc.) | Gross signal < 0.1 Sharpe — costs kill |
| Classical TA (Stochastic+ADX, Aroon, CCI, Williams %R) | Folklore — 0 survivors across 13 symbols |
| Multi-leg spreads (3-4 leg combinations) | Only YEN_TRI survived; 0.69 corr with PAIRS_EXP — duplicates |
| HMM-conditional sleeve weighting | All variants underperformed v15 — HMM signal already absorbed by HMM_BULL_TSMOM sleeve |
| Sleeve-of-sleeves momentum | +0.018 Sharpe (within noise) — portfolio at Sharpe 4 is too smooth to extract meta-signal |
| Per-asset asymmetric vol targeting | Ties or worsens — sleeve-equal already equalizes per-symbol vol |

## When to expect the strategy to break

Three categories of failure modes worth monitoring in live trading.

1. **Sleeve decay.** When a sleeve's trailing-12m Sharpe goes negative, the decay tripwire catches it within ~75 days. The Monday-23 indices effect and the crypto calendar effects both went through this lifecycle during the build. Expect 1–2 sleeves per year to decay below the threshold.

2. **Cost regime shifts.** The model assumes 1 bp FX, 1.5 bp index, 5 bp crypto. If spreads widen (typical in stressed markets), check the cost stress test results: at 2× costs OOS Sharpe is still +2.21 (good). At 3× it's +1.51 (still profitable). Above that the high-turnover sleeves (XSMOM, TREND_NEW, VOL_BREAKOUT) become drags.

3. **Cross-sleeve correlation explosion.** In extreme risk-off events, all directional sleeves can correlate. The CORR_REGIME sleeve is designed to detect this; the regime gates are designed to neutralize directional sleeves automatically. But a multi-week correlated drawdown across mean-reversion and trend simultaneously is theoretically possible and not present in the 2020–2026 backtest.

## What's still on the table

Three concrete avenues for further alpha that this codebase can't yet exploit:

1. **Options data → real tail-protection sleeve.** The naive vol-shorts all failed because they don't actually buy vol, they sell it. Real put-spread protection in SPX options would convert 2022 from +5 %/mo to +6–7 %/mo and improve the 5th-percentile bootstrap Sharpe.

2. **Crypto perpetuals + funding rates.** Currently using spot. Perp basis + funding-rate momentum are documented edges (Glassnode, CryptoQuant research) that this dataset can't see. Adding Binance perp data + funding rate history would likely add 1–2 % monthly.

3. **Sub-bp execution → intraday becomes viable.** ETH M15 momentum had gross OOS Sharpe +1.41 but lost everything to 5 bp/side crypto cost. Direct-to-exchange routing with sub-bp execution would unlock the entire intraday universe (~40+ untested signals).

The strategy as it stands is converged for the data and cost structure available. To push past +5 %/month sustainably, you need new data, not new analysis.
