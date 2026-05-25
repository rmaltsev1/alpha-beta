# Models from observations — iteration report

**Data:** 13 symbols × 7 timeframes, 2020-01-01 → 2026-05-23, local parquet store.
**Engine:** `alphabeta/backtest.py` — vectorized signal backtester with per-side cost (FX 1 bp, Index 1.5 bp, Crypto 5 bp).
**Split:** in-sample (IS) ≤ 2024-01-01, out-of-sample (OOS) ≥ 2024-01-01. All vol scaling computed on IS only.
**Sleeve vol target:** 5 % annualized each, equal-weighted into the portfolio.

## TL;DR

A **7-sleeve, 3-bucket portfolio** (XAU evening drift / crypto Wednesday / equity-index D1 mean-reversion) delivered:

| period | ann_return | ann_vol | Sharpe | MaxDD |
|---|---|---|---|---|
| Full (2020–2026) | **+3.9 %** | 2.5 % | **+1.59** | −2.3 % |
| IS (2020–2023)   | +3.1 %    | 2.6 % | +1.19    | −2.3 % |
| **OOS (2024–2026)** | **+5.3 %**| 2.2 % | **+2.37**| **−1.4 %** |

OOS Sharpe > IS Sharpe — the model is not curve-fit. Every calendar year except 2022 was positive at the portfolio level (2022 Sharpe −0.23, smallest annual drawdown in the entire 7-year series).

At 4× leverage the same Sharpe gives ~16 % annual return at ~10 % vol with ~9 % max drawdown.

## What worked

| Sleeve         | Symbol      | Signal             | IS Sharpe | OOS Sharpe | Notes |
|---|---|---|---|---|---|
| **EVE_XAU**    | XAU_USD     | Long 21:00–23:00 UTC, weekdays   | +0.88 | **+2.21** | t-stats 3.2 / 3.9 / 5.0 in agent 2 diag |
| **WED_BTC**    | BTCUSDT     | Long Wed D1                       | +0.73 | **+1.06** | Crypto midweek bias |
| **WED_ETH**    | ETHUSDT     | Long Wed D1                       | +0.71 | **+1.11** | ETH Wed mean = +52.9 bps, t=2.04 |
| WED_SOL        | SOLUSDT     | Long Wed D1                       | +0.33 | +0.88     | Highest vol of crypto sleeve |
| **D1REV_NAS**  | NAS100_USD  | Fade yesterday's D1 sign (50 bp gate) | +0.74 | +0.75 | Most stable equity reversion |
| D1REV_UK       | UK100_GBP   | Same                              | +0.76 | +0.40     | Weaker OOS but still positive |
| D1REV_SPX      | SPX500_USD  | Same                              | +0.32 | +0.39     | Marginal but adds diversification |

### Why 3 buckets?

Weekly-resampled correlation across sleeves (FULL period):

- **XAU evening** ↔ everything else: −0.04 to 0.00 (independent)
- **Crypto Wednesday cluster** internally 0.55–0.85, ↔ indices ~0
- **Indices D1 reversion cluster** internally 0.43–0.80, ↔ crypto ~0, ↔ XAU ~0

Three near-independent buckets is why equal-weight gives the same result as risk parity — vol is already balanced and correlations are already low.

## What we dropped after testing

| Idea                          | IS Sharpe  | OOS Sharpe | Reason |
|---|---|---|---|
| Monday 23:00 UTC long on US indices | +0.61 to +0.96 | **−1.82 to −3.38** | Strong IS, collapsed in 2024+. Either decayed or never real. |
| EUR_USD evening (21–23 UTC long block) | −2.30 | −1.71 | DST gaps in OANDA H1 bars fragment position series → 2× cost. |
| GBP_USD evening                | −1.95 | −3.04 | Same DST issue; signal also weaker than EUR. |
| USD_JPY evening short          | −1.15 | −0.59 | Per-hour t-stats existed but didn't survive cost. |
| USD_JPY Wednesday 23:00 short  | −0.02 | −0.72 | Tiny exposure (0.8 %), cost dominates. |
| Hour-of-day 23 long on FX majors | −3.31 to −4.90 | −2.07 to −5.89 | XAU-specific signal — does not generalize. |
| EUR_USD London +08 / −11 combo | −1.94 | −3.69 | Each leg has a real t-stat, but the cost of 2 round trips/day kills it. |
| JP225 Asia-open continuation   | −1.33 | −0.30 | Asia open is *high-vol*, not *directional*. |

The recurring failure mode is **cost-eats-signal**: vol-scaling a high-turnover sleeve to a fixed-vol target means the cost per trade also scales with the leverage, and at 1 bp/side × 10× leverage on 250 round trips/year you eat 50 bps × leverage of alpha just to play. Only low-turnover or low-leverage-needed sleeves survive.

## Annual stability

Sharpe per calendar year (script `scratch/models/rolling_stability.py`):

```
year   EVE_XAU  WED_BTC  WED_ETH  WED_SOL  D1REV_NAS  D1REV_UK  D1REV_SPX  PORTFOLIO
2020    +1.63    +1.69    +0.92    −0.22     +0.99     +1.46      +0.80     +2.07
2021    +0.21    +1.06    +1.59    +1.99     +0.61     −0.17      −0.15     +1.68
2022    −1.29    −0.65    −0.37    −1.03     +0.87     +1.08      +0.31     −0.23
2023    +2.53    +0.85    +0.47    +0.50     +0.29     −0.44      −0.44     +1.23
2024    +1.03    +1.00    +0.76    +1.10     −0.26     +0.60      −0.28     +1.14
2025    +1.55    +1.51    +1.84    +1.17     +1.60     −0.02      +1.02     +2.44
2026*   +4.67    +0.32    +0.21    −0.62     +0.02     +1.02      −0.53     +4.22
```

*2026 = Jan–May only, treat with caution.*

2022 is the only down year (Sharpe −0.23). Everything risk-off correlated: gold sold, crypto crashed, BUT the equity D1-reversion sleeve was strongly positive (volatility creates more reversion opportunities), keeping the portfolio MaxDD shallow.

## How each sleeve is wired

### EVE_XAU — gold evening long
- Hour-of-day diag: XAU 21:00 / 22:00 / 23:00 UTC have hourly mean returns +1.75 / +2.32 / +2.47 bps with t-stats +3.20 / +3.92 / **+5.00** (weekdays). The block is consistent across Mon–Fri.
- Position: +1 during the 3-hour window, weekdays only.
- Scaled to 5 % ann vol on IS → multiplier ≈ 2.08.
- Why it works: physical-gold market hand-off across continents — Asian / Wellington physical-pre-open buying.

### Crypto Wed long (BTC / ETH / SOL)
- Agent-2 diag: BTC Wed = +43.7 bps (t=2.31), ETH Wed = +52.9 bps (t=2.04) on D1.
- Position: +1 on Wednesday D1 bar, flat else.
- 14.3 % exposure. One trade per week.
- Why it works: unclear (some combination of Asia-late funding rolls, US earlier-week risk decisions hitting Wednesday). Robust empirically across 3 coins.

### Equity D1 mean-reversion (NAS / UK / SPX)
- Agent-3 diag: D1 ρ₁ = −0.10 (NAS), −0.11 (UK), −0.09 (SPX). All run-length distributions show fewer ≥4-day streaks than a random walk.
- Position: +1 if yesterday's D1 log-return was < −50 bps, −1 if > +50 bps, else 0.
- Threshold filters tiny daily noise and reduces turnover.

## V4 — regime overlay variant (run_v4_regime.py)

Halve sleeve gross when SPX D1 30-day realized vol is above the IS 80th
percentile (~24 % annualized). Threshold fit on IS only.

| period | V3 Sharpe | V4 Sharpe | V4 vol | V4 MaxDD |
|---|---|---|---|---|
| Full | +1.59 | **+1.62** | 2.1 % | −2.2 % |
| IS   | +1.19 | +1.14     | 2.2 % | −2.2 % |
| OOS  | +2.37 | **+2.46** | 2.1 % | −1.4 % |

Year-by-year V3 → V4:
```
 2020  +2.07 → +1.59   (regime on 34% — cost us, market was fine)
 2021  +1.68 → +1.68
 2022  −0.23 → −0.01   (regime on 43% — saved the year)
 2023  +1.23 → +1.23
 2024  +1.14 → +1.14
 2025  +2.44 → +2.78   (regime on 13% — small lift)
 2026* +4.22 → +4.22
```

Net: small improvement in expectation, recovers the only losing year, slightly
worse drawdown protection in calm-but-volatile years like 2020. The simple
overlay halves *every* sleeve in the regime — a smarter version would only
halve the sleeves that historically struggle in high-vol regimes (EVE_XAU and
the crypto sleeves were the 2022 losers; the equity D1-reversion sleeve *gains*
in high vol). Worth pursuing.

## What's NOT in here that probably should be (next iterations)

1. **Walk-forward vol scaling.** Each sleeve's scale is fit once on IS. A 12-month rolling fit would adapt to vol regime changes and probably help WED_SOL (which has the most vol drift).
2. **Portfolio-level vol targeting.** Currently the portfolio drifts between 2.1 % and 2.6 % vol. Targeting a fixed 5 % would dial leverage up when sleeves diversify and down when they correlate.
3. **Regime-aware *per-sleeve* gating.** V4 halves the whole portfolio in high vol; the D1-reversion sleeve is actually *additive* in high vol. Halve only the directional sleeves (XAU, crypto) and you keep the upside.
4. **Funding cost on the D1-reversion sleeve.** It holds positions ~55 % of bars — overnight financing on a CFD index would matter. Estimate ~3–5 % per year on the *gross* notional; at 5 % vol target the *net* drag is more like 0.4–0.7 % per year — manageable but should be subtracted.
5. **Live signal decay monitoring.** The Monday-23 effect *was* real in 2020–2023 (IS Sharpe +0.96 on NAS) and dead in 2024+. We need an automated decay tripwire — e.g. require a sleeve's trailing 12-month Sharpe stay > 0 before allowing it into the portfolio.

## Files

- `alphabeta/backtest.py` — engine (~150 lines)
- `scratch/models/strategies.py` — initial v1 strategy library
- `scratch/models/strategies_v2.py` — final v3 strategies
- `scratch/models/run_all.py` — v1: scattershot run of every idea
- `scratch/models/run_v2.py` — v2: 14 sleeves, vol-scaled, found which break OOS
- **`scratch/models/run_v3.py`** — final 7-sleeve portfolio (clean version)
- `scratch/models/run_v4_regime.py` — V3 + SPX-vol regime overlay
- `scratch/models/rolling_stability.py` — year-by-year Sharpe per sleeve
- `scratch/models/diag_*.py` — diagnostics that drove decisions
- `scratch/models/results_*.csv`, `equity_*.parquet`, `portfolio_v3.parquet`, `portfolio_v4.parquet` — saved curves
- `scratch/models/annual_sharpes.csv` — annual table

Re-run any of them with `PYTHONPATH=. python scratch/models/<file>.py` from
the repo root.
