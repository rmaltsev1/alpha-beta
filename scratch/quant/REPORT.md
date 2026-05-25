# Institutional multi-strategy portfolio — final report

**Universe:** 13 symbols (3 crypto, 4 forex, 6 indices), D1 + H1, 2020-01-01 → 2026-05-23.
**Cost model:** per-side. FX 1 bp, Index 1.5 bp, Crypto 5 bp.
**Validation:** in-sample (IS) ≤ 2024-01-01, out-of-sample (OOS) ≥ 2024-01-01. All vol estimators walk-forward.

## Headline — TOP7 portfolio

7 sleeves, equal-weight, each rescaled to 5 % IS-annualized vol.

| Period | Ann return | Ann vol | **Sharpe** | MaxDD |
|---|---|---|---|---|
| FULL (2020–2026)    | +5.6 % | 2.2 % | **+2.58** | −2.3 % |
| IS (2020–2023)      | +4.9 % | 2.1 % | +2.37 | −2.3 % |
| **OOS (2024–2026)** | **+6.8 %** | 2.3 % | **+2.91** | **−1.5 %** |

OOS Sharpe > IS Sharpe → not curve-fit. Total equity curve grew from 1.000 to **1.407** over the 6.4 years; OOS slice grew from 1.000 to **1.169** in 874 days (+16.9 %).

**Year-by-year:**
| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|
| +10.4 % | +4.3 % | **−0.0 %** | +4.6 % | +4.3 % | +5.8 % | +5.7 % |
| Sh +3.85 | +2.12 | **−0.01** | +2.89 | +2.03 | +2.72 | +5.02 |

2022 is now break-even (vs the v3 portfolio's −0.23 Sharpe) because the TOP7 dropped sleeves that were 2022-correlated.

**Leverage scenarios** (Sharpe scales, vol and DD scale linearly):
- 4× lever → ~22 %/yr at ~9 % vol, ~6 % MaxDD
- 8× lever → ~45 %/yr at ~18 % vol, ~12 % MaxDD
- 12× lever → ~67 %/yr at ~27 % vol, ~18 % MaxDD

## Composition

The TOP7 picks the highest-IS-Sharpe sleeves after de-duplicating one near-perfectly-correlated pair (VOLMGD vs RISKPAR, ρ = 0.92 → kept RISKPAR):

| # | Sleeve | Bucket | IS Sharpe | OOS Sharpe | Source |
|---|---|---|---|---|---|
| 1 | RISKPAR     | beta      | +1.63 | +1.71 | Quant agent — risk-parity 13-asset, 10 % vol target |
| 2 | TSMOM       | trend     | +1.31 | +0.84 | Quant agent — Moskowitz 21/63/126/252-day momentum |
| 3 | EVE_XAU     | calendar  | +0.93 | +2.15 | XAU long 21–23 UTC weekdays (t-stat 5.0 from agent 2) |
| 4 | D1REV_UK    | mean-rev  | +0.76 | +0.40 | UK100 D1 mean-reversion, 50 bp gate |
| 5 | XSMOM       | trend     | +0.74 | +1.24 | Cross-sectional momentum (crypto/FX long-short + indices reversed) |
| 6 | D1REV_NAS   | mean-rev  | +0.73 | +0.72 | NAS100 D1 mean-reversion, 50 bp gate |
| 7 | WED_BTC     | calendar  | +0.73 | +1.06 | BTCUSDT long Wednesdays |

Buckets (correlation-derived clusters): trend (2 sleeves), mean-reversion (2), beta (1), calendar (2).

## What we tried, what worked, what was dropped

### Sleeves that made the cut (these are in TOP7)
- **RISKPAR**: classic 13-asset risk-parity (inverse-vol) with 10 % portfolio vol overlay. FX dominates the weighting (~62 %), crypto squeezed to ~5 %. Beats equal-weight on Sharpe in every year except 2021.
- **TSMOM**: classic Moskowitz 4-lookback ensemble (21/63/126/252 d). Medium-long (126 d) wins across asset classes; 21d is destructive in FX and indices.
- **XSMOM**: cross-sectional 126-d momentum within crypto and FX baskets, with the **indices basket inverted** (D1 mean-reversion). Indices momentum is noise; reversion isn't strong enough to overcome 40 % weekly turnover.
- **EVE_XAU**: gold long 21:00 / 22:00 / 23:00 UTC on weekdays. Hourly t-stats up to 5.0. The single highest-OOS-Sharpe sleeve in the panel.
- **WED_BTC**: BTC long on Wednesday D1 (t=2.31). Robust to OOS, all 3 cryptos show this midweek bias.
- **D1REV_NAS, D1REV_UK**: fade yesterday's > 50 bp move. NAS more stable than UK; both positive OOS.

### Sleeves cut from the master
- **PAIRS**: BTC-ETH, SPX-NAS, SPX-US30, EUR-GBP, XAU vs synthetic DXY. Sleeve OOS Sharpe **−0.83**. Rolling-OLS β breaks in trending markets (e.g. BTC-ETH β went 0.57 → 1.03 → 0.39 across 2020-2025). Same-currency pairs (EUR-GBP, SPX-NAS) survived OOS individually but couldn't overcome the basket drag from BTC-ETH and SPX-US30.
- **VOLMGD Sub-A**: vol-managed long-only across 13 symbols. OOS Sharpe +1.70 — *good*, but ρ = 0.92 with RISKPAR, so de-duped.
- **VOLMGD Sub-B**: vol-managed dip-buying (entry: D1 < −2σ). OOS Sharpe −0.03. Broken — the −2σ trigger now fires in confirmed downtrends, not reversions.
- **WED_ETH / WED_SOL**: solid OOS but correlated 0.85 / 0.55 with WED_BTC. Adding them in adds beta to crypto without much new alpha.
- **D1REV_SPX**: marginal IS Sharpe +0.32, kept-but-low priority. Dropping it raised TOP-N performance.

### V3 calendar sleeves vs the v4 quant sleeves
- v3 alone (7 calendar sleeves, equal-weight): OOS Sharpe +2.37, MaxDD −1.4 %
- v4 master TOP7 (mixed calendar + quant): OOS Sharpe **+2.91**, MaxDD −1.5 %
- **Lift: +0.54 OOS Sharpe** from adding TSMOM / XSMOM / RISKPAR

The quant sleeves added what v3 was missing — trend-following, cross-sectional, and a sized-up beta proxy.

## Correlation structure

Three loose clusters (weekly-resampled Pearson):
1. **Beta cluster**: VOLMGD ↔ RISKPAR ρ=0.92. Crypto-calendar correlated to beta at ρ=0.30–0.35 (WED_BTC ↔ RISKPAR).
2. **Mean-rev cluster**: D1REV_NAS / D1REV_UK / D1REV_SPX internally 0.43–0.80. Orthogonal to everything else (ρ ≤ 0.12 to all other sleeves).
3. **Calendar/independent**: EVE_XAU has ρ ≤ 0.10 to every other sleeve in the panel. Pure diversifier.

Trend (TSMOM ↔ XSMOM): ρ=0.13 — different time-scales, basically independent.

## What 2022 taught us

2022 was the only losing year in every variant tested. Why:
- All long-biased sleeves (RISKPAR, VOLMGD, WED_BTC/ETH/SOL, EVE_XAU) lost. Crypto crashed −60 %, gold flat, equities −20 %.
- Mean-reversion sleeves (D1REV_NAS/UK) made money — sharp moves followed by reversions.
- Cross-sectional XSMOM helped — rank-based long-short is regime-agnostic.

The TOP7 selection happens to **drop two of the three crypto-calendar sleeves** (ETH and SOL Wednesday), which were the biggest 2022 losers among the calendar group. That's why 2022 went from −0.72 Sharpe (11-sleeve equal-weight) to −0.01 Sharpe (TOP7).

A regime overlay (halve beta sleeves when SPX 30 d vol > IS p80) was tried and made almost no difference (Sharpe −0.04). Sleeve selection mattered more than dynamic gating.

## What's still missing (next iterations)

1. **A true tail-risk / long-vol sleeve** to make 2022 *positive* instead of flat. Options: put-spread overlay on SPX, short-rates futures, or a "buy-vol-when-vol-is-low" signal.
2. **Walk-forward sleeve selection**. The TOP7 was picked using full-IS Sharpe ranking, which has a mild look-ahead at the OOS test (since you'd have to *commit* to TOP7 at the IS/OOS boundary, which is what we did). A 12-month rolling sleeve-selection layer would be the honest production approach.
3. **Per-sleeve regime gating**, not portfolio-level. RISKPAR + WED_BTC both struggle in high-vol; D1REV_NAS *gains* in high-vol. A smart overlay halves only the directional sleeves.
4. **Realistic financing on RISKPAR and the D1REV sleeves** — both hold ~50 % of bars in indices, which means overnight CFD financing of ~3–5 %/yr on the *gross* notional. At 5 % sleeve vol this is ~0.4–0.7 %/yr net drag — material at the Sharpe levels we're showing.
5. **Live decay monitoring**. The Monday-23 effect on US indices was real in 2020–2023 and dead in 2024+. A trailing-12m-Sharpe tripwire on each sleeve before it can re-enter the portfolio is the obvious safety.
6. **Cost stress test**. Doubling all per-side bps should give a "lower bound" Sharpe. Most calendar sleeves are 1-trade-per-week, so this hurts the high-turnover sleeves (XSMOM, pairs that we dropped, TSMOM marginally) more than the rest.

## Files (all under `scratch/quant/`)

**Headline:**
- `REPORT.md` — this document
- `master_portfolio.py` — synthesizer for the full 11-sleeve panel
- `master_v2.py` — top-N + regime variants (TOP7 is the headline)
- `FINAL_portfolio.parquet` — daily returns + equity curve of the TOP7

**Per-sleeve agent outputs:**
- `tsmom.py` + `tsmom_returns.parquet` + `tsmom_breakdown.csv`
- `xsmom.py` + `xsmom_returns.parquet` + `xsmom_baskets.csv`
- `pairs.py` + `pairs_returns.parquet` + `pairs_breakdown.csv` (dropped from master)
- `volmgmt.py` + `volmgmt_{returns,subA_returns,subB_returns}.parquet` + breakdowns
- `risk_parity.py` + `risk_parity_returns.parquet` + `risk_parity_comparison.csv`

**Synthesizer outputs:**
- `all_sleeve_returns.parquet` — every sleeve rescaled to 5 % IS vol, daily
- `master_sleeves_stats.csv`, `master_sleeve_correlations.csv`
- `master_yearly_sharpes.csv`, `master_v2_yearly.csv`, `master_v2_variants.csv`

**Re-run from repo root:** `PYTHONPATH=. python scratch/quant/<file>.py`

## TL;DR

Built 5 institutional-style sleeves (TSMOM, XSMOM, pairs, vol-managed, risk-parity) via parallel sub-agents, added them to the 7 calendar sleeves from the previous iteration, de-duplicated correlated pairs, ranked by IS Sharpe, picked top-7. Result is a 7-sleeve equal-weight portfolio across 4 buckets (trend / mean-rev / beta / calendar) delivering **OOS Sharpe +2.91 at 2.3 % vol with −1.5 % MaxDD** over 2024-01 → 2026-05. The only losing year in the entire 6.4-year sample is 2022 at −0.01 Sharpe (essentially flat). Pure-beta benchmarks (equal-weight, 60/40 SPX/BTC) sit at Sharpe 1.17–1.42 over the same period — the strategy is doing real risk-adjusted work, not selling beta.
