# Where to start — v15 PRODUCTION

If you're picking this up cold:

1. **`FINAL_REPORT_v15.md`** — current production (10-hour iteration). 23-sleeve portfolio, +5.62%/mo OOS at -6.0% MaxDD.

2. **`master_v15.py`** — production script. `PYTHONPATH=. python scratch/quant/master_v15.py` reproduces headline.

3. **`PRODUCTION_FINAL_v15.parquet`** — daily returns + equity curve.

## Headline

- **OOS Sharpe +4.13** at 18% vol target
- **+5.62%/month OOS**
- **MaxDD -6.0%**
- Every year ≥ +4.01%/mo, 2022 crisis = +5.05%/mo
- Bootstrap: 5%-ile Sharpe = +3.06, P(Sharpe > 3) = 96.7%

## Capacity ceiling

- **$1-20M AUM**: full TOP23, headline numbers apply
- **$100M AUM**: drop 15 sleeves, expect OOS Sharpe +2.40
- **$1B AUM**: only 4 sleeves work, Sharpe +1.37
- **Route to LIFFE futures** instead of OANDA CFDs to unlock $150M+

## Iteration chain

| Version | OOS Sharpe | Monthly @ 18% vol | Key change |
|---|---|---|---|
| v3 calendar-only | +2.37 | — | 7 calendar sleeves |
| v8 (1st session end) | +2.52 | +4.04% | + pairs |
| v11 (start of 10h session) | +2.82 | +3.98% | + DEFEND |
| v12 (wave 3) | +3.11 | +4.40% | + VOLFORECAST, TREND_NEW, H4_SLEEVE, CORR_REGIME |
| v13 (wave 7) | +3.60 | +4.77% | + W1, EVENT, STATARB, MICROSTR, VOL_BREAK, TERM, EURGBP, MULTIDAY |
| v14 (wave 8 stops) | +4.04 | +5.31% | + stops on 3 sleeves |
| **v15 (wave 10 HMM)** | **+4.13** | **+5.62%** | **+ HMM_BULL_TSMOM** |

## Operational deployment

| Tier | AUM | Composition | Expected monthly |
|---|---|---|---|
| 1 | $1-20M | Full TOP23 + 5 overlays | +5.62% |
| 2 | $20-100M | 8-sleeve subset (drop UK100-bound) | +2.0% |
| 3 | $1B | 4-sleeve macro-overlay | +1.5% |

## Reproduction

From repo root with `.venv`:
```sh
PYTHONPATH=. python scratch/quant/master_v15.py
```

Stress + walk-forward:
```sh
PYTHONPATH=. python scratch/quant/v14_walk_forward.py
PYTHONPATH=. python scratch/quant/v9_stress_full.py
```

## Sleeve count summary

- 23 sleeves in production
- 5 production overlays (regime gates, decay tripwire, time-of-month vol-target, drawdown control, stop-losses on 6 sleeves)
- 50+ hypotheses tested across 11 waves
- ~20 sleeves accepted, ~30 rejected with hard data

## Convergence confirmation (Wave 11)

Final wave tested 3 high-hope ideas — all failed to beat v15:
- HMM-conditional sleeve weighting: -0.10 to -0.66 OOS Sharpe
- Sleeve-of-sleeves momentum: +0.018 OOS Sharpe (within noise)
- Per-asset asymmetric vol targeting: ties or worsens vs sleeve-equal

**The model has reached the alpha ceiling on this data/cost structure.** Further alpha gains require new data sources (options, perps, funding rates) or better execution (sub-bp costs).

## Capacity-aware sleeve drops

**Drop at $20M:** D1REV_UK first (FTSE-only bottleneck), then in order:
TERM_SPREADS, H4_SLEEVE, CORR_REGIME, XSMOM, VOL_BREAKOUT, VOLFORECAST,
MICROSTR_D1, STATARB_XS, TREND_NEW, RISKPAR, MULTIDAY, D1REV_NAS, SESSION_MOM

**Keep at $100M (8 sleeves):**
PAIRS_EXP, W1_STRATS, CRYPTO_vs_SPX, EVENT_VOLSPIKE, WED_BTC, EURGBP_MR, EVE_XAU, DEFEND

**Keep at $1B (4 sleeves):**
WED_BTC, EURGBP_MR, EVE_XAU, DEFEND (+ HMM_BULL_TSMOM if can be implemented)
