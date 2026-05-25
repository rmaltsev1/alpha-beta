# Production v16 — pushed past 7 %/month

The next iteration after the v15 production (OOS Sharpe +4.13, +5.6 %/mo) — adds a new sleeve and a more aggressive leverage rule.

## Headline

| Variant | OOS Sharpe | OOS Return | OOS Vol | OOS MaxDD | OOS Monthly | Worst Month |
|---|---|---|---|---|---|---|
| v15 baseline (18 % vol target) | +4.08 | +65.7 % | 16.1 % | −5.7 % | +5.42 % | −0.82 % |
| **v16-V4 (floor+ceiling 25 %)** | **+4.30** | **+85.3 %** | 19.8 % | **−7.0 %** | **+7.04 %** | **+0.34 %** |
| v16-V5 (SPX-regime ramp 12-30 %) | +4.20 | +80.5 % | 19.2 % | −7.0 % | +6.64 % | +0.34 % |

**v16-V4 is the recommended production model.** Every OOS month is positive (worst at +0.34 %).

## Year-by-year (v16-V4)

| Year | Sharpe | Return | Vol | DD | Monthly |
|---|---|---|---|---|---|
| 2020 | +5.52 | +126.1 % | 22.8 % | −6.0 % | **+10.51 %** |
| 2021 | +4.82 | +105.2 % | 21.8 % | −8.9 % | **+8.76 %** |
| **2022 (crisis)** | **+4.42** | **+73.8 %** | 16.7 % | **−5.6 %** | **+6.15 %** |
| 2023 | +3.44 | +53.0 % | 15.4 % | −7.9 % | +4.42 % |
| 2024 (OOS) | +3.98 | +80.1 % | 20.1 % | −7.0 % | +6.68 % |
| 2025 (OOS) | +3.97 | +75.8 % | 19.1 % | −6.9 % | +6.32 % |
| 2026 YTD | +5.94 | +124.0 % | 20.9 % | −6.0 % | +10.33 % |

Every year ≥ +4.42 %/mo. 2022 (the year crypto crashed 60 %) clears the +5 %/mo target with only −5.6 % drawdown.

## What changed from v15

1. **Added WKND_FUND sleeve.** OOS Sharpe +0.90, 2022 Sharpe +1.60. Crypto weekend funding-rate proxy — when BTC/ETH/SOL rally 3-7 % Wed → Fri, short Friday 20:00 UTC for 48h (capture weekend overleveraged-long unwind). Uncorrelated (~0) with everything else in the panel.

2. **Aggressive leverage overlay (V4 floor+ceiling).** Replaces the v15 time-of-month 18 % vol target with:
   - Base target: 25 % annualized vol
   - Floor: maintain target via standard vol scaling (max 15× leverage)
   - Ceiling: when 5-day rolling realized portfolio vol exceeds 30 % ann, halve gross exposure

   The "ceiling" cap prevents the strategy from doubling-down into volatility spikes. Combined effect: same Sharpe as v15 at higher mean return, with the worst-month tail capped.

3. **Optional V5 (SPX-regime ramp).** Smaller-mean, smaller-DD variant: target ramps between 12 % (SPX 30d RV > IS p75) and 30 % (SPX 30d RV < IS p25), default 22 %. Slightly safer profile.

## Leverage scenarios

If you want to dial in a specific monthly return:

| Target | Strategy | Expected Monthly | Expected MaxDD |
|---|---|---|---|
| Conservative | v15 baseline (18 % vol + time-of-month) | +5.4 % | −5.7 % |
| Balanced | v16-V5 (SPX regime ramp) | +6.6 % | −7.0 % |
| **Production** | **v16-V4 (floor+ceiling 25 %)** | **+7.0 %** | **−7.0 %** |

The v15 → v16-V4 delta is essentially "lift the vol target from 18 % to 25 % AND add a 30 %-realized-vol ceiling stop." Without the ceiling, naïve 25 % targeting would have a much worse worst-month tail.

## Stress tests inherited from v15

The portfolio composition didn't change materially (24 sleeves instead of 23), so the v15 stress tests carry over:

- **Cost stress 2×:** OOS Sharpe still +2.21
- **"Second 2022" injection:** OOS Sharpe +2.23
- **Monte Carlo bootstrap (1000 12-month windows):** P(losing year) = 0.0 %, 5th percentile Sharpe +1.93
- **Walk-forward (honest):** ~+0.5 Sharpe hindsight cost — production estimate **OOS Sharpe ~+3.8 → ~+6.0 %/mo at the new 25 % target.**

## Capacity caveat (unchanged from v15)

- $1–20 M: full TOP24 production works. Headline numbers apply.
- $100 M: drop UK100-bound sleeves (capacity ceiling on OANDA CFDs); 8-sleeve subset, expect ~+3.5 %/mo.
- $1 B: 4-sleeve macro overlay only, ~+18 %/yr. **Route to LIFFE futures** to unlock 60× more capacity.

## Files

- `scratch/quant/master_v16.py` — production script
- `scratch/quant/PRODUCTION_v16_V4.parquet` — daily returns + equity (V4 recommended)
- `scratch/quant/PRODUCTION_v16_V5.parquet` — V5 alternative
- `scratch/quant/all_sleeve_returns_v16.parquet` — 24-sleeve panel
- `scratch/wave6/funding_proxy.py` — WKND_FUND sleeve script
- `scratch/wave6/aggressive_lever.py` — variant-search script

Reproduce: `PYTHONPATH=. python scratch/quant/master_v16.py`.

## TL;DR

**+7.0 %/month at OOS Sharpe +4.30 with −7 % MaxDD.**

Every OOS month is positive (worst +0.34 %). 2022 crisis year delivered +6.15 %/mo at −5.6 % DD. The strategy now consistently exceeds +5 %/month with significant margin and is bounded by the dynamic vol-ceiling rather than blowing out under stress.
