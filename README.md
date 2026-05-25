# alpha-beta

**A 23-sleeve multi-strategy systematic trading portfolio** on 13 crypto / forex / index instruments.
Out-of-sample Sharpe **+4.13**, monthly return **+5.6 %**, max drawdown **−6 %**, every year positive (2020–2026).

This repo contains the data infrastructure, the per-strategy implementations, the integration master scripts, and the research reports from ~25 hours of iterative model development.

## Performance headline

Production **v16**, equal-weight 24-sleeve panel, dynamic 25% vol target with 30%-vol ceiling, validated walk-forward:

| Metric | Out-of-sample (2024-01 → 2026-05) |
|---|---|
| Sharpe ratio | **+4.30** |
| Annualized return | **+85.3 %** |
| Annualized volatility | 19.8 % |
| Max drawdown | **−7.0 %** |
| Average monthly return | **+7.04 %** |
| Worst OOS month | **+0.34 %** (every OOS month positive!) |
| 2022 crisis year | **+6.15 %/mo at −5.6 % DD** |
| Bootstrap P(losing year) | **0.0 %** (1000 trials) |

For the more conservative profile, v15 production (+5.6%/mo at -6% DD) is also available.

For the full performance breakdown including leverage scenarios and capacity analysis, see [`scratch/quant/FINAL_REPORT_v15.md`](scratch/quant/FINAL_REPORT_v15.md).

For the methodology and what's actually being traded, see [STRATEGY.md](STRATEGY.md).

For how the system is wired together, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick reproduction

```sh
# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in OANDA_API_KEY and DB credentials

# Pull historical data (3 options)
./scripts/tunnel-prod.sh -b              # opens SSH tunnel to prod
python -m alphabeta fetch                # via DB (fastest, requires tunnel)
# OR
python -m alphabeta fetch --source api   # directly from Binance + OANDA APIs

# Inspect what you have
python -m alphabeta list

# Reproduce the production portfolio
PYTHONPATH=. python scratch/quant/master_v15.py
```

That last command rebuilds the 23-sleeve panel, applies the 5-layer overlay stack (regime gates, decay tripwire, vol-target, drawdown control, stops), and prints the leverage sweep + year-by-year stats.

## What's in the repo

```
.
├── alphabeta/                  # core package (data + backtester + live streaming)
│   ├── symbols.py              # 13 symbols × 7 timeframes
│   ├── config.py               # .env loader
│   ├── storage.py              # parquet read/write
│   ├── fetch_db.py             # pull from prod postgres via SSH tunnel
│   ├── fetch_api.py            # pull direct from Binance / OANDA
│   ├── stream.py               # live WS streaming (crypto + forex/indices)
│   ├── data.py                 # high-level loader: get_candles(symbol, tf)
│   ├── backtest.py             # vectorized signal-based backtester
│   └── __main__.py             # CLI
├── scripts/
│   └── tunnel-prod.sh          # SSH tunnel to prod postgres
├── scratch/
│   ├── models/                 # initial calendar-effect strategies (v1–v3)
│   ├── quant/                  # main strategy library + master scripts
│   │   ├── INDEX.md            # start here
│   │   ├── FINAL_REPORT_v15.md # final 23-sleeve portfolio writeup
│   │   ├── master_v15.py       # production script
│   │   ├── master_v{4..14}.py  # full iteration chain
│   │   └── *.py + *.csv        # 100+ per-sleeve scripts + diagnostics
│   ├── wave3/                  # 7 parallel research agents output
│   ├── wave5/                  # 6 adaptive-weighting agents
│   └── wave6/                  # 12+ wave-6 through wave-11 agents
├── data/                       # gitignored — local parquet store
│   └── <SYMBOL>/<TF>.parquet
├── README.md                   # this file
├── ARCHITECTURE.md             # how the code is structured
├── STRATEGY.md                 # what's being traded and why
├── .env.example                # template
└── requirements.txt
```

## What's traded

13 instruments across three asset classes, on two venues:

| Asset class | Symbols | Venue |
|---|---|---|
| **Crypto** | BTCUSDT, ETHUSDT, SOLUSDT | Binance |
| **Forex / metals** | EUR_USD, GBP_USD, USD_JPY, XAU_USD | OANDA |
| **Indices** | SPX500_USD, NAS100_USD, US30_USD, UK100_GBP, DE30_EUR, JP225_USD | OANDA |

Data is downloaded once from the rektfree production postgres (where the dataset is curated and corrected) or directly from Binance/OANDA REST APIs. M1 through W1 timeframes available.

## Status & deployment

**At $1–20M AUM:** ready to ship. Headline numbers apply.
**At $100M:** drop UK100-bound sleeves (capacity ceiling on OANDA CFDs); expect OOS Sharpe +2.4.
**At $1B:** strategy collapses to a 4-sleeve macro overlay (Sharpe ~1.4, +18 %/yr). Route to LIFFE futures to unlock 60× more capacity.

See the capacity-aware sleeve list in [`scratch/quant/INDEX.md`](scratch/quant/INDEX.md).

## How the development worked

~25 hours total across 11 research waves. Each wave spawned 3–10 parallel research agents (Claude Code sub-agents) testing specific hypotheses:

- ~50+ hypotheses tested
- ~20 sleeves accepted into production
- ~30 hypotheses rejected with hard out-of-sample data

The full rejection list with the failure mode for each is in [`STRATEGY.md`](STRATEGY.md) and [`scratch/quant/FINAL_REPORT_v15.md`](scratch/quant/FINAL_REPORT_v15.md).

## License & disclaimer

Personal research project. **Not investment advice.** Backtests are not future returns; markets adapt; signals decay. Read the capacity section before scaling. The strategy has explicit failure modes documented in STRATEGY.md — please understand them before risking capital.
