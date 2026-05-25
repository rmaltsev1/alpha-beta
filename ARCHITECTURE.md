# Architecture

How the alpha-beta system is wired together — data → signal → portfolio → execution.

## Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: PRODUCTION OVERLAYS (5 of them, applied in order) │
│  - per-sleeve regime gates                                  │
│  - fast decay tripwire (63-day Sharpe × 2 monthly checks)   │
│  - time-of-month vol-target overlay (18 % base)             │
│  - drawdown control (halve at 3 % rolling DD)               │
│  - per-sleeve stop-losses on 6 sleeves                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: PORTFOLIO INTEGRATION                             │
│  - 23 sleeves equal-weighted                                │
│  - each rescaled to 5 % IS-annualized vol                   │
│  - master_v{4..15}.py                                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: STRATEGY SLEEVES                                  │
│  - 23 independent signal generators                         │
│  - each saves a daily-aligned returns parquet               │
│  - scratch/{models,quant,wave3,wave5,wave6}/<sleeve>.py     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: BACKTEST ENGINE                                   │
│  - vectorized signal-based                                  │
│  - alphabeta/backtest.py                                    │
│  - inputs: dataframe + position series                      │
│  - outputs: equity, returns, stats                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: DATA INFRASTRUCTURE                               │
│  - local parquet store (data/<SYMBOL>/<TF>.parquet)         │
│  - 3 acquisition paths: prod DB, Binance API, OANDA API     │
│  - live streaming for refresh                               │
│  - alphabeta/{fetch_db,fetch_api,stream,storage}.py         │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1 — Data infrastructure

### Storage schema

```
data/
├── BTCUSDT/
│   ├── M1.parquet      # 1-minute bars, ~3.4M rows
│   ├── M5.parquet
│   ├── M15.parquet
│   ├── H1.parquet      # 1-hour bars, ~56k rows
│   ├── H4.parquet
│   ├── D1.parquet      # daily, ~2300 rows
│   └── W1.parquet
├── ETHUSDT/
└── ... (13 symbols total)
```

Each parquet has 6 columns: `timestamp` (UTC tz-aware), `open`, `high`, `low`, `close`, `volume`. All float64. ~30M candles total, ~600 MB on disk.

### Acquisition paths

Three independent ways to populate the local store:

1. **From production DB** (`alphabeta/fetch_db.py`): pulls from the rektfree postgres `candles` table via SSH tunnel. Fastest, deduplicated, curated. Requires `./scripts/tunnel-prod.sh` running.

2. **From upstream APIs** (`alphabeta/fetch_api.py`): direct from Binance for crypto, OANDA for forex/indices. No tunnel needed. Paginated, walk-forward-safe.

3. **Live streaming** (`alphabeta/stream.py`): WebSocket (Binance) + REST stream (OANDA). Appends closed bars to parquet as they arrive. For paper/live trading on top of backtested strategies.

All three converge on the same on-disk schema, so the rest of the system is acquisition-agnostic.

### Configuration

`.env` holds the DB password, OANDA API key, SSH tunnel target. Loaded once by `alphabeta/config.py` into a `Settings` dataclass. Never committed; `.env.example` shows the structure.

## Layer 2 — Backtest engine

`alphabeta/backtest.py` (~150 lines).

Convention: a strategy produces a `position` series (values in [−1, +1]) where `position[t]` is the position held *during* bar t. Engine multiplies position by bar return, subtracts cost on `|Δposition|`, returns equity curve + stats.

```python
from alphabeta.backtest import backtest
result = backtest(df, position, symbol="BTCUSDT", timeframe="D1", name="my_strategy")
result.equity      # cumulative equity (starts at 1.0)
result.returns     # per-bar net returns
result.stats       # dict: sharpe, max_dd, hit_rate, etc
```

Cost model: per-side in basis points, asset-class-specific. FX 1 bp, Index 1.5 bp, Crypto 5 bp. Configurable per call. Cost is charged on absolute position change at each bar (so a flat→long event = 1 cost; long→short = 2 costs because notional turnover is 2×).

The engine is intentionally minimal:
- **Vectorized**, not event-driven. Faster, simpler.
- **No look-ahead enforcement** — strategies are trusted to shift their own signals.
- **No slippage model beyond constant spread.** Real-money deployment needs an execution layer.

## Layer 3 — Strategy sleeves

23 independent signal generators, each producing a daily-aligned returns parquet that the portfolio integration layer consumes.

Sleeves are organized by research wave / theme:

| Folder | Wave | Contents |
|---|---|---|
| `scratch/models/` | v1–v3 | Initial calendar-effect strategies (EVE_XAU, WED_BTC, D1REV_*) + first master scripts |
| `scratch/quant/` | v4–v15 | Master integration scripts + pairs / TSMOM / XSMOM / risk-parity / vol-managed / defensive |
| `scratch/wave3/` | wave 3 | 7 parallel agents: TSMOM ensemble, XSMOM, pairs, vol-managed, risk-parity, defensive, walk-forward |
| `scratch/wave5/` | wave 5 | 6 adaptive-weighting agents (most rejected) |
| `scratch/wave6/` | waves 6–11 | Most accepted sleeves: H4, CORR_REGIME, W1_STRATS, MICROSTR_D1, STATARB_XS, VOL_BREAKOUT, TERM_SPREADS, EURGBP_MR, HMM_BULL_TSMOM, etc. |

Each sleeve script:
1. Reads candles via `alphabeta.data.get_candles(symbol, timeframe)`.
2. Computes a position series following the no-look-ahead convention.
3. Vol-scales the position on IS data to 5 % annualized vol (so sleeves are commensurate).
4. Saves daily returns parquet: `<sleeve>_returns.parquet` with `timestamp` (UTC) + `ret` columns.

The pattern is consistent across all 23 sleeves, which made the integration layer trivial.

## Layer 4 — Portfolio integration

The `master_v{4..15}.py` series progressively combines sleeves. The final production script is `master_v15.py`.

Integration steps:

1. **Load all sleeve returns parquets** → wide DataFrame, one column per sleeve, indexed by UTC timestamp.
2. **Each sleeve already vol-scaled to 5 % IS-annualized vol** (done at the sleeve level). Sanity check.
3. **Apply per-sleeve regime gates** — multiply each column by a regime-specific scaler when the SPX 30-day realized vol is above the IS 80th percentile. Mean-reversion / pairs / defensive sleeves get amplified (×1.5); directional / trend / beta get halved (×0.5).
4. **Apply fast decay tripwire** — for each sleeve, compute trailing-63-day Sharpe at each monthly checkpoint. If negative for 2 consecutive checks, zero the sleeve for the next month. Re-entry when trailing Sharpe > 0.5.
5. **Equal-weight aggregation** across the 23 (gated, tripwired) sleeves → portfolio return series.
6. **Time-of-month vol-target overlay** — base 18 %, lifted to 20 % on days 1–10, trimmed to 16 % on days 11–20. Max leverage cap 15×.
7. **Drawdown control** — when portfolio's 30-day rolling DD exceeds 3 %, halve total gross. Recover to full when DD < 1 %.

## Layer 5 — Production overlays (summary)

Already described inline in Layer 4. The five overlays in order:

| # | Overlay | Effect |
|---|---|---|
| 1 | Per-sleeve regime gates | Reshapes sleeve mix in high-vol regimes |
| 2 | Fast decay tripwire | Catches dead sleeves in ~75 days (vs 6 mo for trailing-12m) |
| 3 | Time-of-month vol-target | +0.09 Sharpe Pareto improvement |
| 4 | Drawdown control | Halves gross at -3 % rolling DD |
| 5 | Per-sleeve stop-losses (TREND_NEW, PAIRS_EXP, RISKPAR, DEFEND, EVENT_VOLSPIKE, CRYPTO_vs_SPX) | +0.13–0.34 Sharpe per sleeve |

## Walk-forward discipline

Everything that *could* peek ahead is walk-forward:

- Volatility estimators use only past data (typically 30d or 60d trailing windows).
- Cointegration tests on pairs use 252d rolling windows ending at t-1.
- Regime gates use IS-fit thresholds, not full-sample percentiles.
- The HMM regime model is refit every 6 months on trailing 252 days.
- Decay tripwire trailing-63d Sharpe is computed on data ending at t-1.

The one exception: **sleeve selection itself.** The 23 sleeves were chosen based on IS+OOS performance. This is hindsight selection bias worth ~0.5 OOS Sharpe (validated via walk-forward simulator in `v9_walk_forward.py` and `v14_walk_forward.py`).

Honest walk-forward production estimate: **OOS Sharpe ~+3.4 → +4.8 %/month at 18 % vol target.**

## CLI

Everything user-facing goes through `python -m alphabeta`:

```
python -m alphabeta list                          # local data inventory
python -m alphabeta fetch                         # incremental refresh from prod DB
python -m alphabeta fetch --source api            # bypass DB, hit upstream APIs
python -m alphabeta fetch --full                  # re-download from 2020-01-01
python -m alphabeta fetch --symbol BTCUSDT        # one symbol
python -m alphabeta fetch --timeframe H1 H4 D1    # subset
python -m alphabeta status                        # local vs prod row counts
python -m alphabeta stream                        # live WebSocket feed
```

Strategy reproduction is via the master scripts:

```sh
PYTHONPATH=. python scratch/quant/master_v15.py            # production portfolio
PYTHONPATH=. python scratch/quant/v14_walk_forward.py      # honest WF validation
PYTHONPATH=. python scratch/quant/v9_stress_full.py        # cost stress + Monte Carlo
```

## Iteration chain

The progression of master scripts shows the cumulative additions:

| Version | OOS Sharpe | Key addition |
|---|---|---|
| v3 | +2.37 | 7 calendar sleeves |
| v8 | +2.52 | + 11-pair cointegrated pairs |
| v11 | +2.82 | + DEFEND safe-haven |
| v12 | +3.11 | + VOLFORECAST, TREND_NEW, H4, CORR_REGIME, SESSION_MOM |
| v13 | +3.60 | + W1, EVENT, STATARB, MICROSTR, VOL_BREAKOUT, TERM_SPREADS, EURGBP, MULTIDAY |
| v14 | +4.04 | + per-sleeve stop-losses |
| **v15** | **+4.13** | + HMM_BULL_TSMOM |

Each version is a separate `master_v{N}.py` and `PRODUCTION_v{N}.parquet` — fully reproducible, version-controlled iterations.

## Live trading pathway (not implemented)

To take this from backtest to live trading you'd need to add:

1. **Execution layer** — wraps the position vector into broker orders (OANDA's REST or Binance's). Already have credentials and tested REST clients in `fetch_api.py` and `stream.py`.
2. **Order management** — child-order slicing, retry logic, fill confirmation.
3. **Real-time position state** — sync local intent with broker actual.
4. **Monitoring** — sleeve-level Sharpe, drawdown, position breaches; alerting.
5. **Capacity routing** — for AUM > $20M, route index trades to futures (LIFFE for FTSE, CME for SPX) instead of OANDA CFDs.

None of these are in the current repo. The backtest infrastructure is solid; the live execution stack is the next building block.
