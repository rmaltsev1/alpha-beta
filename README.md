# alpha-beta

Local backtesting workspace for the crypto / forex / index strategies that
live (in production) on the rektfree stack. Pulls candle data straight from
the prod postgres `candles` table (via SSH tunnel), caches it on disk as
parquet, and exposes a small loader for strategy code.

## Layout

```
.
├── alphabeta/            python package
│   ├── symbols.py        the 13 tracked symbols + 7 timeframes
│   ├── config.py         .env loader
│   ├── storage.py        parquet read/write (data/<symbol>/<tf>.parquet)
│   ├── fetch_db.py       pull from prod postgres via SSH tunnel
│   ├── fetch_api.py      pull directly from Binance / OANDA (no tunnel)
│   ├── data.py           high-level loader for strategies
│   └── __main__.py       CLI — see `python -m alphabeta -h`
├── scripts/
│   └── tunnel-prod.sh    open the SSH tunnel to prod postgres
├── data/                 gitignored — local parquet store
├── .env                  gitignored — DB password, OANDA key, etc.
└── .env.example          checked in
```

## Setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in the secrets
```

## Pulling data

```sh
./scripts/tunnel-prod.sh -b            # opens SSH tunnel in background
python -m alphabeta fetch              # incremental refresh, every symbol/tf
python -m alphabeta list               # see what's on disk
```

Other useful invocations:

```sh
python -m alphabeta fetch --full                          # re-download from 2020-01-01
python -m alphabeta fetch --symbol BTCUSDT ETHUSDT        # just two symbols
python -m alphabeta fetch --timeframe H1 H4 D1            # just three timeframes
python -m alphabeta fetch --source api                    # skip the DB, hit Binance/OANDA directly
python -m alphabeta status                                # diff local rowcounts vs. prod
```

Symbols (13): `BTCUSDT ETHUSDT SOLUSDT EUR_USD GBP_USD USD_JPY XAU_USD SPX500_USD NAS100_USD US30_USD UK100_GBP DE30_EUR JP225_USD`

Timeframes (7): `M1 M5 M15 H1 H4 D1 W1` (Postgres enum names, matches upstream).

Files are written to `data/<symbol>/<tf>.parquet`. Columns: `timestamp` (UTC),
`open`, `high`, `low`, `close`, `volume` (all float64).

## Using the data in a backtest

```python
from alphabeta import get_candles

df = get_candles("BTCUSDT", "H1", start="2023-01-01", end="2024-01-01")
# df is a pandas DataFrame indexed 0..N-1 with the columns above
```

## Refetching

`python -m alphabeta fetch` is idempotent — it only reads rows newer than
the latest local timestamp per (symbol, timeframe). The last bar in the
local file is re-fetched in case it was a still-forming partial when
cached. Run it on a schedule (cron / launchd / manual) to keep the
local store fresh.

If the SSH tunnel is closed, `--source api` falls back to the public
upstreams (Binance for crypto, OANDA for forex/indices). OANDA needs
`OANDA_API_KEY` set in `.env`.

## Live streaming

`python -m alphabeta stream` opens a Binance WebSocket for crypto and an
OANDA REST stream for forex / indices, and appends each closed candle
to the local parquet file as it arrives.

- **Crypto.** Native multi-timeframe streams — Binance emits a fresh
  closed bar for each timeframe you subscribe to.
- **Forex / indices.** OANDA only streams ticks. We aggregate them into
  M1 bars, then resample closed M1s up to M5 / M15 / H1 / H4 / D1 as the
  boundaries hit. Same trick the rektfree backend uses.

```sh
python -m alphabeta stream                       # all 13 symbols, M1..D1
python -m alphabeta stream --symbol BTCUSDT      # one crypto symbol only
python -m alphabeta stream --no-oanda            # crypto only
python -m alphabeta stream --no-binance          # fx + indices only
```

Both reconnect on drop. Ctrl-C to stop.
