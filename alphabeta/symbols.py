"""The 13 assets and 7 timeframes mirrored from rektfree-backend.

Keep this in sync with `app/services/market_data.py` upstream. If they add
a symbol there, add it here too — order matters for log readability.
"""
from __future__ import annotations

from enum import Enum


class AssetType(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDEX = "index"


CRYPTO: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
FOREX:  list[str] = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
INDEX:  list[str] = ["SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP", "DE30_EUR", "JP225_USD"]

ALL_SYMBOLS: list[str] = CRYPTO + FOREX + INDEX

SYMBOL_TYPE: dict[str, AssetType] = (
    {s: AssetType.CRYPTO for s in CRYPTO}
    | {s: AssetType.FOREX for s in FOREX}
    | {s: AssetType.INDEX for s in INDEX}
)

# Postgres stores the SQLAlchemy enum member *names* (M1/M5/M15/H1/H4/D1/W1),
# not the values. We use the same labels here so everything (DB queries,
# parquet filenames, CLI args) speaks one vocabulary.
TIMEFRAMES: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1", "W1"]

# Seconds in each timeframe — useful for paginating API fetches.
TF_SECONDS: dict[str, int] = {
    "M1":  60,
    "M5":  300,
    "M15": 900,
    "H1":  3_600,
    "H4":  14_400,
    "D1":  86_400,
    "W1":  604_800,
}

# Binance interval strings (only used for the API fetch fallback).
BINANCE_TF = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w"}

# OANDA granularity strings.
OANDA_TF = {"M1": "M1", "M5": "M5", "M15": "M15", "H1": "H1", "H4": "H4", "D1": "D", "W1": "W"}
