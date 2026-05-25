#!/usr/bin/env bash
# Full rebuild of the alpha-beta strategy panel.
#
# Refetches market data, re-runs all individual sleeve scripts to produce
# fresh per-sleeve return parquets, then re-runs the master chain (v3 → v16)
# to assemble the integrated portfolio. The end product is a fresh
# PRODUCTION_v16_V4.parquet reflecting the latest market data.
#
# Use case: weekly (or daily) "true freshness" rebuild — without this, the
# scheduled fire reuses frozen sleeve outputs and signals don't update.
#
# Usage:
#   ./scripts/rebuild_all.sh                       # full rebuild (~3-5 min)
#   ./scripts/rebuild_all.sh --no-fetch            # skip candle refetch
#   ./scripts/rebuild_all.sh --skip-sleeves        # only re-run master chain

set -uo pipefail   # NOT -e — we want to continue on non-fatal errors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found in $REPO_ROOT" >&2
    exit 2
fi
source .venv/bin/activate

FETCH=1
SLEEVES=1
for arg in "$@"; do
    case "$arg" in
        --no-fetch) FETCH=0 ;;
        --skip-sleeves) SLEEVES=0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

ts() { date -u +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }
log_err() { echo "[$(ts)] ERROR: $*" >&2; }

START_TS=$(date +%s)

# -----------------------------------------------------------------------------
# Step 1: refetch candles
# -----------------------------------------------------------------------------
if [[ $FETCH -eq 1 ]]; then
    log "step 1/3: fetching latest candles (Binance + OANDA APIs)..."
    if ! python -m alphabeta fetch --source api; then
        log_err "fetch failed — continuing with stale data on disk"
    fi
else
    log "step 1/3: skipping candle fetch (--no-fetch)"
fi

# -----------------------------------------------------------------------------
# Step 2: re-run individual sleeve scripts that feed the master chain
# -----------------------------------------------------------------------------
# These are the sleeve scripts that produce parquets read by master_v3..v16.
# Each is independent and fast (< 5s typically). We run them sequentially —
# parallelism is possible but the file I/O contention isn't worth the gain.
SLEEVE_SCRIPTS=(
    # Initial 11-sleeve panel (master_portfolio -> all_sleeve_returns.parquet)
    "scratch/quant/master_portfolio.py"
    # Individual sleeves feeding the master chain
    "scratch/quant/defensive.py"
    "scratch/quant/risk_parity.py"
    "scratch/quant/rotation.py"
    "scratch/quant/tsmom.py"
    "scratch/quant/xsmom.py"
    "scratch/quant/tail_v2.py"
    "scratch/quant/tail_sleeve.py"
    "scratch/quant/pairs_expanded.py"
    "scratch/quant/intraday.py"
    "scratch/quant/crypto_dom.py"
    "scratch/quant/crypto_micro.py"
    "scratch/quant/volmgmt.py"
    "scratch/quant/volume_signals.py"
    # Wave 3
    "scratch/wave3/trend_strategies.py"
    "scratch/wave3/h4_strategies.py"
    "scratch/wave3/corr_regime.py"
    "scratch/wave3/session_momentum.py"
    "scratch/wave3/volforecast.py"
    "scratch/wave3/multi_confirm.py"
    "scratch/wave3/carry_drift.py"
    "scratch/wave3/synthetic_spreads.py"
    "scratch/wave3/rel_strength.py"
    "scratch/wave3/crypto_alpha.py"
    "scratch/wave3/session_reversion.py"
    "scratch/wave3/vwap_strategies.py"
    "scratch/wave3/kelly_sizing.py"
    "scratch/wave3/ml_meta.py"
    # Wave 6 — most accepted sleeves are here
    "scratch/wave6/funding_proxy.py"
    "scratch/wave6/fx_specific.py"
    "scratch/wave6/hmm_regime.py"
    "scratch/wave6/hmm_weighting.py"
    "scratch/wave6/event_clusters.py"
    "scratch/wave6/multiday_patterns.py"
    "scratch/wave6/statarb_ensembles.py"
    "scratch/wave6/microstructure_d1.py"
    "scratch/wave6/vol_breakout.py"
    "scratch/wave6/term_spreads.py"
    "scratch/wave6/w1_strategies.py"
    "scratch/wave6/stops_extended.py"
    "scratch/wave6/multi_leg_spreads.py"
    "scratch/wave6/mom_quality.py"
    "scratch/wave6/skewness.py"
    "scratch/wave6/anomaly_detection.py"
    "scratch/wave6/adaptive_reversion.py"
    "scratch/wave6/classical_indicators.py"
    "scratch/wave6/dd_conditional.py"
    "scratch/wave6/dd_recovery.py"
    "scratch/wave6/regime_activation.py"
    "scratch/wave6/per_asset_voltarget.py"
)

if [[ $SLEEVES -eq 1 ]]; then
    log "step 2/3: running ${#SLEEVE_SCRIPTS[@]} individual sleeve scripts..."
    SLEEVE_OK=0
    SLEEVE_FAIL=0
    SLEEVE_MISSING=0
    for script in "${SLEEVE_SCRIPTS[@]}"; do
        if [[ ! -f "$script" ]]; then
            SLEEVE_MISSING=$((SLEEVE_MISSING + 1))
            continue
        fi
        if PYTHONPATH=. python "$script" > "/tmp/sleeve_$(basename "$script" .py).log" 2>&1; then
            SLEEVE_OK=$((SLEEVE_OK + 1))
            printf "."
        else
            SLEEVE_FAIL=$((SLEEVE_FAIL + 1))
            printf "x"
            echo
            log_err "$script — last lines:"
            tail -3 "/tmp/sleeve_$(basename "$script" .py).log" | sed 's/^/    /' >&2
        fi
    done
    echo
    log "sleeves: $SLEEVE_OK ok, $SLEEVE_FAIL failed, $SLEEVE_MISSING missing (logs in /tmp/sleeve_*.log)"
else
    log "step 2/3: skipping sleeves (--skip-sleeves)"
fi

# -----------------------------------------------------------------------------
# Step 3: run the master chain v3 → v9 → ... → v16
# -----------------------------------------------------------------------------
# Each master reads the previous master's panel + adds new sleeves. Order
# matters here — must be sequential. master_v6/v7/v8 are intermediate
# "explore" branches, not in the main chain.
MASTER_CHAIN=(v3 v9 v10 v11 v12 v13 v14 v15 v16)

log "step 3/3: running master chain (${#MASTER_CHAIN[@]} versions)..."
MASTER_OK=0
MASTER_FAIL=0
for v in "${MASTER_CHAIN[@]}"; do
    if PYTHONPATH=. python "scratch/quant/master_${v}.py" > "/tmp/master_${v}.log" 2>&1; then
        MASTER_OK=$((MASTER_OK + 1))
        log "  ✓ master_${v}"
    else
        MASTER_FAIL=$((MASTER_FAIL + 1))
        log_err "  ✗ master_${v} — see /tmp/master_${v}.log"
        # Continue — downstream masters may still work if the parquet was saved before crash
    fi
done

# -----------------------------------------------------------------------------
# Done — report
# -----------------------------------------------------------------------------
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

log "================================"
log "rebuild complete in ${ELAPSED}s"
log "masters: $MASTER_OK ok, $MASTER_FAIL failed"
if [[ $SLEEVES -eq 1 ]]; then
    log "sleeves: $SLEEVE_OK ok, $SLEEVE_FAIL failed, $SLEEVE_MISSING missing"
fi

# Check final output
if [[ -f "scratch/quant/PRODUCTION_v16_V4.parquet" ]]; then
    log "latest bar in PRODUCTION_v16_V4.parquet:"
    python -c "
import pandas as pd
df = pd.read_parquet('scratch/quant/PRODUCTION_v16_V4.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
print(f'    last row: {df[\"timestamp\"].iloc[-1]}  ({len(df)} rows)')
"
fi

exit 0
