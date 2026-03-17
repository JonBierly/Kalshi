#!/bin/bash
# run_backfills.sh — Runs all ETL/backfill scripts sequentially inside Docker.
# Called by cron at 9:00 AM ET daily.
#
# Usage:
#   ./run_backfills.sh              # defaults to 2025-26 season, auto-detects data source
#   ./run_backfills.sh 2024-25      # specify season
#   ./run_backfills.sh --use-s3     # force S3 data source (for VPS where stats.nba.com is blocked)

# Parse args: first positional is season, --use-s3 is optional flag
SEASON="2025-26"
S3_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --use-s3) S3_FLAG="--use-s3" ;;
        *)        SEASON="$arg" ;;
    esac
done

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Helper: run a Python module inside the orchestrator container and clean up
run_in_docker() {
    docker compose run --rm orchestrator python "$@"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting daily backfills (season: $SEASON) ${S3_FLAG:+(S3 mode)}"

echo "[$(date '+%H:%M:%S')] Running ETL pipeline..."
run_in_docker -m spread_src.scripts.etl_pipeline --seasons "$SEASON" $S3_FLAG 2>&1 || echo "  ⚠️ ETL pipeline failed"

echo "[$(date '+%H:%M:%S')] Running team logs backfill..."
run_in_docker -m spread_src.scripts.backfill_team_logs --seasons "$SEASON" $S3_FLAG 2>&1 || echo "  ⚠️ Team logs failed"

echo "[$(date '+%H:%M:%S')] Running stats backfill..."
run_in_docker -m spread_src.scripts.backfill_stats --seasons "$SEASON" $S3_FLAG 2>&1 || echo "  ⚠️ Stats backfill failed"

echo "[$(date '+%H:%M:%S')] Running prediction outcome backfill..."
run_in_docker -m spread_src.scripts.backfill_outcomes 2>&1 || echo "  ⚠️ Outcomes backfill failed"

echo "[$(date '+%H:%M:%S')] Running trade settlements..."
run_in_docker -m spread_src.scripts.settle_historical_trades 2>&1 || echo "  ⚠️ Trade settlements failed"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ All backfills complete"
