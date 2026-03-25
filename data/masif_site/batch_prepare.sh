#!/usr/bin/env bash
# batch_prepare.sh: Parallel batch data preparation for MaSIF-site.
#
# Usage:
#   ./batch_prepare.sh [--workers N] [--list FILE] [--skip-existing]
#
# Defaults: 4 parallel workers, lists/training.txt, skip already-processed proteins.
#
# Each entry in the list is a PPI_PAIR_ID like 1A0G_B or 1A2K_AB_CD.
# Progress is logged to logs/batch_prepare.log; per-protein logs go to
# logs/proteins/<ID>.log.

set -euo pipefail

WORKERS=4
LIST="lists/training.txt"
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --list)    LIST="$2";    shift 2 ;;
        --no-skip) SKIP_EXISTING=0; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

masif_root=$(git rev-parse --show-toplevel)
masif_source="$masif_root/source"
export PYTHONPATH="${PYTHONPATH:-}:$masif_source"

PRECOMP_DIR="data_preparation/04a-precomputation_9A/precomputation"
LOG_DIR="logs/proteins"
mkdir -p "$LOG_DIR"
MAIN_LOG="logs/batch_prepare.log"

total=$(wc -l < "$LIST")
echo "[$(date)] Starting batch preparation: $total proteins, $WORKERS workers" | tee -a "$MAIN_LOG"
echo "[$(date)] List: $LIST" | tee -a "$MAIN_LOG"

# Worker function — process one PPI_PAIR_ID
process_one() {
    local PPI_PAIR_ID="$1"
    local LOG="$LOG_DIR/${PPI_PAIR_ID}.log"

    PDB_ID=$(echo "$PPI_PAIR_ID" | cut -d_ -f1)
    CHAIN1=$(echo "$PPI_PAIR_ID" | cut -d_ -f2)
    CHAIN2=$(echo "$PPI_PAIR_ID" | cut -d_ -f3 || true)

    # Skip if precomputation output already exists
    if [[ "$SKIP_EXISTING" -eq 1 && -d "$PRECOMP_DIR/$PPI_PAIR_ID" ]]; then
        echo "[SKIP] $PPI_PAIR_ID (already precomputed)"
        return 0
    fi

    echo "[START] $PPI_PAIR_ID" >> "$MAIN_LOG"
    {
        echo "=== $PPI_PAIR_ID $(date) ==="

        # Download PDB
        python3 -W ignore "$masif_source/data_preparation/00-pdb_download.py" "$PPI_PAIR_ID"

        # Triangulate
        python3 -W ignore "$masif_source/data_preparation/01-pdb_extract_and_triangulate.py" "${PDB_ID}_${CHAIN1}"
        if [[ -n "${CHAIN2:-}" ]]; then
            python3 -W ignore "$masif_source/data_preparation/01-pdb_extract_and_triangulate.py" "${PDB_ID}_${CHAIN2}"
        fi

        # Precompute geodesic patches
        python3 "$masif_source/data_preparation/04-masif_precompute.py" masif_site "$PPI_PAIR_ID"

        echo "=== DONE $PPI_PAIR_ID $(date) ==="
    } > "$LOG" 2>&1

    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[DONE]  $PPI_PAIR_ID" >> "$MAIN_LOG"
    else
        echo "[FAIL]  $PPI_PAIR_ID (exit $rc)" >> "$MAIN_LOG"
    fi
    return $rc
}

export -f process_one
export masif_source PRECOMP_DIR LOG_DIR MAIN_LOG SKIP_EXISTING

# Use xargs for parallel execution (works on macOS without GNU parallel)
cat "$LIST" | xargs -P "$WORKERS" -I{} bash -c 'process_one "$@"' _ {}

echo "[$(date)] Batch preparation complete." | tee -a "$MAIN_LOG"

# Summary
done_count=$(grep -c '^\[DONE\]' "$MAIN_LOG" 2>/dev/null || echo 0)
fail_count=$(grep -c '^\[FAIL\]' "$MAIN_LOG" 2>/dev/null || echo 0)
echo "Done: $done_count / $total   Failed: $fail_count" | tee -a "$MAIN_LOG"
