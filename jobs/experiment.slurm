#!/usr/bin/env bash
# Submit manually after setting site-specific account, partition and resources.
# The array must be contiguous 0..N-1. There are no implicit allocation defaults.
set -euo pipefail

: "${BATCH:?Set BATCH to the frozen batch directory}"
: "${ATTEMPTS:?Set ATTEMPTS to the complete-attempt output directory}"
: "${BACKEND:?Set BACKEND explicitly to default.qubit or lightning.gpu}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the approved environment interpreter}"
: "${SLURM_ARRAY_TASK_ID:?Submit a contiguous array starting at zero}"
: "${SLURM_ARRAY_TASK_MIN:?Slurm array minimum is required}"
: "${SLURM_ARRAY_TASK_MAX:?Slurm array maximum is required}"
SHARD_COUNT=${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:?Set SHARD_COUNT or use the Slurm array count}}
if [[ "$SLURM_ARRAY_TASK_MIN" -ne 0 || "$SLURM_ARRAY_TASK_MAX" -ne $((SHARD_COUNT - 1)) \
      || "${SLURM_ARRAY_TASK_STEP:-1}" -ne 1 \
      || "${SLURM_ARRAY_TASK_COUNT:-$SHARD_COUNT}" -ne "$SHARD_COUNT" ]]; then
    printf '%s\n' 'The Slurm array must be exactly 0..SHARD_COUNT-1, with step one.' >&2
    exit 2
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PYTHON_BIN" -u scripts/experiment.py run --batch "$BATCH" --output "$ATTEMPTS" \
    --backend "$BACKEND" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$SHARD_COUNT" --execute
