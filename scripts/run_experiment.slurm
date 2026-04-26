#!/bin/bash
#SBATCH --job-name=rl-dd
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Uncomment this for a different node type
# #SBATCH --partition=regular

set -euo pipefail

PROJECT_ROOT="/home3/s5238366/thesis2"
cd "$PROJECT_ROOT"

mkdir -p slurm results

PYTHON=(uv run --no-sync python)

ARCH="cnn"
GRID_SIZE=16
MAX_STEPS=256
WIDTHS=(4 8 16 32 64 128 256)
DEPTHS=(3)
RUNS=3
TRAIN_SEEDS="1-100"
TEST_SEEDS="101-200"
EPISODES=250000
LOG_EVERY=5000
LOG_DIR="results/cnn_16x16_100seed_250k"

if [[ "${1:-}" == "--collect-only" ]]; then
  uv sync
  "${PYTHON[@]}" -m rl_dd.experiment \
    --log-dir "${LOG_DIR}" \
    --collect-only
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  uv sync
  TOTAL_TASKS=$(( ${#WIDTHS[@]} * ${#DEPTHS[@]} * RUNS ))
  ARRAY_JOB_ID=$(sbatch --parsable --array=0-$((TOTAL_TASKS - 1)) "$0")
  sbatch --dependency=afterok:"${ARRAY_JOB_ID}" "$0" --collect-only
  echo "Submitted array job ${ARRAY_JOB_ID} with ${TOTAL_TASKS} tasks."
  exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
RUN_ID=$((TASK_ID % RUNS))
PAIR_ID=$((TASK_ID / RUNS))
WIDTH_INDEX=$((PAIR_ID % ${#WIDTHS[@]}))
DEPTH_INDEX=$((PAIR_ID / ${#WIDTHS[@]}))

WIDTH="${WIDTHS[$WIDTH_INDEX]}"
DEPTH="${DEPTHS[$DEPTH_INDEX]}"

"${PYTHON[@]}" -m rl_dd.experiment \
  --algo trpo \
  --arch "${ARCH}" \
  --grid-size "${GRID_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --depths "${DEPTH}" \
  --train-seeds "${TRAIN_SEEDS}" \
  --test-seeds "${TEST_SEEDS}" \
  --runs 1 \
  --run-id "${RUN_ID}" \
  --episodes "${EPISODES}" \
  --video-fps 12 \
  --early-stop-episodes 0 \
  --start 0 \
  --end 2 \
  --log-dir "${LOG_DIR}" \
  --widths "${WIDTH}" \
  --log-every "${LOG_EVERY}"
