#!/bin/bash
#SBATCH --job-name=rl-dd
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Uncomment this for a different job
# # SBATCH --partition=regular

set -euo pipefail

PROJECT_ROOT="/home3/s5238366/thesis2"
cd "$PROJECT_ROOT"

mkdir -p slurm results

eval "$(conda shell.bash hook)"
source .venv/bin/activate

WIDTHS=(1024 2048 4096)
DEPTHS=(8 9 10)
RUNS=3
LOG_DIR="results/100seed_manysizes_50k"

if [[ "${1:-}" == "--collect-only" ]]; then
  python -m rl_dd.experiment \
    --log-dir "${LOG_DIR}" \
    --collect-only
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
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

python -m rl_dd.experiment \
  --algo trpo \
  --depths "${DEPTH}" \
  --train-seeds 1-100 \
  --test-seeds 101-200 \
  --runs 1 \
  --run-id "${RUN_ID}" \
  --episodes 20000 \
  --video-fps 12 \
  --early-stop-episodes 0 \
  --start 0 \
  --end 2 \
  --log-dir "${LOG_DIR}" \
  --widths "${WIDTH}" \
  --video-seeds 96-105
