#!/bin/bash
#SBATCH --job-name=rl-dd
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=regular

set -euo pipefail

PROJECT_ROOT="/home3/$USER/thesis2"
cd "$PROJECT_ROOT"

mkdir -p slurm results

eval "$(conda shell.bash hook)"
source .venv/bin/activate

WIDTHS=(2 4 8 16 32 64 128 256 512 1024)
DEPTHS=(3 4 5 6 7)
RUNS=3
LOG_DIR="results/50seed_manysizes_20ksteps"

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
  --train-seeds 1-50 \
  --test-seeds 51-60 \
  --runs 1 \
  --run-id "${RUN_ID}" \
  --episodes 20000 \
  --eps-decay-steps 256000 \
  --video-fps 12 \
  --cpu \
  --early-stop-episodes 0 \
  --log-dir "${LOG_DIR}" \
  --widths "${WIDTH}" \
  --video-seeds 41-55
