#!/bin/bash
#SBATCH --job-name=rl-dd
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

cd /home3/$USER/thesis

eval "$(conda shell.bash hook)"
source .venv/bin/activate

python -m rl_dd.experiment \
  --algo trpo \
  --depths 2,3,4,5,6 \
  --train-seeds 1-50 \
  --test-seeds 51-60 \
  --runs 3 \
  --episodes 20000 \
  --eps-decay-steps 256000 \
  --video-fps 12 \
  --cpu \
  --early-stop-episodes 0 \
  --log-dir results/50seed_manysizes_20ksteps \
  --widths 2,4,8,16,32,64,128,256,512,1024 \
  --video-seeds 41-55