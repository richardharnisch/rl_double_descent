# DQN Double Descent in Seeded Gridworlds

This repo is a minimal, reproducible setup for probing double descent in RL.
We train a DQN on a small set of seeded environments (e.g., seeds 1-5) and
evaluate on unseen seeds. We then sweep model sizes to observe performance
curves as parameter count increases.

## Setup (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Run an experiment

Example sweep with a 2-layer MLP and width sweep:

```bash
python -m rl_dd.experiment \
  --train-seeds 1-25 \
  --test-seeds 26-30 \
  --widths 16,32,64,128,256 \
  --depths 2 \
  --episodes 2000 \
  --max-steps 64 \
  --log-dir results/run_001
```

If `--log-dir` is omitted, logs are written to `results/<timestamp>`.

Each run creates its own subdirectory inside `--log-dir` (e.g., `w64_d2_run0/`)
with per-episode CSVs, episode plots, train/test GIFs, and the saved model
weights. For TRPO runs, each run directory also includes `trpo_updates.csv`
(mean KL and entropy per update).

The top-level log directory contains:
- `metrics.csv`: per-run metrics including parameter count, train return, and test return.
- `summary.csv`: aggregated metrics per parameter count (mean/std).
- `curve.png`: train/test return vs parameter count plus FIM/parameter panel.

This also logs:
- `fim_trace`: Hutchinson-estimated Fisher trace at the end of training.
- Videos (GIFs) for seeds listed in `--video-seeds`, saved inside each run directory.
- A second plot panel for `fim_trace / num_params` vs parameter count.

Disable FIM if needed:

```bash
python -m rl_dd.experiment --fim-samples 0
```

## Parameters (in depth)

Seed lists
- `--train-seeds` (default: `1-25`): Seed list for training environments. Accepts comma lists and ranges (e.g., `1,2,10-20`). Each training episode samples uniformly from this list.
- `--test-seeds` (default: `26-30`): Seed list for evaluation environments. Same format as `--train-seeds`, used only for evaluation.

Model size
- `--widths` (default: `16,32,64,128,256,512`): Comma-separated hidden layer widths to sweep (e.g., `16,32,64`).
- `--depths` (default: `2`): Comma/range list of depths to sweep (e.g., `2,3,4` or `2-5`). Every width is tested against every depth.
- `--runs` (default: `1`): Number of independent runs per width (different RNG seeds and replay buffer sampling).
- `--base-seed` (default: `0`): Base RNG seed for runs; run `k` uses `base_seed + k` for all RNGs.
- `--run-id` (default: unset): Force a single run index (use with `--runs 1`) so array jobs can map distinct seeds to distinct runs.
- `--algo` (default: `dqn`): Algorithm choice (`dqn` or `trpo`). DQN-specific flags are ignored when using TRPO.

Environment
- `--grid-size` (default: `8`): Square grid side length. Observation uses a per-tile one-hot encoding (4 channels) and is flattened. Also has 2-frame stacking.
- `--obstacle-prob` (default: `0.2`): Bernoulli probability of a wall in each cell (except start/goal). Maps are regenerated per seed until solvable.
- `--max-steps` (default: `64`): Maximum steps per episode before truncation (applies to training, eval, and video rollouts).
- `--start` (default: unset): Start corner index (0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left). Unset means randomized.
- `--end` (default: unset): Goal corner index (0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left). Unset means randomized.
When one of `--start` or `--end` is unset, the other is sampled from the remaining corners; if both are unset, both corners are randomized (but always different).
The observation is a 2-frame stack: two consecutive one-hot grids are flattened and concatenated.

Training
- `--episodes` (default: `2000`): Training episodes per run (episodes are sampled across training seeds).
- `--batch-size` (default: `64`): Replay batch size for each DQN update.
- `--gamma` (default: `0.99`): Discount factor used in TD target computation.
- `--lr` (default: `1e-3`): Adam learning rate.
- `--buffer-size` (default: `50000`): Replay buffer capacity (FIFO overwrite).
- `--target-update` (default: `500`): Target network sync interval in environment steps. Set `0` to disable updates.
- `--eps-start` (default: `1.0`): Initial epsilon for epsilon-greedy exploration.
- `--eps-end` (default: `0.05`): Final epsilon after decay.
- `--eps-decay-episodes` (default: 30% of `--episodes`): Linear decay horizon in episodes from `eps-start` to `eps-end`.
- `--early-stop-return` (default: `0.7`): Return threshold for early stopping.
- `--early-stop-episodes` (default: `10`): Number of consecutive episodes above `early-stop-return` required to stop; set `0` to disable.

TRPO-specific
- `--trpo-max-kl` (default: `1e-2`): KL divergence trust region size for the policy update.
- `--trpo-cg-iters` (default: `10`): Conjugate gradient iterations for the natural gradient step.
- `--trpo-cg-damping` (default: `0.1`): Damping added to the Fisher vector product for numerical stability.
- `--trpo-backtrack-coeff` (default: `0.5`): Backtracking line search shrink factor.
- `--trpo-backtrack-iters` (default: `10`): Maximum line search steps for a safe policy update.
- `--trpo-vf-iters` (default: `5`): Value function optimization steps per policy update.
- `--trpo-vf-lr` (default: `1e-3`): Value function learning rate (Adam).
- `--trpo-gae-lambda` (default: `0.95`): GAE lambda used for advantage estimation.
- `--trpo-batch-episodes` (default: `20`): Number of episodes collected per TRPO policy update.

Evaluation
- `--eval-episodes` (default: `1`): Episodes per seed for evaluation; averages across all test seeds and episodes.

Logging and plots
- `--log-x` / `--no-log-x` (default: log-x enabled): Use or disable log scale on the x-axis for plots.
- `--log-dir` (default: `results/<timestamp>`): Base directory for all logs. Each run writes to its own subdirectory.
- `--collect-only` (default: disabled): Skip training and compile `metrics.csv`, `summary.csv`, and `curve.png` from existing run directories.

Videos
- `--video-seeds` (default: unset): Comma/range seed list to render as GIFs (e.g., `1,2,10-12`). When unset, renders the first 5 training seeds and first 5 test seeds (or all if fewer). Use `--video-seeds none` to disable.
- `--video-fps` (default: `6`): Playback FPS for saved GIFs.

Episode curves
Per-run episode CSVs and plots are always saved inside each run's log directory.

FIM (Hutchinson)
- `--fim-samples` (default: `64`): Number of (state, action) samples used to estimate the Fisher trace; `0` disables FIM logging.
- `--fim-hutchinson` (default: `4`): Number of Rademacher probe vectors per sample. More probes reduces estimator variance.

Sanity check (learnability)
- `--sanity-check` (default: disabled): Run the obstacle-free learnability check before the main experiment.
- `--sanity-only` (default: disabled): Run only the sanity check and exit without running the full sweep.
- `--sanity-episodes` (default: `800`): Training episodes for the sanity check.
- `--sanity-threshold` (default: `0.8`): Minimum average return required to pass the sanity check.

Hardware
- `--cpu` (default: disabled): Force CPU even if CUDA is available. Use for strict determinism.

## Notes

- The environment is a seeded gridworld with random obstacles; each seed is a
  deterministic map with a guaranteed path from start to goal.
- Training samples episodes from the training seeds uniformly.
- Overfitting can be observed by a widening gap between train/test returns.
- RNG is seeded for Python, NumPy, Torch, and the replay buffer. For strongest
  determinism, run on CPU (use `--cpu`).

## Sanity check (learnability)

Run a quick learnability check on an obstacle-free grid:

```bash
python -m rl_dd.experiment --sanity-check --sanity-only --cpu
```

## Evaluate a saved model

Save videos from a trained run:

```bash
python -m rl_dd.eval \
  --algo dqn \
  --model-path results/run_001/w64_d2_run0/model.pt \
  --seeds 1,2,3,10-12 \
  --out-dir results/run_001/w64_d2_run0/eval_videos
```

If `metrics.csv` is present in the same directory as the model, width/depth are
inferred automatically. Otherwise pass `--width` and `--depth`.

## SLURM

Edit and submit the script in `scripts/run_experiment.slurm`:

```bash
sbatch scripts/run_experiment.slurm
```
