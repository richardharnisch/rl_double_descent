# DQN Double Descent in Seeded Gridworlds

This repo is a reproducible setup for probing double descent in RL.
We train a DQN/TRPO on a set of seeded gridworld environments (e.g., seeds 1-5) and
evaluate on unseen seeds. We then sweep model sizes to observe performance
curves as parameter count increases.

The runner also supports a live contextual-bandit task for a short-horizon
stochastic-MDP search. Bandit contexts are regenerated from episode seeds and
reward noise is sampled by the environment on every action; no frozen
observation/reward training table is used.

## Setup (uv)

```bash
uv sync
```

## Run an experiment

Example sweep with a 2-layer MLP and width sweep:

```bash
uv run python -m rl_dd.experiment \
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

Each run also records train/test evaluation return standard deviation, action
entropy, and state-visitation coverage. Bandit runs additionally record the
optimal-action rate. Repeated stochastic evaluation episodes keep the seeded
task context fixed while transition randomness advances.

This also logs:
- `fim_trace`: Fisher trace estimated as the average of $\|\nabla_\theta \log \pi_\theta(a|s)\|^2$ over sampled state-action pairs.
- Videos (GIFs) for seeds listed in `--video-seeds`, saved inside each run directory.
- A second plot panel for `fim_trace / num_params` vs parameter count.

Disable FIM if needed:

```bash
uv run python -m rl_dd.experiment --fim-samples 0
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
- `--arch` (default: `mlp`): Network architecture (`mlp`, `cnn`, or
  `random_features`). For `cnn`, `--widths` controls conv channels and
  `--depths` controls the number of conv layers. `random_features` uses a
  frozen random feature map with a trainable policy/value head and requires
  `--depths 1`.

Environment
- `--task` (default: `gridworld`): Task choice (`gridworld` or `bandit`). The
  bandit task requires `--algo trpo` or `--algo dqn`, `--arch mlp` or
  `--arch random_features`, and `--max-steps 1`.
- `--grid-size` (default: `8`): Square grid side length. Observation uses a per-tile one-hot encoding (4 channels) and is flattened. Also has 2-frame stacking.
- `--obstacle-prob` (default: `0.2`): Bernoulli probability of a wall in each cell (except start/goal). Maps are regenerated per seed until solvable.
- `--max-steps` (default: `64`): Maximum steps per episode before truncation (applies to training, eval, and video rollouts).
- `--frame-stack` (default: `2`): Number of consecutive one-hot grid observations concatenated into the flattened observation.
- `--start` (default: unset): Start corner index (0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left). Unset means randomized.
- `--end` (default: unset): Goal corner index (0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left). Unset means randomized.
When one of `--start` or `--end` is unset, the other is sampled from the remaining corners; if both are unset, both corners are randomized (but always different).
The observation is a 2-frame stack: two consecutive one-hot grids are flattened and concatenated.
- `--sticky-action-prob` (default: `0`): Probability that the previous action is executed.
- `--slip-prob` (default: `0`): Probability that a uniformly random action is executed.
- `--reward-noise-std` (default: `0`): Standard deviation of zero-mean online reward noise.
- `--context-dim` (default: `16`): Observation dimension for the contextual bandit.
- `--bandit-actions` (default: `4`): Number of bandit actions.
- `--bandit-teacher-hidden` (default: `12`): Hidden width of the fixed nonlinear
  teacher used to generate the bandit task's optimal action.
- `--bandit-teacher-seed` (default: `17`): Seed for the bandit task teacher;
  this changes the environment instance, not the independent training runs.

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
- `--plot-only` (default: disabled): Skip training and regenerate only `curve.png` from the existing top-level `metrics.csv`.
- `--periodic-plot-only` (default: disabled): Skip training and regenerate each run's `periodic_eval.png` from existing `w*_d*_run*/periodic_eval.csv` files.
- `--min-return` / `--max-return` (default: unset): With `--plot-only` or `--periodic-plot-only`, rescale train/test returns to proportions using `(return - min_return) / (max_return - min_return)` before plotting.
- `--log-every` (default: `0`): Save `episodes.csv`/`episodes.png`, `trpo_updates.csv` (TRPO), and configured GIFs every N episodes by overwriting the current files; also appends train/test evals plus FIM trace to `periodic_eval.csv` and updates `periodic_eval.png`; set `0` to disable.
  Periodic eval rows include `episode`, `train_return`, `test_return`, `fim_trace`, and the run metadata columns.
- `--save-model` / `--no-save-model` (default: save model enabled): Enable or disable writing `.pt` model checkpoints.

Videos
- `--video-seeds` (default: unset): Comma/range seed list to render as GIFs (e.g., `1,2,10-12`). When unset, renders the first 5 training seeds and first 5 test seeds (or all if fewer). Use `--video-seeds none` to disable.
- `--video-fps` (default: `6`): Playback FPS for saved GIFs.

Episode curves
Per-run episode CSVs and plots are always saved inside each run's log directory.

FIM
- `--fim-samples` (default: `64`): Number of (state, action) samples used to estimate the Fisher trace; `0` disables FIM logging.

## Fixed-sweep analysis

Analyze raw per-run capacity metrics with the fixed rise-dip-rise rule:

```bash
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics testing/online_cnn_sweep_01/metrics.csv \
  --out-dir testing/online_cnn_sweep_01/analysis \
  --min-return -0.32 --max-return 0.959 \
  --fit-threshold 0.95 --practical-effect 0.10
```

For long-run checkpoints, use `scripts/analyze_episodic_dd.py` with a glob of
the raw `periodic_eval.csv` files. Both analyzers write aggregate CSV, a curve,
and JSON containing every tested candidate and whether it passed.

For online LSTD-Q bandit experiments, use:

```bash
uv run --no-sync python scripts/run_online_lstd_bandit.py \
  --task delayed_mdp --widths 2,4,8,16,32,64,128,256 \
  --runs 3 --episodes 1000 --gamma .9 \
  --log-dir testing/lstd_delayed_example
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics testing/lstd_delayed_example/metrics.csv \
  --out-dir testing/lstd_delayed_example/analysis \
  --min-return 0 --max-return 1 --fit-field train_optimal_action_rate \
  --fit-threshold .95 --practical-effect .10
```

The LSTD runner updates sufficient statistics after each live transition and
does not retain a frozen observation/reward dataset. `--fit-field` permits a
direct training optimal-action diagnostic while the curve remains based on
held-out return.

The confirmed online double-descent configuration uses sequential live
contexts, separate-action ReLU features, and five learner seeds:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync \
  python scripts/run_online_lstd_bandit.py --task bandit \
  --widths 4,8,12,16,24,32,48,64,96,128,192,256,384,512 \
  --runs 5 --base-seed 19500 --train-seeds 1001-1200 \
  --test-seeds 1201-1400 --context-dim 3 --bandit-actions 2 \
  --bandit-teacher-hidden 2 --bandit-teacher-seed 3 \
  --reward-noise-std .5 --episodes 1000 --train-sampling sequential \
  --eval-episodes 2 --epsilon-start 1 --epsilon-end 1 \
  --epsilon-decay 1 --gamma 0 --ridge 0 --solve-every 1000 \
  --feature-map relu --separate-action-features \
  --log-dir testing/lstd_bandit_relu_separate_teacher3_split2_confirm_01
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics testing/lstd_bandit_relu_separate_teacher3_split2_confirm_01/metrics.csv \
  --out-dir testing/lstd_bandit_relu_separate_teacher3_split2_confirm_01/analysis \
  --min-return -1 --max-return 1 \
  --fit-field train_optimal_action_rate --fit-threshold .95 \
  --practical-effect .10
```

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
