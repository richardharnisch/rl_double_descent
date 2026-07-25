# Saved online-RL search evidence

Each directory contains raw per-run `metrics.csv`, the repository collector's
`summary.csv` and curve, and the fresh analyzer's aggregate CSV, JSON, and
curve. The episodic directory also contains one raw checkpoint CSV per run.

All runs used live interaction with `uv run --no-sync python -m
rl_dd.experiment`; no frozen observation or trajectory dataset was used.

| Directory | Online configuration |
| --- | --- |
| `online_cnn_sweep_01` | CNN/TRPO, fixed corners, no transition noise, widths 2–32, 3 runs |
| `online_dqn_sweep_01` | CNN/DQN, 4x4 grid, widths 2–48, 3 runs |
| `online_cnn_sticky_01` | CNN/TRPO, sticky actions 0.2, widths 2–32, 3 runs |
| `online_cnn_sticky_01p1` | CNN/TRPO, sticky actions 0.1, widths 2–32, 3 runs |
| `online_cnn_sticky_01p1_confirm` | CNN/TRPO, sticky actions 0.1, widths 2–32, 5 runs, 50 held-out maps |
| `online_cnn_rewardnoise_01` | CNN/TRPO, online reward noise 0.1, widths 2–32, 3 runs |
| `online_cnn_rewardnoise_002` | CNN/TRPO, online reward noise 0.02, widths 2–32, 3 exploratory runs; analyzer pass did not reproduce |
| `online_cnn_rewardnoise_002_confirm` | CNN/TRPO, online reward noise 0.02, widths 2–32, 5 runs, 50 held-out maps; analyzer false |
| `bandit_capacity_01` / `bandit_capacity_02` / `bandit_capacity_03` | Live contextual bandit, TRPO/MLP, 3-run capacity sweeps; no analyzer pass |
| `bandit_teacher3_confirm` | Live contextual bandit, teacher seed 3, 5-run capacity confirmation; analyzer false |
| `bandit_random_features_01` / `bandit_random_features_short_01` / `bandit_random_features_short_02` | Live contextual bandit with frozen random features and online TRPO; no analyzer pass |
| `bandit_dqn_random_features_01` | Live contextual bandit with DQN and frozen random features; analyzer false |
| `bandit_episodic_01` | Live contextual bandit, 50,000 online episodes, 1,000-episode checkpoints, 3 runs; analyzer false |
| `lstd_bandit_01` / `lstd_bandit_02` / `lstd_bandit_clean_01` | Incremental online random-feature LSTD-Q on the live bandit; no analyzer pass |
| `lstd_delayed_01` / `lstd_delayed_02` / `lstd_delayed_highwidth_01` / `lstd_delayed_highwidth_02` | Incremental online LSTD-Q on a two-step delayed MDP; no analyzer pass |
| `lstd_delayed_states_01` / `lstd_delayed_states_highwidth_01` | Delayed online LSTD-Q with 100 live training contexts and widths through 1536; no analyzer pass |
| `lstd_delayed_contrast_full_01` | Delayed online LSTD-Q, 20 live training contexts, complete widths 2–1024, 3 runs; near candidate rejected because recovery was not persistent |
| `lstd_delayed_states_tail_01` | Fresh 100-context delayed online LSTD-Q tail at widths 1536–4096, 3 runs; analyzer false |
| `online_cnn_randomcorners_01` | CNN/TRPO, seeded random start/goal corners, widths 2–32, 3 runs |
| `online_cnn_depth_01` | CNN/TRPO, width 8, depths 1–5, 3 runs |
| `online_episodic_01` | CNN/TRPO width 16/depth 2, 20,000 episodes, 1,000-episode checkpoints, 3 runs |

To regenerate an aggregate from a saved capacity raw file:

```bash
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics evidence/online_cnn_sticky_01p1/metrics.csv \
  --out-dir /tmp/online_cnn_sticky_01p1-analysis \
  --min-return -0.32 --max-return 0.955104949 \
  --fit-threshold 0.95 --practical-effect 0.10
```

To regenerate the episodic analysis:

```bash
uv run --no-sync python scripts/analyze_episodic_dd.py \
  --periodic-glob 'evidence/online_episodic_01/periodic_eval_run*.csv' \
  --out-dir /tmp/online_episodic_01-analysis \
  --min-return -0.32 --max-return 0.9589949493661166 \
  --fit-threshold 0.95 --practical-effect 0.10
```

All confirmatory saved analyses report `passed: false` under the fixed
criterion. The exploratory `online_cnn_rewardnoise_002` analysis passed on its
first three runs, but its five-run confirmation did not reproduce that curve;
both artifacts are retained. The same rule was applied to the contextual
bandit and LSTD experiments, including `--fit-field train_optimal_action_rate`
where noisy rewards made direct return a poor interpolation diagnostic. The
interpretation and exact training commands are in `.cook/dd_report.md`.
