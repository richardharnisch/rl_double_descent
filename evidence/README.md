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
| `online_cnn_rewardnoise_01` | CNN/TRPO, online reward noise 0.1, widths 2–32, 3 runs |
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

All saved analyses report `passed: false` under the fixed criterion. The
interpretation and exact training commands are in `.cook/dd_report.md`.
