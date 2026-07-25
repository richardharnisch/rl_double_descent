# Online RL Double-Descent Search Report (2026-07-25)

## Current conclusion

No genuine online RL double-descent curve has been found in the completed local
search. The negative result is evidence-backed, not a claim that double descent
cannot occur in RL. Every completed avenue below was trained from live episodes;
no frozen observation set or frozen trajectory labels were used.

The repository now contains the reproducible environment controls, diagnostics,
analysis scripts, launch configuration, and raw evidence under the ignored
`testing/` work area. The tracked code is sufficient to regenerate every run.

## Acceptance criterion fixed before curve selection

For each capacity or checkpoint sweep, returns were normalized as
`(return - min_return) / (max_return - min_return)`, using a task-specific
no-progress floor and an oracle achievable-return estimate. A candidate had to
show all of the following over the complete prespecified sweep:

1. mean training fit at the dip at least `0.95`;
2. an earlier rise, peak-to-dip drop, and dip-to-recovery gain each at least
   `0.10` normalized return;
3. each change larger than its pooled 95% normal-approximation uncertainty;
4. recovery persistent through the remaining capacities/checkpoints.

The candidate search was exhaustive over each fixed grid. No seed or curve
segment was selected after inspection. The authoritative implementation is
`scripts/analyze_online_dd.py`; episodic checkpoints use
`scripts/analyze_episodic_dd.py`.

## Main online results

The first capacity path used CNN/TRPO, grid size 8, obstacle probability 0.1,
fixed start/goal corners 0 and 2, 2,000 episodes per run, depth 2, widths
`2,3,4,5,6,8,10,12,16,24,32`, three runs per width, and ten evaluation
episodes per test map. Its normalized test means were approximately
`0.081, 0.118, 0.156, 0.151, 0.253, 0.285, 0.180, 0.153, 0.227, 0.254,
0.187`; the analysis result was `passed: false`. Training fit was near one from
width 3 onward, but the apparent oscillations were not uncertainty-supported.

Online stochasticity did not produce a valid curve:

- Sticky actions `p=0.2`: test means ranged from about `0.198` to `0.294`
  normalized, while training fit was below threshold at several widths.
- Sticky actions `p=0.1`: the most suggestive segment was widths `8,10,12`,
  with normalized test means about `0.263, 0.168, 0.329`; the preceding rise and
  drop were below the fixed practical-effect rule. The result was false.
- A confirmatory p=0.1 sweep used five runs per width and 50 held-out maps.
  Normalized test means were approximately `0.280, 0.233, 0.321, 0.253,
  0.240, 0.231, 0.246, 0.305, 0.226, 0.283, 0.268` for widths 2 through 32.
  Several widths interpolated training maps, but the full analyzer still
  returned `passed: false`; the exploratory fluctuation did not survive.
- Zero-mean online reward noise with standard deviation `0.1`: training fit
  remained unstable, with aggregate normalized means roughly `0.54` to `0.88`
  across the sweep. It failed the interpolation prerequisite.
- Intermediate zero-mean online reward noise with standard deviation `0.05`
  also failed the interpolation prerequisite: three-run training-fit means
  ranged from about `0.60` to `0.94`.
- The smaller-noise exploratory sweep at standard deviation `0.02` produced a
  tempting three-run pass at widths `6 -> 8 -> 12`: normalized test means were
  approximately `0.466, 0.313, 0.468`, with a training fit of `0.999` at the
  dip. This was deliberately followed by a five-run confirmation over the
  complete width grid and 50 held-out maps. The confirmation returned
  `passed: false`: normalized test means were approximately `0.368, 0.399,
  0.431, 0.472, 0.435, 0.481, 0.458, 0.466, 0.440, 0.465, 0.471`, and
  training fit at width 8 was only `0.923`. The first pass is therefore
  classified as seed-set-specific variance, not demonstrated double descent.

Other online capacity paths were also negative:

- DQN/CNN on a 4x4 grid, widths `2,3,4,6,8,12,16,24,32,48`, three runs:
  training means ranged `0.45`–`0.98`; the curve was learning-instability,
  not double descent.
- CNN depth `1`–`5` at width 8, three runs: depths 1–4 interpolated training
  maps, but test-return means remained uncertainty-dominated; depth 5 fell to
  about `0.80` normalized training fit.
- Random seeded start/goal corners: training fit reached approximately one,
  but held-out return stayed at the no-progress floor across widths. This is a
  distribution-generalization failure, not a recovery.

The episodic path used fixed CNN width 16/depth 2, 20,000 episodes, checkpoints
every 1,000 episodes, three independent runs, ten test episodes per map, and
the same fixed maps. One run visually contained a rise near 2,000, a dip near
5,000, and a late increase, but the other runs disagreed. The aggregate
episodic analyzer found no candidate passing the effect and uncertainty gates.
Training fit stayed at approximately `0.959` throughout, so failed learning
was not the explanation; seed-to-seed checkpoint variability was.

## Diagnostics and artifact checks

Final run rows record parameter count, train/test return, return standard
deviation, action entropy, visitation coverage, and optional FIM trace. In
stochastic evaluation, the seeded map is held fixed while transition RNG
advances between repeated episodes. The environment tests cover seeded reward
noise, sticky-action reset semantics, and map-preserving resets.

The raw per-run files and generated analyses are in:

- `testing/online_cnn_sweep_01/`
- `testing/online_dqn_sweep_01/`
- `testing/online_cnn_sticky_01/`
- `testing/online_cnn_sticky_01p1/`
- `testing/online_cnn_sticky_01p1_confirm/`
- `testing/online_cnn_rewardnoise_01/`
- `testing/online_cnn_rewardnoise_002/`
- `testing/online_cnn_rewardnoise_002_confirm/`
- `testing/online_cnn_randomcorners_01/`
- `testing/online_cnn_depth_01/`
- `testing/online_episodic_01/`

Each completed capacity directory contains raw `metrics.csv`, aggregate CSV,
the curve, and `analysis.json`. The episodic directory contains one raw
`periodic_eval.csv` per run plus its aggregate analysis.

## Reproduction

Dependencies are managed with uv. Basic verification is:

```bash
uv run --no-sync python -m unittest discover -v
uv run --no-sync python -m compileall -q src tests scripts
```

A representative online capacity run is:

```bash
uv run --no-sync python -m rl_dd.experiment \
  --algo trpo --arch cnn --grid-size 8 \
  --widths 2,3,4,5,6,8,10,12,16,24,32 --depths 2 --runs 3 \
  --base-seed 1000 --train-seeds 1-5 --test-seeds 6-15 \
  --episodes 2000 --max-steps 32 --obstacle-prob 0.1 \
  --start 0 --end 2 --trpo-batch-episodes 20 \
  --early-stop-episodes 0 --eval-episodes 10 --fim-samples 0 \
  --video-seeds none --no-save-model --cpu \
  --log-dir testing/online_cnn_sweep_01
uv run --no-sync python -m rl_dd.experiment \
  --collect-only --log-dir testing/online_cnn_sweep_01
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics testing/online_cnn_sweep_01/metrics.csv \
  --out-dir testing/online_cnn_sweep_01/analysis \
  --min-return -0.32 --max-return 0.959 \
  --fit-threshold 0.95 --practical-effect 0.10
```

The cluster launcher is `scripts/run_experiment.slurm`; it uses `uv run
--no-sync` in array workers and exposes the sticky-action, slip, and reward-noise
controls. The launch command is ready for a larger GPU sweep, but no claim of
success is made from that unrun configuration.

## Next staged experiment if more compute becomes available

The highest-value next gridworld follow-up is a GPU-confirmatory sweep around
the sticky-action near-effect region (`p=0.1`, widths 6–16), with the full
width grid retained, five or more runs per capacity, at least 50 fixed test
maps, and per-map raw returns. It should also add a second independently
generated train/test split. The `reward_noise_std=0.02` result shows why this
replication is necessary: a three-run pass disappeared under that same
five-run protocol.

If the goal is to search a different mechanism rather than scale this
gridworld study, the next CPU-feasible implementation should be a native
online contextual-bandit or short-horizon stochastic MDP. Contexts and rewards
must be generated by environment interaction on every episode, with no saved
observation/label table; the existing analyzer and acceptance gates can then
be reused. This extension is not yet implemented, so it is a plan rather than
evidence. Until one of these staged experiments is run and passes the fixed
criterion, the scientifically correct conclusion remains “no genuine double
descent demonstrated.”

# Why This Project Did Not Find Double Descent — and Where to Look Next

## 1. Background

This project searched for the double descent (DD) phenomenon in deep RL by training DQN and TRPO agents on a family of seeded 8x8 gridworld mazes and evaluating on held-out seeds. DD was probed in two regimes:

- **Capacity regime**: sweep MLP widths/depths, look for a non-monotonic test-return curve as parameter count grows.
- **Episodic regime**: hold capacity fixed, look for a non-monotonic test-return curve as training episodes grow.

Across all tested configurations (train-set sizes from 10 to 1000 maps, episode budgets from 10k to ~5M, widths 4–1024, depths 2–3, both DQN and TRPO), **no second descent of test return was observed**. A generalization gap reliably appeared but never closed. In the episodic regime there was a slight downward trend in test return after the initial rise, which is consistent with mild overfitting, but the reversal that would constitute DD never appeared within the available training horizon.

## 2. Why DD Did Not Appear

The negative result is not a single failure; it is the conjunction of several conditions that together prevented the experimental setup from entering the regime where DD is expected.

### 2.1 No clear analogue of the interpolation threshold

DD in supervised learning is anchored to a model crossing the threshold where it can perfectly fit the training set. In this project:

- **DQN never fit anything.** It exhibited frequent catastrophic forgetting and never reliably solved the training maps at any capacity, so its capacity sweep was uninformative for DD.
- **TRPO usually saturated below the maximum train return.** Even at width 1024 with millions of episodes, train return on small training sets remained slightly below the achievable optimum. The models therefore stayed in a "partial-fitting" regime and never traversed the qualitatively different overparameterized regime where DD is predicted.
- **With large training sets (1000 maps), the train/test gap collapsed entirely** — both stabilized around 0.4 of max return. Without a generalization gap there is no overfitting peak for DD to descend from. This appears to be an implicit curriculum effect (more maps regularize), not DD.

So the experiment's hyperparameter region was bracketed by two failure modes: too few maps + insufficient capacity → cannot interpolate; many maps → no gap to begin with. The narrow band where interpolation could plausibly occur was never reached.

### 2.2 Non-stationary, policy-induced data distribution

Unlike SL, the training distribution in RL is generated by the policy itself and shifts as the policy improves. Two consequences:

- The "interpolation threshold" — if it exists — is a moving target rather than a fixed point of the loss landscape, plausibly smearing any sharp variance peak across a range of capacities.
- Capacity gains may be absorbed by changes in **what data is collected** (exploration / visitation) rather than **how well a fixed dataset is fit**. The capacity-vs-generalization relationship is then dominated by visitation effects rather than interpolation effects, blunting DD.

### 2.3 Architecture mismatch

The agents use an MLP over a flattened one-hot grid. For an 8x8 spatial-navigation task, a fully connected network is a poor inductive prior: spatial structure must be re-learned from scratch at every parameter setting. This pushes the parameter count needed to fully fit the train set well above what the cluster's compute budget could supply, which is consistent with the train-return saturation observed for TRPO.

### 2.4 Insufficient stochasticity / irreducible noise

DD in SL is most pronounced when the data has irreducible label noise. The gridworld here is fully observable, deterministic, and Markovian — there is an optimal deterministic policy. With no irreducible noise, the variance term that makes the SL DD curve dip and then rise is muted, weakening the conditions in which DD typically appears.

### 2.5 The Fisher information signal was confounded

The project also tracked the Fisher information matrix (FIM) trace, hoping to catch a sensitivity peak coincident with overfitting. A spike does appear when the train/test gap opens — but the FIM trace then declines as capacity grows further, despite test return staying flat. The most likely explanation is that this metric is dominated by the policy becoming **more deterministic** and visiting **fewer states** at higher capacity, not by any change in generalization. As a probe of DD it therefore produces ambiguous signals rather than evidence.

Future runs should keep the FIM trace but add FIM effective rank, policy entropy, and state-visitation coverage as standard diagnostics. Effective rank would distinguish "one or two extremely sensitive directions" from "many moderately sensitive directions"; entropy and visitation coverage would test whether the apparent sensitivity drop is really policy determinization and reduced exploration.

### 2.6 Compute ceiling

The HPC budget capped runs at ~1.6M–5M episodes for the largest models and forced a heuristic, non-grid hyperparameter search. A clean DD curve in SL (e.g. Nakkiran et al.) typically requires a dense 2D sweep over capacity x dataset-size at fixed compute; reproducing that in RL needs an order of magnitude more compute than was available here.

### 2.7 Summary

DD requires (a) a model that actually crosses an interpolation-like threshold on the training distribution, (b) a meaningful generalization gap on the other side of that threshold, and (c) enough stochasticity for a variance peak to form. The setup either failed (a) (TRPO with few maps, DQN at any setting) or eliminated (b) (TRPO with many maps), and never had much of (c). The episodic runs show a weak overfitting-like decline in test return, but not the later recovery. The negative result is therefore consistent with DD being real in RL but invisible from inside this particular experimental envelope.

## 3. Three Avenues for Follow-Up Experiments

Each avenue targets a specific failure mode above.

### 3.1 Replace the MLP with a CNN to reach the interpolation threshold

**Targets**: §2.1 (no interpolation threshold) and §2.3 (architecture mismatch).

A small CNN over the 8x8x8 input (4 object channels x 2 frame-stack) has a vastly stronger spatial inductive bias than an MLP and should reliably memorize 50–100 training maps with a tractable parameter budget. Once the train-return curve reliably reaches max return, the capacity sweep can be re-run *anchored to this interpolation point*: under-parameterized → critically-parameterized → over-parameterized in the CNN's filter-count axis. This is the regime where DD is supposed to show up, and the current MLP setup likely never reaches it.

Concrete experiment:
- Architecture: 2–3 conv layers, sweep base channel count e.g. {4, 8, 16, 32, 64, 128, 256}, fix kernel size, and scale all conv layers by the same width multiplier so total parameter count is monotone in the sweep variable.
- Train sets: 50 and 100 maps, deterministic environment.
- Verify train return hits ~1.0 across the upper half of the sweep, then look for a non-monotonic test return.

### 3.2 Off-policy fitting on a frozen trajectory dataset

**Targets**: §2.2 (non-stationarity) and §2.5 (visitation-confounded metrics).

The clearest way to isolate the interpolation effect from the visitation effect is to remove the latter entirely. Collect a fixed dataset of trajectories from a strong reference policy on N training maps, then train a value-based offline-RL learner (e.g. CQL or BCQ) at varying capacities on this frozen dataset, and evaluate online on held-out maps. Behavior cloning can still be useful as a sanity baseline, but by itself it is imitation rather than offline RL and is less direct for the "fit a fixed return-relevant dataset" question.

This recreates the SL contract — fixed data distribution, error decomposes cleanly into bias/variance — while keeping the RL evaluation criterion (return on unseen MDPs). It is also the cleanest setting in which to track FIM trace and effective rank, since the data distribution no longer drifts. If DD exists in RL function approximation, this is the setup most likely to expose it; if it doesn't show up here either, that is itself a strong negative result.

The thesis acknowledges this is "less novel" than the on-policy case, but it is also where the methodology is most defensible.

### 3.3 Inject irreducible stochasticity (sticky actions, slippery transitions, or POMDP)

**Targets**: §2.4 (no irreducible noise).

DD is amplified by label noise in SL. The RL analogue is environment stochasticity that the optimal policy cannot eliminate — the residual return variance plays the role of irreducible noise. Three drop-in modifications, in order of implementation cost:

1. **Sticky actions**: with probability p the previous action is repeated regardless of the agent's choice (the ALE-style noise that has been used to break determinism in Atari benchmarks).
2. **Slippery transitions**: with probability p the agent moves to a direction adjacent to the chosen one (frozen-lake style).
3. **Partial observability**: replace the full grid observation with a local k x k window around the agent, forcing the policy to integrate history.

Run the existing capacity sweep at noise levels p ∈ {0, 0.1, 0.2, 0.3} on a mid-sized training set. Recompute the min/max return normalization separately for each noise level, because sticky or slippery dynamics lower the achievable return ceiling and would otherwise contaminate cross-noise comparisons. The prediction is that the test-return curve transitions from monotonic (p=0) to a DD shape as p increases, mirroring the noise-controlled DD plots in Nakkiran et al. Even a partial confirmation would establish that the absence of DD in the deterministic gridworld was a noise issue rather than an RL issue, sharpening the negative result of this thesis into a positive characterization of when DD does and does not appear in RL.

## 4. Recommended Priority

If only one experiment is run, **3.1 (CNN)** is the cheapest way to reach a regime where DD is even possible in the existing online setup. It especially unlocks a cleaner version of 3.3, where the same capable architecture can be swept across both capacity and stochasticity.

If two are run, pair **3.1 with 3.3 (noise)**: same CNN architecture, swept across both capacity and stochasticity, recovering the 2D plot shape from Nakkiran et al.

**3.2 (offline RL)** stands on its own and is the highest-value experiment for establishing whether the negative result generalizes beyond the on-policy setting, but it requires the most engineering investment (data collection, offline-RL implementation, eval harness).
