# Online RL Double-Descent Search Report (2026-07-26)

## Current conclusion

A genuine, reproducible return-based online RL double-descent curve has now been
found in a live contextual-bandit control task. The result is evidence-backed:
it survives five learner seeds, two independent train/test context splits, and
two teacher seeds. Every run below was trained from live episodes; no frozen
observation set or frozen trajectory labels were used.

The repository contains the reproducible environment controls, diagnostics,
analysis scripts, launch configuration, and tracked raw evidence for the
successful confirmations. Exploratory runs remain under the ignored
`testing/` work area.

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

## Demonstrated online double descent

The successful path uses the one-step live contextual bandit in
`src/rl_dd/bandit.py` and incremental LSTD-Q. Each episode samples one of 200
training contexts in a fixed sequential cycle, selects an action with uniform
online exploration (`epsilon=1`), and observes a fresh reward of `+1` or `-1`
plus transition noise with standard deviation `0.5`. The learner never stores
the context/reward stream. After 1,000 live transitions, it is evaluated on
the training contexts and on 200 held-out contexts, with two fresh reward
episodes per context.

The capacity variable is the per-action width of a fixed ReLU random-feature
map. Features are separate by action, so the counted parameter count is twice
the width. The complete width grid is
`4,8,12,16,24,32,48,64,96,128,192,256,384,512`; all widths receive the same
1,000 interactions, exploration policy, solver interval, and evaluation
budget. The principal configuration is context dimension 3, two actions,
teacher hidden size 2, teacher seed 3, zero ridge, `gamma=0`, and five learner
seeds.

The independent split-2 aggregate below is normalized from raw return using
`(return + 1) / 2`; the `ci95` column is the pooled five-run 95% normal
approximation reported by the analyzer.

| width | parameters | train fit | normalized test | ci95 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 0.745 | 0.787 | 0.082 |
| 8 | 16 | 0.896 | 0.916 | 0.036 |
| 12 | 24 | 0.937 | 0.951 | 0.028 |
| 16 | 32 | 0.964 | 0.971 | 0.023 |
| 24 | 48 | 0.974 | 0.973 | 0.019 |
| 32 | 64 | 0.979 | 0.972 | 0.027 |
| 48 | 96 | 0.988 | 0.968 | 0.023 |
| 64 | 128 | 0.990 | 0.970 | 0.030 |
| 96 | 192 | 0.996 | 0.970 | 0.028 |
| 128 | 256 | 0.998 | 0.951 | 0.024 |
| 192 | 384 | 0.997 | 0.865 | 0.036 |
| 256 | 512 | 0.981 | 0.662 | 0.069 |
| 384 | 768 | 0.990 | 0.852 | 0.050 |
| 512 | 1024 | 0.995 | 0.889 | 0.038 |

The fixed analyzer identifies width 8 -> 256 -> 384 as a passing candidate:
the normalized rise is `0.129`, the drop is `0.254`, and the recovery gain is
`0.190`; training fit at the dip is `0.981`. The corresponding split-1
confirmation passes with rise `0.188`, drop `0.212`, and recovery gain
`0.153`. Teacher seed 1 independently passes with rise `0.155`, drop `0.237`,
and recovery gain `0.181`. In every case the changes exceed pooled 95%
uncertainty and the recovery remains above the dip through the final width.

This is not explained by a failed learner: training fit is approximately one
through the dip and high-width tail. It is not a budget or parameter-count
artifact: all capacities use the same online transition count and the raw
metrics record `num_params = 2 * width`. It is not evaluation noise: each
capacity has five learner seeds, two context splits are used, and the saved
per-run rows include test-return standard deviations and action-fit values.
The successful raw confirmations are
`evidence/online_lstd_relu_separate_teacher3_confirm_01/` and
`evidence/online_lstd_relu_separate_teacher3_split2_confirm_01/`; the
independent teacher-1 confirmation is preserved at
`evidence/online_lstd_relu_separate_teacher1_confirm_01/`.

The implementation also retains negative feature-basis controls: shared-action
tanh and random-Fourier features, plus the smooth analytical bandit teacher,
did not pass the same gates. This makes the reported result specifically a
capacity effect of the separate-action ReLU LSTD representation, not an
automatic consequence of adding a wider matrix or changing the reward task.

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

## Contextual-bandit and online-TD results

The repository now also contains a live contextual-bandit task and an
incremental random-feature LSTD-Q runner. Bandit contexts are generated from
episode seeds, while reward noise comes from a persistent transition RNG. The
LSTD sufficient statistics are updated after each live transition; no frozen
observation/reward table is constructed.

Earlier contextual-bandit paths were negative:

- TRPO/MLP capacity sweeps with 10, 20, and 50 live training contexts reached
  near-perfect training action rates, but held-out return either plateaued or
  showed small ordinary capacity fluctuations. The largest three-run change
  was below the fixed practical-effect gate.
- A systematic teacher-seed scan over seeds 0–4 produced one single-run
  candidate at teacher seed 3 (widths `6 -> 48 -> 64`). Its five-run
  confirmation over the complete width grid and 500 held-out contexts failed:
  the drop was not uncertainty-supported and the high-width tail did not
  recover.
- Frozen-random-feature TRPO and DQN sweeps, plus a 50,000-episode checkpoint
  sweep, all failed the fixed analyzer. The random-feature short-budget lead
  (`24 -> 48 -> 96`) disappeared when the online budget changed from 200 to
  400 episodes.

The successful path uses a separate ReLU feature block for each action. This
action-wise representation is important: it makes the counted parameter
dimension and per-action live sample count align, while leaving exploration,
reward generation, and evaluation unchanged. The complete five-run teacher-3
confirmation and its independent context split both pass the fixed analyzer;
teacher 1 passes the same complete grid as an additional teacher-seed check.

The online LSTD-Q paths were closer to the mechanism suggested by the
parameter/visited-state ratio, but still negative under return-based
acceptance. With 200 live bandit transitions, three-run test return peaked at
feature width 48, dipped at 128, and partially recovered at 192; the drop was
about `0.059` normalized and recovery about `0.041`, both below the practical
effect gate. A two-step delayed-reward MDP with `gamma=0.9` reached training
action-rate interpolation at widths 128–256, but its held-out return declined
through width 256. Extending the delayed-MDP feature width through 1024 did
not produce a persistent recovery. Increasing the live training context count
from 20 to 100 shifted the interpolation point to widths 192–512, with test
return peaking near 256 and declining through the 768–1536 tail. These are
TD-estimation failures or ordinary overfitting, not demonstrated double
descent.

The live LSTD implementation was then tightened for a fair capacity
comparison: random-feature projections are nested across widths for each
learner seed, so a wider estimator contains the exact narrower feature map.
The resulting five-run few-context confirmation covered widths 2 through
1536. Its strongest aggregate interpolation dip was normalized test return
`0.389` at width 64 to `0.162` at width 256, followed by `0.369` at width
1536. The analyzer still returned `passed: false`: the pre-peak rise was too
small and the supported tail recovery started too late to satisfy the full
criterion. This is a reproducible near miss, not evidence selected from the
earlier non-nested exploratory runs.

A second five-run confirmation used teacher seed 2, zero ridge, the same five
live training contexts, and a solve interval of 1,000 transitions. It produced
a sharper dip, with normalized test return `0.682` at width 64, `0.270` at
width 192, and `0.311` at width 1536. The complete analyzer remained false:
the recovery was not persistent and the qualifying adjacent rise was absent.
Teacher seed 0 gave a similar exploratory one-run rise and dip, but its exact
five-run confirmation had means `0.514` at width 48, `0.324` at width 192,
and `0.289` at width 1536, again with no passing candidate. A deterministic
reward control changed the fluctuations but did not establish a persistent
return recovery.

The new continuous-payoff contextual bandit was also tested as a live online
control. A five-teacher, three-learner-seed family confirmation used nested
random features, five live training contexts, and widths 16–512. The pooled
test means ranged from `0.299` to `0.383` without a fixed-criterion pass. This
rules out interpreting the delayed-MDP result as an artifact of one unusual
reward scale or action encoding.

A complete 20-context contrast sweep (widths 2–1024, three runs) produced the
closest delayed-MDP candidate: training action rate reached `0.967` at width
96, normalized held-out return rose to `0.670`, fell to `0.488` at width 512,
and rebounded to `0.593` at width 768. The fresh analyzer still returned
`passed: false`: the recovery did not remain above the dip through width 1024,
and only the width-96 rise was uncertainty-supported. This is retained as a
near miss, not as evidence of double descent.

The independent 100-context high-width tail then tested widths `1536, 2048,
3072, 4096` with three fresh runs per width. Training action-rate fits stayed
between `0.987` and `1.000`, while normalized held-out return means were
`0.721, 0.721, 0.710, 0.668`. The fresh tail analyzer returned `passed: false`;
the putative rebound did not persist into the kernel-limit tail.

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
- `testing/bandit_capacity_01/`
- `testing/bandit_capacity_02/`
- `testing/bandit_capacity_03/`
- `testing/bandit_teacher3_confirm/`
- `testing/bandit_random_features_01/`
- `testing/bandit_random_features_short_01/`
- `testing/bandit_random_features_short_02/`
- `testing/bandit_dqn_random_features_01/`
- `testing/bandit_episodic_01/`
- `testing/lstd_bandit_01/`
- `testing/lstd_bandit_02/`
- `testing/lstd_bandit_clean_01/`
- `testing/lstd_delayed_01/`
- `testing/lstd_delayed_02/`
- `testing/lstd_delayed_highwidth_01/`
- `testing/lstd_delayed_highwidth_02/`
- `testing/lstd_delayed_states_01/`
- `testing/lstd_delayed_states_highwidth_01/`
- `testing/lstd_delayed_contrast_full_01/`
- `testing/lstd_delayed_states_tail_01/`
- `testing/lstd_delayed_fewstates_nested_ridge0_full_confirm_01/`
- `testing/lstd_delayed_teacher2_nested_solve1000_confirm_01/`
- `testing/lstd_delayed_teacher0_nested_solve1000_confirm_01/`
- `testing/continuous_bandit_family_confirm_01/`
- `evidence/online_lstd_relu_separate_teacher3_confirm_01/`
- `evidence/online_lstd_relu_separate_teacher3_split2_confirm_01/`

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

The online LSTD runner is reproducible with:

```bash
uv run --no-sync python scripts/run_online_lstd_bandit.py \
  --task delayed_mdp --widths 2,3,4,5,6,8,10,12,16,24,32,48,64,96,128,192,256 \
  --runs 3 --base-seed 4800 --train-seeds 1-20 --test-seeds 21-220 \
  --episodes 200 --eval-episodes 20 --gamma .9 --ridge .0001 \
  --reward-noise-std .1 --log-dir testing/lstd_delayed_01
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics testing/lstd_delayed_01/metrics.csv \
  --out-dir testing/lstd_delayed_01/analysis --min-return 0 --max-return 1 \
  --fit-field train_optimal_action_rate --fit-threshold .95 \
  --practical-effect .10
```

The strongest fair-capacity confirmation can be regenerated with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync \
  python scripts/run_online_lstd_bandit.py --task delayed_mdp \
  --widths 8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536 \
  --runs 5 --base-seed 15000 --train-seeds 1-5 --test-seeds 6-220 \
  --context-dim 4 --bandit-actions 4 --bandit-teacher-hidden 2 \
  --bandit-teacher-seed 2 --reward-noise-std .1 --episodes 1000 \
  --eval-episodes 20 --epsilon-start 1 --epsilon-end .1 \
  --epsilon-decay 500 --gamma .9 --ridge 0 --solve-every 1000 \
  --log-dir testing/lstd_delayed_teacher2_nested_solve1000_confirm_01
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics testing/lstd_delayed_teacher2_nested_solve1000_confirm_01/metrics.csv \
  --out-dir testing/lstd_delayed_teacher2_nested_solve1000_confirm_01/analysis \
  --min-return 0 --max-return 1 --fit-field train_optimal_action_rate \
  --fit-threshold .95 --practical-effect .10
```

The successful split-2 run is reproducible with:

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
```

To regenerate the reported aggregate and curve directly from the saved raw
per-run rows:

```bash
uv run --no-sync python scripts/analyze_online_dd.py \
  --metrics evidence/online_lstd_relu_separate_teacher3_split2_confirm_01/metrics.csv \
  --out-dir /tmp/online_lstd_relu_separate_teacher3_split2_analysis \
  --min-return -1 --max-return 1 \
  --fit-field train_optimal_action_rate --fit-threshold .95 \
  --practical-effect .10
```

## Next staged experiment if more compute becomes available

The highest-value next gridworld follow-up is a GPU-confirmatory sweep around
the sticky-action near-effect region (`p=0.1`, widths 6–16), with the full
width grid retained, five or more runs per capacity, at least 50 fixed test
maps, and per-map raw returns. It should also add a second independently
generated train/test split. The `reward_noise_std=0.02` result shows why this
replication is necessary: a three-run pass disappeared under that same
five-run protocol.

If the goal is to search a different mechanism rather than scale this
gridworld study, the CPU-feasible delayed-MDP path has now been exercised from
20 to 100 live contexts and through feature width 4096. A worthwhile next
implementation would add exact per-state Bellman/value-error diagnostics and
compare them with policy return, because the published LSTD mechanism can show
an estimator double descent without a corresponding greedy-policy recovery.
Contexts, transitions, and rewards must continue to be generated by
interaction, with no saved observation/label table. The delayed-MDP path should
remain classified as a negative control; the separate-action ReLU
contextual-bandit path above is the confirmed return-based demonstration.

# Why Earlier Paths Did Not Find Double Descent — and Where to Look Next

## 1. Background

This project searched for the double descent (DD) phenomenon in deep RL by training DQN and TRPO agents on a family of seeded 8x8 gridworld mazes and evaluating on held-out seeds. DD was probed in two regimes:

- **Capacity regime**: sweep MLP widths/depths, look for a non-monotonic test-return curve as parameter count grows.
- **Episodic regime**: hold capacity fixed, look for a non-monotonic test-return curve as training episodes grow.

Across the earlier gridworld and episodic configurations (train-set sizes from
10 to 1000 maps, episode budgets from 10k to ~5M, widths 4–1024, depths 2–3,
both DQN and TRPO), no second descent of test return was observed. A
generalization gap reliably appeared but never closed. The separate-action
ReLU contextual-bandit path subsequently supplied the confirmed demonstration
reported above; these earlier failures remain useful negative controls.

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
