# M23A — Results and Discussion Brief

Last synthesized: 2026-09-10

Result authority: [completed campaign manifest](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M23A/final_last1m_alltasks_ep50_evalSeed20260909_controlled-goal-replay-v1/campaign_manifest.json)
Design/provenance authority: [M23A README](README.md) and [M23A Study](../../../../experiments/M23A_puzzle_direct_rollout_audit/study.yaml)

## Purpose of this brief

This is a discussion-ready interpretation of the completed M23A diagnostic. It
is not a second experimental specification: the linked README and Study remain
authoritative for design and provenance. The goal here is to give a reviewer or
another model enough verified context to assess the result, propose competing
mechanisms, and identify follow-up experiments without confusing inference with
training, or diagnostic evidence with causal proof.

## Executive finding

The twelve `final@1M` GCIQL checkpoints learned useful low-distance Puzzle
pressing routines, but did **not** learn robust long-horizon composition.
Mixer-L2 alpha=0.4 is the strongest of the three available checkpoint families
on 4x4, 4x5, and 4x6 under controlled replay, whereas Flat alpha=1.0 is almost
perfect on 3x3 and Mixer-L2 alpha=1.0 is substantially worse there. Thus the
result supports a task-dependent architecture--alpha interaction, not the
simple claim that Mixer or alpha=0.4 is universally better.

The most diagnostic pattern is complexity-conditioned failure:

- On 4x5, only task 1 (`D*=4`) is ever solved; every policy has 0/50 success
  on task IDs 2--5 (`D*=10,14,16,20`).
- On 4x6, task 1 (`D*=6`) is sometimes solved, task 2 (`D*=8`) is nearly never
  solved, and task IDs 3--5 (`D*=12,16,24`) are never solved.
- High-dimensional failures include repeated source-button presses and late
  `D*` regression. Alpha=0.4 reduces this waste substantially but does not
  produce long-sequence competence.

## What was evaluated

M23A is a post-hoc, observation-only rollout diagnosis. It trained no policy
and did not modify PuzzleEnv or GCIQL/Mixer semantics.

For each of four canonical Puzzle environments, it compared three existing
final checkpoints:

| Policy label | Source study/config family | Alpha | Source status |
| --- | --- | ---: | --- |
| Flat GCIQL | M16B Flat | 1.0 | `verified_clean` |
| Puzzle Mixer-L2 GCIQL | M16B S002 | 1.0 | `verified_clean` |
| Puzzle Mixer-L2 GCIQL | M16D S002 | 0.4 | `scoped_exception` |

All sources use training seed 0 and `last` / `final@1M`. The Flat-versus-Mixer
alpha=1.0 contrast is an M16B contrast. The alpha=0.4 checkpoints come from
M16D attempt 001 and retain the narrow provenance qualification:

```text
provenance_status = scoped_exception
reason = concurrent M23A-F0 diagnostic development
evidence_level = user-attested / partially machine-verified
```

This exception is restricted to the listed M16D source commit, configuration,
environment, seed, attempt and checkpoint hash. The historical dirty diff was
not reconstructed; alpha=0.4 results must not be described as fully verified
clean.

### Exact evaluated source mapping

The campaign manifest contains the full SHA-256 values; the abbreviated values
below are identifiers for human inspection, not substitutes for provenance.

| Environment | M23A policy | Source config | Attempt | Checkpoint SHA-256 prefix | Provenance |
| --- | --- | --- | ---: | --- | --- |
| 3x3 | Flat alpha=1.0 | `M16B-3x3-B000` | 0 | `14416ab7f328` | verified clean |
| 3x3 | Mixer-L2 alpha=1.0 | `M16B-3x3-S002` | 0 | `f62ec6c72a28` | verified clean |
| 3x3 | Mixer-L2 alpha=0.4 | `M16D-P3X3-S002-A04` | 1 | `3b450b99dd36` | scoped exception |
| 4x4 | Flat alpha=1.0 | `M16B-4x4-B000` | 0 | `76d2dba3c3b0` | verified clean |
| 4x4 | Mixer-L2 alpha=1.0 | `M16B-4x4-S002` | 0 | `2f4f408466fa` | verified clean |
| 4x4 | Mixer-L2 alpha=0.4 | `M16D-P4X4-S002-A04` | 1 | `fb7bda382644` | scoped exception |
| 4x5 | Flat alpha=1.0 | `M16B-4x5-B000` | 0 | `a41195316c33` | verified clean |
| 4x5 | Mixer-L2 alpha=1.0 | `M16B-4x5-S002` | 0 | `7164cdc024a3` | verified clean |
| 4x5 | Mixer-L2 alpha=0.4 | `M16D-P4X5-S002-A04` | 1 | `eff669720d7b` | scoped exception |
| 4x6 | Flat alpha=1.0 | `M16B-4x6-B000` | 0 | `22298b7b9563` | verified clean |
| 4x6 | Mixer-L2 alpha=1.0 | `M16B-4x6-S002` | 0 | `fed0585d3770` | verified clean |
| 4x6 | Mixer-L2 alpha=0.4 | `M16D-P4X6-S002-A04` | 1 | `9535e89c793e` | scoped exception |

## Evaluation protocol and validity gates

Each of the twelve cells was evaluated on all five canonical task IDs, with 50
episodes per task: 250 episodes/cell and 3,000 rollouts total. The protocol is:

- `evaluation_seed = 20260909`, common task/episode seed scheme;
- deterministic policy evaluation: `eval_temperature = 0`,
  `eval_gaussian = None`;
- final checkpoint step 1,000,000 only; no per-policy best-checkpoint choice;
- complete policy-facing goal observation replayed identically across the three
  policies for a fixed `(environment, task_id, episode_index)`;
- the board goal and initial observation are also hard paired invariants.

The completed [pairing gate](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M23A/final_last1m_alltasks_ep50_evalSeed20260909_controlled-goal-replay-v1/pairing_invariants.json)
reports 1,000 paired groups, each with exactly three members, and `passed`.
Independent aggregation of the episode tables also found zero mismatches in
goal fingerprint, board-goal fingerprint, initial-observation fingerprint, or
episode/actor/noise seed within any group.

All twelve per-cell manifests record 250 episodes, clean diagnostic code commit
`21e0cb838c1476ac6ec62f63e6f1152702afcde9`, and execution on `cuda:0`.
The campaign was manually run from the clean
`/home/eai/Research/RLC-M23A` feature worktree on branch
`m23a-direct-rollout-audit` under the user's explicit non-training diagnostic
exception; this is execution provenance, not a modification of any source run.
Across the campaign:

- event-effect consistency failures: 0;
- malformed board observations: 0;
- unreachable GF(2) residuals: 0;
- multi-source press events: 9 (rare descriptive events, not detector errors).

Therefore differences between policies are not attributable to unmatched goal
inputs or an observed board-dynamics inconsistency. Once policies take
different actions their later trajectories may, as intended, diverge.

## Main completion results

Every percentage below is success over the same 250 paired episodes for one
environment. Parentheses show mean final algebraic `D*`; lower is better.
`D*` is the minimum number of discrete Puzzle button presses needed to match
the board goal. It is not robot-motion distance, a value estimate, or proof of
reasoning.

| Environment | Mean initial `D*` | Flat alpha=1.0 | Mixer-L2 alpha=1.0 | Mixer-L2 alpha=0.4 |
| --- | ---: | ---: | ---: | ---: |
| Puzzle 3x3 | 6.2 | **99.6%** (0.0) | 73.2% (1.3) | 97.6% (0.1) |
| Puzzle 4x4 | 5.8 | 27.2% (3.6) | 46.0% (2.4) | **64.4%** (1.7) |
| Puzzle 4x5 | 12.8 | 14.0% (10.8) | 16.0% (10.8) | **19.2%** (10.5) |
| Puzzle 4x6 | 13.2 | 8.8% (12.6) | 4.8% (12.9) | **16.8%** (12.5) |

### Exact paired success contrasts

The entries below are net success changes among the same 250 initial states:
positive means the second named policy has more successes than the first.

| Environment | Mixer alpha=1.0 minus Flat | Mixer alpha=0.4 minus Mixer alpha=1.0 | Mixer alpha=0.4 minus Flat |
| --- | ---: | ---: | ---: |
| 3x3 | -66 | +61 | -5 |
| 4x4 | +47 | +46 | +93 |
| 4x5 | +5 | +8 | +13 |
| 4x6 | -10 | +30 | +20 |

For example, on 4x4 the Flat-versus-Mixer alpha=0.4 comparison contains 107
episodes solved only by alpha=0.4, 14 solved only by Flat, 54 solved by both,
and 75 solved by neither. This is strong within-checkpoint, within-protocol
evidence of a behavioral difference; it is not replication over independently
trained seeds.

## Difficulty-stratified result

The fixed canonical tasks expose a sharp boundary rather than a smooth small
success-rate reduction.

| Environment | Task `D*` values | Result by task |
| --- | --- | --- |
| 3x3 | 2, 5, 8, 9, 7 | Flat: 49--50/50 on every task. Mixer alpha=1.0: 33--40/50. Mixer alpha=0.4: 47--50/50. |
| 4x4 | 4, 6, 6, 6, 7 | Alpha=0.4: 46, 5, 43, 35, 32 successes per task. Task 2 is a shared bottleneck despite the same `D*=6` as tasks 3 and 4. |
| 4x5 | 4, 10, 14, 16, 20 | Task 1: Flat/Mixer-1.0/Mixer-0.4 = 35/40/48 successes. Tasks 2--5: 0/50 for all policies. |
| 4x6 | 6, 8, 12, 16, 24 | Task 1: 21/10/40 successes. Task 2: 1/2/2. Tasks 3--5: 0/50 for all policies. |

Task 2 of 4x4 demonstrates that `D*` alone is not the whole difficulty
description: board configuration, policy generalization, and physical approach
can matter even when the algebraic minimum press count agrees.

## Behavioral mechanism evidence

The following rates are aggregated over all single-source press events in a
cell; multi-source events are excluded from progress/regress classification.
Repeat-source rate is repeated button sources divided by all source presses.

| Env. | Policy | Press events / episode | Median first press | Progress / regress | Repeat-source rate |
| --- | --- | ---: | ---: | ---: | ---: |
| 3x3 | Flat alpha=1.0 | 6.50 | 19 | 97.5% / 2.5% | 3.4% |
| 3x3 | Mixer alpha=1.0 | 4.99 | 18 | 98.9% / 1.1% | 1.4% |
| 3x3 | Mixer alpha=0.4 | 6.16 | 14 | 99.3% / 0.7% | 1.2% |
| 4x4 | Flat alpha=1.0 | 10.15 | 21 | 60.9% / 39.1% | 44.1% |
| 4x4 | Mixer alpha=1.0 | 8.42 | 20 | 70.4% / 29.6% | 38.1% |
| 4x4 | Mixer alpha=0.4 | 10.78 | 17 | 68.8% / 31.2% | 51.9% |
| 4x5 | Flat alpha=1.0 | 19.18 | 22 | 55.3% / 44.7% | 70.2% |
| 4x5 | Mixer alpha=1.0 | 11.18 | 23 | 58.8% / 41.2% | 57.8% |
| 4x5 | Mixer alpha=0.4 | 5.84 | 18 | 69.8% / 30.2% | 24.6% |
| 4x6 | Flat alpha=1.0 | 22.74 | 22 | 51.4% / 48.6% | 65.8% |
| 4x6 | Mixer alpha=1.0 | 26.38 | 24 | 50.6% / 49.4% | 69.7% |
| 4x6 | Mixer alpha=0.4 | 8.10 | 19 | 54.1% / 45.9% | 34.4% |

### What these metrics support

1. **3x3 Mixer alpha=1.0 is primarily under-executing, not selecting bad
   presses.** Its press quality is high, but it performs about 1.5 fewer press
   events per episode than Flat and has 67 horizon-exhausted failures. Alpha
   0.4 restores enough execution and lowers first-press latency.
2. **4x4 Mixer alpha=1.0 improves local operator quality relative to Flat.**
   It has fewer regression and repeated-source presses, lower final `D*`, and
   47 additional paired successes. Alpha 0.4 improves completion further, but
   does so with more total and repeated presses than alpha=1.0; it should not
   be summarized simply as “more precise.”
3. **4x5 and 4x6 failures are consistent with sequential-composition and
   late-regression limits.** Policies may temporarily lower `D*`, then return
   away from their best state. Mean final-minus-minimum `D*` backslide for
   Flat/Mixer-1.0/Mixer-0.4 is 1.08/0.70/0.42 on 4x5 and 2.01/2.22/1.74 on
   4x6. Alpha=0.4 reduces action volume and repeated sources, but high-distance
   tasks remain unsolved.

### Concrete, fully paired example

For `puzzle-4x4-play-v0`, `task03_ep000`, all policies received the same full
goal and initial observation; its initial `D*=6`.

| Policy | Outcome | Press pattern |
| --- | --- | --- |
| Flat alpha=1.0 | failure at horizon 500; final/min `D*=5/3` | 13 presses, 8 repeats; reaches 3 then cycles between 3 and 4. |
| Mixer alpha=1.0 | failure at horizon 500; final/min `D*=4/4` | four presses: `6→5→4→5→4`, then no completion. |
| Mixer alpha=0.4 | success at step 191; final/min `D*=0/0` | six non-repeated progress presses: `6→5→4→3→2→1→0`. |

This is an illustrative trajectory, not an aggregate proof. It shows why
`episodes.csv` must be paired with `press_events.csv` before assigning a
mechanism to a success-rate difference.

## Cross-check against the source training evaluations

The source-run `summary.json` / final row of `eval.csv` and M23A tell a
consistent broad story. The source evaluation used its own original protocol;
M23A uses 50 episodes/task plus controlled full-goal replay, so numerical
identity is neither expected nor required.

| Environment | Flat: source final → M23A | Mixer alpha=1.0: source final → M23A | Mixer alpha=0.4: source final → M23A |
| --- | ---: | ---: | ---: |
| 3x3 | 99.0% → 99.6% | 77.0% → 73.2% | 96.0% → 97.6% |
| 4x4 | 26.0% → 27.2% | 47.0% → 46.0% | 60.0% → 64.4% |
| 4x5 | 13.0% → 14.0% | 19.0% → 16.0% | 20.0% → 19.2% |
| 4x6 | 12.0% → 8.8% | 8.0% → 4.8% | 18.0% → 16.8% |

M23A intentionally uses `final@1M`, not each source's individually best
checkpoint. It therefore describes the final checkpoint behavior rather than
the largest transient score seen during source training.

## How to inspect the artifacts

Set the root once:

```bash
export M23A_ROOT=/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M23A/final_last1m_alltasks_ep50_evalSeed20260909_controlled-goal-replay-v1
export RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
```

Read artifacts in this order:

1. `campaign_manifest.json` — global completion status, frozen protocol, all
   source runs/checkpoint hashes, diagnostic source commit.
2. `pairing_invariants.json` — the prerequisite for paired comparisons.
3. `runs/<config>/<environment>/seed_000/manifest.json` — exact source
   checkpoint, restored configuration, provenance status, backend, and local
   diagnostic invariants.
4. `episodes.csv` — one row per task episode; this is the primary aggregate
   table.
5. `press_events.csv` — one row per detected physical press; use this to test
   a mechanism hypothesis for a selected paired episode.
6. `controlled_goal_replay/<environment>/manifest.json` — persisted replay
   archive identity; `records.npz` is an audit artifact rather than a primary
   analysis table.

Useful read-only commands:

```bash
"$RLC_PYTHON" -m json.tool "$M23A_ROOT/campaign_manifest.json" | less
"$RLC_PYTHON" -m json.tool "$M23A_ROOT/pairing_invariants.json"

for f in "$M23A_ROOT"/runs/*/puzzle-4x4-play-v0/seed_000/episodes.csv; do
  echo "=== $f ==="
  rg 'task03_ep000' "$f"
done
```

For the same `paired_episode_id`, repeat the final loop with
`press_events.csv`. Those rows provide `Dstar_before`, `Dstar_after`,
`progress_class`, source-button indices, and time between presses. Use a real
CSV reader for `press_events.csv`: its list-valued fields contain quoted
commas, so naive comma splitting is invalid.

### Reading guide for the primary columns

| Field(s) | Meaning | Safe interpretation |
| --- | --- | --- |
| `paired_episode_id` and the three fingerprints | Link the same controlled initial condition across policies | Compare rows only after this identity is verified. |
| `success`, `terminated`, `truncated`, `horizon_exhausted` | Completion versus timeout | A horizon failure is not evidence that no local progress occurred. |
| `initial_Dstar`, `min_Dstar`, `final_Dstar` | Algebraic board-distance trajectory | Distinguish no progress, temporary progress then regression, and completion. |
| `num_press_events`, `first_press_step` | Whether and when the policy executes a press | Interpret episode length carefully: success terminates episodes early. |
| `num_progress_presses`, `num_regress_presses` | Quality of single-source press choices | Aggregate over presses, not only per-episode averages. |
| `num_repeat_source_presses` | Re-use of prior source buttons | A useful waste/regression signal, not proof that every repetition is wrong. |
| `press_events.csv` `Dstar_before/after` | Effect of each physical press | The strongest evidence for a trajectory-level mechanism. |

## Interpretation boundary

### Supported statements

- These exact final checkpoints exhibit the reported task-dependent behavioral
  differences under identical policy-facing goals and initial observations.
- At high Puzzle dimensions, their failures are accompanied by incomplete
  composition, repeated source pressing, regression after partial progress,
  and horizon exhaustion.
- Alpha=0.4 is a favorable operating point among these available checkpoint
  sources on 4x4--4x6, while being near but not above Flat on 3x3.

### Statements not supported

- Alpha=0.4 is globally optimal.
- Mixer-L2 is universally superior to Flat.
- The effect belongs solely to the actor: M16B's Flat/Mixer comparison changes
  the declared structured computation treatment across actor, value, and
  critic.
- Any result will replicate across independently trained seeds, alternate
  checkpoints, altered goals, or altered training semantics.
- `D*` measures robot control difficulty, value accuracy, representation
  quality in isolation, or reasoning in a general sense.

The 250 paired episodes quantify conditional behavior of one trained seed per
cell. They provide rich within-checkpoint evidence, but do not replace
training-seed replication. The M16D alpha=0.4 provenance qualification remains
part of every resulting claim.

## Useful questions for further discussion

1. Is the main alpha=0.4 benefit a better discrete operator-selection policy,
   a change in execution/press frequency, or a mixture that depends on task
   geometry?
2. Why is 4x4 task 2 difficult for every checkpoint despite sharing `D*=6`
   with tasks 3 and 4? Does the trajectory trace indicate a goal-layout,
   physical-approach, or learned-generalization issue?
3. On 4x5/4x6, do policies fail mostly before reaching a near-solved state, or
   do they reach one and destroy it through repeated presses? The answer should
   be measured task-by-task from `min_Dstar`, `final_Dstar`, and event traces.
4. Would selecting source `best` checkpoints change the diagnosis, or only the
   headline success rate? This requires a separately declared checkpoint
   selection study, not replacing the frozen M23A endpoint ad hoc.
5. Which follow-up would most efficiently separate architecture from alpha:
   matched multi-seed Flat/Mixer alpha grids, checkpoint-time trajectories, or
   a targeted intervention on repeated presses/termination behavior?

## Result locations

- Campaign root: `/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M23A/final_last1m_alltasks_ep50_evalSeed20260909_controlled-goal-replay-v1/`
- Per-cell outputs: `runs/<M23A-config>/<environment>/seed_000/{manifest.json,episodes.csv,press_events.csv}`
- Source checkpoints/runs: exact paths and hashes are in `campaign_manifest.json`.

No source checkpoint, training run, or diagnostic output was modified while
creating this brief.
