# M23A — Puzzle Direct Rollout Behavioral Audit

Status: completed. The full twelve-cell campaign completed under the frozen
protocol recorded here. This README remains the durable design/provenance
authority; the result interpretation and discussion-ready tables are in the
[M23A results and discussion brief](RESULTS_DISCUSSION_BRIEF.md).

## Scientific question

Why does the structured Puzzle MLP-Mixer policy outperform the canonical flat
GCIQL policy? M23A diagnoses whether the difference is primarily associated
with physical execution, useful operator selection, sequential composition,
destructive/repeated presses, or conversion of near-solved states into success.

This is a post-hoc diagnostic study over existing trained policies. It does not
train a new policy or modify GCIQL/Mixer semantics.

## Foundation

M23A-F0 provides observation-only physical press detection, pinned GF(2)
Puzzle dynamics, exact algebraic `D*`, episode/press-event metrics, and
deterministic checkpoint rollout diagnosis. M23A reuses those generic APIs and
adds generic controlled goal replay at the diagnostics layer.

## Treatments and source-policy cells

The twelve cells are the Cartesian product of four canonical Puzzle
environments and these three existing trained-policy conditions:

| Source policy | Training family | alpha |
| --- | --- | --- |
| Flat GCIQL | M16B | 1.0 |
| Puzzle Mixer-L2 | M16B | 1.0 |
| Puzzle Mixer-L2 | M16D | 0.4 |

Environments are `puzzle-3x3-play-v0`, `puzzle-4x4-play-v0`,
`puzzle-4x5-play-v0`, and `puzzle-4x6-play-v0`; all source training seeds are
`0`. The declarative authority for the exact source mapping is the
[M23A Study](../../../../experiments/M23A_puzzle_direct_rollout_audit/study.yaml)
and its twelve configuration files. Primary source checkpoints are always
`final@1M`; primary analyses do not mix `best` and `final` checkpoints.

## Source provenance rule

Every source is validated against runtime metadata, resolved config, completed
status, source seed and attempt, source commit, final checkpoint metadata and
SHA-256, alpha, and architecture identity. The source dependency is not
silently substituted.

The eight M16B sources must satisfy the normal clean-provenance gate. Four
M16D `seed_000__attempt_001` sources are the sole exception:

```text
provenance_status = scoped_exception
reason = concurrent M23A-F0 diagnostic development
evidence_level = user-attested / partially machine-verified
```

Their exception is scoped to the declared study/config/environment/training
seed/run attempt/source commit/checkpoint SHA-256. It does not relax the
clean-provenance gate for any other source. The historical exact dirty diff
cannot be machine-recovered, so these sources must not be described as fully
verified clean; the listed evidence level is the maximum supported claim.

The M16B Mixer-L2 checkpoints predate the M17 modular ownership layout. For
diagnostic inference only, the reevaluation path uses an exact, shape-checked
legacy-Mixer parameter-layout adapter and records
`checkpoint_restore_mode = legacy_mixer_layout_adapter_v1` in its manifest.
It neither mutates a source checkpoint nor changes policy/training semantics.

## Frozen diagnostic evaluation protocol

All twelve source policies use exactly:

- source training seed `0` and primary checkpoint `final@1M`;
- all five canonical task IDs: `1, 2, 3, 4, 5`;
- `episodes_per_task = 50`;
- `evaluation_seed = 20260909`;
- `eval_temperature = 0`;
- `eval_gaussian = None`.

### Controlled goal replay and pairing invariants

For every `(environment, task_id, episode_index)`, diagnostics first capture a
canonical real reset and persist its complete policy-facing goal observation,
board-goal observation, and initial observation. The same persisted full goal
is injected into all three source-policy rollouts for that paired episode;
this occurs only in the diagnostic wrapper and does not change `PuzzleEnv` or
training semantics.

The following are hard invariants, not soft comparisons:

- the three `goal_fingerprint` values are identical;
- the three `board_goal_fingerprint` values are identical;
- the three `initial_observation_fingerprint` values are identical.

The wrapper validates the native reset's board target and initial observation
before replacing the policy-facing goal. An initial-observation mismatch fails
the rollout immediately; that sample is not considered paired. Every episode
output records all three fingerprints and the paired-episode identifier.

## Primary behavioral metrics

Execution metrics include `first_press_step`, `num_press_events`, and
`steps_between_presses`. Operator-quality metrics include progress, neutral,
and regression press counts/rates. Composition and waste include unique button
sources, repeated source presses, and `D*` trajectories. Completion metrics
include `initial_Dstar`, `min_Dstar`, `final_Dstar`, success, and horizon
exhaustion.

`D*` is an algebraic minimum press distance, not robot-motion distance, control
difficulty, policy value, or proof of reasoning. M23A is diagnostic evidence
for distinguishing candidate mechanisms, not a causal result on its own.

## Boundaries and execution authority

M23A does not add oracle policies, Local-GCIQL, solver/executor training, a
neural composer, representation architectures, or new policy training. It does
not modify M16D/M22 source artifacts.

The prepared [campaign runner](../../../../tools/run_puzzle_diagnostic_campaign.py)
may validate dependencies, run targeted tests, and run a declared tiny smoke.
It must not launch the full M23A campaign unless the user explicitly authorizes
it. The completed campaign was manually launched under a user-authorized
non-training diagnostic exception from the clean
`/home/eai/Research/RLC-M23A` feature worktree on
`m23a-direct-rollout-audit`; it did not modify source training artifacts.

## Completed campaign record

The actual completed artifact, rather than the design-time
`formal_evaluation_started: false` field in the Study, is authoritative for
execution status:

- [campaign manifest](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M23A/final_last1m_alltasks_ep50_evalSeed20260909_controlled-goal-replay-v1/campaign_manifest.json)
  records `status: completed`, diagnostic commit
  `21e0cb838c1476ac6ec62f63e6f1152702afcde9`, and `diagnostic_git_dirty: false`;
- twelve cells × five tasks × fifty episodes produced 3,000 rollout records;
- [pairing invariants](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M23A/final_last1m_alltasks_ep50_evalSeed20260909_controlled-goal-replay-v1/pairing_invariants.json)
  records 1,000 three-policy paired groups and `status: passed`.

The completed campaign is an observation-only, single-training-seed diagnostic.
It does not turn alpha=0.4 into an optimum claim or establish a causal
architecture result. See the linked discussion brief for the exact results,
their scope, and recommended follow-up questions.
