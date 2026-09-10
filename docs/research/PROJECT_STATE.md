# RLC project state

Last reviewed: 2026-09-11. This is a compact state summary; run artifacts and
resolved configurations remain authoritative for exact progress and results.

## Current scientific focus

The current thread concerns Puzzle computation, goal-coordinate semantics, and
failure mechanisms in the unified RLC runtime. M24A is configured as the next
direction-selection Study, while M22 remains active/partial and M23A provides
the completed controlled-goal-replay behavioral diagnosis that motivates the
intervention.

## Established evidence

- M16A seed-0 results show a task-dependent Mixer effect: L2/L4 improve over
  Flat on 3x3 and 4x4, but not uniformly on 4x5/4x6. The joint
  actor/value/critic placement prevents assigning the effect to one slot.
- M16A/M16B show an interaction between alpha and architecture. M16C scanned
  `{0.1, 0.2, 0.5, 0.7}` on 4x4 S002; it did not train alpha=0.4. M16D uses
  0.4 as a later empirically motivated operating point, not as an optimum.
- M22's semantic gate classifies CRL as an executable semantic match and
  GCIVL/GCIQL as explicitly documented RLC variants with post-gradient target
  Polyak updates. The campaign is not byte-for-byte upstream equivalence.
- The generic sweep launcher now defaults child processes to
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; M16D's restart note records the
  controlled single- and dual-process memory checks.
- M23A completed its twelve-cell, 3,000-rollout controlled-goal-replay audit.
  The exact final checkpoints show low-distance task competence but no robust
  4x5/4x6 long-sequence composition; alpha=0.4 Mixer-L2 is the best available
  source on 4x4--4x6, while Flat alpha=1.0 is strongest on 3x3. This is
  single-training-seed diagnostic evidence, with M16D alpha=0.4 retaining its
  scoped provenance exception. See the [M23A results brief](experiments/M23A-F0/RESULTS_DISCUSSION_BRIEF.md).
- M24A's preflight validates six seed-0 cells on Puzzle-4x5/4x6, with identical
  raw transition/value-goal/actor-goal sampling and identical parameter trees
  across board, residual, and exact operation-parity conditioning. This is a
  design/runtime validation only; it is not training evidence.

## Active studies

| Item | Current state | Authority |
| --- | --- | --- |
| M16D | Formal attempt 001 is complete for all four cells; analysis and provenance-qualified handoff remain. | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16D/` |
| M22 | Formal campaign is active/partial in the external run tree; the repository completion manifest is not paper-ready. | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M22/` |
| M23A | Completed: 12/12 cells, 3,000 controlled paired rollouts; discussion/replication analysis remains. | [`M23A authority`](experiments/M23A-F0/README.md), [`results brief`](experiments/M23A-F0/RESULTS_DISCUSSION_BRIEF.md), and external diagnostic artifact root |
| M24A | Study/configuration preflight complete; 0/6 formal runs started. Git review, clean frozen source, exact concurrent-GPU smoke, and physical GPU assignment remain launch gates. | [`M24A Study`](../../experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml), [`M24A protocol`](../../experiments/M24A_puzzle_goal_conditioning_intervention/README.md) |

The M16D Study and M22 Study retain design-time `formal_training_started: false`
fields. These declarations conflict with the observed external run artifacts;
they are not silently rewritten here. M24A's same field currently agrees with
the absent formal output namespace and process audit. Use actual run metadata
and process state for execution status and Study/config files for declared
design semantics.

The sampled M16D attempt-001 runtime metadata records source commit
`fdbd7875a031c537b98b415a3c73ce0bf6a02721` with `git_dirty: true`; this is the
effective recorded provenance for those runs and must not be replaced by the
clean documentation snapshot `279d7b6` reviewed on 2026-09-10. M23A permits exactly its
four declared alpha=0.4 sources through a scoped, user-attested / partially
machine-verified provenance exception; it does not claim their historical dirty
diff was recovered or their source was fully verified clean. Sampled M22
runtime metadata records `2617634f29a883581b6fbf582db072f08b9d24ff` with
`git_dirty: false`, matching its frozen worktree.

## Immediate next study

Review and commit the M24 capability and M24A Study definitions separately,
then repeat the dry-run from a clean detached snapshot. Before any formal
launch, assign the physical GPU and pass the exact two-process workload smoke.
Do not start M24A from the currently dirty main worktree.

## Major open scientific questions

- After provenance validation, does M16D alpha=0.4 reproduce a useful
  Mixer-L2 operating point across the four Puzzle sizes under its single-seed
  exploratory scope?
- What are the complete, reproducible M22 baseline outcomes across algorithms,
  environments, and three seeds once all runs are complete?
- Which mechanism best explains M23A's high-distance failure boundary, and
  what matched multi-seed or intervention study can distinguish architecture,
  alpha, and closed-loop execution effects?
- Under fixed GCIQL and Mixer-L2, do residual or exact operation-parity
  coordinates improve final@1M Task2--5 performance over target-board
  conditioning on Puzzle-4x5/4x6?

## Source-of-truth order

`actual run artifacts / resolved configs` > `source at recorded SHA` >
`Study/config files` > `experiment README` > `project context docs` > `chat
history`.
