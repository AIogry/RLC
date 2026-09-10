# RLC project state

Last reviewed: 2026-09-10. This is a compact state summary; run artifacts and
resolved configurations remain authoritative for exact progress and results.

## Current scientific focus

The current thread concerns Puzzle computation, alpha sensitivity, and failure
mechanisms in the unified RLC runtime. The active work combines the M16D
alpha-completion study, the M22 official-baseline campaign, and the M23A-F0
observation-only rollout diagnostic foundation.

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

## Active studies

| Item | Current state | Authority |
| --- | --- | --- |
| M16D | Formal attempt 001 is complete for all four cells; analysis and provenance-qualified handoff remain. | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16D/` |
| M22 | Formal campaign is active/partial in the external run tree; the repository completion manifest is not paper-ready. | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M22/` |
| M23A-F0 | Diagnostic foundation is merged; no M23 formal Study or result artifact is declared here. | M23A-F0 README and source links below |

The M16D Study and M22 Study retain design-time `formal_training_started: false`
fields. These declarations conflict with the observed external run artifacts;
they are not silently rewritten here. Use the actual run metadata/process state
for execution status and the Study/config files for declared design semantics.

The sampled M16D attempt-001 runtime metadata records source commit
`fdbd7875a031c537b98b415a3c73ce0bf6a02721` with `git_dirty: true`; this is the
effective recorded provenance for those runs and must not be replaced by the
current clean documentation worktree at `279d7b6`. Sampled M22 runtime metadata
records `2617634f29a883581b6fbf582db072f08b9d24ff` with `git_dirty: false`,
matching its frozen worktree.

## Immediate next study

M23A is the next planned scientific step after the relevant source checkpoints,
run status, and diagnostic scope are explicitly reviewed. M23A-F0 currently
provides infrastructure only; it does not authorize a formal diagnostic run.

## Major open scientific questions

- After provenance validation, does M16D alpha=0.4 reproduce a useful
  Mixer-L2 operating point across the four Puzzle sizes under its single-seed
  exploratory scope?
- What are the complete, reproducible M22 baseline outcomes across algorithms,
  environments, and three seeds once all runs are complete?
- Do M23A rollout/event metrics distinguish data/task support failures from
  representation, optimization, or closed-loop policy failures?

## Source-of-truth order

`actual run artifacts / resolved configs` > `source at recorded SHA` >
`Study/config files` > `experiment README` > `project context docs` > `chat
history`.
