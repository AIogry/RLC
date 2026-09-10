# M23A-F0 — Puzzle rollout diagnostic foundation

## Scientific purpose

Provide a deterministic, observation-only post-hoc diagnosis of canonical
OGBench Puzzle rollouts. The foundation extracts physical button press events,
checks board effects against the pinned toggle rule, and computes exact GF(2)
residual distance (`D*`) and episode-level progress summaries. It is intended
to help distinguish task/data support failures from policy or closed-loop
failures in later M23A work.

## Fixed protocol

- Input is one existing source Run/checkpoint; no training or optimizer update
  is performed.
- Rollouts use deterministic temperature 0 and an explicit evaluation seed and
  episode count. The source checkpoint is restored through the existing
  reevaluation path and its network fingerprint is checked before/after.
- Press sources come only from the authoritative joint-position threshold
  crossing; board differences are used for consistency checks, not source
  inference. The GF(2) matrix uses row-major boards and press-source columns.
- Outputs are a separate diagnostic root containing `manifest.json`,
  `episodes.csv`, and `press_events.csv`; source run/checkpoint directories are
  not modified or used as output roots.

## Treatment and control

F0 has no learned treatment/control and no formal Study matrix. It is a
controlled measurement procedure over a selected checkpoint, task set, and
seed protocol. Any later comparison between policies or checkpoints belongs to
M23A and must declare those factors separately.

## Current status

The reusable foundation and its tests were merged in commit `3600cf6`
(`M23A-F0: add Puzzle rollout diagnostic foundation`). No formal M23A Study,
diagnostic result root, or scientific performance claim is recorded here.

## Interpretation boundary

Event-effect consistency, `D*`, press timing, and progress classifications are
diagnostic observables. They do not by themselves establish causality,
representation superiority, optimization failure, or a policy's environment
return. Multi-source events and unreachable residuals must remain explicit
invariant outcomes rather than being silently converted into pseudo-distance.

## Relevant artifacts

- [`puzzle diagnostics package`](../../../../impls/diagnostics/puzzle/)
- [`event diagnosis tool`](../../../../tools/run_puzzle_event_diagnosis.py)
- [`diagnostic tests`](../../../../tests/diagnostics/test_puzzle_event_diagnosis.py)
- Source commit: `3600cf6` (`M23A-F0: add Puzzle rollout diagnostic foundation`)

## Next action

Before any M23A execution, select and validate source checkpoints, define the
task/episode and output namespace, and register the scientific comparison in a
Study/config layer if it is a formal experiment. Reuse the F0 APIs; do not
introduce M23A-specific one-off logic into `impls/`.
