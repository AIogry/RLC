# Research decisions

These records preserve decisions that are supported by repository evidence.
They distinguish verified facts from the interpretation or operating policy
derived from them.

## 2026-09-09 — M22 is a unified RLC baseline campaign, not universal byte equivalence

**Verified.** The M22 audit records official OGBench Puzzle hyperparameters and
classifies GCBC, QRL, CRL, and HIQL as `upstream_semantic_match`. GCIVL and
GCIQL are `rlc_variant_documented` because they retain post-gradient target
Polyak updates. The audit explicitly sets byte-for-byte upstream equivalence to
false.

**Decision.** Use the campaign name “M22 — OGBench Puzzle Baselines in Unified
RLC”; report the two disclosed RLC variants rather than treating them as
blockers or hiding them.

Evidence: [`M22_upstream_baseline_audit.md`](../../docs/9-9/M22_upstream_baseline_audit.md),
[`canonical_semantics_audit.json`](../../docs/9-9/canonical_semantics_audit.json),
[`M22 study`](../../experiments/M22_puzzle_baselines_unified_rlc/study.yaml).

## 2026-09-09 — CRL DDPG+BC actor-loss gradient semantics are a match

**Verified.** The executable audit differentiates isolated actor loss with
respect to the full parameter tree. Both upstream and RLC have positive
actor-subtree gradients, numerical-zero critic-subtree gradients, and an
active `dQ/da` pathway. The earlier syntax-only material-difference record is
retained as historical evidence and superseded by this audit.

**Decision.** Classify
`crl.ddpgbc_critic_gradient_in_actor_loss` as
`match_same_gradient_semantics_different_implementation_style`.

Evidence: the CRL correction section of
[`M22_upstream_baseline_audit.md`](../../docs/9-9/M22_upstream_baseline_audit.md).

## 2026-08-30 — Alpha 0.4 is an operating point, not an optimum

**Verified.** M16C trained alpha `{0.1, 0.2, 0.5, 0.7}` on 4x4 S002 and did
not train 0.4. M18 records 0.4 as the midpoint of an empirically favorable
region used for later Puzzle Mixer work.

**Decision.** M16D may use alpha=0.4 as a fixed completion/contrast cell, but
must not call it the best or optimal alpha and must preserve its single-seed
descriptive boundary.

Evidence: [`M16D study`](../../experiments/M16D_puzzle_mixer_alpha04_completion/study.yaml),
[`M16C study`](../../experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml),
[`M18 parameter-scaling record`](../../docs/8-30/M18_fixed_parameter_recurrent_scaling.md).

## 2026-09-09 — Two jobs per GPU is an engineering scheduling policy

**Verified.** The M21 benchmark defines a single-versus-dual workload, records
GPU/process telemetry and throughput, enforces a 90% peak-memory safety limit,
and provides a recommendation rule. The generic scheduler and tests support
`jobs_per_gpu=2`; M22 declares physical GPU1 with two worker slots.

**Boundary.** The final M21 benchmark report is not present in the current
repository, so no numerical speedup is reconstructed here.

**Decision.** `jobs_per_gpu=2` is an available throughput-oriented operating
policy, subject to per-workload memory smoke tests; it is not a scientific
factor or a guarantee that every model should use two workers.

Evidence: [`m21_gpu_concurrency_benchmark.py`](../../tools/m21_gpu_concurrency_benchmark.py),
[`test_m21_gpu_concurrency.py`](../../tests/experiment/test_m21_gpu_concurrency.py),
[`M22 README`](../../experiments/M22_puzzle_baselines_unified_rlc/README.md).

## 2026-09-10 — Generic JAX child allocation defaults to on-demand

**Verified.** M16D attempt 0 showed one child reserving approximately 75% of
the 24-GiB GPU. The reusable `tools/sweep.py` child environment now applies
`setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')`, while preserving an
explicit caller value. Real-data single and concurrent smoke checks are
recorded in the M16D restart note.

**Decision.** Keep the policy in generic launcher infrastructure, not in an
algorithm or M16D-specific implementation. Formal runs must still be checked
for actual peak memory and OOMs.

Evidence: [`tools/sweep.py`](../../tools/sweep.py),
[`M16D restart note`](../../experiments/M16D_puzzle_mixer_alpha04_completion/attempt_001_restart_note.md).

## 2026-09-10 — M16D attempt 001 remains provenance-qualified

**Verified.** Sampled attempt-001 runtime metadata records source commit
`fdbd7875a031c537b98b415a3c73ce0bf6a02721` with `git_dirty: true`. The current
clean main worktree is `279d7b6`; it is not evidence of the code state used by
that run.

**Decision.** Retain the attempt-001 artifacts, but do not combine them into a
clean-source comparison or publish an analysis until the recorded dirty state
and effective source diff have been resolved.

Evidence: external M16D `runtime_metadata.json` files and the
[`M16D context`](experiments/M16D/README.md).

## 2026-09-10 — M16D attempt 001 execution is complete, analysis is separate

**Verified.** The four attempt-001 M16D run summaries record
`status: completed`, with final checkpoint/evaluation artifacts under the
declared M16D run root.

**Decision.** Mark the formal execution phase complete, but keep scientific
analysis and cross-study comparison gated on the recorded dirty-source
provenance. Completion status alone does not establish an alpha optimum or a
multi-seed result.

Evidence: external M16D `summary.json` and checkpoint metadata files; the
[`M16D context`](experiments/M16D/README.md).

## 2026-08-16 onward — Dependency-aware targeted regression

**Verified.** The repository separates computation, agent, experiment, launch,
diagnostic, and analysis tests; prior handoff records run targeted dependent
tests after shared-runtime changes rather than treating a full historical
suite as a prerequisite for every change.

**Decision.** Changes to shared runtime or experiment infrastructure require
the changed-layer tests plus directly dependent Study/lifecycle/launcher tests.
Documentation-only work uses lightweight path/link and whitespace checks.

Evidence: [`experiment execution policy`](../../docs/8-16/experiment_execution.md),
[`M16D targeted tests`](../../tests/experiment/test_m16d.py), and the test tree.

## 2026-08-16 onward — Feature worktree to main to frozen experiment worktree

**Verified.** The formal execution record requires review and commit on the
main development line, followed by a detached frozen worktree whose SHA is
recorded by the launcher. Existing experiment handoffs repeatedly preserve
older frozen worktrees.

**Decision.** Use feature worktrees for repository changes, merge/review on
main, and launch formal training only from a clean detached snapshot. Never
patch a frozen/running experiment tree in place.

Evidence: [`experiment execution policy`](../../docs/8-16/experiment_execution.md),
[`M19A handoff`](../../docs/9-4/M19A_implementation_handoff.md).
