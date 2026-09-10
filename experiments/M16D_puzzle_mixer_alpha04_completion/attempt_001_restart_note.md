# M16D attempt 1 restart note — 2026-09-10

## Retained attempt 0

The original canonical attempt 0 artifacts are retained without alteration:

- `M16D-P3X3-S002-A04` reached training step 55,000.
- `M16D-P4X4-S002-A04` reached training step 45,000.
- `M16D-P4X5-S002-A04` and `M16D-P4X6-S002-A04` were not started.

The sweep parent and its two children were deliberately stopped after the
P4X4 child was observed to consume about 18,464 MiB on GPU0. Its metadata
therefore remains the original, stale `running` record rather than being
rewritten as a completed result.

## Cause and remediation

The generic `tools/sweep.py` launcher had set the CUDA device but had not set
`XLA_PYTHON_CLIENT_PREALLOCATE`. JAX consequently used its default
process-wide preallocation policy for one child. The launcher now defaults
that variable to `false` with `setdefault`, so an explicit caller-provided
value remains authoritative.

Real-data smoke tests on GPU0 verified the new default without setting the
variable in the parent process:

- Single P4X4: the child environment contained
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; observed use was about 4,516 MiB,
  not a fixed approximately 18 GiB reservation.
- Concurrent P3X3 plus P4X4: both reached step 2,000 and passed checkpoint
  save/restore probes; observed uses were about 1,958 MiB and 4,518 MiB,
  with about 15,458 MiB free on the 24,564 MiB GPU.

## Formal restart

Formal execution uses `run_attempt=1` for all four cells. This creates the
`seed_000__attempt_001` paths, preserves attempt 0 for diagnosis, and runs
only on physical GPU0 with two worker slots.
