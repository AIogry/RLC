# RLC status

## HIQL computation and runtime migration

### Completed

- HIQL `high_actor` and `low_actor` computation parity: locally validated.
- HIQL `value` MLP + FeedForward + Direct migration: locally validated.
- `target_value` automatically mirrors the online value architecture; it is
  not an independent computation slot.
- Value ensemble size and independent parameter slices are preserved.
- `goal_rep`, value objective, expectile loss, advantage, actor losses, and
  Polyak target-update semantics remain legacy behavior.
- OGBench `Dataset`, `GCDataset`, and `HGCDataset` are available in the RLC
  runtime with explicit seeded NumPy generators.
- RLC `main.py` now runs the HIQL train/update/evaluation/checkpoint loop while
  importing the canonical `RLC/ogbench` package.
- Real-data determinism, sampling sanity, and strict N=20 legacy-vs-computation
  parity tests pass.
- GPU N=1000 strict parity diagnostic passed on 2 x NVIDIA GeForce RTX 4090;
  matched-initialization loss and semantic parameter errors were 0.0.

### Validation scope

The deterministic CPU parity suite (25 tests including the synthetic smoke and
three real-runtime tests)
covers primitive/network forward passes,
all three losses, semantic full-tree gradients, target forward, Polyak update,
real optimizer updates, parameter counts, and N=10/N=20 regressions.  The
real runtime tests use `antmaze-medium-navigate-v0`, verify 20 deterministic
batches and real trajectory metadata, and compare 20 strict HIQL updates.
Core parity comparisons use `rtol=0`, `atol=1e-6` and observe zero loss and
gradient error.

The optimizer's aggregate `grad/norm` diagnostic can differ by approximately
`6.1e-5` because raw pytree leaf ordering changes when a computation wrapper is
introduced.  This is a reduction-order-only diagnostic; semantic gradients,
optimizer deltas, losses, and target updates remain exact in the CPU tests.

### Integration validation

Both native modes completed a 1,000-step CPU trainer smoke on
`antmaze-medium-navigate-v0` with finite losses, evaluation, CSV logging, and
checkpoint save/load action/value equality.  The real runtime test also
confirmed that `ogbench` resolves to `RLC/ogbench` and that the configured
dataset directory is logged.

The separate GPU diagnostic completed 1,000 matched-initialization updates.
There was no exact, float32-tolerance, or semantic-parameter divergence.  The
only nonzero metric was the aggregate `grad/norm` diagnostic, whose maximum
absolute error was `6.103515625e-05` from pytree reduction ordering.  Native
step-0 initialization already differed, with total loss `16.81831932` for
legacy HIQL and `11.03039646` for computationized HIQL.

## Current recommendation

Mark the HIQL computationized baseline as **integration validated for the
local vertical slice**.  This does not authorize formal long baseline
reproduction or CRL migration; those remain separate milestones.
