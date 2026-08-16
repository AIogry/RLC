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

## CRL migration

- CRL baseline migrated and registered in the shared runtime.
- Reference DDPG+BC and AWR semantics pass synthetic forward/loss/gradient/
  update parity.
- CRL actor-only `MLP + FeedForward + Direct` computationization passes
  synthetic and real `antmaze-medium-navigate-v0` N=20 strict parity.
- The shared main path completes both 1,000-step GPU CRL smoke modes with
  evaluation, logging, and checkpoint save/load.
- CRL `critic_state` and `critic_goal` now independently computationize the
  `phi(s, a)` and `psi(g)` branches; their bilinear interaction remains
  parameter-free and CRL-specific.
- AWR value branches are now independently computationizable through
  `value_state` and `value_goal`; they remain separate from critic branches.

CRL baseline migration, actor computationization, critic computationization,
and AWR value computationization are **integration validated locally**.

## CRL critic computationization (M5)

- CPU regression: `32/32` tests passed.
- Real `antmaze-medium-navigate-v0` `GCDataset` N=20 strict parity has
  `first_divergence=None` and maximum tracked error `0.0`.
- GPU legacy versus actor+critic computation trainer smokes completed 1,000
  steps with finite losses, evaluation, logging, and checkpoint probes.
- Small-config accounting is `1616` total, `1164` critic, `630` critic-state
  core, `534` critic-goal core, and `0` trainable bilinear/readout parameters.

## CRL AWR value computationization (M6)

- Added independent `value_state` and `value_goal` slots for the separate
  `GCBilinearValue(ensemble=False)` AWR value network.
- AWR value forward/loss, `Q1/Q2/V`, advantage, clipped exponential weight,
  actor loss, gradients, one-step update, optimizer state, and RNG parity pass.
- The four representation groups (`critic_state`, `critic_goal`,
  `value_state`, `value_goal`) are independent; value remains `ensemble=False`.
- DDPG+BC does not instantiate `modules_value`, and its parameter count remains
  unchanged even when value slot configuration entries are enabled.
- Full AWR real `antmaze-medium-navigate-v0` N=20 strict parity passes with
  `first_divergence=None` and all tracked maximum errors `0.0`.
- GPU 1000-step native legacy/full-computation AWR smokes pass with finite
  critic/value/actor diagnostics, evaluation, logging, and checkpoint probes.
- Small-config AWR accounting is `2150` total, `452` actor, `1164` critic,
  `534` AWR value, with value-state/value-goal cores `267` each.

CRL AWR value computationization and the full computationized AWR-CRL baseline
are **integration validated locally**. Long training and new computation
primitives remain out of scope.

## M8 computation foundation (2026-08-15)

- Added the frozen computation ontology: `Operator -> Primitive -> Block`.
- Documented the independent dimensions `State Structure`, `Topology`,
  `Execution Schedule`, `Parameter Reuse`, and `Credit Structure`.
- `ComputationCore` now delegates optional state semantics to topology;
  `FeedForward` explicitly rejects non-`None` state.
- Added standalone computation-side `MLPMixerBlock`; vanilla CoGHP remains on
  the frozen `networks.coghp.MixerBlock` implementation.
- CRL vanilla defaults now disable actor, critic, and AWR value computation
  slots; explicit `--computation` still enables migration paths.
- Runtime metadata preserves `computation` and adds resolved `compute_slots`
  provenance snapshots.
- Fixed duplicate `impls/networks/__init__.py` imports/exports.
- M8 foundation tests pass `5/5`; full HIQL/CRL/CoGHP regression passes
  `46/46`. A real shared-runtime provenance smoke confirmed JSON snapshots in
  runtime and checkpoint metadata. No stateful model, HRM, new credit policy,
  or long training was introduced.

## Vanilla CoGHP migration (M7)

- Official source audited from `wlsdn9350/CoGHP` `main` commit
  `8f362e9f86bf97fdbc9ce36d1b7b73b024e18b36`.
- Vanilla `CoGHPAgent`, official `MixerBlock`/
  `HierarchicalPolicyNetwork`, `MultiHGCDataset`, registry, evaluation,
  logging, and checkpoint path are integrated into the shared RLC runtime.
- The single physical `actor_mixer` core is preserved and reused for every
  autoregressive subgoal/action token; high/low heads remain independent.
- Synthetic official/RLC forward, loss, and update parity is exact outside
  the pre-existing `grad/norm` aggregation diagnostic.
- Real `antmaze-medium-navigate-v0` `MultiHGCDataset` N=20 required-field
  parity passes. Matched updates remain within float32 reduction noise:
  loss `7.63e-6`, parameters `1.49e-8`, target `2.91e-11`, optimizer
  state `1.19e-7`, RNG `0.0`.
- `impls/computation/` was intentionally not modified; vanilla CoGHP does
  not accept `--computation`.
- GPU vanilla CoGHP smoke: 1,000 real-data CUDA steps, 10 finite train/validation
  logging rows, evaluation, checkpoint save/load probe, and no NaN/Inf.
- M7 CPU regression: `41/41` tests passed. Vanilla CoGHP migration is
  **integration validated locally**; long training remains out of scope.
