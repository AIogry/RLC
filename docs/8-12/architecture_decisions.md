# Architecture decisions

## ADR: HIQL value is a configurable computation slot

The online HIQL value estimator accepts the shared `compute.value` contract:
`MLP + FeedForward + Direct` in this milestone.  The value body is the
representation-transforming computation; the scalar value readout remains in
`GCValue`.

## ADR: target value mirrors online value architecture

There is no `compute.target_value` slot.  `target_value` is automatically
constructed with the same value computation specification and is updated by
the existing Polyak target-update path.

## ADR: value ensemble members remain independent

The baseline ensemble has two estimators.  Computation migration applies
`nn.vmap` with independent parameter axes to the body and scalar readout, so
member 0 and member 1 do not share computation parameters.

## ADR: goal representation remains outside the value slot

HIQL `goal_rep` is shared and has independent algorithmic representation
semantics.  It remains legacy in this migration and is not included in
`compute.value`.

## ADR: scalar value semantics remain controlled by GCValue

The legacy final `Dense(1)` behavior, initializer, bias, no-final-activation
semantics, and `squeeze(-1)` remain in `GCValue`.  The computation body uses
`activate_final=True` so the split is numerically equivalent to the legacy
MLP whose final scalar Dense was inside the MLP helper.

## ADR: parameter mapping is test-only

Legacy/computation scope differences are handled by semantic parameter grafts
inside parity tests only.  Production code does not add a checkpoint remap
layer or depend on generated names such as `Dense_0`.

## ADR: RLC/ogbench remains canonical

`RLC/ogbench` is the environment and benchmark implementation used by the
runtime. The upstream OGBench package is a reference for algorithm-side
training flow only and must not overwrite the canonical RLC package.

## ADR: upstream impls is a training reference, not a package to copy wholesale

Only the minimum HIQL runtime closure is migrated from
`offline_rl_baselines/ogbench/impls`: Dataset wrappers, environment helper,
evaluation, checkpoint, logging, and main. Unrelated agents and
CoGHP-specific `MultiHGCDataset` remain out of scope.

## ADR: OGBench sampling semantics and fixed RNG semantics are separate concerns

The Dataset/GCDataset/HGCDataset equations and HIQL field names follow the
OGBench reference. The explicit per-dataset and per-episode RNG streams from
the validated CoGHP fixed runtime take precedence over upstream global RNG
state. This preserves sampling distributions while making same-seed runs
reproducible.

## ADR: HIQL is the first end-to-end vertical slice

HIQL is the first agent wired through RLC environment loading, dataset
sampling, computation slots, update, evaluation, logging, and checkpoint.
This milestone does not imply that CRL, CoGHP, or other agents are integrated.

## ADR: integration validation is a separate gate

The gate is now met for the HIQL vertical slice: real Dataset tests, strict
N=20 real-batch parity, the 25-test CPU suite, and paired 1,000-step native
legacy/computation trainer smokes all pass. The result is integration
validated locally, but it is not a formal long-run baseline reproduction.
