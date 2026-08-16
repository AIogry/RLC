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

Only the minimum required runtime closures are migrated from the external
references: Dataset wrappers, environment helper, evaluation, checkpoint,
logging, main, and the explicitly audited vanilla CoGHP agent/network/data
path. Unrelated agents and un-audited variants are not copied wholesale.

## ADR: OGBench sampling semantics and fixed RNG semantics are separate concerns

The Dataset/GCDataset/HGCDataset equations and HIQL field names follow the
OGBench reference. The explicit per-dataset and per-episode RNG streams from
the validated CoGHP fixed runtime take precedence over upstream global RNG
state. This preserves sampling distributions while making same-seed runs
reproducible.

## ADR: HIQL is the first end-to-end vertical slice

HIQL was the first agent wired through RLC environment loading, dataset
sampling, computation slots, update, evaluation, logging, and checkpoint.
CRL and vanilla CoGHP now reuse the same shared runtime through their own
validated migration gates.

## ADR: integration validation is a separate gate

The gate is now met for the HIQL vertical slice: real Dataset tests, strict
N=20 real-batch parity, the 25-test CPU suite, and paired 1,000-step native
legacy/computation trainer smokes all pass. The result is integration
validated locally, but it is not a formal long-run baseline reproduction.

## ADR: CRL critic computation boundary

CRL keeps the bilinear critic boundary explicit. `compute.critic_state` owns
only the state/action representation `phi(s, a)` and `compute.critic_goal`
owns only the goal representation `psi(g)`. Equal specifications do not
imply parameter sharing: each slot constructs its own `ComputationCore`.
The bilinear dot product, ensemble aggregation, normalization, contrastive
objective, and actor interaction remain in the CRL-specific network/agent.
AWR value branches are not merged with these slots.

## ADR: CRL critic computationization is integration-gated

The M5 gate requires synthetic forward/loss/gradient/update parity, all slot
wiring combinations, real antmaze-medium N=20 strict parity, matching
parameter counts, and paired GPU 1000-step trainer smokes. M5 meets this local
integration gate; it is not a long-run baseline reproduction.

## ADR: CRL AWR value uses independent computation slots

The AWR value network exposes `compute.value_state` for `phi_V(s)` and
`compute.value_goal` for `psi_V(g)`. They use independent ComputationCore
instances and remain separate from `critic_state` and `critic_goal`; equal
specifications never imply parameter sharing.

## ADR: AWR value remains ensemble=False

CRL AWR value construction preserves the reference
`GCBilinearValue(ensemble=False)` behavior. Only its representation bodies are
wrapped; the bilinear dot product, latent scaling, contrastive value loss, and
`A=min(Q1,Q2)-V` actor weighting remain unchanged.

## ADR: AWR value is conditional on actor_loss

The value module and its trainable parameters are instantiated only when
`actor_loss='awr'`. DDPG+BC may carry the five-slot configuration schema, but
it does not create `modules_value` or add value parameters. The shared
`--computation` flag enables value slots only for AWR.

## ADR: CRL AWR value computationization is integration-gated

M6 requires isolated value parity, full five-slot AWR parity, independent
branch checks, real antmaze-medium N=20 strict parity, parameter accounting,
and paired GPU 1000-step AWR smokes. M6 meets this local integration gate; it
is not a long-run baseline reproduction.

## ADR: CRL actor computation boundary

CRL uses the shared `compute.actor` slot. Only the actor representation body
is replaced by `ComputationCore`; the distribution/readout, actor loss, and
critic-to-actor interaction remain legacy. The critic branches are the
independent `compute.critic_state` and `compute.critic_goal` slots above.

## ADR: CRL shares the HIQL runtime

CRL is selected through the same agent registry and `impls.main` training
loop. It reuses RLC OGBench, Dataset/GCDataset, evaluation, logging,
checkpoint, and explicit RNG streams; it does not create a second trainer.

## ADR: Vanilla CoGHP follows the official Mixer architecture

Vanilla CoGHP is migrated from `wlsdn9350/CoGHP` `main` without changing its
algorithmic roles. `goal_rep`, `value`, `target_value`, and `actor_mixer`
remain the production module names. The actor mixer owns one physical list of
MixerBlocks reused by every autoregressive token step, including both
subgoal generation and final action generation. High and low actor heads are
independent readouts only; high and low do not receive separate Mixer cores.

## ADR: Vanilla CoGHP is outside the computation-slot system

The official MixerBlock is not reinterpreted as a computation primitive or
topology. M7 does not modify `impls/computation/`, add HRM/recurrence, or
enable `--computation` for CoGHP. CoGHP uses the shared RLC trainer and
checkpoint/evaluation interfaces while preserving the official Mixer and
loss semantics.

## ADR: MixerBlock remains in the vanilla CoGHP network for now

The original official `impls/utils/coghp_network.py` is classified in RLC as
`impls/networks/coghp.py`, which is the more appropriate current layer for an
algorithm-specific network. `MixerBlock` remains there temporarily because
it is the reference-faithful implementation of the official vanilla CoGHP
baseline. If MLP-Mixer is later formally admitted to the unified computation
framework, this block will likely move into a more general primitive/block
layer under `impls/computation/`; M7 cleanup does not move it early.

## ADR: MultiHGCDataset preserves official fields and RLC RNG

CoGHP uses the official `MultiHGCDataset` output keys and trajectory boundary
equations, including `low_actor_goals`, `high_actor_goals`, and
`high_actor_targets` with `num_subgoals`. The RLC explicit NumPy Generator
contract is retained as the injection mechanism; it does not change the
sampling formulas or field semantics.
