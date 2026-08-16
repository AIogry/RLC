# HIQL value computation design and audit

This document defines the M2 value migration boundary.  The only configurable
research slot is `compute.value`; `target_value` is its automatic shadow.  The
current migration supports only `MLP + FeedForward + Direct` and does not
change HIQL's objective, goal representation, target update, or ensemble
semantics.

## Source audit

The audit covered the local RLC implementation and the reference implementation
at `offline_rl_baselines/ogbench/impls/{agents/hiql.py,utils/networks.py,utils/encoders.py}`.

### Current GCValue path

For the state-based HIQL configuration, the production call path is:

```text
HIQLAgent.create
  -> goal_rep_def = MLP(value_hidden_dims + (rep_dim,), activate_final=False)
                    + LengthNormalize
  -> value_encoder_def = GCEncoder(
         state_encoder=Identity(),
         concat_encoder=goal_rep_def,
     )
  -> GCValue(value_hidden_dims, ensemble=True, gc_encoder=value_encoder_def)
  -> GCEncoder(observations, goals)
       = concatenate(Identity(observations), goal_rep([observations, goals]))
  -> value_net(concatenated_input).squeeze(-1)
```

For pixel configurations, `state_encoder=encoder_module()` is used in addition
to the same goal representation path.  The value computation slot does not
include either encoder or `goal_rep`.

`GCValue` currently constructs:

```python
MLP((*hidden_dims, 1), activate_final=False, layer_norm=layer_norm)
```

The exact layer order is therefore:

```text
Dense(hidden_0) -> GELU -> LayerNorm
Dense(hidden_1) -> GELU -> LayerNorm
...
Dense(1)
-> squeeze(-1)
```

The final `Dense(1)` has no activation and no LayerNorm after it.  The MLP
primitive uses OGBench's `variance_scaling(scale=1.0, mode='fan_avg',
distribution='uniform')` initializer, default float dtype, and default Dense
bias.  The migration preserves these details by using a computation body with
`activate_final=True` and leaving the scalar readout as a separate Dense:

```text
hidden MLP body: MLP(hidden_dims, activate_final=True, layer_norm=layer_norm)
                 -> ComputationCore(FeedForward(...))
scalar readout: Dense(1, same default initializer/bias semantics)
output: squeeze(-1)
```

This split is numerically equivalent to the legacy MLP because every hidden
layer remains followed by the same GELU and optional LayerNorm, while the
former final Dense(1) is now the explicit readout and remains unactivated.
The scalar readout is task-specific value semantics and is not part of the
replaceable computation body.

### Online and target value

`HIQLAgent.create` constructs independent `GCValue` definitions for `value` and
`target_value`.  After initialization, the target parameter subtree is copied
from the online value subtree.  During training, `HIQLAgent.target_update`
executes the production Polyak path:

```python
new_target = online * tau + old_target * (1 - tau)
```

The migration keeps that code path and makes target architecture selection
automatic: if `compute.value` resolves to a `ComputationSpec`, both online
`value` and `target_value` receive that same spec.  There is deliberately no
`compute.target_value` configuration and no valid online/target architecture
mismatch.

### Ensemble semantics

`GCValue(ensemble=True)` uses the local `ensemblize` helper, equivalent to:

```python
nn.vmap(
    module,
    variable_axes={'params': 0},
    split_rngs={'params': True},
    in_axes=None,
    out_axes=0,
    axis_size=2,
)
```

The ensemble size is exactly two.  Every Dense kernel/bias and every LayerNorm
scale/bias has a leading axis of size two.  Member 0 and member 1 have
independent parameter slices and independent initialization RNGs.  The value
migration keeps this mapping around the complete computation body and scalar
readout; it does not create one shared core with two heads.

The body ensemble receives the same input in each member (`in_axes=None`) and
returns a leading member axis.  The separate scalar readout is vmapped with
`in_axes=0` so each readout consumes only its corresponding body slice.

For the small parity configuration (`value_hidden_dims=(6, 6)`), the legacy
value subtree is:

```text
modules_value/value_net/Dense_0/{kernel,bias}       shape (2, 7, 6)/(2, 6)
modules_value/value_net/LayerNorm_0/{scale,bias}    shape (2, 6)/(2, 6)
modules_value/value_net/Dense_1/{kernel,bias}       shape (2, 6, 6)/(2, 6)
modules_value/value_net/LayerNorm_1/{scale,bias}    shape (2, 6)/(2, 6)
modules_value/value_net/Dense_2/{kernel,bias}       shape (2, 6, 1)/(2, 1)
```

The target tree has the same structure under `modules_target_value`.

The migrated online tree uses the corresponding paths:

```text
modules_value/value_net/core/topology/primitive/Dense_{0,1}/*
modules_value/value_net/core/topology/primitive/LayerNorm_{0,1}/*
modules_value/value_readout/{kernel,bias}
```

`modules_target_value` mirrors these paths exactly.  The `core` and
`topology` nodes are structural only and contribute no parameters.

## Computation boundary

The scientific boundary is the representation-transforming value body only:

```text
(state, raw goal)
  -> legacy GCEncoder and legacy goal_rep
  -> computation value body
  -> legacy-compatible scalar Dense(1) readout
  -> squeeze(-1)
```

`goal_rep` is intentionally not computationized.  It is shared by high actor,
low actor, and value paths and has independent algorithmic representation
semantics.  Migrating it in this task would change two scientific variables at
once.  The value loss, expectile weighting, advantage definition, AWR losses,
and Polyak coefficient are also outside the boundary.

## Parameter-tree and checkpoint implications

The legacy value body and migrated value body have different raw scopes.  The
exact computation wrapper currently inserts `core/topology/primitive` below
`value_net`, while the scalar readout is a separate `value_readout` module.  A
test-only semantic mapping therefore labels parameters as `body/...` and
`readout/...`, and maps both ensemble slices by their leading axis.  It is not
part of production architecture or runtime checkpoint loading.

Because the scope changes, same-seed raw body kernels are not required to be
tree-identical.  A legacy checkpoint conversion must map the legacy hidden
body leaves into the computation primitive subtree and map the former final
`Dense_2` leaves into the scalar readout.  Target conversion applies the same
mapping under `modules_target_value`.  Readout and body parameter counts remain
equal; wrapper nodes contribute zero trainable parameters.

In the current CPU same-seed audit, semantic body kernel errors are about
`1.316` and `1.269` for the two hidden kernels, and the scalar readout kernel
error is about `1.200`; biases and LayerNorm parameters happen to match.  This
is expected Flax scope/RNG behavior, not a model mismatch.  All parity tests
graft legacy semantic parameters before comparing numerical behavior.

## Architecture decisions

- **ADR: HIQL value is a configurable computation slot.** The slot is named
  `value` and accepts the shared computation factory contract.
- **ADR: target_value is not an independent computation slot.** It mirrors the
  online value architecture and remains controlled by the existing target
  update path.
- **ADR: ensemble members preserve baseline-independent parameters.** The two
  value estimators keep independent body/readout parameter slices.
- **ADR: goal_rep is outside this migration.** Its legacy shared representation
  semantics remain unchanged.
- **ADR: scalar value semantics remain controlled by GCValue.** The final
  scalar Dense readout and squeeze are retained outside the computation core,
  while their operation order and initialization match the baseline.
- **ADR: semantic parameter mapping is test-only.** Production code uses the
  actual Flax module tree and does not add a compatibility remap layer.

## Explicit non-goals

This migration does not add HRM, recurrence, MLP-Mixer, SwiGLU/RMSNorm,
learned schedules, shared ensemble cores, a target computation config, goal
representation computation, value-objective changes, or long baseline
training.

## Validation record

The CPU parity suite currently reports:

| check | result | max absolute error |
| --- | --- | ---: |
| value forward, member 1 | PASS | 0.0 |
| value forward, member 2 | PASS | 0.0 |
| value loss and diagnostics | PASS | 0.0 |
| full value mapped gradient | PASS | 0.0 |
| target forward | PASS | 0.0 |
| Polyak target update | PASS | 0.0 |
| value one-step update | PASS | 0.0 |
| value N=10 | PASS | 0.0 |
| value N=20 | PASS | 0.0 |
| full HIQL total loss/gradient | PASS | 0.0 |
| full HIQL one-step update | PASS | 0.0 |
| full HIQL N=10/N=20 | PASS | 0.0 |
| parameter count | PASS | 0.0 |

For the small configuration, both legacy and computationized value modules
contain 242 parameters, and the full HIQL agent contains 846 parameters.  The
slot counts are value 242, high actor 117, and low actor 104 in both modes.

There is one non-semantic diagnostic caveat: `TrainState.apply_loss_fn`
aggregates `grad/norm` in raw pytree order.  The wrapper changes that order, so
the aggregate can differ by about `6.1e-5` even though every mapped gradient
leaf and every optimizer/target parameter delta is exactly equal.  Core parity
assertions retain `rtol=0`, `atol=1e-6` and exclude only this reduction-order
diagnostic.

The available RLC runtime also passes a three-step synthetic JIT/update,
action/evaluation-path, and temporary checkpoint save/load smoke.  A real
OGBench dataset/trainer/evaluation smoke is not run because the local
`impls/main.py`, dataset, evaluation, and checkpoint utility files remain
scaffolds.
