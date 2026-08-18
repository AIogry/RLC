# HIQL actor computation parity

This document records the HIQL actor migration boundary completed before the
value migration.  It covers only the actor MLP body under the
`MLP -> FeedForward -> Direct` computation configuration.  Current value-slot
architecture and parity are documented separately in
`docs/hiql_value_computation_design.md`.

## Parameter-tree audit

With the small parity configuration, the actor leaves are:

| semantic component | legacy path | computation path |
| --- | --- | --- |
| low body | `modules_low_actor/actor_net/Dense_*` | `modules_low_actor/actor_net/topology/primitive/Dense_*` |
| low mean readout | `modules_low_actor/mean_net/*` | unchanged |
| high body | `modules_high_actor/actor_net/Dense_*` | `modules_high_actor/actor_net/topology/primitive/Dense_*` |
| high mean readout | `modules_high_actor/mean_net/*` | unchanged |

The concrete leaves in this configuration are `Dense_0/{kernel,bias}` and
`Dense_1/{kernel,bias}` for each body, plus `mean_net/{kernel,bias}` for each
readout.  `const_std=True` means there is no trainable `log_stds` leaf.  If a
state-dependent or learned standard deviation option is enabled, its readout
is mapped as another actor-head component and is not placed inside the body.

The actor computation wrapper contributes no trainable leaves.  It only
inserts the `topology/primitive` scope around the existing MLP parameters.
This actor-only document does not describe the later value wrapper; see the
value design document for its separate body/readout boundary.

## Why tests use a semantic graft

Constructing legacy and computation agents with the same seed does not make
all corresponding body arrays equal.  Flax parameter initialization folds
the parameter scope into the RNG; moving a kernel below
`actor_net/topology/primitive` therefore changes its initialization key.  In
the current CPU check, body kernels differ while the actor readout and body
biases happen to match.  The latter is an implementation detail and is not a
contract.

The parity helper in
`tests/computation/test_mlp_parity.py` maps parameters by semantic groups:

- `(<actor>, body, ...)` maps legacy `actor_net` to computation
  `actor_net/topology/primitive`;
- `(<actor>, mean_net, ...)` and any other readout keep their direct module
  names;
- all non-actor modules are copied unchanged, including the legacy value
  networks.

The helper also preserves the parameter-container type used by the existing
HIQL/Optax state.  It is test-only and is not part of checkpoint loading or
runtime training code.

## Gradient expectations

The direct low and high actor losses use the legacy value network as a fixed
weight source.  With the current configuration, low actor representation
gradients are explicitly stopped and the high target representation is
evaluated without `grad_params`.  Consequently, the actor loss gradient is
nonzero only in the selected actor's complete trainable subtree (body plus
readout); disconnected value, target-value, goal-representation, and other
actor leaves are expected to be zero.  The test helper also accepts matching
`None` leaves so a future differentiation path can represent disconnected
parameters that way.

## Covered parity surface

All comparisons use `rtol=0` and `atol=1e-6` unless the primitive reference is
checked with exact array equality.

- primitive MLP forward parity against the OGBench reference;
- FeedForward output shape and no-fake-state contract;
- generic continuous `GCActor` mode, scale, and `log_prob` parity;
- low actor loss/info and full actor-subtree gradient parity;
- true `HIQLAgent.high_actor_loss` distribution, target log-probability,
  loss/info, and full actor-subtree gradient parity;
- integrated total loss with both actors enabled and value still legacy;
- one real `HIQLAgent.update` using the same semantic initial parameters,
  optimizer, batch, and RNG, including actor parameter deltas;
- deterministic 10-step and 20-step update regressions, comparing each
  total/high/low/value loss-info record and final actor deltas.

The integrated and update tests use synthetic batches containing the required
HIQL fields: observations, next observations, value goals, rewards, masks,
low actor goals, actions, high actor goals, and high actor targets.

The current CPU run passed all 12 unittest cases.  The independently collected
maximum absolute errors were:

| check | result | max absolute error |
| --- | --- | ---: |
| primitive reference MLP | pass | 0.0 |
| generic GCActor distribution | pass | 0.0 |
| low actor loss/info/gradient | pass | 0.0 |
| high actor distribution/loss/info/gradient | pass | 0.0 |
| integrated total loss/info | pass | 0.0 |
| one optimizer update, info/semantic params | pass | 0.0 |
| deterministic N=10 regression | pass | 0.0 |
| deterministic N=20 regression | pass | 0.0 |

The test assertions use the documented `atol=1e-6` bound; the zero values
above are the observed CPU results after semantic parameter grafting.

## Checkpoint compatibility

The computation mode is not raw-path compatible with a legacy actor
checkpoint because the body scope is nested one level deeper.  A checkpoint
conversion must perform the same semantic remap for both actor slots:

```text
modules_low_actor/actor_net/*
  -> modules_low_actor/actor_net/topology/primitive/*
modules_high_actor/actor_net/*
  -> modules_high_actor/actor_net/topology/primitive/*
```

Readouts can be copied at their existing paths.  The reverse remap is needed
when loading a computation checkpoint into legacy mode.  No value checkpoint
conversion is included or implied by this actor migration.

## CPU validation command

From `RLC/`:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

The suite is intentionally a short deterministic parity suite; it does not
run a long training job and does not enable value computation migration.
