# M9B：Two-State Hierarchical Computation

日期：2026-08-16

## 结论与边界

M9B extends M9A's actor-only computation with two decision-local internal
states and an asymmetric H/L schedule. It preserves baseline actor width,
GELU MLP semantics, actor objective, network readout and dataset semantics.

This milestone implements and validates the topology, full-BPTT/one-step credit
axis, Study configuration and short GPU smoke. It does not start the formal
32-run M9B sweep or make a performance claim.

## 1. Source and HRM audit

The local RLC tree is the source of truth. At the start of this task it was
dirty because the uncommitted M9A implementation was retained; `HEAD` was
`bc6a16f433ea5b3fc9859f7b1edf4a8ca660850c`.

Local and official OGBench actor configs still agree on:

```text
HIQL high_actor: (512, 512, 512)
HIQL low_actor:  (512, 512, 512)
CRL actor:      (512, 512, 512)
```

Thus `state_dim=512` is justified by the RLC/baseline actor width, not by
copying HRM's hidden size. M9A's real actor audit remains: HIQL high
`D_in=58, output=10`, HIQL low `D_in=39, output=8`, and CRL `D_in=58, output=8`.

The HRM-mini audit confirmed the reference `H_cycles=2`, `L_cycles=6` schedule:
L receives `z_L+z_H+x` and H receives `z_H+z_L`. Its `bptt=False` behavior
motivates one-step credit. M9B deliberately removes HRM's Transformer,
Attention, SwiGLU, RMSNorm, bias-free CastedLinear, bfloat16 choice,
truncated-normal initialization and persistent environment-time carry.

## 2. TwoState topology

Production names are `TwoState` and `topology: two_state`; the class is in
[`impls/computation/topologies/two_state.py`](../impls/computation/topologies/two_state.py).
It is a computation ontology class, not an `HRM`-named network.

For raw actor input `x_raw`:

```text
x = E(x_raw)                         # Dense(512) + GELU
z_H = broadcast(z_h_init)
z_L = broadcast(z_l_init)
```

The two independent physical update modules are:

```text
h_update: Dense(512) + GELU + Dense(512) + GELU
l_update: Dense(512) + GELU + Dense(512) + GELU
```

The canonical equations are:

```text
z_L <- l_update(z_L + z_H + x)
z_H <- h_update(z_H + z_L)
```

Only L receives `x`; H never directly receives it. There is no outer residual.
The final actor representation is `z_H`; `(z_H, z_L)` is diagnostic local
state only. No state is accepted from or written to a later decision.

`z_h_init` and `z_l_init` are independent standard-normal non-trainable Flax
`buffers`, each shape `(512,)`. They are checkpointed/restored and excluded
from params and optimizer state. This differs from HRM-mini's approximate
truncated-normal buffers.

## 3. Schedule

| schedule | trace | L executions | H executions | total |
|---|---|---:|---:|---:|
| H2L1 | `L H L H` | 2 | 2 | 4 |
| H2L6 | `L L L L L L H L L L L L L H` | 12 | 2 | 14 |

In general, `N_L=H_cycles*l_cycles`, `N_H=H_cycles`, and
`N_update=H_cycles*(l_cycles+1)`. Only `(2,1)` and `(2,6)` are accepted.
H2L1 is the two-state four-update control; H2L6 is the HRM-inspired
hierarchical-timescale condition without HRM's primitive.

## 4. Credit axis

`credit: full_bptt` leaves every warm-up state connected to the final actor
loss. `credit: one_step` executes the identical forward trace, then immediately
before the final `L -> H` pair applies `jax.lax.stop_gradient` to accumulated
`z_H` and `z_L`:

```text
z_L* = l_update(sg(z_L) + sg(z_H) + x)
z_H* = h_update(sg(z_H) + z_L*)
```

The final H gradient therefore passes through the final L, but not through
earlier warm-up updates. Warm-up and final updates reuse the same H/L parameter
subtrees. Policies are in
[`impls/computation/credit/full_bptt.py`](../impls/computation/credit/full_bptt.py)
and [`impls/computation/credit/one_step.py`](../impls/computation/credit/one_step.py).

Tests verify forward equality and backward difference between the two policies,
and match both production gradients to hand-written JAX references.

## 5. Accounting

For CRL (`D_in=58`, action output `8`):

| quantity | value |
|---|---:|
| input mapping params | 30208 |
| H update params | 525312 |
| L update params | 525312 |
| core trainable params | 1080832 |
| full actor trainable params | 1084936 |
| buffer elements | 1024 |

HIQL high is `1085962` trainable parameters and HIQL low is `1075208`; each
has `1024` buffer elements. H2L1/H2L6 and full-BPTT/one-step have identical
parameter and buffer counts. HIQL high+low has `2048` buffer elements and two
independent TwoState parameter trees.

Runtime metadata under `actor_parameter_accounting` records topology, primitive,
credit, state dimension, initialization, H/L cycles, executions, trainable
parameters, core parameters and buffers for each actor slot.

## 6. Placements and Study

M9B uses only CRL actor, HIQL high actor, HIQL low actor and HIQL high+low actor.
CRL is fixed to `actor_loss=ddpgbc`; critic/value/goal-representation slots are
not studied, and HIQL value remains baseline. High+low has separate H/L modules
and separate H/L buffer pairs.

[`experiments/M9B_two_state/study.yaml`](../experiments/M9B_two_state/study.yaml)
contains 16 configurations: four placement families × H2L1/H2L6 ×
full_bptt/one_step. With two environments and seed 0, the manifest contains 32
planned runs. M9B references M9A-C001 and M9A-C002 as HIQL/CRL baselines:

```yaml
baseline_reference:
  study: M9A
  hiql: M9A-C001
  crl: M9A-C002
```

M9A remains 52 planned runs, M9B adds 32, and the combined planned inventory is
84 runs without duplicate baseline training.

## 7. Validation and formal boundary

Targeted TwoState and M9B integration tests pass `11/11`, covering schedule
traces, equations, buffer reproducibility/reset, H/L independence, parameter
sharing, full-BPTT gradients, one-step gradients, and the
forward-equal/backward-different invariant.

The complete CPU regression passes `76/76`. Five real-data 1-step GPU smokes
passed with `jax_backend=gpu`, finite loss/gradients, evaluation and checkpoint
action/value restore:

| config | placement | backend | loss |
|---|---|---|---:|
| M9B-C001 | CRL actor H2L1 full-BPTT | gpu | 2.734457 |
| M9B-C006 | HIQL high H2L1 one-step | gpu | 77.493813 |
| M9B-C011 | HIQL low H2L6 full-BPTT | gpu | 77.793938 |
| M9B-C016 | HIQL high+low H2L6 one-step | gpu | 77.483025 |
| M9B-C015 | HIQL high+low H2L6 full-BPTT | gpu | 77.483002 |

The five smoke metadata records include the expected H/L execution counts,
1024 buffers per TwoState actor, and 2048 buffers for high+low. The M9B
manifest contains 32 planned rows, all still `planned`; no formal M9A or M9B
scientific sweep was started.

Formal training still requires confirmation of train steps, batch size,
evaluation/checkpoint schedule, additional seeds, GPU allocation, retry policy
and statistical analysis. M9B reuses `tools/run.py` and `tools/sweep.py`; it has
no separate HRM scheduler. Short smoke values are not performance results.
