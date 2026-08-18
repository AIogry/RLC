# M6 — CRL AWR value computationization

Date: 2026-08-13

## Scope

M6 computationizes the independent AWR value network used only by
`actor_loss='awr'`:

```text
value_state -> phi_V(s)
value_goal  -> psi_V(g)
```

Both branches use the existing `MLP + FeedForward + Direct` factory path and
have independent parameters. The value network remains
`GCBilinearValue(ensemble=False)`. The critic slots and value slots are four
separate representation groups.

## Semantics and call paths

Legacy value path:

```text
CRLAgent.create (actor_loss=awr)
  -> GCBilinearValue(ensemble=False)
  -> phi_V = legacy MLP(state)
  -> psi_V = legacy MLP(goal)
  -> V = phi_V^T psi_V / sqrt(latent_dim)
```

Computation value path:

```text
CRLAgent.create (actor_loss=awr)
  -> resolve_slot_spec('value_state'/'value_goal')
  -> GCBilinearValue(ensemble=False)
  -> phi_V = ComputationCore(MLP + FeedForward + Direct)(state)
  -> psi_V = ComputationCore(MLP + FeedForward + Direct)(goal)
  -> same bilinear V
```

The network retains input construction, latent dimension, dot product,
`sqrt(latent_dim)` scaling, `ensemble=False`, and value contrastive loss. The
agent retains the reference AWR equations:

```text
Q = min(Q1, Q2)
A = Q - V
weight = min(exp(alpha * A), 100)
actor_loss = -(weight * log_prob).mean()
```

The bilinear interaction remains outside ComputationCore. No objective,
dataset, or runtime semantics were changed.

## Conditional instantiation

The public schema contains five independent CRL slots:

```yaml
compute:
  actor: ...
  critic_state: ...
  critic_goal: ...
  value_state: ...
  value_goal: ...
```

`value_state` and `value_goal` are read only while constructing the AWR value
module. DDPG+BC does not instantiate `modules_value`, even if its value slot
entries are enabled. The shared `--computation` flag enables all five slots
for AWR and only actor/critic slots for DDPG+BC.

## Validation

- Isolated value forward parity: `phi_V`, `psi_V`, and `V(s,g)` exact.
- Production contrastive value loss and diagnostics exact.
- `Q1`, `Q2`, `V`, advantage, and clipped AWR weight exact.
- AWR actor loss, semantic gradients, one-step deltas, optimizer state, and
  agent RNG exact.
- Full five-slot AWR synthetic parity passes.
- Value remains `ensemble=False`; critic/value branch parameters are all
  independent.
- CPU regression: `38/38` tests passed.
- Real `antmaze-medium-navigate-v0` `GCDataset` N=20 full AWR strict parity:
  `first_divergence=None`, all tracked maximum absolute errors `0.0`.
- GPU 1000-step AWR trainer smoke passed for native legacy and full
  computation modes. Final aggregate logged losses were `9.641608` and
  `9.450994`; both had finite critic/value/actor diagnostics, evaluation,
  logging, and checkpoint save/load probes. Short-run success was `0.0` for
  both and is not a research conclusion.

The final training CSV rows were:

| Mode | Critic loss | Value loss | Actor loss | Total loss | Success |
|---|---:|---:|---:|---:|---:|
| Legacy AWR | 0.384524 | 0.383256 | 8.873828 | 9.641608 | 0.0 |
| Full computation AWR | 0.377802 | 0.407084 | 8.666108 | 9.450995 | 0.0 |

## Parameter accounting

For `obs_dim=29`, `action_dim=8`, hidden dimensions `(6, 6)`, and latent
dimension `3`:

| Component | Parameters |
|---|---:|
| DDPG+BC total | 1616 |
| AWR total | 2150 |
| actor | 452 |
| critic total | 1164 |
| critic state core | 630 |
| critic goal core | 534 |
| AWR value total | 534 |
| value state core | 267 |
| value goal core | 267 |
| bilinear/readout parameters | 0 |

Legacy and computationized counts match in both DDPG+BC and AWR modes.

## Status

CRL AWR value computationization and the full computationized AWR-CRL baseline
are **integration validated locally**. Long training and new computation
primitives/topologies remain out of scope.
