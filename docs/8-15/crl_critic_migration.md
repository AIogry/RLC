# M5 — CRL critic computationization

Date: 2026-08-13

## Scope

M5 computationizes the two representation branches of the CRL bilinear
critic:

```text
critic_state -> phi(s, a)
critic_goal  -> psi(g)
```

Each branch uses the shared `ComputationSpec` factory with
`MLP + FeedForward + Direct`. The two specifications are independent API
slots and instantiate independent parameters. The AWR bilinear value remains
legacy in this milestone.

## Call paths

Legacy path:

```text
CRLAgent.create
  -> GCBilinearValue / GCDiscreteBilinearCritic
  -> phi = legacy MLP(state/action)
  -> psi = legacy MLP(goal)
  -> bilinear dot product / contrastive CRL loss
```

Computation path:

```text
CRLAgent.create
  -> resolve_slot_spec('critic_state'/'critic_goal')
  -> GCBilinearValue / GCDiscreteBilinearCritic
  -> phi = ComputationCore(MLP + FeedForward + Direct)(state/action)
  -> psi = ComputationCore(MLP + FeedForward + Direct)(goal)
  -> same bilinear dot product / contrastive CRL loss
```

The network still owns input construction, `phi`/`psi` outputs, ensemble
semantics, normalization, exponential option, and the
`/ sqrt(latent_dim)` bilinear score. No contrastive objective or runtime path
was changed.

## Validation

- Synthetic forward parity covers each ensemble member's `phi`, `psi`, and
  final value/logits.
- Both `ddpgbc` and `awr` cover critic loss, actor loss, total loss,
  contrastive diagnostics, semantic gradients, one-step updates, optimizer
  state, and agent RNG.
- Slot wiring initializes `ON/OFF`, `OFF/ON`, and `ON/ON` combinations.
- State and goal computation parameters are independent.
- Legacy and computationized trainable parameter counts are identical.
- CPU regression: `32/32` tests passed.
- Real `antmaze-medium-navigate-v0` `GCDataset` strict N=20 parity:
  `first_divergence=None`, all tracked maximum absolute errors `0.0`.
- GPU 1000-step trainer smoke passed for legacy and all-three-slot
  computation CRL. Final logged losses were `2.308672` and `2.302007`; both
  runs had finite updates, evaluation, logging, and checkpoint save/load
  probes. Short-run success was `0.0` for both and is not a research claim.

## Parameter accounting

For `obs_dim=29`, `action_dim=8`, hidden dimensions `(6, 6)`, and latent
dimension `3`:

| Component | Parameters |
|---|---:|
| CRL total | 1616 |
| actor | 452 |
| critic total | 1164 |
| critic state core | 630 |
| critic goal core | 534 |
| bilinear/readout | 0 |

The 0 bilinear/readout count is expected: CRL's bilinear interaction is a
parameter-free dot product, not a trainable Dense readout.

## Status

CRL critic computationization is **integration validated locally**. AWR value
branches, new computation primitives, recurrent critics, and long baseline
training remain out of scope.
