# CRL migration and computationization

## Scope

The CRL migration uses
`offline_rl_baselines/ogbench/impls/agents/crl.py` as the algorithm reference
and keeps `RLC/ogbench` as the canonical benchmark/environment package.

The migrated agent supports the reference branches:

- `actor_loss='ddpgbc'`;
- `actor_loss='awr'`.

CRL uses the existing `GCDataset` path and the reference fields
`observations`, `value_goals`, `actor_goals`, and `actions`.

## Algorithm boundary

The migrated CRL algorithm preserves:

- ensemble bilinear critic `Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d)`;
- non-ensemble bilinear value used by AWR;
- contrastive binary/categorical objectives and diagnostic statistics;
- DDPG+BC Q normalization and behavior-cloning term;
- AWR advantage weighting and log-probability term;
- actor action clipping, temperature, distribution, and `sample_actions`;
- reference initialization, optimizer, update, and agent RNG semantics.

The required network migration is limited to `GCBilinearValue` and
`GCDiscreteBilinearCritic` in `impls/networks/common.py`. The old OGBench
`utils/networks.py` is not copied wholesale.

## Computation boundary

The CRL actor representation body and the two bilinear critic representation
branches are configurable independently:

```yaml
compute:
  actor:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
  critic_state:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
  critic_goal:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
  value_state:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
  value_goal:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
```

With the slot disabled, CRL uses the legacy `GCActor` or `GCDiscreteActor`.
With the slot enabled, only `actor_net` is wrapped by the existing
`ComputationCore`; the action readout, standard deviation, distribution,
encoder semantics, and actor/critic interaction remain unchanged.

`critic_state` replaces only the state/action branch that produces `phi(s, a)`;
`critic_goal` replaces only the goal branch that produces `psi(g)`. Both use
the same `MLP + FeedForward + Direct` specification by default, but they are
separate Flax submodules and do not share parameters. The bilinear interaction,
ensemble axis, contrastive logits, normalization, and `sqrt(latent_dim)` scale
remain in `GCBilinearValue`/`GCDiscreteBilinearCritic`.

In AWR mode, `value_state` and `value_goal` independently replace the two
branches of the separate `GCBilinearValue(ensemble=False)` value network. They
are never merged with the critic slots and do not share parameters with
`critic_state` or `critic_goal`. In DDPG+BC mode, the value module is not
instantiated at all, so these configuration entries cannot add parameters.

## Runtime path

CRL is registered in the same `agents` registry and uses the same main entry
point as HIQL:

```text
RLC/ogbench
  -> Dataset / GCDataset
  -> CRLAgent
  -> actor ComputationCore (optional)
  -> critic_state/critic_goal ComputationCore (optional)
  -> value_state/value_goal ComputationCore in AWR (optional)
  -> bilinear critic/value semantics
  -> CRL update
  -> shared evaluation
  -> shared logging/checkpoint
```

Select it with `--agent=crl`; select actor computation with `--computation`.
For CRL, `--computation` enables the actor and critic slots in both branches;
it additionally enables the two value slots only when `actor_loss=awr`.

## Validation

### Reference migration

On fixed synthetic batches, both `ddpgbc` and `awr` pass:

- critic output/logits;
- critic and actor losses;
- total loss;
- full gradient;
- one optimizer update;
- agent RNG update.

Reference parameter trees are identical for the tested small configuration.
The aggregate `grad/norm` is excluded from exact assertions because raw
optimizer pytree leaf ordering is a diagnostic reduction detail.

### Actor computation parity

After test-only semantic actor parameter grafting, legacy CRL and
`MLP + FeedForward + Direct` actor CRL pass for both actor-loss branches:

- actor mean/mode, standard deviation, and log-probability;
- actor loss and full total loss;
- semantic full gradient;
- one optimizer update;
- critic parameters remain unchanged in structure;
- agent RNG.

Legacy and computation actor trainable parameter counts are equal.

### Critic computation parity

After test-only semantic parameter grafting, legacy CRL and
`critic_state = MLP + FeedForward + Direct`,
`critic_goal = MLP + FeedForward + Direct` pass for both `ddpgbc` and `awr`:

- each ensemble member's `phi`, `psi`, and final bilinear value/logits;
- critic loss, actor loss, total loss, and contrastive diagnostics;
- all semantic gradients, including both critic branches and actor;
- one real `agent.update(batch)` and optimizer-state deltas;
- agent RNG equality.

The slot wiring tests also initialize `state ON / goal OFF`,
`state OFF / goal ON`, and `state ON / goal ON`. The two enabled cores have
independent parameter subtrees. No trainable parameter is added by the
computation wrapper.

### AWR value computation parity

The separate AWR value network remains `ensemble=False`. After semantic
parameter grafting, legacy and computationized `value_state`/`value_goal`
branches pass:

- `phi_V`, `psi_V`, and `V(s,g)` forward outputs;
- production contrastive value loss and diagnostics;
- `Q1`, `Q2`, `V`, `A=min(Q1,Q2)-V`, and
  `min(exp(alpha*A), 100)` AWR weights;
- AWR actor loss and all semantic gradients;
- one real update, parameter deltas, optimizer state, and agent RNG.

The value and critic branches are four independent parameter groups. DDPG+BC
does not instantiate `modules_value` even when the value slot configuration is
enabled.

### Real data

On `antmaze-medium-navigate-v0`, real `GCDataset` N=20 strict parity passes
with the same batch sequence, semantic initial parameters, optimizer state,
and agent RNG. Critic loss, actor loss, total loss, critic-state parameters,
critic-goal parameters, actor parameters, optimizer state, and agent RNG have
maximum absolute error `0.0`; there is no first divergence step. N=100 was not
run because N=20 passed and the requested short trainer smoke completed.

### Parameter accounting

For the small smoke configuration (`obs_dim=29`, `action_dim=8`, hidden
dimensions `(6, 6)`, latent dimension `3`):

| Component | Parameters |
|---|---:|
| CRL total | 1616 |
| actor | 452 |
| critic | 1164 |
| critic state core (`phi`) | 630 |
| critic goal core (`psi`) | 534 |
| bilinear/readout parameters | 0 |
| AWR value total | 534 |
| AWR value state core (`phi_V`) | 267 |
| AWR value goal core (`psi_V`) | 267 |
| actor body / computation core | 396 |
| actor readout and distribution parameters | 56 |

For DDPG+BC, legacy and computationized CRL both have 1616 total parameters.
For AWR, legacy and computationized CRL both have 2150 total parameters.

### GPU trainer smoke

Both native modes completed 1,000 steps on the real antmaze dataset using the
shared `impls.main` path:

- legacy CRL: final logged loss `2.308672`;
- actor + critic-state + critic-goal computation CRL: final logged loss
  `2.302007`;
- both had finite updates, evaluation, CSV logging, and checkpoint
  save/load action/value equality;
- task-1 success was `0.0` in both short smokes and is not interpreted as an
  algorithmic result.

For AWR, both native modes also completed 1,000 steps:

- legacy AWR CRL: final logged aggregate loss `9.641608`;
- full actor + critic-state + critic-goal + value-state + value-goal
  computation AWR CRL: final logged aggregate loss `9.450994`;
- both had finite critic/value/actor diagnostics, evaluation, CSV logging, and
  checkpoint save/load action/value equality;
- task-1 success was `0.0` in both short smokes and is not interpreted as an
  algorithmic result.

## Status

CRL baseline migration: **integration validated locally**.

CRL actor computationization: **integration validated locally**.

CRL critic computationization: **integration validated locally**.

This does not include recurrent/HRM critics, long training, multi-seed
experiments, or formal baseline reproduction.
