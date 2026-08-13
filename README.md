# RLC — RL Computation

RLC is a research platform for comparing computation modules and computation organizations in goal-conditioned reinforcement learning.

The project keeps the following boundary explicit:

```text
Agent       = learning algorithm
Network     = task-specific input/output semantics
Computation = representation transformation
Compute Slot = replaceable location for computation
```

`RLC/ogbench` is the canonical OGBench environment/runtime copied from the repaired CoGHP version. It must not be overwritten by the older reference runtime. The algorithm reference is `offline_rl_baselines/ogbench` in this workspace.

The first computation implementation is intentionally small: the original
OGBench MLP, a feed-forward topology, and direct backpropagation. HIQL
`high_actor`, `low_actor`, and `value` are parity-validated slots; their
distribution/readout, goal encoding, loss, target-update, and dataset
semantics remain unchanged. The first HIQL OGBench runtime vertical slice is
now integrated locally through real Dataset sampling, evaluation, logging,
checkpointing, and a short native trainer smoke.

## Current milestone — 2026-08-13

HIQL is the first RLC agent with a locally validated end-to-end runtime path.
The legacy and computationized variants both run through the same real OGBench
dataset, HIQL update, evaluation, logging, and checkpoint flow.

The controlled GPU diagnostic also completed 1,000 real-data strict-parity
updates after semantic parameter mapping:

- total/value/high-actor/low-actor losses: maximum error `0.0`;
- semantic online and target parameters: maximum error `0.0`;
- optimizer state and agent RNG: maximum error `0.0`;
- only the aggregate `grad/norm` diagnostic differed, by at most
  `6.1035e-05` due to pytree reduction order;
- native step-0 losses were already different, and native semantic initial
  parameters differed by up to `1.65875`.

This supports the conclusion that the earlier native 1000-step loss gap is
caused by different parameter-tree initialization paths, not by a computation
migration error. This status is local integration validation, not a formal
baseline reproduction.

CRL now has the same explicit computation boundary for its bilinear critic:
`compute.critic_state` produces `phi(s, a)` and `compute.critic_goal` produces
`psi(g)`. They use independent `MLP + FeedForward + Direct` cores, while the
bilinear dot product and contrastive objective remain CRL-specific. AWR value
branches remain legacy.

M5 validation passed: CPU regression `32/32`, real antmaze-medium N=20 strict
parity with maximum tracked error `0.0`, and paired GPU 1000-step legacy versus
actor+critic computation trainer smokes with checkpoint probes. M5 kept the
AWR value branches legacy by scope; M6 below completes their migration.

M6 adds independent AWR value slots: `compute.value_state` produces
`phi_V(s)` and `compute.value_goal` produces `psi_V(g)`. The AWR value remains
`ensemble=False`, preserves the bilinear/contrastive semantics, and does not
share parameters with the critic branches. Full AWR strict N=20 parity,
the `38/38` CPU regression, and paired GPU 1000-step AWR smokes pass.

See the full record in
[`docs/milestone_2026-08-12_hiql.md`](docs/milestone_2026-08-12_hiql.md).

See:

- `docs/milestone_2026-08-12_hiql.md`
- `docs/crl_migration.md`
- `docs/crl_critic_migration.md`
- `docs/crl_awr_value_migration.md`
- `docs/runtime_migration_audit.md`
- `docs/hiql_runtime_integration.md`
- `docs/architecture_decisions.md`
- `docs/computation_migration_plan.md`

## Quick validation

Daily CPU regression, using the available OGBench data directory:

```bash
cd RLC
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
JAX_PLATFORMS=cpu PYTHONPATH=. \
python -m unittest discover -s tests -p 'test_*.py' -v
```

The current suite contains 38 tests, including real-data determinism,
trajectory/goal sampling sanity, and real N=20 legacy-vs-computation parity.

The opt-in GPU N=1000 diagnostic is kept separate from the fast CPU suite:

```bash
cd RLC
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
python tools/diagnose_hiql_gpu_parity.py --steps=1000
```

The diagnostic requires a CUDA-enabled JAX installation and prints the actual
JAX devices/backend before running.

## CRL runtime

CRL is now registered in the same runtime and supports both reference actor
loss branches (`ddpgbc` and `awr`). The actor representation, bilinear critic
branches, and AWR value branches can use independent computation slots; the
bilinear readout and contrastive objectives remain legacy-equivalent.

```bash
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
python -m impls.main --agent=crl --env_name=antmaze-medium-navigate-v0 \
  --train_steps=1000 --computation
```

CRL migration and actor/critic computationization results are documented in
[`docs/crl_migration.md`](docs/crl_migration.md) and
[`docs/crl_critic_migration.md`](docs/crl_critic_migration.md). M6 AWR value
results are in [`docs/crl_awr_value_migration.md`](docs/crl_awr_value_migration.md).

## Vanilla CoGHP

Vanilla CoGHP is available as `--agent=coghp` in the same runtime. It follows
the official `wlsdn9350/CoGHP` Mixer architecture: one physical autoregressive
`actor_mixer` core is reused for subgoals and the final action, while
`high_actor_head` and `low_actor_head` are independent readouts. Its official
`MultiHGCDataset` is included with RLC's explicit seeded RNG contract.

```bash
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
python -m impls.main --agent=coghp --env_name=antmaze-medium-navigate-v0 \
  --train_steps=1000
```

Vanilla CoGHP intentionally does not use `--computation`; M7 leaves
`impls/computation/` unchanged. See
[`docs/coghp_official_migration_audit.md`](docs/coghp_official_migration_audit.md)
and [`docs/coghp_vanilla_migration.md`](docs/coghp_vanilla_migration.md) for
the source audit, sharing/parameter accounting, and official/RLC parity.
