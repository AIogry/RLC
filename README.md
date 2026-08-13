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

## Current milestone — 2026-08-12

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

See the full record in
[`docs/milestone_2026-08-12_hiql.md`](docs/milestone_2026-08-12_hiql.md).

See:

- `docs/milestone_2026-08-12_hiql.md`
- `docs/crl_migration.md`
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

The current suite contains 25 tests, including real-data determinism,
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
loss branches (`ddpgbc` and `awr`). The CRL bilinear critic remains legacy;
only the actor representation body can use the existing computation slot.

```bash
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
python -m impls.main --agent=crl --env_name=antmaze-medium-navigate-v0 \
  --train_steps=1000 --computation
```

CRL migration and actor computationization results are documented in
[`docs/crl_migration.md`](docs/crl_migration.md). CRL critic computationization
is intentionally deferred.
