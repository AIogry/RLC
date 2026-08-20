# RLC — RL Computation

RLC is a research platform for comparing computation modules and computation organizations in goal-conditioned reinforcement learning.

## Research handoff

The current handoff contract, research questions, design invariants, experiment
registry, and external-data ledger are maintained in
[`handoff/HANDOFF.md`](handoff/HANDOFF.md). Before taking ownership, run the
read-only handoff check:

```bash
PYTHONPATH=. python3 tools/handoff_doctor.py
```

The handoff package separates implementation status, scientific evidence,
open hypotheses, immutable source data, and rebuildable analysis outputs.

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

## Current milestone

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
[`docs/8-12/milestone_2026-08-12_hiql.md`](docs/8-12/milestone_2026-08-12_hiql.md).

The 2026-08-13 CRL and Vanilla CoGHP milestone is recorded in
[`docs/8-13/milestone_2026-08-13_coghp.md`](docs/8-13/milestone_2026-08-13_coghp.md).

The M8 computation foundation is documented in
[`docs/8-15/m8_computation_foundation.md`](docs/8-15/m8_computation_foundation.md) and
[`docs/8-15/computation_ontology.md`](docs/8-15/computation_ontology.md).

The file-based Study → Configuration → Run layer is documented in
[`docs/8-15/experiment_management.md`](docs/8-15/experiment_management.md), with the
planned M9A example under `experiments/M9A_single_state_iteration/`.

The 2026-08-15 milestone is recorded in
[`docs/8-15/milestone_2026-08-15.md`](docs/8-15/milestone_2026-08-15.md).

The M9A single-state iterative actor implementation, audit, 26-configuration
matrix, parameter accounting, and short validation record are documented in
[`docs/8-16/m9_single_state_iterative.md`](docs/8-16/m9_single_state_iterative.md).

The M9B two-state hierarchical computation implementation, executable credit
axis, 16-configuration Study, and validation record are documented in
[`docs/8-16/m9b_two_state.md`](docs/8-16/m9b_two_state.md).

The M10A fixed-budget HIQL computation-placement study, 11-configuration matrix,
zero-buffer SingleState semantics, and Dense-MAC/parameter audit are documented in
[`docs/8-18/m10a_fixed_budget_placement.md`](docs/8-18/m10a_fixed_budget_placement.md).

The M10A-R001 checkpoint reevaluation and reusable M10A-A001 analysis pipeline
are documented in [`docs/8-19/m10a_checkpoint_reevaluation.md`](docs/8-19/m10a_checkpoint_reevaluation.md),
[`docs/8-19/analysis_pipeline.md`](docs/8-19/analysis_pipeline.md), and
[`handoff/experiment_registry.yaml`](handoff/experiment_registry.yaml).

The formal frozen-worktree execution procedure, generic GPU launcher,
provenance fingerprint, lifecycle/retry policy, protocol audit, and M9A/M9B
freeze checklist are documented in
[`docs/8-16/experiment_execution.md`](docs/8-16/experiment_execution.md).

The 2026-08-16 M9 milestone is recorded in
[`docs/8-16/milestone_2026-08-16.md`](docs/8-16/milestone_2026-08-16.md).

Formal Studies are launched only from a clean detached worktree with the
explicit common protocol, for example:

```bash
bash scripts/run_study.sh --study experiments/M9B_two_state/study.yaml \
  --gpus 0,1 --run-root /data/.../RLC/runs \
  --dataset-root /data/.../ogbench \
  --train-steps <confirmed> --batch-size <confirmed> \
  --eval-interval <confirmed> --eval-tasks all \
  --eval-episodes <confirmed> --save-interval <confirmed> \
  --eval-temperature <confirmed>
```

Do not use the current dirty development tree or smoke artifacts as formal
scientific results.

See:

- `docs/8-12/milestone_2026-08-12_hiql.md`
- `docs/8-15/crl_migration.md`
- `docs/8-15/crl_critic_migration.md`
- `docs/8-15/crl_awr_value_migration.md`
- `docs/8-16/runtime_migration_audit.md`
- `docs/8-13/hiql_runtime_integration.md`
- `docs/8-15/architecture_decisions.md`
- `docs/8-15/computation_migration_plan.md`

## Quick validation

Daily CPU regression, using the available OGBench data directory:

```bash
cd RLC
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
JAX_PLATFORMS=cpu PYTHONPATH=. \
python -m unittest discover -s tests -p 'test_*.py' -v
```

The current suite contains 76 tests, including experiment-management
serialization/manifest/aggregation checks, M9A/M9B topology and credit tests,
real-data determinism, trajectory/goal sampling sanity, and real N=20
legacy-vs-computation parity.

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
[`docs/8-15/crl_migration.md`](docs/8-15/crl_migration.md) and
[`docs/8-15/crl_critic_migration.md`](docs/8-15/crl_critic_migration.md). M6 AWR value
results are in [`docs/8-15/crl_awr_value_migration.md`](docs/8-15/crl_awr_value_migration.md).

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
[`docs/8-13/coghp_official_migration_audit.md`](docs/8-13/coghp_official_migration_audit.md)
and [`docs/8-13/coghp_vanilla_migration.md`](docs/8-13/coghp_vanilla_migration.md) for
the source audit, sharing/parameter accounting, and official/RLC parity.
