# HIQL runtime migration audit

## Scope and reference selection

This audit is the first step of the codex4 runtime migration.  The benchmark
and environment implementation remains the canonical package in
`RLC/ogbench/`.  It is not replaced by
`offline_rl_baselines/ogbench/ogbench/`.  The algorithm-side implementation in
`offline_rl_baselines/ogbench/impls/` is used as the training-flow reference;
the frozen CoGHP implementation is used as the reproducibility comparison.

The first vertical slice is HIQL only:

```text
impls/main.py
  -> impls/utils/env_utils.py -> RLC/ogbench
  -> impls/utils/datasets.py  -> Dataset/GCDataset/HGCDataset
  -> impls/agents/hiql.py     -> networks + computation slots
  -> impls/utils/evaluation.py
  -> impls/utils/flax_utils.py (checkpoint)
  -> impls/utils/log_utils.py  (CSV/progress logging)
```

CRL, CoGHP, MultiHGCDataset, new computation primitives/topologies, and
formal long-running experiments are outside this migration.

## Reference dependency audit

The upstream OGBench `main.py` was read together with its imports and call
sites.  It requires the following runtime closure:

```text
main
  ├── agent registry and agent config
  ├── env_utils.make_env_and_datasets
  │     ├── ogbench.make_env_and_datasets
  │     └── Dataset.create
  ├── Dataset / GCDataset / HGCDataset
  ├── agent.create / agent.update
  ├── evaluation.evaluate
  ├── flax_utils.save_agent / restore_agent
  ├── log_utils.CsvLogger / optional wandb setup
  └── random, numpy, jax RNG initialization
```

Answers to the required audit questions:

1. `datasets.py + main.py` are not sufficient.  The main path also imports
   environment construction, evaluation, checkpoint serialization, and
   logging utilities.  The RLC copies of those modules were empty except for
   the existing `TrainState`/`ModuleDict` implementation in `flax_utils.py`.
2. The main runtime depends on the dataset wrappers, RLC environment factory,
   HIQL agent/config, evaluation, checkpoint, logging, and independent RNG
   streams.
3. Already implemented in RLC: the HIQL agent, its actor/value networks,
   computation factory/slots, `TrainState`, `ModuleDict`, and the canonical
   `ogbench` environment/dataset loader.
4. Before this migration, `main.py`, `datasets.py`, `evaluation.py`,
   `env_utils.py`, `log_utils.py`, and `reproducibility.py` were scaffolds;
   `flax_utils.py` contained only the model state helpers.
5. The minimum closure is the three dataset classes, an RLC-aware environment
   wrapper, deterministic evaluation, save/restore helpers, CSV logging, and
   a small OGBench-style HIQL training loop.
6. The upstream environment package, unrelated baseline agents, optional
   replay/online utilities, CoGHP-specific `MultiHGCDataset`, and all other
   algorithm implementations are not being copied into RLC.
7. Upstream uses process-global `np.random` in dataset goal/index sampling,
   global Gaussian noise in evaluation, and an unseeded environment reset.
   The frozen CoGHP runtime adds per-dataset `np.random.default_rng(seed)`,
   explicit RNG forwarding through every sampling/augmentation path,
   independent per-episode evaluation seeds, and seeded environment reset.
8. Those explicit streams, per-episode seed derivation, and the CoGHP fixed
   `+1e-6` goal-selection denominator are retained.  The fixed implementation
   does not change the normal HIQL sampling probabilities; it only avoids
   hidden global state.  The only edge-case behavior difference from the old
   upstream code is the already-validated denominator handling when
   `p_curgoal == 1.0`.

## Runtime compatibility decisions

### Canonical OGBench and data directory

`env_utils.py` imports `ogbench` through the current RLC import path and calls
`ogbench.make_env_and_datasets`.  It logs the imported module path, dataset
name, environment name, and resolved dataset directory.  Dataset resolution
continues to honor `OGBENCH_DATASET_DIR`, whose default is the same default
defined by `RLC/ogbench/utils.py`.

### Dataset semantics

The Dataset/GCDataset/HGCDataset implementation follows the frozen CoGHP
implementation of the upstream OGBench sampling equations.  It preserves
the compact-dataset terminal handling, `rewards`, `masks`, observations,
next-observations, actions, value goals, low-level actor goals, high-level
actor goals, high-level actor targets, future/random goal choices, trajectory
boundaries, and `subgoal_steps`.  `MultiHGCDataset` is intentionally omitted.

### RNG streams

```text
training seed
  ├── dataset RNG: explicit NumPy Generator for train wrapper
  ├── validation RNG: independent Generator
  ├── environment seed: independent derived seed
  ├── evaluation episode seed: derive_seed(seed, episode, 0)
  ├── evaluation actor key: derive_seed(seed, episode, 1)
  └── evaluation noise RNG: derive_seed(seed, episode, 2)

agent.update
  └── agent-owned JAX PRNG sequence (HIQLAgent.rng)
```

The dataset stream is never inferred from the global NumPy state.  Evaluation
episodes are independent so a variable episode length cannot perturb later
episodes.

### Checkpoint and logging boundary

RLC's existing `TrainState` remains the serialization source of truth.  The
minimal checkpoint helper serializes the full HIQL PyTree, including optimizer
state and the agent-owned RNG, and restores it with Flax serialization.  CSV
logging is always available without introducing a new mandatory dependency;
wandb is not required for this first vertical slice.

## Validation required by this audit

The migration adds real-data tests against an available locomaze dataset:

* same-seed `HGCDataset` sampling for 20 consecutive batches;
* real metadata trajectory-boundary and high-target interval sanity checks;
* finite, shape, and dtype checks;
* strict legacy-vs-computationized HIQL parity on the same 20 real batches,
  including component losses, mapped parameters, target parameters, and the
  first-divergence report.

Only after those tests and the existing local parity suite pass is the real
trainer smoke run.  The smoke run is limited to roughly 1,000 steps and checks
startup, finite loss/gradients, target updates, evaluation, logging, and
checkpoint save/load.  It is not a scientific baseline reproduction.

