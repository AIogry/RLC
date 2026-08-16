# HIQL runtime integration

The first end-to-end RLC vertical slice is now:

```text
RLC/ogbench
  -> impls.utils.env_utils.make_env_and_datasets
  -> Dataset.create(seed=...) + HGCDataset(rng=...)
  -> impls.agents.hiql.HIQLAgent.create
  -> computation slots: high_actor / low_actor / value
  -> HIQLAgent.update
  -> impls.utils.evaluation.evaluate
  -> CSV logging + Flax checkpoint
```

`RLC/ogbench` remains the benchmark/environment implementation. The
algorithm-side runtime is adapted from the OGBench reference and the fixed
CoGHP runtime; no upstream `ogbench/` package is copied over it.

## Runtime components

`impls/utils/datasets.py` contains only the public wrappers required by HIQL:
`Dataset`, `GCDataset`, and `HGCDataset`. It preserves compact-dataset
terminal handling, future/random/current goal probabilities, trajectory
boundaries, `subgoal_steps`, all HIQL goal fields, rewards, and masks. Each
wrapper accepts an explicit NumPy `Generator`.

`impls/utils/env_utils.py` resolves `OGBENCH_DATASET_DIR`, calls the RLC
canonical `ogbench.make_env_and_datasets`, and prints the imported module
path, environment/dataset name, and resolved directory. The optional frame
stack and episode monitor wrappers follow OGBench behavior.

`impls/utils/evaluation.py` retains task selection, goal reset, temperature,
action clipping/noise, success, return, length, and trajectory recording. It
uses independent derived environment, actor, and noise streams per episode.

`impls/utils/flax_utils.py` adds minimal full-agent save/restore on top of the
existing RLC `TrainState`; the optimizer state and agent-owned JAX RNG are
serialized with the computationized parameter tree. `log_utils.py` provides
dependency-light scalar CSV logging. `impls/main.py` is intentionally a
small OGBench-style loop and exposes `--computation` to select the three
already-implemented slots.

The main entry point is:

```bash
OGBENCH_DATASET_DIR=/path/to/ogbench/data JAX_PLATFORMS=cpu PYTHONPATH=. \
python -m impls.main --env_name=antmaze-medium-navigate-v0 \
  --train_steps=1000 --computation
```

The optional `--width`, `--depth`, and `--batch_size` flags are intended for
short smoke tests; they do not change the production HIQL loss or sampling
algorithm.

## Reproducibility contract

```text
seed
  ├── train Dataset Generator: derive_seed(seed, 1)
  ├── validation Dataset Generator: derive_seed(seed, 2)
  ├── environment reset: derive_seed(seed, 3)
  ├── evaluation episode reset: derive_seed(eval_seed, episode, 0)
  ├── evaluation actor key: derive_seed(eval_seed, episode, 1)
  └── evaluation Gaussian noise: derive_seed(eval_seed, episode, 2)

agent update: HIQLAgent.rng, split by the JAX update path
```

The process-global Python/NumPy seeds are initialized only for compatibility;
dataset and evaluation sampling use explicit streams. The fixed CoGHP goal
selection denominator (`+1e-6`) is retained.

## Validation record

The real tests use the available `antmaze-medium-navigate-v0` dataset and
confirm that the imported module is
`/home/eai/Research/RLC/ogbench/__init__.py`.

* 20 consecutive same-seed `HGCDataset` batches: bitwise identical for every
  required field; max error 0.0.
* Real metadata sanity: indices remain within trajectory boundaries; future
  goals do not cross episodes; high targets equal
  `min(index + subgoal_steps, high_goal)`; all required tensors are finite and
  correctly shaped.
* Real strict parity, N=20: legacy HIQL versus all three MLP + FeedForward +
  Direct slots, with a semantic initial parameter graft and identical batch
  sequence. Total/value/high-actor/low-actor loss, full semantic gradients,
  online/target semantic parameters, update diagnostics, and agent RNG all
  passed at `rtol=0`, `atol=1e-6`; the first divergence step was none. The
  measured loss and gradient maxima were 0.0.
* Existing CPU suite: 25/25 tests passed (the original 22 plus 3 real-runtime
  tests).
* Trainer smoke: both native legacy and native computationized HIQL ran 1,000
  steps on `antmaze-medium-navigate-v0` with width 6, depth 2, batch size 8.
  Losses and diagnostics stayed finite, evaluation ran, CSV logs were
  written, and checkpoint save/load reproduced the same action and value.
  The recorded task-1 success was 0.0 in both short smoke runs and is not
  interpreted as a scientific result.

This is integration validation for the HIQL vertical slice, not a formal
baseline reproduction. No long training or CRL migration was started.
