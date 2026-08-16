# Formal experiment execution

This document defines how a validated RLC Study becomes a reproducible GPU
experiment. It is deliberately separate from the scientific definitions in
`experiments/`: a Study declares the matrix, while this document defines the
frozen code, environment, protocol, and artifact rules.

## Study, Configuration, Run

The execution hierarchy is:

```text
Study
  -> Configuration
      -> Run = Configuration + Environment + Seed + frozen provenance
```

`experiments/<study>/study.yaml` and `configs/*.yaml` are experiment design.
`runs/` (or an explicitly selected external run root) contains results. A
canonical Run path is deterministic:

```text
<run_root>/M9B/M9B-C012__<slug>/antmaze-large-navigate-v0/seed_000/
```

For example, one completed result is expected to look like:

```text
<run_root>/
└── M9B/
    └── M9B-C012__<slug>/
        └── antmaze-large-navigate-v0/
            └── seed_000/
                ├── resolved_config.json
                ├── runtime_metadata.json
                ├── train.csv
                ├── eval.csv
                ├── summary.json
                └── checkpoints/
```

Run identity contains no timestamp and is never silently renamed to `run2`,
`new`, `retry2`, or `final`. `create_run_context()` fails if the canonical
directory already exists.

## Frozen code and worktree procedure

The current development tree is intentionally dirty while M9A/M9B work is
being reviewed. It must not be used for formal execution. After review:

```bash
cd /home/eai/Research/RLC
git status --short
git add <reviewed-files>
git commit -m "RLC: finalize M9 experiment execution foundation"
FROZEN_SHA="$(git rev-parse HEAD)"
git worktree add --detach /home/eai/Research/RLC-exp "$FROZEN_SHA"
```

The worktree command is a user-controlled freeze step; this task does not
create it automatically. The detached worktree must remain unchanged for the
whole M9A/M9B sweep. Do not copy the RLC directory or run from a later mutable
checkout. The formal launcher records the commit and refuses any dirty tree.

## Generic execution chain

There are no per-Study Python launchers. The chain is:

```text
experiments/<study>/study.yaml + configs/*.yaml
    -> scripts/run_study.sh
    -> tools/sweep.py              (generic GPU worker queue)
    -> tools/run.py / impls.main   (generic Run identity and trainer)
    -> training, evaluation, CSVs, checkpoints, summary
```

`tools/sweep.py` assigns one Run to each listed GPU and gives the next pending
Run to whichever worker finishes first. It does not pin a method to a GPU and
does not launch multiple formal Runs on the same GPU. Each child receives the
physical ID through `CUDA_VISIBLE_DEVICES=<physical_gpu>`; seeing `cuda:0`
inside that child is expected. Runtime metadata records the actual JAX device
descriptions.

The human-facing formal entry point is:

```bash
cd /home/eai/Research/RLC-exp
bash scripts/run_study.sh \
  --study experiments/M9B_two_state/study.yaml \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps <confirmed> \
  --batch-size <confirmed> \
  --eval-interval <confirmed> \
  --eval-tasks all \
  --eval-episodes <confirmed> \
  --save-interval <confirmed> \
  --eval-temperature <confirmed>
```

For the first-round protocol frozen below, the two Study invocations differ
only in their training/checkpoint budget:

```text
M9A baseline configurations:  train_steps=1,000,000  save_interval=1,000,000
M9A SingleState variants:     train_steps=500,000    save_interval=500,000
M9B TwoState variants:        train_steps=500,000    save_interval=500,000
```

The launcher validates the Study, run root, dataset directory, GPU IDs, and
clean Git state. It prints commit, Study, roots, GPUs, planned/completed/
remaining counts, and the exact common protocol before invoking the generic
sweep. It has no interactive confirmation, so it is suitable for tmux or a
background process. It sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` by default
uniformly for all workers; an explicit value may be supplied through the same
environment for all workers. No numerical XLA flags are added.

Scientific factors such as `K`, residual choice, `H2L1/H2L6`, credit policy,
or actor placement must remain in the Study configuration. They must not be
encoded in this Bash script.

## Frozen first-round M9A + M9B protocol

The following is the frozen protocol for the first formal M9A/M9B round. It
applies identically to HIQL vanilla, CRL vanilla, M9A SingleState variants,
and M9B TwoState variants unless a field is explicitly listed as a budget
factor below.

### Common settings

```text
batch_size       = 1024
learning_rate    = 3e-4
log_interval     = 5000
eval_interval    = 100000
eval_tasks       = all        # equivalent to OGBench official None
eval_episodes    = 20         # per task
eval_temperature = 0
eval_gaussian    = None
seed             = 0
environments     = antmaze-medium-navigate-v0, antmaze-large-navigate-v0
```

`eval_tasks=all` means every evaluation task exposed by the environment. The
same dataset split, optimizer, learning rate, batch size, evaluation schedule,
temperature, and checkpoint semantics are required for baseline, SingleState,
and TwoState runs. Only the Study-declared scientific factors and the budget
factor below may differ.

### Baseline protocol

The baseline protocol applies to HIQL vanilla and CRL vanilla:

```text
train_steps  = 1,000,000
save_interval = 1,000,000
```

Each baseline trains for the complete 1M-step budget and saves the final
checkpoint at step 1M.

### Exploration protocol

The exploration protocol applies to all M9A SingleState and M9B TwoState
variants:

```text
train_steps  = 500,000
save_interval = 500,000
```

Each structural variant trains for 500k steps and saves its final checkpoint
at step 500k. The different checkpoint intervals express different training
endpoints; they do not define different optimization protocols.

These values preserve the audited OGBench settings wherever applicable. The
only first-round changes are the explicitly declared 500k exploration budget
and its final checkpoint step.

### Evaluation and checkpoint schedule

All three families use `eval_interval=100k`, all tasks, and 20 episodes per
task. Under the current RLC training loop, evaluation is emitted when
`step % eval_interval == 0` or at the final training step; there is no extra
step-1 evaluation in the current runtime. Therefore the formal points are:

```text
SingleState / TwoState @ 500k:
100k, 200k, 300k, 400k, 500k

Vanilla baseline @ 1M:
100k, 200k, 300k, 400k, 500k, 600k, 700k, 800k, 900k, 1M
```

The OGBench official `save_interval=1M` is semantically a single save at the
training endpoint. RLC therefore uses the analogous endpoint-only saves at
1M for baseline and 500k for exploration; this is not a change to optimizer,
gradient, or evaluation protocol.

### Matched-budget comparison

The primary first-round budget-matched comparison is:

```text
Variant @ 500k  vs  corresponding algorithm's Vanilla Baseline @ 500k
```

Because the baseline continues to 1M and is evaluated every 100k, its 500k
learning-curve point is available directly. `Baseline @ 1M` remains a separate
complete-training reference. It must not be used as the main equal-budget
comparison against `Variant @ 500k`.

Every later result table must distinguish at least:

```text
Baseline @ 500k
Baseline @ 1M
Variant @ 500k
```

Seed `0` remains an exploratory first-round seed. These results must not be
described as statistically significant, robust improvement, or final paper
results; selected conditions require later multi-seed confirmation.

### Current runtime/documentation consistency

The frozen protocol is the target formal protocol, but the current runtime has
one known operational mismatch:

- `impls/main.py` currently defaults to `log_interval=100`, while the frozen
  protocol requires `log_interval=5000`;
- `scripts/run_study.sh` currently forwards the common training/evaluation
  fields but does not yet expose or forward `--log_interval`.

The agent defaults and existing generic arguments already agree with the
frozen `learning_rate=3e-4`, `batch_size=1024`, `eval_interval=100k`,
`eval_tasks=all`, `eval_episodes=20`, `eval_temperature=0`, and
`eval_gaussian=None` when explicitly supplied. This documentation-only task
does not change the launcher or training loop. Formal execution must wait for
that generic `log_interval` plumbing to be resolved and revalidated; no run
may silently use the current default of 100.

## Previous smoke protocol

The previous `train_steps=1`, `batch_size=8`, `eval_tasks=1`, and
`eval_episodes=1` GPU checks were infrastructure smoke tests only. Their
artifacts, including `/tmp/rlc_m9b_gpu_runs_final`, are not formal M9A/M9B
results and must not be compared as scientific samples.

The first-round formal protocol above supersedes the previous “not approved”
placeholder values. It does not retroactively change any smoke artifact.

## Seed and statistical interpretation

Seed `0` in the current Studies is exploratory. It must not be described as a
statistically significant result, robust improvement, or final paper result.
Multi-seed confirmation follows only after conditions are selected.

## M9A and M9B freeze checklist

Before passing `--execute`, record the following as a checked checklist:

```text
[ ] validated implementation committed
[ ] Git worktree clean
[ ] frozen commit recorded
[ ] dataset path verified
[ ] run root verified
[ ] 2 GPUs visible
[ ] M9A manifest correct (26 configs, 52 planned runs)
[ ] M9B manifest correct (16 configs, 32 planned runs)
[x] first-round training protocol recorded above
[ ] baseline reuse valid
[ ] dry-run inspected
```

Current inventory is:

```text
M9A: 26 configurations × 2 environments × seed 0 = 52 planned Runs
M9B: 16 configurations × 2 environments × seed 0 = 32 planned Runs
Combined: 84 planned Run identities
```

M9A has two baseline configurations (`M9A-C001` HIQL and `M9A-C002` CRL),
plus six configurations each for CRL actor, HIQL high actor, HIQL low actor,
and HIQL high+low actor. M9B has four configurations each for CRL actor,
HIQL high actor, HIQL low actor, and HIQL high+low actor, covering H2L1/H2L6
and full-BPTT/one-step. M9B references M9A-C001/C002 and does not duplicate
those baseline rows.

Baseline reuse is valid only when the referenced run has the same frozen Git
commit, environment/dataset split, seed, optimizer, learning rate, batch size,
training budget, evaluation schedule, temperature, and checkpoint protocol.
Until that equivalence is verified after protocol approval, M9B baseline reuse
is a declared reference rather than a completed result.

No full M9A/M9B sweep is started by this documentation-only task.
