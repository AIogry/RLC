# M24A — Puzzle Goal-Conditioning Intervention

M24A is the first formal Study in the **M24 — Operation-Aligned Goal
Learning** program. It asks whether a deterministic change in the coordinate
used to express the remaining Puzzle task changes learning and execution while
the algorithm, data pairs, success semantics, architecture, optimization, and
evaluation remain fixed.

This directory declares six formal Runs: three conditions on each of
`puzzle-4x5-play-v0` and `puzzle-4x6-play-v0`, all with training seed `0`.
Creating this Study does not start training.

## Frozen matrix

| Config | Environment | Study label | Production mode | Goal-side binary coordinate |
| --- | --- | --- | --- | --- |
| `M24A-C001` | `puzzle-4x5-play-v0` | G1 | `board` | `g` |
| `M24A-C002` | `puzzle-4x6-play-v0` | G1 | `board` | `g` |
| `M24A-C003` | `puzzle-4x5-play-v0` | G2 | `residual` | `b XOR g` |
| `M24A-C004` | `puzzle-4x6-play-v0` | G2 | `residual` | `b XOR g` |
| `M24A-C005` | `puzzle-4x5-play-v0` | G4 | `oracle_operation` | `inverse(M) * (b XOR g)` over GF(2) |
| `M24A-C006` | `puzzle-4x6-play-v0` | G4 | `oracle_operation` | `inverse(M) * (b XOR g)` over GF(2) |

G4 exposes unordered remaining press parity. It is not an ordered plan,
next-button oracle, scheduler, solver wrapper, or motor program. The policy
continues to emit continuous robot actions directly.

## Controlled comparison

Every cell uses GCIQL DDPG+BC with `alpha=0.4`, Puzzle entity tokens,
Mixer-L2, MeanContextReadout, and structured actor/value/critic placement.
The value/actor goal sampling probabilities, batch size, optimizer,
one-million-step budget, and evaluation schedule match the M16D S002 operating
point. Alpha 0.4 is fixed rather than tuned and is not claimed optimal.

All cells use `PuzzleBoardGCDataset`. Consequently, training success is
board equality and reward/mask are `c-1` and `1-c`. Raw transition, value-goal,
and actor-goal sampling remain identical across G1/G2/G4. Only deterministic
network-input conditioning differs.

The clean causal contrasts are:

1. G2 - G1: Residual minus Board.
2. G4 - G2: OracleOperation minus Residual.

Historical canonical GCIQL (G0) combines a full raw goal with index-equality
training success. It is a dashed/gray descriptive reference only; G0 to G1 is
not a clean representation contrast.

## Endpoints and interpretation

`evaluation/overall_success` remains the infrastructure metric used to retain
the best checkpoint. The scientific endpoint is the **last checkpoint at 1M**,
reported separately for all five tasks. The primary derived summary for each
environment is:

```text
HardTaskMean = mean(Task2, Task3, Task4, Task5 final success at 1M)
```

Task-level values must remain visible; HardTaskMean must not replace them.
Puzzle-4x5 Task4 is an emphasized diagnostic cell, not a separately optimized
treatment.

The Study can reveal whether the supplied coordinate changes learning under
this fixed setup. It cannot by itself prove reasoning, planning, a learned
solver, or a universal representation bottleneck. Because the screening uses
one training seed, differences are direction-selection evidence rather than a
paper-final statistical comparison.

## Required final diagnosis

After all six `last@1M` checkpoints exist, run a separate M23A-style
controlled-goal-replay campaign with 50 episodes per task, evaluation seed
`20260909`, temperature 0, and no Gaussian noise. Reuse the persisted M23A
4x5/4x6 full-goal replay archives so the three M24A policies receive identical
external goals and initial conditions.

In addition to success and Dstar progress, retain zero-press rate, first-press
latency, press count, progress/regress, repeated-source behavior, final stall,
and unique-button coverage. On full-rank 4x5/4x6 single-source events, the
existing `progress_advantage` metric is the requested operation-selection
advantage relative to the random support-size baseline.

This diagnosis is not launched as part of the training Study.

## Git and execution gates

The current capability and Study definitions must be reviewed and committed
separately. Formal execution then requires a clean detached worktree at the
reviewed commit, an empty `M24A` run namespace, verified datasets, and an exact
two-process GPU workload smoke. No historical M16D provenance exception carries
over to M24A.

Two jobs per physical GPU is the planned operational policy only after that
smoke passes. The physical GPU assignment remains intentionally unset until
the user-controlled launch preflight.

The non-executing Study validation command is:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/m24a_doctor.py
```

After Git review, the exact generic-launcher dry run is:

```bash
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
bash scripts/run_study.sh \
  --study experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml \
  --gpus GPU_ID --jobs-per-gpu 2 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --run-attempt 0 \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 --batch-size 1024 \
  --log-interval 5000 --eval-interval 100000 \
  --eval-tasks all --eval-episodes 20 --eval-temperature 0 \
  --save-interval 100000 \
  --save-best-checkpoint --save-last-checkpoint \
  --dry-run
```

`GPU_ID` is intentionally a placeholder, not a default assignment. Repeat the
dry run from the eventual clean frozen worktree after choosing the physical
GPU. Replacing `--dry-run` with `--execute` is not sufficient authorization:
the exact concurrent workload smoke and all other gates above must first pass,
and formal launch remains user-controlled.
