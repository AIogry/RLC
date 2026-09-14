# M25 — Linear Reversible Boolean Control Coordinates

This directory contains the first executable, standalone Stage-1 M25
implementation.  It learns a bijective linear Boolean map from real adjacent
Puzzle board-change edges and evaluates whether those edges become one-axis
edits.  It does not train or modify a goal-conditioned RL agent.

Both checked-in configurations are explicitly engineering-only.  Their
32-layer depth is a capacity candidate, not a theoretical or scientific
default.  Candidate depths must be screened on random invertible 20- and
24-bit maps before a later formal M25 design is frozen.

The repository Run invariant keeps `seed` out of Configuration YAML.  It is
still explicit: Study declares seed 0, the training CLI requires/resolves the
Run seed, and the resolved runtime configuration records it alongside every
model, optimizer, and sampling factor.

The only learner-facing fields are `start_board` and `end_board`.  Event XOR
signatures exist solely for audits and post-hoc grouping.  Actions are retained
indirectly through each event's exact transition index for a future physical
window stage; action values do not enter Stage-1 H.

The canonical OGBench loader is used in compact mode.  Episode observation
boundaries are recovered from its `valids` mask (which is deterministically
derived from the standard `terminals` field), rather than interpreting compact
transition-oriented terminals as observation endpoints.  `valids` is boundary
metadata only and never enters the model.

No command in this directory launches automatically.  An engineering run is
created only by an explicit invocation such as:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python tools/train_control_coordinates.py \
  --config M25-E001 --seed 0 --dataset-dir /path/to/raw_ogbench \
  --run-root /path/to/engineering_runs
```

This is not a formal campaign command.

Audit either real dataset before training:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  python tools/audit_puzzle_control_coordinates.py \
  --environment puzzle-4x5-play-v0 --dataset-dir /path/to/raw_ogbench
```

Probe a candidate depth on independent random invertible matrices (the default
checks both N=20 and N=24):

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python tools/check_boolean_flow_capacity.py \
  --num-layers 32 --steps 10000
```

The capacity output is an architecture/optimization gate, never a scientific
Puzzle score.  A failed gate must remain visible and blocks freezing that depth
for a formal design.

Evaluate a saved engineering checkpoint without changing its source Run:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python tools/evaluate_control_coordinates.py \
  --run-dir /path/to/M25/run --checkpoint-role last \
  --dataset-dir /path/to/raw_ogbench --output /path/to/evaluation.json
```
