# Experiment Management

RLC uses a small file-based layer to make experiments reproducible and
traceable before scientific M9 implementation begins.

## Study → Configuration → Run

- A **Study** is one scientific question. It lives in
  `experiments/<study-name>/study.yaml` and declares factors, fixed design,
  algorithms, environments, seeds, and the primary metric.
- A **Configuration** is one set of scientific factors, without a seed. It
  lives in `experiments/<study-name>/configs/<config_id>.yaml` and has a
  stable ID such as `M9A-C002` plus a readable slug.
- A **Run** is one Configuration + Environment + Seed + Git commit execution.

Filenames are for readability; metadata is the source of experimental truth.
The `config_id` is the stable identity. Slugs must be path-safe and cannot
use mutable labels such as `final`, `new`, `best`, `v2`, or `try2`.

## Artifact roots and identity

Canonical runs use one root:

```text
runs/<study_id>/<config_id>__<slug>/<environment>/seed_<NNN>/
```

The path is deterministic and contains no timestamp. Reusing the same
identity fails fast when the directory already exists. A future explicit
resume/overwrite policy can be added without changing the identity.

`--save_dir` remains a compatibility/debug option. Calls without a Study use
`runs/legacy/sd<seed>_<timestamp>` (or the explicitly supplied save directory)
and are not canonical scientific Runs.

## Run files and provenance

Each canonical run writes:

```text
resolved_config.json
runtime_metadata.json
train.csv
eval.csv
summary.json
checkpoints/
```

`resolved_config.json` contains the Study, Configuration, launcher arguments,
and resolved agent configuration. `runtime_metadata.json` preserves all M8
fields and adds `study_id`, `config_id`, `config_slug`, `algorithm`,
`environment`, `seed`, `git_commit`, `git_dirty`, `start_time`, `hostname`,
JAX backend/device descriptions, dataset identity/path, lifecycle status, and
resolved `compute_slots`.

The Git helper records the current commit and whether the tree is dirty.
Dirty debug runs are allowed, but their metadata remains explicit.

## Lifecycle, summary, and failure retention

Run states are `planned`, `running`, `completed`, `failed`, `aborted`, and
`invalid`. A created run starts as `running`; normal completion writes
`completed`. Exceptions retain the partial directory, metadata, CSVs and
checkpoint files, write `failure.json`, and mark the run `failed`.

`summary.json` is generated from explicit success columns in `eval.csv`:
`evaluation/overall_success`, `overall_success`, or `success`. It records
`final_success`, `best_success`, and `best_step`. If no supported column is
present, these fields remain null rather than being inferred.

## Manifest and aggregation

Build a machine-readable Study manifest with:

```bash
python tools/manifest.py \
  --study experiments/M9A_single_state_iteration/study.yaml \
  --run-root runs
```

The manifest includes planned configurations and observed runs, including
failed runs. It records the configuration factors, status, relative run path,
Git commit, and summary values.

Aggregate only the CSV manifest; raw artifacts are not modified:

```bash
python tools/aggregate_results.py \
  experiments/M9A_single_state_iteration/manifest.csv \
  --metric final_success
```

The output groups by `config_id + environment` and writes `count`, `mean`, and
population `std` to `aggregated.csv`. Rows without a numeric metric are
excluded from the numeric aggregate but remain in the manifest.

## M9A SingleState study

`experiments/M9A_single_state_iteration/` is the canonical M9A Study. It has
26 configurations and 52 planned runs (`2` AntMaze environments × `1` seed).
The matrix contains HIQL and CRL baselines, CRL actor placement, HIQL high-only,
low-only, and high+low placements, each with `K ∈ {1,2,4}` and residual versus
non-residual updates.

The fixed design is decision-local, non-learned normal-buffer state, `z+x`
injection, an MLP update module, shared parameters across iterations, and no
state across environment steps. State initialization alternatives, other
injection forms, gating, normalization recipes, multi-state, and HRM are
deferred. Each canonical configuration controls its slots explicitly through
`agent_overrides`; the launcher does not infer placement from the slug.

The SingleState topology is implemented, but this does not mean the scientific
study has started. `tools/sweep.py` is safe by default and requires explicit
`--execute`; failed/invalid/running artifacts are retained instead of being
silently relaunched. See [`docs/m9_single_state_iterative.md`](m9_single_state_iterative.md)
for the architecture, accounting, audit, and validation record.

## M9B TwoState study

`experiments/M9B_two_state/` contains 16 TwoState configurations and 32 planned
runs over the same two AntMaze environments and seed 0. It has four actor
placement families, two schedules (`H2L1`, `H2L6`) and two real credit policies
(`full_bptt`, `one_step`). M9B references M9A-C001/M9A-C002 as its HIQL/CRL
baselines rather than defining duplicate baseline runs. The topology, credit
semantics, accounting and validation are documented in
[`docs/m9b_two_state.md`](m9b_two_state.md).
