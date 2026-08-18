# Experiment Management Audit

## Scope

This audit was performed before implementing the experiment-management layer
required by `docs/prompt for codex2.md`. The current source of truth is the
working tree under `RLC/`; no existing experiment-management package was
found.

## Existing output behavior

- `impls/main.py` accepted a free-form `--save_dir` and appended
  `sd<seed>_<timestamp>` via `get_exp_name`.
- `runtime_metadata.json` already recorded `agent`, `environment`,
  `dataset_dir`, `ogbench_module`, `seed`, `computation`, and the resolved
  M8 `compute_slots` snapshot.
- `CsvLogger` wrote `train.csv` and `eval.csv` with a `step` column and
  flushed each row.
- `save_agent` stored pickle checkpoints directly in the run directory.
- There was no `resolved_config.json`, `summary.json`, manifest, stable run
  identity, Git provenance helper, or aggregation tool.

The existing M8 smoke output under `/tmp/rlc_m8_provenance_smoke2` confirms
that the actual runtime produced the metadata/CSV/checkpoint pattern above.
It is temporary output and is not adopted as a second project artifact root.

## Existing boundaries preserved

- Agent configuration remains in `impls/agents` and `configs/agents`.
- Computation remains under `impls/computation`; no fields were added to
  `ComputationSpec`.
- The new layer only records resolved configuration and run provenance. It
  does not select a topology, add state, or modify HIQL/CRL/CoGHP semantics.
- Raw artifacts are placed under one canonical `runs/` root for identified
  studies. Legacy/debug calls remain supported through an explicitly supplied
  `--save_dir` or `runs/legacy`.

## M8-to-M9 management gap

M8 supplied the computation-slot snapshot, but not the scientific identity
around it. M9 adds Study/configuration parsing, stable `config_id` and slug
validation, deterministic run paths, lifecycle status, Git/JAX/dataset
provenance, summaries, manifests, and CSV-only aggregation.
