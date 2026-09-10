# M22 — OGBench Puzzle Baselines in Unified RLC

## Scientific purpose

Measure six official OGBench Puzzle baselines in the unified RLC runtime across
four canonical Puzzle-play environments and seeds 0, 1, and 2: 24
configuration cells and 72 planned runs.

## Fixed protocol

- Algorithms: GCBC, GCIVL, GCIQL, QRL, CRL, HIQL.
- Official Puzzle hyperparameters: GCBC default; GCIVL alpha=10; GCIQL
  alpha=1; QRL alpha=.3; CRL alpha=3; HIQL high/low alpha=3 and
  `subgoal_steps=10`.
- One million steps, batch 1024, log/evaluate/save every 5k/100k/100k;
  all tasks, 50 episodes per task, temperature 0, no video; primary endpoint
  `final@1M`.
- Canonical computation is disabled. Intended launcher policy is physical GPU1
  with two worker slots; GPU placement is not a scientific factor.

## Treatment and control

The matrix varies algorithm and environment; seeds provide replication within
each cell. Official OGBench hyperparameters are authoritative. Implementation
semantics are `upstream_semantic_match` for GCBC/QRL/CRL/HIQL and
`rlc_variant_documented` for GCIVL/GCIQL, whose post-gradient target Polyak
update is intentional and disclosed.

## Current status

The Study and original experiment README are design-time declarations and still
contain `formal_training_started: false`. At the 2026-09-10 review, the
external `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M22/` tree contained
partial completed artifacts and active GPU1 processes. The repository
`docs/9-9/M22_results/completion_manifest.json` remains partial/not paper-ready
and reports no completed aggregate cells. This is an explicit provenance
conflict, not a claim that missing values are zero.

The frozen worktree `/home/eai/Research/RLC-m22-baselines` is not modified by
this context work. Its resolved configs, runtime metadata, and current run
artifacts are authoritative for individual runs. Sampled runtime metadata and
checkpoint manifests record source commit
`2617634f29a883581b6fbf582db072f08b9d24ff` with `git_dirty: false`, matching
the frozen worktree.

## Interpretation boundary

Do not call the campaign byte-for-byte upstream equivalence. Do not aggregate
or report a cell as complete until all three seeds have valid `final@1M`
artifacts under a consistent source/config provenance. The audit gate passing
supports launch/readiness semantics; it is not a performance result.

## Relevant artifacts

- [`Study`](../../../../experiments/M22_puzzle_baselines_unified_rlc/study.yaml)
- [`original experiment README`](../../../../experiments/M22_puzzle_baselines_unified_rlc/README.md)
- [`semantic audit`](../../../../docs/9-9/canonical_semantics_audit.json)
- [`baseline audit`](../../../../docs/9-9/M22_upstream_baseline_audit.md)
- [`dataset audit`](../../../../docs/9-9/M22_dataset_audit.json)
- [`partial result manifest`](../../../../docs/9-9/M22_results/completion_manifest.json)
- External run root: `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M22/`

## Next action

Allow the frozen M22 campaign to finish without modifying its worktree. Then
rebuild the completion manifest and result tables from actual run metadata,
resolve source/protocol consistency, and only then make cross-algorithm or
cross-environment claims.
