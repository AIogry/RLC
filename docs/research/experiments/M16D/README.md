# M16D — Puzzle Mixer-L2 alpha=0.4 completion

## Scientific question

Under the historical M16B S002 Puzzle-token Mixer-L2 architecture and matched
protocol, what is the single-seed alpha=0.4 performance on the four canonical
Puzzle-play environments?

M16D completes only the four missing S002 cells:
`puzzle-{3x3,4x4,4x5,4x6}-play-v0`, GCIQL, actor+value+critic structured
computation, seed 0. It is a four-run completion/contrast, not an alpha search.

## Fixed protocol

- `structure=puzzle_tokens`, feed-forward `mlp_mixer`, `num_mixer_blocks=2`;
  token dim 128, token MLP 64, channel MLP 256, index embedding, mean readout,
  `tm_mode=none`.
- GCIQL `actor_loss=ddpgbc`, alpha=0.4, hidden dims 512×3, layer norm,
  batch=1024, 1,000,000 steps, lr=3e-4, discount=.99, expectile=.9,
  tau=.005.
- Evaluation every 100k, all five tasks, 20 episodes per task, temperature 0,
  no Gaussian noise/video; save best/last and periodic checkpoints.
- Formal namespace: `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16D/`.
  Attempt 001 uses `seed_000__attempt_001` paths.

## Treatment and control

The treatment is the explicit alpha=0.4 S002 cell. The historical anchor is
M16B S002 alpha=1.0 after resolved-config and source provenance validation;
M16A/M16B/M16C are not interchangeable controls. Flat, other Mixer depths,
other alpha values, extra seeds, relation/recurrent variants, and other
algorithms are outside this Study.

## Current status

Attempt 000 was stopped after P3X3 reached step 55k and P4X4 reached step 45k;
its artifacts are retained and not reinterpreted as completed results.
Attempt 001 was restarted after the generic JAX allocator fix. All four
attempt-001 `summary.json` files now record `status: completed` and the
corresponding final checkpoints/evaluation artifacts are present. Consult the
external run metadata for exact per-task metrics; do not copy a summary field
into a scientific conclusion without the protocol and provenance checks. The
sampled attempt-001 `runtime_metadata.json` records source commit
`fdbd7875a031c537b98b415a3c73ce0bf6a02721` and `git_dirty: true`; this
provenance must be resolved before treating the results as a clean-source
formal comparison.

## Interpretation boundary

Alpha=0.4 is an empirically motivated later Puzzle Mixer operating point. M16C
did not train alpha=0.4, and M16D does not identify an optimum. With one
training seed and 20 episodes per task, results are descriptive evidence only.
The alpha change is declared in each config's `factors.alpha` and
`agent_overrides.alpha`; no algorithm semantics are modified.

## Relevant artifacts

- [`study.yaml`](../../../../experiments/M16D_puzzle_mixer_alpha04_completion/study.yaml)
- [`configs/`](../../../../experiments/M16D_puzzle_mixer_alpha04_completion/configs/)
- [`attempt_001_restart_note.md`](../../../../experiments/M16D_puzzle_mixer_alpha04_completion/attempt_001_restart_note.md)
- [`M16A report`](../../../../docs/8-28/M16A_puzzle_mixer_depth_scaling_report.md)
- [`M16C design`](../../../../docs/8-29/M16C_puzzle_4x4_mixer_alpha_sweep.md)

## Next action

Validate the four completed cells' resolved configs, checkpoint/evaluation
completeness, and dirty-source provenance; then produce a compact M16D
analysis. Do not overwrite attempt 000 or promote a single-seed result to an
optimal-alpha claim.
