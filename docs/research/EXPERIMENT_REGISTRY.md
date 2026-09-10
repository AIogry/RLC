# Experiment registry

Compact index only. Exact configuration, lifecycle status, and metrics come
from the linked Study/configs and external run artifacts.

| ID | Purpose | Status | Study/path | Main artifact or note |
| --- | --- | --- | --- | --- |
| M15 | Puzzle structured-token MLP-Mixer foundation | completed foundation | historical source in commit `635e551` | [`M15 design`](../../docs/8-26/M15_puzzle_token_mixer.md) |
| M16A | Puzzle complexity × Mixer depth, alpha=0.3 | completed, 16/16 seed-0 runs | [`M16A study`](../../experiments/M16A_puzzle_mixer_depth_scaling/study.yaml) | [`M16A report`](../../docs/8-28/M16A_puzzle_mixer_depth_scaling_report.md) |
| M16B | Puzzle alpha=1.0 correction, Flat/S002 | completed, 8/8 seed-0 runs | [`M16B study`](../../experiments/M16B_puzzle_alpha_correction/study.yaml) | [`M16B design`](../../docs/8-28/M16B_puzzle_alpha_correction.md) |
| M16C | 4x4 S002 alpha exploration | completed, 4/4 seed-0 runs | [`M16C study`](../../experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml) | [`M16C design`](../../docs/8-29/M16C_puzzle_4x4_mixer_alpha_sweep.md) |
| M16D | Four Puzzle S002 Mixer-L2 cells at alpha=0.4 | attempt 001 complete; provenance-qualified analysis pending; attempt 000 retained | [`M16D study`](../../experiments/M16D_puzzle_mixer_alpha04_completion/study.yaml) | [`M16D context`](experiments/M16D/README.md) |
| M18 | Fixed-capacity recurrent computation scaling | completed; M18-D diagnostics documented | [`M18 study`](../../experiments/M18_puzzle_recurrent_compute_scaling/study.yaml) | [`M18-D analysis`](../../docs/9-3/M18-D_full_analysis_report.md) |
| M19A | Puzzle entity-wise MLP isolation control | external artifact exists; historical handoff/status requires reconciliation | `experiments/M19A_puzzle_entity_factorization_isolation/` | [`M19A handoff`](../../docs/9-4/M19A_implementation_handoff.md) |
| M21 | GPU concurrency throughput engineering benchmark | launcher/scheduler policy integrated; numeric report not in repo | [`M21 benchmark`](../../tools/m21_gpu_concurrency_benchmark.py) | [`M21 tests`](../../tests/experiment/test_m21_gpu_concurrency.py) |
| M22 | Six official OGBench Puzzle baselines in unified RLC | active/partial; not paper-ready | [`M22 study`](../../experiments/M22_puzzle_baselines_unified_rlc/study.yaml) | [`M22 context`](experiments/M22/README.md) |
| M23A | Puzzle direct-rollout behavioral audit | completed, 12/12 cells and 3,000 controlled paired rollouts; single-seed/provenance-qualified interpretation | [`M23A Study`](../../experiments/M23A_puzzle_direct_rollout_audit/study.yaml) | [`M23A authority`](experiments/M23A-F0/README.md), [`results brief`](experiments/M23A-F0/RESULTS_DISCUSSION_BRIEF.md) |
