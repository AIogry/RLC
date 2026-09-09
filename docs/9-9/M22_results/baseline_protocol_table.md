# M22 baseline protocol

| Algorithm | Official Puzzle override | Implementation semantics | Documented deviations | Dataset provenance | Training protocol | Evaluation protocol |
|---|---|---|---|---|---|---|
| GCBC | none | upstream_semantic_match | — | official OGBench dataset + audited SHA-256 | 1,000,000 steps; batch=1,024; log=5,000; save=100,000 | all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M |
| GCIVL | alpha=10 | rlc_variant_documented | post-gradient target Polyak update | official OGBench dataset + audited SHA-256 | 1,000,000 steps; batch=1,024; log=5,000; save=100,000 | all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M |
| GCIQL | alpha=1 | rlc_variant_documented | post-gradient target Polyak update | official OGBench dataset + audited SHA-256 | 1,000,000 steps; batch=1,024; log=5,000; save=100,000 | all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M |
| QRL | alpha=0.3 | upstream_semantic_match | — | official OGBench dataset + audited SHA-256 | 1,000,000 steps; batch=1,024; log=5,000; save=100,000 | all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M |
| CRL | alpha=3 | upstream_semantic_match | — | official OGBench dataset + audited SHA-256 | 1,000,000 steps; batch=1,024; log=5,000; save=100,000 | all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M |
| HIQL | high_alpha=3, low_alpha=3, subgoal_steps=10 | upstream_semantic_match | — | official OGBench dataset + audited SHA-256 | 1,000,000 steps; batch=1,024; log=5,000; save=100,000 | all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M |

Protocol fields are declared in `/home/eai/Research/RLC/experiments/M22_puzzle_baselines_unified_rlc/study.yaml` and were not inferred from result files.
`eval_episodes=50` means 50 episodes for each task in the canonical evaluator.
