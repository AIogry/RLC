# M11B — Cross-Task and Cross-Algorithm Computation Demand

M11B is a controlled cross-task and cross-algorithm computation-demand study.
It keeps SingleState K4 non-residual frozen and changes only the designated
CRL actor/critic or HIQL high/low compute slot inside each environment.

The five references are: AntMaze-Large Navigate (campaign calibration
anchor), AntMaze-Giant Navigate (task-scale-associated shift), HumanoidMaze
Large Navigate (embodiment/control-complexity-associated shift), HumanoidMaze
Giant Navigate (difficult embodiment plus scale), and AntMaze-Large Stitch
(dataset compositionality and trajectory-stitching-associated demand).
These are descriptive cross-task comparisons, not strict causal environment
factorials.

The four new references each have four CRL conditions and four HIQL
conditions. Two fresh AntMaze-Large feedforward baselines are calibration
anchors, giving exactly 34 configurations. Formal training is intentionally
not part of this implementation/preflight stage.

The canonical task-context table is
`/home/eai/Research/offline-rl/docs/ALGORITHM_HYPERPARAMETERS.md`; the local
RLC agent defaults supply the shared objective and model defaults. The
requested `antmaze-large-stitch-v0` dataset must exist before a formal GO can
be issued; no similarly named dataset is an allowed substitute.
