# M22 — OGBench Puzzle Baselines in Unified RLC

This Study declares the six official OGBench Puzzle baselines in the unified
RLC runtime. It contains 24 algorithm/environment configurations and expands
each configuration over seeds `0`, `1`, and `2` for 72 planned Runs.

The canonical agent defaults are resolved from the selected RLC agent config.
Each configuration records the audited source reference, resolved baseline
fields, official Puzzle overrides, and the disabled canonical computation
slots. GCIVL and GCIQL retain the user-frozen RLC post-gradient target Polyak
update and are documented RLC variants, not byte-for-byte upstream copies.

The formal run namespace is `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M22`.
GPU placement is a launcher concern; the intended launch policy is physical
GPU 1 with two worker slots and is not a scientific factor in this Study.

Use `tools/m22_doctor.py` before any launch. Formal training is intentionally
not started by creating this Study.
