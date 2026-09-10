# RLC repository rules

This file contains only long-lived collaboration and reproducibility rules. It
is not a milestone log and must not contain temporary results, process IDs, or
current run progress.

## Scientific semantics and implementation

- State the scientific question, manipulated factors, fixed factors, and
  interpretation boundary separately from implementation details.
- Treat executable semantics and resolved runtime configuration as evidence;
  do not classify a difference from source syntax alone when a controlled
  semantic or gradient-flow check is possible.
- Keep infrastructure, diagnostics, and performance claims separate. A
  passing smoke test or audit is not a scientific result.

## Study and Configuration organization

- A Study declares one scientific question and its factor matrix.
- A Configuration has a stable ID and declares one factor combination without
  a seed. Scientific factors belong in Study/config files, not in launch
  scripts or ad-hoc code paths.
- A Run is a Configuration + environment + seed + frozen provenance. Do not
  silently rename, overwrite, or reinterpret an existing Run.
- Prefer reusable generic infrastructure over milestone-named one-off
  functions, scripts, or files.

## Running and frozen worktree safety

- Never modify a running or frozen experiment worktree, run directory,
  checkpoint, or result artifact while diagnosing or documenting it.
- Use a separate feature worktree for repository changes when another tree is
  running formal work. Do not use a mutable development tree for formal
  training.
- GPU assignment is operational metadata, not a scientific factor. Respect
  the physical GPU and jobs-per-GPU policy declared by the active Study.

## Git authority

- Record the actual source commit, branch/worktree, and dirty state; do not
  substitute a prompt-provided or remembered SHA for observed provenance.
- Preserve existing history and user changes. Do not reset, clean, rewrite, or
  force-update history.
- Commit and push are user-controlled unless the user explicitly authorizes
  them in the current task.

## Targeted regression policy

- For a shared-runtime or experiment-management change, run the smallest
  dependency-aware set: changed-layer tests plus the directly affected Study,
  lifecycle, launcher, and provenance tests.
- Do not run the full historical regression suite merely to prepare context
  documentation. Record what was and was not checked.
- A regression failure must remain visible; do not weaken assertions or hide a
  failure to make a gate pass.

## Formal-run reproducibility

- Before formal execution, validate the Study/config matrix, dataset identity,
  protocol, output namespace, GPU policy, and clean frozen source.
- Formal training and long-running/high-cost formal campaigns require a clean
  detached frozen worktree. A non-training post-hoc diagnostic or reevaluation
  may instead use a clean, dedicated feature worktree only when the user
  explicitly authorizes that exception for the current task. Record its
  worktree path, branch, commit, clean state, rationale, GPU assignment, and
  output namespace in the execution manifest and durable study context; do not
  modify that worktree after preflight or while it is executing.
- Use explicit seeds and protocol arguments. Let resolved configs and runtime
  metadata record the effective values, devices, source SHA, and lifecycle
  status.
- Preserve partial or failed artifacts for diagnosis. Any retry must have an
  explicit attempt identity and must not overwrite the original attempt.

## Context discipline

- Keep project context compact and durable. Link to source files, audits,
  manifests, and external artifact roots instead of copying logs, CSVs,
  checkpoints, or large metadata blobs.
- Clearly label verified fact, interpretation, hypothesis, and unresolved
  ambiguity. If documentation conflicts with executable artifacts, report the
  conflict rather than silently reconciling it.
- Update project-level context after a material decision or completed study,
  while keeping transient telemetry in the run artifacts.

## Handoff expectations

Every handoff should state the worktree/branch and source SHA, files changed,
checks performed, source locations consulted, unresolved conflicts, actions
intentionally not taken, and the next human-controlled action. Formal launch,
Git publication, and changes to frozen experiment state require explicit human
ownership.

## Human-controlled Git and launch guidance

- When repository work is ready, but commit, push, merge, or launch has not
  been explicitly authorized in the current task, do not perform it. The
  handoff must still provide a copy-paste-ready sequence tailored to the
  observed worktree/branch/SHA, target branch, remote state, and any existing
  user changes. Distinguish required review/validation steps from optional
  remote publication or pull-request steps.
- Before suggesting a merge, inspect and report whether the target worktree is
  clean and whether uncommitted user changes need to be preserved. Do not
  recommend reset, clean, force-push, or dropping a stash by default.
- Before suggesting a formal experiment launch, inspect and report the active
  Study protocol, source/provenance gates, output-root emptiness, current GPU
  process occupancy, and the exact execution source SHA. Give the exact launch
  command, explicit GPU assignment, expected output namespace, and monitoring
  command when known.
- Never infer that an available GPU may be shared with another formal study.
  Respect the active Study's declared GPU policy and the user's explicit GPU
  allocation; recheck live GPU occupancy immediately before launch.
- State whether the launch source is a frozen worktree or a user-authorized
  clean feature-worktree exception. If it is the latter, make the exception and
  its reproducibility boundary explicit in the handoff and execution record.
