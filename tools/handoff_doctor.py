#!/usr/bin/env python3
"""Read-only validation of the RLC research handoff contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impls.analysis.loaders import source_fingerprint
from impls.analysis.schema import AnalysisError, load_analysis_spec


def _run(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--spec",
        default="experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml",
    )
    parser.add_argument(
        "--analysis-output",
        default="/data/qijunrong/06-RL/offline-rl/exp/RLC/analyses/M10A/M10A-A001",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    spec_path = repo_root / args.spec
    spec = load_analysis_spec(spec_path)
    output = Path(args.analysis_output).resolve()
    metadata_path = output / "analysis_metadata.json"
    if not metadata_path.is_file():
        raise AnalysisError(f"missing analysis metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    head = _run(repo_root, "rev-parse", "HEAD")
    status = _run(repo_root, "status", "--porcelain", "--untracked-files=all")
    failures: list[str] = []
    if status:
        failures.append("git worktree is dirty")
    if metadata.get("status") != "completed":
        failures.append(f"analysis status is not completed: {metadata.get('status')}")
    if metadata.get("execution_mode") != "formal":
        failures.append(f"analysis execution_mode is not formal: {metadata.get('execution_mode')}")
    if metadata.get("analysis_git_dirty") is not False:
        failures.append("analysis metadata records a dirty worktree")
    if metadata.get("analysis_git_commit") != head:
        failures.append(
            f"analysis commit mismatch: metadata={metadata.get('analysis_git_commit')} head={head}"
        )
    if metadata.get("analysis_spec_fingerprint") != spec["_spec_fingerprint"]:
        failures.append("analysis spec fingerprint mismatch")
    source_path = Path(str(metadata.get("source_path", spec["source"]["path"]))).resolve()
    current_source_fingerprint, _ = source_fingerprint(source_path)
    if metadata.get("source_campaign_fingerprint") != current_source_fingerprint:
        failures.append("source campaign fingerprint mismatch")
    expected_ids = {spec["analysis_id"], spec["source"]["study_id"], spec["source"]["reevaluation_id"]}
    actual_ids = {metadata.get("analysis_id"), metadata.get("study_id"), metadata.get("reevaluation_id")}
    if actual_ids != expected_ids:
        failures.append(f"analysis identity mismatch: metadata={actual_ids} expected={expected_ids}")
    required_outputs = [
        "canonical_results.csv",
        "allocation_summary.csv",
        "task_allocation_summary.csv",
        "task_delta_vs_reference.csv",
        "paired_comparisons.csv",
    ]
    missing_outputs = [name for name in required_outputs if not (output / name).is_file()]
    if missing_outputs:
        failures.append(f"missing analysis outputs: {missing_outputs}")
    rendered_views = metadata.get("rendered_views", [])
    for rendered in rendered_views:
        for path in rendered.get("output_files", {}).values():
            if not (output / path).is_file():
                failures.append(f"missing rendered view output: {path}")
    result = {
        "repo_root": str(repo_root),
        "head": head,
        "git_clean": not bool(status),
        "analysis_output": str(output),
        "analysis_id": metadata.get("analysis_id"),
        "analysis_execution_mode": metadata.get("execution_mode"),
        "analysis_status": metadata.get("status"),
        "source_campaign_fingerprint": current_source_fingerprint,
        "analysis_spec_fingerprint": spec["_spec_fingerprint"],
        "rendered_view_count": len(rendered_views),
        "failures": failures,
        "handoff_ready": not failures,
    }
    if failures:
        raise AnalysisError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    try:
        result = run(_args())
    except (AnalysisError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
