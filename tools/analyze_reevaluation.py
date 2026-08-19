#!/usr/bin/env python3
"""Run a provenance-aware analysis over an immutable reevaluation campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impls.analysis.allocation import build_allocation_tables
from impls.analysis.loaders import load_reevaluation, source_fingerprint
from impls.analysis.plotting import generate_figures, write_rows_csv
from impls.analysis.schema import AnalysisError, load_analysis_spec


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _git_state(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    return {
        "analysis_git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "analysis_git_dirty": bool(status),
        "analysis_git_status": status.splitlines(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="analysis YAML specification")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="load and validate without writing")
    mode.add_argument("--execute", action="store_true", help="write analysis outputs")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root used to resolve config_root",
    )
    parser.add_argument(
        "--output-root",
        default="/data/qijunrong/06-RL/offline-rl/exp/RLC/analyses",
        help="root for canonical analysis outputs",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="explicit smoke/test mode; permits a dirty worktree",
    )
    return parser.parse_args()


def _metadata(
    spec: dict[str, Any],
    bundle: dict[str, Any],
    git_state: dict[str, Any],
    *,
    started_at: str,
    ended_at: str,
    status: str,
    execution_mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "analysis_id": spec["analysis_id"],
        "study_id": spec["source"]["study_id"],
        "reevaluation_id": spec["source"]["reevaluation_id"],
        "source_path": bundle["source_root"],
        "source_campaign_fingerprint": bundle["source_fingerprint"],
        "source_campaign_protocol_fingerprint": bundle["campaign_metadata"].get(
            "protocol_fingerprint"
        ),
        "source_file_inventory": bundle["source_file_inventory"],
        "analysis_output_path": str(output_dir),
        "analysis_spec_path": spec["_spec_path"],
        "analysis_spec_fingerprint": spec["_spec_fingerprint"],
        **git_state,
        "execution_mode": execution_mode,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "reference": spec["reference"],
        "task_groups": spec["task_groups"],
        "figure_registry": spec["figures"],
        "statistical_definitions": {
            "training_seed_unit": "training_seed",
            "mean": "arithmetic mean across individual training seeds",
            "population_sd": "sqrt(sum((x-mean)^2)/n) across training seeds",
            "sample_sd": "sqrt(sum((x-mean)^2)/(n-1)); zero for n<2",
            "episode_sampling_uncertainty": "retained in source summaries; not pooled as model replicates",
            "paired_comparison": "exact join on training_seed, task_id, paired_episode_id",
        },
        "output_inventory": {
            "canonical_results": "canonical_results.csv",
            "allocation_summary": "allocation_summary.csv",
            "task_allocation_summary": "task_allocation_summary.csv",
            "task_delta_vs_reference": "task_delta_vs_reference.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "figures": [
                f"figures/{figure_id}.{extension}"
                for figure_id in spec["figures"]
                for extension in ("pdf", "png", "csv")
            ],
        },
    }


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    spec = load_analysis_spec(args.spec)
    started_at = _now()
    git_state = _git_state(repo_root)
    if args.execute and git_state["analysis_git_dirty"] and not args.smoke:
        raise AnalysisError(
            "formal analysis execution requires a clean worktree; use --smoke only for a "
            "deliberate smoke/test run"
        )
    bundle = load_reevaluation(spec, repo_root)
    if args.dry_run:
        print(json.dumps(
            {
                "analysis_id": spec["analysis_id"],
                "source_path": bundle["source_root"],
                "source_campaign_fingerprint": bundle["source_fingerprint"],
                "config_count": len(bundle["config_ids"]),
                "run_count": len(bundle["manifest"]),
                "canonical_row_count": len(bundle["canonical_rows"]),
                "source_unchanged_check": "not applicable (read-only dry-run)",
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / str(spec["source"]["study_id"])
        / str(spec["analysis_id"])
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AnalysisError(f"analysis output already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_allocation_tables(bundle, spec)
    write_rows_csv(output_dir / "canonical_results.csv", bundle["canonical_rows"])
    write_rows_csv(output_dir / "allocation_summary.csv", tables["allocation_summary"])
    write_rows_csv(
        output_dir / "task_allocation_summary.csv", tables["task_allocation_summary"]
    )
    write_rows_csv(
        output_dir / "task_delta_vs_reference.csv", tables["task_delta_vs_reference"]
    )
    write_rows_csv(output_dir / "paired_comparisons.csv", tables["paired_comparisons"])
    generate_figures(
        tables,
        output_dir / "figures",
        figure_ids=spec["figures"],
        reference_config=str(spec["reference"]["config_id"]),
    )
    after_fingerprint, _ = source_fingerprint(bundle["source_root"])
    if after_fingerprint != bundle["source_fingerprint"]:
        raise AnalysisError("source reevaluation files changed during analysis")
    ended_at = _now()
    metadata = _metadata(
        spec,
        bundle,
        git_state,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        execution_mode="smoke" if args.smoke else "formal",
        output_dir=output_dir,
    )
    _write_json(output_dir / "analysis_metadata.json", metadata)
    print(json.dumps(
        {
            "analysis_id": spec["analysis_id"],
            "output_dir": str(output_dir),
            "execution_mode": metadata["execution_mode"],
            "source_campaign_fingerprint": bundle["source_fingerprint"],
            "canonical_row_count": len(bundle["canonical_rows"]),
            "paired_row_count": len(tables["paired_comparisons"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def main() -> int:
    try:
        return run(_parse_args())
    except AnalysisError as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
