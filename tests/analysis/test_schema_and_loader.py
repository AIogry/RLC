import csv
import json
from pathlib import Path

import yaml

from impls.analysis.loaders import load_reevaluation, source_file_hashes
from impls.analysis.schema import analysis_spec_fingerprint, load_analysis_spec


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_loader_is_read_only_and_validates_explicit_allocation(tmp_path):
    repo = tmp_path / "repo"
    config_root = repo / "configs"
    config_root.mkdir(parents=True)
    config = {
        "config_id": "C001",
        "slug": "alloc_h1_l1",
        "factors": {
            "high_iterations_K": 1,
            "low_iterations_K": 1,
            "body_compute_budget": 2,
            "high_budget_fraction": 0.5,
            "low_budget_fraction": 0.5,
        },
    }
    (config_root / "c001.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    source = tmp_path / "reevaluation"
    run = source / "C001__alloc_h1_l1" / "env-v0" / "seed_000"
    run.mkdir(parents=True)
    campaign = {
        "source_study_id": "S",
        "reevaluation_id": "R",
        "completed_run_count": 1,
        "source_run_count": 1,
        "protocol": {"episodes_per_task": 2, "task_selection": "all"},
    }
    (source / "campaign_metadata.json").write_text(json.dumps(campaign), encoding="utf-8")
    _write_csv(
        source / "manifest.csv",
        [
            "study_id", "reevaluation_id", "config_id", "config_slug", "environment",
            "training_seed", "checkpoint_step", "checkpoint_sha256", "status", "output_dir",
        ],
        [{
            "study_id": "S", "reevaluation_id": "R", "config_id": "C001",
            "config_slug": "alloc_h1_l1", "environment": "env-v0", "training_seed": 0,
            "checkpoint_step": 10, "checkpoint_sha256": "abc", "status": "completed",
            "output_dir": str(run),
        }],
    )
    (run / "reevaluation_metadata.json").write_text(json.dumps({
        "config_id": "C001", "training_seed": 0, "checkpoint_step": 10,
        "checkpoint_sha256": "abc",
    }), encoding="utf-8")
    episode_fields = [
        "config_id", "environment", "training_seed", "task_id", "task_name",
        "episode_index", "success", "paired_episode_id",
    ]
    _write_csv(run / "episode_results.csv", episode_fields, [
        {"config_id": "C001", "environment": "env-v0", "training_seed": 0,
         "task_id": "1", "task_name": "task1", "episode_index": 0, "success": 1,
         "paired_episode_id": "task1_ep0"},
        {"config_id": "C001", "environment": "env-v0", "training_seed": 0,
         "task_id": "1", "task_name": "task1", "episode_index": 1, "success": 0,
         "paired_episode_id": "task1_ep1"},
        {"config_id": "C001", "environment": "env-v0", "training_seed": 0,
         "task_id": "2", "task_name": "task2", "episode_index": 0, "success": 1,
         "paired_episode_id": "task2_ep0"},
        {"config_id": "C001", "environment": "env-v0", "training_seed": 0,
         "task_id": "2", "task_name": "task2", "episode_index": 1, "success": 1,
         "paired_episode_id": "task2_ep1"},
    ])
    _write_csv(run / "task_summary.csv", ["task_id", "task_name", "episode_count", "success_count", "success_rate"], [
        {"task_id": "1", "task_name": "task1", "episode_count": 2, "success_count": 1, "success_rate": 0.5},
        {"task_id": "2", "task_name": "task2", "episode_count": 2, "success_count": 2, "success_rate": 1.0},
    ])
    (run / "summary.json").write_text(json.dumps({
        "checkpoint_step": 10, "evaluation/overall_success": 0.75,
    }), encoding="utf-8")
    spec = load_analysis_spec_from_mapping(tmp_path, source, repo)
    before = source_file_hashes(source)
    bundle = load_reevaluation(spec, repo)
    after = source_file_hashes(source)
    assert before == after
    assert bundle["config_metadata"]["C001"]["budget"] == 2
    assert len(bundle["episode_rows"]) == 4
    assert len(bundle["task_rows"]) == 2


def load_analysis_spec_from_mapping(tmp_path, source, repo):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({
        "analysis_id": "A001",
        "source": {"study_id": "S", "reevaluation_id": "R", "path": str(source)},
        "config_root": "configs",
        "reference": {"config_id": "C001", "label": "reference"},
        "task_groups": {"all": {"label": "all", "task_ids": ["1", "2"]}},
    }), encoding="utf-8")
    spec = load_analysis_spec(path)
    # The loader resolves config_root relative to repo_root, so leave only the
    # repository-relative suffix in this synthetic specification.
    assert analysis_spec_fingerprint(spec) == spec["_spec_fingerprint"]
    return spec
