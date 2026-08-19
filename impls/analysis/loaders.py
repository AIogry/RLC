"""Read-only loaders for reevaluation campaigns.

The loader treats the reevaluation directory as immutable input.  It validates
the manifest, run metadata, task summaries, episode identities, configuration
metadata, and checkpoint provenance before constructing a canonical long table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .schema import AnalysisError, CANONICAL_FIELDS, validate_canonical_row


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object: {path}")
    return value


def _number(value: Any, field: str, *, allow_none: bool = True) -> float | int | None:
    if value is None or value == "" or str(value).lower() == "null":
        if allow_none:
            return None
        raise AnalysisError(f"missing numeric field: {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid numeric field {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise AnalysisError(f"non-finite numeric field {field}: {value!r}")
    if number.is_integer():
        return int(number)
    return number


def _int(value: Any, field: str) -> int:
    number = _number(value, field, allow_none=False)
    assert number is not None
    return int(number)


def _float(value: Any, field: str) -> float:
    number = _number(value, field, allow_none=False)
    assert number is not None
    return float(number)


def _same(a: Any, b: Any, *, tolerance: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    return str(a) == str(b)


def _assert_same(name: str, expected: Any, actual: Any, *, tolerance: float = 1e-12) -> None:
    if not _same(expected, actual, tolerance=tolerance):
        raise AnalysisError(f"inconsistent {name}: expected {expected!r}, got {actual!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(source_root: str | Path) -> tuple[str, list[dict[str, str]]]:
    """Hash all regular files in a campaign in deterministic relative order."""

    root = Path(source_root).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        manifest.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), manifest


def _find_config_file(config_root: Path, config_id: str) -> Path:
    candidates = sorted(config_root.glob("*.yaml")) + sorted(config_root.glob("*.yml"))
    matches = []
    for path in candidates:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if isinstance(payload, Mapping) and str(payload.get("config_id")) == config_id:
            matches.append(path)
    if len(matches) != 1:
        raise AnalysisError(
            f"expected exactly one config file for {config_id}, found {len(matches)}"
        )
    return matches[0]


def _config_metadata(config_path: Path, config_id: str) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise AnalysisError(f"config is not a mapping: {config_path}")
    factors = config.get("factors") or {}
    if not isinstance(factors, Mapping):
        raise AnalysisError(f"factors is not a mapping: {config_path}")
    if str(config.get("config_id")) != config_id:
        raise AnalysisError(f"config ID mismatch in {config_path}")

    metadata = {
        "config_id": config_id,
        "config_slug": str(config.get("slug", "")),
        "environment": str(config.get("environment", "")),
        "budget": _number(factors.get("body_compute_budget"), f"{config_id}.body_compute_budget"),
        "k_high": _number(factors.get("high_iterations_K"), f"{config_id}.high_iterations_K"),
        "k_low": _number(factors.get("low_iterations_K"), f"{config_id}.low_iterations_K"),
        "high_fraction": _number(
            factors.get("high_budget_fraction"), f"{config_id}.high_budget_fraction"
        ),
        "low_fraction": _number(
            factors.get("low_budget_fraction"), f"{config_id}.low_budget_fraction"
        ),
        "config_path": str(config_path.resolve()),
    }
    if metadata["budget"] is not None and metadata["k_high"] is not None and metadata["k_low"] is not None:
        _assert_same(
            f"{config_id}.budget=K_H+K_L",
            metadata["budget"],
            float(metadata["k_high"]) + float(metadata["k_low"]),
        )
    if metadata["budget"] is not None and metadata["high_fraction"] is not None:
        _assert_same(
            f"{config_id}.high_fraction",
            metadata["high_fraction"],
            float(metadata["k_high"]) / float(metadata["budget"]),
        )
    if metadata["budget"] is not None and metadata["low_fraction"] is not None:
        _assert_same(
            f"{config_id}.low_fraction",
            metadata["low_fraction"],
            float(metadata["k_low"]) / float(metadata["budget"]),
        )
    if not metadata["config_slug"]:
        raise AnalysisError(f"missing slug for {config_id}")
    return metadata


def _source_run_dir(source_root: Path, output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = source_root / path
    return path.resolve()


def _base_row(
    *,
    campaign: Mapping[str, Any],
    manifest_row: Mapping[str, str],
    config: Mapping[str, Any],
    environment: str,
    seed: int,
    checkpoint_step: int,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "study_id": str(campaign["source_study_id"]),
        "reevaluation_id": str(campaign["reevaluation_id"]),
        "config_id": str(config["config_id"]),
        "config_slug": str(config["config_slug"]),
        "environment": environment,
        "training_seed": seed,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "budget": config["budget"],
        "k_high": config["k_high"],
        "k_low": config["k_low"],
        "high_fraction": config["high_fraction"],
        "low_fraction": config["low_fraction"],
        "source_run_dir": str(manifest_row["output_dir"]),
        "episode_sampling_se": "",
    }


def _validate_episode_rows(
    rows: list[dict[str, str]],
    *,
    config: Mapping[str, Any],
    seed: int,
    environment: str,
    episodes_per_task: int,
) -> None:
    required = {
        "config_id",
        "environment",
        "training_seed",
        "task_id",
        "task_name",
        "episode_index",
        "success",
        "paired_episode_id",
    }
    if not rows:
        raise AnalysisError(f"empty episode_results.csv for {config['config_id']} seed {seed}")
    missing = required.difference(rows[0])
    if missing:
        raise AnalysisError(f"episode_results.csv missing columns: {sorted(missing)}")
    episode_keys: set[tuple[int, str, int]] = set()
    paired_keys: set[tuple[int, str, str]] = set()
    task_indices: set[tuple[str, int]] = set()
    task_names: dict[str, str] = {}
    for row in rows:
        _assert_same("episode config_id", config["config_id"], row["config_id"])
        _assert_same("episode environment", environment, row["environment"])
        _assert_same("episode training_seed", seed, _int(row["training_seed"], "training_seed"))
        task_id = str(row["task_id"])
        index = _int(row["episode_index"], "episode_index")
        if index < 0 or index >= episodes_per_task:
            raise AnalysisError(f"episode index out of range: {task_id}/{index}")
        key = (seed, task_id, index)
        if key in episode_keys:
            raise AnalysisError(f"duplicate episode key: {key}")
        episode_keys.add(key)
        task_indices.add(key)
        task_name = str(row["task_name"])
        if task_id in task_names:
            _assert_same(f"task name for {task_id}", task_names[task_id], task_name)
        task_names[task_id] = task_name
        paired = str(row["paired_episode_id"])
        pair_key = (seed, task_id, paired)
        if pair_key in paired_keys:
            raise AnalysisError(f"duplicate paired episode key: {pair_key}")
        paired_keys.add(pair_key)
        success = _int(row["success"], "success")
        if success not in (0, 1):
            raise AnalysisError(f"success must be 0 or 1, got {success}")
        if not paired:
            raise AnalysisError(f"missing paired_episode_id for {key}")
    task_counts: dict[str, int] = {}
    for _, task_id, _ in task_indices:
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
    if not task_counts or any(count != episodes_per_task for count in task_counts.values()):
        raise AnalysisError(f"unexpected per-task episode counts: {task_counts}")


def _validate_task_summary(
    task_rows: list[dict[str, str]],
    episode_rows: list[dict[str, str]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    by_task: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for row in episode_rows:
        task_id = str(row["task_id"])
        by_task.setdefault(task_id, []).append(_int(row["success"], "success"))
        names[task_id] = str(row["task_name"])
    seen: set[str] = set()
    rates: dict[str, float] = {}
    for row in task_rows:
        task_id = str(row["task_id"])
        if task_id in seen:
            raise AnalysisError(f"duplicate task summary row: {task_id}")
        seen.add(task_id)
        if task_id not in by_task:
            raise AnalysisError(f"task summary has unknown task: {task_id}")
        count = _int(row["episode_count"], "episode_count")
        successes = _int(row["success_count"], "success_count")
        actual = by_task[task_id]
        _assert_same(f"episode count for {task_id}", len(actual), count)
        _assert_same(f"success count for {task_id}", sum(actual), successes)
        rate = _float(row["success_rate"], "success_rate")
        expected = sum(actual) / len(actual)
        if not math.isclose(rate, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise AnalysisError(f"task success rate mismatch for {task_id}")
        rates[task_id] = rate
    if set(rates) != set(by_task):
        raise AnalysisError("task summary does not cover exactly the episode tasks")
    return rates


def load_reevaluation(spec: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Load and validate one reevaluation campaign without writing to it."""

    source_root = Path(str(spec["source"]["path"])).expanduser().resolve()
    if not source_root.is_dir():
        raise AnalysisError(f"source reevaluation directory does not exist: {source_root}")
    campaign_path = source_root / "campaign_metadata.json"
    manifest_path = source_root / "manifest.csv"
    if not campaign_path.is_file() or not manifest_path.is_file():
        raise AnalysisError(f"source is missing campaign_metadata.json or manifest.csv: {source_root}")
    campaign = read_json(campaign_path)
    manifest = read_csv_rows(manifest_path)
    _assert_same("source study_id", spec["source"]["study_id"], campaign.get("source_study_id"))
    _assert_same("reevaluation_id", spec["source"]["reevaluation_id"], campaign.get("reevaluation_id"))
    if str(campaign.get("protocol", {}).get("task_selection")) != "all":
        raise AnalysisError("analysis requires task_selection=all")
    if not manifest:
        raise AnalysisError("manifest.csv is empty")
    required_manifest = {
        "study_id",
        "reevaluation_id",
        "config_id",
        "config_slug",
        "environment",
        "training_seed",
        "checkpoint_step",
        "checkpoint_sha256",
        "status",
        "output_dir",
    }
    missing = required_manifest.difference(manifest[0])
    if missing:
        raise AnalysisError(f"manifest.csv missing columns: {sorted(missing)}")
    config_root = Path(str(repo_root)).resolve() / str(spec["config_root"])
    config_cache: dict[str, dict[str, Any]] = {}
    canonical_rows: list[dict[str, Any]] = []
    episode_rows_all: list[dict[str, Any]] = []
    task_rows_all: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    expected_episodes = _int(campaign.get("protocol", {}).get("episodes_per_task"), "episodes_per_task")

    for manifest_row in sorted(manifest, key=lambda row: (row["config_id"], int(row["training_seed"]))):
        _assert_same("manifest study_id", spec["source"]["study_id"], manifest_row["study_id"])
        _assert_same("manifest reevaluation_id", spec["source"]["reevaluation_id"], manifest_row["reevaluation_id"])
        if manifest_row["status"] != "completed":
            raise AnalysisError(f"incomplete reevaluation row: {manifest_row}")
        config_id = str(manifest_row["config_id"])
        if config_id not in config_cache:
            config_cache[config_id] = _config_metadata(_find_config_file(config_root, config_id), config_id)
        config = config_cache[config_id]
        _assert_same("config slug", config["config_slug"], manifest_row["config_slug"])
        environment = str(manifest_row["environment"])
        if config["environment"] and config["environment"] != environment:
            _assert_same("config environment", config["environment"], environment)
        seed = _int(manifest_row["training_seed"], "training_seed")
        checkpoint_step = _int(manifest_row["checkpoint_step"], "checkpoint_step")
        checkpoint_hash = str(manifest_row["checkpoint_sha256"])
        run_dir = _source_run_dir(source_root, manifest_row["output_dir"])
        metadata_path = run_dir / "reevaluation_metadata.json"
        if not metadata_path.is_file():
            # Older reevaluation protocol revisions used metadata.json.  The
            # filename is a schema compatibility detail; both files are read
            # without changing the immutable source campaign.
            metadata_path = run_dir / "metadata.json"
        metadata = read_json(metadata_path)
        summary = read_json(run_dir / "summary.json")
        episodes = read_csv_rows(run_dir / "episode_results.csv")
        task_summary = read_csv_rows(run_dir / "task_summary.csv")
        checkpoint_metadata = metadata.get("checkpoint_metadata", {})
        metadata_config_id = metadata.get("config_id", metadata.get("source_config_id", checkpoint_metadata.get("config_id")))
        metadata_seed = metadata.get("training_seed", metadata.get("source_training_seed", checkpoint_metadata.get("seed")))
        _assert_same("run config_id", config_id, metadata_config_id)
        _assert_same("run training_seed", seed, metadata_seed)
        _assert_same("run checkpoint_step", checkpoint_step, metadata.get("checkpoint_step"))
        _assert_same("run checkpoint_sha256", checkpoint_hash, metadata.get("checkpoint_sha256"))
        _assert_same("summary checkpoint_step", checkpoint_step, summary.get("checkpoint_step"))
        _validate_episode_rows(
            episodes,
            config=config,
            seed=seed,
            environment=environment,
            episodes_per_task=expected_episodes,
        )
        task_rates = _validate_task_summary(task_summary, episodes)
        overall_value = summary.get(
            "evaluation/overall_success",
            summary.get("evaluation", {}).get("overall_success"),
        )
        overall = _float(overall_value, "overall_success")
        expected_overall = sum(task_rates.values()) / len(task_rates)
        if not math.isclose(overall, expected_overall, rel_tol=1e-12, abs_tol=1e-12):
            raise AnalysisError(f"overall success mismatch for {config_id} seed {seed}")
        base = _base_row(
            campaign=campaign,
            manifest_row=manifest_row,
            config=config,
            environment=environment,
            seed=seed,
            checkpoint_step=checkpoint_step,
            checkpoint_sha256=checkpoint_hash,
        )
        for episode in sorted(episodes, key=lambda row: (str(row["task_id"]), int(row["episode_index"]))):
            row = dict(base)
            row.update(
                {
                    "task_id": str(episode["task_id"]),
                    "task_name": str(episode["task_name"]),
                    "metric": "episode_success",
                    "value": _int(episode["success"], "success"),
                    "episode_sampling_se": "",
                    "episode_index": _int(episode["episode_index"], "episode_index"),
                    "paired_episode_id": str(episode["paired_episode_id"]),
                }
            )
            validate_canonical_row(row)
            canonical_rows.append(row)
            episode_rows_all.append(row)
        for task in sorted(task_summary, key=lambda row: str(row["task_id"])):
            row = dict(base)
            row.update(
                {
                    "task_id": str(task["task_id"]),
                    "task_name": str(task["task_name"]),
                    "metric": "task_success_rate",
                    "value": _float(task["success_rate"], "success_rate"),
                    "episode_sampling_se": _float(
                        task.get("success_standard_error"), "success_standard_error"
                    )
                    if task.get("success_standard_error") not in (None, "")
                    else "",
                    "episode_index": "",
                    "paired_episode_id": "",
                }
            )
            validate_canonical_row(row)
            canonical_rows.append(row)
            task_rows_all.append(row)
        overall_row = dict(base)
        overall_row.update(
            {
                "task_id": "overall",
                "task_name": "overall",
                "metric": "overall_success",
                "value": overall,
                "episode_sampling_se": _float(
                    summary.get("overall_episode_sampling_se"),
                    "overall_episode_sampling_se",
                )
                if summary.get("overall_episode_sampling_se") not in (None, "")
                else "",
                "episode_index": "",
                "paired_episode_id": "",
            }
        )
        validate_canonical_row(overall_row)
        canonical_rows.append(overall_row)
        overall_rows.append(overall_row)

    config_ids = {row["config_id"] for row in manifest}
    if str(campaign.get("completed_run_count")) != str(len(manifest)):
        raise AnalysisError("campaign completed_run_count does not match manifest")
    declared_count = campaign.get("source_run_count")
    if declared_count is not None and int(declared_count) != len(manifest):
        raise AnalysisError("campaign source_run_count does not match manifest")
    if str(campaign.get("reevaluation_id")) != str(spec["source"]["reevaluation_id"]):
        raise AnalysisError("campaign reevaluation ID mismatch")
    fingerprint, file_inventory = source_fingerprint(source_root)
    return {
        "source_root": str(source_root),
        "campaign_metadata": campaign,
        "manifest": manifest,
        "config_metadata": config_cache,
        "config_ids": sorted(config_ids),
        "canonical_rows": canonical_rows,
        "episode_rows": episode_rows_all,
        "task_rows": task_rows_all,
        "overall_rows": overall_rows,
        "source_fingerprint": fingerprint,
        "source_file_inventory": file_inventory,
        "canonical_fields": list(CANONICAL_FIELDS) + [
            "episode_index",
            "paired_episode_id",
            "source_run_dir",
            "episode_sampling_se",
        ],
    }


def source_file_hashes(source_root: str | Path) -> dict[str, str]:
    """Convenience helper used by tests to prove that loading is read-only."""

    _, inventory = source_fingerprint(source_root)
    return {item["path"]: item["sha256"] for item in inventory}
