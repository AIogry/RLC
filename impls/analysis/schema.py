"""Schemas and validation for reproducible reevaluation analyses.

This module deliberately contains no M10A-specific scientific assumptions.  A
study-specific YAML file supplies the reference configuration and task groups;
the loaders and statistics code consume that declaration generically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class AnalysisError(ValueError):
    """Raised when an analysis specification or source is inconsistent."""


CANONICAL_FIELDS = (
    "study_id",
    "reevaluation_id",
    "config_id",
    "config_slug",
    "environment",
    "training_seed",
    "checkpoint_step",
    "checkpoint_sha256",
    "budget",
    "k_high",
    "k_low",
    "high_fraction",
    "low_fraction",
    "task_id",
    "task_name",
    "metric",
    "value",
    "episode_sampling_se",
)

FIGURE_IDS = (
    "allocation_response_overall",
    "allocation_response_by_task",
    "allocation_response_focal_remaining",
    "allocation_seed_consistency",
    "task_delta_vs_reference",
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{name} must be a mapping")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{name} must be a non-empty string")
    return value.strip()


def _normalise_task_ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"{name} must be a non-empty list")
    result = [str(item) for item in value]
    if len(set(result)) != len(result):
        raise AnalysisError(f"{name} contains duplicate task IDs")
    return result


def load_analysis_spec(path: str | Path) -> dict[str, Any]:
    """Read and validate an analysis YAML specification.

    The returned object is a detached, normalised dictionary so callers cannot
    accidentally mutate the parsed YAML object while performing analysis.
    """

    spec_path = Path(path).resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    spec = dict(_require_mapping(loaded, "analysis spec"))

    spec["analysis_id"] = _require_nonempty_string(spec.get("analysis_id"), "analysis_id")
    source = dict(_require_mapping(spec.get("source"), "source"))
    source["study_id"] = _require_nonempty_string(source.get("study_id"), "source.study_id")
    source["reevaluation_id"] = _require_nonempty_string(
        source.get("reevaluation_id"), "source.reevaluation_id"
    )
    source["path"] = _require_nonempty_string(source.get("path"), "source.path")
    spec["source"] = source

    reference = dict(_require_mapping(spec.get("reference"), "reference"))
    reference["config_id"] = _require_nonempty_string(
        reference.get("config_id"), "reference.config_id"
    )
    reference["label"] = _require_nonempty_string(
        reference.get("label", "reference"), "reference.label"
    )
    spec["reference"] = reference

    groups = dict(_require_mapping(spec.get("task_groups"), "task_groups"))
    normalised_groups: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for group_name, group_value in groups.items():
        group = dict(_require_mapping(group_value, f"task_groups.{group_name}"))
        group["label"] = _require_nonempty_string(
            group.get("label", group_name), f"task_groups.{group_name}.label"
        )
        group["task_ids"] = _normalise_task_ids(
            group.get("task_ids"), f"task_groups.{group_name}.task_ids"
        )
        overlap = seen.intersection(group["task_ids"])
        if overlap:
            raise AnalysisError(
                f"task groups overlap on task IDs: {sorted(overlap)}"
            )
        seen.update(group["task_ids"])
        normalised_groups[str(group_name)] = group
    if not normalised_groups:
        raise AnalysisError("task_groups must not be empty")
    spec["task_groups"] = normalised_groups

    config_root = spec.get("config_root", "experiments/M10A_fixed_budget_placement/configs")
    spec["config_root"] = _require_nonempty_string(config_root, "config_root")

    figures = spec.get("figures", list(FIGURE_IDS))
    if not isinstance(figures, list) or not figures:
        raise AnalysisError("figures must be a non-empty list")
    figures = [str(item) for item in figures]
    unknown = sorted(set(figures).difference(FIGURE_IDS))
    if unknown:
        raise AnalysisError(f"unknown figure IDs: {unknown}")
    if len(set(figures)) != len(figures):
        raise AnalysisError("figures contains duplicate IDs")
    spec["figures"] = figures

    spec["_spec_path"] = str(spec_path)
    spec["_spec_fingerprint"] = analysis_spec_fingerprint(spec)
    return spec


def _fingerprintable_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in spec.items()
        if not str(key).startswith("_")
    }


def analysis_spec_fingerprint(spec: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for the declared analysis spec."""

    payload = json.dumps(
        _fingerprintable_spec(spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_canonical_row(row: Mapping[str, Any]) -> None:
    missing = [field for field in CANONICAL_FIELDS if field not in row]
    if missing:
        raise AnalysisError(f"canonical row is missing fields: {missing}")
