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

_SUCCESS_FIGURE_IDS = {
    "allocation_response_overall",
    "allocation_response_by_task",
    "allocation_response_focal_remaining",
    "allocation_seed_consistency",
}
_DELTA_FIGURE_IDS = {"task_delta_vs_reference"}
_SPLIT_FIGURE_IDS = {
    "allocation_response_focal_remaining",
    "task_delta_vs_reference",
}
_VIEW_IDS = {"full", "zoom", "split_zoom"}
_VIEW_KEYS = {"y_range", "panels", "panel_layout", "subtitle"}
_PANEL_KEYS = {"y_range", "subtitle"}


def _default_full_range(figure_id: str) -> list[float]:
    """Backward-compatible, predeclared full range for legacy figure lists."""

    if figure_id in _SUCCESS_FIGURE_IDS:
        return [0.0, 1.0]
    if figure_id in _DELTA_FIGURE_IDS:
        return [-1.0, 1.0]
    raise AnalysisError(f"cannot determine default view range for {figure_id}")


def _validate_range(value: Any, *, name: str, figure_id: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AnalysisError(f"{name}.y_range must be a two-element list")
    try:
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{name}.y_range must contain numbers") from exc
    if not all(map(lambda item: item == item and abs(item) != float("inf"), (lower, upper))):
        raise AnalysisError(f"{name}.y_range must contain finite numbers")
    if lower >= upper:
        raise AnalysisError(f"{name}.y_range requires y_min < y_max")
    if figure_id in _SUCCESS_FIGURE_IDS and not (0.0 <= lower < upper <= 1.0):
        raise AnalysisError(
            f"{name}.y_range is not a valid success-rate range: [{lower}, {upper}]"
        )
    if figure_id in _DELTA_FIGURE_IDS and not (-1.0 <= lower < upper <= 1.0):
        raise AnalysisError(
            f"{name}.y_range is not a valid success-rate delta range: [{lower}, {upper}]"
        )
    return [lower, upper]


def _normalise_view(
    figure_id: str,
    view_id: str,
    value: Any,
    *,
    task_group_names: set[str],
) -> dict[str, Any]:
    view = dict(_require_mapping(value, f"figures.{figure_id}.views.{view_id}"))
    unknown = sorted(set(view).difference(_VIEW_KEYS))
    if unknown:
        raise AnalysisError(
            f"unknown view fields for {figure_id}__{view_id}: {unknown}"
        )
    if view_id not in _VIEW_IDS:
        raise AnalysisError(f"unknown view ID for {figure_id}: {view_id}")
    if view_id == "split_zoom":
        if figure_id not in _SPLIT_FIGURE_IDS:
            raise AnalysisError(f"split_zoom is not supported for {figure_id}")
        if "panels" not in view:
            raise AnalysisError(f"{figure_id}__split_zoom requires panels")
        panels = dict(_require_mapping(view["panels"], f"{figure_id}__split_zoom.panels"))
        if set(panels) != task_group_names:
            raise AnalysisError(
                f"{figure_id}__split_zoom panels must exactly match task groups "
                f"{sorted(task_group_names)}"
            )
        normalised_panels: dict[str, dict[str, Any]] = {}
        for panel_id, panel_value in panels.items():
            panel = dict(_require_mapping(panel_value, f"panel {panel_id}"))
            unknown_panel = sorted(set(panel).difference(_PANEL_KEYS))
            if unknown_panel:
                raise AnalysisError(
                    f"unknown panel fields for {figure_id}__split_zoom/{panel_id}: "
                    f"{unknown_panel}"
                )
            normalised_panels[str(panel_id)] = {
                "y_range": _validate_range(
                    panel.get("y_range"),
                    name=f"{figure_id}__split_zoom/{panel_id}",
                    figure_id=figure_id,
                ),
                "subtitle": panel.get("subtitle"),
            }
        if "y_range" in view:
            raise AnalysisError(f"{figure_id}__split_zoom cannot also define y_range")
        return {
            "view_id": view_id,
            "panel_layout": str(view.get("panel_layout", "horizontal")),
            "panels": normalised_panels,
            "subtitle": view.get("subtitle", "Zoomed y-axis"),
        }
    if "y_range" not in view:
        raise AnalysisError(f"{figure_id}__{view_id} requires y_range")
    if "panels" in view:
        raise AnalysisError(f"{figure_id}__{view_id} cannot define panels")
    return {
        "view_id": view_id,
        "y_range": _validate_range(
            view["y_range"],
            name=f"{figure_id}__{view_id}",
            figure_id=figure_id,
        ),
        "panel_layout": str(view.get("panel_layout", "single")),
        "subtitle": view.get("subtitle", "Zoomed y-axis" if view_id == "zoom" else None),
    }


def _normalise_figure_views(
    raw_figures: Any,
    *,
    task_group_names: set[str],
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    if isinstance(raw_figures, list):
        figure_ids = [str(item) for item in raw_figures]
        raw_view_mapping = {
            figure_id: {"full": {"y_range": _default_full_range(figure_id)}}
            for figure_id in figure_ids
        }
    elif isinstance(raw_figures, Mapping):
        figure_ids = [str(item) for item in raw_figures]
        raw_view_mapping = {}
        for figure_id, figure_value in raw_figures.items():
            figure = dict(_require_mapping(figure_value, f"figures.{figure_id}"))
            unknown_figure = sorted(set(figure).difference({"views"}))
            if unknown_figure:
                raise AnalysisError(
                    f"unknown figure fields for {figure_id}: {unknown_figure}"
                )
            views = figure.get("views")
            if views is None:
                views = {"full": {"y_range": _default_full_range(str(figure_id))}}
            views = dict(_require_mapping(views, f"figures.{figure_id}.views"))
            raw_view_mapping[str(figure_id)] = views
    else:
        raise AnalysisError("figures must be a list (legacy) or mapping with views")
    if not figure_ids:
        raise AnalysisError("figures must not be empty")
    if len(set(figure_ids)) != len(figure_ids):
        raise AnalysisError("figures contains duplicate IDs")
    unknown_figures = sorted(set(figure_ids).difference(FIGURE_IDS))
    if unknown_figures:
        raise AnalysisError(f"unknown figure IDs: {unknown_figures}")
    normalised: dict[str, dict[str, dict[str, Any]]] = {}
    for figure_id in figure_ids:
        views = raw_view_mapping[figure_id]
        if not views:
            raise AnalysisError(f"figures.{figure_id}.views must not be empty")
        normalised_views: dict[str, dict[str, Any]] = {}
        for view_id, view_value in views.items():
            normalised_views[str(view_id)] = _normalise_view(
                figure_id,
                str(view_id),
                view_value,
                task_group_names=task_group_names,
            )
        if "full" not in normalised_views:
            raise AnalysisError(f"figures.{figure_id}.views must include full")
        normalised[figure_id] = normalised_views
    return figure_ids, normalised


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

    figures, figure_views = _normalise_figure_views(
        spec.get("figures", list(FIGURE_IDS)),
        task_group_names=set(normalised_groups),
    )
    spec["figures"] = figures
    spec["figure_views"] = figure_views

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
