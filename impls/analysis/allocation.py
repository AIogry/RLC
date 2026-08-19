"""Derived M10A allocation tables and paired comparisons.

The functions are intentionally parameterised by the analysis specification.
There is no special case for a particular task being difficult or long horizon;
task-group labels come only from YAML.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .schema import AnalysisError
from .statistics import join_paired_episode_results, summarize_values


def _sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in ("config_id", "task_id", "training_seed"))


def _allocation_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_id": config["config_id"],
        "config_slug": config["config_slug"],
        "budget": config["budget"],
        "k_high": config["k_high"],
        "k_low": config["k_low"],
        "high_fraction": config["high_fraction"],
        "low_fraction": config["low_fraction"],
    }


def _attach_seed_statistics(
    rows: list[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    seed_value_field: str,
    output_prefix: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        group_rows = grouped[key]
        seen: set[Any] = set()
        values: list[float] = []
        for row in group_rows:
            seed = row.get("training_seed")
            if seed in seen:
                raise AnalysisError(f"duplicate seed in derived group {key}: {seed}")
            seen.add(seed)
            values.append(float(row[seed_value_field]))
        summary = summarize_values(values)
        for row in sorted(group_rows, key=lambda item: str(item.get("training_seed"))):
            copy = dict(row)
            copy[f"{output_prefix}_mean"] = summary["mean"]
            copy[f"{output_prefix}_population_sd"] = summary["population_sd"]
            copy[f"{output_prefix}_sample_sd"] = summary["sample_sd"]
            copy[f"{output_prefix}_n_training_seeds"] = summary["n_training_seeds"]
            output.append(copy)
    return output


def _base_seed_table(
    source_rows: Sequence[Mapping[str, Any]],
    config_metadata: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    value_name: str,
    include_non_allocated: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        if row["metric"] != metric:
            continue
        config = config_metadata[str(row["config_id"])]
        if not include_non_allocated and config["budget"] is None:
            continue
        result = _allocation_fields(config)
        result.update(
            {
                "environment": row["environment"],
                "training_seed": row["training_seed"],
                value_name: float(row["value"]),
            }
        )
        if "task_id" in row and metric == "task_success_rate":
            result.update({"task_id": row["task_id"], "task_name": row["task_name"]})
        output.append(result)
    return output


def build_allocation_tables(bundle: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build all tabular outputs consumed by M10A-A001 figures."""

    config_metadata = bundle["config_metadata"]
    overall_seed = _base_seed_table(
        bundle["overall_rows"],
        config_metadata,
        metric="overall_success",
        value_name="seed_overall_success",
        include_non_allocated=False,
    )
    overall = _attach_seed_statistics(
        overall_seed,
        group_fields=("config_id", "budget", "high_fraction"),
        seed_value_field="seed_overall_success",
        output_prefix="overall_success",
    )

    task_seed = _base_seed_table(
        bundle["task_rows"],
        config_metadata,
        metric="task_success_rate",
        value_name="seed_task_success_rate",
        include_non_allocated=True,
    )
    task = _attach_seed_statistics(
        task_seed,
        group_fields=("config_id", "task_id", "budget", "high_fraction"),
        seed_value_field="seed_task_success_rate",
        output_prefix="task_success_rate",
    )

    reference_id = str(spec["reference"]["config_id"])
    task_by_key: dict[tuple[str, str, Any], Mapping[str, Any]] = {}
    for row in task_seed:
        key = (str(row["config_id"]), str(row["task_id"]), row["training_seed"])
        if key in task_by_key:
            raise AnalysisError(f"duplicate task seed row: {key}")
        task_by_key[key] = row
    reference_rows = {
        (str(row["task_id"]), row["training_seed"]): row
        for row in task_seed
        if str(row["config_id"]) == reference_id
    }
    if not reference_rows:
        raise AnalysisError(f"reference config has no task-level rows: {reference_id}")
    delta_seed: list[dict[str, Any]] = []
    for row in task_seed:
        task_id = str(row["task_id"])
        ref = reference_rows.get((task_id, row["training_seed"]))
        if ref is None:
            raise AnalysisError(
                f"reference missing task/seed pair: {task_id}/{row['training_seed']}"
            )
        result = dict(_allocation_fields(config_metadata[str(row["config_id"])]))
        result.update(
            {
                "reference_config": reference_id,
                "target_config": row["config_id"],
                "task_id": task_id,
                "task_name": row["task_name"],
                "training_seed": row["training_seed"],
                "reference_task_success_rate": float(ref["seed_task_success_rate"]),
                "target_task_success_rate": float(row["seed_task_success_rate"]),
                "seed_delta": float(row["seed_task_success_rate"])
                - float(ref["seed_task_success_rate"]),
            }
        )
        delta_seed.append(result)
    delta = _attach_seed_statistics(
        delta_seed,
        group_fields=("target_config", "task_id", "budget", "high_fraction"),
        seed_value_field="seed_delta",
        output_prefix="delta",
    )

    task_group_seed: list[dict[str, Any]] = []
    groups = spec["task_groups"]
    task_lookup: dict[tuple[str, Any, str], Mapping[str, Any]] = {
        (str(row["config_id"]), row["training_seed"], str(row["task_id"])): row
        for row in task_seed
    }
    for config_id in sorted(config_metadata):
        config = config_metadata[config_id]
        if config["budget"] is None:
            continue
        seeds = sorted(
            {row["training_seed"] for row in task_seed if str(row["config_id"]) == config_id},
            key=str,
        )
        for seed in seeds:
            for group_name, group in groups.items():
                group_rows = [
                    task_lookup[(config_id, seed, task_id)]
                    for task_id in group["task_ids"]
                    if (config_id, seed, task_id) in task_lookup
                ]
                if len(group_rows) != len(group["task_ids"]):
                    raise AnalysisError(
                        f"task group {group_name} incomplete for {config_id} seed {seed}"
                    )
                result = dict(_allocation_fields(config))
                result.update(
                    {
                        "group_name": str(group_name),
                        "group_label": group["label"],
                        "training_seed": seed,
                        "seed_group_success_rate": sum(
                            float(item["seed_task_success_rate"]) for item in group_rows
                        )
                        / len(group_rows),
                    }
                )
                task_group_seed.append(result)
    focal_remaining = _attach_seed_statistics(
        task_group_seed,
        group_fields=("group_name", "config_id", "budget", "high_fraction"),
        seed_value_field="seed_group_success_rate",
        output_prefix="group_success_rate",
    )

    paired: list[dict[str, Any]] = []
    for target_config in sorted(bundle["config_ids"]):
        if target_config == reference_id:
            continue
        paired.extend(
            join_paired_episode_results(
                bundle["episode_rows"],
                reference_config=reference_id,
                target_config=target_config,
            )
        )

    return {
        "allocation_summary": sorted(overall, key=_sort_key),
        "task_allocation_summary": sorted(task, key=_sort_key),
        "task_delta_vs_reference": sorted(delta, key=_sort_key),
        "focal_remaining_summary": sorted(focal_remaining, key=_sort_key),
        "paired_comparisons": paired,
    }
