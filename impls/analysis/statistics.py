"""Small, explicit statistical helpers used by reevaluation analyses.

Training seeds are the independent model replicates in this analysis.  Episode
rows are only used to calculate each seed's task metric and paired outcomes;
they are never pooled as independent training-seed observations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .schema import AnalysisError


def population_sd(values: Sequence[float]) -> float:
    if not values:
        raise AnalysisError("population SD requires at least one value")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise AnalysisError("cannot summarize an empty value sequence")
    values = [float(value) for value in values]
    return {
        "n_training_seeds": len(values),
        "mean": sum(values) / len(values),
        "population_sd": population_sd(values),
        "sample_sd": sample_sd(values),
    }


def _key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def seed_aggregate(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    value_field: str = "value",
    seed_field: str = "training_seed",
    seed_value_field: str = "seed_value",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return seed-level rows and one aggregate row per declared group.

    The function rejects duplicate seed observations within a group.  This is
    important because a duplicated episode or task summary must not silently
    receive additional weight.
    """

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row, group_fields)].append(row)
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        group_rows = grouped[group_key]
        seen_seeds: set[Any] = set()
        values: list[float] = []
        for source in sorted(group_rows, key=lambda row: str(row.get(seed_field))):
            seed = source.get(seed_field)
            if seed in seen_seeds:
                raise AnalysisError(f"duplicate training seed in group {group_key}: {seed}")
            seen_seeds.add(seed)
            value = float(source[value_field])
            values.append(value)
            output = dict(source)
            output[seed_value_field] = value
            seed_rows.append(output)
        summary = summarize_values(values)
        aggregate = {field: value for field, value in zip(group_fields, group_key)}
        aggregate.update(summary)
        aggregate_rows.append(aggregate)
    return seed_rows, aggregate_rows


def join_paired_episode_results(
    episode_rows: Iterable[Mapping[str, Any]],
    *,
    reference_config: str,
    target_config: str,
) -> list[dict[str, Any]]:
    """Join two configurations on seed/task/paired episode identity.

    The join is exact: missing or duplicate keys are errors rather than being
    silently dropped.  This makes paired comparisons auditable and prevents a
    partial comparison from looking like a complete one.
    """

    key_fields = ("training_seed", "task_id", "paired_episode_id")
    selected = [
        row
        for row in episode_rows
        if str(row.get("config_id")) in {reference_config, target_config}
    ]
    by_config: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {
        reference_config: {},
        target_config: {},
    }
    for row in selected:
        config_id = str(row.get("config_id"))
        key = tuple(row.get(field) for field in key_fields)
        if not all(value not in (None, "") for value in key):
            raise AnalysisError(f"paired row has incomplete key: {config_id}, {key}")
        if key in by_config[config_id]:
            raise AnalysisError(f"duplicate paired key for {config_id}: {key}")
        by_config[config_id][key] = row
    reference_keys = set(by_config[reference_config])
    target_keys = set(by_config[target_config])
    if reference_keys != target_keys:
        missing_reference = sorted(target_keys - reference_keys, key=str)
        missing_target = sorted(reference_keys - target_keys, key=str)
        raise AnalysisError(
            "paired key mismatch: "
            f"missing_reference={missing_reference[:5]}, missing_target={missing_target[:5]}"
        )
    output: list[dict[str, Any]] = []
    for key in sorted(reference_keys, key=lambda value: tuple(str(item) for item in value)):
        reference = by_config[reference_config][key]
        target = by_config[target_config][key]
        reference_success = int(float(reference["value"]))
        target_success = int(float(target["value"]))
        if reference_success not in (0, 1) or target_success not in (0, 1):
            raise AnalysisError("paired episode success must be binary")
        if reference_success and target_success:
            outcome = "both_success"
        elif target_success:
            outcome = "target_only"
        elif reference_success:
            outcome = "reference_only"
        else:
            outcome = "both_fail"
        output.append(
            {
                "reference_config": reference_config,
                "target_config": target_config,
                "training_seed": key[0],
                "task_id": key[1],
                "paired_episode_id": key[2],
                "reference_success": reference_success,
                "target_success": target_success,
                "outcome": outcome,
            }
        )
    return output
