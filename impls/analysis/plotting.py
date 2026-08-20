"""Centralised, declarative plotting for reevaluation analyses.

The plotting functions describe what is drawn.  ``apply_view`` and
``validate_rows_within_view`` own the visual axis policy, so a zoomed view can
never silently filter data or derive its range from the data being plotted.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .schema import AnalysisError


PLOT_STYLE = {
    "figure_size": (7.2, 4.8),
    "dpi": 180,
    "font_size": 10,
    "grid_alpha": 0.25,
    "budget_colors": {2: "#6C757D", 5: "#1F77B4", 16: "#D95F02"},
    "task_colors": {
        "1": "#4C78A8",
        "2": "#F58518",
        "3": "#54A24B",
        "4": "#E45756",
        "5": "#B279A2",
    },
    "group_colors": {"focal_task": "#E45756", "remaining_tasks": "#4C78A8"},
}


def write_rows_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        raise AnalysisError(f"cannot write an empty schema CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _setup(title: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update(
        {
            "font.size": PLOT_STYLE["font_size"],
            "axes.titlesize": PLOT_STYLE["font_size"] + 1,
            "axes.labelsize": PLOT_STYLE["font_size"],
            "legend.fontsize": PLOT_STYLE["font_size"] - 1,
        }
    )
    figure, axes = plt.subplots(figsize=PLOT_STYLE["figure_size"])
    axes.set_title(title)
    axes.set_xlabel("high_fraction")
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=PLOT_STYLE["grid_alpha"])
    return figure, axes


def _save(figure: plt.Figure, figure_dir: Path, figure_id: str, view_id: str) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{figure_id}__{view_id}"
    pdf = figure_dir / f"{stem}.pdf"
    png = figure_dir / f"{stem}.png"
    csv_path = figure_dir / f"{stem}.csv"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(figure)
    return {"pdf": str(pdf), "png": str(png), "csv": str(csv_path)}


def _float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise AnalysisError(f"plot row is missing {field}: {row}")
    return float(value)


def _budget_color(budget: Any) -> str:
    return PLOT_STYLE["budget_colors"].get(int(float(budget)), "#333333")


def _allocated(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("budget") not in (None, "")]


def _value_range(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fields: Sequence[str],
    mean_field: str | None = None,
    sd_field: str | None = None,
) -> tuple[float, float]:
    values: list[float] = []
    for row in rows:
        for field in value_fields:
            values.append(_float(row, field))
        if mean_field is not None:
            mean = _float(row, mean_field)
            values.append(mean)
            if sd_field is not None:
                sd = _float(row, sd_field)
                values.extend((mean - sd, mean + sd))
    if not values:
        raise AnalysisError("cannot validate an empty view")
    return min(values), max(values)


def validate_rows_within_view(
    rows: Sequence[Mapping[str, Any]],
    *,
    y_range: Sequence[float],
    figure_id: str,
    view_id: str,
    panel_id: str | None,
    value_fields: Sequence[str],
    mean_field: str | None = None,
    sd_field: str | None = None,
) -> dict[str, Any]:
    """Validate plotted values against a declared range; never auto-expand it."""

    lower, upper = float(y_range[0]), float(y_range[1])
    natural_min, natural_max = _value_range(
        rows,
        value_fields=value_fields,
        mean_field=mean_field,
        sd_field=sd_field,
    )
    for index, row in enumerate(rows):
        values: list[tuple[str, float]] = [
            (field, _float(row, field)) for field in value_fields
        ]
        if mean_field is not None:
            mean = _float(row, mean_field)
            values.append((mean_field, mean))
            if sd_field is not None:
                sd = _float(row, sd_field)
                values.extend(
                    [(f"{mean_field}-sd", mean - sd), (f"{mean_field}+sd", mean + sd)]
                )
        for field, value in values:
            if value < lower or value > upper:
                location = {
                    "row_index": index,
                    "config_id": row.get("config_id", row.get("target_config")),
                    "training_seed": row.get("training_seed"),
                    "task_id": row.get("task_id"),
                    "field": field,
                    "value": value,
                    "declared_range": [lower, upper],
                }
                if panel_id is not None:
                    location["panel_id"] = panel_id
                raise AnalysisError(
                    f"data exceeds declared range for {figure_id}__{view_id}: {location}"
                )
    return {
        "y_axis": {"y_min": lower, "y_max": upper},
        "natural_data_range": {"min": natural_min, "max": natural_max},
        "y_axis_truncated": False,
    }


def apply_view(
    axes: plt.Axes,
    *,
    view: Mapping[str, Any],
    figure_id: str,
    view_id: str,
    rows: Sequence[Mapping[str, Any]],
    value_fields: Sequence[str],
    mean_field: str | None = None,
    sd_field: str | None = None,
    panel_id: str | None = None,
) -> dict[str, Any]:
    """Validate and apply one declarative axis view to an existing axis."""

    if "y_range" not in view:
        raise AnalysisError(f"single-panel view lacks y_range: {figure_id}__{view_id}")
    record = validate_rows_within_view(
        rows,
        y_range=view["y_range"],
        figure_id=figure_id,
        view_id=view_id,
        panel_id=panel_id,
        value_fields=value_fields,
        mean_field=mean_field,
        sd_field=sd_field,
    )
    axes.set_ylim(*view["y_range"])
    if view_id != "full":
        axes.text(
            0.02,
            0.98,
            "Zoomed y-axis",
            transform=axes.transAxes,
            va="top",
            ha="left",
            fontsize=PLOT_STYLE["font_size"] - 1,
            color="#555555",
        )
    if view.get("subtitle") and view_id != "full":
        axes.set_title(f"{axes.get_title()} — {view['subtitle']}")
    return record


def _raw_and_mean(
    axes: plt.Axes,
    rows: list[Mapping[str, Any]],
    *,
    x_field: str,
    seed_field: str,
    mean_field: str,
    sd_field: str,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    by_x: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_x[_float(row, x_field)].append(row)
    for x, point_rows in sorted(by_x.items()):
        for row in point_rows:
            axes.scatter(
                x,
                _float(row, seed_field),
                color=color,
                alpha=0.55,
                marker=marker,
                s=28,
                linewidths=0,
            )
        first = point_rows[0]
        axes.errorbar(
            x,
            _float(first, mean_field),
            yerr=_float(first, sd_field),
            color=color,
            marker="D",
            markersize=5,
            capsize=3,
            linestyle="none",
            label=label if x == min(by_x) else None,
        )


def _view_record(
    *,
    figure_id: str,
    view_id: str,
    panel_layout: str,
    source_table: str,
    output_files: dict[str, str],
    axis_records: Mapping[str, Mapping[str, Any]],
    truncated_override: bool | None = None,
) -> dict[str, Any]:
    if len(axis_records) == 1 and "default" in axis_records:
        axis = dict(axis_records["default"])
        y_axis = axis.pop("y_axis")
        truncated = axis.pop("y_axis_truncated")
        if truncated_override is not None:
            truncated = truncated_override
        return {
            "figure_id": figure_id,
            "view_id": view_id,
            "y_axis": y_axis,
            "panel_layout": panel_layout,
            "y_axis_truncated": truncated,
            "natural_data_range": axis.pop("natural_data_range"),
            "source_table": source_table,
            "output_files": output_files,
        }
    return {
        "figure_id": figure_id,
        "view_id": view_id,
        "y_axis": {panel: record["y_axis"] for panel, record in axis_records.items()},
        "panel_layout": panel_layout,
        "y_axis_truncated": (
            truncated_override
            if truncated_override is not None
            else any(record["y_axis_truncated"] for record in axis_records.values())
        ),
        "natural_data_range": {
            panel: record["natural_data_range"] for panel, record in axis_records.items()
        },
        "panels": {
            panel: {
                "y_axis": record["y_axis"],
                "y_axis_truncated": (
                    truncated_override
                    if truncated_override is not None
                    else record["y_axis_truncated"]
                ),
                "natural_data_range": record["natural_data_range"],
            }
            for panel, record in axis_records.items()
        },
        "source_table": source_table,
        "output_files": output_files,
    }


def _success_truncation(record: dict[str, Any], *, view_id: str) -> dict[str, Any]:
    """Annotate success views against their fixed natural metric domain [0, 1]."""

    y_range = record["y_axis"]
    record["y_axis_truncated"] = view_id != "full" or y_range != {"y_min": 0.0, "y_max": 1.0}
    return record


def plot_allocation_response_overall(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
    reference_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    figure, axes = _setup("Allocation response: overall success", "overall success")
    axis_record = apply_view(
        axes,
        view=view,
        figure_id="allocation_response_overall",
        view_id=view_id,
        rows=rows,
        value_fields=("seed_overall_success",),
        mean_field="overall_success_mean",
        sd_field="overall_success_population_sd",
    )
    for budget in sorted({int(float(row["budget"])) for row in rows}):
        _raw_and_mean(
            axes,
            [row for row in rows if int(float(row["budget"])) == budget],
            x_field="high_fraction",
            seed_field="seed_overall_success",
            mean_field="overall_success_mean",
            sd_field="overall_success_population_sd",
            label=f"B={budget}",
            color=_budget_color(budget),
        )
    if reference_row is not None:
        axes.axhline(
            _float(reference_row, "overall_success_mean"),
            color="#555555",
            linestyle="--",
            linewidth=1,
            label="reference mean",
        )
    axes.legend(loc="best", frameon=False)
    output_files = _save(figure, figure_dir, "allocation_response_overall", view_id)
    return _view_record(
        figure_id="allocation_response_overall",
        view_id=view_id,
        panel_layout="single",
        source_table="allocation_summary",
        output_files=output_files,
        axis_records={"default": _success_truncation(axis_record, view_id=view_id)},
    )


def plot_allocation_response_by_task(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
) -> dict[str, Any]:
    task_ids = sorted({str(row["task_id"]) for row in rows})
    ncols = 2
    nrows = max(1, (len(task_ids) + ncols - 1) // ncols)
    figure, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.6, 3.2 * nrows),
        squeeze=False,
    )
    axes_list = [axis for row in axes_grid for axis in row]
    axis_records: dict[str, dict[str, Any]] = {}
    for axis, task_id in zip(axes_list, task_ids):
        axis.set_title(f"task {task_id}")
        axis.set_xlabel("high_fraction")
        axis.set_ylabel("task success")
        axis.grid(True, alpha=PLOT_STYLE["grid_alpha"])
        task_rows = [row for row in rows if str(row["task_id"]) == task_id]
        axis_records[task_id] = apply_view(
            axis,
            view=view,
            figure_id="allocation_response_by_task",
            view_id=view_id,
            rows=task_rows,
            value_fields=("seed_task_success_rate",),
            mean_field="task_success_rate_mean",
            sd_field="task_success_rate_population_sd",
            panel_id=task_id,
        )
        for budget in sorted({int(float(row["budget"])) for row in task_rows}):
            _raw_and_mean(
                axis,
                [row for row in task_rows if int(float(row["budget"])) == budget],
                x_field="high_fraction",
                seed_field="seed_task_success_rate",
                mean_field="task_success_rate_mean",
                sd_field="task_success_rate_population_sd",
                label=f"B={budget}",
                color=_budget_color(budget),
            )
        axis.legend(loc="best", frameon=False)
    for axis in axes_list[len(task_ids) :]:
        axis.set_visible(False)
    figure.suptitle("Allocation response by task")
    figure.tight_layout()
    output_files = _save(figure, figure_dir, "allocation_response_by_task", view_id)
    return _view_record(
        figure_id="allocation_response_by_task",
        view_id=view_id,
        panel_layout="task_grid",
        source_table="task_allocation_summary",
        output_files=output_files,
        axis_records=axis_records,
    )


def plot_focal_remaining(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
) -> dict[str, Any]:
    figure, axes = _setup("Task-group response", "group success")
    axis_record = apply_view(
        axes,
        view=view,
        figure_id="allocation_response_focal_remaining",
        view_id=view_id,
        rows=rows,
        value_fields=("seed_group_success_rate",),
        mean_field="group_success_rate_mean",
        sd_field="group_success_rate_population_sd",
    )
    for group_name in sorted({str(row["group_name"]) for row in rows}):
        group_rows = [row for row in rows if str(row["group_name"]) == group_name]
        for budget in sorted({int(float(row["budget"])) for row in group_rows}):
            _raw_and_mean(
                axes,
                [row for row in group_rows if int(float(row["budget"])) == budget],
                x_field="high_fraction",
                seed_field="seed_group_success_rate",
                mean_field="group_success_rate_mean",
                sd_field="group_success_rate_population_sd",
                label=f"{row_label(group_rows)}, B={budget}",
                color=PLOT_STYLE["group_colors"].get(group_name, "#333333"),
                marker="s" if group_name != "focal_task" else "o",
            )
    axes.legend(loc="best", frameon=False)
    output_files = _save(figure, figure_dir, "allocation_response_focal_remaining", view_id)
    return _view_record(
        figure_id="allocation_response_focal_remaining",
        view_id=view_id,
        panel_layout="single",
        source_table="focal_remaining_summary",
        output_files=output_files,
        axis_records={"default": _success_truncation(axis_record, view_id=view_id)},
    )


def row_label(rows: Sequence[Mapping[str, Any]]) -> str:
    labels = {str(row.get("group_label", row.get("group_name", "group"))) for row in rows}
    if len(labels) != 1:
        raise AnalysisError(f"group rows have inconsistent labels: {labels}")
    return next(iter(labels))


def plot_focal_remaining_split(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
) -> dict[str, Any]:
    group_names = list(view["panels"])
    figure, axes_grid = plt.subplots(1, len(group_names), figsize=(10.5, 4.8), squeeze=False)
    axis_records: dict[str, dict[str, Any]] = {}
    for axis, group_name in zip(axes_grid[0], group_names):
        group_rows = [row for row in rows if str(row["group_name"]) == group_name]
        if not group_rows:
            raise AnalysisError(f"no rows for declared task group panel: {group_name}")
        axis.set_title(row_label(group_rows))
        axis.set_xlabel("high_fraction")
        axis.set_ylabel("group success")
        axis.grid(True, alpha=PLOT_STYLE["grid_alpha"])
        panel = view["panels"][group_name]
        axis_records[group_name] = apply_view(
            axis,
            view=panel,
            figure_id="allocation_response_focal_remaining",
            view_id=view_id,
            rows=group_rows,
            value_fields=("seed_group_success_rate",),
            mean_field="group_success_rate_mean",
            sd_field="group_success_rate_population_sd",
            panel_id=group_name,
        )
        for budget in sorted({int(float(row["budget"])) for row in group_rows}):
            _raw_and_mean(
                axis,
                [row for row in group_rows if int(float(row["budget"])) == budget],
                x_field="high_fraction",
                seed_field="seed_group_success_rate",
                mean_field="group_success_rate_mean",
                sd_field="group_success_rate_population_sd",
                label=f"B={budget}",
                color=_budget_color(budget),
                marker="s" if group_name != "focal_task" else "o",
            )
        axis.legend(loc="best", frameon=False)
    figure.suptitle("Task-group response — split zoom")
    figure.tight_layout()
    output_files = _save(
        figure,
        figure_dir,
        "allocation_response_focal_remaining",
        view_id,
    )
    return _view_record(
        figure_id="allocation_response_focal_remaining",
        view_id=view_id,
        panel_layout="horizontal",
        source_table="focal_remaining_summary",
        output_files=output_files,
        axis_records=axis_records,
        truncated_override=True,
    )


def plot_seed_consistency(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
) -> dict[str, Any]:
    figure, axes = _setup("Training-seed consistency", "overall success")
    axis_record = apply_view(
        axes,
        view=view,
        figure_id="allocation_seed_consistency",
        view_id=view_id,
        rows=rows,
        value_fields=("seed_overall_success",),
        mean_field="overall_success_mean",
        sd_field="overall_success_population_sd",
    )
    seed_markers = {"0": "o", "1": "s", "2": "^"}
    for budget in sorted({int(float(row["budget"])) for row in rows}):
        budget_rows = [row for row in rows if int(float(row["budget"])) == budget]
        for row in budget_rows:
            axes.scatter(
                _float(row, "high_fraction"),
                _float(row, "seed_overall_success"),
                color=_budget_color(budget),
                marker=seed_markers.get(str(row["training_seed"]), "o"),
                alpha=0.65,
                s=34,
            )
        for x in sorted({_float(row, "high_fraction") for row in budget_rows}):
            point = next(row for row in budget_rows if _float(row, "high_fraction") == x)
            axes.errorbar(
                x,
                _float(point, "overall_success_mean"),
                yerr=_float(point, "overall_success_population_sd"),
                color=_budget_color(budget),
                marker="D",
                markersize=5,
                capsize=3,
                linestyle="none",
                label=f"B={budget}" if x == min(_float(item, "high_fraction") for item in budget_rows) else None,
            )
    axes.legend(loc="best", frameon=False)
    output_files = _save(figure, figure_dir, "allocation_seed_consistency", view_id)
    return _view_record(
        figure_id="allocation_seed_consistency",
        view_id=view_id,
        panel_layout="single",
        source_table="allocation_summary",
        output_files=output_files,
        axis_records={"default": _success_truncation(axis_record, view_id=view_id)},
    )


def _plot_delta_points(
    axes: plt.Axes,
    rows: list[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
) -> None:
    for task_id in task_ids:
        task_rows = [row for row in rows if str(row["task_id"]) == str(task_id)]
        color = PLOT_STYLE["task_colors"].get(str(task_id), "#333333")
        for row in task_rows:
            axes.scatter(
                _float(row, "high_fraction"),
                _float(row, "seed_delta"),
                color=color,
                alpha=0.35,
                s=24,
            )
        for budget in sorted({int(float(row["budget"])) for row in task_rows}):
            budget_rows = [row for row in task_rows if int(float(row["budget"])) == budget]
            by_x: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for row in budget_rows:
                by_x[_float(row, "high_fraction")].append(row)
            for x, points in sorted(by_x.items()):
                first = points[0]
                axes.errorbar(
                    x,
                    _float(first, "delta_mean"),
                    yerr=_float(first, "delta_population_sd"),
                    color=color,
                    marker="o",
                    markersize=4,
                    capsize=3,
                    linestyle="none",
                    label=f"task {task_id}" if x == min(by_x) and budget == min(
                        int(float(item["budget"])) for item in task_rows
                    ) else None,
                )


def plot_task_delta(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
    task_ids: Sequence[str],
) -> dict[str, Any]:
    figure, axes = _setup("Task delta versus reference", "target - reference")
    axis_record = apply_view(
        axes,
        view=view,
        figure_id="task_delta_vs_reference",
        view_id=view_id,
        rows=rows,
        value_fields=("seed_delta",),
        mean_field="delta_mean",
        sd_field="delta_population_sd",
    )
    _plot_delta_points(axes, rows, task_ids=task_ids)
    axes.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    axes.legend(loc="best", frameon=False)
    output_files = _save(figure, figure_dir, "task_delta_vs_reference", view_id)
    return _view_record(
        figure_id="task_delta_vs_reference",
        view_id=view_id,
        panel_layout="single",
        source_table="task_delta_vs_reference",
        output_files=output_files,
        axis_records={"default": axis_record},
    )


def plot_task_delta_split(
    rows: list[dict[str, Any]],
    figure_dir: Path,
    *,
    view_id: str,
    view: Mapping[str, Any],
    task_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    group_names = list(view["panels"])
    figure, axes_grid = plt.subplots(1, len(group_names), figsize=(10.5, 4.8), squeeze=False)
    axis_records: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for axis, group_name in zip(axes_grid[0], group_names):
        task_ids = [str(task_id) for task_id in task_groups[group_name]["task_ids"]]
        group_rows = [row for row in rows if str(row["task_id"]) in task_ids]
        if not group_rows:
            raise AnalysisError(f"no rows for declared task group panel: {group_name}")
        panel_rows = [dict(row, panel_id=group_name) for row in group_rows]
        csv_rows.extend(panel_rows)
        axis.set_title(str(task_groups[group_name]["label"]))
        axis.set_xlabel("high_fraction")
        axis.set_ylabel("target - reference")
        axis.grid(True, alpha=PLOT_STYLE["grid_alpha"])
        panel = view["panels"][group_name]
        axis_records[group_name] = apply_view(
            axis,
            view=panel,
            figure_id="task_delta_vs_reference",
            view_id=view_id,
            rows=group_rows,
            value_fields=("seed_delta",),
            mean_field="delta_mean",
            sd_field="delta_population_sd",
            panel_id=group_name,
        )
        _plot_delta_points(axis, group_rows, task_ids=task_ids)
        axis.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
        axis.legend(loc="best", frameon=False)
    figure.suptitle("Task delta versus reference — split zoom")
    figure.tight_layout()
    output_files = _save(figure, figure_dir, "task_delta_vs_reference", view_id)
    return _view_record(
        figure_id="task_delta_vs_reference",
        view_id=view_id,
        panel_layout="horizontal",
        source_table="task_delta_vs_reference",
        output_files=output_files,
        axis_records=axis_records,
        truncated_override=True,
    ), csv_rows


def _with_panel(rows: Sequence[Mapping[str, Any]], panel_id: str) -> list[dict[str, Any]]:
    return [dict(row, panel_id=panel_id) for row in rows]


def generate_figures(
    tables: Mapping[str, list[dict[str, Any]]],
    figure_dir: str | Path,
    *,
    figure_ids: Iterable[str],
    figure_views: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reference_config: str,
    task_groups: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Render every declared view and return its provenance records."""

    figure_dir = Path(figure_dir)
    rendered: list[dict[str, Any]] = []
    for figure_id in figure_ids:
        if figure_id not in figure_views:
            raise AnalysisError(f"missing view spec for figure {figure_id}")
        for view_id, view in figure_views[figure_id].items():
            csv_rows: list[dict[str, Any]]
            if figure_id == "allocation_response_overall":
                csv_rows = _allocated(tables["allocation_summary"])
                ref_rows = [row for row in csv_rows if str(row["config_id"]) == reference_config]
                record = plot_allocation_response_overall(
                    csv_rows,
                    figure_dir,
                    view_id=view_id,
                    view=view,
                    reference_row=ref_rows[0] if ref_rows else None,
                )
            elif figure_id == "allocation_response_by_task":
                csv_rows = _allocated(tables["task_allocation_summary"])
                record = plot_allocation_response_by_task(
                    csv_rows,
                    figure_dir,
                    view_id=view_id,
                    view=view,
                )
            elif figure_id == "allocation_response_focal_remaining":
                csv_rows = _allocated(tables["focal_remaining_summary"])
                if view_id == "split_zoom":
                    record = plot_focal_remaining_split(
                        csv_rows,
                        figure_dir,
                        view_id=view_id,
                        view=view,
                    )
                    csv_rows = [
                        dict(row, panel_id=str(row["group_name"])) for row in csv_rows
                    ]
                else:
                    record = plot_focal_remaining(
                        csv_rows,
                        figure_dir,
                        view_id=view_id,
                        view=view,
                    )
            elif figure_id == "allocation_seed_consistency":
                csv_rows = _allocated(tables["allocation_summary"])
                record = plot_seed_consistency(
                    csv_rows,
                    figure_dir,
                    view_id=view_id,
                    view=view,
                )
            elif figure_id == "task_delta_vs_reference":
                csv_rows = _allocated(tables["task_delta_vs_reference"])
                if view_id == "split_zoom":
                    record, csv_rows = plot_task_delta_split(
                        csv_rows,
                        figure_dir,
                        view_id=view_id,
                        view=view,
                        task_groups=task_groups,
                    )
                else:
                    all_task_ids = sorted({str(row["task_id"]) for row in csv_rows})
                    record = plot_task_delta(
                        csv_rows,
                        figure_dir,
                        view_id=view_id,
                        view=view,
                        task_ids=all_task_ids,
                    )
            else:
                raise AnalysisError(f"no plotting implementation for {figure_id}")
            write_rows_csv(record["output_files"]["csv"], csv_rows)
            rendered.append(record)
    return rendered
