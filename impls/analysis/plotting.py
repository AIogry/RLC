"""Centralised, deterministic plotting for reevaluation analyses."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _save(figure: plt.Figure, figure_dir: Path, figure_id: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{figure_id}.pdf", bbox_inches="tight")
    figure.savefig(
        figure_dir / f"{figure_id}.png",
        dpi=PLOT_STYLE["dpi"],
        bbox_inches="tight",
    )
    plt.close(figure)


def _float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise AnalysisError(f"plot row is missing {field}: {row}")
    return float(value)


def _budget_color(budget: Any) -> str:
    return PLOT_STYLE["budget_colors"].get(int(float(budget)), "#333333")


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
    if not rows:
        return
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


def _allocated(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("budget") not in (None, "")]


def plot_allocation_response_overall(
    rows: list[dict[str, Any]], figure_dir: Path, *, reference_row: Mapping[str, Any] | None
) -> None:
    rows = _allocated(rows)
    figure, axes = _setup("Allocation response: overall success", "overall success")
    for budget in sorted({int(float(row["budget"])) for row in rows}):
        budget_rows = [row for row in rows if int(float(row["budget"])) == budget]
        _raw_and_mean(
            axes,
            budget_rows,
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
    axes.set_ylim(-0.02, 1.02)
    axes.legend(loc="best", frameon=False)
    _save(figure, figure_dir, "allocation_response_overall")


def plot_allocation_response_by_task(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = _allocated(rows)
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
    for axis, task_id in zip(axes_list, task_ids):
        axis.set_title(f"task {task_id}")
        axis.set_xlabel("high_fraction")
        axis.set_ylabel("task success")
        axis.grid(True, alpha=PLOT_STYLE["grid_alpha"])
        task_rows = [row for row in rows if str(row["task_id"]) == task_id]
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
        axis.set_ylim(-0.02, 1.02)
        axis.legend(loc="best", frameon=False)
    for axis in axes_list[len(task_ids) :]:
        axis.set_visible(False)
    figure.suptitle("Allocation response by task")
    figure.tight_layout()
    _save(figure, figure_dir, "allocation_response_by_task")


def plot_focal_remaining(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = _allocated(rows)
    figure, axes = _setup("Task-group response", "group success")
    for group_name in sorted({str(row["group_name"]) for row in rows}):
        group_rows = [row for row in rows if str(row["group_name"]) == group_name]
        for budget in sorted({int(float(row["budget"])) for row in group_rows}):
            budget_rows = [row for row in group_rows if int(float(row["budget"])) == budget]
            _raw_and_mean(
                axes,
                budget_rows,
                x_field="high_fraction",
                seed_field="seed_group_success_rate",
                mean_field="group_success_rate_mean",
                sd_field="group_success_rate_population_sd",
                label=f"{group_name}, B={budget}",
                color=PLOT_STYLE["group_colors"].get(group_name, "#333333"),
                marker="s" if group_name != "focal_task" else "o",
            )
    axes.set_ylim(-0.02, 1.02)
    axes.legend(loc="best", frameon=False)
    _save(figure, figure_dir, "allocation_response_focal_remaining")


def plot_seed_consistency(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = _allocated(rows)
    figure, axes = _setup("Training-seed consistency", "overall success")
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
    axes.set_ylim(-0.02, 1.02)
    axes.legend(loc="best", frameon=False)
    _save(figure, figure_dir, "allocation_seed_consistency")


def plot_task_delta(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = _allocated(rows)
    figure, axes = _setup("Task delta versus reference", "target - reference")
    task_ids = sorted({str(row["task_id"]) for row in rows})
    for task_id in task_ids:
        task_rows = [row for row in rows if str(row["task_id"]) == task_id]
        color = PLOT_STYLE["task_colors"].get(task_id, "#333333")
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
    axes.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    axes.legend(loc="best", frameon=False)
    _save(figure, figure_dir, "task_delta_vs_reference")


def generate_figures(
    tables: Mapping[str, list[dict[str, Any]]],
    figure_dir: str | Path,
    *,
    figure_ids: Iterable[str],
    reference_config: str,
) -> None:
    """Write every registered figure as PDF, PNG, and its exact CSV input."""

    figure_dir = Path(figure_dir)
    registry = set(figure_ids)
    if "allocation_response_overall" in registry:
        rows = _allocated(tables["allocation_summary"])
        ref_rows = [row for row in rows if str(row["config_id"]) == reference_config]
        write_rows_csv(figure_dir / "allocation_response_overall.csv", rows)
        plot_allocation_response_overall(
            rows,
            figure_dir,
            reference_row=ref_rows[0] if ref_rows else None,
        )
    if "allocation_response_by_task" in registry:
        rows = _allocated(tables["task_allocation_summary"])
        write_rows_csv(figure_dir / "allocation_response_by_task.csv", rows)
        plot_allocation_response_by_task(rows, figure_dir)
    if "allocation_response_focal_remaining" in registry:
        rows = _allocated(tables["focal_remaining_summary"])
        write_rows_csv(figure_dir / "allocation_response_focal_remaining.csv", rows)
        plot_focal_remaining(rows, figure_dir)
    if "allocation_seed_consistency" in registry:
        rows = _allocated(tables["allocation_summary"])
        write_rows_csv(figure_dir / "allocation_seed_consistency.csv", rows)
        plot_seed_consistency(rows, figure_dir)
    if "task_delta_vs_reference" in registry:
        rows = _allocated(tables["task_delta_vs_reference"])
        write_rows_csv(figure_dir / "task_delta_vs_reference.csv", rows)
        plot_task_delta(rows, figure_dir)
