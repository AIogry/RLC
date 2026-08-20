import pytest

from impls.analysis.plotting import apply_view, generate_figures, write_rows_csv
from impls.analysis.schema import AnalysisError


def _row(seed, value):
    return {
        "config_id": "C",
        "config_slug": "c",
        "budget": 5,
        "k_high": 4,
        "k_low": 1,
        "high_fraction": 0.8,
        "low_fraction": 0.2,
        "environment": "env",
        "training_seed": seed,
        "seed_overall_success": value,
        "overall_success_mean": 0.5,
        "overall_success_population_sd": 0.1,
        "overall_success_sample_sd": 0.122474487,
        "overall_success_n_training_seeds": 2,
    }


def test_figure_csv_is_reproducible_and_names_are_stable(tmp_path):
    rows = [_row(0, 0.4), _row(1, 0.6)]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_rows_csv(first, rows)
    write_rows_csv(second, rows)
    assert first.read_bytes() == second.read_bytes()

    figure_dir = tmp_path / "figures"
    generate_figures(
        {"allocation_summary": rows},
        figure_dir,
        figure_ids=["allocation_response_overall"],
        figure_views={
            "allocation_response_overall": {
                "full": {
                    "view_id": "full",
                    "y_range": [0.0, 1.0],
                    "panel_layout": "single",
                    "subtitle": None,
                },
                "zoom": {
                    "view_id": "zoom",
                    "y_range": [0.3, 0.7],
                    "panel_layout": "single",
                    "subtitle": "Zoomed y-axis",
                },
            }
        },
        reference_config="C",
        task_groups={},
    )
    assert (figure_dir / "allocation_response_overall__full.csv").read_bytes() == (
        figure_dir / "allocation_response_overall__zoom.csv"
    ).read_bytes()
    assert sorted(path.name for path in figure_dir.iterdir()) == [
        "allocation_response_overall__full.csv",
        "allocation_response_overall__full.pdf",
        "allocation_response_overall__full.png",
        "allocation_response_overall__zoom.csv",
        "allocation_response_overall__zoom.pdf",
        "allocation_response_overall__zoom.png",
    ]


def test_apply_view_uses_declared_range_and_rejects_clipping():
    import matplotlib.pyplot as plt

    rows = [_row(0, 0.4), _row(1, 0.6)]
    figure, axes = plt.subplots()
    apply_view(
        axes,
        view={"y_range": [0.3, 0.7], "subtitle": "Zoomed y-axis"},
        figure_id="allocation_response_overall",
        view_id="zoom",
        rows=rows,
        value_fields=("seed_overall_success",),
        mean_field="overall_success_mean",
        sd_field="overall_success_population_sd",
    )
    assert axes.get_ylim() == (0.3, 0.7)
    plt.close(figure)

    figure, axes = plt.subplots()
    with pytest.raises(AnalysisError, match="data exceeds declared range"):
        apply_view(
            axes,
            view={"y_range": [0.45, 0.55]},
            figure_id="allocation_response_overall",
            view_id="zoom",
            rows=rows,
            value_fields=("seed_overall_success",),
            mean_field="overall_success_mean",
            sd_field="overall_success_population_sd",
        )
    plt.close(figure)


def test_split_views_use_declared_groups_and_preserve_task_identity(tmp_path):
    group_rows = []
    for group_name, value in (("focal_task", 0.5), ("remaining_tasks", 0.9)):
        group_rows.append(
            {
                "config_id": "C",
                "config_slug": "c",
                "budget": 5,
                "k_high": 4,
                "k_low": 1,
                "high_fraction": 0.8,
                "low_fraction": 0.2,
                "training_seed": 0,
                "group_name": group_name,
                "group_label": group_name,
                "seed_group_success_rate": value,
                "group_success_rate_mean": value,
                "group_success_rate_population_sd": 0.0,
                "group_success_rate_sample_sd": 0.0,
                "group_success_rate_n_training_seeds": 1,
            }
        )
    focal_views = {
        "allocation_response_focal_remaining": {
            "split_zoom": {
                "view_id": "split_zoom",
                "panel_layout": "horizontal",
                "panels": {
                    "focal_task": {"y_range": [0.15, 0.9]},
                    "remaining_tasks": {"y_range": [0.84, 0.98]},
                },
            }
        }
    }
    records = generate_figures(
        {"focal_remaining_summary": group_rows},
        tmp_path / "focal",
        figure_ids=["allocation_response_focal_remaining"],
        figure_views=focal_views,
        reference_config="C",
        task_groups={
            "focal_task": {"label": "focal_task", "task_ids": ["2"]},
            "remaining_tasks": {"label": "remaining_tasks", "task_ids": ["1"]},
        },
    )
    assert records[0]["y_axis_truncated"] is True
    assert records[0]["y_axis"]["focal_task"] == {"y_min": 0.15, "y_max": 0.9}
    csv_rows = list(__import__("csv").DictReader(
        (tmp_path / "focal" / "allocation_response_focal_remaining__split_zoom.csv").open()
    ))
    assert {row["panel_id"] for row in csv_rows} == {"focal_task", "remaining_tasks"}

    delta_rows = []
    for task_id, value in (("1", 0.05), ("2", -0.4)):
        delta_rows.append(
            {
                "config_id": "C",
                "target_config": "C",
                "config_slug": "c",
                "budget": 5,
                "k_high": 4,
                "k_low": 1,
                "high_fraction": 0.8,
                "low_fraction": 0.2,
                "training_seed": 0,
                "task_id": task_id,
                "task_name": f"task{task_id}",
                "seed_delta": value,
                "delta_mean": value,
                "delta_population_sd": 0.0,
                "delta_sample_sd": 0.0,
                "delta_n_training_seeds": 1,
            }
        )
    delta_views = {
        "task_delta_vs_reference": {
            "split_zoom": {
                "view_id": "split_zoom",
                "panel_layout": "horizontal",
                "panels": {
                    "focal_task": {"y_range": [-0.7, 0.7]},
                    "remaining_tasks": {"y_range": [-0.18, 0.18]},
                },
            }
        }
    }
    delta_records = generate_figures(
        {"task_delta_vs_reference": delta_rows},
        tmp_path / "delta",
        figure_ids=["task_delta_vs_reference"],
        figure_views=delta_views,
        reference_config="C",
        task_groups={
            "focal_task": {"label": "focal_task", "task_ids": ["2"]},
            "remaining_tasks": {"label": "remaining_tasks", "task_ids": ["1"]},
        },
    )
    assert delta_records[0]["y_axis_truncated"] is True
    delta_csv = list(__import__("csv").DictReader(
        (tmp_path / "delta" / "task_delta_vs_reference__split_zoom.csv").open()
    ))
    assert {row["task_id"] for row in delta_csv} == {"1", "2"}
    assert {row["panel_id"] for row in delta_csv} == {"focal_task", "remaining_tasks"}
