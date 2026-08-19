from impls.analysis.plotting import generate_figures, write_rows_csv


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
        reference_config="C",
    )
    assert sorted(path.name for path in figure_dir.iterdir()) == [
        "allocation_response_overall.csv",
        "allocation_response_overall.pdf",
        "allocation_response_overall.png",
    ]
