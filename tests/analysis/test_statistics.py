from impls.analysis.statistics import (
    join_paired_episode_results,
    population_sd,
    sample_sd,
    seed_aggregate,
)


def test_population_and_sample_sd_are_distinct():
    assert abs(population_sd([0.0, 1.0, 1.0]) - (2.0 / 9.0) ** 0.5) < 1e-15
    assert abs(sample_sd([0.0, 1.0, 1.0]) - (1.0 / 3.0) ** 0.5) < 1e-15


def test_seed_aggregate_keeps_training_seeds_separate():
    rows = [
        {"config_id": "C", "training_seed": 0, "value": 0.0},
        {"config_id": "C", "training_seed": 1, "value": 1.0},
        {"config_id": "C", "training_seed": 2, "value": 1.0},
    ]
    seed_rows, aggregate_rows = seed_aggregate(rows, group_fields=("config_id",))
    assert [row["training_seed"] for row in seed_rows] == [0, 1, 2]
    assert aggregate_rows[0]["mean"] == 2.0 / 3.0
    assert aggregate_rows[0]["n_training_seeds"] == 3
