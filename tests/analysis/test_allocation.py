from impls.analysis.allocation import build_allocation_tables


def _config(config_id, budget, high, low):
    return {
        "config_id": config_id,
        "config_slug": config_id.lower(),
        "budget": budget,
        "k_high": high,
        "k_low": low,
        "high_fraction": high / budget,
        "low_fraction": low / budget,
    }


def _task_row(config_id, seed, task_id, value):
    return {
        "config_id": config_id,
        "config_slug": config_id.lower(),
        "environment": "env",
        "training_seed": seed,
        "task_id": str(task_id),
        "task_name": f"task{task_id}",
        "metric": "task_success_rate",
        "value": value,
    }


def test_focal_group_order_and_reference_delta_are_seed_first():
    config = {
        "ref": _config("ref", 2, 1, 1),
        "target": _config("target", 5, 4, 1),
    }
    task_rows = []
    overall_rows = []
    episode_rows = []
    values = {
        "ref": {0: {"1": 0.2, "2": 0.4, "3": 0.6}, 1: {"1": 0.4, "2": 0.6, "3": 0.8}},
        "target": {0: {"1": 0.3, "2": 0.9, "3": 0.7}, 1: {"1": 0.5, "2": 0.7, "3": 0.9}},
    }
    for config_id, seeds in values.items():
        for seed, task_values in seeds.items():
            for task_id, value in task_values.items():
                task_rows.append(_task_row(config_id, seed, task_id, value))
                for episode in range(2):
                    episode_rows.append({
                        "config_id": config_id,
                        "training_seed": seed,
                        "task_id": task_id,
                        "paired_episode_id": f"{task_id}-{episode}",
                        "value": int(value >= 0.5),
                    })
            overall_rows.append({
                "config_id": config_id,
                "config_slug": config_id.lower(),
                "environment": "env",
                "training_seed": seed,
                "metric": "overall_success",
                "value": sum(task_values.values()) / 3,
            })
    bundle = {
        "config_metadata": config,
        "config_ids": ["ref", "target"],
        "task_rows": task_rows,
        "overall_rows": overall_rows,
        "episode_rows": episode_rows,
    }
    spec = {
        "reference": {"config_id": "ref", "label": "reference"},
        "task_groups": {
            "focal": {"label": "focal", "task_ids": ["2"]},
            "remaining": {"label": "remaining", "task_ids": ["1", "3"]},
        },
    }
    tables = build_allocation_tables(bundle, spec)
    focal = [
        row for row in tables["focal_remaining_summary"]
        if row["config_id"] == "target" and row["group_name"] == "focal"
    ]
    assert [row["training_seed"] for row in focal] == [0, 1]
    assert focal[0]["seed_group_success_rate"] == 0.9
    assert focal[0]["group_success_rate_mean"] == 0.8
    deltas = [
        row for row in tables["task_delta_vs_reference"]
        if row["target_config"] == "target" and row["task_id"] == "2"
    ]
    assert abs(deltas[0]["seed_delta"] - 0.5) < 1e-12
    assert abs(deltas[1]["seed_delta"] - 0.1) < 1e-12
    assert abs(deltas[0]["delta_mean"] - 0.3) < 1e-12
