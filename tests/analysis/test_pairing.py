import pytest

from impls.analysis.schema import AnalysisError
from impls.analysis.statistics import join_paired_episode_results


def _row(config_id, seed=0, task_id="1", pair="p0", value=0):
    return {
        "config_id": config_id,
        "training_seed": seed,
        "task_id": task_id,
        "paired_episode_id": pair,
        "value": value,
    }


def test_paired_join_classifies_outcomes():
    rows = [
        _row("ref", pair="p0", value=1),
        _row("target", pair="p0", value=0),
        _row("ref", pair="p1", value=0),
        _row("target", pair="p1", value=1),
    ]
    joined = join_paired_episode_results(rows, reference_config="ref", target_config="target")
    assert [row["outcome"] for row in joined] == ["reference_only", "target_only"]


def test_paired_join_rejects_missing_pair():
    rows = [_row("ref"), _row("target", pair="different")]
    with pytest.raises(AnalysisError, match="paired key mismatch"):
        join_paired_episode_results(rows, reference_config="ref", target_config="target")


def test_paired_join_rejects_duplicate_pair():
    rows = [_row("ref"), _row("ref"), _row("target")]
    with pytest.raises(AnalysisError, match="duplicate paired key"):
        join_paired_episode_results(rows, reference_config="ref", target_config="target")
