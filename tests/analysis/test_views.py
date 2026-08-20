import copy

import pytest

from impls.analysis.schema import AnalysisError, analysis_spec_fingerprint, load_analysis_spec


def _write_spec(path, *, figures):
    path.write_text(
        """
analysis_id: A
source:
  study_id: S
  reevaluation_id: R
  path: /tmp/source
reference:
  config_id: C
  label: reference
task_groups:
  focal_task:
    label: focal_task
    task_ids: [2]
figures:
"""
        + figures,
        encoding="utf-8",
    )


def test_valid_full_zoom_views_and_fingerprint(tmp_path):
    path = tmp_path / "spec.yaml"
    _write_spec(
        path,
        figures="""  allocation_response_overall:
    views:
      full: {y_range: [0.0, 1.0]}
      zoom: {y_range: [0.75, 0.95]}
""",
    )
    spec = load_analysis_spec(path)
    assert spec["figure_views"]["allocation_response_overall"]["zoom"]["y_range"] == [0.75, 0.95]
    changed = copy.deepcopy(spec)
    changed["figure_views"]["allocation_response_overall"]["zoom"]["y_range"] = [0.7, 0.95]
    assert analysis_spec_fingerprint(spec) != analysis_spec_fingerprint(changed)


def test_invalid_range_and_unknown_view_field_fail_loudly(tmp_path):
    invalid_range = tmp_path / "invalid_range.yaml"
    _write_spec(
        invalid_range,
        figures="""  allocation_response_overall:
    views:
      full: {y_range: [1.0, 0.0]}
""",
    )
    with pytest.raises(AnalysisError, match="y_min < y_max"):
        load_analysis_spec(invalid_range)

    unknown_field = tmp_path / "unknown_field.yaml"
    _write_spec(
        unknown_field,
        figures="""  allocation_response_overall:
    views:
      full: {y_range: [0.0, 1.0], autoscale: true}
""",
    )
    with pytest.raises(AnalysisError, match="unknown view fields"):
        load_analysis_spec(unknown_field)


def test_legacy_figure_list_defaults_to_full_view(tmp_path):
    path = tmp_path / "legacy.yaml"
    _write_spec(path, figures="  figures_placeholder: {}\n")
    # Replace the intentionally simple mapping with the legacy list form.
    path.write_text(
        path.read_text(encoding="utf-8").replace("figures:\n  figures_placeholder: {}", "figures:\n  - allocation_response_overall"),
        encoding="utf-8",
    )
    spec = load_analysis_spec(path)
    assert list(spec["figure_views"]["allocation_response_overall"]) == ["full"]
    assert spec["figure_views"]["allocation_response_overall"]["full"]["y_range"] == [0.0, 1.0]
