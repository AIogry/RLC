"""Targeted design, preflight, and analysis tests for M22."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from tools import analyze_m22, m22_doctor


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M22_puzzle_baselines_unified_rlc/study.yaml'
BASELINE_AUDIT = ROOT / 'docs/9-9/M22_upstream_baseline_audit.json'


def _audit_data():
    return json.loads(BASELINE_AUDIT.read_text())


def _write_eval(path: Path, values: dict[int, float], *, task_value: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['step', 'evaluation/overall_success', 'evaluation/task_a_success']
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for step, value in values.items():
            writer.writerow({
                'step': step,
                'evaluation/overall_success': value,
                'evaluation/task_a_success': value if task_value is None else task_value,
            })


def _write_run(run: dict, *, root: Path, commit: str = 'abc123', values: dict[int, float] | None = None):
    run_dir = Path(run['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'runtime_metadata.json').write_text(json.dumps({
        'study_id': run['study_id'],
        'config_id': run['config_id'],
        'algorithm': run['algorithm'],
        'environment': run['environment'],
        'seed': run['seed'],
        'run_attempt': 0,
        'git_commit': commit,
        'status': 'completed',
    }))
    _write_eval(
        run_dir / 'eval.csv',
        values or {step: 0.5 for step in analyze_m22.EXPECTED_EVAL_STEPS},
    )


def test_exact_m22_config_ids_and_factorial():
    study = m22_doctor.load_study(STUDY)
    configs = [
        m22_doctor.prepare_run_design(STUDY, path)[1]
        for path in sorted((STUDY.parent / 'configs').glob('*.yaml'))
    ]
    assert len(configs) == 24
    assert {config.config_id for config in configs} == m22_doctor.CONFIG_IDS
    assert {config.data['algorithm'] for config in configs} == set(m22_doctor.ALGORITHMS)
    assert {config.data['environment'] for config in configs} == set(m22_doctor.ENVIRONMENTS)
    assert all('seed' not in config.data for config in configs)
    assert study.data['seeds'] == [0, 1, 2]


def test_expansion_has_72_unique_run_identities(tmp_path):
    runs = analyze_m22.expected_runs(STUDY, tmp_path / 'runs')
    assert len(runs) == 72
    identities = {
        (run['study_id'], run['config_id'], run['algorithm'], run['environment'], run['seed'])
        for run in runs
    }
    assert len(identities) == 72
    assert len({str(run['run_dir']) for run in runs}) == 72
    assert {run['seed'] for run in runs} == {0, 1, 2}


def test_official_overrides_and_actual_resolved_compute_disabled():
    baseline = _audit_data()
    errors = []
    study, configs = m22_doctor._check_study_and_configs(STUDY, baseline, errors)
    assert study is not None
    assert not errors
    m22_doctor._check_resolved_configs(configs, baseline, errors)
    assert not errors
    for config in configs:
        algorithm = config.data['algorithm']
        overrides = config.data['agent_overrides']
        for key, value in m22_doctor.OFFICIAL_OVERRIDES[algorithm].items():
            assert overrides[key] == value


def test_doctor_rejects_failed_semantic_gate(tmp_path):
    source = json.loads((ROOT / 'docs/9-9/canonical_semantics_audit.json').read_text())
    source['status'] = 'fail'
    semantic_path = tmp_path / 'semantic.json'
    semantic_path.write_text(json.dumps(source))
    errors = []
    m22_doctor._check_audits(
        semantic_path,
        ROOT / 'docs/9-9/M22_upstream_baseline_audit.json',
        ROOT / 'docs/9-9/M22_dataset_audit.json',
        errors,
    )
    assert any('semantic audit status is not pass' in error for error in errors)


def test_doctor_rejects_wrong_official_override(tmp_path):
    path = ROOT / 'experiments/M22_puzzle_baselines_unified_rlc/configs/M22-QRL-P3X3.yaml'
    data = m22_doctor.yaml.safe_load(path.read_text())
    data['agent_overrides']['alpha'] = 0.003
    study_dir = tmp_path / 'study'
    shutil.copytree(STUDY.parent, study_dir)
    wrong = study_dir / 'configs' / path.name
    wrong.write_text(m22_doctor.yaml.safe_dump(data, sort_keys=False))
    errors = []
    study, _ = m22_doctor._check_study_and_configs(
        study_dir / 'study.yaml', _audit_data(), errors
    )
    assert study is not None
    assert any('override alpha expected' in error for error in errors)


def test_analyzer_statistics_use_final_at_1m_and_sample_std(tmp_path):
    values = {step: 0.5 for step in analyze_m22.EXPECTED_EVAL_STEPS}
    values[1_000_000] = 0.8
    values[900_000] = 0.7
    eval_path = tmp_path / 'eval.csv'
    _write_eval(eval_path, values)
    parsed = analyze_m22._parse_eval(eval_path)
    assert parsed['final_success'] == 0.8
    assert parsed['final_step'] == 1_000_000
    assert parsed['best_success'] == 0.8
    assert parsed['best_step'] == 1_000_000
    assert parsed['auc_status'] == 'complete'
    expected_auc = (0.5 * 700_000 + 0.6 * 100_000 + 0.75 * 100_000) / 900_000
    assert np.isclose(parsed['normalized_auc'], expected_auc)
    mean, std = analyze_m22._mean_std([0.8, 0.6, 0.7])
    assert np.isclose(mean, 0.7)
    assert np.isclose(std, 0.1)


def test_analyzer_marks_missing_auc_points_without_interpolation(tmp_path):
    values = {step: 0.5 for step in analyze_m22.EXPECTED_EVAL_STEPS if step != 500_000}
    eval_path = tmp_path / 'eval.csv'
    _write_eval(eval_path, values)
    parsed = analyze_m22._parse_eval(eval_path)
    assert parsed['auc_status'] == 'incomplete_curve'
    assert parsed['normalized_auc'] is None
    assert parsed['auc_missing_steps'] == [500_000]


def test_partial_analyzer_writes_explicit_not_paper_ready_outputs(tmp_path):
    run_root = tmp_path / 'runs'
    runs = analyze_m22.expected_runs(STUDY, run_root)
    _write_run(runs[0], root=run_root)
    output_root = tmp_path / 'results'
    analysis = analyze_m22.analyze_and_write(
        study_path=STUDY,
        run_root=run_root,
        output_root=output_root,
        allow_partial=True,
    )
    assert not analysis['paper_ready']
    assert (output_root / 'raw_seed_results.csv').is_file()
    assert 'PARTIAL / NOT PAPER READY' in (output_root / 'paper_table_final.md').read_text()
    manifest = json.loads((output_root / 'completion_manifest.json').read_text())
    assert manifest['counts']['completed'] == 1
    assert manifest['paper_ready'] is False


def test_paper_ready_gate_requires_72_runs_three_seeds_and_one_commit():
    records = []
    for algorithm in analyze_m22.ALGORITHMS:
        for environment in analyze_m22.ENVIRONMENTS:
            for seed in analyze_m22.SEEDS:
                records.append({
                    'algorithm': algorithm,
                    'environment': environment,
                    'seed': seed,
                    'status': 'completed',
                    'final_step': 1_000_000,
                    'final_success': 0.5,
                    'source_commit': 'abc123',
                })
    cells = {
        (algorithm, environment): analyze_m22._cell_summary([
            record for record in records
            if record['algorithm'] == algorithm and record['environment'] == environment
        ])
        for algorithm in analyze_m22.ALGORITHMS
        for environment in analyze_m22.ENVIRONMENTS
    }
    ready, reasons = analyze_m22._paper_ready(records, cells)
    assert ready
    assert reasons == []
    records[-1]['final_step'] = 900_000
    cells[(analyze_m22.ALGORITHMS[-1], analyze_m22.ENVIRONMENTS[-1])] = analyze_m22._cell_summary([
        record for record in records
        if record['algorithm'] == analyze_m22.ALGORITHMS[-1]
        and record['environment'] == analyze_m22.ENVIRONMENTS[-1]
    ])
    ready, reasons = analyze_m22._paper_ready(records, cells)
    assert not ready
    assert any('final@1M' in reason for reason in reasons)
