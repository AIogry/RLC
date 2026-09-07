"""Schema-first, within-task-only analyzer for future M20A Phase-2 results.

Phase 2 is frozen but may not yet have formal M20A training results.  This
tool validates the frozen 18-cell design and emits an explicitly incomplete
analysis skeleton; it never launches a run or pools raw success across task
families.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from impls.experiment import load_configuration, load_study, make_run_path


STUDY_DEFAULT = 'experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml'
RUN_ROOT_DEFAULT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'
OUTPUT_DEFAULT = 'docs/9-8/M20A_phase1_analysis_skeleton.json'
MARKDOWN_DEFAULT = 'docs/9-8/M20A_phase1_analysis_skeleton.md'
EXPECTED_STEPS = list(range(100_000, 1_000_001, 100_000))
METRIC = 'evaluation/overall_success'


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_records(path):
    path = Path(path)
    if not path.is_file():
        return [], ()
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        task_columns = tuple(
            name for name in (reader.fieldnames or ())
            if name.startswith('evaluation/')
            and name.endswith('_success')
            and name != METRIC
        )
        records = []
        for row in reader:
            step = _number(row.get('step'))
            success = _number(row.get(METRIC) or row.get('overall_success'))
            if step is None or success is None:
                continue
            records.append({
                'step': int(step),
                'overall_success': success,
                'per_task_success': {name: _number(row.get(name)) for name in task_columns},
            })
    return sorted(records, key=lambda item: item['step']), task_columns


def _auc(records):
    points = [
        (item['step'], item['overall_success']) for item in records
        if 100_000 <= item['step'] <= 1_000_000
    ]
    if [step for step, _ in points] != EXPECTED_STEPS:
        return None
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / 900_000.0


def curve_summary(records):
    observed_steps = [item['step'] for item in records]
    final = next((item for item in records if item['step'] == 1_000_000), None)
    best = max(records, key=lambda item: (item['overall_success'], -item['step'])) if records else None
    values = [item['overall_success'] for item in records]
    return {
        'curve_status': 'complete' if observed_steps == EXPECTED_STEPS else ('partial' if records else 'missing'),
        'observed_steps': observed_steps,
        'final_at_1m': None if final is None else final['overall_success'],
        'per_task_final_at_1m': {} if final is None else final['per_task_success'],
        'best': None if best is None else best['overall_success'],
        'best_step': None if best is None else best['step'],
        'last3_mean': None if not values else sum(values[-3:]) / len(values[-3:]),
        'normalized_auc': _auc(records),
    }


def _contrast(left, right, key='final_at_1m'):
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return {'status': 'missing_required_results', 'delta': None}
    return {'status': 'available', 'delta': float(left_value - right_value)}


def collect(study_path, run_root):
    study = load_study(study_path)
    if study.study_id != 'M20A':
        raise ValueError(f'Expected M20A study, got {study.study_id!r}')
    config_paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    if len(config_paths) != 18:
        raise ValueError(f'M20A requires 18 configs, found {len(config_paths)}')
    rows = []
    by_task = {}
    for path in config_paths:
        configuration = load_configuration(study, path)
        data = configuration.data
        if data.get('protocol_stage') != 'phase2_formal' or data.get('executable') is not True:
            raise ValueError(f'{configuration.config_id}: expected executable frozen Phase-2 configuration')
        factors = data['factors']
        task = factors['task_family']
        relation_mode = factors['relation_mode']
        readout = factors['readout']
        run_dir = make_run_path(
            run_root, study.study_id, configuration.config_id, configuration.slug,
            data['environment'], 0,
        )
        records, task_columns = _read_records(run_dir / 'eval.csv')
        row = {
            'config_id': configuration.config_id,
            'task_family': task,
            'environment': data['environment'],
            'relation_mode': relation_mode,
            'readout': readout,
            'run_dir': str(run_dir),
            'formal_executable': data['executable'],
            'blocked_by': data.get('blocked_by'),
            'eval_task_columns': list(task_columns),
            **curve_summary(records),
        }
        rows.append(row)
        by_task.setdefault(task, {})[(relation_mode, readout)] = row

    contrasts = {}
    for task, cells in by_task.items():
        contrasts[task] = {
            'correct_minus_shuffled_mean': _contrast(
                cells.get(('correct', 'mean_context'), {}),
                cells.get(('shuffled', 'mean_context'), {}),
            ),
            'correct_minus_zero_mean': _contrast(
                cells.get(('correct', 'mean_context'), {}),
                cells.get(('zero', 'mean_context'), {}),
            ),
            'hybrid_minus_mean_correct': _contrast(
                cells.get(('correct', 'hybrid_context_query'), {}),
                cells.get(('correct', 'mean_context'), {}),
            ),
        }
    return {
        'schema': 'm20a_future_analysis_v1',
        'study_id': study.study_id,
        'formal_results_present': any(row['curve_status'] != 'missing' for row in rows),
        'analysis_scope': {
            'primary_endpoint': 'final@1M',
            'secondary_endpoints': ['best', 'best_step', 'last3_mean', 'normalized_auc', 'per_task_final@1M'],
            'within_task_contrasts_only': True,
            'cross_task_raw_success_pooling': 'prohibited',
            'hybrid_interpretation': 'readout_package_contrast_not_parameter_or_mac_matched',
        },
        'cells': rows,
        'within_task_contrasts': contrasts,
        'phase2_status': 'No formal M20A result is interpreted by this analyzer until an evaluation curve exists.',
    }


def _markdown(report):
    lines = [
        '# M20A future analysis skeleton',
        '',
        'No completed formal M20A Phase-2 result is available at this report-generation time.',
        '',
        '| Task | Relation | Readout | Curve status | final@1M | best | best step | last3 | AUC |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in report['cells']:
        def display(value):
            return '—' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)
        lines.append(
            f'| {row["task_family"]} | {row["relation_mode"]} | {row["readout"]} | '
            f'{row["curve_status"]} | {display(row["final_at_1m"])} | '
            f'{display(row["best"])} | {display(row["best_step"])} | '
            f'{display(row["last3_mean"])} | {display(row["normalized_auc"])} |'
        )
    lines.extend([
        '',
        'Contrasts are deliberately calculated per task only: Correct−Shuffled, Correct−Zero, and HybridContextQuery−Mean. Raw Puzzle/Cube/Scene success is never pooled.',
        '',
    ])
    return '\n'.join(lines)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=STUDY_DEFAULT)
    parser.add_argument('--run-root', default=RUN_ROOT_DEFAULT)
    parser.add_argument('--output', default=OUTPUT_DEFAULT)
    parser.add_argument('--markdown-output', default=MARKDOWN_DEFAULT)
    args = parser.parse_args(argv)
    report = collect(args.study, args.run_root)
    _write(args.output, report)
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report) + '\n')
    print(f'M20A analysis skeleton written: {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
