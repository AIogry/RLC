"""Within-environment analyzer for the M20B Cube transfer screen.

The analyzer is intentionally descriptive.  It computes final@1M, best,
best step, last-three mean, complete-curve normalized AUC, and exact task
columns for each run.  Contrasts are formed only between B000 and S002 inside
the same environment; no cross-environment pooling, entity-count regression,
or seed-0 significance calculation is implemented.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import load_configuration, load_study, make_run_path
from tools.m20b_doctor import ENV_SPECS, EXPECTED_CONFIG_IDS


STUDY_ID = 'M20B'
STUDY_DEFAULT = 'experiments/M20B_cube_entity_mixer_scaling/study.yaml'
RUN_ROOT_DEFAULT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'
OUTPUT_DEFAULT = 'docs/9-8/M20B_analysis.json'
MARKDOWN_DEFAULT = 'docs/9-8/M20B_analysis.md'
METRIC = 'evaluation/overall_success'
EXPECTED_STEPS = list(range(100_000, 1_000_001, 100_000))


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_eval_records(path, *, expected_task_names=()):
    path = Path(path)
    if not path.is_file():
        return [], (), 'missing'
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        fields = tuple(reader.fieldnames or ())
        task_columns = tuple(
            name for name in fields
            if name.startswith('evaluation/')
            and name.endswith('_success')
            and name != METRIC
        )
        expected_columns = {
            f'evaluation/{name}_success' for name in expected_task_names
        }
        unexpected = set(task_columns) - expected_columns
        if unexpected:
            raise ValueError(
                f'{path}: task columns are not authoritative Cube task names: '
                f'{sorted(unexpected)!r}'
            )
        records = []
        seen_steps = set()
        for row in reader:
            step = _number(row.get('step'))
            success_value = row.get(METRIC)
            if success_value is None:
                success_value = row.get('overall_success')
            success = _number(success_value)
            if step is None or success is None:
                continue
            step = int(step)
            if step in seen_steps:
                raise ValueError(f'{path}: duplicate evaluation step {step}')
            seen_steps.add(step)
            records.append({
                'step': step,
                'overall_success': success,
                'per_task_success': {
                    name: _number(row.get(name)) for name in task_columns
                },
            })
    return sorted(records, key=lambda item: item['step']), task_columns, 'available'


def _normalized_auc(records):
    points = {
        item['step']: item['overall_success']
        for item in records
        if 100_000 <= item['step'] <= 1_000_000
    }
    if set(points) != set(EXPECTED_STEPS) or len(points) != len(EXPECTED_STEPS):
        return None
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for left_step, right_step in zip(EXPECTED_STEPS, EXPECTED_STEPS[1:])
        for left_value, right_value in ((points[left_step], points[right_step]),)
    )
    return area / 900_000.0


def curve_summary(records, task_columns, expected_task_names):
    observed_steps = [item['step'] for item in records]
    final = next((item for item in records if item['step'] == 1_000_000), None)
    best = max(records, key=lambda item: (item['overall_success'], -item['step'])) if records else None
    last3 = records[-3:] if len(records) >= 3 else []
    if not records:
        curve_status = 'missing'
    elif observed_steps == EXPECTED_STEPS:
        curve_status = 'complete'
    else:
        curve_status = 'incomplete_curve'
    expected_columns = [f'evaluation/{name}_success' for name in expected_task_names]
    task_final = {
        column: None if final is None else final['per_task_success'].get(column)
        for column in expected_columns
    }
    return {
        'curve_status': curve_status,
        'observed_steps': observed_steps,
        'final_at_1m': None if final is None else final['overall_success'],
        'flat_headroom': None,
        'per_task_final_at_1m': task_final,
        'task_columns_observed': list(task_columns),
        'best': None if best is None else best['overall_success'],
        'best_step': None if best is None else best['step'],
        'last3_mean': None if not last3 else sum(item['overall_success'] for item in last3) / 3.0,
        'last3_status': 'complete' if len(last3) == 3 else 'incomplete_curve',
        'normalized_auc': _normalized_auc(records),
        'auc_status': 'complete' if _normalized_auc(records) is not None else 'incomplete_curve',
    }


def _delta(left, right, key):
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _headroom(value):
    return None if value is None else float(1.0 - value)


def _configuration_rows(study, run_root):
    paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    if len(paths) != 6:
        raise ValueError(f'M20B requires exactly six configs, found {len(paths)}')
    rows = []
    for path in paths:
        configuration = load_configuration(study, path)
        config_id = configuration.config_id
        if config_id not in EXPECTED_CONFIG_IDS:
            raise ValueError(f'Unexpected M20B config: {config_id!r}')
        data = configuration.data
        environment = data['environment']
        condition = data['condition_id']
        task_names = ENV_SPECS[environment]['tasks']
        run_dir = make_run_path(
            run_root,
            study.study_id,
            config_id,
            configuration.slug,
            environment,
            0,
        )
        records, task_columns, file_status = _read_eval_records(
            run_dir / 'eval.csv',
            expected_task_names=task_names,
        )
        summary = curve_summary(records, task_columns, task_names)
        summary['flat_headroom'] = _headroom(summary['final_at_1m'])
        rows.append({
            'config_id': config_id,
            'condition': condition,
            'environment': environment,
            'num_cubes': ENV_SPECS[environment]['num_cubes'],
            'run_dir': str(run_dir),
            'eval_file_status': file_status,
            'authoritative_task_names': task_names,
            'formal_executable': data.get('executable', True),
            **summary,
        })
    if {row['config_id'] for row in rows} != EXPECTED_CONFIG_IDS:
        raise ValueError('M20B configuration ID set mismatch')
    return rows


def collect(study_path=STUDY_DEFAULT, run_root=RUN_ROOT_DEFAULT):
    study = load_study(study_path)
    if study.study_id != STUDY_ID:
        raise ValueError(f'Expected {STUDY_ID} study, got {study.study_id!r}')
    rows = _configuration_rows(study, run_root)
    by_environment = {}
    for row in rows:
        by_environment.setdefault(row['environment'], {})[row['condition']] = row
    contrasts = {}
    for environment in ENV_SPECS:
        cells = by_environment.get(environment, {})
        if set(cells) != {'B000', 'S002'}:
            raise ValueError(f'{environment}: expected B000 and S002')
        flat = cells['B000']
        mixer = cells['S002']
        raw_delta = _delta(mixer, flat, 'final_at_1m')
        contrasts[environment] = {
            'flat_config_id': flat['config_id'],
            'mixer_config_id': mixer['config_id'],
            'flat_final': flat['final_at_1m'],
            'mixer_final': mixer['final_at_1m'],
            'raw_delta': raw_delta,
            'flat_headroom': flat['flat_headroom'],
            'mixer_headroom': mixer['flat_headroom'],
            'delta_best': _delta(mixer, flat, 'best'),
            'delta_best_step': _delta(mixer, flat, 'best_step'),
            'delta_last3_mean': _delta(mixer, flat, 'last3_mean'),
            'delta_normalized_auc': _delta(mixer, flat, 'normalized_auc'),
            'per_task_final_delta': {
                task_column: _delta(
                    {'value': mixer['per_task_final_at_1m'].get(task_column)},
                    {'value': flat['per_task_final_at_1m'].get(task_column)},
                    'value',
                )
                for task_column in flat['per_task_final_at_1m']
            },
        }
    short_names = {'cube-single-play-v0': 'single', 'cube-double-play-v0': 'double', 'cube-triple-play-v0': 'triple'}
    delta_by_count = {
        f'delta_{short_names[environment]}': contrasts[environment]['raw_delta']
        for environment in ENV_SPECS
    }
    return {
        'schema': 'm20b_within_environment_transfer_v1',
        'study_id': STUDY_ID,
        'formal_results_present': any(row['eval_file_status'] == 'available' for row in rows),
        'analysis_scope': {
            'primary_endpoint': 'final@1M',
            'secondary_endpoints': ['best', 'best_step', 'last3_mean', 'normalized_auc', 'per_task_final@1M'],
            'primary_environment': 'cube-double-play-v0',
            'within_environment_contrast': 'S002 - B000',
            'cross_environment_raw_success_pooling': 'prohibited',
            'entity_count_gap_regression': 'prohibited',
            'seed0_significance': 'prohibited',
            'interpretation': 'full structured package contrast; no token-mixing or relation attribution',
        },
        'cells': sorted(rows, key=lambda row: row['config_id']),
        'within_environment_contrasts': contrasts,
        **delta_by_count,
    }


def _display(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def markdown(report):
    lines = [
        '# M20B Cube Entity-Structured Mixer Transfer Screen',
        '',
        '对每个 Cube environment 单独比较 `S002 - B000`；不合并 raw success，不拟合实体数量到 gap 的回归，也不做 seed-0 显著性检验。',
        '',
        '| Environment | Condition | Curve | final@1M | best | best step | last3 | normalized AUC | Flat headroom |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in report['cells']:
        lines.append(
            f'| {row["environment"]} | {row["condition"]} | {row["curve_status"]} | '
            f'{_display(row["final_at_1m"])} | {_display(row["best"])} | '
            f'{_display(row["best_step"])} | {_display(row["last3_mean"])} | '
            f'{_display(row["normalized_auc"])} | {_display(row["flat_headroom"])} |'
        )
    lines.extend(['', '## Within-environment contrasts', ''])
    for environment, contrast in report['within_environment_contrasts'].items():
        lines.append(
            f'- `{environment}`: raw Δ = {_display(contrast["raw_delta"])}, '
            f'best Δ = {_display(contrast["delta_best"])}, '
            f'last3 Δ = {_display(contrast["delta_last3_mean"])}, '
            f'AUC Δ = {_display(contrast["delta_normalized_auc"])}.'
        )
    lines.extend([
        '',
        'Task columns are retained under their exact authoritative `evaluation/<task_name>_success` names.',
        '',
    ])
    return '\n'.join(lines)


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=STUDY_DEFAULT)
    parser.add_argument('--run-root', default=RUN_ROOT_DEFAULT)
    parser.add_argument('--output', default=OUTPUT_DEFAULT)
    parser.add_argument('--markdown-output', default=MARKDOWN_DEFAULT)
    args = parser.parse_args(argv)
    report = collect(args.study, args.run_root)
    _write(args.output, __import__('json').dumps(report, indent=2, sort_keys=True) + '\n')
    _write(args.markdown_output, markdown(report) + '\n')
    print(f'M20B analysis written: {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
