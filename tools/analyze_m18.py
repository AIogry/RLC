"""Aggregate completed M18 recurrent-computation scaling runs.

This post-training tool reads only run artifacts and writes a separate report
directory.  It never launches training, restores/resaves checkpoints, or
silently treats missing/partial seed-0 runs as completed results.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from impls.experiment import load_study, make_run_path, prepare_run_design


STUDY_ID = 'M18'
ENVIRONMENT = 'puzzle-4x4-play-v0'
METRIC = 'evaluation/overall_success'
EXPECTED_STEPS = list(range(100_000, 1_000_001, 100_000))
K_VALUES = (1, 2, 4, 8)


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open() as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def _task_columns(fieldnames):
    return tuple(
        name for name in (fieldnames or ())
        if name.startswith('evaluation/')
        and name.endswith('_success')
        and name != METRIC
    )


def _read_eval_records(path):
    path = Path(path)
    if not path.is_file():
        return [], ()
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        tasks = _task_columns(reader.fieldnames)
        rows = []
        for row in reader:
            step = _float(row.get('step'))
            overall = _float(row.get(METRIC) or row.get('overall_success'))
            if step is None or overall is None:
                continue
            rows.append({
                'step': int(step),
                'overall_success': overall,
                'task_success': {name: _float(row.get(name)) for name in tasks},
            })
    return sorted(rows, key=lambda item: item['step']), tasks


def _normalized_auc(records):
    if len(records) < 2:
        return None
    points = [
        (record['step'], record['overall_success'])
        for record in records
        if 100_000 <= record['step'] <= 1_000_000
    ]
    if len(points) < 2:
        return None
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
    )
    return area / 900_000.0


def curve_summary(records):
    steps = [record['step'] for record in records]
    values = [record['overall_success'] for record in records]
    complete = steps == EXPECTED_STEPS
    best = max(records, key=lambda record: (record['overall_success'], -record['step'])) if records else None
    final = records[-1] if records and records[-1]['step'] == 1_000_000 else None
    return {
        'curve_status': 'complete' if complete else ('partial' if records else 'missing'),
        'num_eval_points': len(records),
        'observed_steps': steps,
        'final_success': None if final is None else final['overall_success'],
        'task_final_success': {} if final is None else final['task_success'],
        'best_success': None if best is None else best['overall_success'],
        'best_step': None if best is None else best['step'],
        'last3_mean': None if not values else sum(values[-3:]) / len(values[-3:]),
        'normalized_eval_auc': _normalized_auc(records) if complete else None,
    }


def _runtime_validation(metadata, resolved, configuration):
    if not metadata:
        return 'missing_metadata'
    if metadata.get('study_id') != STUDY_ID:
        return f"study_id={metadata.get('study_id')!r}"
    if metadata.get('config_id') != configuration.config_id:
        return f"config_id={metadata.get('config_id')!r}"
    if metadata.get('environment') != ENVIRONMENT or int(metadata.get('seed', -1)) != 0:
        return 'environment_or_seed_mismatch'
    agent = resolved.get('algorithm_config', {}).get('agent', {})
    if not isinstance(agent, dict):
        return 'missing_resolved_agent'
    try:
        resolved_alpha = float(agent.get('alpha'))
    except (TypeError, ValueError):
        return 'missing_or_invalid_resolved_alpha'
    if abs(resolved_alpha - 0.4) > 1e-12:
        return 'resolved_alpha_not_0p4'
    k = configuration.data.get('factors', {}).get('recurrent_compute_budget_K')
    for slot_name in ('actor', 'value', 'critic'):
        slot = agent.get('compute', {}).get(slot_name, {})
        topology = slot.get('topology_kwargs', {})
        if (
            slot.get('structure') != 'puzzle_tokens'
            or slot.get('block') != 'mlp_mixer'
            or slot.get('topology') != 'single_state'
            or topology.get('iterations') != k
            or topology.get('input_mapping') != 'identity'
            or topology.get('state_init') != 'zero_buffer'
            or topology.get('input_injection') != 'z_plus_x'
            or topology.get('residual') is not False
            or slot.get('parameter_sharing') != 'shared'
            or slot.get('readout') != 'mean_context'
        ):
            return f'{slot_name}_resolved_semantics_mismatch'
    return 'valid'


def _architecture(metadata):
    architecture = metadata.get('architecture_accounting', {})
    slots = architecture.get('slots', {}) if isinstance(architecture, dict) else {}
    slot_accounting = metadata.get('computation_slot_accounting', {})
    actor = slot_accounting.get('actor', {}) if isinstance(slot_accounting, dict) else {}
    return {
        'total_trainable_params': architecture.get('total_trainable_params'),
        'total_dense_macs': architecture.get('total_dense_macs'),
        'actor_trainable_params': actor.get('trainable_params'),
        'actor_body_dense_macs': actor.get('structured_body_dense_macs'),
        'actor_unique_mixer_layers': actor.get('unique_mixer_layers'),
        'actor_executed_mixer_layers': actor.get('executed_mixer_layers'),
        'actor_executed_depth': actor.get('executed_sequential_depth'),
        'actor_buffer_elements': actor.get('buffer_elements'),
        'slot_depths': {
            slot_name: slots.get(slot_name, {}).get('sequential_depth')
            for slot_name in ('actor', 'value', 'critic')
        },
        'slot_params': {
            slot_name: slot_accounting.get(slot_name, {}).get('trainable_params')
            for slot_name in ('actor', 'value', 'critic')
        },
    }


def collect(study_path, run_root, run_attempt=0):
    study = load_study(study_path)
    if study.study_id != STUDY_ID:
        raise ValueError(f'Expected M18 Study, got {study.study_id!r}')
    rows = []
    config_paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    for config_path in config_paths:
        _, configuration = prepare_run_design(study.path, config_path)
        k = configuration.data.get('factors', {}).get('recurrent_compute_budget_K')
        run_dir = make_run_path(
            run_root, study.study_id, configuration.config_id, configuration.slug,
            ENVIRONMENT, 0, run_attempt=run_attempt,
        )
        metadata = _read_json(run_dir / 'runtime_metadata.json')
        resolved = _read_json(run_dir / 'resolved_config.json')
        records, task_columns = _read_eval_records(run_dir / 'eval.csv')
        rows.append({
            'study_id': study.study_id,
            'config_id': configuration.config_id,
            'environment': ENVIRONMENT,
            'seed': 0,
            'run_attempt': run_attempt,
            'K': k,
            'L': configuration.data.get('factors', {}).get('block_depth_L'),
            'alpha': configuration.data.get('factors', {}).get('alpha'),
            'run_dir': str(run_dir),
            'run_status': metadata.get('status', 'missing'),
            'runtime_validation': _runtime_validation(metadata, resolved, configuration),
            'task_columns': list(task_columns),
            **curve_summary(records),
            **_architecture(metadata),
        })
    return sorted(rows, key=lambda row: row['K'])


def descriptive_effects(rows):
    by_k = {row['K']: row for row in rows}
    pairs = ((2, 1), (4, 2), (8, 4), (4, 1), (8, 1))
    effects = []
    for high, low in pairs:
        row = {'contrast': f'K{high}-K{low}', 'K_high': high, 'K_low': low}
        for metric in ('final_success', 'normalized_eval_auc', 'last3_mean'):
            left = by_k.get(high, {}).get(metric)
            right = by_k.get(low, {}).get(metric)
            row[f'delta_{metric}'] = None if left is None or right is None else left - right
        effects.append(row)
    return effects


def _number(value):
    if value is None:
        return ''
    return f'{value:.6f}' if isinstance(value, float) else str(value)


def markdown(rows, effects):
    complete = [
        row for row in rows
        if row['run_status'] == 'completed'
        and row['curve_status'] == 'complete'
        and row['runtime_validation'] == 'valid'
    ]
    lines = [
        '# M18 结果汇总（自动生成）',
        '',
        '- 主终点：`evaluation/overall_success` 的 final@1M。',
        '- AUC：100k–1M 共 10 个 checkpoint 的梯形积分除以 900k；缺点时留空。',
        '- 参数、unique Mixer、MAC 和 depth 的主表为 actor 单一路径口径；critic ensemble 的物理层数在 JSON 中单列。',
        '- 本 Study 只有 seed=0；下表与差值均为描述性结果，不能用于显著性检验或宣称通用/单调 scaling law。',
        '',
        f'- 完整且 provenance 有效的运行：{len(complete)}/{len(rows)}。',
        '',
        '| K | Run 状态 | 曲线 | provenance | Params(total/actor) | Unique Mixer | Executed Mixer | Actor MAC | Depth | Buffer | Final@1M | Best | Best step | Last-3 | Norm AUC |',
        '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f'| {row["K"]} | {row["run_status"]} | {row["curve_status"]} | '
            f'{row["runtime_validation"]} | '
            f'{_number(row["total_trainable_params"])}/{_number(row["actor_trainable_params"])} | '
            f'{_number(row["actor_unique_mixer_layers"])} | '
            f'{_number(row["actor_executed_mixer_layers"])} | '
            f'{_number(row["actor_body_dense_macs"])} | '
            f'{_number(row["actor_executed_depth"])} | {_number(row["actor_buffer_elements"])} | '
            f'{_number(row["final_success"])} | {_number(row["best_success"])} | '
            f'{_number(row["best_step"])} | {_number(row["last3_mean"])} | '
            f'{_number(row["normalized_eval_auc"])} |'
        )
    lines.extend([
        '',
        '## 描述性相邻/锚点差值',
        '',
        '| Contrast | Δ final@1M | Δ normalized AUC | Δ last-3 |',
        '|---|---:|---:|---:|',
    ])
    for effect in effects:
        lines.append(
            f'| {effect["contrast"]} | {_number(effect["delta_final_success"])} | '
            f'{_number(effect["delta_normalized_eval_auc"])} | '
            f'{_number(effect["delta_last3_mean"])} |'
        )
    lines.extend(['', '## Task-level final success', ''])
    task_names = sorted({name for row in rows for name in row.get('task_columns', [])})
    if task_names:
        lines.append('| K | ' + ' | '.join(task_names) + ' |')
        lines.append('|' + '---:|' * (len(task_names) + 1))
        for row in rows:
            values = row.get('task_final_success', {})
            lines.append('| ' + str(row['K']) + ' | ' + ' | '.join(
                _number(values.get(name)) for name in task_names
            ) + ' |')
    else:
        lines.append('尚无完整 `eval.csv` task-level 记录。')
    if len(complete) != len(rows):
        lines.extend([
            '',
            '> 结果尚未齐全或 provenance 未通过；不得对 K scaling 作正式科学结论。',
        ])
    return '\n'.join(lines) + '\n'


def _write_rows_csv(rows, path):
    fields = (
        'study_id', 'config_id', 'environment', 'seed', 'run_attempt', 'K', 'L', 'alpha',
        'run_status', 'curve_status', 'runtime_validation', 'num_eval_points',
        'final_success', 'best_success', 'best_step', 'last3_mean', 'normalized_eval_auc',
        'total_trainable_params', 'total_dense_macs', 'actor_trainable_params',
        'actor_body_dense_macs', 'actor_unique_mixer_layers', 'actor_executed_mixer_layers',
        'actor_executed_depth', 'actor_buffer_elements', 'run_dir',
    )
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _write_raw_eval(rows, path):
    fields = ('study_id', 'config_id', 'K', 'step', 'overall_success', 'task_metric', 'task_success', 'run_dir')
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            records, _ = _read_eval_records(Path(row['run_dir']) / 'eval.csv')
            for record in records:
                for task_metric, value in record['task_success'].items():
                    writer.writerow({
                        'study_id': row['study_id'],
                        'config_id': row['config_id'],
                        'K': row['K'],
                        'step': record['step'],
                        'overall_success': record['overall_success'],
                        'task_metric': task_metric,
                        'task_success': value,
                        'run_dir': row['run_dir'],
                    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M18_puzzle_recurrent_compute_scaling/study.yaml')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--run-attempt', type=int, default=0)
    parser.add_argument('--output-dir', default='docs/8-30/M18_results')
    args = parser.parse_args(argv)
    if args.run_attempt < 0:
        raise SystemExit('--run-attempt must be non-negative')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.study, args.run_root, args.run_attempt)
    effects = descriptive_effects(rows)
    _write_rows_csv(rows, output_dir / 'm18_results_long.csv')
    (output_dir / 'm18_results_summary.json').write_text(
        json.dumps({'rows': rows, 'descriptive_effects': effects}, indent=2, sort_keys=True) + '\n'
    )
    _write_raw_eval(rows, output_dir / 'm18_raw_eval_long.csv')
    (output_dir / 'M18_results_summary.md').write_text(markdown(rows, effects))
    complete = sum(
        row['run_status'] == 'completed'
        and row['curve_status'] == 'complete'
        and row['runtime_validation'] == 'valid'
        for row in rows
    )
    print(f'Wrote {len(rows)} M18 rows to {output_dir.resolve()}')
    print(f'complete_and_valid={complete}/{len(rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
