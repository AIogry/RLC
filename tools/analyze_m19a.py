"""Aggregate the fixed M19A Flat → EntityMLP → Mixer comparison.

The tool reads only immutable run artifacts.  It does not launch training,
restore checkpoints, or replace missing M19A results with any other study.
Before E001 completes it writes an explicitly incomplete planning report
instead of manufacturing a three-method conclusion.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from impls.experiment import load_study, make_run_path, prepare_run_design


STUDY_ID = 'M19A'
ENVIRONMENT = 'puzzle-4x4-play-v0'
SEED = 0
METRIC = 'evaluation/overall_success'
EXPECTED_STEPS = list(range(100_000, 1_000_001, 100_000))
METHODS = (
    ('Flat MLP', 'anchor_flat', 'M16B-4x4-B000'),
    ('Entity Token + Entity MLP', 'entity', 'M19A-4x4-E001'),
    ('Entity Token + MLP-Mixer', 'anchor_mixer', 'M16B-4x4-S002'),
)


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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
            value = _float(row.get(METRIC) or row.get('overall_success'))
            if step is None or value is None:
                continue
            rows.append({
                'step': int(step),
                'overall_success': value,
                'task_success': {name: _float(row.get(name)) for name in tasks},
            })
    return sorted(rows, key=lambda row: row['step']), tasks


def _auc(records):
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
    final = records[-1] if records and records[-1]['step'] == 1_000_000 else None
    best = max(records, key=lambda row: (row['overall_success'], -row['step'])) if records else None
    return {
        'curve_status': 'complete' if steps == EXPECTED_STEPS else ('partial' if records else 'missing'),
        'num_eval_points': len(records),
        'observed_steps': steps,
        'final_success': None if final is None else final['overall_success'],
        'task_final_success': {} if final is None else final['task_success'],
        'best_success': None if best is None else best['overall_success'],
        'best_step': None if best is None else best['step'],
        'last3_mean': sum(values[-3:]) / len(values[-3:]) if values else None,
        'normalized_eval_auc': _auc(records) if steps == EXPECTED_STEPS else None,
    }


def _architecture(metadata):
    architecture = metadata.get('architecture_accounting', {})
    slots = architecture.get('slots', {}) if isinstance(architecture, dict) else {}
    actor = slots.get('actor', {}) if isinstance(slots, dict) else {}
    return {
        'total_trainable_params': architecture.get('total_trainable_params'),
        'total_dense_macs': architecture.get('total_dense_macs'),
        # This is deliberately the structured/vector body depth before the
        # algorithm action/scalar head, making Entity=6 and Mixer=10 visible.
        'body_depth': actor.get('computation_body_sequential_depth'),
        'total_slot_depth': actor.get('sequential_depth'),
        'actor_block': actor.get('block'),
    }


def _throughput(run_dir):
    records, _ = _read_train_records(Path(run_dir) / 'train.csv')
    if not records:
        return {
            'training_total_seconds': None,
            'seconds_per_step': None,
            'steps_per_second': None,
        }
    latest = records[-1]
    total_seconds = latest.get('time/total_seconds')
    if total_seconds is None:
        total_seconds = latest.get('total_seconds')
    step = latest.get('step')
    total_seconds = _float(total_seconds)
    step = _float(step)
    if total_seconds is None or total_seconds <= 0 or step is None or step <= 0:
        return {
            'training_total_seconds': total_seconds,
            'seconds_per_step': None,
            'steps_per_second': None,
        }
    return {
        'training_total_seconds': total_seconds,
        'seconds_per_step': total_seconds / step,
        'steps_per_second': step / total_seconds,
    }


def _read_train_records(path):
    path = Path(path)
    if not path.is_file():
        return [], ()
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
        return rows, tuple(reader.fieldnames or ())


def _runtime_validation(kind, metadata, resolved, expected_config_id):
    if not metadata:
        return 'missing_metadata'
    if metadata.get('status') != 'completed':
        return f'status={metadata.get("status")!r}'
    expected_study = 'M19A' if kind == 'entity' else 'M16B'
    if metadata.get('study_id') != expected_study:
        return f'study_id={metadata.get("study_id")!r}'
    if metadata.get('config_id') != expected_config_id:
        return f'config_id={metadata.get("config_id")!r}'
    if metadata.get('environment') != ENVIRONMENT or int(metadata.get('seed', -1)) != SEED:
        return 'environment_or_seed_mismatch'
    agent = resolved.get('algorithm_config', {}).get('agent', {})
    if not isinstance(agent, dict) or agent.get('agent_name') != 'gciql':
        return 'missing_or_invalid_resolved_gciql_agent'
    if _float(agent.get('alpha')) != 1.0:
        return 'resolved_alpha_not_1p0'
    for slot_name in ('actor', 'value', 'critic'):
        slot = agent.get('compute', {}).get(slot_name, {})
        if kind == 'anchor_flat':
            if slot.get('enabled') is not False or slot.get('structure') != 'vector':
                return f'{slot_name}_flat_semantics_mismatch'
        elif kind == 'anchor_mixer':
            structure = slot.get('structure_kwargs', {})
            if not (
                slot.get('enabled') is True
                and slot.get('structure') == 'puzzle_tokens'
                and slot.get('block') == 'mlp_mixer'
                and slot.get('topology') == 'feedforward'
                and slot.get('credit') == 'direct'
                and structure.get('num_mixer_blocks') == 2
                and structure.get('tm_mode') == 'none'
            ):
                return f'{slot_name}_mixer_semantics_mismatch'
        else:
            block = slot.get('block_kwargs', {})
            if not (
                slot.get('enabled') is True
                and slot.get('structure') == 'puzzle_tokens'
                and slot.get('block') == 'entity_mlp'
                and slot.get('topology') == 'feedforward'
                and slot.get('credit') == 'direct'
                and slot.get('readout') == 'mean_context'
                and block.get('num_blocks') == 2
                and block.get('channel_hidden_dim') == 256
            ):
                return f'{slot_name}_entity_semantics_mismatch'
    return 'valid'


def _source_rows(study, configuration, run_root, run_attempt):
    anchors = study.data.get('historical_anchors', {})
    flat_path = Path(anchors.get('anchor_flat', {}).get('source_run', ''))
    mixer_path = Path(anchors.get('anchor_mixer', {}).get('source_run', ''))
    entity_path = make_run_path(
        run_root, study.study_id, configuration.config_id, configuration.slug,
        ENVIRONMENT, SEED, run_attempt=run_attempt,
    )
    source_paths = {
        'anchor_flat': flat_path,
        'entity': entity_path,
        'anchor_mixer': mixer_path,
    }
    rows = []
    for method, kind, config_id in METHODS:
        run_dir = source_paths[kind]
        metadata = _read_json(run_dir / 'runtime_metadata.json')
        resolved = _read_json(run_dir / 'resolved_config.json')
        records, task_columns = _read_eval_records(run_dir / 'eval.csv')
        rows.append({
            'method': method,
            'source_kind': kind,
            'study_id': 'M19A' if kind == 'entity' else 'M16B',
            'config_id': config_id,
            'environment': ENVIRONMENT,
            'seed': SEED,
            'run_attempt': run_attempt if kind == 'entity' else 0,
            'run_dir': str(run_dir),
            'run_status': metadata.get('status', 'missing'),
            'runtime_validation': _runtime_validation(kind, metadata, resolved, config_id),
            'task_columns': list(task_columns),
            **curve_summary(records),
            **_architecture(metadata),
            **_throughput(run_dir),
        })
    return rows


def collect(study_path, run_root, run_attempt=0):
    study = load_study(study_path)
    if study.study_id != STUDY_ID:
        raise ValueError(f'Expected M19A study, got {study.study_id!r}')
    configs = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    if len(configs) != 1:
        raise ValueError(f'M19A requires exactly one config, found {len(configs)}')
    _, configuration = prepare_run_design(study.path, configs[0])
    if configuration.config_id != 'M19A-4x4-E001':
        raise ValueError(f'Unexpected M19A config ID: {configuration.config_id!r}')
    return _source_rows(study, configuration, run_root, run_attempt)


def _delta(left, right, key):
    left_value = left.get(key)
    right_value = right.get(key)
    return None if left_value is None or right_value is None else left_value - right_value


def descriptive_effects(rows):
    by_kind = {row['source_kind']: row for row in rows}
    entity = by_kind['entity']
    flat = by_kind['anchor_flat']
    mixer = by_kind['anchor_mixer']
    effects = []
    for name, high, low, definition in (
        (
            'Delta_structured_factorization_package', entity, flat,
            'J(EntityMLP) - J(FlatMLP)',
        ),
        (
            'Delta_added_token_mixing_branch', mixer, entity,
            'J(MLPMixer) - J(EntityMLP)',
        ),
    ):
        task_names = sorted(set(high.get('task_final_success', {})) | set(low.get('task_final_success', {})))
        effects.append({
            'effect': name,
            'definition': definition,
            'delta_final_success': _delta(high, low, 'final_success'),
            'delta_best_success': _delta(high, low, 'best_success'),
            'delta_last3_mean': _delta(high, low, 'last3_mean'),
            'delta_normalized_eval_auc': _delta(high, low, 'normalized_eval_auc'),
            'task_final_deltas': {
                task: _delta(high.get('task_final_success', {}), low.get('task_final_success', {}), task)
                for task in task_names
            },
        })
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
        '# M19A 结果汇总（自动生成）',
        '',
        '- 固定顺序：Flat MLP → Entity Token + Entity MLP → Entity Token + MLP-Mixer。',
        '- Flat/Mixer 为完成的 M16B alpha=1.0 历史锚点；仅 EntityMLP 是 M19A 的新 formal run。',
        '- 主终点为 `evaluation/overall_success` 的 final@1M；AUC 为 100k–1M 十个 checkpoint 的梯形积分除以 900k。',
        '- `Depth` 是 actor computation body 的 Dense depth（不含算法 action/scalar head）：Entity L2=6、Mixer L2=10。',
        '- 只有 seed=0；所有差值均为描述性量，不能用于显著性、因果、预算匹配或跨 Puzzle 大小结论。',
        '',
        f'- 完整且 provenance 有效的三组结果：{len(complete)}/{len(rows)}。',
        '',
        '| Method | Final@1M | Best | Best Step | Last-3 | AUC | Params | MAC | Depth | Run status | Provenance |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in rows:
        lines.append(
            f'| {row["method"]} | {_number(row["final_success"])} | '
            f'{_number(row["best_success"])} | {_number(row["best_step"])} | '
            f'{_number(row["last3_mean"])} | {_number(row["normalized_eval_auc"])} | '
            f'{_number(row["total_trainable_params"])} | {_number(row["total_dense_macs"])} | '
            f'{_number(row["body_depth"])} | {row["run_status"]} | '
            f'{row["runtime_validation"]} |'
        )
    lines.extend(['', '## Task-level final success', ''])
    task_names = sorted({task for row in rows for task in row.get('task_columns', [])})
    if task_names:
        lines.append('| Method | ' + ' | '.join(task_names) + ' |')
        lines.append('|' + '---|' * (len(task_names) + 1))
        for row in rows:
            values = row.get('task_final_success', {})
            lines.append('| ' + row['method'] + ' | ' + ' | '.join(
                _number(values.get(task)) for task in task_names
            ) + ' |')
    else:
        lines.append('尚无可用的 task-level `eval.csv` 记录。')
    lines.extend(['', '## Descriptive deltas', ''])
    lines.append('| Effect | Definition | Δ Final@1M | Δ Best | Δ Last-3 | Δ AUC |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for effect in effects:
        lines.append(
            f'| {effect["effect"]} | {effect["definition"]} | '
            f'{_number(effect["delta_final_success"])} | '
            f'{_number(effect["delta_best_success"])} | '
            f'{_number(effect["delta_last3_mean"])} | '
            f'{_number(effect["delta_normalized_eval_auc"])} |'
        )
    if task_names:
        lines.extend(['', '### Task-level final-success deltas', ''])
        lines.append('| Effect | ' + ' | '.join(task_names) + ' |')
        lines.append('|' + '---|' * (len(task_names) + 1))
        for effect in effects:
            lines.append('| ' + effect['effect'] + ' | ' + ' | '.join(
                _number(effect['task_final_deltas'].get(task)) for task in task_names
            ) + ' |')
    throughputs = [row for row in rows if row.get('training_total_seconds') is not None]
    if throughputs:
        lines.extend(['', '## Optional wall-clock / throughput', ''])
        lines.append('| Method | Training seconds | Seconds / step | Steps / second |')
        lines.append('|---|---:|---:|---:|')
        for row in throughputs:
            lines.append(
                f'| {row["method"]} | {_number(row["training_total_seconds"])} | '
                f'{_number(row["seconds_per_step"])} | {_number(row["steps_per_second"])} |'
            )
    if len(complete) != len(rows):
        lines.extend([
            '',
            '> E001 尚未完成或 provenance 未通过；本文件目前是锚点与分析计划汇总，不得给出 M19A 三组科学结论。',
        ])
    return '\n'.join(lines) + '\n'


def _write_rows(rows, path):
    fields = (
        'method', 'source_kind', 'study_id', 'config_id', 'environment', 'seed',
        'run_attempt', 'run_status', 'runtime_validation', 'curve_status',
        'num_eval_points', 'final_success', 'best_success', 'best_step',
        'last3_mean', 'normalized_eval_auc', 'total_trainable_params',
        'total_dense_macs', 'body_depth', 'total_slot_depth', 'actor_block',
        'training_total_seconds', 'seconds_per_step', 'steps_per_second', 'run_dir',
    )
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _write_task_rows(rows, path):
    fields = ('method', 'source_kind', 'config_id', 'task_metric', 'final_success', 'run_dir')
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for task, value in row.get('task_final_success', {}).items():
                writer.writerow({
                    'method': row['method'],
                    'source_kind': row['source_kind'],
                    'config_id': row['config_id'],
                    'task_metric': task,
                    'final_success': value,
                    'run_dir': row['run_dir'],
                })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M19A_puzzle_entity_factorization_isolation/study.yaml')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--run-attempt', type=int, default=0)
    parser.add_argument('--output-dir', default='docs/9-4/M19A_results')
    args = parser.parse_args(argv)
    if args.run_attempt < 0:
        raise SystemExit('--run-attempt must be non-negative')
    rows = collect(args.study, args.run_root, args.run_attempt)
    effects = descriptive_effects(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_dir / 'm19a_results_summary.csv')
    _write_task_rows(rows, output_dir / 'm19a_task_final_success.csv')
    (output_dir / 'm19a_results_summary.json').write_text(
        json.dumps({'rows': rows, 'descriptive_effects': effects}, indent=2, sort_keys=True) + '\n'
    )
    (output_dir / 'M19A_results_summary.md').write_text(markdown(rows, effects))
    complete = sum(
        row['run_status'] == 'completed'
        and row['curve_status'] == 'complete'
        and row['runtime_validation'] == 'valid'
        for row in rows
    )
    print(f'Wrote {len(rows)} M19A comparison rows to {output_dir.resolve()}')
    print(f'complete_and_valid={complete}/{len(rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
