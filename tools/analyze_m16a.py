"""Aggregate M16A evaluation curves into auditable tables and summaries.

The script is intended for use after the user has completed the formal runs.
It never starts training and never modifies checkpoints.  Missing or partial
runs are reported explicitly instead of being silently dropped.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from impls.experiment import load_study, make_run_path, prepare_run_design


EXPECTED_STEPS = list(range(100_000, 1_000_001, 100_000))
METRIC = 'evaluation/overall_success'
CONDITIONS = ('B000', 'S001', 'S002', 'S004')


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_eval(path):
    if not path.is_file():
        return []
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            step = _float(row.get('step'))
            value = _float(row.get(METRIC) or row.get('overall_success'))
            if step is not None and value is not None:
                rows.append((int(step), value))
    return sorted(rows)


def _auc(rows):
    if not rows:
        return None
    points = [(step, value) for step, value in rows if 100_000 <= step <= 1_000_000]
    if len(points) < 2:
        return None
    area = 0.0
    for (left_step, left_value), (right_step, right_value) in zip(points, points[1:]):
        area += (right_step - left_step) * (left_value + right_value) / 2.0
    return area / 900_000.0


def _curve_summary(rows):
    steps = [step for step, _ in rows]
    values = [value for _, value in rows]
    expected = steps == EXPECTED_STEPS
    best_step, best_value = max(rows, key=lambda item: (item[1], -item[0])) if rows else (None, None)
    return {
        'status': 'complete' if expected else ('partial' if rows else 'missing'),
        'metric': METRIC,
        'num_eval_points': len(rows),
        'observed_steps': steps,
        'final_success': values[-1] if rows and steps[-1] == 1_000_000 else None,
        'best_success': best_value,
        'best_step': best_step,
        'last3_mean': sum(values[-3:]) / len(values[-3:]) if values else None,
        'normalized_eval_auc': _auc(rows) if expected else None,
    }


def _architecture(metadata):
    report = metadata.get('architecture_accounting', {})
    slots = report.get('slots', {}) if isinstance(report, dict) else {}
    return {
        'total_trainable_params': report.get('total_trainable_params'),
        'total_dense_macs': report.get('total_dense_macs'),
        'actor_depth': slots.get('actor', {}).get('sequential_depth'),
        'value_depth': slots.get('value', {}).get('sequential_depth'),
        'critic_depth': slots.get('critic', {}).get('sequential_depth'),
    }


def collect(study_path, run_root, run_attempt=0):
    study = load_study(study_path)
    config_dir = Path(study.path).parent / 'configs'
    configurations = [
        prepare_run_design(study_path, path)[1]
        for path in sorted(config_dir.glob('*.yaml'))
    ]
    rows = []
    for configuration in configurations:
        environment = configuration.data.get('environment')
        condition = configuration.data.get('condition_id')
        run_dir = make_run_path(
            run_root, study.study_id, configuration.config_id,
            configuration.slug, environment, 0, run_attempt=run_attempt,
        )
        metadata_path = run_dir / 'runtime_metadata.json'
        metadata = {}
        if metadata_path.is_file():
            with metadata_path.open() as file:
                metadata = json.load(file)
        curve = _curve_summary(_read_eval(run_dir / 'eval.csv'))
        rows.append({
            'study_id': study.study_id,
            'config_id': configuration.config_id,
            'environment': environment,
            'condition_id': condition,
            'seed': 0,
            'run_attempt': run_attempt,
            'run_dir': str(run_dir),
            'run_status': metadata.get('status', 'missing'),
            **curve,
            **_architecture(metadata),
        })
    return rows


def _number(value):
    return '' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)


def _write_csv(rows, path):
    fields = [
        'study_id', 'config_id', 'environment', 'condition_id', 'seed',
        'run_attempt', 'run_status', 'status', 'num_eval_points',
        'final_success', 'best_success', 'best_step', 'last3_mean',
        'normalized_eval_auc', 'total_trainable_params', 'total_dense_macs',
        'actor_depth', 'value_depth', 'critic_depth', 'run_dir',
    ]
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _write_json(rows, path):
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')


def _write_plots(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plots skipped: matplotlib is unavailable.')
        return
    environments = sorted({row['environment'] for row in rows})
    colors = {'B000': 'black', 'S001': 'tab:blue', 'S002': 'tab:orange', 'S004': 'tab:red'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, environment in zip(axes.flat, environments):
        for condition in CONDITIONS:
            row = next(
                (item for item in rows
                 if item['environment'] == environment and item['condition_id'] == condition),
                None,
            )
            if row is None:
                continue
            points = _read_eval(Path(row['run_dir']) / 'eval.csv')
            if points:
                axis.plot(
                    [step / 1_000_000 for step, _ in points], [value for _, value in points],
                    marker='o', label=condition, color=colors[condition],
                )
        axis.set_title(environment)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.supxlabel('Training step (M)')
    fig.supylabel('Overall success')
    fig.tight_layout()
    fig.savefig(output_dir / 'learning_curves.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for axis, metric, title in (
        (axes[0], 'final_success', 'Final success at 1M'),
        (axes[1], 'normalized_eval_auc', 'Normalized evaluation AUC'),
    ):
        for condition in CONDITIONS:
            values = [
                row.get(metric) if row['condition_id'] == condition else None
                for row in rows
            ]
            values = [value for value in values if value is not None]
            if values:
                axis.scatter([condition] * len(values), values, label=condition, color=colors[condition])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel('Metric')
    fig.tight_layout()
    fig.savefig(output_dir / 'endpoint_metrics.png', dpi=180)
    plt.close(fig)
    print(f'Plots written to {output_dir.resolve()}')


def _markdown(rows):
    complete = sum(row['status'] == 'complete' and row['run_status'] == 'completed' for row in rows)
    lines = [
        '# M16A 结果汇总（自动生成）',
        '',
        f'- 运行单元：{len(rows)}；完整单元：{complete}/{len(rows)}。',
        '- 主指标：`evaluation/overall_success`；主终点：`final_success` at 1M。',
        '- `normalized_eval_auc` 按 100k–1M 的梯形积分除以 900k 计算；缺少完整 10 个评估点时留空。',
        '- 所有跨环境比较仅作复杂度相关的描述；B000 与同环境结构化条件的差值才是受控对比。',
        '',
        '| 环境 | 条件 | 状态 | last@1M | best | best step | last-3 mean | norm AUC | params | Dense MACs | depth A/V/C |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        depth = '/'.join(_number(row.get(key)) for key in ('actor_depth', 'value_depth', 'critic_depth'))
        lines.append(
            f'| {row["environment"]} | {row["condition_id"]} | '
            f'{row["run_status"]}/{row["status"]} | {_number(row["final_success"])} | '
            f'{_number(row["best_success"])} | {_number(row["best_step"])} | '
            f'{_number(row["last3_mean"])} | {_number(row["normalized_eval_auc"])} | '
            f'{_number(row["total_trainable_params"])} | {_number(row["total_dense_macs"])} | {depth} |'
        )
    lines.extend(['', '## 同环境相对 B000 的差值', '', '| 环境 | 条件 | Δ last@1M | Δ best | Δ norm AUC |', '|---|---:|---:|---:|---:|'])
    by_cell = {(row['environment'], row['condition_id']): row for row in rows}
    for environment in sorted({row['environment'] for row in rows}):
        baseline = by_cell.get((environment, 'B000'), {})
        for condition in CONDITIONS[1:]:
            row = by_cell.get((environment, condition), {})
            deltas = []
            for field in ('final_success', 'best_success', 'normalized_eval_auc'):
                left, right = row.get(field), baseline.get(field)
                deltas.append(None if left is None or right is None else left - right)
            lines.append(f'| {environment} | {condition} | ' + ' | '.join(_number(value) for value in deltas) + ' |')
    if complete < len(rows):
        lines.extend(['', '> 当前结果尚不完整；不得据此作 M16A 正式结论。'])
    else:
        lines.extend(['', '> 表格为描述性汇总；正式结论仍需结合预注册假设、任务级曲线和统计不确定性。'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M16A_puzzle_mixer_depth_scaling/study.yaml')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--run-attempt', type=int, default=0)
    parser.add_argument('--output-dir', default='docs/milestones/M16A_results')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.study, args.run_root, args.run_attempt)
    _write_csv(rows, output_dir / 'results_long.csv')
    _write_json(rows, output_dir / 'results_summary.json')
    (output_dir / 'results_summary.md').write_text(_markdown(rows))
    if args.plots:
        _write_plots(rows, output_dir)
    print(f'Wrote {len(rows)} rows to {output_dir.resolve()}')
    print(f'complete={sum(row["status"] == "complete" and row["run_status"] == "completed" for row in rows)}/{len(rows)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
