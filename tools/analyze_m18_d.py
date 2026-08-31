"""Aggregate M18-D diagnostics into non-causal tables, plots, and a report.

This is a post-hoc reader: it never launches experiments, restores source
checkpoints, writes under a source run, or overwrites an existing report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DIAGNOSTIC_ID_D1 = 'M18-D1'
DIAGNOSTIC_ID_TRACE = 'M18-D234'
K_VALUES = (1, 2, 4, 8)
DEFAULT_DIAGNOSTICS_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'

TRACE_FIELDS = (
    'record_type', 'K_train', 'K_actor_test', 'checkpoint_role', 'slot',
    'ensemble_member', 'iteration_k', 'is_depth_extrapolation', 'metric',
    'count', 'mean', 'std', 'median', 'p10', 'p25', 'p75', 'p90',
    'source_checkpoint_step', 'source_checkpoint_sha256', 'source_path',
)

PLOT_SPECS = (
    ('D2_state_rms.png', 'state_rms', None, 'D2 state RMS'),
    ('D3_relative_update.png', 'relative_update_from_previous', None, 'D2 relative update ratio'),
    ('D4_state_cosine.png', 'state_cosine_from_previous', None, 'D2 consecutive-state cosine'),
    ('D5_token_variance.png', 'token_variance', None, 'D2 token variance'),
    ('D6_pairwise_token_cosine.png', 'pairwise_token_cosine', None, 'D2 pairwise token cosine'),
    ('D7_action_delta.png', 'action_delta_from_previous', ('actor',), 'D3 action delta'),
    ('D8_dataset_action_mse.png', 'dataset_action_mse', ('actor',), 'D3 dataset-action MSE'),
    ('D9_action_saturation.png', 'action_mean_saturation_fraction', ('actor',), 'D3 mean saturation'),
    ('D10_qmin.png', 'qmin', ('actor',), 'D4 Qmin intermediate actor action'),
    ('D11_qgap_vs_dataset.png', 'qgap_vs_dataset_action', ('actor',), 'D4 Qmin gap versus dataset'),
)


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path):
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return value


def _read_csv(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])


def _m18d_root(diagnostics_root):
    return Path(diagnostics_root) / 'M18D'


def _collect_cross_k(diagnostics_root):
    """Use D1 aggregate if present; otherwise reconstruct it from job summaries."""

    m18d_root = _m18d_root(diagnostics_root)
    aggregate_path = m18d_root / 'summary' / 'checkpoint_best' / 'm18d_cross_k_success.csv'
    source_paths = []
    rows = []
    if aggregate_path.is_file():
        source_paths.append(str(aggregate_path))
        for raw in _read_csv(aggregate_path):
            train_k = _int(raw.get('K_train'))
            test_k = _int(raw.get('K_actor_test'))
            success = _float(raw.get('overall_success'))
            if train_k is None or test_k is None or success is None:
                continue
            rows.append({
                'K_train': train_k,
                'K_actor_test': test_k,
                'overall_success': success,
                'checkpoint_role': raw.get('checkpoint_role', 'best') or 'best',
                'checkpoint_step': _int(raw.get('checkpoint_step')),
                'checkpoint_sha256': raw.get('checkpoint_sha256'),
                'source_path': raw.get('summary_path') or str(aggregate_path),
            })
    else:
        cross_root = m18d_root / 'cross_k' / 'checkpoint_best'
        paths = sorted(cross_root.rglob('summary.json')) if cross_root.is_dir() else ()
        for path in paths:
            try:
                summary = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if summary.get('status') != 'completed' or summary.get('diagnostic_id') != DIAGNOSTIC_ID_D1:
                continue
            train_k = _int(summary.get('K_train'))
            test_k = _int(summary.get('K_actor_test'))
            success = _float(summary.get('overall_success'))
            if train_k is None or test_k is None or success is None:
                continue
            rows.append({
                'K_train': train_k,
                'K_actor_test': test_k,
                'overall_success': success,
                'checkpoint_role': summary.get('checkpoint_role', 'best'),
                'checkpoint_step': _int(summary.get('checkpoint_step')),
                'checkpoint_sha256': summary.get('checkpoint_sha256'),
                'source_path': str(path),
            })
            source_paths.append(str(path))
    unique = {}
    for row in rows:
        key = (row['K_train'], row['K_actor_test'])
        if row['checkpoint_role'] != 'best':
            raise ValueError(f'D1 artifact is not checkpoint=best: {row["source_path"]}')
        if key in unique:
            raise ValueError(f'Duplicate completed D1 cell for K_train/K_actor_test={key!r}')
        unique[key] = row
    return [unique[key] for key in sorted(unique)], source_paths


def _collect_trace_rows(diagnostics_root):
    trace_root = _m18d_root(diagnostics_root) / 'trace' / 'checkpoint_best'
    rows = []
    source_paths = []
    if not trace_root.is_dir():
        return rows, source_paths
    for path in sorted(trace_root.rglob('trace_summary.csv')):
        metadata_path = path.with_name('m18d_metadata.json')
        try:
            metadata = _read_json(metadata_path)
            raw_rows = _read_csv(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if metadata.get('status') != 'completed' or metadata.get('diagnostic_id') != DIAGNOSTIC_ID_TRACE:
            continue
        if metadata.get('source_checkpoint_role') != 'best':
            raise ValueError(f'Trace artifact is not checkpoint=best: {metadata_path}')
        source_paths.extend((str(path), str(metadata_path)))
        for raw in raw_rows:
            train_k = _int(raw.get('K_train'))
            iteration_k = _int(raw.get('iteration_k'))
            metric = raw.get('metric')
            if train_k is None or iteration_k is None or not metric:
                continue
            row = {
                'record_type': 'trace',
                'K_train': train_k,
                'K_actor_test': '',
                'checkpoint_role': raw.get('checkpoint_role') or 'best',
                'slot': raw.get('slot') or '',
                'ensemble_member': raw.get('ensemble_member') or '',
                'iteration_k': iteration_k,
                'is_depth_extrapolation': str(raw.get('is_depth_extrapolation')).lower() == 'true',
                'metric': metric,
                'count': _int(raw.get('count')),
                'mean': _float(raw.get('mean')),
                'std': _float(raw.get('std')),
                'median': _float(raw.get('median')),
                'p10': _float(raw.get('p10')),
                'p25': _float(raw.get('p25')),
                'p75': _float(raw.get('p75')),
                'p90': _float(raw.get('p90')),
                'source_checkpoint_step': _int(metadata.get('source_checkpoint_step')),
                'source_checkpoint_sha256': metadata.get('source_checkpoint_sha256'),
                'source_path': str(path),
            }
            if row['checkpoint_role'] != 'best':
                raise ValueError(f'Trace summary row is not checkpoint=best: {path}')
            rows.append(row)
    rows.sort(key=lambda row: (
        row['K_train'], row['slot'], str(row['ensemble_member']),
        row['metric'], row['iteration_k'],
    ))
    return rows, source_paths


def _delta_rows(cross_rows):
    lookup = {(row['K_train'], row['K_actor_test']): row['overall_success'] for row in cross_rows}
    return [
        row | {
            'diagonal_success': lookup.get((row['K_train'], row['K_train'])),
            'delta_vs_trained_actor_depth': (
                None
                if lookup.get((row['K_train'], row['K_train'])) is None
                else row['overall_success'] - lookup[(row['K_train'], row['K_train'])]
            ),
        }
        for row in cross_rows
    ]


def _validate_shared_provenance(cross_rows, trace_rows):
    """Reject mixing D1/D234 artifacts from different best checkpoints per K."""

    d1_hashes = defaultdict(set)
    trace_hashes = defaultdict(set)
    for row in cross_rows:
        value = row.get('checkpoint_sha256')
        if value:
            d1_hashes[row['K_train']].add(str(value))
    for row in trace_rows:
        value = row.get('source_checkpoint_sha256')
        if value:
            trace_hashes[row['K_train']].add(str(value))
    for label, mapping in (('D1', d1_hashes), ('D2/D3/D4', trace_hashes)):
        for train_k, values in mapping.items():
            if len(values) > 1:
                raise ValueError(f'{label} mixes multiple checkpoint SHA256 values for K_train={train_k}')
    for train_k in sorted(set(d1_hashes) & set(trace_hashes)):
        if d1_hashes[train_k] != trace_hashes[train_k]:
            raise ValueError(
                f'D1 and D2/D3/D4 do not share checkpoint provenance for K_train={train_k}: '
                f'{d1_hashes[train_k]!r} vs {trace_hashes[train_k]!r}'
            )


def _series(trace_rows, metric, slots=None):
    groups = defaultdict(list)
    for row in trace_rows:
        if row['metric'] != metric or row['mean'] is None:
            continue
        if slots is not None and row['slot'] not in slots:
            continue
        groups[(row['K_train'], row['slot'], str(row['ensemble_member']))].append(
            (row['iteration_k'], row['mean'])
        )
    return {key: sorted(values) for key, values in groups.items() if values}


def _trend(trace_rows, metric, train_k, slot='actor'):
    groups = _series(trace_rows, metric, slots=(slot,))
    points = []
    for (candidate_k, _, _), values in groups.items():
        if candidate_k == train_k:
            points.extend(values)
    by_iteration = defaultdict(list)
    for iteration_k, value in points:
        by_iteration[iteration_k].append(value)
    points = sorted((iteration_k, float(np.mean(values))) for iteration_k, values in by_iteration.items())
    if len(points) < 2:
        return None
    first_k, first_value = points[0]
    last_k, last_value = points[-1]
    return {
        'first_iteration_k': first_k,
        'first_value': first_value,
        'last_iteration_k': last_k,
        'last_value': last_value,
        'delta': last_value - first_value,
    }


def _plot_heatmap(cross_rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.full((len(K_VALUES), len(K_VALUES)), np.nan, dtype=np.float64)
    lookup = {(row['K_train'], row['K_actor_test']): row['overall_success'] for row in cross_rows}
    for row_index, train_k in enumerate(K_VALUES):
        for column_index, test_k in enumerate(K_VALUES):
            if (train_k, test_k) in lookup:
                values[row_index, column_index] = lookup[(train_k, test_k)]
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(values, aspect='auto')
    axis.set_xticks(range(len(K_VALUES)), K_VALUES)
    axis.set_yticks(range(len(K_VALUES)), K_VALUES)
    axis.set_xlabel('K_actor_test (actor-only inference depth)')
    axis.set_ylabel('K_train (joint actor/value/critic training depth)')
    axis.set_title('D1 actor success heatmap | checkpoint=best | slot=actor')
    for row_index in range(len(K_VALUES)):
        for column_index in range(len(K_VALUES)):
            value = values[row_index, column_index]
            if math.isfinite(value):
                axis.text(column_index, row_index, f'{value:.3f}', ha='center', va='center')
    figure.colorbar(image, ax=axis, label='overall success')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_metric(trace_rows, metric, slots, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    groups = _series(trace_rows, metric, slots=slots)
    for (train_k, slot, member), points in sorted(groups.items()):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        member_text = '' if not member else f', member={member}'
        axis.plot(xs, ys, marker='o', label=f'K_train={train_k}, slot={slot}{member_text}')
    if groups:
        axis.legend()
    else:
        axis.text(0.5, 0.5, 'No completed trace data', ha='center', va='center')
    axis.set_xlabel('recurrent iteration k')
    axis.set_ylabel(metric)
    axis.set_title(f'{title} | checkpoint=best | K_train/slot in legend')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _status(signs):
    """Map directional observations to explicitly non-causal labels."""

    signs = [sign for sign in signs if sign is not None]
    if not signs:
        return 'insufficient evidence'
    if all(sign > 0 for sign in signs):
        return 'consistent'
    if all(sign <= 0 for sign in signs):
        return 'not observed'
    return 'mixed'


def _qmin_at(trace_rows, train_k, iteration_k):
    values = [
        row['mean']
        for row in trace_rows
        if row['metric'] == 'qmin'
        and row['slot'] == 'actor'
        and row['K_train'] == train_k
        and row['iteration_k'] == iteration_k
        and row['mean'] is not None
    ]
    return None if not values else float(np.mean(values))


def _decision_table(cross_rows, trace_rows):
    """Produce a threshold-light mechanism table, never causal TRUE/FALSE."""

    cross = {(row['K_train'], row['K_actor_test']): row['overall_success'] for row in cross_rows}
    rows = []

    signs, observations = [], []
    for train_k in K_VALUES:
        diagonal = cross.get((train_k, train_k))
        shallow = [
            (test_k, success)
            for (candidate_k, test_k), success in cross.items()
            if candidate_k == train_k and test_k < train_k
        ]
        if diagonal is None or not shallow:
            continue
        best_k, best_success = max(shallow, key=lambda item: item[1])
        delta = best_success - diagonal
        signs.append(1 if delta > 0 else -1)
        observations.append(
            f'K{train_k}: best shallower K={best_k}, success={best_success:.3f}; '
            f'diagonal={diagonal:.3f}; delta={delta:+.3f}'
        )
    rows.append({
        'hypothesis': 'H_overprocessing',
        'status': _status(signs),
        'evidence': 'D1 shallow actor depth success relative to the source diagonal',
        'observation': '; '.join(observations) if observations else 'No comparable shallow D1 cells.',
        'supports': 'A descriptive actor-depth response where extra inference steps can coincide with lower success.',
        'does_not_prove': 'That recurrence is the sole cause or that the response generalizes beyond these checkpoints.',
    })

    signs, observations = [], []
    for train_k in K_VALUES:
        rms = _trend(trace_rows, 'state_rms', train_k)
        relative = _trend(trace_rows, 'relative_update_from_previous', train_k)
        if rms is None or relative is None:
            continue
        signs.append(1 if rms['delta'] > 0 and relative['delta'] > 0 else -1)
        observations.append(
            f'K{train_k}: delta RMS={rms["delta"]:+.4g}; '
            f'delta relative-update={relative["delta"]:+.4g}'
        )
    rows.append({
        'hypothesis': 'H_state_instability',
        'status': _status(signs),
        'evidence': 'Actor D2 state RMS and relative-update trajectories',
        'observation': '; '.join(observations) if observations else 'Insufficient actor D2 trajectories.',
        'supports': 'A descriptive indication of increasing representation scale and update magnitude.',
        'does_not_prove': 'Numerical divergence or a causal performance mechanism.',
    })

    signs, observations = [], []
    for train_k in K_VALUES:
        variance = _trend(trace_rows, 'token_variance', train_k)
        cosine = _trend(trace_rows, 'pairwise_token_cosine', train_k)
        if variance is None or cosine is None:
            continue
        signs.append(1 if variance['delta'] < 0 and cosine['delta'] > 0 else -1)
        observations.append(
            f'K{train_k}: delta token-variance={variance["delta"]:+.4g}; '
            f'delta pairwise-cos={cosine["delta"]:+.4g}'
        )
    rows.append({
        'hypothesis': 'H_token_oversmoothing',
        'status': _status(signs),
        'evidence': 'Actor D2 token variance jointly with off-diagonal token cosine',
        'observation': '; '.join(observations) if observations else 'Insufficient actor token trajectories.',
        'supports': 'The joint descriptive signature expected from token homogenization.',
        'does_not_prove': 'That token collapse caused a rollout outcome.',
    })

    signs, observations = [], []
    for train_k in K_VALUES:
        action_delta = _trend(trace_rows, 'action_delta_from_previous', train_k)
        drift = _trend(trace_rows, 'action_drift_from_k1', train_k)
        if action_delta is None or drift is None:
            continue
        signs.append(
            1
            if action_delta['delta'] > 0 and drift['last_value'] > drift['first_value']
            else -1
        )
        observations.append(
            f'K{train_k}: delta action-delta={action_delta["delta"]:+.4g}; '
            f'delta drift={drift["delta"]:+.4g}'
        )
    rows.append({
        'hypothesis': 'H_action_instability',
        'status': _status(signs),
        'evidence': 'D3 deterministic action delta and drift from a1',
        'observation': '; '.join(observations) if observations else 'Insufficient D3 action trajectories.',
        'supports': 'A descriptive pattern of increasingly changing deterministic actions.',
        'does_not_prove': 'That any action movement is harmful in the environment.',
    })

    signs, observations = [], []
    for train_k in K_VALUES:
        pairs = []
        for test_k in K_VALUES:
            success = cross.get((train_k, test_k))
            qmin = _qmin_at(trace_rows, train_k, test_k)
            if success is not None and qmin is not None:
                pairs.append((success, qmin))
        if len(pairs) < 2:
            continue
        successes, qmins = np.asarray(pairs).T
        correlation = float(np.corrcoef(successes, qmins)[0, 1])
        if not math.isfinite(correlation):
            continue
        signs.append(1 if correlation < 0 else -1)
        observations.append(f'K{train_k}: corr(success, Qmin)={correlation:+.3f}')
    rows.append({
        'hypothesis': 'H_critic_misalignment',
        'status': _status(signs),
        'evidence': 'Matched D1 success versus D4 fixed-source-critic Qmin across actor depths',
        'observation': '; '.join(observations) if observations else 'No sufficient matched D1/D4 depths.',
        'supports': 'A descriptive inverse association if more successful actor depths receive lower Qmin.',
        'does_not_prove': 'Critic error because Q scale and controlled offline states are not rollout returns.',
    })

    k8_values = [success for (train_k, _), success in cross.items() if train_k == 8]
    k4_diagonal = cross.get((4, 4))
    if not k8_values or k4_diagonal is None:
        training_status = 'insufficient evidence'
        training_observation = 'Need K8 depth-response cells and the K4 diagonal.'
    else:
        k8_best = max(k8_values)
        training_status = 'consistent' if k8_best <= k4_diagonal else 'not observed'
        training_observation = (
            f'K8 best tested actor-depth success={k8_best:.3f}; '
            f'K4 diagonal success={k4_diagonal:.3f}.'
        )
    rows.append({
        'hypothesis': 'H_training_failure',
        'status': training_status,
        'evidence': 'Best D1 K8 actor-depth success relative to the available K4 diagonal',
        'observation': training_observation,
        'supports': 'Whether the present depth probe leaves open a broad K8 training-time failure interpretation.',
        'does_not_prove': 'A training-cause diagnosis, especially with one training seed and best checkpoints.',
    })
    return rows


def _number(value):
    return '' if value is None else f'{float(value):.6g}'


def _render_report(cross_rows, trace_rows, delta_rows, decisions, input_paths, figure_names):
    lines = [
        '# M18-D 诊断汇总（自动生成）',
        '',
        '本报告只汇总已完成的 checkpoint=best M18-D1/D2/D3/D4 artifact。所有判断均是描述性证据，不构成因果诊断。',
        '',
        '## 输入与范围',
        '',
        f'- D1 completed cells: {len(cross_rows)}。',
        f'- D2/D3/D4 aggregate rows: {len(trace_rows)}。',
        f'- 输入 artifact 数: {len(input_paths)}。',
        '- D1 改变的仅是 actor inference depth；D4 中 critic 始终使用 source K_train。',
        '- fixed offline batch 是受控 representation/action comparison，不是 rollout-state distribution。',
        '',
        '## D1：actor inference-depth response',
        '',
        '| K_train | K_actor_test | Overall success | Diagonal success | Delta vs diagonal |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in delta_rows:
        lines.append(
            f'| {row["K_train"]} | {row["K_actor_test"]} | {_number(row["overall_success"])} | '
            f'{_number(row["diagonal_success"])} | {_number(row["delta_vs_trained_actor_depth"])} |'
        )
    lines.extend([
        '',
        '## D2/D3/D4 coverage',
        '',
        '| K_train | Slot | Metrics | Iteration range | Source checkpoint step |',
        '|---:|---|---:|---|---:|',
    ])
    coverage = defaultdict(list)
    for row in trace_rows:
        coverage[(row['K_train'], row['slot'], row['source_checkpoint_step'])].append(row)
    for (train_k, slot, step), values in sorted(coverage.items()):
        metrics = {row['metric'] for row in values}
        iterations = sorted({row['iteration_k'] for row in values})
        iteration_range = '' if not iterations else f'{iterations[0]}..{iterations[-1]}'
        lines.append(f'| {train_k} | {slot} | {len(metrics)} | {iteration_range} | {step or ""} |')
    lines.extend([
        '',
        '## Mechanism decision table（非因果）',
        '',
        '| Hypothesis | Status | Evidence | Observation | Supports | Does not prove |',
        '|---|---|---|---|---|---|',
    ])
    for row in decisions:
        lines.append(
            f'| {row["hypothesis"]} | {row["status"]} | {row["evidence"]} | '
            f'{row["observation"]} | {row["supports"]} | {row["does_not_prove"]} |'
        )
    lines.extend([
        '',
        '## Interpretation boundaries',
        '',
        '- state RMS 增长只表示 representation scale 改变；需结合 relative update、action drift、success 与 Q ranking 解释。',
        '- pairwise token cosine 接近 1 必须与 token variance 同时下降，才是较强 oversmoothing evidence。',
        '- Qmin 是 source critic 对 controlled offline states/action 的输出，不等同环境 return。',
        '- 此 Study 的训练 seed 为 0；没有显著性检验或跨 seed 泛化结论。',
        '',
        '## Generated figures',
        '',
    ])
    lines.extend(f'- {name}' for name in figure_names)
    lines.append('')
    return '\n'.join(lines)


def analyze(diagnostics_root, output_dir, *, dry_run=False):
    diagnostics_root = Path(diagnostics_root)
    output_dir = Path(output_dir)
    cross_rows, cross_paths = _collect_cross_k(diagnostics_root)
    trace_rows, trace_paths = _collect_trace_rows(diagnostics_root)
    if not cross_rows:
        raise ValueError('No completed M18-D1 checkpoint=best rows found')
    if not trace_rows:
        raise ValueError('No completed M18-D2/D3/D4 checkpoint=best rows found')
    if output_dir.exists():
        raise FileExistsError(f'M18-D report directory exists; refusing overwrite: {output_dir}')
    _validate_shared_provenance(cross_rows, trace_rows)
    delta_rows = _delta_rows(cross_rows)
    decisions = _decision_table(cross_rows, trace_rows)
    input_paths = sorted(set(cross_paths + trace_paths))
    if dry_run:
        return {
            'status': 'dry-run',
            'output_dir': str(output_dir),
            'cross_k_rows': len(cross_rows),
            'trace_rows': len(trace_rows),
            'input_paths': input_paths,
            'decisions': decisions,
        }

    output_dir.mkdir(parents=True)
    figure_names = ['D1_cross_k_success_heatmap.png']
    _plot_heatmap(cross_rows, output_dir / figure_names[0])
    for filename, metric, slots, title in PLOT_SPECS:
        _plot_metric(trace_rows, metric, slots, title, output_dir / filename)
        figure_names.append(filename)
    _write_csv(output_dir / 'm18d_summary.csv', trace_rows, TRACE_FIELDS)
    _write_csv(
        output_dir / 'm18d_cross_k_success.csv',
        cross_rows,
        ('K_train', 'K_actor_test', 'overall_success', 'checkpoint_role', 'checkpoint_step', 'checkpoint_sha256', 'source_path'),
    )
    _write_csv(
        output_dir / 'm18d_cross_k_delta.csv',
        delta_rows,
        (
            'K_train', 'K_actor_test', 'overall_success', 'diagonal_success',
            'delta_vs_trained_actor_depth', 'checkpoint_role', 'checkpoint_step',
            'checkpoint_sha256', 'source_path',
        ),
    )
    _write_json(output_dir / 'm18d_summary.json', {
        'diagnostic_id': 'M18-D',
        'checkpoint_role': 'best',
        'input_paths': input_paths,
        'cross_k_rows': cross_rows,
        'cross_k_delta_rows': delta_rows,
        'trace_rows': trace_rows,
        'mechanism_decision_table': decisions,
        'figures': figure_names,
        'non_causal_interpretation': True,
    })
    (output_dir / 'M18D_report.md').write_text(
        _render_report(cross_rows, trace_rows, delta_rows, decisions, input_paths, figure_names) + '\n'
    )
    return {
        'status': 'completed',
        'output_dir': str(output_dir),
        'cross_k_rows': len(cross_rows),
        'trace_rows': len(trace_rows),
        'figures': figure_names,
        'decisions': decisions,
    }


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-root', default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument('--checkpoint', default='best')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='New report directory; default is diagnostics-root/M18D/reports/checkpoint_best.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.checkpoint != 'best':
        parser.error('M18-D analyzer currently accepts --checkpoint best only')
    if args.dry_run == args.execute:
        parser.error('Exactly one of --dry-run or --execute is required')
    if args.output_dir is None:
        args.output_dir = _m18d_root(args.diagnostics_root) / 'reports' / 'checkpoint_best'
    return args


def main(argv=None):
    args = _args(argv)
    try:
        result = analyze(args.diagnostics_root, args.output_dir, dry_run=bool(args.dry_run))
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f'M18-D analyzer: FAIL: {error}')
        return 2
    print(
        f'M18-D analyzer {result["status"]}: D1 rows={result["cross_k_rows"]} '
        f'trace rows={result["trace_rows"]} output={result["output_dir"]}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
