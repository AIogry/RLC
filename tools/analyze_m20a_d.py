"""Post-hoc analysis for M20A-D relation-utilization artifacts.

This reader consumes only the completed ``diagnostic_summary.json`` and
``per_sample_metrics.npz`` files produced by ``m20a_relation_utilization``.
It never restores a checkpoint, constructs an agent, or performs a forward
pass.  A report directory is created once and is never overwritten.

The report deliberately uses the conservative vocabulary required by M20A:
``consistent``, ``mixed``, ``not observed``, and ``insufficient evidence``.
Those labels describe the observed diagnostic evidence; they are not causal
or performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DIAGNOSTIC_ID = 'M20AD'
DIAGNOSTIC_SCHEMA = 'm20a_relation_utilization_v1'
STUDY_ID = 'M20A'
DEFAULT_CONFIG_ID = 'M20A-CUBE-C-Q'
# Backwards-compatible alias for callers that used the original Q reader.
CONFIG_ID = DEFAULT_CONFIG_ID
RELATION_NAMES = ('current_support', 'goal_support', 'goal_conflict')
INPUT_GROUPS = ('actor', 'value', 'critic_value', 'critic_actor')
CONCLUSION_STATUSES = frozenset({
    'consistent', 'mixed', 'not observed', 'insufficient evidence',
})
NUMERICAL_TOLERANCE = 1e-7
DEFAULT_DIAGNOSTICS_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'


class M20ADAnalysisError(ValueError):
    """Raised for incomplete, mixed, or malformed M20A-D artifacts."""


def _read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise M20ADAnalysisError(f'Cannot read JSON artifact {path}: {error}') from error
    if not isinstance(value, dict):
        raise M20ADAnalysisError(f'Expected JSON object in {path}')
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, '') for field in fields} for row in rows)


def _finite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _stats(values):
    values = _finite(values)
    if values.size == 0:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'median': None,
            'p10': None,
            'p25': None,
            'p75': None,
            'p90': None,
            'max': None,
            'fraction_gt_numerical_tolerance': None,
        }
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'max': float(np.max(values)),
        'fraction_gt_numerical_tolerance': float(
            np.mean(np.abs(values) > NUMERICAL_TOLERANCE)
        ),
    }


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _campaign_id(source_config):
    if source_config == 'M20A-CUBE-C-M':
        return 'M20A-D-M'
    if source_config == DEFAULT_CONFIG_ID:
        return 'M20A-D-Q'
    return f'M20A-D-{source_config}'


def _artifact_root_candidates(diagnostics_root, source_config=DEFAULT_CONFIG_ID):
    root = Path(diagnostics_root).resolve() / DIAGNOSTIC_ID
    if source_config == DEFAULT_CONFIG_ID:
        # The first layout is the pre-prompt5 Q layout.  The second is the
        # generalized layout emitted by the current tool for a future Q run.
        return (
            root / 'relation_utilization' / source_config,
            root / 'relation_utilization',
        )
    return (root / 'campaigns' / source_config / 'relation_utilization',)


def _artifact_root(diagnostics_root, source_config=DEFAULT_CONFIG_ID):
    candidates = _artifact_root_candidates(diagnostics_root, source_config)
    populated = [
        candidate for candidate in candidates
        if candidate.is_dir() and any(candidate.glob('checkpoint_*/diagnostic_summary.json'))
    ]
    if len(populated) > 1:
        raise M20ADAnalysisError(
            f'Artifacts for {source_config} appear in multiple namespaces: '
            f'{[str(path) for path in populated]!r}'
        )
    return populated[0] if populated else candidates[0]


def _load_npz(path):
    path = Path(path)
    if not path.is_file():
        raise M20ADAnalysisError(f'Missing per-sample artifact: {path}')
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError) as error:
        raise M20ADAnalysisError(f'Cannot read per-sample artifact {path}: {error}') from error


def _identity(summary, metadata, summary_path, *, source_config):
    source_run = metadata.get('source_run_path')
    config_fingerprint = metadata.get('source_resolved_config_fingerprint')
    batch_fingerprint = (
        summary.get('fixed_batch_fingerprint_sha256')
        or metadata.get('fixed_batch_fingerprint_sha256')
    )
    checkpoint_step = summary.get('source_checkpoint_step')
    checkpoint_sha = summary.get('source_checkpoint_sha256')
    observed_config = summary.get('source_config') or metadata.get('source_config')
    source_readout = summary.get('source_readout') or metadata.get('source_readout')
    if source_run is None or config_fingerprint is None or batch_fingerprint is None:
        raise M20ADAnalysisError(f'Incomplete provenance in {summary_path}')
    if checkpoint_step is None or checkpoint_sha is None:
        raise M20ADAnalysisError(f'Incomplete checkpoint identity in {summary_path}')
    if observed_config is not None and observed_config != source_config:
        raise M20ADAnalysisError(
            f'Source config mismatch in {summary_path}: '
            f'expected={source_config!r}, observed={observed_config!r}'
        )
    try:
        checkpoint_step = int(checkpoint_step)
    except (TypeError, ValueError) as error:
        raise M20ADAnalysisError(f'Invalid checkpoint step in {summary_path}') from error
    return {
        'source_run_path': str(source_run),
        'source_resolved_config_fingerprint': str(config_fingerprint),
        'fixed_batch_fingerprint_sha256': str(batch_fingerprint),
        'source_checkpoint_step': checkpoint_step,
        'source_checkpoint_sha256': str(checkpoint_sha),
        'source_config': source_config,
        'source_readout': source_readout,
    }


def _completed_artifacts(diagnostics_root, source_config=DEFAULT_CONFIG_ID):
    root = _artifact_root(diagnostics_root, source_config)
    expected_diagnostic_ids = {DIAGNOSTIC_ID, _campaign_id(source_config)}
    paths = sorted(root.glob('checkpoint_*/diagnostic_summary.json')) if root.is_dir() else []
    artifacts = []
    for summary_path in paths:
        summary = _read_json(summary_path)
        metadata_path = summary_path.parent / 'm20ad_metadata.json'
        metadata = _read_json(metadata_path)
        if summary.get('diagnostic_id') not in expected_diagnostic_ids:
            raise M20ADAnalysisError(f'Diagnostic ID mismatch in {summary_path}')
        if summary.get('status') != 'completed' or metadata.get('status') != 'completed':
            continue
        if metadata.get('diagnostic_id') not in expected_diagnostic_ids:
            raise M20ADAnalysisError(f'Metadata diagnostic ID mismatch in {metadata_path}')
        identity = _identity(
            summary, metadata, summary_path, source_config=source_config,
        )
        if source_config == 'M20A-CUBE-C-M':
            attention_status = (
                summary.get('attention_status')
                or metadata.get('attention_status')
            )
            if attention_status != 'not_applicable_mean_readout':
                raise M20ADAnalysisError(
                    f'Mean-readout attention status is not explicit in {summary_path}: '
                    f'{attention_status!r}'
                )
        artifact_ref = summary.get('artifact_paths', {}).get('per_sample_metrics')
        per_sample_path = Path(artifact_ref) if artifact_ref else summary_path.parent / 'per_sample_metrics.npz'
        arrays = _load_npz(per_sample_path)
        if 'sample_id' not in arrays or arrays['sample_id'].ndim != 1:
            raise M20ADAnalysisError(f'Missing one-dimensional sample_id in {per_sample_path}')
        for required in ('actor_relation_counts', 'value_relation_counts'):
            if required not in arrays:
                raise M20ADAnalysisError(f'Missing {required} in {per_sample_path}')
        sample_count = arrays['sample_id'].shape[0]
        if any(array.ndim > 0 and array.shape[0] not in (sample_count, 2) for array in arrays.values()):
            raise M20ADAnalysisError(f'Unexpected leading dimension in {per_sample_path}')
        artifacts.append({
            'summary_path': str(summary_path),
            'metadata_path': str(metadata_path),
            'per_sample_path': str(per_sample_path),
            'summary': summary,
            'metadata': metadata,
            'identity': identity,
            'arrays': arrays,
        })
    if not artifacts:
        raise M20ADAnalysisError(
            f'No completed {DIAGNOSTIC_ID} artifacts found below {root}; '
            'post-hoc analysis is blocked'
        )

    identity_fields = (
        'source_run_path', 'source_resolved_config_fingerprint',
        'fixed_batch_fingerprint_sha256', 'source_config',
    )
    for field in identity_fields:
        values = {item['identity'][field] for item in artifacts}
        if len(values) != 1:
            raise M20ADAnalysisError(f'Artifacts mix {field}: {sorted(values)!r}')
    steps = [item['identity']['source_checkpoint_step'] for item in artifacts]
    if len(set(steps)) != len(steps):
        raise M20ADAnalysisError(f'Duplicate checkpoint artifact steps: {steps!r}')
    reference_sample_id = artifacts[0]['arrays']['sample_id']
    reference_sample_count = reference_sample_id.shape[0]
    for artifact in artifacts[1:]:
        sample_id = artifact['arrays']['sample_id']
        if sample_id.shape != reference_sample_id.shape or not np.array_equal(sample_id, reference_sample_id):
            raise M20ADAnalysisError(
                'Completed artifacts do not share the same per-sample sample_id axis'
            )
        if artifact['arrays']['actor_relation_counts'].shape[0] != reference_sample_count:
            raise M20ADAnalysisError(
                'Completed artifacts do not share the same fixed-batch sample count'
            )
    return sorted(artifacts, key=lambda item: item['identity']['source_checkpoint_step'])


def _group_mask(arrays, input_name, group_name):
    key = f'{input_name}_group_{group_name}'
    if key not in arrays:
        return None
    return np.asarray(arrays[key], dtype=bool)


def _status_from_values(values, *, mask=None):
    values = np.asarray(values)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    finite = _finite(values)
    if finite.size == 0:
        return 'not observed' if values.size == 0 else 'insufficient evidence'
    if np.any(np.abs(finite) > NUMERICAL_TOLERANCE):
        return 'consistent'
    return 'not observed'


def _combine_statuses(statuses):
    statuses = [status for status in statuses if status in CONCLUSION_STATUSES]
    if not statuses:
        return 'insufficient evidence'
    unique = set(statuses)
    if unique == {'consistent'}:
        return 'consistent'
    if unique == {'not observed'}:
        return 'not observed'
    if unique == {'insufficient evidence'}:
        return 'insufficient evidence'
    return 'mixed'


def _relation_coverage(artifacts):
    rows = []
    for artifact in artifacts:
        step = artifact['identity']['source_checkpoint_step']
        arrays = artifact['arrays']
        for input_name in INPUT_GROUPS:
            key = f'{input_name}_relation_counts'
            if key not in arrays:
                continue
            counts = np.asarray(arrays[key])
            if counts.ndim != 2 or counts.shape[1] != len(RELATION_NAMES):
                raise M20ADAnalysisError(f'Invalid relation count shape in {artifact["per_sample_path"]}: {counts.shape}')
            rows.append({
                'checkpoint_step': step,
                'input_group': input_name,
                'sample_count': int(counts.shape[0]),
                'any_relation_active_fraction': float(np.mean(np.any(counts > 0, axis=1))),
                'current_support_active_fraction': float(np.mean(counts[:, 0] > 0)),
                'goal_support_active_fraction': float(np.mean(counts[:, 1] > 0)),
                'goal_conflict_active_fraction': float(np.mean(counts[:, 2] > 0)),
                'total_edge_count': int(np.sum(counts)),
            })
    return rows


_METRIC_SPECS = (
    ('hidden', 'delta_rel'),
    ('hidden', 'relative_relation_update'),
    ('core', 'delta_mix'),
    ('core', 'relative_core_difference'),
    ('core', 'propagation_ratio'),
    ('readout', 'delta_readout'),
    ('readout', 'relative_readout_difference'),
    ('attention', 'attention_tv'),
    ('output', 'delta_mean'),
    ('output', 'delta_action'),
    ('output', 'delta_value'),
    ('output', 'relative_value_difference'),
    ('output', 'delta_q1'),
    ('output', 'delta_q2'),
    ('output', 'delta_qmin'),
)


def _metric_rows(artifacts):
    rows = []
    evidence = []
    for artifact in artifacts:
        step = artifact['identity']['source_checkpoint_step']
        arrays = artifact['arrays']
        for input_name in INPUT_GROUPS:
            observed_groups = ('inactive', 'any_relation_active', *RELATION_NAMES)
            for activity_group in observed_groups:
                mask = _group_mask(arrays, input_name, activity_group)
                if mask is None:
                    continue
                for category, metric in _METRIC_SPECS:
                    key = f'{input_name}_{metric}'
                    if key not in arrays:
                        continue
                    values = np.asarray(arrays[key])
                    if values.shape[0] != mask.shape[0]:
                        raise M20ADAnalysisError(
                            f'{key} and {activity_group} length mismatch in {artifact["per_sample_path"]}'
                        )
                    summary = _stats(values[mask])
                    rows.append({
                        'checkpoint_step': step,
                        'input_group': input_name,
                        'activity_group': activity_group,
                        'category': category,
                        'metric': metric,
                        **summary,
                    })
                    if activity_group == 'any_relation_active':
                        evidence.append({
                            'checkpoint_step': step,
                            'input_group': input_name,
                            'category': category,
                            'metric': metric,
                            'status': _status_from_values(values, mask=mask),
                            'count': summary['count'],
                        })

            for channel in RELATION_NAMES:
                for metric in ('channel_delta_rel', 'channel_delta_mix', 'channel_delta_readout', 'channel_output_effect'):
                    key = f'{input_name}_channel_{channel}_{metric}'
                    if key not in arrays:
                        continue
                    values = np.asarray(arrays[key])
                    relation_count_key = f'{input_name}_relation_counts'
                    if relation_count_key not in arrays:
                        raise M20ADAnalysisError(
                            f'Missing {relation_count_key} for channel artifact {key}'
                        )
                    relation_counts = np.asarray(arrays[relation_count_key])
                    channel_index = RELATION_NAMES.index(channel)
                    masks = {
                        'any_relation_active': _group_mask(
                            arrays, input_name, 'any_relation_active'
                        ),
                        f'{channel}_active': relation_counts[:, channel_index] > 0,
                        f'{channel}_inactive': relation_counts[:, channel_index] == 0,
                    }
                    if values.ndim == 0 or values.shape[0] != relation_counts.shape[0]:
                        raise M20ADAnalysisError(f'Invalid channel artifact shape for {key}')
                    for activity_group, mask in masks.items():
                        if mask is None:
                            continue
                        summary = _stats(values[mask])
                        rows.append({
                            'checkpoint_step': step,
                            'input_group': input_name,
                            'activity_group': activity_group,
                            'category': 'channel_ablation',
                            'metric': f'{channel}:{metric}',
                            **summary,
                        })
                        if activity_group == f'{channel}_active':
                            evidence.append({
                                'checkpoint_step': step,
                                'input_group': input_name,
                                'category': 'channel_ablation',
                                'metric': f'{channel}:{metric}',
                                'status': _status_from_values(values, mask=mask),
                                'count': summary['count'],
                            })
    return rows, evidence


def _channel_negative_controls(artifacts):
    """Recheck ALL == DROP on samples inactive for the dropped channel."""

    rows = []
    metrics = (
        'channel_delta_rel', 'channel_delta_mix',
        'channel_delta_readout', 'channel_output_effect',
    )
    for artifact in artifacts:
        step = artifact['identity']['source_checkpoint_step']
        arrays = artifact['arrays']
        for input_name in INPUT_GROUPS:
            relation_key = f'{input_name}_relation_counts'
            if relation_key not in arrays:
                continue
            relation_counts = np.asarray(arrays[relation_key])
            for channel_index, channel in enumerate(RELATION_NAMES):
                inactive = relation_counts[:, channel_index] == 0
                max_by_metric = {}
                for metric in metrics:
                    key = f'{input_name}_channel_{channel}_{metric}'
                    if key not in arrays:
                        continue
                    values = np.asarray(arrays[key])
                    max_by_metric[metric] = (
                        float(np.max(np.abs(values[inactive])))
                        if np.any(inactive) else None
                    )
                if not max_by_metric:
                    continue
                observed = [value for value in max_by_metric.values() if value is not None]
                max_abs = max(observed) if observed else None
                status = 'not observed' if not np.any(inactive) else (
                    'consistent' if max_abs is not None and max_abs <= NUMERICAL_TOLERANCE
                    else 'mixed'
                )
                if status == 'mixed':
                    raise M20ADAnalysisError(
                        f'Channel inactive negative control failed in '
                        f'{artifact["per_sample_path"]}: {input_name}/{channel} '
                        f'{max_by_metric}'
                    )
                rows.append({
                    'checkpoint_step': step,
                    'input_group': input_name,
                    'channel': channel,
                    'inactive_count': int(np.sum(inactive)),
                    'max_abs': max_abs,
                    'status': status,
                    **{f'{metric}_max_abs': value for metric, value in max_by_metric.items()},
                })
    return rows


def _trajectory_rows(metric_rows):
    return [
        row for row in metric_rows
        if row['activity_group'] == 'any_relation_active'
        and row['category'] != 'channel_ablation'
        and row['count'] > 0
    ]


def _trajectory_table(artifacts, coverage_rows, metric_rows, optimization_rows, *, attention_status):
    """Build the compact checkpoint trajectory required by the M20A handoff."""

    def metric_mean(step, categories, metric, input_names=INPUT_GROUPS):
        values = [
            row['mean'] for row in metric_rows
            if row['checkpoint_step'] == step
            and row['input_group'] in input_names
            and row['category'] in categories
            and row['metric'] == metric
            and row['activity_group'] == 'any_relation_active'
            and row['mean'] is not None
        ]
        return None if not values else float(np.mean(values))

    def optimization_mean(step, field):
        values = [
            row[field] for row in optimization_rows
            if row['checkpoint_step'] == step and row.get(field) is not None
        ]
        return None if not values else float(np.mean(values))

    coverage_by_step = defaultdict(list)
    for row in coverage_rows:
        coverage_by_step[row['checkpoint_step']].append(
            row['any_relation_active_fraction']
        )
    steps = sorted({row['checkpoint_step'] for row in coverage_rows})
    rows = []
    for step in steps:
        active_values = coverage_by_step[step]
        rows.append({
            'step': step,
            'active_fraction': float(np.mean(active_values)) if active_values else None,
            'delta_h_rel': metric_mean(step, {'hidden'}, 'delta_rel'),
            'delta_h_mix': metric_mean(step, {'core'}, 'delta_mix'),
            'propagation_ratio': metric_mean(step, {'core'}, 'propagation_ratio'),
            'delta_readout': metric_mean(step, {'readout'}, 'delta_readout'),
            'delta_actor': metric_mean(step, {'output'}, 'delta_action', ('actor',)),
            'delta_v': metric_mean(step, {'output'}, 'delta_value', ('value',)),
            'delta_q': metric_mean(
                step, {'output'}, 'delta_qmin', ('critic_value', 'critic_actor')
            ),
            'relation_grad_norm': optimization_mean(step, 'relation_grad_l2'),
            'relation_param_norm': optimization_mean(step, 'relation_param_l2'),
            'attention_status': attention_status,
        })
    return rows


def _evidence_table(artifacts, evidence):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row['input_group'], row['category'], row['metric'])].append(row['status'])
    rows = []
    for key in sorted(grouped):
        input_name, category, metric = key
        rows.append({
            'input_group': input_name,
            'category': category,
            'metric': metric,
            'status': _combine_statuses(grouped[key]),
            'checkpoint_count': len(grouped[key]),
        })

    # Explicit hard-gate evidence is kept separate from numeric magnitudes.
    parity_statuses = []
    integrity_statuses = []
    for artifact in artifacts:
        parity = artifact['summary'].get('normal_forward_parity', {})
        parity_statuses.append(
            'consistent' if parity.get('status') == 'pass' else 'mixed'
        )
        integrity = artifact['summary'].get('state_integrity', {})
        integrity_statuses.append(
            'consistent' if integrity.get('unchanged') is True else 'mixed'
        )
    rows.extend((
        {
            'input_group': 'all',
            'category': 'hard_gate',
            'metric': 'normal_forward_parity',
            'status': _combine_statuses(parity_statuses),
            'checkpoint_count': len(parity_statuses),
        },
        {
            'input_group': 'all',
            'category': 'hard_gate',
            'metric': 'state_integrity',
            'status': _combine_statuses(integrity_statuses),
            'checkpoint_count': len(integrity_statuses),
        },
    ))
    return rows


def _plot_coverage(coverage_rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [f'{row["checkpoint_step"]}\n{row["input_group"]}' for row in coverage_rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.5))
    channel_fields = {
        'any_relation_active': 'any_relation_active_fraction',
        'current_support': 'current_support_active_fraction',
        'goal_support': 'goal_support_active_fraction',
        'goal_conflict': 'goal_conflict_active_fraction',
    }
    for channel in ('any_relation_active', *RELATION_NAMES):
        values = [
            row[channel_fields[channel]]
            for row in coverage_rows
        ]
        axis.plot(x, values, marker='o', label=channel)
    axis.set_ylim(0, 1)
    axis.set_ylabel('active-sample fraction')
    axis.set_title('M20A-D relation coverage by channel (fixed batch)')
    axis.set_xticks(x, labels, rotation=45, ha='right')
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_trajectory(rows, path, *, category, metric, title, ylabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in rows:
        if row['category'] == category and row['metric'] == metric:
            grouped[row['input_group']].append((row['checkpoint_step'], row['mean']))
    for input_name, points in sorted(grouped.items()):
        points.sort()
        x, y = zip(*points)
        axis.plot(x, y, marker='o', label=input_name)
    axis.set_title(title)
    axis.set_xlabel('checkpoint step')
    axis.set_ylabel(ylabel)
    if grouped:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_active_inactive(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in rows:
        if row['category'] == 'readout' and row['metric'] == 'delta_readout':
            grouped[(row['input_group'], row['activity_group'])].append(
                (row['checkpoint_step'], row['mean'])
            )
    for (input_name, activity_group), points in sorted(grouped.items()):
        points.sort()
        x, y = zip(*points)
        axis.plot(x, y, marker='o', label=f'{input_name}:{activity_group}')
    axis.set_title('M20A-D active versus inactive relation samples')
    axis.set_xlabel('checkpoint step')
    axis.set_ylabel('readout RMS effect')
    if grouped:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_channel_ablation(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in rows:
        if row['category'] == 'channel_ablation' and row['metric'].endswith(':channel_output_effect'):
            channel = row['metric'].split(':', 1)[0]
            if row['activity_group'] == f'{channel}_active':
                grouped[(row['input_group'], channel)].append(
                    (row['checkpoint_step'], row['mean'])
                )
    for (input_name, channel), points in sorted(grouped.items()):
        points.sort()
        x, y = zip(*points)
        axis.plot(x, y, marker='o', label=f'{input_name}:{channel}')
    axis.set_title('M20A-D channel-wise output ablation effects')
    axis.set_xlabel('checkpoint step')
    axis.set_ylabel('ALL versus DROP channel effect')
    if grouped:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _optimization_rows(artifacts):
    rows = []
    for artifact in artifacts:
        summary = artifact['summary']
        step = artifact['identity']['source_checkpoint_step']
        gradient_metrics = summary.get('gradient_analysis', {}).get('metrics', {})
        parameter_metrics = summary.get('parameter_learning', {})
        for slot in ('actor', 'value', 'critic'):
            gradient = gradient_metrics.get(slot, {})
            parameter = parameter_metrics.get(slot, {})
            rows.append({
                'checkpoint_step': step,
                'slot': slot,
                'relation_param_l1': parameter.get('relation_param_l1'),
                'relation_param_l2': parameter.get('relation_param_l2'),
                'initial_parameter_displacement': parameter.get('initial_parameter_displacement'),
                'initial_parameter_displacement_status': parameter.get('initial_parameter_displacement_status'),
                'relation_grad_l1': gradient.get('relation_grad_l1'),
                'relation_grad_l2': gradient.get('relation_grad_l2'),
                'normalized_relation_grad': gradient.get('normalized_relation_grad'),
                'full_module_grad_l2': gradient.get('full_module_grad_l2'),
                'relation_grad_fraction': gradient.get('relation_grad_fraction'),
            })
    return rows


def _plot_optimization(rows, path, *, metric, title, ylabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in rows:
        value = _float(row.get(metric))
        if value is not None:
            grouped[row['slot']].append((row['checkpoint_step'], value))
    for slot, points in sorted(grouped.items()):
        points.sort()
        x, y = zip(*points)
        axis.plot(x, y, marker='o', label=slot)
    axis.set_title(title)
    axis.set_xlabel('checkpoint step')
    axis.set_ylabel(ylabel)
    if grouped:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _lookup_evidence(evidence_rows, *, category, metric, input_names=None):
    statuses = []
    input_names = set(input_names or INPUT_GROUPS)
    for row in evidence_rows:
        if (
            row['category'] == category
            and row['metric'] == metric
            and row['input_group'] in input_names
        ):
            statuses.append(row['status'])
    return _combine_statuses(statuses)


def _decision_table(coverage_rows, evidence_rows):
    active_observed = any(
        row['any_relation_active_fraction'] > 0 for row in coverage_rows
    )
    relation_status = _lookup_evidence(
        evidence_rows, category='hidden', metric='delta_rel',
    )
    core_status = _lookup_evidence(
        evidence_rows, category='core', metric='delta_mix',
    )
    readout_status = _lookup_evidence(
        evidence_rows, category='readout', metric='delta_readout',
    )
    output_status = _combine_statuses([
        _lookup_evidence(evidence_rows, category='output', metric=metric)
        for metric in ('delta_action', 'delta_value', 'delta_qmin')
    ])
    channel_statuses = [
        row['status'] for row in evidence_rows
        if row['category'] == 'channel_ablation'
    ]
    if not active_observed:
        coverage_status = 'not observed'
    else:
        coverage_status = 'consistent'
    case_a = (
        'consistent'
        if active_observed and relation_status == 'not observed'
        else 'insufficient evidence'
    )
    case_b = (
        'consistent'
        if relation_status == 'consistent' and core_status == 'not observed'
        else 'insufficient evidence'
    )
    case_c = (
        'consistent'
        if readout_status == 'consistent' and output_status == 'not observed'
        else 'insufficient evidence'
    )
    case_d = (
        'consistent'
        if active_observed and output_status == 'consistent'
        else 'insufficient evidence'
    )
    case_e = (
        'consistent'
        if len(set(channel_statuses)) > 1
        else ('not observed' if not channel_statuses else channel_statuses[0])
    )
    return [
        {
            'case': 'A',
            'status': case_a,
            'pattern': 'active relation edges with relation-update effect not observed',
            'frozen_interpretation': 'RelationAugmenter pathway did not show a detectable utilization effect in the selected evidence.',
        },
        {
            'case': 'B',
            'status': case_b,
            'pattern': 'relation update observed but core propagation not observed',
            'frozen_interpretation': 'Relation effect entered tokens but was not observed after the Mixer core.',
        },
        {
            'case': 'C',
            'status': case_c,
            'pattern': 'representation/readout effect observed but output effect not observed',
            'frozen_interpretation': 'The inspected representation depended on relations while downstream output dependence was not observed.',
        },
        {
            'case': 'D',
            'status': case_d,
            'pattern': 'same-checkpoint Correct-vs-Zero actor/value/Q output dependence',
            'frozen_interpretation': 'The C-M model shows numerical dependence on the relation input in the inspected actor/value/critic outputs; this is not a causal or generalization claim.',
        },
        {
            'case': 'E',
            'status': case_e,
            'pattern': 'slot/channel evidence is not uniform',
            'frozen_interpretation': 'Relation utilization is component/channel-specific and must not be treated as one uniform mechanism.',
        },
        {
            'case': 'coverage',
            'status': coverage_status,
            'pattern': 'fixed-batch relation coverage',
            'frozen_interpretation': 'Coverage is descriptive and is not a performance or causal claim.',
        },
    ]


def _markdown_report(
    provenance,
    coverage_rows,
    evidence_rows,
    optimization_rows,
    decision_rows,
    artifacts,
    figure_names,
):
    source_config = provenance.get('source_config', CONFIG_ID)
    source_readout = provenance.get('source_readout') or (
        'mean_context' if source_config == 'M20A-CUBE-C-M' else 'hybrid_context_query'
    )
    attention_status = provenance.get('attention_status') or (
        'not_applicable_mean_readout'
        if source_readout == 'mean_context'
        else 'applicable_hybrid_context_query'
    )
    if source_readout == 'mean_context':
        readout_heading = '## F. MeanContextReadout sensitivity'
        readout_text = (
            'MeanContextReadout is the selected readout. Readout sensitivity is represented '
            'by `delta_readout` and `relative_readout_difference`; attention is explicitly '
            f'`{attention_status}` because this readout has no attention mechanism.'
        )
    else:
        readout_heading = '## F. HybridContextQuery sensitivity'
        readout_text = (
            'Readout and attention sensitivity are represented by `delta_readout`, '
            '`relative_readout_difference`, and `attention_tv`; the per-entity attention '
            'arrays remain in each checkpoint NPZ.'
        )
    lines = [
        '# M20A-D relation-utilization post-hoc report',
        '',
        'This report is a post-hoc read of fixed diagnostic artifacts. It performs no checkpoint restore and no neural forward pass.',
        '',
        '## A. Source/model provenance',
        '',
        f'- Study/config: `{STUDY_ID}/{source_config}`',
        f'- Readout: `{source_readout}`',
        f'- Attention status: `{attention_status}`',
        f'- Source run: `{provenance["source_run_path"]}`',
        f'- Resolved-config fingerprint: `{provenance["source_resolved_config_fingerprint"]}`',
        f'- Fixed-batch fingerprint: `{provenance["fixed_batch_fingerprint_sha256"]}`',
        f'- Checkpoints observed: `{", ".join(str(step) for step in provenance["checkpoint_steps"])}`',
        '',
        '| checkpoint | source checkpoint SHA256 |',
        '|---:|---|',
    ]
    for artifact in artifacts:
        identity = artifact['identity']
        lines.append(f'| {identity["source_checkpoint_step"]} | `{identity["source_checkpoint_sha256"]}` |')
    lines.extend((
        '',
        '## B. Fixed batch provenance',
        '',
        'The fixed batch fingerprint above is required to match across every checkpoint artifact; the analyzer does not resample or reconstruct it.',
        '',
        '## C. Relation coverage',
        '',
        '| checkpoint | input group | any active | current support | goal support | goal conflict |',
        '|---:|---|---:|---:|---:|---:|',
    ))
    for row in coverage_rows:
        lines.append(
            f'| {row["checkpoint_step"]} | {row["input_group"]} | '
            f'{row["any_relation_active_fraction"]:.6f} | '
            f'{row["current_support_active_fraction"]:.6f} | '
            f'{row["goal_support_active_fraction"]:.6f} | '
            f'{row["goal_conflict_active_fraction"]:.6f} |'
        )
    lines.extend((
        '',
        '## D. RelationAugmenter utilization',
        '',
        '`consistent` means the requested numerical effect was observed in the selected artifact subset; `mixed` means the selected evidence is not uniform; `not observed` means no effect or no active samples were observed; `insufficient evidence` means the artifact does not support the check.',
        '',
        '| input group | category | metric | status | checkpoints |',
        '|---|---|---|---|---:|',
    ))
    for row in evidence_rows:
        lines.append(
            f'| {row["input_group"]} | {row["category"]} | {row["metric"]} | '
            f'{row["status"]} | {row["checkpoint_count"]} |'
        )
    lines.extend((
        '',
        '## E. Mixer propagation',
        '',
        'Core propagation is represented by the `delta_mix`, `relative_core_difference`, and `propagation_ratio` rows in `metric_summary.csv` and `trajectory.csv`.',
        '',
        readout_heading,
        '',
        readout_text,
        '',
        '## G. Actor / Value / Critic dependence',
        '',
        'Output metrics are separated by actor, value, critic-value-goal, and critic-actor-goal input groups. Critic differences use the fixed action selected during the formal diagnostic.',
        '',
        '## H. Channel-wise dependence',
        '',
        'Channel-ablation rows are computed only from the same checkpoint and same input, and are interpreted on channel-active samples separately from the inactive negative control. The explicit ALL==DROP inactive checks are recorded in `channel_negative_controls.csv`.',
        '',
        '## I. Optimization-signal and parameter-learning evidence',
        '',
        '| checkpoint | slot | relation parameter L2 | relation gradient L2 | normalized relation gradient | displacement status |',
        '|---:|---|---:|---:|---:|---|',
    ))
    for row in optimization_rows:
        lines.append(
            f'| {row["checkpoint_step"]} | {row["slot"]} | '
            f'{row.get("relation_param_l2") if row.get("relation_param_l2") is not None else ""} | '
            f'{row.get("relation_grad_l2") if row.get("relation_grad_l2") is not None else ""} | '
            f'{row.get("normalized_relation_grad") if row.get("normalized_relation_grad") is not None else ""} | '
            f'{row.get("initial_parameter_displacement_status") or "insufficient evidence"} |'
        )
    lines.extend((
        '',
        '## J. Training-time evolution',
        '',
        'The compact `trajectory_table.csv` reports step, active fraction, ΔH_rel, ΔH_mix, propagation ratio, Δreadout, Δactor, ΔV, ΔQ, relation gradient norm, and relation parameter norm. The long-form trajectory CSV and checkpoint-indexed figures are descriptive comparisons on one fixed batch; they do not compare raw output scales across different checkpoints as a causal quantity.',
        '',
        '## K. Frozen decision table',
        '',
        '| case | observed status | frozen pattern | frozen interpretation |',
        '|---|---|---|---|',
    ))
    for row in decision_rows:
        lines.append(
            f'| {row["case"]} | {row["status"]} | {row["pattern"]} | '
            f'{row["frozen_interpretation"]} |'
        )
    lines.extend((
        '',
        '## Interpretation boundary',
        '',
        'These artifacts establish only whether the frozen relation intervention was numerically propagated through the inspected actor/value/critic paths on the fixed batch. They do not establish a causal training explanation, task performance improvement, or generalization beyond the selected checkpoints and sampled inputs.',
        '',
        '## Figures',
        '',
    ))
    for figure_name in figure_names:
        lines.append(f'- `{figure_name}`')
    lines.append('')
    return '\n'.join(lines)


def analyze(diagnostics_root, output_root=None, *, source_config=DEFAULT_CONFIG_ID):
    """Read completed M20A-D artifacts and create a reproducible report."""

    artifacts = _completed_artifacts(diagnostics_root, source_config)
    first_identity = artifacts[0]['identity']
    source_readouts = {
        item['identity'].get('source_readout')
        for item in artifacts
        if item['identity'].get('source_readout') is not None
    }
    if len(source_readouts) > 1:
        raise M20ADAnalysisError(f'Artifacts mix source readouts: {sorted(source_readouts)!r}')
    source_readout = next(iter(source_readouts), None)
    if source_readout is None and source_config == 'M20A-CUBE-C-M':
        source_readout = 'mean_context'
    mean_readout = source_readout == 'mean_context'
    provenance = {
        'diagnostic_id': _campaign_id(source_config),
        'source_config': source_config,
        'source_readout': source_readout,
        'attention_status': (
            'not_applicable_mean_readout'
            if mean_readout else 'applicable_hybrid_context_query'
        ),
        'source_run_path': first_identity['source_run_path'],
        'source_resolved_config_fingerprint': first_identity['source_resolved_config_fingerprint'],
        'fixed_batch_fingerprint_sha256': first_identity['fixed_batch_fingerprint_sha256'],
        'checkpoint_steps': [item['identity']['source_checkpoint_step'] for item in artifacts],
        'source_checkpoint_sha256': {
            str(item['identity']['source_checkpoint_step']): item['identity']['source_checkpoint_sha256']
            for item in artifacts
        },
    }
    coverage_rows = _relation_coverage(artifacts)
    metric_rows, evidence = _metric_rows(artifacts)
    channel_negative_controls = _channel_negative_controls(artifacts)
    evidence_rows = _evidence_table(artifacts, evidence)
    trajectory_rows = _trajectory_rows(metric_rows)
    optimization_rows = _optimization_rows(artifacts)
    trajectory_table_rows = _trajectory_table(
        artifacts,
        coverage_rows,
        metric_rows,
        optimization_rows,
        attention_status=(
            'not_applicable_mean_readout'
            if mean_readout else 'applicable_hybrid_context_query'
        ),
    )
    decision_rows = _decision_table(coverage_rows, evidence_rows)

    diagnostics_root = Path(diagnostics_root).resolve()
    output_dir = (
        Path(output_root).resolve()
        if output_root is not None
        else (
            diagnostics_root / DIAGNOSTIC_ID / 'campaigns' / source_config / 'summary' / 'posthoc'
            if source_config != DEFAULT_CONFIG_ID
            else diagnostics_root / DIAGNOSTIC_ID / 'summary' / 'posthoc'
        )
    )
    source_dir = Path(provenance['source_run_path']).resolve()
    output_is_under_source = True
    source_is_under_output = True
    try:
        output_dir.relative_to(source_dir)
    except ValueError:
        output_is_under_source = False
    try:
        source_dir.relative_to(output_dir)
    except ValueError:
        source_is_under_output = False
    if output_is_under_source or source_is_under_output:
        raise M20ADAnalysisError(
            f'Post-hoc output must be independent of source run: {output_dir}'
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    coverage_fields = [
        'checkpoint_step', 'input_group', 'sample_count',
        'any_relation_active_fraction', 'current_support_active_fraction',
        'goal_support_active_fraction', 'goal_conflict_active_fraction',
        'total_edge_count',
    ]
    metric_fields = [
        'checkpoint_step', 'input_group', 'activity_group', 'category', 'metric',
        'count', 'mean', 'std', 'median', 'p10', 'p25', 'p75', 'p90', 'max',
        'fraction_gt_numerical_tolerance',
    ]
    evidence_fields = ['input_group', 'category', 'metric', 'status', 'checkpoint_count']
    optimization_fields = [
        'checkpoint_step', 'slot', 'relation_param_l1', 'relation_param_l2',
        'initial_parameter_displacement', 'initial_parameter_displacement_status',
        'relation_grad_l1', 'relation_grad_l2', 'normalized_relation_grad',
        'full_module_grad_l2', 'relation_grad_fraction',
    ]
    trajectory_table_fields = [
        'step', 'active_fraction', 'delta_h_rel', 'delta_h_mix',
        'propagation_ratio', 'delta_readout', 'delta_actor', 'delta_v',
        'delta_q', 'relation_grad_norm', 'relation_param_norm',
        'attention_status',
    ]
    channel_negative_control_fields = [
        'checkpoint_step', 'input_group', 'channel', 'inactive_count',
        'max_abs', 'status', 'channel_delta_rel_max_abs',
        'channel_delta_mix_max_abs', 'channel_delta_readout_max_abs',
        'channel_output_effect_max_abs',
    ]
    decision_fields = ['case', 'status', 'pattern', 'frozen_interpretation']
    _write_csv(output_dir / 'coverage.csv', coverage_rows, coverage_fields)
    _write_csv(output_dir / 'metric_summary.csv', metric_rows, metric_fields)
    _write_csv(output_dir / 'trajectory.csv', trajectory_rows, metric_fields)
    _write_csv(
        output_dir / 'trajectory_table.csv',
        trajectory_table_rows,
        trajectory_table_fields,
    )
    _write_csv(
        output_dir / 'channel_negative_controls.csv',
        channel_negative_controls,
        channel_negative_control_fields,
    )
    _write_csv(output_dir / 'evidence.csv', evidence_rows, evidence_fields)
    _write_csv(output_dir / 'optimization.csv', optimization_rows, optimization_fields)
    _write_csv(output_dir / 'decision_table.csv', decision_rows, decision_fields)

    figure_names = [
        'relation_coverage_by_channel.png',
        'delta_rel_trajectory.png',
        'delta_mix_trajectory.png',
        'propagation_ratio_trajectory.png',
    ]
    if not mean_readout:
        figure_names.append('attention_tv_trajectory.png')
    figure_names.extend([
        'actor_sensitivity_trajectory.png',
        'value_sensitivity_trajectory.png',
        'critic_sensitivity_trajectory.png',
        'active_vs_inactive.png',
        'channel_ablation_effects.png',
        'relation_gradient_norm_trajectory.png',
    ])
    figure_paths = {name: output_dir / name for name in figure_names}
    _plot_coverage(coverage_rows, figure_paths['relation_coverage_by_channel.png'])
    _plot_trajectory(
        trajectory_rows, figure_paths['delta_rel_trajectory.png'], category='hidden', metric='delta_rel',
        title='M20A-D relation update on active relation samples', ylabel='RMS effect',
    )
    _plot_trajectory(
        trajectory_rows, figure_paths['delta_mix_trajectory.png'], category='core', metric='delta_mix',
        title='M20A-D Mixer propagation on active relation samples', ylabel='RMS effect',
    )
    _plot_trajectory(
        trajectory_rows, figure_paths['propagation_ratio_trajectory.png'], category='core', metric='propagation_ratio',
        title='M20A-D relation-to-core propagation ratio', ylabel='propagation ratio',
    )
    if not mean_readout:
        _plot_trajectory(
            trajectory_rows, figure_paths['attention_tv_trajectory.png'], category='attention', metric='attention_tv',
            title='M20A-D HybridContextQuery attention variation', ylabel='attention TV',
        )
    _plot_trajectory(
        trajectory_rows, figure_paths['actor_sensitivity_trajectory.png'], category='output', metric='delta_action',
        title='M20A-D actor action sensitivity', ylabel='clipped action RMS effect',
    )
    _plot_trajectory(
        trajectory_rows, figure_paths['value_sensitivity_trajectory.png'], category='output', metric='delta_value',
        title='M20A-D value sensitivity', ylabel='absolute value effect',
    )
    _plot_trajectory(
        trajectory_rows, figure_paths['critic_sensitivity_trajectory.png'], category='output', metric='delta_qmin',
        title='M20A-D critic Qmin sensitivity', ylabel='absolute Qmin effect',
    )
    _plot_active_inactive(metric_rows, figure_paths['active_vs_inactive.png'])
    _plot_channel_ablation(metric_rows, figure_paths['channel_ablation_effects.png'])
    _plot_optimization(
        optimization_rows, figure_paths['relation_gradient_norm_trajectory.png'], metric='relation_grad_l2',
        title='M20A-D RelationAugmenter gradient norm trajectory', ylabel='relation gradient L2',
    )

    report = {
        'status': 'completed',
        'diagnostic_id': _campaign_id(source_config),
        'artifact_schema': f'{DIAGNOSTIC_SCHEMA}_posthoc_v1',
        'analysis_mode': 'posthoc_npz_json_only_no_forward',
        'provenance': provenance,
        'coverage_rows': len(coverage_rows),
        'metric_rows': len(metric_rows),
        'evidence_rows': len(evidence_rows),
        'optimization_rows': len(optimization_rows),
        'trajectory_table_rows': len(trajectory_table_rows),
        'channel_negative_control_rows': len(channel_negative_controls),
        'decision_table': decision_rows,
        'evidence': evidence_rows,
        'artifacts': [
            {
                'summary_path': item['summary_path'],
                'metadata_path': item['metadata_path'],
                'per_sample_path': item['per_sample_path'],
                **item['identity'],
            }
            for item in artifacts
        ],
        'outputs': {
            'coverage_csv': str(output_dir / 'coverage.csv'),
            'metric_summary_csv': str(output_dir / 'metric_summary.csv'),
            'trajectory_csv': str(output_dir / 'trajectory.csv'),
            'trajectory_table_csv': str(output_dir / 'trajectory_table.csv'),
            'channel_negative_controls_csv': str(
                output_dir / 'channel_negative_controls.csv'
            ),
            'evidence_csv': str(output_dir / 'evidence.csv'),
            'optimization_csv': str(output_dir / 'optimization.csv'),
            'decision_table_csv': str(output_dir / 'decision_table.csv'),
            'figures': [str(output_dir / name) for name in figure_names],
            'report_markdown': str(output_dir / 'report.md'),
        },
    }
    _write_json(output_dir / 'posthoc_summary.json', report)
    (output_dir / 'report.md').write_text(
        _markdown_report(
            provenance, coverage_rows, evidence_rows, optimization_rows,
            decision_rows, artifacts, figure_names,
        )
    )
    return report


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-root', default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument('--output-root', default=None)
    parser.add_argument(
        '--source-config', default=DEFAULT_CONFIG_ID,
        help='Diagnostic source configuration ID; default preserves the Q reader.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    try:
        report = analyze(
            args.diagnostics_root, args.output_root,
            source_config=args.source_config,
        )
        print(
            f'M20AD post-hoc report completed: artifacts={len(report["artifacts"])} '
            f'output={report["outputs"]["report_markdown"]}'
        )
        return 0
    except (M20ADAnalysisError, FileExistsError, OSError, ValueError) as error:
        print(f'M20AD post-hoc analysis blocked: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
