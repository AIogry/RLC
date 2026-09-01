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
DIAGNOSTIC_ID_D5 = 'M18-D5'
DIAGNOSTIC_ID_D6 = 'M18-D6'
K_VALUES = (1, 2, 4, 8)
DEFAULT_DIAGNOSTICS_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'
RETAINED_ENERGY_EPSILON = 1e-8

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


def _aggregate_values(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            'count': 0, 'mean': None, 'std': None, 'median': None,
            'p10': None, 'p25': None, 'p75': None, 'p90': None,
        }
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
    }


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


def _npz_scalar(arrays, name, path):
    if name not in arrays:
        raise ValueError(f'Missing {name!r} in trace artifact: {path}')
    value = np.asarray(arrays[name])
    if value.size != 1:
        raise ValueError(f'Expected scalar {name!r} in {path}, got shape {value.shape!r}')
    return value.reshape(()).item()


def _collect_retained_energy(diagnostics_root):
    """Derive D2+ from saved D2 per-sample metrics without neural execution."""

    trace_root = _m18d_root(diagnostics_root) / 'trace' / 'checkpoint_best'
    rows = []
    source_paths = []
    per_sample_parts = defaultdict(list)
    if not trace_root.is_dir():
        return rows, {}, source_paths
    for path in sorted(trace_root.rglob('*_metrics.npz')):
        metadata_path = path.with_name('m18d_metadata.json')
        try:
            metadata = _read_json(metadata_path)
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if metadata.get('status') != 'completed' or metadata.get('diagnostic_id') != DIAGNOSTIC_ID_TRACE:
            continue
        required = {'sample_id', 'iteration_k', 'slot', 'K_train', 'checkpoint_role', 'state_rms', 'mean_token_rms', 'token_variance'}
        if not required.issubset(arrays):
            continue
        train_k = _int(_npz_scalar(arrays, 'K_train', path))
        slot = str(_npz_scalar(arrays, 'slot', path))
        member = str(_npz_scalar(arrays, 'ensemble_member', path)) if 'ensemble_member' in arrays else ''
        role = str(_npz_scalar(arrays, 'checkpoint_role', path))
        if train_k is None or role != 'best' or slot not in ('actor', 'value', 'critic'):
            raise ValueError(f'Malformed D2 trace identity in {path}')
        if int(metadata.get('K_train', -1)) != train_k:
            raise ValueError(f'Trace K_train disagreement between metadata and artifact: {path}')
        sample_id = np.asarray(arrays['sample_id'], dtype=np.int64)
        iteration_k = np.asarray(arrays['iteration_k'], dtype=np.int64)
        state_rms = np.asarray(arrays['state_rms'], dtype=np.float64)
        mean_token_rms = np.asarray(arrays['mean_token_rms'], dtype=np.float64)
        token_variance = np.asarray(arrays['token_variance'], dtype=np.float64)
        expected_shape = (len(sample_id), len(iteration_k))
        if (
            state_rms.shape != expected_shape
            or mean_token_rms.shape != expected_shape
            or token_variance.shape != expected_shape
            or not np.array_equal(iteration_k, np.arange(len(iteration_k), dtype=np.int64))
        ):
            raise ValueError(f'D2 per-sample trace shape/iteration contract mismatch: {path}')
        state_energy = state_rms ** 2
        mean_energy = mean_token_rms ** 2
        decomposition_energy = mean_energy + token_variance
        nonzero = state_energy > RETAINED_ENERGY_EPSILON
        rho = np.full(expected_shape, np.nan, dtype=np.float64)
        rho_from_variance = np.full(expected_shape, np.nan, dtype=np.float64)
        rho[nonzero] = mean_energy[nonzero] / (state_energy[nonzero] + RETAINED_ENERGY_EPSILON)
        decomposition_nonzero = decomposition_energy > RETAINED_ENERGY_EPSILON
        rho_from_variance[decomposition_nonzero] = (
            mean_energy[decomposition_nonzero] /
            (decomposition_energy[decomposition_nonzero] + RETAINED_ENERGY_EPSILON)
        )
        if np.any((rho[nonzero] < -1e-6) | (rho[nonzero] > 1.0 + 1e-6)):
            raise ValueError(f'D2+ retained-energy fraction is outside [0,1] in {path}')
        # The two per-sample identities must agree.  Do not recombine prior
        # aggregate means, which would be a different statistic.
        identity_error = state_energy - decomposition_energy
        valid_identity = nonzero & decomposition_nonzero
        if np.any(np.abs(identity_error[valid_identity]) > 1e-6 * np.maximum(1.0, state_energy[valid_identity])):
            raise ValueError(f'D2+ RMS/variance energy identity fails in {path}')
        if np.any(np.abs(rho[valid_identity] - rho_from_variance[valid_identity]) > 2e-6):
            raise ValueError(f'D2+ retained-energy ratio identities disagree in {path}')
        discarded = np.where(np.isfinite(rho), 1.0 - rho, np.nan)
        source_paths.extend((str(path), str(metadata_path)))
        for index, iteration in enumerate(iteration_k):
            common = {
                'record_type': 'd2_plus',
                'K_train': int(train_k),
                'K_actor_test': '',
                'checkpoint_role': 'best',
                'slot': slot,
                'ensemble_member': member,
                'iteration_k': int(iteration),
                'is_depth_extrapolation': bool(int(iteration) > int(train_k)),
                'source_checkpoint_step': _int(metadata.get('source_checkpoint_step')),
                'source_checkpoint_sha256': metadata.get('source_checkpoint_sha256'),
                'source_path': str(path),
            }
            for metric, values in (
                ('mean_pooling_retained_energy_fraction', rho[:, index]),
                ('discarded_energy_fraction', discarded[:, index]),
            ):
                rows.append(common | {'metric': metric, **_aggregate_values(values)})
        grid_sample_id, grid_iteration = np.meshgrid(sample_id, iteration_k, indexing='ij')
        per_sample_parts['sample_id'].append(grid_sample_id.reshape(-1))
        per_sample_parts['K_train'].append(np.full(grid_sample_id.size, int(train_k), dtype=np.int64))
        per_sample_parts['slot'].append(np.full(grid_sample_id.size, slot, dtype='U16'))
        per_sample_parts['ensemble_member'].append(np.full(grid_sample_id.size, member, dtype='U16'))
        per_sample_parts['iteration_k'].append(grid_iteration.reshape(-1))
        per_sample_parts['is_depth_extrapolation'].append((grid_iteration.reshape(-1) > int(train_k)).astype(np.int8))
        per_sample_parts['mean_pooling_retained_energy_fraction'].append(rho.reshape(-1))
        per_sample_parts['discarded_energy_fraction'].append(discarded.reshape(-1))
        per_sample_parts['rho_from_token_variance_identity'].append(rho_from_variance.reshape(-1))
        per_sample_parts['state_energy_minus_mean_plus_variance'].append(identity_error.reshape(-1))
    rows.sort(key=lambda row: (row['K_train'], row['slot'], row['ensemble_member'], row['metric'], row['iteration_k']))
    arrays = {
        name: np.concatenate(values) if values else np.asarray([])
        for name, values in per_sample_parts.items()
    }
    return rows, arrays, sorted(set(source_paths))


def _collect_completed_singleton(root, *, diagnostic_id, summary_name):
    """Read at most one non-smoke formal D5/D6 artifact for the final report."""

    root = Path(root)
    found = []
    if not root.is_dir():
        return None, []
    for metadata_path in sorted(root.rglob('metadata.json')):
        try:
            metadata = _read_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            metadata.get('status') != 'completed'
            or metadata.get('diagnostic_id') != diagnostic_id
            or bool(metadata.get('smoke_only'))
        ):
            continue
        summary_path = metadata_path.with_name(summary_name)
        if not summary_path.is_file():
            # D5 has one completed metadata file per model underneath its
            # aggregate root.  Those worker files intentionally do not own
            # the aggregate summary and are not candidates for final mixing.
            continue
        found.append((metadata_path, metadata, summary_path, _read_json(summary_path)))
    if len(found) > 1:
        raise ValueError(f'Multiple formal {diagnostic_id} artifacts found; refuse to mix them: {[str(item[0]) for item in found]!r}')
    if not found:
        return None, []
    metadata_path, metadata, summary_path, summary = found[0]
    return {
        'metadata_path': str(metadata_path),
        'metadata': metadata,
        'summary_path': str(summary_path),
        'summary': summary,
        'root': str(metadata_path.parent),
    }, [str(metadata_path), str(summary_path)]


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


def _plot_retained_energy(rows, path):
    """Plot D2+ as one independent, non-subplot retained-energy figure."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    groups = defaultdict(list)
    for row in rows:
        if row['metric'] != 'mean_pooling_retained_energy_fraction' or row['mean'] is None:
            continue
        groups[(row['K_train'], row['slot'], str(row['ensemble_member']), str(row['source_checkpoint_sha256']))].append(
            (row['iteration_k'], row['mean'])
        )
    for (train_k, slot, member, checkpoint_sha), points in sorted(groups.items()):
        points = sorted(points)
        member_text = '' if not member else f', member={member}'
        hash_text = checkpoint_sha[:12] if checkpoint_sha else 'unknown'
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker='o',
            label=f'K_train={train_k}, slot={slot}{member_text}, SHA={hash_text}',
        )
    if groups:
        axis.legend()
    else:
        axis.text(0.5, 0.5, 'No completed D2+ per-sample trace data', ha='center', va='center')
    axis.set_xlabel('recurrent iteration k')
    axis.set_ylabel('mean-pooling retained-energy fraction')
    axis.set_title('D2+ mean-pooling retained energy | checkpoint role=best; checkpoint identity/K_train/slot in legend')
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_distribution(values, *, title, xlabel, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    if len(values):
        axis.hist(values, bins='auto')
        axis.axvline(0.0, linestyle='--')
    else:
        axis.text(0.5, 0.5, 'No finite formal diagnostic values', ha='center', va='center')
    axis.set_xlabel(xlabel)
    axis.set_ylabel('sample count')
    axis.set_title(title)
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


def _d2_plus_trend(rows, train_k, *, slot='actor'):
    points = sorted(
        (row['iteration_k'], row['mean'])
        for row in rows
        if row['metric'] == 'mean_pooling_retained_energy_fraction'
        and row['K_train'] == int(train_k)
        and row['slot'] == slot
        and row['mean'] is not None
    )
    if len(points) < 2:
        return None
    return {
        'first_iteration_k': points[0][0],
        'first_value': points[0][1],
        'last_iteration_k': points[-1][0],
        'last_value': points[-1][1],
        'delta': points[-1][1] - points[0][1],
    }


def _summary_metric(summary, metric, *, model_k=None, record_type=None, scope='overall'):
    if not summary:
        return None
    candidates = []
    for row in summary.get('summary_rows', []):
        if row.get('metric') != metric:
            continue
        if record_type is not None and row.get('record_type') != record_type:
            continue
        if model_k is not None and str(row.get('model_K')) != str(model_k):
            continue
        if scope is not None and row.get('scope') != scope:
            continue
        value = _float(row.get('mean'))
        if value is not None:
            candidates.append(value)
    return None if not candidates else float(np.mean(candidates))


def _final_decision_table(cross_rows, trace_rows, d2_plus_rows, d5_artifact, d6_artifact):
    """The pre-registered final six-hypothesis table, never causal TRUE/FALSE."""

    cross = {(row['K_train'], row['K_actor_test']): row['overall_success'] for row in cross_rows}
    rows = []

    depth_signs, depth_observations = [], []
    for train_k in (4, 8):
        diagonal = cross.get((train_k, train_k))
        shallower = [
            (test_k, success)
            for (candidate_k, test_k), success in cross.items()
            if candidate_k == train_k and test_k < train_k
        ]
        if diagonal is None or not shallower:
            continue
        test_k, success = max(shallower, key=lambda item: item[1])
        delta = success - diagonal
        depth_signs.append(1 if delta > 0 else -1)
        depth_observations.append(
            f'K{train_k}: best shallower actor K={test_k}, Δsuccess={delta:+.3f}'
        )
    rows.append({
        'hypothesis': 'H_depth_specialization',
        'status': _status(depth_signs),
        'evidence': 'D1 native-depth versus shallower actor-only inference',
        'observation': '; '.join(depth_observations) if depth_observations else 'No matched D1 depth cells.',
        'supports': 'A depth-specific actor inference response.',
        'does_not_prove': 'A causal training-depth mechanism.',
    })

    state_signs, state_observations = [], []
    for train_k in (4, 8):
        rms = _trend(trace_rows, 'state_rms', train_k)
        update = _trend(trace_rows, 'relative_update_from_previous', train_k)
        if rms is None or update is None:
            continue
        state_signs.append(1 if rms['delta'] > 0 and update['delta'] > 0 else -1)
        state_observations.append(
            f'K{train_k}: ΔRMS={rms["delta"]:+.4g}, Δrelative-update={update["delta"]:+.4g}'
        )
    rows.append({
        'hypothesis': 'H_state_instability',
        'status': _status(state_signs),
        'evidence': 'D2 actor state RMS and relative-update trajectories',
        'observation': '; '.join(state_observations) if state_observations else 'Insufficient D2 actor traces.',
        'supports': 'A descriptive scale/update-growth signature.',
        'does_not_prove': 'Numerical divergence or a rollout cause.',
    })

    action_signs, action_observations = [], []
    for train_k in (4, 8):
        action_delta = _trend(trace_rows, 'action_delta_from_previous', train_k)
        drift = _trend(trace_rows, 'action_drift_from_k1', train_k)
        if action_delta is None or drift is None:
            continue
        action_signs.append(1 if action_delta['delta'] > 0 and drift['delta'] > 0 else -1)
        action_observations.append(
            f'K{train_k}: Δaction-delta={action_delta["delta"]:+.4g}, Δdrift={drift["delta"]:+.4g}'
        )
    rows.append({
        'hypothesis': 'H_action_instability',
        'status': _status(action_signs),
        'evidence': 'D3 deterministic action refinement traces',
        'observation': '; '.join(action_observations) if action_observations else 'Insufficient D3 action traces.',
        'supports': 'A descriptive action-change signature.',
        'does_not_prove': 'That action change is harmful.',
    })

    pooling_signs, pooling_observations = [], []
    for train_k in (4, 8):
        rho = _d2_plus_trend(d2_plus_rows, train_k)
        variance = _trend(trace_rows, 'token_variance', train_k)
        if rho is None or variance is None:
            continue
        pooling_signs.append(1 if rho['delta'] < 0 and variance['delta'] > 0 else -1)
        pooling_observations.append(
            f'K{train_k}: Δrho={rho["delta"]:+.4g}, Δtoken-variance={variance["delta"]:+.4g}'
        )
    rows.append({
        'hypothesis': 'H_mean_pooling_mismatch',
        'status': _status(pooling_signs),
        'evidence': 'D2+ retained energy with D2 token variance',
        'observation': '; '.join(pooling_observations) if pooling_observations else 'No completed D2+ per-sample artifacts.',
        'supports': 'More energy in token-specific components discarded by mean pooling.',
        'does_not_prove': 'That mean pooling causes failure or measures task information.',
    })

    d6_summary = None if d6_artifact is None else d6_artifact['summary']
    p4 = _summary_metric(d6_summary, 'self_preference_4')
    p8 = _summary_metric(d6_summary, 'self_preference_8')
    joint = _summary_metric(d6_summary, 'joint_self_preference')
    if p4 is None or p8 is None or joint is None:
        coadaptation_status = 'insufficient evidence'
        coadaptation_observation = 'No completed formal D6 artifact.'
    elif p4 > 0.5 and p8 > 0.5 and joint > 0.5:
        coadaptation_status = 'consistent'
        coadaptation_observation = f'P4_self={p4:.3f}, P8_self={p8:.3f}, P_joint_self={joint:.3f}.'
    elif p4 <= 0.5 and p8 <= 0.5:
        coadaptation_status = 'not observed'
        coadaptation_observation = f'P4_self={p4:.3f}, P8_self={p8:.3f}, P_joint_self={joint:.3f}.'
    else:
        coadaptation_status = 'mixed'
        coadaptation_observation = f'P4_self={p4:.3f}, P8_self={p8:.3f}, P_joint_self={joint:.3f}.'
    rows.append({
        'hypothesis': 'H_actor_critic_coadaptation',
        'status': coadaptation_status,
        'evidence': 'D6 within-critic self-preference rates on fixed D234 actions',
        'observation': coadaptation_observation,
        'supports': 'Depth-specific actor–critic action-preference geometry when present.',
        'does_not_prove': 'A globally incorrect critic or cross-critic raw-Q ordering.',
    })

    d5_summary = None if d5_artifact is None else d5_artifact['summary']
    k4_progress = _summary_metric(d5_summary, 'net_logical_progress', model_k=4, record_type='model_episode_aggregate')
    k8_progress = _summary_metric(d5_summary, 'net_logical_progress', model_k=8, record_type='model_episode_aggregate')
    k4_regression = _summary_metric(d5_summary, 'number_of_regressive_transitions', model_k=4, record_type='model_episode_aggregate')
    k8_regression = _summary_metric(d5_summary, 'number_of_regressive_transitions', model_k=8, record_type='model_episode_aggregate')
    if None in (k4_progress, k8_progress, k4_regression, k8_regression):
        closed_status = 'insufficient evidence'
        closed_observation = 'No completed formal D5 logical rollout aggregate.'
    elif k8_progress < k4_progress or k8_regression > k4_regression:
        closed_status = 'consistent'
        closed_observation = (
            f'net progress K4={k4_progress:.3g}, K8={k8_progress:.3g}; '
            f'regressive transitions K4={k4_regression:.3g}, K8={k8_regression:.3g}.'
        )
    elif k8_progress >= k4_progress and k8_regression <= k4_regression:
        closed_status = 'not observed'
        closed_observation = (
            f'net progress K4={k4_progress:.3g}, K8={k8_progress:.3g}; '
            f'regressive transitions K4={k4_regression:.3g}, K8={k8_regression:.3g}.'
        )
    else:
        closed_status = 'mixed'
        closed_observation = (
            f'net progress K4={k4_progress:.3g}, K8={k8_progress:.3g}; '
            f'regressive transitions K4={k4_regression:.3g}, K8={k8_regression:.3g}.'
        )
    rows.append({
        'hypothesis': 'H_closed_loop_progress_failure',
        'status': closed_status,
        'evidence': 'D5 paired episode-level logical progress/regression',
        'observation': closed_observation,
        'supports': 'A closed-loop behavioral degradation signature when present.',
        'does_not_prove': 'An identifiable reasoning-versus-control attribution.',
    })
    return rows


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


def _render_report(
    cross_rows,
    trace_rows,
    delta_rows,
    d2_plus_rows,
    d5_artifact,
    d6_artifact,
    decisions,
    input_paths,
    figure_names,
):
    lines = [
        '# M18-D 诊断汇总（自动生成）',
        '',
        '本报告只汇总已完成的 M18-D artifacts。所有判断均是描述性证据，不构成因果诊断。',
        '',
        '## 输入与范围',
        '',
        f'- D1 completed cells: {len(cross_rows)}。',
        f'- D2/D3/D4 aggregate rows: {len(trace_rows)}。',
        f'- D2+ retained-energy aggregate rows: {len(d2_plus_rows)}。',
        f'- D5 formal artifact: {"已读取" if d5_artifact is not None else "尚未执行或不可用"}。',
        f'- D6 formal artifact: {"已读取" if d6_artifact is not None else "尚未执行或不可用"}。',
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
        '## D2：hidden-state geometry',
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
        '## D2+：mean-pooling retained energy',
        '',
        '正式指标为 `mean_pooling_retained_energy_fraction = mean_token_rms² / (state_rms² + 1e-8)`，先在每个 sample/iteration 计算，再聚合。它表示保留在共享 mean-token 成分中的平均 token-state energy fraction；不是 task information、mutual information 或因果证据。',
        '',
        '| K_train | Slot | Member | Iteration | Retained-energy mean | Discarded-energy mean | Depth scope |',
        '|---:|---|---|---:|---:|---:|---|',
    ])
    lookup = {}
    for row in d2_plus_rows:
        key = (row['K_train'], row['slot'], row['ensemble_member'], row['iteration_k'])
        lookup.setdefault(key, {})[row['metric']] = row
    for key in sorted(lookup):
        retained = lookup[key].get('mean_pooling_retained_energy_fraction', {})
        discarded = lookup[key].get('discarded_energy_fraction', {})
        lines.append(
            f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | '
            f'{_number(retained.get("mean"))} | {_number(discarded.get("mean"))} | '
            f'{"depth extrapolation" if retained.get("is_depth_extrapolation") else "within trained depth"} |'
        )
    if not d2_plus_rows:
        lines.append('|  |  |  |  |  |  | No completed per-sample trace artifact |')
    lines.extend([
        '',
        '## D3：action refinement',
        '',
        'D3 保存的 actor intermediate clipped actions 是后续 D6 的唯一 a4/a8 来源；没有在 D6 中重新 forward actor。',
        '',
        '## D4：source-critic ranking',
        '',
        'D4 的 Qmin 是 source critic 对受控 offline `(s,g,a)` 的输出，不等同环境 return，也不允许跨 critic raw-Q scale 比较。',
        '',
        '## D5：paired closed-loop behavior',
        '',
    ])
    if d5_artifact is None:
        lines.append('没有可纳入的正式 D5 artifact。tiny smoke 不作为科学证据，也不会被本汇总器混入。')
    else:
        d5_summary = d5_artifact['summary']
        lines.extend([
            f'- Summary: `{d5_artifact["summary_path"]}`。',
            f'- Paired episodes: {d5_summary.get("paired_episode_count")}；exact d* availability: {d5_summary.get("exact_shortest_distance_available_by_model")}。',
            '- 配对只保证相同 task/reset/goal/episode seeds；模型 action 一旦不同，轨迹不做逐 timestep 状态距离对齐。',
            '- 在当前 continuous low-level policy interface 中，未观测到 policy intended button，因此不能将 no logical interaction 唯一归因为 reasoning 或 motor-control failure。',
        ])
    lines.extend([
        '',
        '## D6：cross actor × cross critic preference',
        '',
    ])
    if d6_artifact is None:
        lines.append('没有可纳入的正式 D6 artifact。tiny smoke 不作为科学证据，也不会被本汇总器混入。')
    else:
        d6_summary = d6_artifact['summary']
        values = {
            name: _summary_metric(d6_summary, name)
            for name in ('self_preference_4', 'self_preference_8', 'joint_self_preference', 'tie_Q4_self', 'tie_Q8_self')
        }
        lines.extend([
            f'- Summary: `{d6_artifact["summary_path"]}`。',
            f'- P4_self={_number(values["self_preference_4"])}；P8_self={_number(values["self_preference_8"])}；P_joint_self={_number(values["joint_self_preference"])}。',
            f'- tie(Q4 self)={_number(values["tie_Q4_self"])}；tie(Q8 self)={_number(values["tie_Q8_self"])}。',
            '- 解释严格限于同一 critic 内对 a_data/a4/a8 的 preference；不同 critic 的 absolute Q scale 不作直接比较。',
        ])
    lines.extend([
        '',
        '## M18-D final hypothesis table（非因果）',
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
        '- retained-energy fraction 测量 representation energy，不测信息论意义上的 task information。',
        '- D5 的 logical progress 是可观测行为；除非环境暴露 intended target，reasoning/control 归因不可识别。',
        '- D6 的 critic disagreement 不是 uncertainty calibration。',
        '- 此 Study 的训练 seed 为 0；没有显著性检验或跨 seed 泛化结论。',
        '',
        '## M18-D STOP RULE',
        '',
        'M18-D is considered diagnostically complete after D2+, D5, and D6. No additional post-hoc latent/action/norm/cosine diagnostics will be added unless these analyses uncover an implementation correctness issue. Further hypotheses must be tested through intervention experiments.',
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
    d2_plus_rows, d2_plus_per_sample, d2_plus_paths = _collect_retained_energy(diagnostics_root)
    d5_artifact, d5_paths = _collect_completed_singleton(
        _m18d_root(diagnostics_root) / 'closed_loop' / 'checkpoint_locked',
        diagnostic_id=DIAGNOSTIC_ID_D5,
        summary_name='m18d_d5_summary.json',
    )
    d6_artifact, d6_paths = _collect_completed_singleton(
        _m18d_root(diagnostics_root) / 'cross_actor_critic' / 'checkpoint_locked',
        diagnostic_id=DIAGNOSTIC_ID_D6,
        summary_name='m18d_d6_summary.json',
    )
    if not cross_rows:
        raise ValueError('No completed M18-D1 checkpoint=best rows found')
    if not trace_rows:
        raise ValueError('No completed M18-D2/D3/D4 checkpoint=best rows found')
    if output_dir.exists():
        raise FileExistsError(f'M18-D report directory exists; refusing overwrite: {output_dir}')
    _validate_shared_provenance(cross_rows, trace_rows)
    delta_rows = _delta_rows(cross_rows)
    decisions = _final_decision_table(cross_rows, trace_rows, d2_plus_rows, d5_artifact, d6_artifact)
    input_paths = sorted(set(cross_paths + trace_paths + d2_plus_paths + d5_paths + d6_paths))
    if dry_run:
        return {
            'status': 'dry-run',
            'output_dir': str(output_dir),
            'cross_k_rows': len(cross_rows),
            'trace_rows': len(trace_rows),
            'd2_plus_rows': len(d2_plus_rows),
            'has_d5': d5_artifact is not None,
            'has_d6': d6_artifact is not None,
            'input_paths': input_paths,
            'decisions': decisions,
        }

    output_dir.mkdir(parents=True)
    figure_names = ['D1_cross_k_success_heatmap.png']
    _plot_heatmap(cross_rows, output_dir / figure_names[0])
    for filename, metric, slots, title in PLOT_SPECS:
        _plot_metric(trace_rows, metric, slots, title, output_dir / filename)
        figure_names.append(filename)
    if d2_plus_rows:
        retained_figure = 'D12_mean_pooling_retained_energy.png'
        _plot_retained_energy(d2_plus_rows, output_dir / retained_figure)
        figure_names.append(retained_figure)
        _write_csv(output_dir / 'm18d_d2_plus_summary.csv', d2_plus_rows, TRACE_FIELDS)
        np.savez_compressed(output_dir / 'm18d_d2_plus_per_sample.npz', **d2_plus_per_sample)
    if d6_artifact is not None:
        d6_per_sample_path = Path(d6_artifact['root']) / 'm18d_d6_per_sample.npz'
        if not d6_per_sample_path.is_file():
            raise ValueError(f'Completed D6 artifact lacks per-sample values: {d6_per_sample_path}')
        with np.load(d6_per_sample_path, allow_pickle=False) as loaded:
            q4_margin = np.asarray(loaded['Delta_Q4_self'], dtype=np.float64)
            q8_margin = np.asarray(loaded['Delta_Q8_self'], dtype=np.float64)
        _plot_distribution(
            q4_margin,
            title='D13 D6 within-Q4 self-preference margin | Q4(a4)-Q4(a8)',
            xlabel='Q4(a4) - Q4(a8)',
            path=output_dir / 'D13_D6_self_preference_margin_K4.png',
        )
        _plot_distribution(
            q8_margin,
            title='D14 D6 within-Q8 self-preference margin | Q8(a8)-Q8(a4)',
            xlabel='Q8(a8) - Q8(a4)',
            path=output_dir / 'D14_D6_self_preference_margin_K8.png',
        )
        figure_names.extend(('D13_D6_self_preference_margin_K4.png', 'D14_D6_self_preference_margin_K8.png'))
        input_paths.append(str(d6_per_sample_path))
    if d5_artifact is not None:
        paired_path = Path(d5_artifact['root']) / 'paired_episode_summary.csv'
        if not paired_path.is_file():
            raise ValueError(f'Completed D5 artifact lacks paired episode summary: {paired_path}')
        paired_rows = _read_csv(paired_path)
        d5_plot_specs = (
            ('D15_final_logical_distance_paired.png', 'K4_minus_K8_final_d_star', 'D15 paired final logical distance | K4-K8', 'final d* K4 - K8'),
            ('D16_best_logical_progress_paired.png', 'K4_minus_K8_best_logical_progress', 'D16 paired best logical progress | K4-K8', 'best logical progress K4 - K8'),
            ('D17_time_to_first_progress_paired.png', 'K4_minus_K8_time_to_first_logical_progress', 'D17 paired time to first logical progress | K4-K8', 'time to first progress K4 - K8'),
        )
        for filename, field, title, xlabel in d5_plot_specs:
            values = [_float(row.get(field)) for row in paired_rows]
            if any(value is not None for value in values):
                _plot_distribution(values, title=title, xlabel=xlabel, path=output_dir / filename)
                figure_names.append(filename)
        input_paths.append(str(paired_path))
    combined_rows = trace_rows + d2_plus_rows
    _write_csv(output_dir / 'm18d_summary.csv', combined_rows, TRACE_FIELDS)
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
        'd2_plus_rows': d2_plus_rows,
        'd5_artifact': d5_artifact,
        'd6_artifact': d6_artifact,
        'final_hypothesis_table': decisions,
        'figures': figure_names,
        'non_causal_interpretation': True,
    })
    (output_dir / 'M18D_report.md').write_text(
        _render_report(
            cross_rows,
            trace_rows,
            delta_rows,
            d2_plus_rows,
            d5_artifact,
            d6_artifact,
            decisions,
            input_paths,
            figure_names,
        ) + '\n'
    )
    return {
        'status': 'completed',
        'output_dir': str(output_dir),
        'cross_k_rows': len(cross_rows),
        'trace_rows': len(trace_rows),
        'd2_plus_rows': len(d2_plus_rows),
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
