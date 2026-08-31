"""M18-D2/D3/D4 fixed-batch recurrent trace diagnostics.

The tool restores one immutable M18 semantic best checkpoint, samples one
reproducible offline Puzzle-4x4 batch, and measures the actor trajectory
Z^0...Z^K, deterministic intermediate actions, and trained-critic ranking.
It does not update, finetune, save source checkpoints, or write under a source
run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.reevaluation import ReevaluationError, _resolved_agent_config
from impls.utils.checkpointing import sha256_file
from impls.utils.reproducibility import derive_seed
from tools import m18_cross_k_eval as d1


STUDY_ID = 'M18'
DIAGNOSTIC_ID = 'M18-D234'
ENVIRONMENT = 'puzzle-4x4-play-v0'
K_VALUES = (1, 2, 4, 8)
SLOT_NAMES = ('actor', 'value', 'critic')
DEFAULT_OUTPUT_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'
EPSILON = 1e-8


def _parse_csv_ints(value, option, *, allowed=None, min_value=None, max_value=None):
    return d1._parse_csv_ints(
        value, option, allowed=allowed, min_value=min_value, max_value=max_value,
    )


def _parse_slots(value):
    slots = tuple(item.strip() for item in str(value).split(',') if item.strip())
    if not slots:
        raise ValueError('--slots must contain at least one slot')
    if len(set(slots)) != len(slots) or any(slot not in SLOT_NAMES for slot in slots):
        raise ValueError(f'--slots must be a non-repeating subset of {SLOT_NAMES!r}')
    if 'actor' not in slots:
        raise ValueError('M18-D2/D3/D4 requires actor because D3/D4 use its intermediate actions')
    return slots


def _checkpoint_label(selector):
    return d1._checkpoint_label(selector)


def _array_fingerprint(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode('utf-8'))
        digest.update(str(array.dtype).encode('utf-8'))
        digest.update(repr(tuple(array.shape)).encode('utf-8'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])


def _configurations(study_path, train_ks):
    study = load_study(study_path)
    if study.study_id != STUDY_ID:
        raise ReevaluationError(f'Expected M18 Study, got {study.study_id!r}')
    if study.data.get('algorithms') != ['gciql']:
        raise ReevaluationError('M18-D234 requires the frozen GCIQL-only study')
    if study.data.get('environments') != [ENVIRONMENT] or study.data.get('seeds') != [0]:
        raise ReevaluationError('M18-D234 requires the frozen Puzzle-4x4, seed-0 study')
    configurations = [
        prepare_run_design(study.path, path)[1]
        for path in sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    ]
    by_k = {
        int(configuration.data['factors']['recurrent_compute_budget_K']): configuration
        for configuration in configurations
    }
    if set(by_k) != set(K_VALUES) or len(configurations) != len(K_VALUES):
        raise ReevaluationError(f'M18-D234 requires exactly K={K_VALUES!r}')
    return study, [by_k[k] for k in train_ks]


def _trace_root(output_root, checkpoint_selector, batch_size, diagnostic_seed):
    return (
        Path(output_root) / 'M18D' / 'trace'
        / f'checkpoint_{_checkpoint_label(checkpoint_selector)}'
        / f'fixed_batch_N{int(batch_size)}_seed{int(diagnostic_seed)}'
    )


def _job_output_dir(output_root, checkpoint_selector, batch_size, diagnostic_seed, train_k, max_trace_k):
    return _trace_root(output_root, checkpoint_selector, batch_size, diagnostic_seed) / (
        f'trainK{int(train_k)}'
    ) / f'maxTraceK{int(max_trace_k)}'


def _make_dataset_from_source(provenance):
    from impls.utils.datasets import GCDataset, HGCDataset, MultiHGCDataset
    from impls.utils.env_utils import make_env_and_datasets

    metadata = provenance['source_metadata']
    config = _resolved_agent_config(provenance['resolved_config'])
    source_k = d1._uniform_train_k(config, label=provenance['source_config_id'])
    d1.validate_m18_agent_config(config, source_k, label='M18-D234 source config')
    dataset_dir = metadata.get('dataset_dir')
    if not dataset_dir:
        raise ReevaluationError('M18 source metadata has no dataset_dir')
    env, raw_train, _ = make_env_and_datasets(
        ENVIRONMENT,
        frame_stack=config.get('frame_stack'),
        seed=derive_seed(provenance['source_training_seed'], 3),
        dataset_seed=derive_seed(provenance['source_training_seed'], 1),
        dataset_dir=dataset_dir,
    )
    classes = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset, 'MultiHGCDataset': MultiHGCDataset}
    dataset_name = config.get('dataset_class')
    if dataset_name not in classes:
        env.close()
        raise ReevaluationError(f'Unsupported M18 source dataset class: {dataset_name!r}')
    dataset = classes[dataset_name](
        raw_train, config, rng=derive_seed(provenance['source_training_seed'], 11),
    )
    return env, dataset, config


def _build_restored_agent(provenance):
    import jax
    from impls.agents import agents
    from impls.computation.accounting import count_parameters
    from impls.main import _computation_slot_accounting
    from impls.utils.flax_utils import restore_agent_from_checkpoint

    env, dataset, config = _make_dataset_from_source(provenance)
    try:
        example_batch = dataset.sample(1)
        agent = agents['gciql'].create(
            provenance['source_training_seed'],
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
        source_accounting = provenance['source_metadata'].get('computation_slot_accounting', {})
        target_accounting = _computation_slot_accounting(agent, config)
        if not isinstance(source_accounting, dict):
            raise ReevaluationError('M18 source lacks computation_slot_accounting')
        for slot_name in SLOT_NAMES:
            if source_accounting.get(slot_name, {}).get('trainable_params') != target_accounting.get(slot_name, {}).get('trainable_params'):
                raise ReevaluationError(f'{slot_name}: parameter count does not match source provenance')
        initial_count = count_parameters(agent.network.params)
        restored = restore_agent_from_checkpoint(agent, provenance['checkpoint_path'])
        if count_parameters(restored.network.params) != initial_count:
            raise ReevaluationError('Checkpoint restore changed trainable parameter count')
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(restored.network.params)):
            raise ReevaluationError('Restored checkpoint has non-finite parameters')
        return restored, env, dataset, config
    except BaseException:
        env.close()
        raise


def _fixed_batch_paths(root):
    root = Path(root)
    return root / 'fixed_batch.npz', root / 'fixed_batch_metadata.json'


def _create_fixed_batch(root, provenance, batch_size, diagnostic_seed):
    root = Path(root)
    artifact_path, metadata_path = _fixed_batch_paths(root)
    if root.exists():
        if not artifact_path.is_file() or not metadata_path.is_file():
            raise FileExistsError(f'Fixed-batch root is incomplete; refusing overwrite: {root}')
        return _load_fixed_batch(root, provenance, batch_size, diagnostic_seed)

    env, dataset, config = _make_dataset_from_source(provenance)
    try:
        rng = np.random.default_rng(int(diagnostic_seed))
        indices = np.asarray(dataset.dataset.get_random_idxs(int(batch_size), rng=rng), dtype=np.int64)
        sampled = dataset.sample(int(batch_size), idxs=indices, rng=rng)
        arrays = {
            'sample_id': np.arange(int(batch_size), dtype=np.int64),
            'sample_indices': indices,
            'observations': np.asarray(sampled['observations']),
            'actor_goals': np.asarray(sampled['actor_goals']),
            'value_goals': np.asarray(sampled['value_goals']),
            'dataset_actions': np.asarray(sampled['actions']),
        }
        fingerprint = _array_fingerprint(arrays)
        root.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(artifact_path, **arrays)
        metadata = {
            'diagnostic_id': DIAGNOSTIC_ID,
            'environment': ENVIRONMENT,
            'dataset_dir': provenance['source_metadata'].get('dataset_dir'),
            'dataset_class': config.get('dataset_class'),
            'batch_size': int(batch_size),
            'diagnostic_seed': int(diagnostic_seed),
            'sampling_rng': 'numpy.default_rng(diagnostic_seed)',
            'goal_semantics_for_actor_value_critic_trace': 'actor_goals',
            'goal_semantics_rationale': (
                'All slots receive the same stored actor_goals so D2/D3/D4 '
                'compare one controlled (s,g) batch. value_goals are retained '
                'only as provenance of the original GCDataset draw.'
            ),
            'source_study_id': provenance['source_study_id'],
            'batch_reference_source_config_id': provenance['source_config_id'],
            'batch_reference_source_run_dir': provenance['source_run_dir'],
            'batch_reference_dataset_dir': provenance['source_metadata'].get('dataset_dir'),
            'sample_indices_saved': True,
            'array_shapes': {key: list(value.shape) for key, value in arrays.items()},
            'batch_fingerprint_sha256': fingerprint,
            'artifact_path': str(artifact_path),
        }
        _write_json(metadata_path, metadata)
        return arrays, metadata
    finally:
        env.close()


def _load_fixed_batch(root, provenance, batch_size, diagnostic_seed):
    artifact_path, metadata_path = _fixed_batch_paths(root)
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'Missing fixed diagnostic batch artifact under {root}')
    with metadata_path.open() as file:
        metadata = json.load(file)
    if int(metadata.get('batch_size', -1)) != int(batch_size):
        raise ReevaluationError('Existing fixed batch has a different batch_size')
    if int(metadata.get('diagnostic_seed', -1)) != int(diagnostic_seed):
        raise ReevaluationError('Existing fixed batch has a different diagnostic_seed')
    if metadata.get('environment') != ENVIRONMENT:
        raise ReevaluationError('Existing fixed batch has a different environment')
    if metadata.get('source_study_id') != provenance['source_study_id']:
        raise ReevaluationError('Existing fixed batch has a different source study')
    if metadata.get('batch_reference_dataset_dir') != provenance['source_metadata'].get('dataset_dir'):
        raise ReevaluationError('Existing fixed batch has a different dataset directory')
    with np.load(artifact_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    expected = {'sample_id', 'sample_indices', 'observations', 'actor_goals', 'value_goals', 'dataset_actions'}
    if set(arrays) != expected:
        raise ReevaluationError(f'Fixed batch fields mismatch: {sorted(arrays)!r}')
    if arrays['observations'].shape[0] != int(batch_size):
        raise ReevaluationError('Fixed batch has an inconsistent leading dimension')
    fingerprint = _array_fingerprint(arrays)
    if metadata.get('batch_fingerprint_sha256') != fingerprint:
        raise ReevaluationError('Fixed batch artifact fingerprint mismatch')
    return arrays, metadata


def _nan_matrix(batch_size, iterations):
    return np.full((batch_size, iterations), np.nan, dtype=np.float64)


def _state_metrics(token_states, readout_states, epsilon=EPSILON):
    """Compute D2 metrics; row k describes state Z^k and k-1 -> k updates."""

    states = np.asarray(token_states, dtype=np.float64)
    readouts = np.asarray(readout_states, dtype=np.float64)
    if states.ndim != 4:
        raise ValueError(f'Expected token states [B,K,T,D], got {states.shape}')
    if readouts.ndim != 3 or readouts.shape[:2] != states.shape[:2]:
        raise ValueError(f'Readout trace shape mismatch: states={states.shape}, readouts={readouts.shape}')
    batch_size, steps, tokens, width = states.shape
    result = {
        'state_rms': np.sqrt(np.mean(states ** 2, axis=(2, 3))),
        'absolute_update_from_previous': _nan_matrix(batch_size, steps),
        'relative_update_from_previous': _nan_matrix(batch_size, steps),
        'state_cosine_from_previous': _nan_matrix(batch_size, steps),
        'token_variance': np.mean(
            (states - np.mean(states, axis=2, keepdims=True)) ** 2,
            axis=(2, 3),
        ),
        'pairwise_token_cosine': np.empty((batch_size, steps), dtype=np.float64),
        'mean_token_rms': np.sqrt(np.mean(np.mean(states, axis=2) ** 2, axis=2)),
        'readout_rms': np.sqrt(np.mean(readouts ** 2, axis=2)),
    }
    norms = np.linalg.norm(states, axis=(2, 3))
    if steps > 1:
        diffs = states[:, 1:] - states[:, :-1]
        result['absolute_update_from_previous'][:, 1:] = np.sqrt(np.mean(diffs ** 2, axis=(2, 3)))
        relative = np.linalg.norm(diffs, axis=(2, 3)) / (norms[:, :-1] + float(epsilon))
        # Z^0 is zero under the M18 zero_buffer contract.  Do not interpret
        # the first relative update/cosine as a finite convergence statistic.
        if steps > 2:
            result['relative_update_from_previous'][:, 2:] = relative[:, 1:]
            cosine = np.sum(states[:, 2:] * states[:, 1:-1], axis=(2, 3)) / (
                norms[:, 2:] * norms[:, 1:-1] + float(epsilon)
            )
            result['state_cosine_from_previous'][:, 2:] = cosine
    token_norms = np.linalg.norm(states, axis=3, keepdims=True)
    token_unit = states / np.maximum(token_norms, float(epsilon))
    pairwise = np.einsum('bktd,bksd->bkts', token_unit, token_unit)
    off_diagonal = 1.0 - np.eye(tokens, dtype=np.float64)
    result['pairwise_token_cosine'][:, :] = np.sum(pairwise * off_diagonal[None, None], axis=(2, 3)) / (
        tokens * (tokens - 1)
    )
    return result


def _action_metrics(action_means, dataset_actions):
    """Compute D3 metrics on deterministic actor means for k >= 1."""

    means = np.asarray(action_means, dtype=np.float64)
    actions = np.clip(means, -1.0, 1.0)
    data_actions = np.asarray(dataset_actions, dtype=np.float64)
    if means.ndim != 3 or data_actions.shape != (means.shape[0], means.shape[2]):
        raise ValueError(f'Actor mean/data action shape mismatch: {means.shape} vs {data_actions.shape}')
    batch_size, steps, action_dim = means.shape
    result = {
        'action_delta_from_previous': _nan_matrix(batch_size, steps),
        'action_drift_from_k1': _nan_matrix(batch_size, steps),
        'action_mean_saturation_fraction': _nan_matrix(batch_size, steps),
        'action_near_boundary_fraction': _nan_matrix(batch_size, steps),
        'dataset_action_mse': _nan_matrix(batch_size, steps),
    }
    if steps > 1:
        primary_means = means[:, 1:]
        primary_actions = actions[:, 1:]
        result['action_mean_saturation_fraction'][:, 1:] = np.mean(np.abs(primary_means) >= 1.0, axis=2)
        result['action_near_boundary_fraction'][:, 1:] = np.mean(np.abs(primary_actions) >= 0.95, axis=2)
        result['dataset_action_mse'][:, 1:] = np.mean(
            (primary_actions - data_actions[:, None]) ** 2,
            axis=2,
        )
        result['action_drift_from_k1'][:, 1:] = np.sqrt(np.mean(
            (primary_actions - primary_actions[:, :1]) ** 2,
            axis=2,
        ))
    if steps > 2:
        result['action_delta_from_previous'][:, 2:] = np.sqrt(np.mean(
            (actions[:, 2:] - actions[:, 1:-1]) ** 2,
            axis=2,
        ))
    return actions, result


def _critic_action_metrics(agent, batch, clipped_actions):
    """Evaluate the trained source-K critic on actor actions a^1...a^K."""

    observations = np.asarray(batch['observations'])
    goals = np.asarray(batch['actor_goals'])
    dataset_actions = np.asarray(batch['dataset_actions'])
    actions = np.asarray(clipped_actions)
    batch_size, steps, action_dim = actions.shape
    if steps <= 1:
        raise ValueError('Actor trace must include at least Z^0 and Z^1')
    evaluation_actions = actions[:, 1:]
    trace_steps = evaluation_actions.shape[1]
    flat_observations = np.repeat(observations[:, None], trace_steps, axis=1).reshape(batch_size * trace_steps, -1)
    flat_goals = np.repeat(goals[:, None], trace_steps, axis=1).reshape(batch_size * trace_steps, -1)
    flat_actions = evaluation_actions.reshape(batch_size * trace_steps, action_dim)
    q_trace = np.asarray(agent.network.select('critic')(flat_observations, flat_goals, flat_actions))
    q_dataset = np.asarray(agent.network.select('critic')(observations, goals, dataset_actions))
    if q_trace.shape != (2, batch_size * trace_steps) or q_dataset.shape != (2, batch_size):
        raise ReevaluationError(
            f'Expected two-member critic outputs, got trace={q_trace.shape}, dataset={q_dataset.shape}'
        )
    q_trace = q_trace.reshape(2, batch_size, trace_steps)
    q_min = np.minimum(q_trace[0], q_trace[1])
    q_min_dataset = np.minimum(q_dataset[0], q_dataset[1])
    metrics = {
        'q1': _nan_matrix(batch_size, steps),
        'q2': _nan_matrix(batch_size, steps),
        'qmin': _nan_matrix(batch_size, steps),
        'qgap_vs_dataset_action': _nan_matrix(batch_size, steps),
        'critic_disagreement': _nan_matrix(batch_size, steps),
    }
    metrics['q1'][:, 1:] = q_trace[0]
    metrics['q2'][:, 1:] = q_trace[1]
    metrics['qmin'][:, 1:] = q_min
    metrics['qgap_vs_dataset_action'][:, 1:] = q_min - q_min_dataset[:, None]
    metrics['critic_disagreement'][:, 1:] = np.abs(q_trace[0] - q_trace[1])
    return metrics, q_min_dataset


def _finite_or_nan(metrics):
    for name, values in metrics.items():
        values = np.asarray(values)
        finite = values[~np.isnan(values)]
        if not np.all(np.isfinite(finite)):
            raise ReevaluationError(f'Non-finite diagnostic metric {name}')


def _aggregate_metric(values):
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


def _summary_rows(metrics, *, train_k, slot, ensemble_member, max_trace_k):
    rows = []
    for metric, values in sorted(metrics.items()):
        values = np.asarray(values)
        if values.ndim != 2:
            continue
        for iteration_k in range(values.shape[1]):
            rows.append({
                'K_train': int(train_k),
                'checkpoint_role': 'best',
                'slot': slot,
                'ensemble_member': ensemble_member,
                'iteration_k': int(iteration_k),
                'is_depth_extrapolation': bool(iteration_k > int(train_k)),
                'metric': metric,
                **_aggregate_metric(values[:, iteration_k]),
            })
    return rows


def _save_slot_artifact(
    path,
    *,
    sample_id,
    iteration_k,
    metrics,
    slot,
    train_k,
    checkpoint_step,
    ensemble_member,
    extra=None,
    raw_states=None,
):
    arrays = {
        'sample_id': np.asarray(sample_id, dtype=np.int64),
        'iteration_k': np.asarray(iteration_k, dtype=np.int64),
        'slot': np.asarray(str(slot)),
        'K_train': np.asarray(int(train_k), dtype=np.int64),
        'checkpoint_role': np.asarray('best'),
        'checkpoint_step': np.asarray(int(checkpoint_step), dtype=np.int64),
        'ensemble_member': np.asarray(str(ensemble_member)),
        **{name: np.asarray(value) for name, value in metrics.items()},
    }
    if extra:
        arrays.update({name: np.asarray(value) for name, value in extra.items()})
    if raw_states is not None:
        arrays['raw_token_states'] = np.asarray(raw_states)
    np.savez_compressed(path, **arrays)


def _trace_slot(agent, slot_name, batch, max_trace_k):
    observations = batch['observations']
    goals = batch['actor_goals']
    if slot_name == 'actor':
        trace = agent.network(
            observations, goals, name='actor', method='diagnostic_trace', max_iterations=int(max_trace_k),
        )
    elif slot_name == 'value':
        trace = agent.network(
            observations, goals, name='value', method='diagnostic_trace', max_iterations=int(max_trace_k),
        )
    elif slot_name == 'critic':
        trace = agent.network(
            observations, goals, batch['dataset_actions'],
            name='critic', method='diagnostic_trace', max_iterations=int(max_trace_k),
        )
    else:
        raise ValueError(f'Unsupported trace slot: {slot_name}')
    return {name: np.asarray(value) for name, value in trace.items()}


def _run_trace_job(job, batch, batch_metadata, *, max_trace_k, slots, save_raw_states, diagnostic_code_commit):
    output_dir = Path(job['output_dir'])
    if output_dir.exists():
        raise FileExistsError(f'M18-D234 output exists; refusing overwrite: {output_dir}')
    provenance = job['provenance']
    source_hash_before = d1._stable_checkpoint_sha256(provenance['checkpoint_path'])
    if source_hash_before != provenance['checkpoint_sha256']:
        raise ReevaluationError('Source checkpoint changed after planning; refusing trace execution')
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / 'm18d_metadata.json'
    metadata = {
        'status': 'running',
        'diagnostic_id': DIAGNOSTIC_ID,
        'diagnostic_code_commit': str(diagnostic_code_commit),
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'source_config_slug': provenance['source_config_slug'],
        'source_run_dir': provenance['source_run_dir'],
        'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'source_run_status_at_diagnostic': provenance['source_run_status_at_validation'],
        'source_training_latest_step_at_diagnostic': provenance['source_training_latest_step_at_diagnostic'],
        'source_git_commit': provenance['source_git_commit'],
        'source_checkpoint_role': provenance['resolved_checkpoint_role'],
        'source_checkpoint_step': provenance['checkpoint_step'],
        'source_checkpoint_selection_metric': provenance['source_checkpoint_selection_metric'],
        'source_checkpoint_selection_metric_value': provenance['source_checkpoint_selection_metric_value'],
        'source_checkpoint_sha256': provenance['checkpoint_sha256'],
        'source_checkpoint_path': provenance['checkpoint_path'],
        'source_checkpoint_hash_before': source_hash_before,
        'K_train': int(job['K_train']),
        'max_trace_k': int(max_trace_k),
        'is_depth_extrapolation_by_iteration': {
            str(k): bool(k > int(job['K_train'])) for k in range(int(max_trace_k) + 1)
        },
        'slots': list(slots),
        'fixed_batch_metadata_path': str(Path(batch_metadata['artifact_path']).with_name('fixed_batch_metadata.json')),
        'fixed_batch_fingerprint_sha256': batch_metadata['batch_fingerprint_sha256'],
        'fixed_batch_goal_semantics': batch_metadata['goal_semantics_for_actor_value_critic_trace'],
        'save_raw_states': bool(save_raw_states),
        'epsilon': EPSILON,
        'metric_definitions': {
            'state_rms': 'RMS(Z^k) over token and feature axes',
            'absolute_update_from_previous': 'RMS(Z^k - Z^(k-1)); NaN at k=0',
            'relative_update_from_previous': (
                '||Z^k-Z^(k-1)||_2 / (||Z^(k-1)||_2 + epsilon); '
                'NaN at k=0 and k=1 because Z^0=0 is not interpreted'
            ),
            'state_cosine_from_previous': 'cos(vec(Z^k), vec(Z^(k-1))); NaN at k=0 and k=1',
            'token_variance': 'mean squared deviation from the per-sample mean token',
            'pairwise_token_cosine': 'mean off-diagonal token cosine with epsilon-stabilized norms',
            'mean_token_rms': 'RMS(mean_token(Z^k))',
            'readout_rms': 'RMS(same restored MeanContextReadout(Z^k, context))',
            'action_delta_from_previous': 'RMS(a^k-a^(k-1)); primary values begin at k=2',
            'action_drift_from_k1': 'RMS(a^k-a^1); defined for k>=1',
            'action_mean_saturation_fraction': 'fraction(|mu^k_j| >= 1.0)',
            'action_near_boundary_fraction': 'fraction(|clip(mu^k)_j| >= 0.95)',
            'dataset_action_mse': 'mean_j((clip(mu^k)_j-a_dataset_j)^2)',
            'qgap_vs_dataset_action': 'Qmin(s,g,a^k)-Qmin(s,g,a_dataset)',
            'critic_disagreement': '|Q1(s,g,a^k)-Q2(s,g,a^k)|',
        },
        'critic_evaluation_policy': (
            'D4 always uses the restored source critic at source K_train; '
            'actor intermediate k never changes critic inference depth.'
        ),
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
    }
    _write_json(metadata_path, metadata)
    env = None
    try:
        agent, env, _, config = _build_restored_agent(provenance)
        train_k = int(job['K_train'])
        all_rows = []
        artifacts = []
        actor_trace = _trace_slot(agent, 'actor', batch, max_trace_k)
        actor_states = actor_trace['token_states']
        actor_readouts = actor_trace['readout_states']
        actor_means = actor_trace['action_means']
        actor_state_metrics = _state_metrics(actor_states, actor_readouts)
        clipped_actions, actor_action_metrics = _action_metrics(actor_means, batch['dataset_actions'])
        critic_action_metrics, qmin_dataset = _critic_action_metrics(agent, batch, clipped_actions)
        actor_metrics = actor_state_metrics | actor_action_metrics | critic_action_metrics
        _finite_or_nan(actor_metrics)
        normal_actor_mean = np.asarray(agent.network.select('actor')(
            batch['observations'], batch['actor_goals'], temperature=0.0,
        ).mode())
        final_actor_error = float(np.max(np.abs(actor_means[:, train_k] - normal_actor_mean)))
        if final_actor_error > 1e-6:
            raise ReevaluationError(
                f'Intermediate actor action at trained K does not match normal actor mode: {final_actor_error}'
            )
        actor_artifact = output_dir / 'actor_metrics.npz'
        _save_slot_artifact(
            actor_artifact,
            sample_id=batch['sample_id'],
            iteration_k=np.arange(int(max_trace_k) + 1),
            metrics=actor_metrics,
            slot='actor',
            train_k=train_k,
            checkpoint_step=provenance['checkpoint_step'],
            ensemble_member='',
            extra={
                'unclipped_action_mean': actor_means,
                'clipped_action': clipped_actions,
                'qmin_dataset_action': qmin_dataset,
                'normal_actor_mode_at_train_k': normal_actor_mean,
            },
            raw_states=actor_states if save_raw_states else None,
        )
        artifacts.append(str(actor_artifact))
        all_rows.extend(_summary_rows(
            actor_metrics, train_k=train_k, slot='actor', ensemble_member='', max_trace_k=max_trace_k,
        ))
        for slot_name in slots:
            if slot_name == 'actor':
                continue
            trace = _trace_slot(agent, slot_name, batch, max_trace_k)
            token_states = trace['token_states']
            readout_states = trace['readout_states']
            scalar_values = trace['values']
            if token_states.ndim == 4:
                token_states = token_states[None]
                readout_states = readout_states[None]
                scalar_values = scalar_values[None]
            if token_states.ndim != 5:
                raise ReevaluationError(f'{slot_name} trace has unexpected token shape {token_states.shape}')
            for member in range(token_states.shape[0]):
                metrics = _state_metrics(token_states[member], readout_states[member])
                metrics['slot_scalar_output'] = np.asarray(scalar_values[member], dtype=np.float64)
                _finite_or_nan(metrics)
                member_suffix = '' if token_states.shape[0] == 1 else f'_member{member}'
                artifact = output_dir / f'{slot_name}{member_suffix}_metrics.npz'
                _save_slot_artifact(
                    artifact,
                    sample_id=batch['sample_id'],
                    iteration_k=np.arange(int(max_trace_k) + 1),
                    metrics=metrics,
                    slot=slot_name,
                    train_k=train_k,
                    checkpoint_step=provenance['checkpoint_step'],
                    ensemble_member=member if token_states.shape[0] > 1 else '',
                    extra=None,
                    raw_states=token_states[member] if save_raw_states else None,
                )
                artifacts.append(str(artifact))
                all_rows.extend(_summary_rows(
                    metrics,
                    train_k=train_k,
                    slot=slot_name,
                    ensemble_member=member if token_states.shape[0] > 1 else '',
                    max_trace_k=max_trace_k,
                ))
        summary_fields = (
            'K_train', 'checkpoint_role', 'slot', 'ensemble_member', 'iteration_k',
            'is_depth_extrapolation', 'metric', 'count', 'mean', 'std', 'median',
            'p10', 'p25', 'p75', 'p90',
        )
        _write_csv(output_dir / 'trace_summary.csv', all_rows, summary_fields)
        source_hash_after = d1._stable_checkpoint_sha256(provenance['checkpoint_path'])
        if source_hash_after != source_hash_before:
            raise ReevaluationError('Source checkpoint SHA256 changed during M18-D234 execution')
        summary = {
            'status': 'completed',
            'diagnostic_id': DIAGNOSTIC_ID,
            'K_train': train_k,
            'checkpoint_role': 'best',
            'checkpoint_step': provenance['checkpoint_step'],
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            'source_checkpoint_hash_before': source_hash_before,
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_immutable': True,
            'optimizer_updates': 0,
            'evaluation_only': True,
            'finetuning': False,
            'max_trace_k': int(max_trace_k),
            'batch_size': int(batch['sample_id'].shape[0]),
            'actor_final_mean_vs_normal_mode_max_abs_error': final_actor_error,
            'artifacts': artifacts,
            'summary_row_count': len(all_rows),
        }
        _write_json(output_dir / 'trace_summary.json', summary)
        metadata.update({
            'status': 'completed',
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_immutable': True,
            'actor_final_mean_vs_normal_mode_max_abs_error': final_actor_error,
            'artifact_paths': artifacts,
        })
        _write_json(metadata_path, metadata)
        return summary
    except BaseException as error:
        try:
            source_hash_after = d1._stable_checkpoint_sha256(provenance['checkpoint_path'])
            metadata['source_checkpoint_hash_after'] = source_hash_after
            metadata['source_checkpoint_immutable'] = source_hash_after == source_hash_before
        except BaseException as hash_error:
            metadata['source_checkpoint_hash_after_error'] = f'{type(hash_error).__name__}: {hash_error}'
        metadata.update({'status': 'failed', 'failure_reason': f'{type(error).__name__}: {error}'})
        _write_json(metadata_path, metadata)
        raise
    finally:
        if env is not None:
            env.close()


def plan_jobs(
    study_path,
    source_run_root,
    output_root,
    checkpoint_selector='best',
    *,
    train_ks=(4, 8),
    batch_size=1024,
    diagnostic_seed=18018,
    max_trace_k=8,
):
    """Plan D2/D3/D4 sources without writing batch or trace artifacts."""

    selector = d1.normalize_checkpoint_selector(checkpoint_selector)
    if selector['selector'] != 'best':
        raise ReevaluationError('M18-D234 currently requires --checkpoint best')
    study, configurations = _configurations(study_path, train_ks)
    jobs = []
    for configuration in configurations:
        train_k = int(configuration.data['factors']['recurrent_compute_budget_K'])
        source_run_dir = make_run_path(
            source_run_root, study.study_id, configuration.config_id, configuration.slug,
            ENVIRONMENT, 0, run_attempt=0,
        )
        provenance = None
        error = None
        try:
            provenance = d1.validate_m18_best_source(source_run_dir)
            config = _resolved_agent_config(provenance['resolved_config'])
            if d1._uniform_train_k(config, label=configuration.config_id) != train_k:
                raise ReevaluationError('Resolved source K does not match M18 declarative config')
            d1.validate_m18_agent_config(config, train_k, label=configuration.config_id)
        except (FileNotFoundError, OSError, ValueError, ReevaluationError) as caught:
            error = str(caught)
        output_dir = _job_output_dir(
            output_root, selector, batch_size, diagnostic_seed, train_k, max_trace_k,
        )
        jobs.append({
            'study': study,
            'configuration': configuration,
            'K_train': train_k,
            'source_run_dir': Path(source_run_dir),
            'provenance': provenance,
            'output_dir': output_dir,
            'batch_root': _trace_root(output_root, selector, batch_size, diagnostic_seed),
            'status': 'planned' if provenance is not None and not output_dir.exists() else (
                'output_exists' if provenance is not None else 'invalid_source'
            ),
            'error': error,
        })
    return jobs


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M18_puzzle_recurrent_compute_scaling/study.yaml')
    parser.add_argument('--source-run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--checkpoint', default='best', help='M18-D234 primary selector; currently best only.')
    parser.add_argument('--train-ks', default='4,8', help='M18 source K values; all 1,2,4,8 are supported.')
    parser.add_argument('--max-trace-k', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--diagnostic-seed', type=int, default=18018)
    parser.add_argument('--slots', default='actor,value,critic')
    parser.add_argument('--save-raw-states', action='store_true')
    parser.add_argument('--diagnostic-code-commit', default=None, help='User-supplied reviewed diagnostic code commit.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.max_trace_k < 1 or args.max_trace_k > 8:
        parser.error('--max-trace-k must be in [1, 8]')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    return args


def main(argv=None):
    args = _args(argv)
    try:
        selector = d1.normalize_checkpoint_selector(args.checkpoint)
        if selector['selector'] != 'best':
            raise ReevaluationError('M18-D234 currently requires --checkpoint best')
        if args.dry_run == args.execute:
            raise ReevaluationError('Exactly one of --dry-run or --execute is required')
        train_ks = _parse_csv_ints(args.train_ks, '--train-ks', allowed=K_VALUES)
        slots = _parse_slots(args.slots)
        if args.execute and not args.diagnostic_code_commit:
            raise ReevaluationError('--execute requires --diagnostic-code-commit from the user-reviewed commit')
        jobs = plan_jobs(
            args.study,
            args.source_run_root,
            args.output_root,
            selector,
            train_ks=train_ks,
            batch_size=args.batch_size,
            diagnostic_seed=args.diagnostic_seed,
            max_trace_k=args.max_trace_k,
        )
        counts = {key: sum(job['status'] == key for job in jobs) for key in ('planned', 'output_exists', 'invalid_source')}
        print(
            f'M18-D234 trace plan: total={len(jobs)} planned={counts["planned"]} '
            f'output_exists={counts["output_exists"]} invalid_source={counts["invalid_source"]}'
        )
        for job in jobs:
            if job['status'] == 'planned':
                print(
                    f'[PLANNED] Ktrain={job["K_train"]} maxTraceK={args.max_trace_k} '
                    f'source_status={job["provenance"]["source_run_status_at_validation"]} '
                    f'batch={job["batch_root"]} output={job["output_dir"]}'
                )
            else:
                print(
                    f'[{job["status"].upper()}] Ktrain={job["K_train"]} '
                    f'{job.get("error") or job["output_dir"]}', file=sys.stderr,
                )
        if args.dry_run:
            return 0 if counts['invalid_source'] == 0 and counts['output_exists'] == 0 else 2
        if counts['invalid_source'] or counts['output_exists']:
            raise ReevaluationError('M18-D234 execute requires every source valid and every output path absent')
        batch_root = jobs[0]['batch_root']
        batch, batch_metadata = _create_fixed_batch(
            batch_root, jobs[0]['provenance'], args.batch_size, args.diagnostic_seed,
        )
        summaries = []
        for job in jobs:
            summaries.append(_run_trace_job(
                job,
                batch,
                batch_metadata,
                max_trace_k=args.max_trace_k,
                slots=slots,
                save_raw_states=args.save_raw_states,
                diagnostic_code_commit=args.diagnostic_code_commit,
            ))
        print(
            f'M18-D234 execute completed: jobs={len(summaries)} batch={batch_root} '
            f'optimizer_updates=0'
        )
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError, ReevaluationError) as error:
        print(f'M18-D234: FAIL: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
