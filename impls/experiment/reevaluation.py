"""Reusable, provenance-checked checkpoint reevaluation infrastructure.

This module intentionally treats reevaluation as a separate experiment object.
It reconstructs an agent from the immutable source run's resolved config,
streams one compact row per episode, and never writes to the source run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .management import config_fingerprint, jsonable
from ..utils.reproducibility import derive_seed


COMMON_EPISODE_SEED_SCHEME = 'common_task_episode_v1'


REEVALUATION_STATUSES = {'running', 'completed', 'failed', 'aborted', 'invalid'}
EPISODE_FIELDS = (
    'study_id',
    'config_id',
    'config_slug',
    'environment',
    'training_seed',
    'checkpoint_step',
    'task_id',
    'task_name',
    'episode_index',
    'evaluation_seed',
    'task_seed',
    'episode_seed',
    'actor_seed',
    'noise_seed',
    'success',
    'episode_return',
    'episode_length',
    'terminated',
    'truncated',
    'paired_episode_id',
    'final_info_json',
)
TASK_SUMMARY_FIELDS = (
    'task_id',
    'task_name',
    'episode_count',
    'success_count',
    'success_rate',
    'success_standard_error',
    'success_wilson_low_95',
    'success_wilson_high_95',
    'return_mean',
    'return_std',
    'episode_length_mean',
    'episode_length_std',
)


class ReevaluationError(ValueError):
    """Raised when a reevaluation source, protocol, or artifact is invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(jsonable(value), file, indent=2, sort_keys=True)
        file.write('\n')


def _read_json(path):
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ReevaluationError(f'Expected JSON mapping: {path}')
    return dict(value)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root=None):
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=all'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {'git_commit': commit, 'git_dirty': bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': None, 'git_dirty': None}


def load_reevaluation_spec(path):
    path = Path(path).resolve()
    with path.open() as file:
        spec = yaml.safe_load(file) or {}
    if not isinstance(spec, dict):
        raise ReevaluationError(f'Reevaluation spec must be a mapping: {path}')
    required = ('reevaluation_id', 'source_study_id', 'source_run_root', 'checkpoint_step', 'environments', 'protocol')
    missing = [key for key in required if key not in spec]
    if missing:
        raise ReevaluationError(f'Spec {path} is missing fields: {missing}')
    protocol = dict(spec['protocol'] or {})
    protocol_defaults = {
        'task_selection': 'all',
        'episodes_per_task': 100,
        'evaluation_seed': 20260819,
        'seed_scheme': COMMON_EPISODE_SEED_SCHEME,
        'eval_temperature': 0.0,
        'eval_gaussian': None,
        'video_episodes': 0,
    }
    for key, value in protocol_defaults.items():
        protocol.setdefault(key, value)
    if protocol['task_selection'] != 'all':
        raise ReevaluationError('The generic runner currently requires task_selection=all')
    if int(protocol['episodes_per_task']) <= 0:
        raise ReevaluationError('episodes_per_task must be positive')
    if int(protocol['video_episodes']) != 0:
        raise ReevaluationError('Post-hoc reevaluation requires video_episodes=0')
    if protocol['seed_scheme'] != COMMON_EPISODE_SEED_SCHEME:
        raise ReevaluationError(f'Unsupported seed_scheme: {protocol["seed_scheme"]!r}')
    spec['protocol'] = protocol
    spec['environments'] = list(spec['environments'])
    spec['training_seeds'] = [int(seed) for seed in spec.get('training_seeds', [0, 1, 2])]
    spec['configs'] = spec.get('configs', 'all')
    spec['_spec_path'] = str(path)
    return spec


def protocol_fingerprint(protocol):
    """Fingerprint only scientific reevaluation protocol fields."""

    fields = {
        'task_selection': protocol['task_selection'],
        'episodes_per_task': int(protocol['episodes_per_task']),
        'evaluation_seed': int(protocol['evaluation_seed']),
        'seed_scheme': protocol['seed_scheme'],
        'eval_temperature': float(protocol['eval_temperature']),
        'eval_gaussian': protocol['eval_gaussian'],
        'video_episodes': int(protocol['video_episodes']),
    }
    return config_fingerprint(fields)


def campaign_root(reeval_root, spec):
    return Path(reeval_root) / spec['source_study_id'] / spec['reevaluation_id']


def _split_config_identity(source_run_dir):
    try:
        config_component = Path(source_run_dir).parent.parent.name
        config_id, config_slug = config_component.split('__', 1)
        environment = Path(source_run_dir).parent.name
        study_id = Path(source_run_dir).parent.parent.parent.name
        seed_component = Path(source_run_dir).name
        if not seed_component.startswith('seed_'):
            raise ValueError
        training_seed = int(seed_component.removeprefix('seed_'))
    except (ValueError, IndexError) as error:
        raise ReevaluationError(f'Cannot parse canonical source run path: {source_run_dir}') from error
    return study_id, config_id, config_slug, environment, training_seed


def _resolved_payload(resolved):
    return {
        'study': resolved.get('study'),
        'configuration': resolved.get('configuration'),
        'algorithm_config': resolved.get('algorithm_config', {}),
    }


def validate_source_run(
    source_run_dir,
    *,
    checkpoint_step,
    expected_study_id=None,
    expected_environment=None,
    check_checkpoint_metadata=True,
):
    """Validate a source run and return immutable provenance information."""

    source_run_dir = Path(source_run_dir).resolve()
    metadata_path = source_run_dir / 'runtime_metadata.json'
    resolved_path = source_run_dir / 'resolved_config.json'
    if not metadata_path.is_file():
        raise ReevaluationError(f'Missing source runtime_metadata.json: {metadata_path}')
    if not resolved_path.is_file():
        raise ReevaluationError(f'Missing source resolved_config.json: {resolved_path}')
    metadata = _read_json(metadata_path)
    resolved = _read_json(resolved_path)
    if metadata.get('status') != 'completed':
        raise ReevaluationError(f'Source run is not completed: {metadata.get("status")!r}')
    if metadata.get('git_dirty') is not False:
        raise ReevaluationError(f'Formal source run is not clean: {metadata.get("git_dirty")!r}')

    study_id, config_id, config_slug, environment, training_seed = _split_config_identity(source_run_dir)
    expected = {
        'study_id': expected_study_id,
        'environment': expected_environment,
        'config_id': config_id,
        'config_slug': config_slug,
        'training_seed': training_seed,
    }
    observed = {
        'study_id': metadata.get('study_id'),
        'environment': metadata.get('environment'),
        'config_id': metadata.get('config_id'),
        'config_slug': metadata.get('config_slug'),
        'training_seed': metadata.get('seed'),
    }
    for key, wanted in expected.items():
        if wanted is not None and str(observed[key]) != str(wanted):
            raise ReevaluationError(
                f'Source path/metadata mismatch for {key}: path={wanted!r}, metadata={observed[key]!r}'
            )
    if metadata.get('run_dir') and Path(metadata['run_dir']).resolve() != source_run_dir:
        raise ReevaluationError('runtime_metadata.run_dir does not match source run directory')
    stored_fingerprint = metadata.get('resolved_config_fingerprint')
    resolved_fingerprint = resolved.get('resolved_config_fingerprint')
    calculated_fingerprint = config_fingerprint(_resolved_payload(resolved))
    if not stored_fingerprint or stored_fingerprint != resolved_fingerprint or stored_fingerprint != calculated_fingerprint:
        raise ReevaluationError(
            'Resolved config fingerprint mismatch: '
            f'metadata={stored_fingerprint!r}, file={resolved_fingerprint!r}, calculated={calculated_fingerprint!r}'
        )

    checkpoint_path = source_run_dir / 'checkpoints' / f'params_{int(checkpoint_step)}.pkl'
    if not checkpoint_path.is_file():
        raise ReevaluationError(f'Missing requested checkpoint: {checkpoint_path}')
    if int(checkpoint_step) == 500000 and checkpoint_path.name != 'params_500000.pkl':
        raise ReevaluationError('M10A-R001 requires exactly params_500000.pkl')
    checkpoint_metadata = {}
    if check_checkpoint_metadata:
        with checkpoint_path.open('rb') as file:
            checkpoint = pickle.load(file)
        if not isinstance(checkpoint, Mapping) or 'agent' not in checkpoint:
            raise ReevaluationError(f'Checkpoint is not a serialized RLC agent: {checkpoint_path}')
        checkpoint_metadata = checkpoint.get('checkpoint_metadata') or {}
        for key, source_key in (
            ('environment', 'environment'),
            ('study_id', 'study_id'),
            ('config_id', 'config_id'),
            ('config_slug', 'config_slug'),
            ('git_commit', 'git_commit'),
        ):
            if checkpoint_metadata.get(key) != metadata.get(source_key):
                raise ReevaluationError(
                    f'Checkpoint metadata mismatch for {key}: '
                    f'checkpoint={checkpoint_metadata.get(key)!r}, source={metadata.get(source_key)!r}'
                )
        if int(checkpoint_metadata.get('seed', -1)) != training_seed:
            raise ReevaluationError('Checkpoint metadata seed does not match source training seed')

    return {
        'source_run_dir': str(source_run_dir),
        'source_study_id': study_id,
        'source_config_id': config_id,
        'source_config_slug': config_slug,
        'source_environment': environment,
        'source_training_seed': training_seed,
        'source_git_commit': metadata.get('git_commit'),
        'source_git_dirty': metadata.get('git_dirty'),
        'source_resolved_config_fingerprint': stored_fingerprint,
        'source_metadata': metadata,
        'resolved_config': resolved,
        'checkpoint_step': int(checkpoint_step),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'checkpoint_metadata': jsonable(checkpoint_metadata),
    }


def _resolved_agent_config(resolved):
    algorithm_config = resolved.get('algorithm_config', {})
    if isinstance(algorithm_config, Mapping) and isinstance(algorithm_config.get('agent'), Mapping):
        return dict(algorithm_config['agent'])
    if isinstance(resolved.get('agent'), Mapping):
        return dict(resolved['agent'])
    raise ReevaluationError('resolved_config.json has no algorithm_config.agent mapping')


def _make_restored_agent(provenance):
    from ..agents import agents
    from ..utils.datasets import GCDataset, HGCDataset, MultiHGCDataset
    from ..utils.env_utils import make_env_and_datasets
    from ..utils.flax_utils import restore_agent

    metadata = provenance['source_metadata']
    resolved = provenance['resolved_config']
    config = _resolved_agent_config(resolved)
    algorithm = metadata.get('algorithm') or config.get('agent_name')
    if algorithm not in agents:
        raise ReevaluationError(f'Unsupported source algorithm for reevaluation: {algorithm!r}')
    if config.get('agent_name') not in (None, algorithm):
        raise ReevaluationError(
            f'Agent mismatch between runtime metadata and resolved config: {algorithm!r} vs {config.get("agent_name")!r}'
        )
    dataset_dir = metadata.get('dataset_dir')
    environment = provenance['source_environment']
    training_seed = provenance['source_training_seed']
    env, raw_train, _ = make_env_and_datasets(
        environment,
        frame_stack=config.get('frame_stack'),
        seed=derive_seed(training_seed, 3),
        dataset_seed=derive_seed(training_seed, 1),
        dataset_dir=dataset_dir,
    )
    dataset_classes = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'MultiHGCDataset': MultiHGCDataset,
    }
    dataset_name = config.get('dataset_class')
    if dataset_name not in dataset_classes:
        raise ReevaluationError(f'Unsupported source dataset_class: {dataset_name!r}')
    train_dataset = dataset_classes[dataset_name](
        raw_train,
        config,
        rng=derive_seed(training_seed, 11),
    )
    example_batch = train_dataset.sample(1)
    if config.get('discrete', False):
        example_batch['actions'] = np.full_like(
            example_batch['actions'], env.action_space.n - 1
        )
    agent = agents[algorithm].create(
        training_seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    restored = restore_agent(agent, provenance['source_run_dir'], provenance['checkpoint_step'])
    leaves = []
    leaves.extend(jax_leaf for jax_leaf in _tree_leaves(restored.network.params))
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise ReevaluationError('Restored checkpoint contains non-finite parameters')
    return restored, env, config, example_batch


def _tree_leaves(value):
    import jax

    return jax.tree_util.tree_leaves(value)


def _restore_probe(agent, example_batch, algorithm, training_seed):
    import jax

    observations = example_batch['observations']
    goals = example_batch.get('high_actor_goals', example_batch.get('actor_goals'))
    if algorithm == 'coghp':
        observations = observations[0]
        if goals is not None and np.asarray(goals).ndim > 1:
            goals = goals[0]
    action = agent.sample_actions(
        observations,
        goals,
        seed=jax.random.PRNGKey(derive_seed(training_seed, 0xA11CE)),
    )
    if not np.all(np.isfinite(np.asarray(action))):
        raise ReevaluationError('Restored action probe is non-finite')


def _wilson_interval(success_count, episode_count, z=1.959963984540054):
    if episode_count <= 0:
        raise ValueError('episode_count must be positive')
    p = success_count / episode_count
    denominator = 1.0 + z * z / episode_count
    center = (p + z * z / (2.0 * episode_count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / episode_count + z * z / (4.0 * episode_count**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def _read_episode_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
            raise ReevaluationError(f'Unexpected episode_results.csv header: {path}')
        rows = list(reader)
    seen = set()
    for row in rows:
        key = (int(row['task_id']), int(row['episode_index']))
        if key in seen:
            raise ReevaluationError(f'Duplicate episode key in {path}: {key}')
        seen.add(key)
        if row['paired_episode_id'] != f'task{key[0]:02d}_ep{key[1]:03d}':
            raise ReevaluationError(f'Invalid paired_episode_id in {path}: {row["paired_episode_id"]}')
    return rows


def _write_task_and_overall_summaries(output_dir, rows, *, task_names, episodes_per_task, checkpoint_step, evaluation_seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row['task_id']), []).append(row)
    task_rows = []
    task_success = {}
    for task_id in sorted(task_names):
        task_rows_for_id = grouped.get(task_id, [])
        if len(task_rows_for_id) != episodes_per_task:
            raise ReevaluationError(
                f'Task {task_id} has {len(task_rows_for_id)} rows; expected {episodes_per_task}'
            )
        success_count = sum(float(row['success']) for row in task_rows_for_id)
        returns = [float(row['episode_return']) for row in task_rows_for_id]
        lengths = [float(row['episode_length']) for row in task_rows_for_id]
        return_mean, return_std = _mean_std(returns)
        length_mean, length_std = _mean_std(lengths)
        success_rate = success_count / episodes_per_task
        wilson_low, wilson_high = _wilson_interval(int(success_count), episodes_per_task)
        task_success[f'task{task_id}'] = success_rate
        task_rows.append({
            'task_id': task_id,
            'task_name': task_names[task_id],
            'episode_count': episodes_per_task,
            'success_count': int(success_count),
            'success_rate': success_rate,
            'success_standard_error': math.sqrt(success_rate * (1.0 - success_rate) / episodes_per_task),
            'success_wilson_low_95': wilson_low,
            'success_wilson_high_95': wilson_high,
            'return_mean': return_mean,
            'return_std': return_std,
            'episode_length_mean': length_mean,
            'episode_length_std': length_std,
        })
    with (Path(output_dir) / 'task_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=TASK_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(task_rows)
    task_count = len(task_rows)
    overall_success = sum(task_success.values()) / task_count
    overall_se = math.sqrt(
        sum(row['success_rate'] * (1.0 - row['success_rate']) / episodes_per_task for row in task_rows)
    ) / task_count
    summary = {
        'status': 'completed',
        'evaluation/overall_success': overall_success,
        'task_count': task_count,
        'total_episodes': len(rows),
        'checkpoint_step': int(checkpoint_step),
        'evaluation_seed': int(evaluation_seed),
        'episodes_per_task': int(episodes_per_task),
        'task_success': task_success,
        'overall_episode_sampling_se': overall_se,
    }
    _write_json(Path(output_dir) / 'summary.json', summary)
    return summary


def _metadata_for_reevaluation(provenance, spec, *, output_dir, assigned_gpu=None, repo_root=None):
    source_meta = provenance['source_metadata']
    protocol = dict(spec['protocol'])
    git_info = _git_metadata(repo_root)
    return {
        'reevaluation_id': spec['reevaluation_id'],
        'status': 'running',
        'start_time': _utc_now(),
        'source_run_dir': provenance['source_run_dir'],
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'source_config_slug': provenance['source_config_slug'],
        'source_environment': provenance['source_environment'],
        'source_training_seed': provenance['source_training_seed'],
        'source_git_commit': provenance['source_git_commit'],
        'source_git_dirty': provenance['source_git_dirty'],
        'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'checkpoint_step': provenance['checkpoint_step'],
        'checkpoint_path': provenance['checkpoint_path'],
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'checkpoint_metadata': provenance['checkpoint_metadata'],
        'reevaluation_git_commit': git_info['git_commit'],
        'reevaluation_git_dirty': git_info['git_dirty'],
        'dataset_dir': source_meta.get('dataset_dir'),
        'ogbench_module': source_meta.get('ogbench_module'),
        'jax_backend': None,
        'jax_devices': [],
        'hostname': socket.gethostname(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'assigned_gpu': None if assigned_gpu is None else str(assigned_gpu),
        'evaluation_protocol': protocol | {
            'task_count': int(spec.get('expected_task_count', 0)),
            'total_episodes': int(spec.get('expected_task_count', 0)) * int(protocol['episodes_per_task']),
        },
        'reevaluation_protocol_fingerprint': protocol_fingerprint(protocol),
        'source_resolved_config_path': str(Path(provenance['source_run_dir']) / 'resolved_config.json'),
        'output_dir': str(Path(output_dir).resolve()),
    }


def _update_status(metadata_path, metadata, status, failure_reason=None):
    metadata = dict(metadata)
    metadata['status'] = status
    metadata['end_time'] = _utc_now()
    if failure_reason is not None:
        metadata['failure_reason'] = str(failure_reason)
    _write_json(metadata_path, metadata)


def run_checkpoint_reevaluation(
    source_run_dir,
    spec,
    *,
    reeval_root,
    resume=False,
    assigned_gpu=None,
    repo_root=None,
):
    """Evaluate one source checkpoint with safe incremental output."""

    provenance = validate_source_run(
        source_run_dir,
        checkpoint_step=spec['checkpoint_step'],
        expected_study_id=spec['source_study_id'],
        expected_environment=spec['environments'][0] if len(spec['environments']) == 1 else None,
    )
    if provenance['source_environment'] not in spec['environments']:
        raise ReevaluationError(
            f'Source environment {provenance["source_environment"]!r} is not in spec environments'
        )
    output_dir = (
        campaign_root(reeval_root, spec)
        / f'{provenance["source_config_id"]}__{provenance["source_config_slug"]}'
        / provenance['source_environment']
        / f'seed_{provenance["source_training_seed"]:03d}'
    )
    metadata_path = output_dir / 'reevaluation_metadata.json'
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f'Reevaluation output exists; use --resume after validating it: {output_dir}'
        )
    if output_dir.exists() and resume and not metadata_path.exists():
        raise ReevaluationError(
            f'Cannot resume an output directory without reevaluation_metadata.json: {output_dir}'
        )
    if output_dir.exists() and resume and metadata_path.exists():
        existing_metadata = _read_json(metadata_path)
        if existing_metadata.get('checkpoint_sha256') != provenance['checkpoint_sha256']:
            raise ReevaluationError('Resume checkpoint SHA256 mismatch')
        if existing_metadata.get('reevaluation_protocol_fingerprint') != protocol_fingerprint(spec['protocol']):
            raise ReevaluationError('Resume reevaluation protocol fingerprint mismatch')
        if existing_metadata.get('status') == 'completed':
            return _read_json(output_dir / 'summary.json')
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_for_reevaluation(
        provenance,
        spec,
        output_dir=output_dir,
        assigned_gpu=assigned_gpu,
        repo_root=repo_root,
    )
    if metadata_path.exists():
        previous = _read_json(metadata_path)
        metadata['start_time'] = previous.get('start_time', metadata['start_time'])
    _write_json(metadata_path, metadata)

    episode_path = output_dir / 'episode_results.csv'
    try:
        rows = _read_episode_rows(episode_path)
        protocol = spec['protocol']
        expected_episodes = int(protocol['episodes_per_task'])
        restored, env, config, example_batch = _make_restored_agent(provenance)
        _restore_probe(
            restored,
            example_batch,
            provenance['source_metadata'].get('algorithm', config.get('agent_name')),
            provenance['source_training_seed'],
        )
        import jax

        metadata['jax_backend'] = jax.default_backend()
        metadata['jax_devices'] = [str(device) for device in jax.devices()]
        task_infos = getattr(env.unwrapped, 'task_infos', None)
        if task_infos is None:
            raise ReevaluationError('Expected task_infos for task_selection=all')
        task_names = {index + 1: str(item['task_name']) for index, item in enumerate(task_infos)}
        expected_task_count = int(spec.get('expected_task_count', len(task_names)))
        if len(task_names) != expected_task_count:
            raise ReevaluationError(
                f'Environment task count {len(task_names)} != expected {expected_task_count}'
            )
        metadata['evaluation_protocol']['task_count'] = len(task_names)
        metadata['evaluation_protocol']['total_episodes'] = len(task_names) * expected_episodes
        _write_json(metadata_path, metadata)

        for row in rows:
            expected_identity = {
                'study_id': provenance['source_study_id'],
                'config_id': provenance['source_config_id'],
                'config_slug': provenance['source_config_slug'],
                'environment': provenance['source_environment'],
                'training_seed': str(provenance['source_training_seed']),
                'checkpoint_step': str(provenance['checkpoint_step']),
                'evaluation_seed': str(protocol['evaluation_seed']),
            }
            for field, expected_value in expected_identity.items():
                if str(row.get(field)) != expected_value:
                    raise ReevaluationError(
                        f'Existing episode row has incompatible {field}: '
                        f'{row.get(field)!r} != {expected_value!r}'
                    )
            task_id = int(row['task_id'])
            if task_id not in task_names or row.get('task_name') != task_names[task_id]:
                raise ReevaluationError('Existing episode row has an invalid task identity')

        existing_keys = {(int(row['task_id']), int(row['episode_index'])) for row in rows}
        expected_keys = {
            (task_id, episode_index)
            for task_id in task_names
            for episode_index in range(expected_episodes)
        }
        if not existing_keys.issubset(expected_keys):
            raise ReevaluationError('Existing episode rows contain keys outside the requested protocol')
        new_file = not episode_path.exists()
        with episode_path.open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
            if new_file:
                writer.writeheader()
                file.flush()
            for task_id in sorted(task_names):
                missing_indices = [
                    index for index in range(expected_episodes)
                    if (task_id, index) not in existing_keys
                ]
                from ..utils.evaluation import evaluate_episodes

                task_records = evaluate_episodes(
                    restored,
                    env,
                    task_id=task_id,
                    task_name=task_names[task_id],
                    config=config,
                    evaluation_seed=int(protocol['evaluation_seed']),
                    episode_indices=missing_indices,
                    eval_temperature=float(protocol['eval_temperature']),
                    eval_gaussian=protocol['eval_gaussian'],
                    seed_scheme=protocol['seed_scheme'],
                )
                for record in task_records:
                    output_record = {
                        'study_id': provenance['source_study_id'],
                        'config_id': provenance['source_config_id'],
                        'config_slug': provenance['source_config_slug'],
                        'environment': provenance['source_environment'],
                        'training_seed': provenance['source_training_seed'],
                        'checkpoint_step': provenance['checkpoint_step'],
                        **record,
                    }
                    writer.writerow(output_record)
                    file.flush()
                    rows.append(output_record)
                    existing_keys.add((task_id, int(record['episode_index'])))
        if len(rows) != len(expected_keys) or existing_keys != expected_keys:
            raise ReevaluationError(
                f'Completed row set has {len(rows)} rows; expected {len(expected_keys)}'
            )
        summary = _write_task_and_overall_summaries(
            output_dir,
            rows,
            task_names=task_names,
            episodes_per_task=expected_episodes,
            checkpoint_step=provenance['checkpoint_step'],
            evaluation_seed=protocol['evaluation_seed'],
        )
        metadata['status'] = 'completed'
        metadata['end_time'] = _utc_now()
        _write_json(metadata_path, metadata)
        return summary
    except KeyboardInterrupt as error:
        _update_status(metadata_path, metadata, 'aborted', error)
        raise
    except BaseException as error:
        _update_status(metadata_path, metadata, 'failed', error)
        raise


def _sample_sd(values):
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_campaign(spec, *, reeval_root, source_runs):
    """Rebuild campaign-level summaries from per-run immutable outputs."""

    root = campaign_root(reeval_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    completed_task_rows = []
    for provenance in source_runs:
        output_dir = (
            root
            / f'{provenance["source_config_id"]}__{provenance["source_config_slug"]}'
            / provenance['source_environment']
            / f'seed_{provenance["source_training_seed"]:03d}'
        )
        metadata_path = output_dir / 'reevaluation_metadata.json'
        summary_path = output_dir / 'summary.json'
        status = 'planned'
        summary = {}
        if metadata_path.exists():
            metadata = _read_json(metadata_path)
            status = metadata.get('status', 'invalid')
            if metadata.get('checkpoint_sha256') != provenance['checkpoint_sha256']:
                status = 'invalid'
        if summary_path.exists():
            summary = _read_json(summary_path)
        manifest.append({
            'study_id': provenance['source_study_id'],
            'reevaluation_id': spec['reevaluation_id'],
            'config_id': provenance['source_config_id'],
            'config_slug': provenance['source_config_slug'],
            'environment': provenance['source_environment'],
            'training_seed': provenance['source_training_seed'],
            'checkpoint_step': provenance['checkpoint_step'],
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            'status': status,
            'overall_success': summary.get('evaluation/overall_success'),
            'output_dir': str(output_dir),
        })
        task_path = output_dir / 'task_summary.csv'
        if status == 'completed' and task_path.exists():
            with task_path.open(newline='') as file:
                for row in csv.DictReader(file):
                    completed_task_rows.append({
                        **row,
                        'config_id': provenance['source_config_id'],
                        'config_slug': provenance['source_config_slug'],
                        'environment': provenance['source_environment'],
                        'training_seed': provenance['source_training_seed'],
                    })
    manifest_fields = (
        'study_id', 'reevaluation_id', 'config_id', 'config_slug', 'environment',
        'training_seed', 'checkpoint_step', 'checkpoint_sha256', 'status',
        'overall_success', 'output_dir',
    )
    with (root / 'manifest.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda row: (row['config_id'], row['training_seed'])))

    config_groups = {}
    for row in manifest:
        config_groups.setdefault((row['config_id'], row['config_slug'], row['environment']), []).append(row)
    config_fields = (
        'config_id', 'config_slug', 'environment', 'number_training_seeds',
        'overall_success_seed0', 'overall_success_seed1', 'overall_success_seed2',
        'overall_success_mean', 'overall_success_population_sd', 'overall_success_sample_sd',
    )
    with (root / 'config_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=config_fields)
        writer.writeheader()
        for (config_id, slug, environment), group in sorted(config_groups.items()):
            values = {int(row['training_seed']): row['overall_success'] for row in group if row['overall_success'] not in (None, '')}
            numeric = [float(value) for value in values.values()]
            writer.writerow({
                'config_id': config_id,
                'config_slug': slug,
                'environment': environment,
                'number_training_seeds': len(numeric),
                'overall_success_seed0': values.get(0, ''),
                'overall_success_seed1': values.get(1, ''),
                'overall_success_seed2': values.get(2, ''),
                'overall_success_mean': statistics.mean(numeric) if numeric else '',
                'overall_success_population_sd': statistics.pstdev(numeric) if len(numeric) > 1 else (0.0 if numeric else ''),
                'overall_success_sample_sd': _sample_sd(numeric) if numeric else '',
            })

    task_groups = {}
    for row in completed_task_rows:
        key = (row['config_id'], row['config_slug'], row['environment'], row['task_id'], row['task_name'])
        task_groups.setdefault(key, {})[int(row['training_seed'])] = float(row['success_rate'])
    task_fields = (
        'config_id', 'config_slug', 'environment', 'task_id', 'task_name',
        'success_seed0', 'success_seed1', 'success_seed2', 'number_training_seeds',
        'success_mean', 'success_population_sd', 'success_sample_sd',
    )
    with (root / 'task_config_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=task_fields)
        writer.writeheader()
        for key, values in sorted(task_groups.items()):
            numeric = list(values.values())
            writer.writerow({
                'config_id': key[0], 'config_slug': key[1], 'environment': key[2],
                'task_id': key[3], 'task_name': key[4],
                'success_seed0': values.get(0, ''), 'success_seed1': values.get(1, ''),
                'success_seed2': values.get(2, ''), 'number_training_seeds': len(numeric),
                'success_mean': statistics.mean(numeric) if numeric else '',
                'success_population_sd': statistics.pstdev(numeric) if len(numeric) > 1 else (0.0 if numeric else ''),
                'success_sample_sd': _sample_sd(numeric) if numeric else '',
            })
    campaign_metadata = {
        'reevaluation_id': spec['reevaluation_id'],
        'source_study_id': spec['source_study_id'],
        'checkpoint_step': spec['checkpoint_step'],
        'protocol': spec['protocol'],
        'protocol_fingerprint': protocol_fingerprint(spec['protocol']),
        'source_run_count': len(source_runs),
        'completed_run_count': sum(row['status'] == 'completed' for row in manifest),
        'manifest_path': str(root / 'manifest.csv'),
        'generated_at': _utc_now(),
    }
    _write_json(root / 'campaign_metadata.json', campaign_metadata)
    return root
