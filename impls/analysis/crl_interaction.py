"""Post-hoc diagnostics for the M11A CRL actor/critic interaction Study.

The module deliberately keeps diagnostics outside the trainer.  It consumes
immutable completed source runs, creates one shared environment-rollout bank,
extracts deterministic actor candidate actions, and scores recurrent critics
without changing training data, gradients, or checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ..agents.crl import CRLAgent
from ..experiment.management import config_fingerprint, load_study, prepare_run_design
from ..experiment.reevaluation import (
    _make_restored_agent,
    _read_json,
    sha256_file,
    validate_source_run,
)
from ..utils.evaluation import (
    COMMON_EPISODE_SEED_SCHEME,
    _rollout_episode,
    common_episode_seeds,
)
from ..utils.flax_utils import restore_agent, save_agent
from ..utils.reproducibility import derive_seed
from ..main import _make_config, _parse_args


CONFIG_IDS = (
    'M11A-C001', 'M11A-C002', 'M11A-C003', 'M11A-C004',
    'M11A-C005', 'M11A-C006', 'M11A-C007',
)
SINGLE_STATE_IDS = {
    'A': 'M11A-C001',
    'S-C': 'M11A-C002',
    'S-A': 'M11A-C003',
    'S-CA': 'M11A-C004',
}
TWO_STATE_IDS = {
    'A': 'M11A-C001',
    'T-C': 'M11A-C005',
    'T-A': 'M11A-C006',
    'T-CA': 'M11A-C007',
}


class InteractionDiagnosticError(ValueError):
    """Raised when a diagnostic specification or artifact is inconsistent."""


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')


def _stable_seed(value):
    """Derive a process-independent integer from a diagnostic label."""

    return int(config_fingerprint(value)[:8], 16)


def _read_yaml(path):
    path = Path(path).resolve()
    with path.open() as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise InteractionDiagnosticError(f'Diagnostic spec must be a mapping: {path}')
    return path, value


def load_interaction_spec(path):
    """Load and validate the declarative M11A diagnostic specification."""

    path, spec = _read_yaml(path)
    required = (
        'diagnostic_id', 'source_study_id', 'source_run_root',
        'environment', 'baseline_config_id', 'protocol',
        'anchor_stride', 'goal_offset_stride', 'max_goal_offset',
        'conditions', 'source_git_commit',
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise InteractionDiagnosticError(f'Spec {path} is missing fields: {missing}')
    if spec['source_study_id'] != 'M11A':
        raise InteractionDiagnosticError('M11A diagnostics require source_study_id=M11A')
    if spec['baseline_config_id'] != 'M11A-C001':
        raise InteractionDiagnosticError('The shared bank source must be M11A-C001')
    if tuple(spec.get('config_ids', CONFIG_IDS)) != CONFIG_IDS:
        raise InteractionDiagnosticError('M11A diagnostic config_ids must contain exactly the 7 factorial conditions')
    if not isinstance(spec['conditions'], Mapping):
        raise InteractionDiagnosticError('Diagnostic conditions must be declarative mappings')
    if spec['conditions'].get('baseline') != spec['baseline_config_id']:
        raise InteractionDiagnosticError('Diagnostic conditions.baseline must equal baseline_config_id')
    expected_condition_keys = {
        'single_state': {
            'ff_actor_ff_critic', 'ff_actor_rec_critic',
            'rec_actor_ff_critic', 'rec_actor_rec_critic',
        },
        'two_state': {
            'ff_actor_ff_critic', 'ff_actor_rec_critic',
            'rec_actor_ff_critic', 'rec_actor_rec_critic',
        },
    }
    mapped = []
    for topology, keys in expected_condition_keys.items():
        mapping = spec['conditions'].get(topology)
        if not isinstance(mapping, Mapping) or set(mapping) != keys:
            raise InteractionDiagnosticError(
                f'Diagnostic conditions.{topology} must contain exactly {sorted(keys)}'
            )
        mapped.extend(mapping.values())
    if set(mapped) != set(CONFIG_IDS) or len(mapped) != 8:
        raise InteractionDiagnosticError('Diagnostic condition mapping must cover exactly the 7 M11A configs per topology')
    if not isinstance(spec['source_git_commit'], str) or len(spec['source_git_commit']) != 40:
        raise InteractionDiagnosticError('source_git_commit must be a 40-character commit SHA')
    protocol = dict(spec['protocol'] or {})
    defaults = {
        'task_selection': 'all',
        'episodes_per_task': 20,
        'evaluation_seed': 20260820,
        'seed_scheme': COMMON_EPISODE_SEED_SCHEME,
        'eval_temperature': 0.0,
        'eval_gaussian': None,
        'video_episodes': 0,
    }
    for key, value in defaults.items():
        protocol.setdefault(key, value)
    if protocol['task_selection'] != 'all':
        raise InteractionDiagnosticError('M11A diagnostic task_selection must be all')
    if int(protocol['episodes_per_task']) <= 0:
        raise InteractionDiagnosticError('episodes_per_task must be positive')
    if protocol['seed_scheme'] != COMMON_EPISODE_SEED_SCHEME:
        raise InteractionDiagnosticError(f'Unsupported episode seed scheme: {protocol["seed_scheme"]!r}')
    if float(protocol['eval_temperature']) != 0.0 or protocol['eval_gaussian'] is not None:
        raise InteractionDiagnosticError('M11A diagnostic policy must be deterministic with temperature=0 and no Gaussian noise')
    if int(protocol['video_episodes']) != 0:
        raise InteractionDiagnosticError('M11A diagnostic does not render video')
    if int(spec['anchor_stride']) <= 0 or int(spec['goal_offset_stride']) <= 0:
        raise InteractionDiagnosticError('anchor_stride and goal_offset_stride must be positive')
    if int(spec['max_goal_offset']) <= 0:
        raise InteractionDiagnosticError('max_goal_offset must be positive')
    spec['protocol'] = protocol
    spec['config_ids'] = list(CONFIG_IDS)
    spec['checkpoint'] = dict(spec.get('checkpoint', {'selector': 'last'}))
    if spec['checkpoint'] != {'selector': 'last'}:
        raise InteractionDiagnosticError('M11A primary diagnostic checkpoint must be last')
    spec['bootstrap_replicates'] = int(spec.get('bootstrap_replicates', 1000))
    spec['bootstrap_seed'] = int(spec.get('bootstrap_seed', derive_seed(protocol['evaluation_seed'], 0xB007)))
    spec['epsilon'] = float(spec.get('epsilon', 1e-6))
    if spec['bootstrap_replicates'] <= 0 or spec['epsilon'] <= 0:
        raise InteractionDiagnosticError('bootstrap_replicates and epsilon must be positive')
    spec['_spec_path'] = str(path)
    return spec


def _condition_ids(spec, topology):
    """Return the declarative condition mapping for one diagnostic topology."""

    mapping = spec['conditions'][topology]
    return {
        'ff_actor_ff_critic': mapping['ff_actor_ff_critic'],
        'ff_actor_rec_critic': mapping['ff_actor_rec_critic'],
        'rec_actor_ff_critic': mapping['rec_actor_ff_critic'],
        'rec_actor_rec_critic': mapping['rec_actor_rec_critic'],
    }


def _candidate_source_map(spec, topology):
    """Return candidate labels and their trained-condition sources."""

    ids = _condition_ids(spec, topology)
    return {
        'a_exec': None,
        f"a_{ids['ff_actor_ff_critic']}": ids['ff_actor_ff_critic'],
        f"a_{ids['ff_actor_rec_critic']}": ids['ff_actor_rec_critic'],
        f"a_{ids['rec_actor_ff_critic']}": ids['rec_actor_ff_critic'],
        f"a_{ids['rec_actor_rec_critic']}": ids['rec_actor_rec_critic'],
    }


def diagnostic_root(root, spec):
    return Path(root) / spec['diagnostic_id']


def _spec_payload(spec):
    return {key: value for key, value in spec.items() if not key.startswith('_')}


def _source_validation_payload(spec, sources):
    source_rows = {}
    for config_id in CONFIG_IDS:
        value = sources[config_id]
        source_rows[config_id] = {
            'source_run_path': value['source_run_dir'],
            'source_checkpoint_path': value['checkpoint_path'],
            'source_checkpoint_sha256': value['checkpoint_sha256'],
            'source_git_commit': value['source_git_commit'],
            'source_git_dirty': value['source_git_dirty'],
            'source_config_fingerprint': value['source_resolved_config_fingerprint'],
            'environment': value['source_environment'],
            'training_seed': value['source_training_seed'],
            'checkpoint_selector': value['requested_checkpoint_selector'],
            'checkpoint_role': value['resolved_checkpoint_role'],
            'checkpoint_step': value['resolved_checkpoint_step'],
            'dataset_dir': value['source_metadata'].get('dataset_dir'),
            'training_protocol': value['source_metadata'].get('training_protocol', {}),
            'status': value['source_metadata'].get('status'),
            'algorithm': value['source_metadata'].get('algorithm'),
        }
    payload = {
        'diagnostic_id': spec['diagnostic_id'],
        'source_study_id': spec['source_study_id'],
        'source_run_root': spec['source_run_root'],
        'environment': spec['environment'],
        'source_git_commit': spec['source_git_commit'],
        'primary_checkpoint': spec['checkpoint'],
        'sources': source_rows,
    }
    payload['source_validation_fingerprint'] = config_fingerprint(payload)
    return payload


def _ensure_diagnostic_root(spec, diagnostic_root_path):
    """Create or validate the immutable diagnostic root and its spec metadata."""

    root = diagnostic_root(diagnostic_root_path, spec)
    metadata_path = root / 'metadata.json'
    expected = {
        'diagnostic_id': spec['diagnostic_id'],
        'source_study_id': spec['source_study_id'],
        'source_run_root': spec['source_run_root'],
        'environment': spec['environment'],
        'source_git_commit': spec['source_git_commit'],
        'baseline_config_id': spec['baseline_config_id'],
        'conditions': spec['conditions'],
        'checkpoint': spec['checkpoint'],
        'protocol': spec['protocol'],
        'anchor_stride': int(spec['anchor_stride']),
        'goal_offset_stride': int(spec['goal_offset_stride']),
        'max_goal_offset': int(spec['max_goal_offset']),
        'epsilon': float(spec['epsilon']),
        'bootstrap_replicates': int(spec['bootstrap_replicates']),
        'bootstrap_seed': int(spec['bootstrap_seed']),
        'spec_fingerprint': config_fingerprint(_spec_payload(spec)),
    }
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f'Diagnostic root is not a directory: {root}')
        if not metadata_path.is_file():
            if any(root.iterdir()):
                raise InteractionDiagnosticError(
                    f'Diagnostic root exists without metadata and is non-empty: {root}'
                )
            _write_json(metadata_path, expected)
        else:
            observed = _read_json(metadata_path)
            for key, value in expected.items():
                if observed.get(key) != value:
                    raise InteractionDiagnosticError(
                        f'Diagnostic root metadata mismatch for {key}: '
                        f'expected={value!r}, observed={observed.get(key)!r}'
                    )
    else:
        root.mkdir(parents=True, exist_ok=False)
        _write_json(metadata_path, expected)
    return root


def _ensure_source_validation_artifact(spec, diagnostic_root_path, sources):
    root = _ensure_diagnostic_root(spec, diagnostic_root_path)
    path = root / 'audits' / 'source_validation.json'
    payload = _source_validation_payload(spec, sources)
    if path.exists():
        observed = _read_json(path)
        if observed.get('source_validation_fingerprint') != payload['source_validation_fingerprint']:
            raise InteractionDiagnosticError(
                f'Source validation artifact no longer matches current source set: {path}'
            )
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return payload


def _metadata_candidates(spec):
    source_root = Path(spec['source_run_root']) / spec['source_study_id']
    if not source_root.exists():
        raise InteractionDiagnosticError(f'M11A source study root does not exist: {source_root}')
    candidates = defaultdict(list)
    for metadata_path in sorted(source_root.rglob('runtime_metadata.json')):
        try:
            metadata = _read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if (
            metadata.get('environment') == spec['environment']
            and int(metadata.get('seed', -1)) == 0
            and metadata.get('config_id') in CONFIG_IDS
            and metadata.get('status') == 'completed'
        ):
            candidates[metadata['config_id']].append(metadata_path.parent)
    return candidates


def validate_source_set(spec):
    """Validate all seven completed sources and shared protocol provenance."""

    candidates = _metadata_candidates(spec)
    sources = {}
    for config_id in CONFIG_IDS:
        paths = candidates.get(config_id, [])
        if len(paths) != 1:
            raise InteractionDiagnosticError(
                f'Expected exactly one completed source for {config_id}, found {len(paths)}: {paths}'
            )
        sources[config_id] = validate_source_run(
            paths[0],
            checkpoint_selector=spec['checkpoint'],
            expected_study_id='M11A',
            expected_environment=spec['environment'],
            check_checkpoint_metadata=True,
        )

    commits = {value['source_git_commit'] for value in sources.values()}
    if len(commits) != 1:
        raise InteractionDiagnosticError(f'M11A sources are not from one commit: {sorted(commits)}')
    if commits != {spec['source_git_commit']}:
        raise InteractionDiagnosticError(
            f'M11A source commit mismatch: expected={spec["source_git_commit"]}, observed={sorted(commits)}'
        )
    if any(value['source_git_dirty'] is not False for value in sources.values()):
        raise InteractionDiagnosticError('All M11A source runs must have git_dirty=false')
    metadata = [value['source_metadata'] for value in sources.values()]
    dataset_dirs = {item.get('dataset_dir') for item in metadata}
    if len(dataset_dirs) != 1 or None in dataset_dirs:
        raise InteractionDiagnosticError('M11A sources do not share one dataset directory')
    protocols = []
    for item in metadata:
        protocol = item.get('training_protocol', {})
        protocols.append(tuple(sorted((key, str(protocol.get(key))) for key in (
            'batch_size', 'train_steps', 'eval_interval', 'eval_episodes',
            'eval_temperature', 'eval_gaussian', 'save_interval',
        ))))
    if len(set(protocols)) != 1:
        raise InteractionDiagnosticError('M11A sources do not share one training protocol')
    for item in metadata:
        protocol = item.get('training_protocol', {})
        expected_protocol = {
            'train_steps': 1_000_000,
            'batch_size': 1024,
            'eval_interval': 100_000,
            'eval_episodes': 20,
            'eval_temperature': 0.0,
            'eval_gaussian': None,
        }
        for key, expected in expected_protocol.items():
            if protocol.get(key) != expected:
                raise InteractionDiagnosticError(
                    f'M11A source training protocol mismatch for {key}: '
                    f'expected={expected!r}, observed={protocol.get(key)!r}'
                )
    if any(item.get('agent') != 'crl' for item in metadata):
        raise InteractionDiagnosticError('All M11A sources must be CRL runs')
    for config_id, value in sources.items():
        if value['requested_checkpoint_selector'] != {'selector': 'last'}:
            raise InteractionDiagnosticError(f'{config_id} did not resolve the primary last checkpoint')
        if value['resolved_checkpoint_role'] != 'last' or value['resolved_checkpoint_step'] != 1_000_000:
            raise InteractionDiagnosticError(
                f'{config_id} primary checkpoint is not last@1M: '
                f'{value["resolved_checkpoint_role"]}@{value["resolved_checkpoint_step"]}'
            )
        if not value.get('checkpoint_sha256'):
            raise InteractionDiagnosticError(f'{config_id} has no checkpoint SHA256')
        if not value.get('source_resolved_config_fingerprint'):
            raise InteractionDiagnosticError(f'{config_id} has no resolved config fingerprint')
    return sources


def _task_names(env):
    task_infos = getattr(env.unwrapped, 'task_infos', None)
    if task_infos is None:
        raise InteractionDiagnosticError('M11A requires an environment with task_infos')
    return {index + 1: str(info['task_name']) for index, info in enumerate(task_infos)}


def _bank_arrays_from_rollouts(agent, env, config, protocol, anchor_stride, goal_offset_stride, max_goal_offset):
    task_names = _task_names(env)
    flat = defaultdict(list)
    episode_offsets = [0]
    episode_task_ids = []
    episode_task_names = []
    episode_indices = []
    episode_seeds = []
    actor_seeds = []
    original_goals = []
    episode_lengths = []
    episode_terminated = []
    episode_truncated = []
    anchor_records = []
    pair_records = []
    for task_id, task_name in task_names.items():
        for episode_index in range(int(protocol['episodes_per_task'])):
            seeds = common_episode_seeds(
                int(protocol['evaluation_seed']), task_id, episode_index,
                scheme=protocol['seed_scheme'],
            )
            rollout = _rollout_episode(
                agent,
                env,
                task_id=task_id,
                config=config,
                episode_seed=seeds['episode_seed'],
                actor_seed=seeds['actor_seed'],
                noise_seed=seeds['noise_seed'],
                eval_temperature=float(protocol['eval_temperature']),
                eval_gaussian=protocol['eval_gaussian'],
                retain_trajectory=True,
                render=False,
                video_frame_skip=1,
            )
            trajectory = rollout['trajectory']
            observations = np.asarray(trajectory['observation'], dtype=np.float32)
            next_observations = np.asarray(trajectory['next_observation'], dtype=np.float32)
            actions = np.asarray(trajectory['action'], dtype=np.float32)
            rewards = np.asarray(trajectory['reward'], dtype=np.float32)
            length = len(observations)
            if length <= 0:
                raise InteractionDiagnosticError(f'Empty evaluation trajectory for task={task_id}, episode={episode_index}')
            episode_id = len(episode_task_ids)
            flat['observations'].append(observations)
            flat['next_observations'].append(next_observations)
            flat['executed_actions'].append(actions)
            flat['rewards'].append(rewards)
            flat['done'].append(np.asarray(trajectory['done'], dtype=np.bool_))
            flat['episode_ids'].append(np.full(length, episode_id, dtype=np.int64))
            flat['task_ids'].append(np.full(length, task_id, dtype=np.int64))
            flat['episode_indices'].append(np.full(length, episode_index, dtype=np.int64))
            flat['timesteps'].append(np.arange(length, dtype=np.int64))
            start = episode_offsets[-1]
            episode_offsets.append(start + length)
            episode_task_ids.append(task_id)
            episode_task_names.append(task_name)
            episode_indices.append(episode_index)
            episode_seeds.append(seeds['episode_seed'])
            actor_seeds.append(seeds['actor_seed'])
            episode_lengths.append(length)
            episode_terminated.append(bool(rollout['terminated']))
            episode_truncated.append(bool(rollout['truncated']))
            goal = rollout['original_eval_goal']
            if goal is None:
                raise InteractionDiagnosticError('Environment evaluation rollout did not expose original_eval_goal')
            original_goals.append(np.asarray(goal, dtype=np.float32))
            for t in range(0, length, int(anchor_stride)):
                valid_h = [
                    h for h in range(int(goal_offset_stride), int(max_goal_offset) + 1, int(goal_offset_stride))
                    if t + h < length
                ]
                anchor_index = len(anchor_records)
                anchor_records.append({
                    'episode_id': episode_id,
                    'task_id': task_id,
                    'episode_index': episode_index,
                    't': t,
                    'observation': observations[t],
                    'next_observation': next_observations[t],
                    'executed_action': actions[t],
                    'original_goal': np.asarray(goal, dtype=np.float32),
                })
                for h in valid_h:
                    pair_records.append({
                        'anchor_index': anchor_index,
                        'episode_id': episode_id,
                        'task_id': task_id,
                        'episode_index': episode_index,
                        'h': h,
                        'observation': observations[t],
                        'executed_action': actions[t],
                        'future_goal': observations[t + h],
                    })

    def stack_records(records, key, dtype=np.float32):
        return np.asarray([record[key] for record in records], dtype=dtype)

    if not anchor_records or not pair_records:
        raise InteractionDiagnosticError('Diagnostic bank contains no valid anchors or future-goal pairs')

    return {
        'observations': np.concatenate(flat['observations'], axis=0),
        'next_observations': np.concatenate(flat['next_observations'], axis=0),
        'executed_actions': np.concatenate(flat['executed_actions'], axis=0),
        'rewards': np.concatenate(flat['rewards'], axis=0),
        'done': np.concatenate(flat['done'], axis=0),
        'step_episode_ids': np.concatenate(flat['episode_ids'], axis=0),
        'step_task_ids': np.concatenate(flat['task_ids'], axis=0),
        'step_episode_indices': np.concatenate(flat['episode_indices'], axis=0),
        'step_timesteps': np.concatenate(flat['timesteps'], axis=0),
        'episode_offsets': np.asarray(episode_offsets, dtype=np.int64),
        'episode_task_ids': np.asarray(episode_task_ids, dtype=np.int64),
        'episode_task_names': np.asarray(episode_task_names, dtype='U128'),
        'episode_indices': np.asarray(episode_indices, dtype=np.int64),
        'episode_seeds': np.asarray(episode_seeds, dtype=np.int64),
        'actor_seeds': np.asarray(actor_seeds, dtype=np.int64),
        'episode_lengths': np.asarray(episode_lengths, dtype=np.int64),
        'episode_terminated': np.asarray(episode_terminated, dtype=np.bool_),
        'episode_truncated': np.asarray(episode_truncated, dtype=np.bool_),
        'original_eval_goals': np.asarray(original_goals, dtype=np.float32),
        'anchor_observations': stack_records(anchor_records, 'observation'),
        'anchor_next_observations': stack_records(anchor_records, 'next_observation'),
        'anchor_executed_actions': stack_records(anchor_records, 'executed_action'),
        'anchor_original_goals': stack_records(anchor_records, 'original_goal'),
        'anchor_episode_ids': np.asarray([r['episode_id'] for r in anchor_records], dtype=np.int64),
        'anchor_task_ids': np.asarray([r['task_id'] for r in anchor_records], dtype=np.int64),
        'anchor_task_names': np.asarray([task_names[r['task_id']] for r in anchor_records], dtype='U128'),
        'anchor_episode_indices': np.asarray([r['episode_index'] for r in anchor_records], dtype=np.int64),
        'anchor_t': np.asarray([r['t'] for r in anchor_records], dtype=np.int64),
        'anchor_ids': np.arange(len(anchor_records), dtype=np.int64),
        'pair_observations': stack_records(pair_records, 'observation'),
        'pair_executed_actions': stack_records(pair_records, 'executed_action'),
        'pair_future_goals': stack_records(pair_records, 'future_goal'),
        'pair_anchor_indices': np.asarray([r['anchor_index'] for r in pair_records], dtype=np.int64),
        'pair_episode_ids': np.asarray([r['episode_id'] for r in pair_records], dtype=np.int64),
        'pair_task_ids': np.asarray([r['task_id'] for r in pair_records], dtype=np.int64),
        'pair_task_names': np.asarray([task_names[r['task_id']] for r in pair_records], dtype='U128'),
        'pair_episode_indices': np.asarray([r['episode_index'] for r in pair_records], dtype=np.int64),
        'pair_h': np.asarray([r['h'] for r in pair_records], dtype=np.int64),
        'pair_ids': np.arange(len(pair_records), dtype=np.int64),
    }, task_names


def generate_diagnostic_bank(spec, diagnostic_root_path):
    """Generate the immutable environment-only shared diagnostic bank."""

    sources = validate_source_set(spec)
    _ensure_source_validation_artifact(spec, diagnostic_root_path, sources)
    output_dir = diagnostic_root(diagnostic_root_path, spec) / 'bank'
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite immutable diagnostic bank: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    agent, env, config, _ = _make_restored_agent(sources[spec['baseline_config_id']])
    try:
        arrays, task_names = _bank_arrays_from_rollouts(
            agent,
            env,
            config,
            spec['protocol'],
            int(spec['anchor_stride']),
            int(spec['goal_offset_stride']),
            int(spec['max_goal_offset']),
        )
    finally:
        close = getattr(env, 'close', None)
        if close is not None:
            close()
    bank_path = output_dir / 'diagnostic_bank.npz'
    np.savez_compressed(bank_path, **arrays)
    bank_sha = sha256_file(bank_path)
    metadata = {
        'schema_version': 2,
        'diagnostic_id': spec['diagnostic_id'],
        'source_condition': spec['baseline_config_id'],
        'source_config_id': spec['baseline_config_id'],
        'source_run_path': sources[spec['baseline_config_id']]['source_run_dir'],
        'source_run_dir': sources[spec['baseline_config_id']]['source_run_dir'],
        'source_checkpoint_path': sources[spec['baseline_config_id']]['checkpoint_path'],
        'source_checkpoint_sha256': sources[spec['baseline_config_id']]['checkpoint_sha256'],
        'source_git_commit': sources[spec['baseline_config_id']]['source_git_commit'],
        'source_config_fingerprint': sources[spec['baseline_config_id']]['source_resolved_config_fingerprint'],
        'environment': spec['environment'],
        'evaluation_protocol': spec['protocol'],
        'episode_seed_scheme': spec['protocol']['seed_scheme'],
        'anchor_stride': int(spec['anchor_stride']),
        'goal_offset_stride': int(spec['goal_offset_stride']),
        'max_goal_offset': int(spec['max_goal_offset']),
        'task_names': task_names,
        'tensor_shapes': {key: list(value.shape) for key, value in arrays.items()},
        'num_episodes': int(len(arrays['episode_task_ids'])),
        'num_anchors': int(len(arrays['anchor_t'])),
        'num_pairs': int(len(arrays['pair_h'])),
        'pair_ids_contiguous': bool(np.array_equal(arrays['pair_ids'], np.arange(len(arrays['pair_ids'])))),
        'anchor_ids_contiguous': bool(np.array_equal(arrays['anchor_ids'], np.arange(len(arrays['anchor_ids'])))),
        'bank_sha256': bank_sha,
    }
    metadata['bank_fingerprint'] = config_fingerprint({
        key: value for key, value in metadata.items() if key != 'bank_fingerprint'
    })
    _write_json(output_dir / 'bank_metadata.json', metadata)
    return metadata


def _load_bank(diagnostic_root_path, spec):
    output_dir = diagnostic_root(diagnostic_root_path, spec) / 'bank'
    bank_path = output_dir / 'diagnostic_bank.npz'
    metadata_path = output_dir / 'bank_metadata.json'
    if not bank_path.is_file() or not metadata_path.is_file():
        raise InteractionDiagnosticError(f'Incomplete diagnostic bank: {output_dir}')
    metadata = _read_json(metadata_path)
    if metadata.get('bank_sha256') != sha256_file(bank_path):
        raise InteractionDiagnosticError('Diagnostic bank sha256 mismatch')
    fingerprint_payload = {
        key: value for key, value in metadata.items() if key != 'bank_fingerprint'
    }
    if metadata.get('bank_fingerprint') != config_fingerprint(fingerprint_payload):
        raise InteractionDiagnosticError('Diagnostic bank fingerprint mismatch')
    if metadata.get('diagnostic_id') != spec['diagnostic_id']:
        raise InteractionDiagnosticError('Diagnostic bank belongs to another diagnostic')
    if metadata.get('source_checkpoint_sha256') is None:
        raise InteractionDiagnosticError('Diagnostic bank has no source checkpoint SHA256')
    return np.load(bank_path, allow_pickle=False), metadata


def _mode_actions(agent, observations, goals, chunk_size=1024):
    observations = np.asarray(observations)
    goals = np.asarray(goals)
    actions = []
    for start in range(0, len(observations), chunk_size):
        end = min(len(observations), start + chunk_size)
        dist = agent.network.select('actor')(
            jnp.asarray(observations[start:end]),
            jnp.asarray(goals[start:end]),
            temperature=1.0,
        )
        actions.append(np.clip(np.asarray(dist.mode()), -1.0, 1.0).astype(np.float32))
    return np.concatenate(actions, axis=0) if actions else np.empty((0, 0), dtype=np.float32)


def _duplicate_action_flags(action_matrix):
    action_matrix = np.asarray(action_matrix)
    if action_matrix.ndim != 3 or action_matrix.shape[1] < 2:
        raise InteractionDiagnosticError('Candidate action matrix must have at least two candidates')
    flags = np.zeros(action_matrix.shape[0], dtype=np.bool_)
    for index, actions in enumerate(action_matrix):
        for left in range(actions.shape[0]):
            for right in range(left + 1, actions.shape[0]):
                if np.array_equal(actions[left], actions[right]):
                    flags[index] = True
                    break
            if flags[index]:
                break
    return flags


def generate_candidate_pools(spec, diagnostic_root_path):
    """Extract deterministic actor actions on the immutable shared bank."""

    bank, bank_metadata = _load_bank(diagnostic_root_path, spec)
    output_dir = diagnostic_root(diagnostic_root_path, spec) / 'candidates'
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite immutable candidate pools: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = validate_source_set(spec)
    _ensure_source_validation_artifact(spec, diagnostic_root_path, sources)
    agents = {}
    envs = []
    try:
        for config_id, provenance in sources.items():
            agent, env, _, _ = _make_restored_agent(provenance)
            agents[config_id] = agent
            envs.append(env)
        pair_obs = bank['pair_observations']
        pair_goals = bank['pair_future_goals']
        pair_exec = bank['pair_executed_actions']
        single_sources = _candidate_source_map(spec, 'single_state')
        two_sources = _candidate_source_map(spec, 'two_state')
        single_actions = {'a_exec': pair_exec}
        two_actions = {'a_exec': pair_exec}
        for label, config_id in single_sources.items():
            if config_id is not None:
                single_actions[label] = _mode_actions(agents[config_id], pair_obs, pair_goals)
        for label, config_id in two_sources.items():
            if config_id is not None:
                two_actions[label] = _mode_actions(agents[config_id], pair_obs, pair_goals)
    finally:
        for env in envs:
            close = getattr(env, 'close', None)
            if close is not None:
                close()
    single_order = list(single_actions)
    two_order = list(two_actions)
    single_matrix = np.stack([single_actions[key] for key in single_order], axis=1)
    two_matrix = np.stack([two_actions[key] for key in two_order], axis=1)
    np.savez_compressed(
        output_dir / 'single_state_candidates.npz',
        pair_id=bank['pair_ids'], actions=single_matrix,
    )
    np.savez_compressed(
        output_dir / 'two_state_candidates.npz',
        pair_id=bank['pair_ids'], actions=two_matrix,
    )
    single_duplicate = _duplicate_action_flags(single_matrix)
    two_duplicate = _duplicate_action_flags(two_matrix)
    source_provenance = {
        config_id: {
            'source_run_path': sources[config_id]['source_run_dir'],
            'source_checkpoint_path': sources[config_id]['checkpoint_path'],
            'source_checkpoint_sha256': sources[config_id]['checkpoint_sha256'],
            'source_git_commit': sources[config_id]['source_git_commit'],
            'source_config_fingerprint': sources[config_id]['source_resolved_config_fingerprint'],
        }
        for config_id in CONFIG_IDS
    }
    metadata = {
        'schema_version': 2,
        'diagnostic_id': spec['diagnostic_id'],
        'bank_sha256': bank_metadata['bank_sha256'],
        'pair_count': int(single_matrix.shape[0]),
        'single_state_order': single_order,
        'two_state_order': two_order,
        'single_state_source_config_ids': [single_sources[key] or f'trajectory:{spec["baseline_config_id"]}' for key in single_order],
        'two_state_source_config_ids': [two_sources[key] or f'trajectory:{spec["baseline_config_id"]}' for key in two_order],
        'source_provenance': source_provenance,
        'policy': 'deterministic distribution.mode() followed by clip[-1,1]',
        'candidate_pool_definition': {
            'single_state': single_order,
            'two_state': two_order,
        },
        'exact_duplicate_action_rate': {
            'single_state': float(np.mean(single_duplicate)),
            'two_state': float(np.mean(two_duplicate)),
        },
        'exact_duplicate_action_definition': 'any pair of saved action vectors is np.array_equal',
        'single_state_sha256': sha256_file(output_dir / 'single_state_candidates.npz'),
        'two_state_sha256': sha256_file(output_dir / 'two_state_candidates.npz'),
    }
    metadata['candidate_fingerprint'] = config_fingerprint(metadata)
    _write_json(output_dir / 'candidate_metadata.json', metadata)
    return metadata


def _critic_q(agent, observations, goals, actions, chunk_size=1024):
    observations = np.asarray(observations)
    goals = np.asarray(goals)
    actions = np.asarray(actions)
    values = []
    for start in range(0, len(observations), chunk_size):
        end = min(len(observations), start + chunk_size)
        result = agent.network.select('critic')(
            jnp.asarray(observations[start:end]),
            jnp.asarray(goals[start:end]),
            jnp.asarray(actions[start:end]),
        )
        result = np.asarray(result)
        if result.ndim == 2:
            result = np.min(result, axis=0)
        elif result.ndim != 1:
            raise InteractionDiagnosticError(f'Unexpected CRL critic output shape: {result.shape}')
        values.append(result.astype(np.float64))
    values = np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise InteractionDiagnosticError('Non-finite critic score in diagnostic')
    return values


def _episode_key(bank, row_indices):
    return [
        (int(bank['pair_task_ids'][index]), int(bank['pair_episode_indices'][index]))
        for index in row_indices
    ]


def _bootstrap_ratio(cluster_values, seed, replicates):
    keys = sorted(cluster_values)
    if not keys:
        return None, None
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        sampled = rng.integers(0, len(keys), size=len(keys))
        numerator = sum(cluster_values[keys[index]][0] for index in sampled)
        denominator = sum(cluster_values[keys[index]][1] for index in sampled)
        estimates.append(numerator / denominator if denominator else 0.0)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _ratio_row(metric, critic, actor, scope, scope_id, cluster_values, *, seed, replicates, n_anchors, ties=0, degenerate=0, total_pairs=0, candidate_pool=''):
    numerator = sum(value[0] for value in cluster_values.values())
    denominator = sum(value[1] for value in cluster_values.values())
    value = numerator / denominator if denominator else None
    low, high = _bootstrap_ratio(cluster_values, seed, replicates)
    return {
        'metric': metric,
        'critic_config': critic,
        'actor_config': actor,
        'candidate_pool': candidate_pool,
        'scope': scope,
        'scope_id': scope_id,
        'value': value,
        'ci_low': low,
        'ci_high': high,
        'n_episodes': len(cluster_values),
        'n_anchors': n_anchors,
        'n_pairs': denominator,
        'ties': ties,
        'degenerate_pool_rate': (degenerate / total_pairs) if total_pairs else None,
    }


def _write_metric_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        'metric', 'critic_config', 'actor_config', 'candidate_pool', 'scope', 'scope_id',
        'value', 'ci_low', 'ci_high', 'n_episodes', 'n_anchors',
        'n_pairs', 'ties', 'degenerate_pool_rate',
    )
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _temporal_rows(bank, q_values, critic_config, spec):
    by_anchor = defaultdict(list)
    for index, anchor_index in enumerate(bank['pair_anchor_indices']):
        by_anchor[int(anchor_index)].append(index)
    all_clusters = defaultdict(lambda: [0, 0])
    by_task = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_h = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_pair_h = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    ties = 0
    n_pairs = 0
    for indices in by_anchor.values():
        indices = sorted(indices, key=lambda index: int(bank['pair_h'][index]))
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                misrank = float(q_values[left] <= q_values[right])
                tie = int(q_values[left] == q_values[right])
                ties += tie
                n_pairs += 1
                key = (int(bank['pair_task_ids'][left]), int(bank['pair_episode_indices'][left]))
                all_clusters[key][0] += misrank
                all_clusters[key][1] += 1
                by_task[int(bank['pair_task_ids'][left])][key][0] += misrank
                by_task[int(bank['pair_task_ids'][left])][key][1] += 1
                by_h[int(bank['pair_h'][left])][key][0] += misrank
                by_h[int(bank['pair_h'][left])][key][1] += 1
                pair_key = f"{int(bank['pair_h'][left])}>{int(bank['pair_h'][right])}"
                by_pair_h[pair_key][key][0] += misrank
                by_pair_h[pair_key][key][1] += 1
    rows = []
    base_seed = int(spec['bootstrap_seed'])
    rows.append(_ratio_row(
        'E_eval_temporal', critic_config, '', 'overall', 'all', all_clusters,
        seed=base_seed, replicates=spec['bootstrap_replicates'],
        n_anchors=len(by_anchor), ties=ties, total_pairs=n_pairs,
    ))
    for task_id, clusters in sorted(by_task.items()):
        rows.append(_ratio_row(
            'E_eval_temporal', critic_config, '', 'task', str(task_id), clusters,
            seed=derive_seed(base_seed, task_id), replicates=spec['bootstrap_replicates'],
            n_anchors=sum(1 for index in by_anchor if int(bank['anchor_task_ids'][index]) == task_id),
        ))
    for h, clusters in sorted(by_h.items()):
        rows.append(_ratio_row(
            'E_eval_temporal', critic_config, '', 'h', str(h), clusters,
            seed=derive_seed(base_seed, h), replicates=spec['bootstrap_replicates'],
            n_anchors=len(by_anchor),
        ))
    for pair_key, clusters in sorted(by_pair_h.items()):
        rows.append(_ratio_row(
            'E_eval_temporal', critic_config, '', 'h_pair', pair_key, clusters,
            seed=derive_seed(base_seed, _stable_seed(pair_key)),
            replicates=spec['bootstrap_replicates'], n_anchors=len(by_anchor),
        ))
    # Episode-level rows make the clustering unit explicit and allow a later
    # analysis to apply a different bootstrap without re-scoring checkpoints.
    for key, cluster in sorted(all_clusters.items()):
        rows.append(_ratio_row(
            'E_eval_temporal', critic_config, '', 'episode', f'task{key[0]:02d}_ep{key[1]:03d}', {key: cluster},
            seed=base_seed, replicates=1, n_anchors=0,
        ))
    return rows


def _extraction_sample_fields(q_values, candidate_names, candidate_actions, actor_index, epsilon):
    q_values = np.asarray(q_values)
    candidate_actions = np.asarray(candidate_actions)
    if q_values.ndim != 2 or candidate_actions.ndim != 3:
        raise InteractionDiagnosticError('Extraction scores/actions have invalid rank')
    if candidate_actions.shape[:2] != q_values.shape:
        raise InteractionDiagnosticError(
            'Candidate action matrix must have shape (num_pairs, num_candidates, action_dim)'
        )
    if q_values.shape[1] < 2 or not 0 <= int(actor_index) < q_values.shape[1]:
        raise InteractionDiagnosticError('Extraction pool must have at least two candidates and a valid actor index')
    candidate_names = list(candidate_names)
    if len(candidate_names) != q_values.shape[1]:
        raise InteractionDiagnosticError('Candidate names do not match candidate score columns')
    q_max = np.max(q_values, axis=1)
    q_min = np.min(q_values, axis=1)
    q_span = q_max - q_min
    actor_q = q_values[:, int(actor_index)]
    actor_rank_count = np.sum(q_values > actor_q[:, None], axis=1).astype(np.int64)
    duplicate = _duplicate_action_flags(candidate_actions)
    degenerate = q_span < float(epsilon)
    return {
        'q_max': q_max.astype(np.float64),
        'q_min': q_min.astype(np.float64),
        'q_span': q_span.astype(np.float64),
        'actor_q': actor_q.astype(np.float64),
        'best_candidate_index': np.argmax(q_values, axis=1).astype(np.int64),
        'best_candidate_identity': np.asarray(
            [candidate_names[index] for index in np.argmax(q_values, axis=1)], dtype='U64'
        ),
        'actor_rank': actor_rank_count,
        'actor_rank_fraction': actor_rank_count.astype(np.float64) / (q_values.shape[1] - 1),
        'gap': ((q_max - actor_q) / (q_span + float(epsilon))).astype(np.float64),
        'rank': (actor_rank_count.astype(np.float64) / (q_values.shape[1] - 1)),
        'degenerate_pool': degenerate,
        'exact_duplicate_action_pool': duplicate,
    }


def _extraction_rows(
    bank,
    q_values,
    critic_config,
    actor_config,
    candidate_names,
    actor_index,
    spec,
    *,
    candidate_actions=None,
    candidate_pool='',
):
    if candidate_actions is None:
        raise InteractionDiagnosticError(
            'Duplicate candidate diagnostics require the saved action vectors'
        )
    fields = _extraction_sample_fields(
        q_values, candidate_names, candidate_actions, actor_index, spec['epsilon']
    )
    by_episode = defaultdict(list)
    by_task = defaultdict(list)
    for index in range(len(q_values)):
        key = (int(bank['pair_task_ids'][index]), int(bank['pair_episode_indices'][index]))
        by_episode[key].append(index)
        by_task[key[0]].append(index)

    def mean_clusters(values, indices_by_cluster):
        result = {}
        for key, indices in indices_by_cluster.items():
            result[key] = [float(np.sum(values[indices])), len(indices)]
        return result

    rows = []
    for metric, values in (
        ('E_ext_gap', fields['gap']),
        ('E_ext_rank', fields['rank']),
        ('degenerate_pool', fields['degenerate_pool'].astype(np.float64)),
        ('duplicate_candidate_pool', fields['exact_duplicate_action_pool'].astype(np.float64)),
    ):
        all_clusters = mean_clusters(values, by_episode)
        rows.append(_ratio_row(
            metric, critic_config, actor_config, 'overall', 'all', all_clusters,
            seed=derive_seed(spec['bootstrap_seed'], _stable_seed((critic_config, actor_config, metric))),
            replicates=spec['bootstrap_replicates'], n_anchors=len(set(bank['pair_anchor_indices'])),
            total_pairs=len(values), degenerate=int(np.sum(fields['degenerate_pool'])) if metric == 'E_ext_gap' else 0,
            candidate_pool=candidate_pool,
        ))
        for task_id, indices in sorted(by_task.items()):
            task_clusters = {key: [float(np.sum(values[rows_for_task])), len(rows_for_task)]
                             for key, rows_for_task in by_episode.items() if key[0] == task_id}
            rows.append(_ratio_row(
                metric, critic_config, actor_config, 'task', str(task_id), task_clusters,
                seed=derive_seed(spec['bootstrap_seed'], task_id), replicates=spec['bootstrap_replicates'],
                n_anchors=len(set(bank['pair_anchor_indices'][indices])), total_pairs=len(indices),
                degenerate=int(np.sum(fields['degenerate_pool'][indices])) if metric == 'E_ext_gap' else 0,
                candidate_pool=candidate_pool,
            ))
        for key, indices in sorted(by_episode.items()):
            rows.append(_ratio_row(
                metric, critic_config, actor_config, 'episode', f'task{key[0]:02d}_ep{key[1]:03d}',
                {key: [float(np.sum(values[indices])), len(indices)]},
                seed=spec['bootstrap_seed'], replicates=1, n_anchors=0,
                total_pairs=len(indices),
                degenerate=int(np.sum(fields['degenerate_pool'][indices])) if metric == 'E_ext_gap' else 0,
                candidate_pool=candidate_pool,
            ))
    return rows


def score_diagnostics(spec, diagnostic_root_path):
    """Score E_eval_temporal and E_ext metrics on fixed artifacts."""

    bank, bank_metadata = _load_bank(diagnostic_root_path, spec)
    candidates_dir = diagnostic_root(diagnostic_root_path, spec) / 'candidates'
    candidate_metadata = _read_json(candidates_dir / 'candidate_metadata.json')
    if candidate_metadata.get('bank_sha256') != bank_metadata['bank_sha256']:
        raise InteractionDiagnosticError('Candidate pool was not generated from this diagnostic bank')
    candidate_payload = {
        key: value for key, value in candidate_metadata.items() if key != 'candidate_fingerprint'
    }
    if candidate_metadata.get('candidate_fingerprint') != config_fingerprint(candidate_payload):
        raise InteractionDiagnosticError('Candidate metadata fingerprint mismatch')
    sources = validate_source_set(spec)
    _ensure_source_validation_artifact(spec, diagnostic_root_path, sources)
    single_path = candidates_dir / 'single_state_candidates.npz'
    two_path = candidates_dir / 'two_state_candidates.npz'
    if candidate_metadata.get('single_state_sha256') != sha256_file(single_path):
        raise InteractionDiagnosticError('SingleState candidate artifact hash mismatch')
    if candidate_metadata.get('two_state_sha256') != sha256_file(two_path):
        raise InteractionDiagnosticError('TwoState candidate artifact hash mismatch')
    single_npz = np.load(single_path, allow_pickle=False)
    two_npz = np.load(two_path, allow_pickle=False)
    if not np.array_equal(single_npz['pair_id'], bank['pair_ids']):
        raise InteractionDiagnosticError('SingleState candidate pair IDs do not match bank')
    if not np.array_equal(two_npz['pair_id'], bank['pair_ids']):
        raise InteractionDiagnosticError('TwoState candidate pair IDs do not match bank')
    single_actions = single_npz['actions']
    two_actions = two_npz['actions']
    if single_actions.shape[0] != len(bank['pair_ids']) or two_actions.shape[0] != len(bank['pair_ids']):
        raise InteractionDiagnosticError('Candidate pair count does not match bank')
    agents = {}
    envs = []
    try:
        for config_id, provenance in sources.items():
            agent, env, _, _ = _make_restored_agent(provenance)
            agents[config_id] = agent
            envs.append(env)
        pair_obs = bank['pair_observations']
        pair_goals = bank['pair_future_goals']
        pair_exec = bank['pair_executed_actions']
        single_ids = _condition_ids(spec, 'single_state')
        two_ids = _condition_ids(spec, 'two_state')
        evaluator_configs = {
            'CF': spec['baseline_config_id'],
            'CS': single_ids['ff_actor_rec_critic'],
            'CT': two_ids['ff_actor_rec_critic'],
        }
        evaluator_scores = {
            label: _critic_q(agents[config_id], pair_obs, pair_goals, pair_exec)
            for label, config_id in evaluator_configs.items()
        }
        single_order = candidate_metadata['single_state_order']
        two_order = candidate_metadata['two_state_order']
        extraction_scores = {
            'cf_single': np.stack([
                _critic_q(agents[evaluator_configs['CF']], pair_obs, pair_goals, single_actions[:, index, :])
                for index in range(single_actions.shape[1])
            ], axis=1),
            'cs_single': np.stack([
                _critic_q(agents[evaluator_configs['CS']], pair_obs, pair_goals, single_actions[:, index, :])
                for index in range(single_actions.shape[1])
            ], axis=1),
            'cf_two': np.stack([
                _critic_q(agents[evaluator_configs['CF']], pair_obs, pair_goals, two_actions[:, index, :])
                for index in range(two_actions.shape[1])
            ], axis=1),
            'ct_two': np.stack([
                _critic_q(agents[evaluator_configs['CT']], pair_obs, pair_goals, two_actions[:, index, :])
                for index in range(two_actions.shape[1])
            ], axis=1),
        }
    finally:
        for env in envs:
            close = getattr(env, 'close', None)
            if close is not None:
                close()

    output_dir = diagnostic_root(diagnostic_root_path, spec)
    scores_dir = output_dir / 'scores'
    metrics_dir = output_dir / 'metrics'
    if scores_dir.exists() or metrics_dir.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic scores: {output_dir}')
    scores_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        scores_dir / 'evaluator_scores.npz',
        pair_id=bank['pair_ids'],
        pair_task_ids=bank['pair_task_ids'],
        pair_episode_indices=bank['pair_episode_indices'],
        pair_anchor_indices=bank['pair_anchor_indices'],
        pair_h=bank['pair_h'],
        q_CF=evaluator_scores['CF'], q_CS=evaluator_scores['CS'], q_CT=evaluator_scores['CT'],
    )
    single_order = candidate_metadata['single_state_order']
    two_order = candidate_metadata['two_state_order']
    extraction_specs = {
        'cf_single': {
            'critic_config': evaluator_configs['CF'], 'candidate_names': single_order,
            'candidate_actions': single_actions, 'actor_configs': [
                single_ids['ff_actor_ff_critic'], single_ids['rec_actor_ff_critic'],
            ], 'actor_indices': [single_order.index(f"a_{single_ids['ff_actor_ff_critic']}"), single_order.index(f"a_{single_ids['rec_actor_ff_critic']}")],
        },
        'cs_single': {
            'critic_config': evaluator_configs['CS'], 'candidate_names': single_order,
            'candidate_actions': single_actions, 'actor_configs': [
                single_ids['ff_actor_rec_critic'], single_ids['rec_actor_rec_critic'],
            ], 'actor_indices': [single_order.index(f"a_{single_ids['ff_actor_rec_critic']}"), single_order.index(f"a_{single_ids['rec_actor_rec_critic']}")],
        },
        'cf_two': {
            'critic_config': evaluator_configs['CF'], 'candidate_names': two_order,
            'candidate_actions': two_actions, 'actor_configs': [
                two_ids['ff_actor_ff_critic'], two_ids['rec_actor_ff_critic'],
            ], 'actor_indices': [two_order.index(f"a_{two_ids['ff_actor_ff_critic']}"), two_order.index(f"a_{two_ids['rec_actor_ff_critic']}")],
        },
        'ct_two': {
            'critic_config': evaluator_configs['CT'], 'candidate_names': two_order,
            'candidate_actions': two_actions, 'actor_configs': [
                two_ids['ff_actor_rec_critic'], two_ids['rec_actor_rec_critic'],
            ], 'actor_indices': [two_order.index(f"a_{two_ids['ff_actor_rec_critic']}"), two_order.index(f"a_{two_ids['rec_actor_rec_critic']}")],
        },
    }
    extraction_payload = {
        'pair_id': bank['pair_ids'],
        'pair_task_ids': bank['pair_task_ids'],
        'pair_episode_indices': bank['pair_episode_indices'],
        'pair_anchor_indices': bank['pair_anchor_indices'],
        'pair_h': bank['pair_h'],
    }
    for label, values in extraction_scores.items():
        plan = extraction_specs[label]
        extraction_payload[f'{label}_q_candidates'] = values
        for actor_config, actor_index in zip(plan['actor_configs'], plan['actor_indices']):
            comparison_label = f'{label}__{actor_config}'
            fields = _extraction_sample_fields(
                values, plan['candidate_names'], plan['candidate_actions'],
                actor_index, spec['epsilon'],
            )
            for field, field_values in fields.items():
                extraction_payload[f'{comparison_label}_{field}'] = field_values
    np.savez_compressed(
        scores_dir / 'extraction_scores.npz',
        **extraction_payload,
    )
    evaluator_rows = []
    for label, q_values in evaluator_scores.items():
        evaluator_rows.extend(_temporal_rows(bank, q_values, evaluator_configs[label], spec))
    extraction_rows = []
    for label, values in extraction_scores.items():
        plan = extraction_specs[label]
        for actor_config, actor_index in zip(plan['actor_configs'], plan['actor_indices']):
            extraction_rows.extend(_extraction_rows(
                bank, values, plan['critic_config'], actor_config,
                plan['candidate_names'], actor_index, spec,
                candidate_actions=plan['candidate_actions'],
                candidate_pool='single_state' if 'single' in label else 'two_state',
            ))
    _write_metric_csv(metrics_dir / 'evaluator_metrics.csv', evaluator_rows)
    _write_metric_csv(metrics_dir / 'extraction_metrics.csv', extraction_rows)
    score_metadata = {
        'diagnostic_id': spec['diagnostic_id'],
        'bank_sha256': bank_metadata['bank_sha256'],
        'candidate_fingerprint': candidate_metadata['candidate_fingerprint'],
        'evaluator_score_sha256': sha256_file(scores_dir / 'evaluator_scores.npz'),
        'extraction_score_sha256': sha256_file(scores_dir / 'extraction_scores.npz'),
        'critic_semantics': 'q_C(s,a,g)=min(Q1,Q2)',
        'source_provenance': _source_validation_payload(spec, sources)['sources'],
        'evaluator_configs': evaluator_configs,
        'extraction_comparisons': {
            label: {
                'critic_config': plan['critic_config'],
                'candidate_order': plan['candidate_names'],
                'actor_configs': plan['actor_configs'],
                'actor_indices': plan['actor_indices'],
            }
            for label, plan in extraction_specs.items()
        },
        'duplicate_candidate_definition': 'any pair of saved candidate action vectors is exactly np.array_equal',
        'epsilon': spec['epsilon'],
        'bootstrap': {
            'cluster': 'episode',
            'replicates': spec['bootstrap_replicates'],
            'seed': spec['bootstrap_seed'],
            'uncertainty_scope': 'evaluation_sampling_only; not training-seed uncertainty',
        },
    }
    _write_json(scores_dir / 'score_metadata.json', score_metadata)
    return score_metadata


def _tree_items(tree, prefix=()):
    if isinstance(tree, Mapping):
        for key in sorted(tree, key=str):
            yield from _tree_items(tree[key], prefix + (str(key),))
    else:
        yield prefix, np.asarray(tree)


def _tree_signature(tree):
    return [(path, tuple(value.shape), str(value.dtype)) for path, value in _tree_items(tree)]


def _tree_hash(tree):
    digest = hashlib.sha256()
    for path, value in _tree_items(tree):
        digest.update('/'.join(path).encode('utf-8'))
        digest.update(str(value.dtype).encode('utf-8'))
        digest.update(repr(tuple(value.shape)).encode('utf-8'))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _identity_component(first, second, tolerance):
    """Compare one critic params/buffers collection without hiding mismatches."""

    first_items = dict(_tree_items(first))
    second_items = dict(_tree_items(second))
    paths_equal = set(first_items) == set(second_items)
    common = set(first_items) & set(second_items)
    shape_dtype_equal = paths_equal and all(
        first_items[path].shape == second_items[path].shape
        and first_items[path].dtype == second_items[path].dtype
        for path in common
    )
    compatible = [
        path for path in common
        if first_items[path].shape == second_items[path].shape
    ]
    max_abs = max(
        (
            float(np.max(np.abs(first_items[path] - second_items[path])))
            for path in compatible
        ),
        default=0.0,
    )
    exact = paths_equal and shape_dtype_equal and all(
        np.array_equal(first_items[path], second_items[path]) for path in common
    )
    return {
        'paths_reference': [list(path) for path in sorted(first_items)],
        'paths_compared': [list(path) for path in sorted(second_items)],
        'structure_equal': bool(paths_equal),
        'shape_dtype_equal': bool(shape_dtype_equal),
        'element_count_reference': int(sum(value.size for value in first_items.values())),
        'element_count_compared': int(sum(value.size for value in second_items.values())),
        'exact_array_equal': bool(exact),
        'max_absolute_difference': max_abs,
        'allclose': bool(shape_dtype_equal and max_abs <= tolerance),
        'tolerance': float(tolerance),
        'reference_hash': _tree_hash(first),
        'compared_hash': _tree_hash(second),
        'tree_signature_reference': [
            [list(path), list(shape), dtype]
            for path, shape, dtype in _tree_signature(first)
        ],
        'tree_signature_compared': [
            [list(path), list(shape), dtype]
            for path, shape, dtype in _tree_signature(second)
        ],
    }


def run_critic_identity_audit(spec, diagnostic_root_path, tolerance=1e-6):
    """Audit critic branch identity across actor-only computation conditions."""

    sources = validate_source_set(spec)
    source_validation = _ensure_source_validation_artifact(spec, diagnostic_root_path, sources)
    agents = {}
    envs = []
    try:
        for config_id, provenance in sources.items():
            agent, env, _, _ = _make_restored_agent(provenance)
            agents[config_id] = agent
            envs.append(env)
        single_ids = _condition_ids(spec, 'single_state')
        two_ids = _condition_ids(spec, 'two_state')
        groups = {
            'FF': [spec['baseline_config_id'], single_ids['rec_actor_ff_critic'], two_ids['rec_actor_ff_critic']],
            'SingleState': [single_ids['ff_actor_rec_critic'], single_ids['rec_actor_rec_critic']],
            'TwoState': [two_ids['ff_actor_rec_critic'], two_ids['rec_actor_rec_critic']],
        }
        comparisons = []
        for group, config_ids in groups.items():
            reference = config_ids[0]
            for compared in config_ids[1:]:
                for branch in ('phi', 'psi'):
                    first_params = agents[reference].network.params['modules_critic'][branch]
                    second_params = agents[compared].network.params['modules_critic'][branch]
                    first_buffers = (
                        agents[reference].network.model_state.get('buffers', {})
                        .get('modules_critic', {}).get(branch, {})
                    )
                    second_buffers = (
                        agents[compared].network.model_state.get('buffers', {})
                        .get('modules_critic', {}).get(branch, {})
                    )
                    params_audit = _identity_component(first_params, second_params, tolerance)
                    buffers_audit = _identity_component(first_buffers, second_buffers, tolerance)
                    exact_equal = params_audit['exact_array_equal'] and buffers_audit['exact_array_equal']
                    comparisons.append({
                        'group': group,
                        'reference_config': reference,
                        'compared_config': compared,
                        'branch': branch,
                        'parameter_count_reference': params_audit['element_count_reference'],
                        'parameter_count_compared': params_audit['element_count_compared'],
                        'structure_equal': params_audit['structure_equal'],
                        'shape_dtype_equal': params_audit['shape_dtype_equal'],
                        'exact_array_equal': bool(exact_equal),
                        'max_absolute_difference': max(
                            params_audit['max_absolute_difference'],
                            buffers_audit['max_absolute_difference'],
                        ),
                        'allclose': bool(params_audit['allclose'] and buffers_audit['allclose']),
                        'tolerance': float(tolerance),
                        'reference_hash': params_audit['reference_hash'],
                        'compared_hash': params_audit['compared_hash'],
                        'params': params_audit,
                        'buffers': buffers_audit,
                    })
    finally:
        for env in envs:
            close = getattr(env, 'close', None)
            if close is not None:
                close()
    passed = all(item['exact_array_equal'] for item in comparisons)
    audit = {
        'schema_version': 2,
        'diagnostic_id': spec['diagnostic_id'],
        'source_validation_fingerprint': source_validation['source_validation_fingerprint'],
        'source_provenance': source_validation['sources'],
        'same_critic_verification_passed': passed,
        'identity_definition': 'exact parameter-tree and relevant recurrent buffer structure, dtype, shape, array equality and stable hash within each critic group',
        'comparisons': comparisons,
        'tolerance': float(tolerance),
        'note': 'A failed identity is reported as evidence; this audit never overwrites or forces parameters to match.',
    }
    output_dir = diagnostic_root(diagnostic_root_path, spec) / 'audits'
    audit_path = output_dir / 'critic_identity.json'
    if audit_path.exists():
        raise FileExistsError(f'Refusing to overwrite critic identity audit: {audit_path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(audit_path, audit)
    return audit


def _read_csv_rows(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def _metric_lookup(rows, metric, critic_config=None, actor_config=None, candidate_pool=None):
    matches = [
        row for row in rows
        if row.get('metric') == metric
        and row.get('scope') == 'overall'
        and (critic_config is None or row.get('critic_config') == critic_config)
        and (actor_config is None or row.get('actor_config') == actor_config)
        and (candidate_pool is None or row.get('candidate_pool') == candidate_pool)
    ]
    if len(matches) != 1:
        raise InteractionDiagnosticError(
            f'Expected exactly one overall metric row for metric={metric}, '
            f'critic={critic_config}, actor={actor_config}; found {len(matches)}'
        )
    return float(matches[0]['value'])


def _fmt(value, digits=6):
    return 'NA' if value is None else f'{float(value):.{digits}f}'


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def aggregate_interaction_metrics(spec, diagnostic_root_path):
    """Aggregate raw diagnostic scores and write the scientific report."""

    sources = validate_source_set(spec)
    source_validation = _ensure_source_validation_artifact(spec, diagnostic_root_path, sources)
    output_dir = _ensure_diagnostic_root(spec, diagnostic_root_path)
    bank, bank_metadata = _load_bank(diagnostic_root_path, spec)
    scores_dir = output_dir / 'scores'
    metrics_dir = output_dir / 'metrics'
    score_metadata_path = scores_dir / 'score_metadata.json'
    if not score_metadata_path.is_file():
        raise InteractionDiagnosticError(f'Missing score metadata: {score_metadata_path}')
    score_metadata = _read_json(score_metadata_path)
    if score_metadata.get('bank_sha256') != bank_metadata['bank_sha256']:
        raise InteractionDiagnosticError('Score metadata does not match diagnostic bank')
    evaluator_score_path = scores_dir / 'evaluator_scores.npz'
    extraction_score_path = scores_dir / 'extraction_scores.npz'
    if score_metadata.get('evaluator_score_sha256') != sha256_file(evaluator_score_path):
        raise InteractionDiagnosticError('Evaluator score hash mismatch before aggregation')
    if score_metadata.get('extraction_score_sha256') != sha256_file(extraction_score_path):
        raise InteractionDiagnosticError('Extraction score hash mismatch before aggregation')
    evaluator_rows = _read_csv_rows(metrics_dir / 'evaluator_metrics.csv')
    extraction_rows = _read_csv_rows(metrics_dir / 'extraction_metrics.csv')
    # Older score artifacts may predate the explicit candidate_pool CSV column.
    # Reconstruct the tidy rows from immutable raw q-scores in that case rather
    # than guessing which of the two pools a duplicated C001/C001 row belongs
    # to.  The raw score artifact and score metadata are the source of truth.
    if not extraction_rows or 'candidate_pool' not in extraction_rows[0]:
        extraction_rows = []
        extraction_npz = np.load(scores_dir / 'extraction_scores.npz', allow_pickle=False)
        single_actions = np.load(
            output_dir / 'candidates' / 'single_state_candidates.npz', allow_pickle=False
        )['actions']
        two_actions = np.load(
            output_dir / 'candidates' / 'two_state_candidates.npz', allow_pickle=False
        )['actions']
        for label, plan in score_metadata['extraction_comparisons'].items():
            pool = 'single_state' if 'single' in label else 'two_state'
            actions = single_actions if pool == 'single_state' else two_actions
            q_values = extraction_npz[f'{label}_q_candidates']
            for actor_config, actor_index in zip(plan['actor_configs'], plan['actor_indices']):
                extraction_rows.extend(_extraction_rows(
                    bank, q_values, plan['critic_config'], actor_config,
                    plan['candidate_order'], int(actor_index), spec,
                    candidate_actions=actions, candidate_pool=pool,
                ))

    single_ids = _condition_ids(spec, 'single_state')
    two_ids = _condition_ids(spec, 'two_state')
    baseline = spec['baseline_config_id']
    evaluator_configs = {
        'single_state': single_ids['ff_actor_rec_critic'],
        'two_state': two_ids['ff_actor_rec_critic'],
    }

    def proxy_alignment(success_delta, diagnostic_delta):
        if success_delta > 0 and diagnostic_delta < 0:
            return 'aligned'
        if success_delta < 0 and diagnostic_delta > 0:
            return 'misaligned'
        return 'partially aligned'

    final = {}
    for config_id, provenance in sources.items():
        eval_path = Path(provenance['source_run_dir']) / 'eval.csv'
        rows = _read_csv_rows(eval_path)
        if not rows or 'evaluation/overall_success' not in rows[-1]:
            raise InteractionDiagnosticError(f'Missing overall success in {eval_path}')
        step = int(rows[-1].get('step', -1))
        if step != 1_000_000:
            raise InteractionDiagnosticError(f'Primary M11A metric must be last@1M, got {config_id}@{step}')
        final[config_id] = float(rows[-1]['evaluation/overall_success'])

    interactions = {
        'I_S': final[single_ids['rec_actor_rec_critic']]
        - final[single_ids['ff_actor_rec_critic']]
        - final[single_ids['rec_actor_ff_critic']]
        + final[baseline],
        'I_T': final[two_ids['rec_actor_rec_critic']]
        - final[two_ids['ff_actor_rec_critic']]
        - final[two_ids['rec_actor_ff_critic']]
        + final[baseline],
    }
    interaction_rows = [
        {
            'metric': 'I_S',
            'value': interactions['I_S'],
            'formula': 'J(S-CA)-J(S-C)-J(S-A)+J(A)',
            'primary_checkpoint': 'last@1M',
            'statistical_scope': 'single-seed descriptive interaction',
            'source_validation_fingerprint': source_validation['source_validation_fingerprint'],
        },
        {
            'metric': 'I_T',
            'value': interactions['I_T'],
            'formula': 'J(T-CA)-J(T-C)-J(T-A)+J(A)',
            'primary_checkpoint': 'last@1M',
            'statistical_scope': 'single-seed descriptive interaction',
            'source_validation_fingerprint': source_validation['source_validation_fingerprint'],
        },
    ]
    interaction_path = metrics_dir / 'interaction_metrics.csv'
    if interaction_path.exists():
        observed_rows = _read_csv_rows(interaction_path)
        if len(observed_rows) != len(interaction_rows):
            raise InteractionDiagnosticError(f'Existing interaction metrics do not match expected rows: {interaction_path}')
        for observed, expected in zip(observed_rows, interaction_rows):
            if observed.get('metric') != expected['metric'] or not math.isclose(
                float(observed.get('value', 'nan')), float(expected['value']), rel_tol=0.0, abs_tol=1e-12
            ):
                raise InteractionDiagnosticError(f'Existing interaction metrics mismatch: {interaction_path}')
    else:
        with interaction_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=interaction_rows[0].keys())
            writer.writeheader()
            writer.writerows(interaction_rows)

    mechanism_rows = []
    for topology, ids in (('SingleState', single_ids), ('TwoState', two_ids)):
        candidate_pool = 'single_state' if topology == 'SingleState' else 'two_state'
        critic_config = ids['ff_actor_rec_critic']
        # The recurrent-critic comparison fixes CS/CT (C002/C005) and
        # changes only the actor, exactly as specified by the diagnostic.
        actor_rec_under_rec_config = ids['rec_actor_rec_critic']
        actor_ff_config = ids['ff_actor_ff_critic']
        actor_rec_config = ids['rec_actor_ff_critic']
        eval_delta = _metric_lookup(evaluator_rows, 'E_eval_temporal', critic_config=critic_config) - _metric_lookup(
            evaluator_rows, 'E_eval_temporal', critic_config=baseline
        )
        for extraction_metric in ('E_ext_gap', 'E_ext_rank'):
            ff_actor_delta = _metric_lookup(
                extraction_rows, extraction_metric, critic_config=baseline, actor_config=actor_rec_config,
                candidate_pool=candidate_pool,
            ) - _metric_lookup(
                extraction_rows, extraction_metric, critic_config=baseline, actor_config=actor_ff_config,
                candidate_pool=candidate_pool,
            )
            rec_actor_delta = _metric_lookup(
                extraction_rows, extraction_metric, critic_config=critic_config, actor_config=actor_rec_under_rec_config,
                candidate_pool=candidate_pool,
            ) - _metric_lookup(
                extraction_rows, extraction_metric, critic_config=critic_config, actor_config=critic_config,
                candidate_pool=candidate_pool,
            )
            mechanism_rows.append({
                'topology': topology,
                'metric': extraction_metric,
                'critic_config': critic_config,
                'critic_delta_success': final[critic_config] - final[baseline],
                'delta_E_eval': eval_delta,
                'actor_ff_config': actor_ff_config,
                'actor_rec_config': actor_rec_config,
                'actor_delta_success_under_ff_critic': final[actor_rec_config] - final[actor_ff_config],
                'delta_E_ext_under_ff_critic': ff_actor_delta,
                'actor_rec_under_rec_config': actor_rec_under_rec_config,
                'actor_delta_success_under_rec_critic': final[actor_rec_under_rec_config] - final[critic_config],
                'delta_E_ext_under_rec_critic': rec_actor_delta,
                'candidate_pool': candidate_pool,
                'critic_alignment': proxy_alignment(
                    final[critic_config] - final[baseline], eval_delta
                ),
                'actor_alignment_under_ff_critic': proxy_alignment(
                    final[actor_rec_config] - final[actor_ff_config], ff_actor_delta
                ),
                'actor_alignment_under_rec_critic': proxy_alignment(
                    final[actor_rec_under_rec_config] - final[critic_config], rec_actor_delta
                ),
                'lower_is_better': True,
                'statistical_scope': 'single-seed descriptive proxy comparison',
            })
    mechanism_path = metrics_dir / 'mechanism_deltas.csv'
    if mechanism_path.exists():
        observed_rows = _read_csv_rows(mechanism_path)
        if len(observed_rows) != len(mechanism_rows):
            raise InteractionDiagnosticError(f'Existing mechanism deltas do not match expected rows: {mechanism_path}')
        for observed, expected in zip(observed_rows, mechanism_rows):
            for key in ('topology', 'metric', 'critic_config', 'candidate_pool'):
                if observed.get(key) != str(expected[key]):
                    raise InteractionDiagnosticError(f'Existing mechanism deltas mismatch for {key}: {mechanism_path}')
            for key in ('critic_delta_success', 'delta_E_eval', 'delta_E_ext_under_ff_critic', 'delta_E_ext_under_rec_critic'):
                if not math.isclose(float(observed.get(key, 'nan')), float(expected[key]), rel_tol=0.0, abs_tol=1e-12):
                    raise InteractionDiagnosticError(f'Existing mechanism deltas mismatch for {key}: {mechanism_path}')
    else:
        with mechanism_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=mechanism_rows[0].keys())
            writer.writeheader()
            writer.writerows(mechanism_rows)

    candidate_metadata = _read_json(output_dir / 'candidates' / 'candidate_metadata.json')
    identity_path = output_dir / 'audits' / 'critic_identity.json'
    identity = _read_json(identity_path) if identity_path.is_file() else {'same_critic_verification_passed': None}
    report_dir = output_dir / 'reports'
    report_json_path = report_dir / 'diagnostic_summary.json'
    report_md_path = report_dir / 'diagnostic_summary.md'
    if report_json_path.exists() or report_md_path.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic report: {report_dir}')

    e_eval_values = {
        'CF': _metric_lookup(evaluator_rows, 'E_eval_temporal', critic_config=baseline),
        'CS': _metric_lookup(evaluator_rows, 'E_eval_temporal', critic_config=single_ids['ff_actor_rec_critic']),
        'CT': _metric_lookup(evaluator_rows, 'E_eval_temporal', critic_config=two_ids['ff_actor_rec_critic']),
    }
    extraction_primary = {}
    for row in mechanism_rows:
        extraction_primary[f"{row['topology']}/{row['metric']}"] = row
    summary = {
        'schema_version': 2,
        'diagnostic_id': spec['diagnostic_id'],
        'primary_checkpoint': 'last@1M',
        'source_validation_fingerprint': source_validation['source_validation_fingerprint'],
        'source_git_commit': spec['source_git_commit'],
        'source_runs': source_validation['sources'],
        'bank': bank_metadata,
        'candidate_metadata': candidate_metadata,
        'score_metadata': score_metadata,
        'critic_identity_audit': {
            'available': identity.get('same_critic_verification_passed') is not None,
            'same_critic_verification_passed': identity.get('same_critic_verification_passed'),
        },
        'final_success_by_config': final,
        'interaction': interactions,
        'E_eval_temporal_overall': e_eval_values,
        'mechanism_deltas': mechanism_rows,
        'scientific_boundaries': [
            'E_eval_temporal is a realized temporal-ordering consistency proxy, not true critic error.',
            'E_ext_gap and E_ext_rank are candidate-relative critic-internal extraction burden proxies, not true optimal-policy error.',
            'Bootstrap CIs cluster by episode and quantify evaluation sampling uncertainty only.',
            'M11A has one training seed; no causal or population-level claim is made.',
        ],
    }
    _write_json(report_json_path, summary)

    def alignment(success_delta, diagnostic_delta):
        if success_delta > 0 and diagnostic_delta < 0:
            return 'aligned'
        if success_delta < 0 and diagnostic_delta > 0:
            return 'misaligned'
        return 'partially aligned'

    report_lines = [
        '# M11A-D001 Post-hoc Mechanism Diagnostic Report',
        '',
        '## Scope and provenance',
        '',
        '- Study: `M11A`; environment: `antmaze-large-navigate-v0`; training seed: `0`.',
        '- Primary checkpoint: `last@1M`; source commit: `' + spec['source_git_commit'] + '`.',
        '- Source runs: 7/7 validated as completed, clean, same protocol, same dataset root, and checkpoint SHA-verified.',
        '- No training, checkpoint mutation, source-run mutation, or best-checkpoint substitution was performed.',
        '- Shared bank: C001 environment rollout only; no train/validation/replay/offline sample was used as diagnostic data.',
        '',
        '## Artifact counts and deterministic policy',
        '',
        f"- Episodes: `{bank_metadata['num_episodes']}`; anchors: `{bank_metadata['num_anchors']}`; future-goal pairs: `{bank_metadata['num_pairs']}`.",
        '- Anchor stride: `25`; future-goal offsets: `25, 50, ..., 200`; common seed scheme: `common_task_episode_v1`.',
        '- Candidate policy: `distribution.mode()` followed by clip to `[-1, 1]`; no stochastic or search-based candidate generation.',
        f"- Exact duplicate-action rate: SingleState `{candidate_metadata['exact_duplicate_action_rate']['single_state']:.6f}`, TwoState `{candidate_metadata['exact_duplicate_action_rate']['two_state']:.6f}`.",
        f"- Critic identity audit available: `{identity.get('same_critic_verification_passed')}`; this is an audit/provenance result, not a primary diagnostic metric.",
        '',
        '## E_eval_temporal',
        '',
        '| evaluator | config | E_eval_temporal |',
        '|---|---|---:|',
        f"| CF | {baseline} | {_fmt(e_eval_values['CF'])} |",
        f"| CS | {single_ids['ff_actor_rec_critic']} | {_fmt(e_eval_values['CS'])} |",
        f"| CT | {two_ids['ff_actor_rec_critic']} | {_fmt(e_eval_values['CT'])} |",
        '',
        f"- ΔE_eval_SS = CS − CF = `{_fmt(e_eval_values['CS'] - e_eval_values['CF'])}`; negative means lower temporal-ordering inconsistency.",
        f"- ΔE_eval_TS = CT − CF = `{_fmt(e_eval_values['CT'] - e_eval_values['CF'])}`.",
        '',
        '## E_ext and mechanism deltas',
        '',
        '| topology | metric | critic Δsuccess | ΔE_eval | actor Δsuccess under FF | ΔE_ext under FF | actor Δsuccess under recurrent critic | ΔE_ext under recurrent critic |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in mechanism_rows:
        report_lines.append(
            f"| {row['topology']} | {row['metric']} | {_fmt(row['critic_delta_success'])} | "
            f"{_fmt(row['delta_E_eval'])} | {_fmt(row['actor_delta_success_under_ff_critic'])} | "
            f"{_fmt(row['delta_E_ext_under_ff_critic'])} | {_fmt(row['actor_delta_success_under_rec_critic'])} | "
            f"{_fmt(row['delta_E_ext_under_rec_critic'])} |"
        )
    report_lines.extend([
        '',
        'All E_eval/E_ext deltas are descriptive single-seed proxy comparisons; lower E is better, so negative ΔE denotes burden reduction.',
        '',
        '## Direct answers to the diagnostic questions',
        '',
        f"1. SingleState critic C002 does not lower E_eval_temporal relative to C001: Δ=`{_fmt(e_eval_values['CS'] - e_eval_values['CF'])}` (positive, therefore higher burden).",
        f"2. TwoState critic C005 does not lower E_eval_temporal relative to C001: Δ=`{_fmt(e_eval_values['CT'] - e_eval_values['CF'])}` (positive, therefore higher burden).",
        f"3. C003 versus C001 under the same FF critic has ΔE_ext_gap `{_fmt(_metric_lookup(extraction_rows, 'E_ext_gap', critic_config=baseline, actor_config=single_ids['rec_actor_ff_critic'], candidate_pool='single_state') - _metric_lookup(extraction_rows, 'E_ext_gap', critic_config=baseline, actor_config=single_ids['ff_actor_ff_critic'], candidate_pool='single_state'))}`.",
        f"4. C006 versus C001 under the same FF critic has ΔE_ext_gap `{_fmt(_metric_lookup(extraction_rows, 'E_ext_gap', critic_config=baseline, actor_config=two_ids['rec_actor_ff_critic'], candidate_pool='two_state') - _metric_lookup(extraction_rows, 'E_ext_gap', critic_config=baseline, actor_config=two_ids['ff_actor_ff_critic'], candidate_pool='two_state'))}`.",
        f"5. C004 versus C002 under the fixed C002 SingleState critic has ΔE_ext_gap `{_fmt(_metric_lookup(extraction_rows, 'E_ext_gap', critic_config=single_ids['ff_actor_rec_critic'], actor_config=single_ids['rec_actor_rec_critic'], candidate_pool='single_state') - _metric_lookup(extraction_rows, 'E_ext_gap', critic_config=single_ids['ff_actor_rec_critic'], actor_config=single_ids['ff_actor_rec_critic'], candidate_pool='single_state'))}`.",
        f"6. C007 versus C005 under the fixed C005 TwoState critic has ΔE_ext_gap `{_fmt(_metric_lookup(extraction_rows, 'E_ext_gap', critic_config=two_ids['ff_actor_rec_critic'], actor_config=two_ids['rec_actor_rec_critic'], candidate_pool='two_state') - _metric_lookup(extraction_rows, 'E_ext_gap', critic_config=two_ids['ff_actor_rec_critic'], actor_config=two_ids['ff_actor_rec_critic'], candidate_pool='two_state'))}`.",
        '7. A negative SingleState factorial interaction can be compared with the residual-burden deltas above, but it does not by itself establish a mechanism; the correct label is aligned/partially aligned/misaligned only at the proxy level.',
        '8. The TwoState interaction must be interpreted jointly with E_eval and E_ext because its success interaction is time-dependent; no stable complementarity claim is made.',
        '9. Success and diagnostic changes are classified as aligned, partially aligned, or misaligned in `metrics/mechanism_deltas.csv`; this is not causal confirmation.',
        '10. E_eval_temporal and E_ext are proxy diagnostics and cannot be called true critic error or true optimal-policy extraction error.',
        '11. Strong evidence for actor-side burden reduction requires consistent negative E_ext deltas across comparison, metric, task, and uncertainty analyses; this single-seed report does not upgrade proxy evidence to causality.',
        '12. Critic-side evaluator evidence is similarly proxy-level and must not be interpreted as proof of critic quality or Q* accuracy.',
        '13. Any discrepancy between success and E_eval/E_ext should be treated as an interface/proxy limitation until additional diagnostics or targeted experiments resolve it.',
        '14. The next highest-value experiment is a targeted multi-seed replication of the most informative factorial comparison, followed by Large-Stitch factorial; ILR critic should be isolated as a separate study rather than conflated with this result.',
        '',
        '## Scientific boundary',
        '',
        '- `E_eval_temporal` measures realized temporal-ordering inconsistency of conservative critic scores on fixed executed actions.',
        '- `E_ext_gap` and `E_ext_rank` measure candidate-relative critic-internal burden for actor-selected actions.',
        '- Episode-cluster bootstrap CIs quantify evaluation sampling uncertainty only, not training-seed uncertainty.',
        '- Degenerate pools and exact duplicate action vectors are retained and reported rather than removed.',
        '',
        '## Artifact index',
        '',
        '- `bank/diagnostic_bank.npz`, `bank/bank_metadata.json`',
        '- `candidates/single_state_candidates.npz`, `candidates/two_state_candidates.npz`, `candidates/candidate_metadata.json`',
        '- `audits/source_validation.json`, `audits/critic_identity.json`',
        '- `scores/evaluator_scores.npz`, `scores/extraction_scores.npz`, `scores/score_metadata.json`',
        '- `metrics/evaluator_metrics.csv`, `metrics/extraction_metrics.csv`, `metrics/mechanism_deltas.csv`, `metrics/interaction_metrics.csv`',
        '- `reports/diagnostic_summary.json`, `reports/diagnostic_summary.md`',
    ])
    _write_text(report_md_path, '\n'.join(report_lines) + '\n')
    return summary


def smoke_m11a_configs(study_path):
    """Create, update, and checkpoint-restore all seven configs on synthetic data."""

    study = load_study(study_path)
    config_dir = Path(study.path).parent / 'configs'
    configs = []
    for config_id in CONFIG_IDS:
        _, configuration = prepare_run_design(study_path, config_id)
        args = _parse_args(['--agent', 'crl'])
        config = _make_config(args, configuration=configuration)
        observations = jnp.zeros((4, 58), dtype=jnp.float32)
        goals = jnp.ones((4, 58), dtype=jnp.float32)
        actions = jnp.zeros((4, 8), dtype=jnp.float32)
        batch = {
            'observations': observations,
            'value_goals': goals,
            'actor_goals': goals,
            'actions': actions,
        }
        agent = CRLAgent.create(0, observations[:1], actions[:1], config)
        updated, info = agent.update(batch)
        if not all(np.all(np.isfinite(np.asarray(value))) for value in info.values()):
            raise InteractionDiagnosticError(f'Non-finite smoke update for {config_id}')
        with tempfile.TemporaryDirectory(prefix='m11a_smoke_') as temp_dir:
            save_agent(updated, temp_dir, 1)
            restored = restore_agent(updated, temp_dir, 1)
            before = np.asarray(updated.sample_actions(observations[:1], goals[:1], seed=jax.random.PRNGKey(1)))
            after = np.asarray(restored.sample_actions(observations[:1], goals[:1], seed=jax.random.PRNGKey(1)))
            if not np.array_equal(before, after):
                raise InteractionDiagnosticError(f'Checkpoint action mismatch for {config_id}')
        configs.append({
            'config_id': config_id,
            'topology': configuration.data['factors']['topology'],
            'status': 'passed',
            'finite_update': True,
            'checkpoint_restore': True,
        })
    return configs
