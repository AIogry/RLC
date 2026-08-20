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


def diagnostic_root(root, spec):
    return Path(root) / spec['diagnostic_id']


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
    if any(value['source_git_dirty'] is not False for value in sources.values()):
        raise InteractionDiagnosticError('All M11A source runs must have git_dirty=false')
    metadata = [value['source_metadata'] for value in sources.values()]
    if {item.get('dataset_dir') for item in metadata}.__len__() != 1:
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
    if any(item.get('agent') != 'crl' for item in metadata):
        raise InteractionDiagnosticError('All M11A sources must be CRL runs')
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
    episode_indices = []
    episode_seeds = []
    actor_seeds = []
    original_goals = []
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
            flat['observations'].append(observations)
            flat['next_observations'].append(next_observations)
            flat['executed_actions'].append(actions)
            flat['rewards'].append(rewards)
            flat['done'].append(np.asarray(trajectory['done'], dtype=np.bool_))
            start = episode_offsets[-1]
            episode_offsets.append(start + length)
            episode_task_ids.append(task_id)
            episode_indices.append(episode_index)
            episode_seeds.append(seeds['episode_seed'])
            actor_seeds.append(seeds['actor_seed'])
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
                    'episode_id': len(episode_task_ids) - 1,
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
                        'episode_id': len(episode_task_ids) - 1,
                        'task_id': task_id,
                        'episode_index': episode_index,
                        'h': h,
                        'observation': observations[t],
                        'executed_action': actions[t],
                        'future_goal': observations[t + h],
                    })

    def stack_records(records, key, dtype=np.float32):
        return np.asarray([record[key] for record in records], dtype=dtype)

    return {
        'observations': np.concatenate(flat['observations'], axis=0),
        'next_observations': np.concatenate(flat['next_observations'], axis=0),
        'executed_actions': np.concatenate(flat['executed_actions'], axis=0),
        'rewards': np.concatenate(flat['rewards'], axis=0),
        'done': np.concatenate(flat['done'], axis=0),
        'episode_offsets': np.asarray(episode_offsets, dtype=np.int64),
        'episode_task_ids': np.asarray(episode_task_ids, dtype=np.int64),
        'episode_indices': np.asarray(episode_indices, dtype=np.int64),
        'episode_seeds': np.asarray(episode_seeds, dtype=np.int64),
        'actor_seeds': np.asarray(actor_seeds, dtype=np.int64),
        'original_eval_goals': np.asarray(original_goals, dtype=np.float32),
        'anchor_observations': stack_records(anchor_records, 'observation'),
        'anchor_next_observations': stack_records(anchor_records, 'next_observation'),
        'anchor_executed_actions': stack_records(anchor_records, 'executed_action'),
        'anchor_original_goals': stack_records(anchor_records, 'original_goal'),
        'anchor_episode_ids': np.asarray([r['episode_id'] for r in anchor_records], dtype=np.int64),
        'anchor_task_ids': np.asarray([r['task_id'] for r in anchor_records], dtype=np.int64),
        'anchor_episode_indices': np.asarray([r['episode_index'] for r in anchor_records], dtype=np.int64),
        'anchor_t': np.asarray([r['t'] for r in anchor_records], dtype=np.int64),
        'pair_observations': stack_records(pair_records, 'observation'),
        'pair_executed_actions': stack_records(pair_records, 'executed_action'),
        'pair_future_goals': stack_records(pair_records, 'future_goal'),
        'pair_anchor_indices': np.asarray([r['anchor_index'] for r in pair_records], dtype=np.int64),
        'pair_episode_ids': np.asarray([r['episode_id'] for r in pair_records], dtype=np.int64),
        'pair_task_ids': np.asarray([r['task_id'] for r in pair_records], dtype=np.int64),
        'pair_episode_indices': np.asarray([r['episode_index'] for r in pair_records], dtype=np.int64),
        'pair_h': np.asarray([r['h'] for r in pair_records], dtype=np.int64),
    }, task_names


def generate_diagnostic_bank(spec, diagnostic_root_path):
    """Generate the immutable environment-only shared diagnostic bank."""

    sources = validate_source_set(spec)
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
        'diagnostic_id': spec['diagnostic_id'],
        'source_condition': 'A',
        'source_config_id': spec['baseline_config_id'],
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


def generate_candidate_pools(spec, diagnostic_root_path):
    """Extract deterministic actor actions on the immutable shared bank."""

    bank, bank_metadata = _load_bank(diagnostic_root_path, spec)
    output_dir = diagnostic_root(diagnostic_root_path, spec) / 'candidates'
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite immutable candidate pools: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = validate_source_set(spec)
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
        single_actions = {
            'a_exec': pair_exec,
            'a_A': _mode_actions(agents[SINGLE_STATE_IDS['A']], pair_obs, pair_goals),
            'a_S-C': _mode_actions(agents[SINGLE_STATE_IDS['S-C']], pair_obs, pair_goals),
            'a_S-A': _mode_actions(agents[SINGLE_STATE_IDS['S-A']], pair_obs, pair_goals),
            'a_S-CA': _mode_actions(agents[SINGLE_STATE_IDS['S-CA']], pair_obs, pair_goals),
        }
        two_actions = {
            'a_exec': pair_exec,
            'a_A': _mode_actions(agents[TWO_STATE_IDS['A']], pair_obs, pair_goals),
            'a_T-C': _mode_actions(agents[TWO_STATE_IDS['T-C']], pair_obs, pair_goals),
            'a_T-A': _mode_actions(agents[TWO_STATE_IDS['T-A']], pair_obs, pair_goals),
            'a_T-CA': _mode_actions(agents[TWO_STATE_IDS['T-CA']], pair_obs, pair_goals),
        }
    finally:
        for env in envs:
            close = getattr(env, 'close', None)
            if close is not None:
                close()
    single_order = list(single_actions)
    two_order = list(two_actions)
    single_matrix = np.stack([single_actions[key] for key in single_order], axis=1)
    two_matrix = np.stack([two_actions[key] for key in two_order], axis=1)
    np.savez_compressed(output_dir / 'single_state_candidates.npz', actions=single_matrix)
    np.savez_compressed(output_dir / 'two_state_candidates.npz', actions=two_matrix)
    metadata = {
        'diagnostic_id': spec['diagnostic_id'],
        'bank_sha256': bank_metadata['bank_sha256'],
        'pair_count': int(single_matrix.shape[0]),
        'single_state_order': single_order,
        'two_state_order': two_order,
        'single_state_source_config_ids': [SINGLE_STATE_IDS[key[2:]] if key.startswith('a_') and key != 'a_exec' else 'trajectory:A' for key in single_order],
        'two_state_source_config_ids': [TWO_STATE_IDS[key[2:]] if key.startswith('a_') and key != 'a_exec' else 'trajectory:A' for key in two_order],
        'policy': 'deterministic distribution.mode() followed by clip[-1,1]',
        'candidate_pool_definition': {
            'single_state': ['a_exec', 'a_A', 'a_S-C', 'a_S-A', 'a_S-CA'],
            'two_state': ['a_exec', 'a_A', 'a_T-C', 'a_T-A', 'a_T-CA'],
        },
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


def _ratio_row(metric, critic, actor, scope, scope_id, cluster_values, *, seed, replicates, n_anchors, ties=0, degenerate=0, total_pairs=0):
    numerator = sum(value[0] for value in cluster_values.values())
    denominator = sum(value[1] for value in cluster_values.values())
    value = numerator / denominator if denominator else None
    low, high = _bootstrap_ratio(cluster_values, seed, replicates)
    return {
        'metric': metric,
        'critic_config': critic,
        'actor_config': actor,
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
        'metric', 'critic_config', 'actor_config', 'scope', 'scope_id',
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
):
    if candidate_actions is None:
        raise InteractionDiagnosticError(
            'Duplicate candidate diagnostics require the saved action vectors'
        )
    candidate_actions = np.asarray(candidate_actions)
    if candidate_actions.ndim != 3 or candidate_actions.shape[:2] != q_values.shape:
        raise InteractionDiagnosticError(
            'Candidate action matrix must have shape (num_pairs, num_candidates, action_dim)'
        )
    by_episode = defaultdict(list)
    by_task = defaultdict(list)
    q_max = np.max(q_values, axis=1)
    q_min = np.min(q_values, axis=1)
    gap = (q_max - q_values[:, actor_index]) / (q_max - q_min + float(spec['epsilon']))
    rank = np.sum(q_values > q_values[:, actor_index, None], axis=1) / (q_values.shape[1] - 1)
    degenerate = (q_max - q_min) < float(spec['epsilon'])
    duplicate = np.zeros(len(q_values), dtype=np.bool_)
    for index, actions in enumerate(candidate_actions):
        for left in range(actions.shape[0]):
            for right in range(left + 1, actions.shape[0]):
                if np.array_equal(actions[left], actions[right]):
                    duplicate[index] = True
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
        ('E_ext_gap', gap),
        ('E_ext_rank', rank),
        ('degenerate_pool', degenerate.astype(np.float64)),
        ('duplicate_candidate_pool', duplicate.astype(np.float64)),
    ):
        all_clusters = mean_clusters(values, by_episode)
        rows.append(_ratio_row(
            metric, critic_config, actor_config, 'overall', 'all', all_clusters,
            seed=derive_seed(spec['bootstrap_seed'], _stable_seed((critic_config, actor_config, metric))),
            replicates=spec['bootstrap_replicates'], n_anchors=len(set(bank['pair_anchor_indices'])),
            total_pairs=len(values), degenerate=int(np.sum(degenerate)) if metric == 'E_ext_gap' else 0,
        ))
        for task_id, indices in sorted(by_task.items()):
            task_clusters = {key: [float(np.sum(values[rows_for_task])), len(rows_for_task)]
                             for key, rows_for_task in by_episode.items() if key[0] == task_id}
            rows.append(_ratio_row(
                metric, critic_config, actor_config, 'task', str(task_id), task_clusters,
                seed=derive_seed(spec['bootstrap_seed'], task_id), replicates=spec['bootstrap_replicates'],
                n_anchors=len(set(bank['pair_anchor_indices'][indices])), total_pairs=len(indices),
                degenerate=int(np.sum(degenerate[indices])) if metric == 'E_ext_gap' else 0,
            ))
        for key, indices in sorted(by_episode.items()):
            rows.append(_ratio_row(
                metric, critic_config, actor_config, 'episode', f'task{key[0]:02d}_ep{key[1]:03d}',
                {key: [float(np.sum(values[indices])), len(indices)]},
                seed=spec['bootstrap_seed'], replicates=1, n_anchors=0,
                total_pairs=len(indices),
                degenerate=int(np.sum(degenerate[indices])) if metric == 'E_ext_gap' else 0,
            ))
    return rows


def score_diagnostics(spec, diagnostic_root_path):
    """Score E_eval_temporal and E_ext metrics on fixed artifacts."""

    bank, bank_metadata = _load_bank(diagnostic_root_path, spec)
    candidates_dir = diagnostic_root(diagnostic_root_path, spec) / 'candidates'
    candidate_metadata = _read_json(candidates_dir / 'candidate_metadata.json')
    if candidate_metadata.get('bank_sha256') != bank_metadata['bank_sha256']:
        raise InteractionDiagnosticError('Candidate pool was not generated from this diagnostic bank')
    sources = validate_source_set(spec)
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
        evaluator_configs = {
            'CF': 'M11A-C001',
            'CS': 'M11A-C002',
            'CT': 'M11A-C005',
        }
        evaluator_scores = {
            label: _critic_q(agents[config_id], pair_obs, pair_goals, pair_exec)
            for label, config_id in evaluator_configs.items()
        }
        single_actions = np.load(candidates_dir / 'single_state_candidates.npz', allow_pickle=False)['actions']
        two_actions = np.load(candidates_dir / 'two_state_candidates.npz', allow_pickle=False)['actions']
        extraction_scores = {
            'cf_single': np.stack([
                _critic_q(agents['M11A-C001'], pair_obs, pair_goals, single_actions[:, index, :])
                for index in range(single_actions.shape[1])
            ], axis=1),
            'cs_single': np.stack([
                _critic_q(agents['M11A-C002'], pair_obs, pair_goals, single_actions[:, index, :])
                for index in range(single_actions.shape[1])
            ], axis=1),
            'cf_two': np.stack([
                _critic_q(agents['M11A-C001'], pair_obs, pair_goals, two_actions[:, index, :])
                for index in range(two_actions.shape[1])
            ], axis=1),
            'ct_two': np.stack([
                _critic_q(agents['M11A-C005'], pair_obs, pair_goals, two_actions[:, index, :])
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
        pair_task_ids=bank['pair_task_ids'],
        pair_episode_indices=bank['pair_episode_indices'],
        pair_anchor_indices=bank['pair_anchor_indices'],
        pair_h=bank['pair_h'],
        q_cf=evaluator_scores['CF'], q_cs=evaluator_scores['CS'], q_ct=evaluator_scores['CT'],
    )
    np.savez_compressed(
        scores_dir / 'extraction_scores.npz',
        pair_task_ids=bank['pair_task_ids'],
        pair_episode_indices=bank['pair_episode_indices'],
        pair_anchor_indices=bank['pair_anchor_indices'],
        pair_h=bank['pair_h'],
        cf_single=extraction_scores['cf_single'],
        cs_single=extraction_scores['cs_single'],
        cf_two=extraction_scores['cf_two'],
        ct_two=extraction_scores['ct_two'],
    )
    evaluator_rows = []
    for label, q_values in evaluator_scores.items():
        evaluator_rows.extend(_temporal_rows(bank, q_values, evaluator_configs[label], spec))
    extraction_rows = []
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['cf_single'], 'M11A-C001', 'M11A-C001', candidate_metadata['single_state_order'], 1, spec, candidate_actions=single_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['cf_single'], 'M11A-C001', 'M11A-C003', candidate_metadata['single_state_order'], 3, spec, candidate_actions=single_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['cs_single'], 'M11A-C002', 'M11A-C002', candidate_metadata['single_state_order'], 2, spec, candidate_actions=single_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['cs_single'], 'M11A-C002', 'M11A-C004', candidate_metadata['single_state_order'], 4, spec, candidate_actions=single_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['cf_two'], 'M11A-C001', 'M11A-C001', candidate_metadata['two_state_order'], 1, spec, candidate_actions=two_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['cf_two'], 'M11A-C001', 'M11A-C006', candidate_metadata['two_state_order'], 3, spec, candidate_actions=two_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['ct_two'], 'M11A-C005', 'M11A-C005', candidate_metadata['two_state_order'], 2, spec, candidate_actions=two_actions))
    extraction_rows.extend(_extraction_rows(bank, extraction_scores['ct_two'], 'M11A-C005', 'M11A-C007', candidate_metadata['two_state_order'], 4, spec, candidate_actions=two_actions))
    _write_metric_csv(metrics_dir / 'evaluator_metrics.csv', evaluator_rows)
    _write_metric_csv(metrics_dir / 'extraction_metrics.csv', extraction_rows)
    score_metadata = {
        'diagnostic_id': spec['diagnostic_id'],
        'bank_sha256': bank_metadata['bank_sha256'],
        'candidate_fingerprint': candidate_metadata['candidate_fingerprint'],
        'evaluator_score_sha256': sha256_file(scores_dir / 'evaluator_scores.npz'),
        'extraction_score_sha256': sha256_file(scores_dir / 'extraction_scores.npz'),
        'critic_semantics': 'q_C(s,a,g)=min(Q1,Q2)',
        'duplicate_candidate_definition': 'any pair of saved candidate action vectors is exactly np.array_equal',
        'epsilon': spec['epsilon'],
        'bootstrap': {
            'cluster': 'episode',
            'replicates': spec['bootstrap_replicates'],
            'seed': spec['bootstrap_seed'],
            'uncertainty_scope': 'evaluation_sampling_only; not training-seed uncertainty',
        },
    }
    _write_json(output_dir / 'score_metadata.json', score_metadata)
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
    agents = {}
    envs = []
    try:
        for config_id, provenance in sources.items():
            agent, env, _, _ = _make_restored_agent(provenance)
            agents[config_id] = agent
            envs.append(env)
        groups = {
            'FF': ['M11A-C001', 'M11A-C003', 'M11A-C006'],
            'SingleState': ['M11A-C002', 'M11A-C004'],
            'TwoState': ['M11A-C005', 'M11A-C007'],
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
        'diagnostic_id': spec['diagnostic_id'],
        'same_critic_verification_passed': passed,
        'identity_definition': 'exact parameter-tree and relevant recurrent buffer structure, dtype, shape, array equality and stable hash within each critic group',
        'comparisons': comparisons,
        'tolerance': float(tolerance),
        'note': 'A failed identity is reported as evidence; this audit never overwrites or forces parameters to match.',
    }
    output_dir = diagnostic_root(diagnostic_root_path, spec) / 'audits'
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite critic identity audit: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / 'critic_identity.json', audit)
    return audit


def aggregate_interaction_metrics(spec, diagnostic_root_path):
    """Compute last@1M factorial interaction metrics from source eval.csv."""

    sources = validate_source_set(spec)
    final = {}
    for config_id, provenance in sources.items():
        eval_path = Path(provenance['source_run_dir']) / 'eval.csv'
        with eval_path.open(newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows or 'evaluation/overall_success' not in rows[-1]:
            raise InteractionDiagnosticError(f'Missing overall success in {eval_path}')
        step = int(rows[-1].get('step', -1))
        if step != 1_000_000:
            raise InteractionDiagnosticError(f'Primary M11A metric must be last@1M, got {config_id}@{step}')
        final[config_id] = float(rows[-1]['evaluation/overall_success'])
    interactions = {
        'I_S': final['M11A-C004'] - final['M11A-C002'] - final['M11A-C003'] + final['M11A-C001'],
        'I_T': final['M11A-C007'] - final['M11A-C005'] - final['M11A-C006'] + final['M11A-C001'],
    }
    rows = [
        {
            'metric': 'I_S',
            'value': interactions['I_S'],
            'formula': 'J(S-CA)-J(S-C)-J(S-A)+J(A)',
            'primary_checkpoint': 'last@1M',
            'statistical_scope': 'single-seed descriptive interaction',
        },
        {
            'metric': 'I_T',
            'value': interactions['I_T'],
            'formula': 'J(T-CA)-J(T-C)-J(T-A)+J(A)',
            'primary_checkpoint': 'last@1M',
            'statistical_scope': 'single-seed descriptive interaction',
        },
    ]
    output_dir = diagnostic_root(diagnostic_root_path, spec)
    metrics_dir = output_dir / 'metrics'
    if metrics_dir.exists() and (metrics_dir / 'interaction_metrics.csv').exists():
        raise FileExistsError(f'Refusing to overwrite interaction metrics: {metrics_dir / "interaction_metrics.csv"}')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / 'interaction_metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'diagnostic_id': spec['diagnostic_id'],
        'primary_checkpoint': 'last@1M',
        'final_success_by_config': final,
        'interaction': interactions,
        'interpretation_rule': {'I<0': 'descriptive substitution', 'I>0': 'descriptive complementarity', 'I≈0': 'descriptive additive/independent'},
        'scientific_boundary': 'seed=0 only; no statistical significance claim',
        'source_git_commit': next(iter({value['source_git_commit'] for value in sources.values()})),
    }
    _write_json(output_dir / 'summary.json', summary)
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
