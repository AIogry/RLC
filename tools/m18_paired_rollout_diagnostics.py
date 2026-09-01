"""M18-D5 paired closed-loop Puzzle-4x4 rollout diagnostic.

K4 and K8 native-depth policies are evaluated from common task/reset/actor
seeds.  The tool records observable logical state transitions and, only after
a live-environment parity audit passes, exact shortest valid-press distance
``d*``.  It is evaluation-only and locks checkpoints to the original M18-D1
and M18-D234 artifacts rather than resolving the current semantic best.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.diagnostics.puzzle_logic import (
    Puzzle4x4LogicalOracle,
    PuzzleLogicalError,
    array_sha256,
    audit_real_puzzle_environment,
)
from impls.experiment.reevaluation import ReevaluationError
from impls.utils.checkpointing import tree_fingerprint
from impls.utils.evaluation import (
    COMMON_EPISODE_SEED_SCHEME,
    common_episode_seeds,
    extract_episode_success,
    supply_rng,
)
from tools import m18_d_reference as reference
from tools import m18_trace_diagnostics as trace


DIAGNOSTIC_ID = 'M18-D5'
STUDY_ID = 'M18'
ENVIRONMENT = 'puzzle-4x4-play-v0'
TRAIN_KS = (4, 8)
DEFAULT_OUTPUT_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'
DEFAULT_SOURCE_RUN_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'
DEFAULT_EVALUATION_SEED = 18018
DEFAULT_EPISODES_PER_TASK = 20


EVENT_FIELDS = (
    'paired_episode_id', 'model_K', 'task_id', 'task_name', 'episode_index',
    'evaluation_seed', 'task_seed', 'episode_seed', 'actor_seed', 'noise_seed',
    'timestep', 'own_Q1', 'own_Q2', 'own_Qmin', 'reward', 'success',
    'terminated', 'truncated', 'logical_configuration', 'next_logical_configuration',
    'goal_configuration', 'exact_d_star', 'next_exact_d_star', 'logical_distance_delta',
    'logical_transition_event', 'verified_single_press_event', 'pressed_button_id',
    'press_event_kind', 'changed_mask',
)

EPISODE_FIELDS = (
    'paired_episode_id', 'model_K', 'task_id', 'task_name', 'episode_index',
    'evaluation_seed', 'task_seed', 'episode_seed', 'actor_seed', 'noise_seed',
    'initial_observation_sha256', 'goal_observation_sha256',
    'initial_logical_configuration', 'goal_configuration',
    'exact_shortest_distance_available', 'initial_d_star', 'final_d_star',
    'minimum_d_star_reached', 'net_logical_progress', 'best_logical_progress',
    'time_to_first_logical_progress', 'number_of_logical_transitions',
    'number_of_verified_single_press_transitions', 'number_of_unidentified_logical_transitions',
    'number_of_progress_transitions', 'number_of_regressive_transitions',
    'number_of_neutral_press_transitions', 'number_of_no_logical_interaction_timesteps',
    'fraction_progress_among_logical_transitions', 'fraction_regression_among_logical_transitions',
    'success', 'episode_return', 'episode_length', 'terminated', 'truncated',
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])


def _read_csv(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def _parse_int_csv(value, option, *, allowed=None):
    try:
        parsed = tuple(sorted({int(item.strip()) for item in str(value).split(',') if item.strip()}))
    except ValueError as error:
        raise ReevaluationError(f'{option} must contain comma-separated integers') from error
    if not parsed:
        raise ReevaluationError(f'{option} must contain at least one integer')
    if allowed is not None and not set(parsed).issubset(set(allowed)):
        raise ReevaluationError(f'{option} must be a subset of {tuple(allowed)!r}, got {parsed!r}')
    return parsed


def _parse_gpus(value):
    parsed = tuple(item.strip() for item in str(value).split(',') if item.strip())
    if not parsed or len(parsed) != len(set(parsed)) or any(not item.isdigit() for item in parsed):
        raise ReevaluationError('--gpus must contain unique numeric physical GPU IDs')
    return parsed


def paired_episode_plan(task_ids, episodes_per_task, evaluation_seed):
    """Return one deterministic, policy-independent paired seed plan."""

    task_ids = tuple(int(task_id) for task_id in task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids).issubset({1, 2, 3, 4, 5}):
        raise ReevaluationError(f'Puzzle task IDs must be a unique subset of (1,2,3,4,5), got {task_ids!r}')
    if int(episodes_per_task) <= 0:
        raise ReevaluationError('episodes_per_task must be positive')
    result = []
    for task_id in sorted(task_ids):
        for episode_index in range(int(episodes_per_task)):
            seeds = common_episode_seeds(int(evaluation_seed), task_id, episode_index)
            result.append({
                'paired_episode_id': f'task{task_id:02d}_ep{episode_index:03d}',
                'task_id': int(task_id),
                'episode_index': int(episode_index),
                'evaluation_seed': int(evaluation_seed),
                **seeds,
            })
    return result


def _output_dir(output_root, *, task_ids, episodes_per_task, evaluation_seed):
    task_label = '-'.join(str(task_id) for task_id in task_ids)
    return (
        Path(output_root) / 'M18D' / 'closed_loop' / 'checkpoint_locked'
        / f'puzzle4x4_tasks{task_label}_episodes{int(episodes_per_task)}_evalSeed{int(evaluation_seed)}'
    )


def _paired_goal_manifest_paths(output_dir):
    output_dir = Path(output_dir)
    return output_dir / 'paired_goal_manifest.npz', output_dir / 'paired_goal_manifest_metadata.json'


def _create_paired_goal_manifest(output_dir, plan_rows):
    """Create one real-environment goal vector shared byte-for-byte by K4/K8.

    The current Puzzle environment constructs ``info['goal']`` after several
    calls to a freshly-created ``action_space.sample()``.  That continuous
    robot component is not controlled solely by ``env.reset(seed=...)``.
    Therefore native K4/K8 environments can have identical logical targets
    but non-identical raw goal vectors.  A single, separately recorded real
    reset per paired episode is the minimal diagnostic control: its exact goal
    vector is fed to both policies while each rollout environment still keeps
    the same task's own logical target for success evaluation.
    """

    manifest_path, metadata_path = _paired_goal_manifest_paths(output_dir)
    if manifest_path.exists() or metadata_path.exists():
        raise FileExistsError(f'Paired D5 goal manifest already exists: {manifest_path}')
    import ogbench

    env = ogbench.make_env_and_datasets(ENVIRONMENT, env_only=True)
    try:
        records = []
        oracle = Puzzle4x4LogicalOracle.from_environment(env)
        for paired in plan_rows:
            observation, info = env.reset(
                seed=int(paired['episode_seed']),
                options={'task_id': int(paired['task_id']), 'render_goal': False},
            )
            if 'goal' not in info:
                raise ReevaluationError(f'Paired D5 goal reset lacks info[goal]: {paired["paired_episode_id"]}')
            goal = np.asarray(info['goal']).copy()
            records.append({
                **paired,
                'policy_goal': goal,
                'policy_goal_logical_configuration': oracle.encode(oracle.extract_logical_state(goal)),
                'anchor_initial_logical_configuration': oracle.encode(oracle.extract_logical_state(observation)),
            })
        arrays = {
            'paired_episode_id': np.asarray([row['paired_episode_id'] for row in records], dtype='U32'),
            'task_id': np.asarray([row['task_id'] for row in records], dtype=np.int64),
            'episode_index': np.asarray([row['episode_index'] for row in records], dtype=np.int64),
            'evaluation_seed': np.asarray([row['evaluation_seed'] for row in records], dtype=np.int64),
            'task_seed': np.asarray([row['task_seed'] for row in records], dtype=np.int64),
            'episode_seed': np.asarray([row['episode_seed'] for row in records], dtype=np.int64),
            'actor_seed': np.asarray([row['actor_seed'] for row in records], dtype=np.int64),
            'noise_seed': np.asarray([row['noise_seed'] for row in records], dtype=np.int64),
            'policy_goal': np.stack([row['policy_goal'] for row in records]),
            'policy_goal_logical_configuration': np.asarray(
                [row['policy_goal_logical_configuration'] for row in records], dtype=np.int64,
            ),
            'anchor_initial_logical_configuration': np.asarray(
                [row['anchor_initial_logical_configuration'] for row in records], dtype=np.int64,
            ),
        }
        fingerprint = trace._array_fingerprint(arrays)
        np.savez_compressed(manifest_path, **arrays)
        _write_json(metadata_path, {
            'diagnostic_id': DIAGNOSTIC_ID,
            'artifact_type': 'paired_shared_policy_goal_manifest',
            'environment': ENVIRONMENT,
            'paired_episode_count': len(records),
            'paired_seed_scheme': COMMON_EPISODE_SEED_SCHEME,
            'goal_source': (
                'one real PuzzleEnv reset per paired episode; exact emitted info[goal] '
                'is reused byte-for-byte for native K4 and K8 policy/critic calls'
            ),
            'goal_manifest_fingerprint_sha256': fingerprint,
            'array_shapes': {name: list(value.shape) for name, value in arrays.items()},
            'exact_shortest_distance_goal_validation': 'logical goal code stored for worker parity check',
        })
        return arrays, {
            'path': str(manifest_path),
            'metadata_path': str(metadata_path),
            'fingerprint': fingerprint,
        }
    finally:
        env.close()


def _load_paired_goal_manifest(output_dir, plan_rows):
    manifest_path, metadata_path = _paired_goal_manifest_paths(output_dir)
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise ReevaluationError(f'Missing paired D5 goal manifest under {Path(output_dir)}')
    try:
        with np.load(manifest_path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ReevaluationError(f'Cannot load paired D5 goal manifest: {error}') from error
    required = {
        'paired_episode_id', 'task_id', 'episode_index', 'evaluation_seed', 'task_seed', 'episode_seed',
        'actor_seed', 'noise_seed', 'policy_goal', 'policy_goal_logical_configuration',
        'anchor_initial_logical_configuration',
    }
    if set(arrays) != required:
        raise ReevaluationError(f'Paired D5 goal manifest fields mismatch: {sorted(arrays)!r}')
    fingerprint = trace._array_fingerprint(arrays)
    if metadata.get('goal_manifest_fingerprint_sha256') != fingerprint:
        raise ReevaluationError('Paired D5 goal manifest fingerprint mismatch')
    expected_keys = [(str(row['paired_episode_id']), int(row['task_id']), int(row['episode_index'])) for row in plan_rows]
    observed_keys = [
        (str(identifier), int(task_id), int(episode_index))
        for identifier, task_id, episode_index in zip(
            arrays['paired_episode_id'], arrays['task_id'], arrays['episode_index'], strict=True,
        )
    ]
    if observed_keys != expected_keys:
        raise ReevaluationError('Paired D5 goal manifest ordering does not match the deterministic job plan')
    if arrays['policy_goal'].shape[0] != len(plan_rows):
        raise ReevaluationError('Paired D5 goal manifest has an invalid goal leading dimension')
    by_id = {
        str(identifier): {
            'policy_goal': np.asarray(goal).copy(),
            'policy_goal_logical_configuration': int(goal_code),
            'anchor_initial_logical_configuration': int(initial_code),
        }
        for identifier, goal, goal_code, initial_code in zip(
            arrays['paired_episode_id'],
            arrays['policy_goal'],
            arrays['policy_goal_logical_configuration'],
            arrays['anchor_initial_logical_configuration'],
            strict=True,
        )
    }
    return by_id, {
        'path': str(manifest_path),
        'metadata_path': str(metadata_path),
        'fingerprint': fingerprint,
    }


def _native_actor_and_critic_k(config, expected_k):
    try:
        actor_k = int(config['compute']['actor']['topology_kwargs']['iterations'])
        critic_k = int(config['compute']['critic']['topology_kwargs']['iterations'])
    except (KeyError, TypeError, ValueError) as error:
        raise ReevaluationError('Restored D5 config has no native actor/critic iteration fields') from error
    if actor_k != int(expected_k) or critic_k != int(expected_k):
        raise ReevaluationError(
            f'M18-D5 K{expected_k} must use native actor/critic K={expected_k}; '
            f'got actor={actor_k}, critic={critic_k}'
        )


def _critic_values(agent, observation, goal, action, *, model_k):
    values = np.asarray(
        agent.network.select('critic')(
            np.asarray(observation)[None], np.asarray(goal)[None], np.asarray(action)[None],
        ),
        dtype=np.float64,
    )
    if values.shape != (2, 1) or not np.all(np.isfinite(values)):
        raise ReevaluationError(f'K{model_k} own critic produced invalid values with shape {values.shape!r}')
    return float(values[0, 0]), float(values[1, 0]), float(min(values[0, 0], values[1, 0]))


def _float_or_none(value):
    if value is None:
        return None
    return float(value)


def _int_or_none(value):
    if value is None:
        return None
    return int(value)


def _npz_from_step_records(records):
    """Store raw vectors in NPZ while CSV retains only scalar/event fields."""

    if not records:
        return {
            'paired_episode_id': np.asarray([], dtype='U1'),
            'model_K': np.asarray([], dtype=np.int64),
            'timestep': np.asarray([], dtype=np.int64),
            'observation': np.empty((0, 83), dtype=np.float32),
            'next_observation': np.empty((0, 83), dtype=np.float32),
            'goal': np.empty((0, 83), dtype=np.float32),
            'action': np.empty((0, 5), dtype=np.float32),
        }
    return {
        'paired_episode_id': np.asarray([row['paired_episode_id'] for row in records], dtype='U32'),
        'model_K': np.asarray([row['model_K'] for row in records], dtype=np.int64),
        'task_id': np.asarray([row['task_id'] for row in records], dtype=np.int64),
        'episode_index': np.asarray([row['episode_index'] for row in records], dtype=np.int64),
        'timestep': np.asarray([row['timestep'] for row in records], dtype=np.int64),
        'observation': np.stack([np.asarray(row['_observation']) for row in records]),
        'next_observation': np.stack([np.asarray(row['_next_observation']) for row in records]),
        'goal': np.stack([np.asarray(row['_goal']) for row in records]),
        'action': np.stack([np.asarray(row['_action']) for row in records]),
        'own_Q1': np.asarray([row['own_Q1'] for row in records], dtype=np.float64),
        'own_Q2': np.asarray([row['own_Q2'] for row in records], dtype=np.float64),
        'own_Qmin': np.asarray([row['own_Qmin'] for row in records], dtype=np.float64),
    }


def _rollout_model(
    provenance,
    *,
    model_k,
    plan_rows,
    paired_goals,
    paired_goal_manifest,
    output_dir,
    diagnostic_code_commit,
):
    """Run native-depth rollouts for one model under one immutable checkpoint."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'M18-D5 model output exists; refusing overwrite: {output_dir}')
    checkpoint_hash_before = reference.stable_checkpoint_sha256(provenance['checkpoint_path'])
    if checkpoint_hash_before != provenance['checkpoint_sha256']:
        raise ReevaluationError(f'K{model_k} locked source checkpoint changed after planning')
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / 'metadata.json'
    metadata = {
        'status': 'running',
        'diagnostic_id': DIAGNOSTIC_ID,
        'diagnostic_family': 'paired_closed_loop_logical_rollout',
        'diagnostic_code_commit': str(diagnostic_code_commit),
        'source_study_id': STUDY_ID,
        'environment': ENVIRONMENT,
        'model_K': int(model_k),
        'native_actor_test_K': int(model_k),
        'native_critic_K': int(model_k),
        'checkpoint_selector': reference.LOCKED_REFERENCE_SELECTOR,
        'source_config_id': provenance['source_config_id'],
        'source_run_dir': provenance['source_run_dir'],
        'source_checkpoint_role': provenance['resolved_checkpoint_role'],
        'source_checkpoint_step': provenance['checkpoint_step'],
        'source_checkpoint_path': provenance['checkpoint_path'],
        'source_checkpoint_sha256': provenance['checkpoint_sha256'],
        'source_checkpoint_hash_before': checkpoint_hash_before,
        'reference_d1_summary_path': provenance['reference_d1_summary_path'],
        'reference_trace_metadata_path': provenance['reference_trace_metadata_path'],
        'paired_seed_scheme': COMMON_EPISODE_SEED_SCHEME,
        'paired_episode_count': int(len(plan_rows)),
        'shared_policy_goal_manifest_path': paired_goal_manifest['path'],
        'shared_policy_goal_manifest_fingerprint_sha256': paired_goal_manifest['fingerprint'],
        'shared_policy_goal_contract': (
            'the same stored real-environment goal vector is used for K4 and K8 policy and own-critic calls'
        ),
        'evaluation_seed': int(plan_rows[0]['evaluation_seed']) if plan_rows else None,
        'eval_temperature': 0.0,
        'eval_gaussian': None,
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
    }
    _write_json(metadata_path, metadata)
    env = None
    try:
        import jax

        agent, env, _, config = trace._build_restored_agent(provenance)
        _native_actor_and_critic_k(config, model_k)
        params_before = tree_fingerprint(agent.network.params)
        network_step_before = int(np.asarray(agent.network.step))
        semantics_audit = audit_real_puzzle_environment(
            env,
            validation_seed=int(plan_rows[0]['evaluation_seed']) if plan_rows else DEFAULT_EVALUATION_SEED,
            transition_cases=16,
        )
        exact_available = bool(semantics_audit.get('exact_shortest_distance_available'))
        oracle = Puzzle4x4LogicalOracle.from_environment(env) if exact_available else None
        task_infos = getattr(env.unwrapped, 'task_infos', None)
        if task_infos is None or len(task_infos) != 5:
            raise ReevaluationError('M18-D5 requires the five audited Puzzle-4x4 task definitions')
        scalar_records = []
        raw_records = []
        episode_rows = []
        for paired in plan_rows:
            task_id = int(paired['task_id'])
            task_name = str(task_infos[task_id - 1]['task_name'])
            observation, reset_info = env.reset(
                seed=int(paired['episode_seed']),
                options={'task_id': task_id, 'render_goal': False},
            )
            observation = np.asarray(observation)
            if 'goal' not in reset_info:
                raise ReevaluationError(f'K{model_k} reset did not expose a goal for task {task_id}')
            try:
                paired_goal = paired_goals[paired['paired_episode_id']]
            except KeyError as error:
                raise ReevaluationError(f'Missing shared goal for {paired["paired_episode_id"]}') from error
            goal = np.asarray(paired_goal['policy_goal']).copy()
            initial_observation = observation.copy()
            initial_observation_sha = array_sha256(initial_observation)
            goal_sha = array_sha256(goal)
            current_states = None
            goal_states = None
            current_code = None
            goal_code = None
            current_distance = None
            exact_for_episode = False
            if oracle is not None:
                current_states = oracle.extract_logical_state(observation)
                goal_states = oracle.extract_logical_state(goal)
                reset_goal_states = oracle.extract_logical_state(np.asarray(reset_info['goal']))
                if not np.array_equal(goal_states, reset_goal_states):
                    raise ReevaluationError(
                        f'K{model_k} shared policy goal logical target differs from environment target at '
                        f'{paired["paired_episode_id"]}'
                    )
                if oracle.encode(goal_states) != int(paired_goal['policy_goal_logical_configuration']):
                    raise ReevaluationError(f'Shared policy goal code mismatch at {paired["paired_episode_id"]}')
                current_code = oracle.encode(current_states)
                goal_code = oracle.encode(goal_states)
                current_distance = oracle.distance(current_states, goal_states)
                exact_for_episode = current_distance is not None
            actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(int(paired['actor_seed'])))
            done = False
            timestep = 0
            episode_return = 0.0
            final_info = reset_info
            final_terminated = False
            final_truncated = False
            distance_history = [] if current_distance is None else [current_distance]
            logical_transition_count = 0
            verified_single_press_count = 0
            unidentified_transition_count = 0
            progress_count = 0
            regression_count = 0
            neutral_press_count = 0
            first_progress_timestep = None
            while not done:
                action = np.asarray(
                    actor_fn(observations=observation, goals=goal, temperature=0.0), dtype=np.float64,
                )
                action = np.clip(action, -1.0, 1.0)
                if action.shape != env.action_space.shape or not np.all(np.isfinite(action)):
                    raise ReevaluationError(f'K{model_k} actor emitted invalid action at {paired["paired_episode_id"]}')
                q1, q2, qmin = _critic_values(agent, observation, goal, action, model_k=model_k)
                next_observation, reward, terminated, truncated, final_info = env.step(action)
                next_observation = np.asarray(next_observation)
                done = bool(terminated or truncated)
                final_terminated = bool(terminated)
                final_truncated = bool(truncated)
                episode_return += float(reward)
                next_states = None
                next_code = None
                next_distance = None
                event = {
                    'logical_transition_event': False,
                    'verified_single_press_event': False,
                    'pressed_button_id': None,
                    'press_event_kind': 'logical_state_unavailable',
                    'changed_mask': None,
                }
                distance_delta = None
                if oracle is not None:
                    next_states = oracle.extract_logical_state(next_observation)
                    next_code = oracle.encode(next_states)
                    event = oracle.classify_observed_transition(current_states, next_states)
                    next_distance = oracle.distance(next_states, goal_states)
                    if current_distance is not None and next_distance is not None:
                        distance_delta = int(next_distance) - int(current_distance)
                        distance_history.append(next_distance)
                    if event['logical_transition_event']:
                        logical_transition_count += 1
                        if event['verified_single_press_event']:
                            verified_single_press_count += 1
                        else:
                            unidentified_transition_count += 1
                        if distance_delta is not None:
                            if distance_delta < 0:
                                progress_count += 1
                                if first_progress_timestep is None:
                                    first_progress_timestep = int(timestep)
                            elif distance_delta > 0:
                                regression_count += 1
                            elif event['verified_single_press_event']:
                                neutral_press_count += 1
                scalar = {
                    'paired_episode_id': paired['paired_episode_id'],
                    'model_K': int(model_k),
                    'task_id': task_id,
                    'task_name': task_name,
                    'episode_index': int(paired['episode_index']),
                    'evaluation_seed': int(paired['evaluation_seed']),
                    'task_seed': int(paired['task_seed']),
                    'episode_seed': int(paired['episode_seed']),
                    'actor_seed': int(paired['actor_seed']),
                    'noise_seed': int(paired['noise_seed']),
                    'timestep': int(timestep),
                    'own_Q1': q1,
                    'own_Q2': q2,
                    'own_Qmin': qmin,
                    'reward': float(reward),
                    'success': bool(final_info.get('success', False)),
                    'terminated': bool(terminated),
                    'truncated': bool(truncated),
                    'logical_configuration': _int_or_none(current_code),
                    'next_logical_configuration': _int_or_none(next_code),
                    'goal_configuration': _int_or_none(goal_code),
                    'exact_d_star': _int_or_none(current_distance),
                    'next_exact_d_star': _int_or_none(next_distance),
                    'logical_distance_delta': _int_or_none(distance_delta),
                    'logical_transition_event': bool(event['logical_transition_event']),
                    'verified_single_press_event': bool(event['verified_single_press_event']),
                    'pressed_button_id': _int_or_none(event['pressed_button_id']),
                    'press_event_kind': event['press_event_kind'],
                    'changed_mask': _int_or_none(event['changed_mask']),
                }
                scalar_records.append(scalar)
                raw_records.append(scalar | {
                    '_observation': observation.copy(),
                    '_next_observation': next_observation.copy(),
                    '_goal': goal.copy(),
                    '_action': action.copy(),
                })
                observation = next_observation
                current_states = next_states
                current_code = next_code
                current_distance = next_distance
                timestep += 1
            success = float(extract_episode_success(final_info))
            final_distance = current_distance
            minimum_distance = min(distance_history) if distance_history else None
            initial_distance = distance_history[0] if distance_history else None
            denominator = logical_transition_count
            episode_rows.append({
                'paired_episode_id': paired['paired_episode_id'],
                'model_K': int(model_k),
                'task_id': task_id,
                'task_name': task_name,
                'episode_index': int(paired['episode_index']),
                'evaluation_seed': int(paired['evaluation_seed']),
                'task_seed': int(paired['task_seed']),
                'episode_seed': int(paired['episode_seed']),
                'actor_seed': int(paired['actor_seed']),
                'noise_seed': int(paired['noise_seed']),
                'initial_observation_sha256': initial_observation_sha,
                'goal_observation_sha256': goal_sha,
                'initial_logical_configuration': _int_or_none(oracle.encode(oracle.extract_logical_state(initial_observation)) if oracle else None),
                'goal_configuration': _int_or_none(goal_code),
                'exact_shortest_distance_available': bool(exact_for_episode),
                'initial_d_star': _int_or_none(initial_distance),
                'final_d_star': _int_or_none(final_distance),
                'minimum_d_star_reached': _int_or_none(minimum_distance),
                'net_logical_progress': _int_or_none(None if initial_distance is None or final_distance is None else initial_distance - final_distance),
                'best_logical_progress': _int_or_none(None if initial_distance is None or minimum_distance is None else initial_distance - minimum_distance),
                'time_to_first_logical_progress': _int_or_none(first_progress_timestep),
                'number_of_logical_transitions': int(logical_transition_count),
                'number_of_verified_single_press_transitions': int(verified_single_press_count),
                'number_of_unidentified_logical_transitions': int(unidentified_transition_count),
                'number_of_progress_transitions': int(progress_count),
                'number_of_regressive_transitions': int(regression_count),
                'number_of_neutral_press_transitions': int(neutral_press_count),
                'number_of_no_logical_interaction_timesteps': int(timestep - logical_transition_count),
                'fraction_progress_among_logical_transitions': _float_or_none(None if denominator == 0 else progress_count / denominator),
                'fraction_regression_among_logical_transitions': _float_or_none(None if denominator == 0 else regression_count / denominator),
                'success': success,
                'episode_return': float(episode_return),
                'episode_length': int(timestep),
                'terminated': final_terminated,
                'truncated': final_truncated,
            })
        params_after = tree_fingerprint(agent.network.params)
        network_step_after = int(np.asarray(agent.network.step))
        if params_before != params_after:
            raise ReevaluationError(f'K{model_k} online parameters changed during D5 rollout')
        if network_step_before != network_step_after:
            raise ReevaluationError(f'K{model_k} optimizer/network step changed during D5 rollout')
        checkpoint_hash_after = reference.stable_checkpoint_sha256(provenance['checkpoint_path'])
        if checkpoint_hash_after != checkpoint_hash_before:
            raise ReevaluationError(f'K{model_k} source checkpoint SHA256 changed during D5 rollout')
        np.savez_compressed(output_dir / 'per_step_rollouts.npz', **_npz_from_step_records(raw_records))
        _write_csv(output_dir / 'per_step_events.csv', scalar_records, EVENT_FIELDS)
        _write_csv(output_dir / 'episode_summary.csv', episode_rows, EPISODE_FIELDS)
        metadata.update({
            'status': 'completed',
            'environment_semantics_audit': semantics_audit,
            'exact_shortest_distance_available': bool(exact_available),
            'source_checkpoint_hash_after': checkpoint_hash_after,
            'source_checkpoint_immutable': True,
            'online_parameter_fingerprint_before': params_before,
            'online_parameter_fingerprint_after': params_after,
            'online_network_step_before': network_step_before,
            'online_network_step_after': network_step_after,
            'artifacts': [
                str(output_dir / 'per_step_rollouts.npz'),
                str(output_dir / 'per_step_events.csv'),
                str(output_dir / 'episode_summary.csv'),
            ],
        })
        _write_json(metadata_path, metadata)
        return episode_rows, metadata
    except BaseException as error:
        try:
            checkpoint_hash_after = reference.stable_checkpoint_sha256(provenance['checkpoint_path'])
            metadata['source_checkpoint_hash_after'] = checkpoint_hash_after
            metadata['source_checkpoint_immutable'] = checkpoint_hash_after == checkpoint_hash_before
        except BaseException as hash_error:
            metadata['source_checkpoint_hash_after_error'] = f'{type(hash_error).__name__}: {hash_error}'
        metadata.update({'status': 'failed', 'failure_reason': f'{type(error).__name__}: {error}'})
        _write_json(metadata_path, metadata)
        raise
    finally:
        if env is not None:
            env.close()


def _number(value):
    if value in (None, ''):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _summary(values):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'std': None, 'median': None, 'p10': None, 'p90': None}
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
    }


def _paired_rows(k4_rows, k8_rows, plan_rows):
    """Validate paired initial conditions, then derive episode-level deltas."""

    plan_by_key = {(int(row['task_id']), int(row['episode_index'])): row for row in plan_rows}
    by_k = {}
    for model_k, rows in ((4, k4_rows), (8, k8_rows)):
        mapped = {(int(row['task_id']), int(row['episode_index'])): row for row in rows}
        if set(mapped) != set(plan_by_key):
            raise ReevaluationError(f'K{model_k} episode output does not exactly match the paired job plan')
        by_k[model_k] = mapped
    paired = []
    compared_fields = ('task_seed', 'episode_seed', 'actor_seed', 'noise_seed', 'initial_observation_sha256', 'goal_observation_sha256', 'initial_logical_configuration', 'goal_configuration')
    for key in sorted(plan_by_key):
        left, right = by_k[4][key], by_k[8][key]
        for field in compared_fields:
            if str(left.get(field, '')) != str(right.get(field, '')):
                raise ReevaluationError(
                    f'Paired initial-condition mismatch at {left.get("paired_episode_id")}: {field} '
                    f'K4={left.get(field)!r}, K8={right.get(field)!r}'
                )
        entry = {
            'paired_episode_id': left['paired_episode_id'],
            'task_id': int(left['task_id']),
            'task_name': left['task_name'],
            'episode_index': int(left['episode_index']),
            'evaluation_seed': int(left['evaluation_seed']),
            'task_seed': int(left['task_seed']),
            'episode_seed': int(left['episode_seed']),
            'actor_seed': int(left['actor_seed']),
            'noise_seed': int(left['noise_seed']),
            'initial_observation_sha256': left['initial_observation_sha256'],
            'goal_observation_sha256': left['goal_observation_sha256'],
            'exact_shortest_distance_available': str(left['exact_shortest_distance_available']).lower() == 'true' and str(right['exact_shortest_distance_available']).lower() == 'true',
        }
        for metric in (
            'success', 'final_d_star', 'best_logical_progress', 'time_to_first_logical_progress',
            'number_of_logical_transitions', 'number_of_progress_transitions',
            'number_of_regressive_transitions', 'number_of_neutral_press_transitions', 'episode_length',
        ):
            k4_value, k8_value = _number(left.get(metric)), _number(right.get(metric))
            entry[f'K4_{metric}'] = k4_value
            entry[f'K8_{metric}'] = k8_value
            entry[f'K4_minus_K8_{metric}'] = None if k4_value is None or k8_value is None else k4_value - k8_value
        paired.append(entry)
    return paired


def _aggregate_rows(model_rows, paired_rows):
    rows = []
    model_metrics = (
        'success', 'initial_d_star', 'final_d_star', 'minimum_d_star_reached',
        'net_logical_progress', 'best_logical_progress', 'time_to_first_logical_progress',
        'number_of_logical_transitions', 'number_of_progress_transitions',
        'number_of_regressive_transitions', 'number_of_neutral_press_transitions', 'episode_length',
    )
    for model_k in TRAIN_KS:
        source = [row for row in model_rows if int(row['model_K']) == model_k]
        for task_id in [None, 1, 2, 3, 4, 5]:
            subset = source if task_id is None else [row for row in source if int(row['task_id']) == task_id]
            if not subset:
                continue
            for metric in model_metrics:
                rows.append({
                    'record_type': 'model_episode_aggregate',
                    'scope': 'overall' if task_id is None else 'task',
                    'task_id': '' if task_id is None else task_id,
                    'model_K': model_k,
                    'metric': metric,
                    **_summary([_number(row.get(metric)) for row in subset]),
                })
    paired_metrics = [key for key in paired_rows[0] if key.startswith('K4_minus_K8_')] if paired_rows else []
    for task_id in [None, 1, 2, 3, 4, 5]:
        subset = paired_rows if task_id is None else [row for row in paired_rows if int(row['task_id']) == task_id]
        if not subset:
            continue
        for metric in paired_metrics:
            rows.append({
                'record_type': 'paired_difference_aggregate',
                'scope': 'overall' if task_id is None else 'task',
                'task_id': '' if task_id is None else task_id,
                'model_K': 'K4_minus_K8',
                'metric': metric.removeprefix('K4_minus_K8_'),
                **_summary([_number(row.get(metric)) for row in subset]),
            })
    return rows


def aggregate(output_dir, plan_rows):
    """Combine two completed native-model outputs without aligning trajectories."""

    output_dir = Path(output_dir)
    model_rows = {}
    model_metadata = {}
    for model_k in TRAIN_KS:
        model_dir = output_dir / f'model_K{model_k}'
        metadata_path = model_dir / 'metadata.json'
        if not metadata_path.is_file() or not (model_dir / 'episode_summary.csv').is_file():
            raise ReevaluationError(f'Missing M18-D5 K{model_k} model artifact under {model_dir}')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('status') != 'completed' or metadata.get('diagnostic_id') != DIAGNOSTIC_ID:
            raise ReevaluationError(f'M18-D5 K{model_k} worker did not complete successfully')
        if int(metadata.get('model_K', -1)) != model_k:
            raise ReevaluationError(f'M18-D5 K{model_k} metadata model identity mismatch')
        if not metadata.get('source_checkpoint_immutable') or int(metadata.get('optimizer_updates', -1)) != 0:
            raise ReevaluationError(f'M18-D5 K{model_k} violates immutable evaluation-only contract')
        model_rows[model_k] = _read_csv(model_dir / 'episode_summary.csv')
        model_metadata[model_k] = metadata
    paired = _paired_rows(model_rows[4], model_rows[8], plan_rows)
    all_model_rows = model_rows[4] + model_rows[8]
    summary_rows = _aggregate_rows(all_model_rows, paired)
    paired_fields = tuple(paired[0]) if paired else ()
    _write_csv(output_dir / 'paired_episode_summary.csv', paired, paired_fields)
    summary_fields = ('record_type', 'scope', 'task_id', 'model_K', 'metric', 'count', 'mean', 'std', 'median', 'p10', 'p90')
    _write_csv(output_dir / 'm18d_d5_summary.csv', summary_rows, summary_fields)
    summary = {
        'status': 'completed',
        'diagnostic_id': DIAGNOSTIC_ID,
        'diagnostic_family': 'paired_closed_loop_logical_rollout',
        'paired_episode_count': len(paired),
        'model_episode_count': len(all_model_rows),
        'summary_rows': summary_rows,
        'exact_shortest_distance_available_by_model': {
            f'K{model_k}': bool(model_metadata[model_k].get('exact_shortest_distance_available'))
            for model_k in TRAIN_KS
        },
        'model_metadata_paths': {f'K{model_k}': str(output_dir / f'model_K{model_k}' / 'metadata.json') for model_k in TRAIN_KS},
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
        'source_checkpoint_immutable': True,
    }
    _write_json(output_dir / 'm18d_d5_summary.json', summary)
    return summary


def plan(
    reference_diagnostics_root,
    source_run_root,
    output_root,
    *,
    task_ids=(1, 2, 3, 4, 5),
    episodes_per_task=DEFAULT_EPISODES_PER_TASK,
    evaluation_seed=DEFAULT_EVALUATION_SEED,
):
    contract = reference.load_reference_contract(reference_diagnostics_root, train_ks=TRAIN_KS)
    provenance = {
        model_k: reference.locked_provenance(contract, source_run_root, model_k)
        for model_k in TRAIN_KS
    }
    rows = paired_episode_plan(task_ids, episodes_per_task, evaluation_seed)
    return {
        'contract': contract,
        'provenance': provenance,
        'plan_rows': rows,
        'output_dir': _output_dir(
            output_root,
            task_ids=tuple(sorted(task_ids)),
            episodes_per_task=episodes_per_task,
            evaluation_seed=evaluation_seed,
        ),
        'smoke_only': bool(tuple(sorted(task_ids)) != (1, 2, 3, 4, 5) or int(episodes_per_task) != DEFAULT_EPISODES_PER_TASK),
    }


def _worker_command(args, *, model_k, assigned_gpu):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        '--reference-diagnostics-root', str(args.reference_diagnostics_root),
        '--source-run-root', str(args.source_run_root),
        '--output-root', str(args.output_root),
        '--task-ids', str(args.task_ids),
        '--episodes-per-task', str(args.episodes_per_task),
        '--evaluation-seed', str(args.evaluation_seed),
        '--checkpoint', str(args.checkpoint),
        '--diagnostic-code-commit', str(args.diagnostic_code_commit),
        '--worker-model-k', str(model_k),
        '--assigned-gpu', str(assigned_gpu),
    ]


def _dispatch_workers(args):
    jobs = queue.Queue()
    for model_k in TRAIN_KS:
        jobs.put(model_k)

    def worker(assigned_gpu):
        failures = 0
        while True:
            try:
                model_k = jobs.get_nowait()
            except queue.Empty:
                return failures
            try:
                if subprocess.run(_worker_command(args, model_k=model_k, assigned_gpu=assigned_gpu), check=False).returncode != 0:
                    failures += 1
            finally:
                jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        return sum(future.result() for future in [executor.submit(worker, gpu) for gpu in args.gpus])


def _worker_execute(args):
    if args.worker_model_k not in TRAIN_KS:
        raise ReevaluationError(f'Invalid D5 worker model K: {args.worker_model_k!r}')
    task_ids = _parse_int_csv(args.task_ids, '--task-ids', allowed=(1, 2, 3, 4, 5))
    plan_data = plan(
        args.reference_diagnostics_root,
        args.source_run_root,
        args.output_root,
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        evaluation_seed=args.evaluation_seed,
    )
    output_dir = Path(plan_data['output_dir'])
    if not output_dir.is_dir():
        raise ReevaluationError(f'D5 worker parent output directory does not exist: {output_dir}')
    paired_goals, paired_goal_manifest = _load_paired_goal_manifest(output_dir, plan_data['plan_rows'])
    _rollout_model(
        plan_data['provenance'][int(args.worker_model_k)],
        model_k=int(args.worker_model_k),
        plan_rows=plan_data['plan_rows'],
        paired_goals=paired_goals,
        paired_goal_manifest=paired_goal_manifest,
        output_dir=output_dir / f'model_K{int(args.worker_model_k)}',
        diagnostic_code_commit=args.diagnostic_code_commit,
    )
    return output_dir


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-diagnostics-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--source-run-root', default=DEFAULT_SOURCE_RUN_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--train-ks', default='4,8')
    parser.add_argument('--checkpoint', default=reference.LOCKED_REFERENCE_SELECTOR)
    parser.add_argument('--task-ids', default='1,2,3,4,5')
    parser.add_argument('--episodes-per-task', type=int, default=DEFAULT_EPISODES_PER_TASK)
    parser.add_argument('--evaluation-seed', type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--diagnostic-code-commit', default=None, help='User-supplied reviewed diagnostic code commit.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-model-k', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--assigned-gpu', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_model_k is None and args.dry_run == args.execute:
        parser.error('Exactly one of --dry-run or --execute is required')
    if args.worker_model_k is not None and (args.dry_run or args.execute):
        parser.error('D5 worker mode cannot combine --dry-run or --execute')
    if args.episodes_per_task <= 0:
        parser.error('--episodes-per-task must be positive')
    return args


def main(argv=None):
    args = _args(argv)
    if args.assigned_gpu is not None:
        # This occurs before a worker restores an agent and initializes JAX.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.assigned_gpu)
    try:
        if args.checkpoint != reference.LOCKED_REFERENCE_SELECTOR:
            raise ReevaluationError(
                f'M18-D5 only accepts --checkpoint {reference.LOCKED_REFERENCE_SELECTOR!r}; '
                'current semantic-best selection is prohibited'
            )
        if _parse_int_csv(args.train_ks, '--train-ks') != TRAIN_KS:
            raise ReevaluationError('M18-D5 requires exactly --train-ks 4,8')
        if args.worker_model_k is not None:
            if not args.diagnostic_code_commit:
                raise ReevaluationError('D5 worker requires --diagnostic-code-commit')
            output_dir = _worker_execute(args)
            print(f'M18-D5 worker completed: K{args.worker_model_k} output={output_dir}')
            return 0
        task_ids = _parse_int_csv(args.task_ids, '--task-ids', allowed=(1, 2, 3, 4, 5))
        args.gpus = _parse_gpus(args.gpus)
        plan_data = plan(
            args.reference_diagnostics_root,
            args.source_run_root,
            args.output_root,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            evaluation_seed=args.evaluation_seed,
        )
        output_dir = Path(plan_data['output_dir'])
        print(
            f'M18-D5 locked paired plan: pairs={len(plan_data["plan_rows"])} tasks={task_ids} '
            f'episodes_per_task={args.episodes_per_task} smoke_only={plan_data["smoke_only"]} output={output_dir}'
        )
        for model_k in TRAIN_KS:
            item = plan_data['provenance'][model_k]
            print(
                f'[LOCKED] K{model_k} role={item["resolved_checkpoint_role"]} '
                f'step={item["checkpoint_step"]} sha256={item["checkpoint_sha256"]} '
                f'path={item["checkpoint_path"]}'
            )
        if output_dir.exists():
            raise FileExistsError(f'M18-D5 output exists; refusing overwrite: {output_dir}')
        if args.dry_run:
            return 0
        if not args.diagnostic_code_commit:
            raise ReevaluationError('--execute requires --diagnostic-code-commit from the user-reviewed commit')
        output_dir.mkdir(parents=True)
        root_metadata_path = output_dir / 'metadata.json'
        root_metadata = {
            'status': 'running',
            'diagnostic_id': DIAGNOSTIC_ID,
            'diagnostic_family': 'paired_closed_loop_logical_rollout',
            'diagnostic_code_commit': str(args.diagnostic_code_commit),
            'source_study_id': STUDY_ID,
            'environment': ENVIRONMENT,
            'checkpoint_selector': reference.LOCKED_REFERENCE_SELECTOR,
            'paired_seed_scheme': COMMON_EPISODE_SEED_SCHEME,
            'evaluation_seed': int(args.evaluation_seed),
            'task_ids': list(task_ids),
            'episodes_per_task': int(args.episodes_per_task),
            'paired_episode_count': len(plan_data['plan_rows']),
            'smoke_only': bool(plan_data['smoke_only']),
            'reference_m18d_root': plan_data['contract']['reference_m18d_root'],
            'source_checkpoints': {
                f'K{k}': {
                    'source_checkpoint_role': plan_data['provenance'][k]['resolved_checkpoint_role'],
                    'source_checkpoint_step': plan_data['provenance'][k]['checkpoint_step'],
                    'source_checkpoint_sha256': plan_data['provenance'][k]['checkpoint_sha256'],
                    'source_checkpoint_path': plan_data['provenance'][k]['checkpoint_path'],
                    'reference_d1_summary_path': plan_data['provenance'][k]['reference_d1_summary_path'],
                    'reference_trace_metadata_path': plan_data['provenance'][k]['reference_trace_metadata_path'],
                }
                for k in TRAIN_KS
            },
            'evaluation_only': True,
            'finetuning': False,
            'optimizer_updates': 0,
        }
        _write_json(root_metadata_path, root_metadata)
        _, paired_goal_manifest = _create_paired_goal_manifest(output_dir, plan_data['plan_rows'])
        root_metadata.update({
            'shared_policy_goal_manifest_path': paired_goal_manifest['path'],
            'shared_policy_goal_manifest_metadata_path': paired_goal_manifest['metadata_path'],
            'shared_policy_goal_manifest_fingerprint_sha256': paired_goal_manifest['fingerprint'],
            'shared_policy_goal_contract': (
                'one real-environment goal per paired episode is stored and reused byte-for-byte by K4/K8'
            ),
        })
        _write_json(root_metadata_path, root_metadata)
        failures = _dispatch_workers(args)
        if failures:
            root_metadata.update({'status': 'failed', 'failure_reason': f'{failures} D5 worker(s) failed'})
            _write_json(root_metadata_path, root_metadata)
            raise ReevaluationError(f'M18-D5 has {failures} failed worker(s); refusing aggregate')
        summary = aggregate(output_dir, plan_data['plan_rows'])
        root_metadata.update({
            'status': 'completed',
            'source_checkpoint_immutable': bool(summary['source_checkpoint_immutable']),
            'artifacts': [
                str(output_dir / 'paired_episode_summary.csv'),
                str(output_dir / 'm18d_d5_summary.csv'),
                str(output_dir / 'm18d_d5_summary.json'),
            ],
        })
        _write_json(root_metadata_path, root_metadata)
        print(
            f'M18-D5 execute completed: pairs={summary["paired_episode_count"]} '
            f'optimizer_updates=0 output={output_dir}'
        )
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError, ReevaluationError, PuzzleLogicalError) as error:
        print(f'M18-D5: FAIL: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
