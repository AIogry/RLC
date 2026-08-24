"""Formal evaluator rollout banks and deterministic progress balancing."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from impls.utils.evaluation import _rollout_episode, common_episode_seeds, extract_episode_success

DEFAULT_PROGRESS_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def environment_task_ids(env):
    task_infos = getattr(env, 'task_infos', None)
    if task_infos is None:
        task_infos = getattr(getattr(env, 'unwrapped', None), 'task_infos', None)
    if task_infos is None:
        raise AttributeError('Formal evaluator environment has no task_infos')
    return list(range(1, len(task_infos) + 1))


def _task_infos(env):
    return getattr(env, 'task_infos', None) or getattr(getattr(env, 'unwrapped', None), 'task_infos', None)


def eval_goals_from_resets(env, *, task_ids, evaluation_seed):
    goals, names = {}, {}
    infos = _task_infos(env)
    for task_id in task_ids:
        seeds = common_episode_seeds(evaluation_seed, task_id, 0)
        _, info = env.reset(
            seed=int(seeds['episode_seed']),
            options={'task_id': int(task_id), 'render_goal': False},
        )
        if 'goal' not in info:
            raise KeyError(f'Formal reset did not expose info[goal] for task {task_id}')
        goals[int(task_id)] = np.asarray(info['goal']).copy()
        task_info = infos[int(task_id) - 1] if infos is not None else {}
        names[int(task_id)] = str(task_info.get('task_name', f'task_{task_id}'))
    return goals, names


def collect_rollout_records(actor, env, *, actor_name, task_ids, episodes,
                            evaluation_seed, eval_temperature=0.0, eval_gaussian=None):
    records = []
    for task_id in task_ids:
        for episode_index in range(int(episodes)):
            seeds = common_episode_seeds(evaluation_seed, task_id, episode_index)
            rollout = _rollout_episode(
                actor.agent, env, task_id=int(task_id), config=actor.config,
                episode_seed=seeds['episode_seed'], actor_seed=seeds['actor_seed'],
                noise_seed=seeds['noise_seed'], eval_temperature=float(eval_temperature),
                eval_gaussian=eval_gaussian, retain_trajectory=True, render=False,
                video_frame_skip=3,
            )
            goal = rollout['original_eval_goal']
            if goal is None:
                raise ValueError(f'No evaluator goal for task {task_id}')
            trajectory = rollout['trajectory']
            length = int(rollout['episode_length'])
            success = extract_episode_success(rollout['final_info'])
            for timestep, observation in enumerate(trajectory['observation']):
                progress = 0.0 if length <= 1 else timestep / (length - 1)
                records.append({
                    'actor_name': str(actor_name),
                    'actor_seed': int(seeds['actor_seed']),
                    'critic_seed': int(actor.seed),
                    'task_id': int(task_id),
                    'episode_index': int(episode_index),
                    'timestep': int(timestep),
                    'episode_length': length,
                    'progress': float(progress),
                    'episode_success': float(success),
                    'observation': np.asarray(observation).copy(),
                    'eval_goal': np.asarray(goal).copy(),
                })
    return records


def _bin_index(progress, bins):
    for index, (left, right) in enumerate(zip(bins, bins[1:])):
        if (progress >= left and progress < right) or (index == len(bins) - 2 and progress <= right):
            return index
    return None


def select_progress_balanced_states(records, bins=DEFAULT_PROGRESS_BINS):
    bins = tuple(float(value) for value in bins)
    if len(bins) < 2 or any(r <= l for l, r in zip(bins, bins[1:])):
        raise ValueError(f'Invalid progress bins: {bins}')
    grouped = defaultdict(list)
    for record in records:
        index = _bin_index(float(record['progress']), bins)
        if index is not None:
            grouped[(record['actor_name'], record['task_id'], record['episode_index'], index)].append(record)
    selected = []
    for key in sorted(grouped, key=lambda value: tuple(map(str, value))):
        index = key[-1]
        midpoint = (bins[index] + bins[index + 1]) / 2
        choice = min(grouped[key], key=lambda row: (abs(row['progress'] - midpoint), row['timestep']))
        row = dict(choice)
        row['progress_bin'] = int(index)
        selected.append(row)
    return selected


def build_rollout_bank(records, *, actor_names, task_names, bins=DEFAULT_PROGRESS_BINS,
                       environment, source_commit, evaluation_seed, episodes_per_task,
                       provenance=None):
    selected = select_progress_balanced_states(records, bins=bins)
    actor_names = list(actor_names)
    task_ids = sorted(int(k) for k in task_names)
    expected = {
        (actor, task, episode, index)
        for actor in actor_names for task in task_ids
        for episode in range(int(episodes_per_task))
        for index in range(len(tuple(bins)) - 1)
    }
    observed = {(r['actor_name'], r['task_id'], r['episode_index'], r['progress_bin']) for r in selected}
    missing = sorted(expected - observed, key=lambda value: tuple(map(str, value)))
    if missing:
        raise ValueError(f'Unbalanced rollout bank; missing cells: {missing[:20]} total={len(missing)}')
    selected.sort(key=lambda row: (
        actor_names.index(row['actor_name']), row['task_id'], row['episode_index'],
        row['progress_bin'], row['timestep'],
    ))
    codes = {name: index for index, name in enumerate(actor_names)}
    arrays = {
        'observations': np.stack([r['observation'] for r in selected]),
        'eval_goals': np.stack([r['eval_goal'] for r in selected]),
        'origin_actor_code': np.asarray([codes[r['actor_name']] for r in selected], dtype=np.int64),
        'actor_seed': np.asarray([r['actor_seed'] for r in selected], dtype=np.int64),
        'critic_seed': np.asarray([r['critic_seed'] for r in selected], dtype=np.int64),
        'task_id': np.asarray([r['task_id'] for r in selected], dtype=np.int64),
        'episode_index': np.asarray([r['episode_index'] for r in selected], dtype=np.int64),
        'timestep': np.asarray([r['timestep'] for r in selected], dtype=np.int64),
        'episode_length': np.asarray([r['episode_length'] for r in selected], dtype=np.int64),
        'normalized_progress': np.asarray([r['progress'] for r in selected], dtype=np.float32),
        'episode_success': np.asarray([r['episode_success'] for r in selected], dtype=np.float32),
        'progress_bin': np.asarray([r['progress_bin'] for r in selected], dtype=np.int64),
    }
    rows = [{
        'sample_index': i, 'origin_actor': r['actor_name'], 'task_id': r['task_id'],
        'task_name': task_names[r['task_id']], 'episode_index': r['episode_index'],
        'timestep': r['timestep'], 'progress_bin': r['progress_bin'],
    } for i, r in enumerate(selected)]
    manifest = {
        'bank_type': 'B_R', 'environment': environment, 'source_commit': source_commit,
        'evaluation_seed': int(evaluation_seed),
        'episodes_per_actor_task': int(episodes_per_task), 'actor_names': actor_names,
        'task_names': {str(k): v for k, v in task_names.items()},
        'progress_bins': list(map(float, bins)),
        'balancing': 'one_midpoint_nearest_state_per_actor_task_episode_bin',
        'same_state_goal_pairs_for_cross_evaluation': True,
        'provenance': provenance or {},
    }
    return arrays, manifest, rows

