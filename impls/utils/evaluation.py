"""Deterministic OGBench-style evaluation for RLC agents.

The legacy :func:`evaluate` API intentionally keeps its historical return
signature.  ``evaluate_episodes`` is the streaming API used by post-hoc
checkpoint reevaluation; both APIs share the same episode rollout function.
"""

from collections import defaultdict
import json

import jax
import numpy as np
from tqdm import trange

from .reproducibility import derive_seed


COMMON_EPISODE_SEED_SCHEME = 'common_task_episode_v1'


def supply_rng(function, rng=jax.random.PRNGKey(0)):
    """Split a JAX key before each policy call."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return function(*args, seed=key, **kwargs)

    return wrapped


def flatten(dictionary, parent_key='', sep='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if hasattr(value, 'items'):
            items.extend(flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for key, value in single_dict.items():
        dict_of_lists[key].append(value)


def common_episode_seeds(evaluation_seed, task_id, episode_index, scheme=COMMON_EPISODE_SEED_SCHEME):
    """Return the versioned common-random-number seed hierarchy.

    ``task_id`` is deliberately the only policy-independent task component;
    configuration ID, training seed, checkpoint hash, and GPU ID must not
    affect these seeds.
    """

    if scheme != COMMON_EPISODE_SEED_SCHEME:
        raise ValueError(f'Unsupported episode seed scheme: {scheme!r}')
    task_id = int(task_id)
    episode_index = int(episode_index)
    if task_id < 0 or episode_index < 0:
        raise ValueError('task_id and episode_index must be non-negative')
    task_seed = derive_seed(evaluation_seed, task_id)
    return {
        'task_seed': task_seed,
        'episode_seed': derive_seed(task_seed, episode_index, 0),
        'actor_seed': derive_seed(task_seed, episode_index, 1),
        'noise_seed': derive_seed(task_seed, episode_index, 2),
    }


def _scalar_value(value):
    array = np.asarray(value)
    if array.size != 1:
        return None
    item = array.reshape(-1)[0]
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (int, float, np.integer, np.floating)):
        value = float(item)
        if np.isfinite(value):
            return value
    return None


def scalar_info(info):
    """Return stable scalar-only flattened final-info fields."""

    result = {}
    for key, value in flatten(info or {}).items():
        scalar = _scalar_value(value)
        if scalar is not None:
            result[key] = scalar
    return result


def extract_episode_success(info):
    """Extract one unambiguous scalar success signal from final ``info``.

    OGBench environments expose ``info['success']``.  The suffix fallback is
    retained for compatible wrappers, but conflicting scalar success fields
    fail loudly instead of silently averaging different scientific signals.
    """

    flattened = scalar_info(info)
    exact = [(key, value) for key, value in flattened.items() if key == 'success']
    candidates = exact or [
        (key, value)
        for key, value in flattened.items()
        if key.endswith('.success') or key.endswith('/success') or key.endswith('_success')
    ]
    if not candidates:
        raise KeyError(f'No scalar success signal found in final info: {sorted(flattened)}')
    values = [float(value) for _, value in candidates]
    if any(not np.isfinite(value) for value in values):
        raise ValueError(f'Non-finite success signal in final info: {candidates!r}')
    if any(abs(value - values[0]) > 1e-8 for value in values[1:]):
        raise ValueError(f'Ambiguous success signals in final info: {candidates!r}')
    return float(values[0])


def _rollout_episode(
    agent,
    env,
    *,
    task_id,
    config,
    episode_seed,
    actor_seed,
    noise_seed,
    eval_temperature,
    eval_gaussian,
    retain_trajectory,
    render,
    video_frame_skip,
):
    """Run one episode for both legacy and streaming evaluation callers."""

    trajectory = defaultdict(list) if retain_trajectory else None
    reset_options = {'render_goal': bool(render)}
    if task_id is not None:
        reset_options['task_id'] = task_id
    observation, info = env.reset(seed=int(episode_seed), options=reset_options)
    goal = info.get('goal')
    goal_frame = info.get('goal_rendered')
    done = False
    step = 0
    episode_return = 0.0
    render_frames = []
    final_info = info
    noise_rng = np.random.default_rng(int(noise_seed))
    actor_fn = supply_rng(
        agent.sample_actions,
        rng=jax.random.PRNGKey(int(actor_seed)),
    )

    while not done:
        action = np.asarray(
            actor_fn(
                observations=observation,
                goals=goal,
                temperature=eval_temperature,
            )
        )
        if not config.get('discrete', False):
            if eval_gaussian is not None:
                action = noise_rng.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)
        next_observation, reward, terminated, truncated, final_info = env.step(action)
        done = bool(terminated or truncated)
        episode_return += float(reward)
        step += 1

        if render and (step % video_frame_skip == 0 or done):
            frame = env.render().copy()
            render_frames.append(
                np.concatenate([goal_frame, frame], axis=0)
                if goal_frame is not None else frame
            )

        if retain_trajectory:
            add_to(
                trajectory,
                dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    info=final_info,
                ),
            )
        observation = next_observation

    return {
        'final_info': final_info,
        'trajectory': trajectory,
        'render': render_frames,
        'episode_return': float(episode_return),
        'episode_length': int(step),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
    }


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    seed=None,
):
    """Evaluate HIQL while keeping environment, policy, and noise streams independent."""
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    config = {} if config is None else config
    trajs = []
    stats = defaultdict(list)
    renders = []

    for episode in trange(num_eval_episodes + num_video_episodes, leave=False):
        should_render = episode >= num_eval_episodes
        episode_seed = derive_seed(seed, episode, 0)
        actor_seed = derive_seed(seed, episode, 1)
        rollout = _rollout_episode(
            agent,
            env,
            task_id=task_id,
            config=config,
            episode_seed=episode_seed,
            actor_seed=actor_seed,
            noise_seed=derive_seed(seed, episode, 2),
            eval_temperature=eval_temperature,
            eval_gaussian=eval_gaussian,
            retain_trajectory=True,
            render=should_render,
            video_frame_skip=video_frame_skip,
        )

        if episode < num_eval_episodes:
            add_to(stats, flatten(rollout['final_info']))
            trajs.append(rollout['trajectory'])
        else:
            renders.append(np.asarray(rollout['render']))

    for key, values in stats.items():
        stats[key] = np.mean(values)
    return stats, trajs, renders


def evaluate_episodes(
    agent,
    env,
    *,
    task_id,
    task_name,
    config=None,
    evaluation_seed,
    episode_indices,
    eval_temperature=0,
    eval_gaussian=None,
    seed_scheme=COMMON_EPISODE_SEED_SCHEME,
):
    """Stream compact, deterministically seeded episode records.

    No trajectory is retained.  The caller may pass any subset of episode
    indices, which is what makes safe resume possible.
    """

    config = {} if config is None else config
    records = []
    for episode_index in episode_indices:
        seeds = common_episode_seeds(
            evaluation_seed,
            task_id,
            episode_index,
            scheme=seed_scheme,
        )
        rollout = _rollout_episode(
            agent,
            env,
            task_id=task_id,
            config=config,
            episode_seed=seeds['episode_seed'],
            actor_seed=seeds['actor_seed'],
            noise_seed=seeds['noise_seed'],
            eval_temperature=eval_temperature,
            eval_gaussian=eval_gaussian,
            retain_trajectory=False,
            render=False,
            video_frame_skip=1,
        )
        final_info = scalar_info(rollout['final_info'])
        records.append({
            'task_id': int(task_id),
            'task_name': str(task_name),
            'episode_index': int(episode_index),
            'evaluation_seed': int(evaluation_seed),
            'task_seed': seeds['task_seed'],
            'episode_seed': seeds['episode_seed'],
            'actor_seed': seeds['actor_seed'],
            'noise_seed': seeds['noise_seed'],
            'success': extract_episode_success(rollout['final_info']),
            'episode_return': rollout['episode_return'],
            'episode_length': rollout['episode_length'],
            'terminated': rollout['terminated'],
            'truncated': rollout['truncated'],
            'paired_episode_id': f'task{int(task_id):02d}_ep{int(episode_index):03d}',
            'final_info_json': json.dumps(final_info, sort_keys=True, separators=(',', ':')),
        })
    return records
