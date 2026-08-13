"""Deterministic OGBench-style evaluation for HIQL."""

from collections import defaultdict

import jax
import numpy as np
from tqdm import trange

from .reproducibility import derive_seed


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
        trajectory = defaultdict(list)
        should_render = episode >= num_eval_episodes
        episode_seed = derive_seed(seed, episode, 0)
        actor_seed = derive_seed(seed, episode, 1)
        noise_rng = np.random.default_rng(derive_seed(seed, episode, 2))
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_seed))
        reset_options = {'render_goal': should_render}
        if task_id is not None:
            reset_options['task_id'] = task_id
        observation, info = env.reset(seed=episode_seed, options=reset_options)
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = np.asarray(actor_fn(observations=observation, goals=goal, temperature=eval_temperature))
            if not config.get('discrete', False):
                if eval_gaussian is not None:
                    action = noise_rng.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(np.concatenate([goal_frame, frame], axis=0) if goal_frame is not None else frame)

            add_to(
                trajectory,
                dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                ),
            )
            observation = next_observation

        if episode < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(trajectory)
        else:
            renders.append(np.asarray(render))

    for key, values in stats.items():
        stats[key] = np.mean(values)
    return stats, trajs, renders
