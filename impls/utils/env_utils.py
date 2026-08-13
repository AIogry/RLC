"""Environment helpers bound to the canonical RLC OGBench package."""

import collections
import os

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from ogbench.utils import DEFAULT_DATASET_DIR

from .datasets import Dataset


def resolve_dataset_dir(dataset_dir=None):
    """Resolve the dataset directory without falling back to a legacy path."""
    if dataset_dir is not None:
        return os.path.abspath(os.path.expanduser(dataset_dir))
    return os.path.abspath(
        os.path.expanduser(
            os.environ.get('OGBENCH_DATASET_DIR', DEFAULT_DATASET_DIR)
        )
    )


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}
        if terminated or truncated:
            info['episode'] = {
                'return': self.reward_sum,
                'length': self.episode_length,
            }
            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = self.unwrapped.get_normalized_score(self.reward_sum) * 100.0
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Stack observations along the final dimension."""

    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(observation)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None, seed=None, dataset_seed=None, dataset_dir=None):
    """Create an RLC OGBench environment and seeded raw Dataset wrappers."""
    resolved_dir = resolve_dataset_dir(dataset_dir)
    print(f'Imported ogbench module: {os.path.abspath(ogbench.__file__)}')
    print(f'Environment/dataset name: {dataset_name}')
    print(f'Resolved dataset directory: {resolved_dir}')

    env, train_data, val_data = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=resolved_dir,
        compact_dataset=True,
    )
    train_dataset = Dataset.create(seed=dataset_seed, **train_data)
    val_dataset = (
        None
        if val_data is None
        else Dataset.create(seed=None if dataset_seed is None else dataset_seed + 1, **val_data)
    )
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    env.reset(seed=seed)
    return env, train_dataset, val_dataset
