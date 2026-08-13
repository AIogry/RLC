"""OGBench dataset wrappers used by the HIQL runtime.

The sampling equations follow the OGBench implementation.  The explicit
``numpy.random.Generator`` plumbing follows the reproducibility-fixed CoGHP
runtime and keeps dataset sampling independent from process-global RNG state.
"""

import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of a dataset tree."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Immutable OGBench dataset with an explicit sampling RNG."""

    @classmethod
    def create(cls, freeze=True, seed=None, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data, seed=seed)

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs, rng=None):
        rng = self.rng if rng is None else rng
        if 'valids' in self._dict:
            return self.valid_idxs[rng.integers(len(self.valid_idxs), size=num_idxs)]
        return rng.integers(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None, rng=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size, rng=rng)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned OGBench dataset wrapper."""

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True
    rng: Any = dataclasses.field(default_factory=np.random.default_rng, repr=False)

    def __post_init__(self):
        if not isinstance(self.rng, np.random.Generator):
            self.rng = np.random.default_rng(self.rng)
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert len(self.terminal_locs) > 0 and self.terminal_locs[-1] == self.size - 1

        assert np.isclose(
            self.config['value_p_curgoal']
            + self.config['value_p_trajgoal']
            + self.config['value_p_randomgoal'],
            1.0,
        )
        assert np.isclose(
            self.config['actor_p_curgoal']
            + self.config['actor_p_trajgoal']
            + self.config['actor_p_randomgoal'],
            1.0,
        )

        if self.config['frame_stack'] is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False, rng=None):
        rng = self.rng if rng is None else rng
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size, rng=rng)

        batch = self.dataset.sample(batch_size, idxs, rng=rng)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
            rng,
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
            rng,
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if rng.random() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'], rng)
        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, rng):
        del p_randomgoal
        batch_size = len(idxs)
        random_goal_idxs = self.dataset.get_random_idxs(batch_size, rng=rng)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = rng.geometric(p=1 - self.config['discount'], size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = rng.random(batch_size)
            traj_goal_idxs = np.round(
                np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances)
            ).astype(int)
        goal_idxs = np.where(
            rng.random(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6),
            traj_goal_idxs,
            random_goal_idxs,
        )
        return np.where(rng.random(batch_size) < p_curgoal, idxs, goal_idxs)

    def augment(self, batch, keys, rng):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = rng.integers(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Hierarchical goal-conditioned wrapper used by HIQL."""

    def sample(self, batch_size, idxs=None, evaluation=False, rng=None):
        rng = self.rng if rng is None else rng
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size, rng=rng)

        batch = self.dataset.sample(batch_size, idxs, rng=rng)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
            rng,
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        if self.config['actor_geom_sample']:
            offsets = rng.geometric(p=1 - self.config['discount'], size=batch_size)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = rng.random(batch_size)
            high_traj_goal_idxs = np.round(
                np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances)
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size, rng=rng)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        pick_random = rng.random(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if rng.random() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                    rng,
                )
        return batch
