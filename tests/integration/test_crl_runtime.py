"""Reference, computation-slot, and real-data parity tests for CRL."""

import copy
import importlib.util
import os
import pathlib
import sys
import unittest
from collections.abc import Mapping

import flax
import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.crl import CRLAgent, get_config
from impls.computation.accounting import count_parameters
from impls.main import _make_config, _parse_args
from impls.utils.datasets import Dataset, GCDataset
from impls.utils.env_utils import make_env_and_datasets, resolve_dataset_dir


ATOL = 1e-6
DATASET_NAME = 'antmaze-medium-navigate-v0'
REQUIRED_BATCH_KEYS = ('observations', 'value_goals', 'actor_goals', 'actions')


def _small_config(
    actor_enabled=False,
    actor_loss='ddpgbc',
    critic_state_enabled=False,
    critic_goal_enabled=False,
    value_state_enabled=False,
    value_goal_enabled=False,
):
    config = get_config()
    config['batch_size'] = 4
    config['actor_hidden_dims'] = (6, 6)
    config['value_hidden_dims'] = (6, 6)
    config['latent_dim'] = 3
    config['actor_loss'] = actor_loss
    config['p_aug'] = 0.0
    config['compute']['actor']['enabled'] = actor_enabled
    config['compute']['critic_state']['enabled'] = critic_state_enabled
    config['compute']['critic_goal']['enabled'] = critic_goal_enabled
    config['compute']['value_state']['enabled'] = value_state_enabled
    config['compute']['value_goal']['enabled'] = value_goal_enabled
    return config


def _batch(batch_size=4, obs_dim=4, action_dim=2):
    observations = jnp.arange(batch_size * obs_dim, dtype=jnp.float32).reshape(batch_size, obs_dim) / 11.0
    goals = jnp.flip(observations, axis=0) + 0.17
    actions = jnp.arange(batch_size * action_dim, dtype=jnp.float32).reshape(batch_size, action_dim) / 13.0 - 0.2
    return {
        'observations': observations,
        'value_goals': goals,
        'actor_goals': goals * 0.7,
        'actions': actions,
    }


def _leaf_items(tree, prefix=()):
    if isinstance(tree, Mapping):
        for key in sorted(tree):
            yield from _leaf_items(tree[key], prefix + (str(key),))
    else:
        yield prefix, tree


def _path_get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _actor_body_path(actor_params, enabled):
    return ('actor_net', 'topology', 'primitive') if enabled else ('actor_net',)


def _critic_body_path(critic_params, branch, enabled):
    if not enabled:
        return (branch,)
    return (branch, 'core', 'topology', 'primitive')


def _value_body_path(value_params, branch, enabled):
    if not enabled:
        return (branch,)
    return (branch, 'core', 'topology', 'primitive')


def _semantic_tree(
    params,
    actor_enabled,
    critic_state_enabled=False,
    critic_goal_enabled=False,
    value_state_enabled=False,
    value_goal_enabled=False,
):
    """Normalize actor, critic, and AWR value computation wrappers."""
    result = {}
    for root_key, root_value in params.items():
        if root_key == 'modules_actor':
            body = _path_get(root_value, _actor_body_path(root_value, actor_enabled))
            for path, value in _leaf_items(body):
                result[('actor', 'body') + path] = value
            for module_key, module_value in root_value.items():
                if module_key != 'actor_net':
                    for path, value in _leaf_items(module_value):
                        result[('actor', str(module_key)) + path] = value
        elif root_key == 'modules_critic':
            critic_slots = {'phi': critic_state_enabled, 'psi': critic_goal_enabled}
            for branch, enabled in critic_slots.items():
                body = _path_get(root_value, _critic_body_path(root_value, branch, enabled))
                for path, value in _leaf_items(body):
                    result[('critic', branch, 'body') + path] = value
        elif root_key == 'modules_value':
            value_slots = {'phi': value_state_enabled, 'psi': value_goal_enabled}
            for branch, enabled in value_slots.items():
                body = _path_get(root_value, _value_body_path(root_value, branch, enabled))
                for path, value in _leaf_items(body):
                    result[('value', branch, 'body') + path] = value
        else:
            for path, value in _leaf_items(root_value):
                result[(str(root_key),) + path] = value
    return result


def _tree_error(
    old_tree,
    new_tree,
    old_actor_enabled=False,
    new_actor_enabled=True,
    old_critic_state_enabled=False,
    new_critic_state_enabled=False,
    old_critic_goal_enabled=False,
    new_critic_goal_enabled=False,
    old_value_state_enabled=False,
    new_value_state_enabled=False,
    old_value_goal_enabled=False,
    new_value_goal_enabled=False,
):
    old_items = _semantic_tree(
        old_tree,
        old_actor_enabled,
        old_critic_state_enabled,
        old_critic_goal_enabled,
        old_value_state_enabled,
        old_value_goal_enabled,
    )
    new_items = _semantic_tree(
        new_tree,
        new_actor_enabled,
        new_critic_state_enabled,
        new_critic_goal_enabled,
        new_value_state_enabled,
        new_value_goal_enabled,
    )
    if set(old_items) != set(new_items):
        raise AssertionError(f'semantic labels differ: {sorted(set(old_items) ^ set(new_items))[:5]}')
    max_error = 0.0
    for label in old_items:
        old_value = old_items[label]
        new_value = new_items[label]
        if old_value is None or new_value is None:
            if old_value is not new_value:
                raise AssertionError(f'None mismatch at {label}')
            continue
        old_array = np.asarray(old_value)
        new_array = np.asarray(new_value)
        if old_array.shape != new_array.shape:
            raise AssertionError(f'shape mismatch at {label}: {old_array.shape} vs {new_array.shape}')
        if old_array.size:
            max_error = max(max_error, float(np.max(np.abs(old_array - new_array))))
    return max_error


def _branch_tree_error(old_tree, new_tree, root_key, branch, old_enabled, new_enabled):
    """Compare one semantic representation branch across parameter trees."""
    old_body = _path_get(old_tree[root_key], _value_body_path(old_tree[root_key], branch, old_enabled))
    new_body = _path_get(new_tree[root_key], _value_body_path(new_tree[root_key], branch, new_enabled))
    old_items = dict(_leaf_items(old_body))
    new_items = dict(_leaf_items(new_body))
    if set(old_items) != set(new_items):
        raise AssertionError(f'branch labels differ for {root_key}/{branch}')
    max_error = 0.0
    for path in old_items:
        old_array = np.asarray(old_items[path])
        new_array = np.asarray(new_items[path])
        if old_array.shape != new_array.shape:
            raise AssertionError(f'branch shape differs for {root_key}/{branch}/{path}')
        if old_array.size:
            max_error = max(max_error, float(np.max(np.abs(old_array - new_array))))
    return max_error


def _graft_actor(old_params, new_params):
    return _graft_semantic_params(old_params, new_params, actor_enabled=True)


def _graft_semantic_params(
    old_params,
    new_params,
    *,
    actor_enabled=False,
    critic_state_enabled=False,
    critic_goal_enabled=False,
    value_state_enabled=False,
    value_goal_enabled=False,
):
    old_root = flax.core.unfreeze(old_params)
    new_root = flax.core.unfreeze(new_params)
    special_roots = {'modules_actor', 'modules_critic', 'modules_value'}
    for root_key in old_root:
        if root_key not in special_roots:
            new_root[root_key] = old_root[root_key]

    old_actor = old_root['modules_actor']
    new_actor = new_root['modules_actor']
    if actor_enabled:
        new_actor['actor_net']['topology']['primitive'] = old_actor['actor_net']
    else:
        new_actor['actor_net'] = old_actor['actor_net']
    for key, value in old_actor.items():
        if key != 'actor_net':
            new_actor[key] = value
    new_root['modules_actor'] = new_actor

    old_critic = old_root['modules_critic']
    new_critic = new_root['modules_critic']
    for branch, enabled in (
        ('phi', critic_state_enabled),
        ('psi', critic_goal_enabled),
    ):
        if enabled:
            new_critic[branch]['core']['topology']['primitive'] = old_critic[branch]
        else:
            new_critic[branch] = old_critic[branch]
    new_root['modules_critic'] = new_critic

    if 'modules_value' in old_root:
        old_value = old_root['modules_value']
        new_value = new_root['modules_value']
        for branch, enabled in (
            ('phi', value_state_enabled),
            ('psi', value_goal_enabled),
        ):
            if enabled:
                new_value[branch]['core']['topology']['primitive'] = old_value[branch]
            else:
                new_value[branch] = old_value[branch]
        new_root['modules_value'] = new_value
    return new_root


def _assert_info_equal(testcase, old_info, new_info, context, skip=()):
    testcase.assertEqual(set(old_info) - set(skip), set(new_info) - set(skip), msg=context)
    max_error = 0.0
    for key in sorted(set(old_info) - set(skip)):
        error = float(np.max(np.abs(np.asarray(old_info[key]) - np.asarray(new_info[key]))))
        max_error = max(max_error, error)
        testcase.assertTrue(np.allclose(old_info[key], new_info[key], rtol=0.0, atol=ATOL), msg=f'{context}: {key}')
    return max_error


def _load_reference_crl():
    ref_impls = '/home/eai/Research/offline_rl_baselines/ogbench/impls'
    if not os.path.exists(os.path.join(ref_impls, 'agents', 'crl.py')):
        raise unittest.SkipTest('offline_rl_baselines reference CRL is unavailable')
    if ref_impls not in sys.path:
        sys.path.insert(0, ref_impls)
    module_name = 'rlc_reference_crl_agent'
    if module_name not in sys.modules:
        path = pathlib.Path(ref_impls) / 'agents' / 'crl.py'
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


class CRLReferenceMigrationTest(unittest.TestCase):
    def test_reference_crl_ddpgbc_and_awr_forward_gradient_and_update(self):
        reference = _load_reference_crl()
        for actor_loss in ('ddpgbc', 'awr'):
            with self.subTest(actor_loss=actor_loss):
                config = _small_config(False, actor_loss)
                reference_config = reference.get_config()
                for key, value in config.items():
                    if key != 'compute':
                        reference_config[key] = copy.deepcopy(value)
                reference_config['encoder'] = None
                reference_config['frame_stack'] = None
                batch = _batch()
                old_agent = CRLAgent.create(17, batch['observations'][:2], batch['actions'][:2], config)
                reference_agent = reference.CRLAgent.create(
                    17, batch['observations'][:2], batch['actions'][:2], reference_config,
                )
                self.assertEqual(_tree_error(old_agent.network.params, reference_agent.network.params, False, False), 0.0)

                old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
                ref_loss, ref_info = reference_agent.total_loss(batch, reference_agent.network.params)
                np.testing.assert_array_equal(np.asarray(old_loss), np.asarray(ref_loss))
                _assert_info_equal(self, old_info, ref_info, f'reference {actor_loss} forward')

                old_grads = jax.grad(lambda params: old_agent.total_loss(batch, params)[0])(old_agent.network.params)
                ref_grads = jax.grad(lambda params: reference_agent.total_loss(batch, params)[0])(reference_agent.network.params)
                self.assertEqual(_tree_error(old_grads, ref_grads, False, False), 0.0)

                old_agent, old_update = old_agent.update(batch)
                reference_agent, ref_update = reference_agent.update(batch)
                _assert_info_equal(self, old_update, ref_update, f'reference {actor_loss} update', skip=('grad/norm',))
                self.assertEqual(_tree_error(old_agent.network.params, reference_agent.network.params, False, False), 0.0)
                np.testing.assert_array_equal(np.asarray(old_agent.rng), np.asarray(reference_agent.rng))


class CRLRuntimeConfigTest(unittest.TestCase):
    def test_computation_flag_enables_value_slots_only_for_awr(self):
        for actor_loss, expected_value_enabled in (('ddpgbc', False), ('awr', True)):
            args = _parse_args(['--agent', 'crl', '--actor_loss', actor_loss, '--computation'])
            config = _make_config(args)
            enabled = {
                name: bool(config['compute'][name]['enabled'])
                for name in ('actor', 'critic_state', 'critic_goal', 'value_state', 'value_goal')
            }
            self.assertEqual(enabled['actor'], True)
            self.assertEqual(enabled['critic_state'], True)
            self.assertEqual(enabled['critic_goal'], True)
            self.assertEqual(enabled['value_state'], expected_value_enabled)
            self.assertEqual(enabled['value_goal'], expected_value_enabled)


class CRLActorComputationParityTest(unittest.TestCase):
    def test_legacy_and_computation_actor_match_after_semantic_graft(self):
        for actor_loss in ('ddpgbc', 'awr'):
            with self.subTest(actor_loss=actor_loss):
                old_config = _small_config(False, actor_loss)
                new_config = _small_config(True, actor_loss)
                batch = _batch()
                old_agent = CRLAgent.create(23, batch['observations'][:2], batch['actions'][:2], old_config)
                new_agent = CRLAgent.create(23, batch['observations'][:2], batch['actions'][:2], new_config)
                new_agent = new_agent.replace(network=new_agent.network.replace(
                    params=_graft_actor(old_agent.network.params, new_agent.network.params)
                ))

                old_dist = old_agent.network.select('actor')(batch['observations'], batch['actor_goals'])
                new_dist = new_agent.network.select('actor')(batch['observations'], batch['actor_goals'])
                np.testing.assert_array_equal(np.asarray(old_dist.mode()), np.asarray(new_dist.mode()))
                np.testing.assert_array_equal(np.asarray(old_dist.scale_diag), np.asarray(new_dist.scale_diag))
                np.testing.assert_array_equal(
                    np.asarray(old_dist.log_prob(batch['actions'])),
                    np.asarray(new_dist.log_prob(batch['actions'])),
                )

                old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
                new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
                np.testing.assert_array_equal(np.asarray(old_loss), np.asarray(new_loss))
                _assert_info_equal(self, old_info, new_info, f'computation {actor_loss} total')
                old_grads = jax.grad(lambda params: old_agent.total_loss(batch, params)[0])(old_agent.network.params)
                new_grads = jax.grad(lambda params: new_agent.total_loss(batch, params)[0])(new_agent.network.params)
                self.assertEqual(_tree_error(old_grads, new_grads, False, True), 0.0)

                old_agent, old_update = old_agent.update(batch)
                new_agent, new_update = new_agent.update(batch)
                _assert_info_equal(self, old_update, new_update, f'computation {actor_loss} update', skip=('grad/norm',))
                self.assertEqual(_tree_error(old_agent.network.params, new_agent.network.params, False, True), 0.0)
                np.testing.assert_array_equal(np.asarray(old_agent.rng), np.asarray(new_agent.rng))

    def test_actor_parameter_count_is_unchanged_by_computation_wrapper(self):
        batch = _batch()
        old = CRLAgent.create(31, batch['observations'][:2], batch['actions'][:2], _small_config(False))
        new = CRLAgent.create(31, batch['observations'][:2], batch['actions'][:2], _small_config(True))
        old_actor = old.network.params['modules_actor']['actor_net']
        new_actor = new.network.params['modules_actor']['actor_net']['topology']['primitive']
        self.assertEqual(count_parameters(old_actor), count_parameters(new_actor))
        self.assertEqual(
            count_parameters(old.network.params['modules_actor']),
            count_parameters(new.network.params['modules_actor']),
        )


class CRLCriticComputationParityTest(unittest.TestCase):
    def _make_pair(self, actor_loss):
        old_config = _small_config(False, actor_loss)
        new_config = _small_config(
            False,
            actor_loss,
            critic_state_enabled=True,
            critic_goal_enabled=True,
        )
        batch = _batch()
        old_agent = CRLAgent.create(61, batch['observations'][:2], batch['actions'][:2], old_config)
        new_agent = CRLAgent.create(61, batch['observations'][:2], batch['actions'][:2], new_config)
        new_agent = new_agent.replace(network=new_agent.network.replace(
            params=_graft_semantic_params(
                old_agent.network.params,
                new_agent.network.params,
                critic_state_enabled=True,
                critic_goal_enabled=True,
            )
        ))
        return old_agent, new_agent, batch

    def test_forward_loss_gradient_and_one_step_update_match(self):
        for actor_loss in ('ddpgbc', 'awr'):
            with self.subTest(actor_loss=actor_loss):
                old_agent, new_agent, batch = self._make_pair(actor_loss)
                old_critic = old_agent.network.select('critic')(
                    batch['observations'], batch['value_goals'], batch['actions'], info=True,
                )
                new_critic = new_agent.network.select('critic')(
                    batch['observations'], batch['value_goals'], batch['actions'], info=True,
                )
                for old_value, new_value in zip(old_critic, new_critic):
                    np.testing.assert_array_equal(np.asarray(old_value), np.asarray(new_value))

                old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
                new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
                np.testing.assert_array_equal(np.asarray(old_loss), np.asarray(new_loss))
                _assert_info_equal(self, old_info, new_info, f'critic computation {actor_loss} loss')

                old_grads = jax.grad(lambda params: old_agent.total_loss(batch, params)[0])(
                    old_agent.network.params,
                )
                new_grads = jax.grad(lambda params: new_agent.total_loss(batch, params)[0])(
                    new_agent.network.params,
                )
                self.assertEqual(
                    _tree_error(
                        old_grads,
                        new_grads,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                    ),
                    0.0,
                )

                old_agent, old_update = old_agent.update(batch)
                new_agent, new_update = new_agent.update(batch)
                _assert_info_equal(
                    self,
                    old_update,
                    new_update,
                    f'critic computation {actor_loss} update',
                    skip=('grad/norm',),
                )
                self.assertEqual(
                    _tree_error(
                        old_agent.network.params,
                        new_agent.network.params,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                    ),
                    0.0,
                )
                for state_name in ('mu', 'nu'):
                    self.assertEqual(
                        _tree_error(
                            getattr(old_agent.network.opt_state[0], state_name),
                            getattr(new_agent.network.opt_state[0], state_name),
                            False,
                            False,
                            False,
                            True,
                            False,
                            True,
                        ),
                        0.0,
                    )
                np.testing.assert_array_equal(np.asarray(old_agent.rng), np.asarray(new_agent.rng))

    def test_state_and_goal_slots_are_independent_and_all_wirings_initialize(self):
        batch = _batch()
        for state_enabled, goal_enabled in (
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(state_enabled=state_enabled, goal_enabled=goal_enabled):
                config = _small_config(
                    False,
                    critic_state_enabled=state_enabled,
                    critic_goal_enabled=goal_enabled,
                )
                agent = CRLAgent.create(67, batch['observations'][:2], batch['actions'][:2], config)
                value, phi, psi = agent.network.select('critic')(
                    batch['observations'], batch['value_goals'], batch['actions'], info=True,
                )
                self.assertEqual(value.shape, (2, 4))
                self.assertEqual(phi.shape, (2, 4, 3))
                self.assertEqual(psi.shape, (2, 4, 3))

        config = _small_config(False, critic_state_enabled=True, critic_goal_enabled=True)
        agent = CRLAgent.create(71, batch['observations'][:2], batch['actions'][:2], config)
        critic = agent.network.params['modules_critic']
        state_core = critic['phi']['core']
        goal_core = critic['psi']['core']
        self.assertGreater(count_parameters(state_core), 0)
        self.assertGreater(count_parameters(goal_core), 0)
        state_leaf = np.asarray(state_core['topology']['primitive']['Dense_0']['kernel'])
        goal_leaf = np.asarray(goal_core['topology']['primitive']['Dense_0']['kernel'])
        self.assertFalse(np.shares_memory(state_leaf, goal_leaf))

    def test_computationized_critic_does_not_change_trainable_parameter_count(self):
        batch = _batch()
        old = CRLAgent.create(73, batch['observations'][:2], batch['actions'][:2], _small_config(False))
        new = CRLAgent.create(
            73,
            batch['observations'][:2],
            batch['actions'][:2],
            _small_config(False, critic_state_enabled=True, critic_goal_enabled=True),
        )
        old_critic = old.network.params['modules_critic']
        new_critic = new.network.params['modules_critic']
        self.assertEqual(count_parameters(old.network.params), count_parameters(new.network.params))
        self.assertEqual(count_parameters(old.network.params['modules_actor']), count_parameters(new.network.params['modules_actor']))
        self.assertEqual(count_parameters(old_critic), count_parameters(new_critic))
        self.assertEqual(count_parameters(new_critic['phi']['core']), count_parameters(old_critic['phi']))
        self.assertEqual(count_parameters(new_critic['psi']['core']), count_parameters(old_critic['psi']))
        self.assertEqual(
            count_parameters(new_critic)
            - count_parameters(new_critic['phi']['core'])
            - count_parameters(new_critic['psi']['core']),
            0,
        )


class CRLAWRValueComputationParityTest(unittest.TestCase):
    @staticmethod
    def _awr_signals(agent, batch, params):
        value, value_phi, value_psi = agent.network.select('value')(
            batch['observations'], batch['value_goals'], info=True, params=params,
        )
        q1, q2 = agent.network.select('critic')(
            batch['observations'], batch['actor_goals'], batch['actions'], params=params,
        )
        advantage = jnp.minimum(q1, q2) - value
        weight = jnp.minimum(jnp.exp(advantage * agent.config['alpha']), 100.0)
        return value_phi, value_psi, value, q1, q2, advantage, weight

    def _value_pair(self):
        old_config = _small_config(False, 'awr')
        new_config = _small_config(
            False,
            'awr',
            value_state_enabled=True,
            value_goal_enabled=True,
        )
        batch = _batch()
        old_agent = CRLAgent.create(79, batch['observations'][:2], batch['actions'][:2], old_config)
        new_agent = CRLAgent.create(79, batch['observations'][:2], batch['actions'][:2], new_config)
        new_agent = new_agent.replace(network=new_agent.network.replace(
            params=_graft_semantic_params(
                old_agent.network.params,
                new_agent.network.params,
                value_state_enabled=True,
                value_goal_enabled=True,
            )
        ))
        return old_agent, new_agent, batch

    def test_value_forward_loss_advantage_actor_gradient_and_update_match(self):
        old_agent, new_agent, batch = self._value_pair()
        old_signals = self._awr_signals(old_agent, batch, old_agent.network.params)
        new_signals = self._awr_signals(new_agent, batch, new_agent.network.params)
        for old_value, new_value in zip(old_signals, new_signals):
            np.testing.assert_array_equal(np.asarray(old_value), np.asarray(new_value))
        self.assertEqual(old_signals[0].shape, (4, 3))
        self.assertEqual(old_signals[1].shape, (4, 3))
        self.assertEqual(old_signals[2].shape, (4,))

        old_value_loss, old_value_info = old_agent.contrastive_loss(
            batch, old_agent.network.params, 'value',
        )
        new_value_loss, new_value_info = new_agent.contrastive_loss(
            batch, new_agent.network.params, 'value',
        )
        np.testing.assert_array_equal(np.asarray(old_value_loss), np.asarray(new_value_loss))
        _assert_info_equal(self, old_value_info, new_value_info, 'AWR value computation loss')

        old_actor_loss, old_actor_info = old_agent.actor_loss(batch, old_agent.network.params)
        new_actor_loss, new_actor_info = new_agent.actor_loss(batch, new_agent.network.params)
        np.testing.assert_array_equal(np.asarray(old_actor_loss), np.asarray(new_actor_loss))
        _assert_info_equal(self, old_actor_info, new_actor_info, 'AWR value computation actor loss')

        old_grads = jax.grad(lambda params: old_agent.total_loss(batch, params)[0])(
            old_agent.network.params,
        )
        new_grads = jax.grad(lambda params: new_agent.total_loss(batch, params)[0])(
            new_agent.network.params,
        )
        self.assertEqual(
            _tree_error(
                old_grads,
                new_grads,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
            ),
            0.0,
        )

        old_agent, old_update = old_agent.update(batch)
        new_agent, new_update = new_agent.update(batch)
        _assert_info_equal(self, old_update, new_update, 'AWR value computation update', skip=('grad/norm',))
        self.assertEqual(
            _tree_error(
                old_agent.network.params,
                new_agent.network.params,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
            ),
            0.0,
        )
        for state_name in ('mu', 'nu'):
            self.assertEqual(
                _tree_error(
                    getattr(old_agent.network.opt_state[0], state_name),
                    getattr(new_agent.network.opt_state[0], state_name),
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    True,
                ),
                0.0,
            )
        np.testing.assert_array_equal(np.asarray(old_agent.rng), np.asarray(new_agent.rng))

    def test_value_state_goal_and_critic_state_goal_are_four_independent_branches(self):
        batch = _batch()
        config = _small_config(
            False,
            'awr',
            critic_state_enabled=True,
            critic_goal_enabled=True,
            value_state_enabled=True,
            value_goal_enabled=True,
        )
        agent = CRLAgent.create(83, batch['observations'][:2], batch['actions'][:2], config)
        critic_value, critic_phi, critic_psi = agent.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], info=True,
        )
        value_value, value_phi, value_psi = agent.network.select('value')(
            batch['observations'], batch['value_goals'], info=True,
        )
        self.assertEqual(critic_value.shape, (2, 4))
        self.assertEqual(critic_phi.shape, (2, 4, 3))
        self.assertEqual(critic_psi.shape, (2, 4, 3))
        self.assertEqual(value_value.shape, (4,))
        self.assertEqual(value_phi.shape, (4, 3))
        self.assertEqual(value_psi.shape, (4, 3))

        params = agent.network.params
        branch_arrays = [
            np.asarray(params['modules_critic']['phi']['core']['topology']['primitive']['Dense_0']['kernel']),
            np.asarray(params['modules_critic']['psi']['core']['topology']['primitive']['Dense_0']['kernel']),
            np.asarray(params['modules_value']['phi']['core']['topology']['primitive']['Dense_0']['kernel']),
            np.asarray(params['modules_value']['psi']['core']['topology']['primitive']['Dense_0']['kernel']),
        ]
        for index, left in enumerate(branch_arrays):
            for right in branch_arrays[index + 1:]:
                self.assertFalse(np.shares_memory(left, right))

    def test_value_is_ensemble_false_and_ddpgbc_does_not_instantiate_value(self):
        batch = _batch()
        awr = CRLAgent.create(
            89,
            batch['observations'][:2],
            batch['actions'][:2],
            _small_config(False, 'awr', value_state_enabled=True, value_goal_enabled=True),
        )
        legacy_awr = CRLAgent.create(
            89,
            batch['observations'][:2],
            batch['actions'][:2],
            _small_config(False, 'awr'),
        )
        value_params = awr.network.params['modules_value']
        self.assertEqual(len(value_params['phi']['core']['topology']['primitive']['Dense_0']['kernel'].shape), 2)
        self.assertEqual(count_parameters(legacy_awr.network.params), count_parameters(awr.network.params))
        self.assertEqual(count_parameters(awr.network.params['modules_value']), 234)
        self.assertEqual(count_parameters(awr.network.params['modules_value']['phi']['core']), 117)
        self.assertEqual(count_parameters(awr.network.params['modules_value']['psi']['core']), 117)
        ddpgbc_with_value_config = _small_config(
            False,
            'ddpgbc',
            value_state_enabled=True,
            value_goal_enabled=True,
        )
        ddpgbc_without_value_config = _small_config(False, 'ddpgbc')
        with_value = CRLAgent.create(97, batch['observations'][:2], batch['actions'][:2], ddpgbc_with_value_config)
        without_value = CRLAgent.create(97, batch['observations'][:2], batch['actions'][:2], ddpgbc_without_value_config)
        self.assertNotIn('modules_value', with_value.network.params)
        self.assertEqual(count_parameters(with_value.network.params), count_parameters(without_value.network.params))

    def test_full_awr_computation_matches_legacy(self):
        old_config = _small_config(False, 'awr')
        new_config = _small_config(
            True,
            'awr',
            critic_state_enabled=True,
            critic_goal_enabled=True,
            value_state_enabled=True,
            value_goal_enabled=True,
        )
        batch = _batch()
        old_agent = CRLAgent.create(101, batch['observations'][:2], batch['actions'][:2], old_config)
        new_agent = CRLAgent.create(101, batch['observations'][:2], batch['actions'][:2], new_config)
        new_agent = new_agent.replace(network=new_agent.network.replace(
            params=_graft_semantic_params(
                old_agent.network.params,
                new_agent.network.params,
                actor_enabled=True,
                critic_state_enabled=True,
                critic_goal_enabled=True,
                value_state_enabled=True,
                value_goal_enabled=True,
            )
        ))
        old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
        new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
        np.testing.assert_array_equal(np.asarray(old_loss), np.asarray(new_loss))
        _assert_info_equal(self, old_info, new_info, 'full AWR computation loss')
        for old_value, new_value in zip(
            self._awr_signals(old_agent, batch, old_agent.network.params),
            self._awr_signals(new_agent, batch, new_agent.network.params),
        ):
            np.testing.assert_array_equal(np.asarray(old_value), np.asarray(new_value))

        old_grads = jax.grad(lambda params: old_agent.total_loss(batch, params)[0])(
            old_agent.network.params,
        )
        new_grads = jax.grad(lambda params: new_agent.total_loss(batch, params)[0])(
            new_agent.network.params,
        )
        self.assertEqual(
            _tree_error(
                old_grads,
                new_grads,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
            ),
            0.0,
        )
        old_agent, old_update = old_agent.update(batch)
        new_agent, new_update = new_agent.update(batch)
        _assert_info_equal(self, old_update, new_update, 'full AWR computation update', skip=('grad/norm',))
        self.assertEqual(
            _tree_error(
                old_agent.network.params,
                new_agent.network.params,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
            ),
            0.0,
        )
        for state_name in ('mu', 'nu'):
            self.assertEqual(
                _tree_error(
                    getattr(old_agent.network.opt_state[0], state_name),
                    getattr(new_agent.network.opt_state[0], state_name),
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ),
                0.0,
            )
        np.testing.assert_array_equal(np.asarray(old_agent.rng), np.asarray(new_agent.rng))


class CRLRealRuntimeParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(resolve_dataset_dir(), DATASET_NAME + '.npz')
        if not os.path.exists(path):
            raise unittest.SkipTest(f'real OGBench dataset is unavailable: {path}')
        cls.env, cls.raw_train, _ = make_env_and_datasets(DATASET_NAME, seed=1234, dataset_seed=5678)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_real_gcdataset_n20_strict_parity(self):
        old_config = _small_config(False)
        new_config = _small_config(
            True,
            critic_state_enabled=True,
            critic_goal_enabled=True,
        )
        old_config['batch_size'] = 8
        new_config['batch_size'] = 8
        old_dataset = GCDataset(Dataset.create(seed=2026, **dict(self.raw_train)), old_config, rng=2026)
        new_dataset = GCDataset(Dataset.create(seed=2026, **dict(self.raw_train)), new_config, rng=2026)
        init_dataset = GCDataset(Dataset.create(seed=2026, **dict(self.raw_train)), old_config, rng=2026)
        initial = init_dataset.sample(2)
        old_agent = CRLAgent.create(44, initial['observations'], initial['actions'], old_config)
        new_agent = CRLAgent.create(44, initial['observations'], initial['actions'], new_config)
        new_agent = new_agent.replace(network=new_agent.network.replace(
            params=_graft_semantic_params(
                old_agent.network.params,
                new_agent.network.params,
                actor_enabled=True,
                critic_state_enabled=True,
                critic_goal_enabled=True,
            )
        ))

        max_errors = {'critic_loss': 0.0, 'actor_loss': 0.0, 'total_loss': 0.0, 'actor_params': 0.0, 'critic_params': 0.0, 'optimizer': 0.0, 'rng': 0.0}
        first_divergence = None
        for step in range(1, 21):
            old_batch = old_dataset.sample(8)
            new_batch = new_dataset.sample(8)
            for key in REQUIRED_BATCH_KEYS:
                np.testing.assert_array_equal(old_batch[key], new_batch[key], err_msg=f'batch {step} {key}')
            old_loss, old_info = old_agent.total_loss(old_batch, old_agent.network.params)
            new_loss, new_info = new_agent.total_loss(new_batch, new_agent.network.params)
            errors = {
                'critic_loss': abs(float(old_info['critic/contrastive_loss']) - float(new_info['critic/contrastive_loss'])),
                'actor_loss': abs(float(old_info['actor/actor_loss']) - float(new_info['actor/actor_loss'])),
                'total_loss': abs(float(old_loss) - float(new_loss)),
            }
            for key, error in errors.items():
                max_errors[key] = max(max_errors[key], error)
            if first_divergence is None and max(errors.values()) > 1e-6:
                first_divergence = step
            np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0.0, atol=ATOL)
            _assert_info_equal(self, old_info, new_info, f'real CRL step={step} loss')
            old_agent, old_update = old_agent.update(old_batch)
            new_agent, new_update = new_agent.update(new_batch)
            actor_error = _tree_error(
                {'modules_actor': old_agent.network.params['modules_actor']},
                {'modules_actor': new_agent.network.params['modules_actor']},
                False,
                True,
            )
            critic_error = _tree_error(
                {'modules_critic': old_agent.network.params['modules_critic']},
                {'modules_critic': new_agent.network.params['modules_critic']},
                False,
                True,
                False,
                True,
                False,
                True,
            )
            optimizer_error = max(
                _tree_error(
                    old_agent.network.opt_state[0].mu,
                    new_agent.network.opt_state[0].mu,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ),
                _tree_error(
                    old_agent.network.opt_state[0].nu,
                    new_agent.network.opt_state[0].nu,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ),
            )
            rng_error = float(np.max(np.abs(np.asarray(old_agent.rng) - np.asarray(new_agent.rng))))
            max_errors['actor_params'] = max(max_errors['actor_params'], actor_error)
            max_errors['critic_params'] = max(max_errors['critic_params'], critic_error)
            max_errors['optimizer'] = max(max_errors['optimizer'], optimizer_error)
            max_errors['rng'] = max(max_errors['rng'], rng_error)
            _assert_info_equal(self, old_update, new_update, f'real CRL step={step} update', skip=('grad/norm',))
            self.assertLessEqual(actor_error, ATOL)
            self.assertLessEqual(critic_error, ATOL)
            self.assertLessEqual(optimizer_error, ATOL)
            self.assertEqual(rng_error, 0.0)

        self.assertIsNone(first_divergence)
        self.assertEqual(max_errors['critic_loss'], 0.0)
        self.assertEqual(max_errors['actor_loss'], 0.0)
        self.assertEqual(max_errors['total_loss'], 0.0)
        self.assertEqual(max_errors['actor_params'], 0.0)
        self.assertEqual(max_errors['critic_params'], 0.0)
        self.assertEqual(max_errors['optimizer'], 0.0)
        self.assertEqual(max_errors['rng'], 0.0)

    def test_real_gcdataset_n20_strict_awr_full_computation_parity(self):
        old_config = _small_config(False, 'awr')
        new_config = _small_config(
            True,
            'awr',
            critic_state_enabled=True,
            critic_goal_enabled=True,
            value_state_enabled=True,
            value_goal_enabled=True,
        )
        old_config['batch_size'] = 8
        new_config['batch_size'] = 8
        old_dataset = GCDataset(Dataset.create(seed=3026, **dict(self.raw_train)), old_config, rng=3026)
        new_dataset = GCDataset(Dataset.create(seed=3026, **dict(self.raw_train)), new_config, rng=3026)
        init_dataset = GCDataset(Dataset.create(seed=3026, **dict(self.raw_train)), old_config, rng=3026)
        initial = init_dataset.sample(2)
        old_agent = CRLAgent.create(144, initial['observations'], initial['actions'], old_config)
        new_agent = CRLAgent.create(144, initial['observations'], initial['actions'], new_config)
        new_agent = new_agent.replace(network=new_agent.network.replace(
            params=_graft_semantic_params(
                old_agent.network.params,
                new_agent.network.params,
                actor_enabled=True,
                critic_state_enabled=True,
                critic_goal_enabled=True,
                value_state_enabled=True,
                value_goal_enabled=True,
            )
        ))

        max_errors = {
            key: 0.0 for key in (
                'critic_loss', 'value_loss', 'actor_loss', 'total_loss',
                'q1', 'q2', 'value', 'advantage', 'awr_weight',
                'actor_params', 'critic_state_params', 'critic_goal_params',
                'value_state_params', 'value_goal_params', 'optimizer', 'rng',
            )
        }
        first_divergence = None
        for step in range(1, 21):
            old_batch = old_dataset.sample(8)
            new_batch = new_dataset.sample(8)
            for key in REQUIRED_BATCH_KEYS:
                np.testing.assert_array_equal(old_batch[key], new_batch[key], err_msg=f'AWR batch {step} {key}')
            old_loss, old_info = old_agent.total_loss(old_batch, old_agent.network.params)
            new_loss, new_info = new_agent.total_loss(new_batch, new_agent.network.params)
            old_signals = CRLAWRValueComputationParityTest._awr_signals(
                old_agent, old_batch, old_agent.network.params,
            )
            new_signals = CRLAWRValueComputationParityTest._awr_signals(
                new_agent, new_batch, new_agent.network.params,
            )
            errors = {
                'critic_loss': abs(float(old_info['critic/contrastive_loss']) - float(new_info['critic/contrastive_loss'])),
                'value_loss': abs(float(old_info['value/contrastive_loss']) - float(new_info['value/contrastive_loss'])),
                'actor_loss': abs(float(old_info['actor/actor_loss']) - float(new_info['actor/actor_loss'])),
                'total_loss': abs(float(old_loss) - float(new_loss)),
                'value': float(np.max(np.abs(np.asarray(old_signals[2]) - np.asarray(new_signals[2])))),
                'q1': float(np.max(np.abs(np.asarray(old_signals[3]) - np.asarray(new_signals[3])))),
                'q2': float(np.max(np.abs(np.asarray(old_signals[4]) - np.asarray(new_signals[4])))),
                'advantage': float(np.max(np.abs(np.asarray(old_signals[5]) - np.asarray(new_signals[5])))),
                'awr_weight': float(np.max(np.abs(np.asarray(old_signals[6]) - np.asarray(new_signals[6])))),
            }
            for key, error in errors.items():
                max_errors[key] = max(max_errors[key], error)
            if first_divergence is None and max(errors.values()) > ATOL:
                first_divergence = step
            np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0.0, atol=ATOL)
            _assert_info_equal(self, old_info, new_info, f'real AWR step={step} loss')
            old_agent, old_update = old_agent.update(old_batch)
            new_agent, new_update = new_agent.update(new_batch)
            actor_error = _tree_error(
                {'modules_actor': old_agent.network.params['modules_actor']},
                {'modules_actor': new_agent.network.params['modules_actor']},
                old_actor_enabled=False,
                new_actor_enabled=True,
            )
            critic_state_error = _branch_tree_error(
                old_agent.network.params,
                new_agent.network.params,
                'modules_critic',
                'phi',
                False,
                True,
            )
            critic_goal_error = _branch_tree_error(
                old_agent.network.params,
                new_agent.network.params,
                'modules_critic',
                'psi',
                False,
                True,
            )
            value_state_error = _branch_tree_error(
                old_agent.network.params,
                new_agent.network.params,
                'modules_value',
                'phi',
                False,
                True,
            )
            value_goal_error = _branch_tree_error(
                old_agent.network.params,
                new_agent.network.params,
                'modules_value',
                'psi',
                False,
                True,
            )
            optimizer_error = max(
                _tree_error(
                    old_agent.network.opt_state[0].mu,
                    new_agent.network.opt_state[0].mu,
                    old_actor_enabled=False,
                    new_actor_enabled=True,
                    old_critic_state_enabled=False,
                    new_critic_state_enabled=True,
                    old_critic_goal_enabled=False,
                    new_critic_goal_enabled=True,
                    old_value_state_enabled=False,
                    new_value_state_enabled=True,
                    old_value_goal_enabled=False,
                    new_value_goal_enabled=True,
                ),
                _tree_error(
                    old_agent.network.opt_state[0].nu,
                    new_agent.network.opt_state[0].nu,
                    old_actor_enabled=False,
                    new_actor_enabled=True,
                    old_critic_state_enabled=False,
                    new_critic_state_enabled=True,
                    old_critic_goal_enabled=False,
                    new_critic_goal_enabled=True,
                    old_value_state_enabled=False,
                    new_value_state_enabled=True,
                    old_value_goal_enabled=False,
                    new_value_goal_enabled=True,
                ),
            )
            rng_error = float(np.max(np.abs(np.asarray(old_agent.rng) - np.asarray(new_agent.rng))))
            max_errors['actor_params'] = max(max_errors['actor_params'], actor_error)
            max_errors['critic_state_params'] = max(max_errors['critic_state_params'], critic_state_error)
            max_errors['critic_goal_params'] = max(max_errors['critic_goal_params'], critic_goal_error)
            max_errors['value_state_params'] = max(max_errors['value_state_params'], value_state_error)
            max_errors['value_goal_params'] = max(max_errors['value_goal_params'], value_goal_error)
            max_errors['optimizer'] = max(max_errors['optimizer'], optimizer_error)
            max_errors['rng'] = max(max_errors['rng'], rng_error)
            _assert_info_equal(self, old_update, new_update, f'real AWR step={step} update', skip=('grad/norm',))
            self.assertLessEqual(actor_error, ATOL)
            self.assertLessEqual(critic_state_error, ATOL)
            self.assertLessEqual(critic_goal_error, ATOL)
            self.assertLessEqual(value_state_error, ATOL)
            self.assertLessEqual(value_goal_error, ATOL)
            self.assertLessEqual(optimizer_error, ATOL)
            self.assertEqual(rng_error, 0.0)

        self.assertIsNone(first_divergence, msg=f'first AWR divergence: {first_divergence}')
        for key, error in max_errors.items():
            self.assertEqual(error, 0.0, msg=f'AWR max error {key}={error}')


if __name__ == '__main__':
    unittest.main()
