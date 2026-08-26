"""M13 parity, equation, target, checkpoint, and RNG tests."""

import copy
import pickle
import tempfile
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import agent_configs, agents
from impls.agents.qrl import QRLAgent
from impls.main import _loss_metric
from impls.networks.common import GCIQEValue, GCMRNValue
from impls.utils.datasets import Dataset, GCDataset
from impls.utils.flax_utils import restore_agent_from_checkpoint


def _small_config(name):
    config = copy.deepcopy(agent_configs[name]())
    config.actor_hidden_dims = (8, 8)
    if 'value_hidden_dims' in config:
        config.value_hidden_dims = (8, 8)
    config.batch_size = 3
    if name == 'qrl':
        config.latent_dim = 8
        config.dim_per_component = 4
    return config


def _batch():
    observations = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 10
    actions = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
    return {
        'observations': observations,
        'next_observations': observations + 0.1,
        'actions': actions,
        'value_goals': observations + 0.2,
        'actor_goals': observations + 0.3,
        'rewards': jnp.array([-1.0, 0.0, -1.0]),
        'masks': jnp.array([1.0, 0.0, 1.0]),
    }


def _discrete_batch():
    batch = _batch()
    batch['actions'] = jnp.array([0, 1, 2], dtype=jnp.int32)
    return batch


def _tree_assert_equal(testcase, left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    testcase.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _tree_assert_allclose(testcase, left, right, atol=1e-6):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    testcase.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(np.asarray(left_leaf), np.asarray(right_leaf), atol=atol, rtol=0)


class M13ConfigParityTest(unittest.TestCase):
    def test_all_upstream_scientific_config_fields(self):
        expected = {
            'gcbc': {
                'agent_name': 'gcbc', 'lr': 3e-4, 'batch_size': 1024,
                'actor_hidden_dims': (512, 512, 512), 'discount': 0.99,
                'const_std': True, 'discrete': False, 'encoder': None,
                'dataset_class': 'GCDataset', 'value_p_curgoal': 0.0,
                'value_p_trajgoal': 1.0, 'value_p_randomgoal': 0.0,
                'value_geom_sample': False, 'actor_p_curgoal': 0.0,
                'actor_p_trajgoal': 1.0, 'actor_p_randomgoal': 0.0,
                'actor_geom_sample': False, 'gc_negative': True,
                'p_aug': 0.0, 'frame_stack': None,
            },
            'gciql': {
                'agent_name': 'gciql', 'lr': 3e-4, 'batch_size': 1024,
                'actor_hidden_dims': (512, 512, 512),
                'value_hidden_dims': (512, 512, 512), 'layer_norm': True,
                'discount': 0.99, 'tau': 0.005, 'expectile': 0.9,
                'actor_loss': 'ddpgbc', 'alpha': 0.3, 'const_std': True,
                'discrete': False, 'encoder': None, 'dataset_class': 'GCDataset',
                'value_p_curgoal': 0.2, 'value_p_trajgoal': 0.5,
                'value_p_randomgoal': 0.3, 'value_geom_sample': True,
                'actor_p_curgoal': 0.0, 'actor_p_trajgoal': 1.0,
                'actor_p_randomgoal': 0.0, 'actor_geom_sample': False,
                'gc_negative': True, 'p_aug': 0.0, 'frame_stack': None,
            },
            'gcivl': {
                'agent_name': 'gcivl', 'lr': 3e-4, 'batch_size': 1024,
                'actor_hidden_dims': (512, 512, 512),
                'value_hidden_dims': (512, 512, 512), 'layer_norm': True,
                'discount': 0.99, 'tau': 0.005, 'expectile': 0.9,
                'alpha': 10.0, 'const_std': True, 'discrete': False,
                'encoder': None, 'dataset_class': 'GCDataset',
                'value_p_curgoal': 0.2, 'value_p_trajgoal': 0.5,
                'value_p_randomgoal': 0.3, 'value_geom_sample': True,
                'actor_p_curgoal': 0.0, 'actor_p_trajgoal': 1.0,
                'actor_p_randomgoal': 0.0, 'actor_geom_sample': False,
                'gc_negative': True, 'p_aug': 0.0, 'frame_stack': None,
            },
            'qrl': {
                'agent_name': 'qrl', 'lr': 3e-4, 'batch_size': 1024,
                'actor_hidden_dims': (512, 512, 512),
                'value_hidden_dims': (512, 512, 512),
                'quasimetric_type': 'iqe', 'latent_dim': 512,
                'layer_norm': True, 'discount': 0.99, 'eps': 0.05,
                'actor_loss': 'ddpgbc', 'alpha': 0.003, 'const_std': True,
                'discrete': False, 'encoder': None, 'dataset_class': 'GCDataset',
                'value_p_curgoal': 0.0, 'value_p_trajgoal': 0.0,
                'value_p_randomgoal': 1.0, 'value_geom_sample': True,
                'actor_p_curgoal': 0.0, 'actor_p_trajgoal': 1.0,
                'actor_p_randomgoal': 0.0, 'actor_geom_sample': False,
                'gc_negative': False, 'p_aug': 0.0, 'frame_stack': None,
            },
        }
        for name, fields in expected.items():
            config = agent_configs[name]()
            for key, value in fields.items():
                self.assertEqual(config[key], value, msg=f'{name}.{key}')

    def test_registry_contains_all_canonical_agents(self):
        self.assertEqual(
            set(('gcbc', 'gciql', 'gcivl', 'qrl', 'crl', 'hiql', 'coghp')),
            set(agents),
        )
        self.assertEqual(set(agents), set(agent_configs))


class M13TargetAndRuntimeTest(unittest.TestCase):
    def test_target_initialization_and_updated_online_polyak(self):
        batch = _batch()
        for name, online_name, target_name in (
            ('gciql', 'modules_critic', 'modules_target_critic'),
            ('gcivl', 'modules_value', 'modules_target_value'),
        ):
            config = _small_config(name)
            agent = agents[name].create(
                17, batch['observations'], batch['actions'], config
            )
            _tree_assert_equal(
                self,
                agent.network.params[online_name],
                agent.network.params[target_name],
            )
            old_target = jax.tree_util.tree_map(
                lambda value: jnp.array(value),
                agent.network.params[target_name],
            )
            updated, info = agent.update(batch)
            self.assertTrue(all(np.all(np.isfinite(np.asarray(value))) for value in info.values()))
            expected = jax.tree_util.tree_map(
                lambda online, target: config['tau'] * online + (1 - config['tau']) * target,
                updated.network.params[online_name],
                old_target,
            )
            _tree_assert_allclose(self, expected, updated.network.params[target_name], atol=2e-6)

    def test_parameter_tree_and_one_update_for_all_four_agents(self):
        batch = _batch()
        expected_keys = {
            'gcbc': {'modules_actor'},
            'gciql': {'modules_value', 'modules_critic', 'modules_target_critic', 'modules_actor'},
            'gcivl': {'modules_value', 'modules_target_value', 'modules_actor'},
            'qrl': {'modules_value', 'modules_actor', 'modules_dynamics', 'modules_lam'},
        }
        for name, keys in expected_keys.items():
            config = _small_config(name)
            agent = agents[name].create(3, batch['observations'], batch['actions'], config)
            self.assertEqual(set(agent.network.params), keys)
            updated, info = agent.update(batch)
            self.assertTrue(all(np.all(np.isfinite(np.asarray(value))) for value in info.values()))
            self.assertTrue(np.all(np.isfinite(np.asarray(updated.sample_actions(
                batch['observations'][:1], batch['actor_goals'][:1], seed=jax.random.PRNGKey(8)
            )))))

    def test_discrete_actor_and_critic_paths(self):
        batch = _discrete_batch()
        for name in ('gcbc', 'gciql', 'gcivl', 'qrl'):
            config = _small_config(name)
            config.discrete = True
            if name in ('gciql', 'qrl'):
                config.actor_loss = 'awr'
            agent = agents[name].create(13, batch['observations'], batch['actions'], config)
            updated, info = agent.update(batch)
            self.assertTrue(all(np.all(np.isfinite(np.asarray(value))) for value in info.values()))
            actions = updated.sample_actions(
                batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(14)
            )
            self.assertEqual(actions.shape, (3,))

    def test_checkpoint_roundtrip_preserves_auxiliary_modules(self):
        batch = _batch()
        for name in ('gcbc', 'gciql', 'gcivl', 'qrl'):
            config = _small_config(name)
            agent = agents[name].create(9, batch['observations'], batch['actions'], config)
            agent, _ = agent.update(batch)
            with tempfile.TemporaryDirectory() as directory:
                path = f'{directory}/params.pkl'
                with open(path, 'wb') as handle:
                    pickle.dump({'agent': flax.serialization.to_state_dict(agent)}, handle)
                restored = restore_agent_from_checkpoint(agent, path)
            actions = agent.sample_actions(
                batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(11)
            )
            restored_actions = restored.sample_actions(
                batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(11)
            )
            np.testing.assert_array_equal(np.asarray(actions), np.asarray(restored_actions))
            for key in ('modules_target_critic', 'modules_target_value', 'modules_lam', 'modules_dynamics'):
                if key in agent.network.params:
                    _tree_assert_equal(self, agent.network.params[key], restored.network.params[key])


class M13EquationTest(unittest.TestCase):
    def test_gcbc_actor_loss_is_negative_mean_log_probability(self):
        batch = _batch()
        agent = agents['gcbc'].create(1, batch['observations'], batch['actions'], _small_config('gcbc'))
        loss, info = agent.actor_loss(batch, agent.network.params)
        np.testing.assert_allclose(float(loss), float(-info['bc_log_prob']), rtol=0, atol=1e-6)

    def test_expectile_weighting_and_loss_metric_no_double_count(self):
        adv = jnp.array([-2.0, 3.0])
        diff = jnp.array([4.0, 5.0])
        expected = jnp.array([(1 - 0.9) * 16.0, 0.9 * 25.0])
        np.testing.assert_allclose(
            np.asarray(agents['gciql'].expectile_loss(adv, diff, 0.9)),
            np.asarray(expected),
        )
        self.assertEqual(float(_loss_metric({
            'value/total_loss': 1.0,
            'value/value_loss': 2.0,
            'value/lam_loss': 3.0,
            'dynamics/dynamics_loss': 4.0,
            'actor/actor_loss': 5.0,
        })), 10.0)
        self.assertEqual(float(_loss_metric({
            'value/value_loss': 1.0,
            'critic/critic_loss': 2.0,
            'actor/actor_loss': 3.0,
        })), 6.0)

    def test_qrl_value_loss_contains_dual_lambda_terms(self):
        class Value:
            def __init__(self):
                self.calls = 0

            def __call__(self, *args, **kwargs):
                del args, kwargs
                self.calls += 1
                return jnp.array([2.0, 4.0]) if self.calls == 1 else jnp.array([0.5, 2.0])

        class Select:
            def __init__(self, value):
                self.value = value

            def __call__(self, *args, **kwargs):
                del args, kwargs
                return self.value

        class Network:
            def __init__(self):
                self.value = Value()

            def select(self, name):
                if name == 'value':
                    return self.value
                return Select(jnp.array(2.0))

        agent = QRLAgent(None, Network(), {'eps': 0.05})
        batch = {'observations': None, 'next_observations': None, 'value_goals': None}
        total, info = agent.value_loss(batch, {})
        d_neg_loss = jnp.mean(100 * jax.nn.softplus(5 - jnp.array([2.0, 4.0]) / 100))
        d_pos_loss = jnp.mean(jax.nn.relu(jnp.array([0.5, 2.0]) - 1) ** 2)
        expected = d_neg_loss + d_pos_loss * 2 + 2 * (0.05 - d_pos_loss)
        np.testing.assert_allclose(float(total), float(expected), atol=1e-6)
        np.testing.assert_allclose(float(info['lam_loss']), float(2 * (0.05 - d_pos_loss)), atol=1e-6)

    def test_iqe_and_mrn_forward_equations(self):
        state = jnp.array([[0.0, 2.0, 1.0, 3.0], [1.0, 4.0, 2.0, 5.0]])
        goal = jnp.array([[1.0, 1.0, 4.0, 2.0], [0.0, 6.0, 1.0, 8.0]])
        iqe = GCIQEValue(hidden_dims=(8,), latent_dim=4, dim_per_component=2)
        iqe_vars = iqe.init(jax.random.PRNGKey(0), state, goal, is_phi=True)
        actual_iqe = iqe.apply(iqe_vars, state, goal, is_phi=True)

        expected_iqe = []
        for x, y in zip(np.asarray(state), np.asarray(goal)):
            x = x.reshape(2, 2)
            y = y.reshape(2, 2)
            components = []
            for x_group, y_group in zip(x, y):
                events = []
                for x_value, y_value, valid in zip(x_group, y_group, x_group < y_group):
                    if valid:
                        events.extend([(float(x_value), -1), (float(y_value), 1)])
                events.sort()
                count = 0
                component = 0.0
                for value, sign in events:
                    previous = count
                    count += sign
                    if previous < 0:
                        component += value
                    if count < 0:
                        component -= value
                components.append(component)
            expected_iqe.append(0.5 * np.mean(components) + 0.5 * np.max(components))
        np.testing.assert_allclose(np.asarray(actual_iqe), np.asarray(expected_iqe), atol=1e-6)

        mrn = GCMRNValue(hidden_dims=(8,), latent_dim=4)
        mrn_vars = mrn.init(jax.random.PRNGKey(1), state, goal, is_phi=True)
        actual_mrn = mrn.apply(mrn_vars, state, goal, is_phi=True)
        sym = np.sqrt(np.sum((np.asarray(state[:, :2]) - np.asarray(goal[:, :2])) ** 2, axis=-1))
        asym = np.maximum(np.max(np.asarray(state[:, 2:]) - np.asarray(goal[:, 2:]), axis=-1), 0)
        np.testing.assert_allclose(np.asarray(actual_mrn), sym + asym, atol=1e-6)


class M13DatasetRNGTest(unittest.TestCase):
    def test_gcdataset_explicit_rng_is_reproducible(self):
        observations = np.arange(40, dtype=np.float32).reshape(10, 4)
        actions = np.zeros((10, 2), dtype=np.float32)
        terminals = np.zeros(10, dtype=np.float32)
        terminals[[4, 9]] = 1
        fields = {
            'observations': observations,
            'actions': actions,
            'terminals': terminals,
        }
        config = copy.deepcopy(agent_configs['gcbc']())
        first = GCDataset(Dataset.create(freeze=False, **fields), config, rng=123)
        second = GCDataset(Dataset.create(freeze=False, **fields), config, rng=123)
        for _ in range(4):
            left, right = first.sample(5), second.sample(5)
            self.assertEqual(set(left), set(right))
            for key in left:
                np.testing.assert_array_equal(np.asarray(left[key]), np.asarray(right[key]))

    def test_gcdataset_goal_sampler_matches_official_current_goal_boundary(self):
        observations = np.arange(20, dtype=np.float32).reshape(5, 4)
        actions = np.zeros((5, 2), dtype=np.float32)
        terminals = np.array([0, 0, 0, 0, 1], dtype=np.float32)
        config = copy.deepcopy(agent_configs['gcbc']())
        config.value_p_curgoal = 1.0
        config.value_p_trajgoal = 0.0
        config.value_p_randomgoal = 0.0
        dataset = GCDataset(
            Dataset.create(
                freeze=False,
                observations=observations,
                actions=actions,
                terminals=terminals,
            ),
            config,
            rng=77,
        )
        idxs = np.array([0, 2, 4])
        np.testing.assert_array_equal(
            dataset.sample_goals(
                idxs, 1.0, 0.0, 0.0, False, np.random.default_rng(77)
            ),
            idxs,
        )


if __name__ == '__main__':
    unittest.main()
