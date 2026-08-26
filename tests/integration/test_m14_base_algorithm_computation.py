"""M14 computation-slot integration and canonical parity tests."""

import copy
import pickle
import tempfile
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import agent_configs, agents
from impls.computation.slots import SLOT_DESCRIPTORS, validate_compute_slots
from impls.main import (
    _accounting_consistency_audit,
    _computation_runtime_extras,
    _computation_slot_accounting,
)
from impls.utils.flax_utils import restore_agent_from_checkpoint


ALGORITHMS = ('gcbc', 'gciql', 'gcivl', 'qrl')


def _small_config(name):
    config = copy.deepcopy(agent_configs[name]())
    config.actor_hidden_dims = (8, 8)
    if 'value_hidden_dims' in config:
        config.value_hidden_dims = (8, 8)
    if name == 'qrl':
        config.latent_dim = 6
        config.dim_per_component = 3
    config.batch_size = 3
    return config


def _enable(config, slot_name, topology='single_state', credit='direct'):
    slot = config.compute[slot_name]
    slot.enabled = True
    slot.topology = topology
    slot.credit = credit
    if topology == 'single_state':
        slot.topology_kwargs = {
            'iterations': 1,
            'residual': False,
            'input_injection': 'z_plus_x',
            'state_dim': 6 if config.agent_name == 'qrl' and slot_name in {'value', 'dynamics'} else 8,
            'state_init': 'normal_buffer',
            'state_init_std': 1.0,
            'update_depth': 2,
        }
    elif topology == 'two_state':
        slot.topology_kwargs = {
            'h_cycles': 2,
            'l_cycles': 1,
            'input_injection': 'l_receives_x',
            'state_dim': 6 if config.agent_name == 'qrl' and slot_name in {'value', 'dynamics'} else 8,
            'state_init': 'normal_buffer',
            'state_init_std': 1.0,
            'update_depth': 2,
        }


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


def _tree_assert_equal(testcase, left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    testcase.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _tree_assert_allclose(testcase, left, right, atol=2e-6):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    testcase.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(
            np.asarray(left_leaf), np.asarray(right_leaf), atol=atol, rtol=0
        )


def _finite_info(info):
    return all(np.all(np.isfinite(np.asarray(value))) for value in info.values())


class M14SlotSchemaTest(unittest.TestCase):
    def test_exact_canonical_slot_ontology(self):
        self.assertEqual(set(SLOT_DESCRIPTORS['gcbc']), {'actor'})
        self.assertEqual(set(SLOT_DESCRIPTORS['gciql']), {'actor', 'value', 'critic'})
        self.assertEqual(set(SLOT_DESCRIPTORS['gcivl']), {'actor', 'value'})
        self.assertEqual(set(SLOT_DESCRIPTORS['qrl']), {'actor', 'value', 'dynamics'})
        self.assertEqual(SLOT_DESCRIPTORS['qrl']['value'].state_dim_source, 'latent_dim')
        self.assertEqual(SLOT_DESCRIPTORS['qrl']['dynamics'].state_dim_source, 'latent_dim')
        self.assertEqual(
            SLOT_DESCRIPTORS['gciql']['critic'].core_path,
            ('value_net', 'core', 'topology'),
        )

    def test_invalid_slots_fail_loudly(self):
        invalid = {
            'gcbc': 'critic',
            'gcivl': 'critic',
            'gciql': 'dynamics',
            'qrl': 'critic',
        }
        for agent_name, slot_name in invalid.items():
            with self.subTest(agent_name=agent_name):
                config = _small_config(agent_name)
                config.compute[slot_name] = {
                    'enabled': False,
                    'primitive': 'mlp',
                    'topology': 'feedforward',
                    'credit': 'direct',
                }
                with self.assertRaisesRegex(ValueError, 'Unsupported computation slot'):
                    validate_compute_slots(agent_name, config)
                with self.assertRaisesRegex(ValueError, 'Unsupported computation slot'):
                    agents[agent_name].create(
                        0, _batch()['observations'], _batch()['actions'], config
                    )


class M14CanonicalDisabledParityTest(unittest.TestCase):
    def test_m13_config_without_compute_matches_disabled_compute(self):
        """Absent compute config is the M13 baseline; disabled slots are M14."""

        batch = _batch()
        for name in ALGORITHMS:
            with self.subTest(agent_name=name):
                disabled = _small_config(name)
                baseline = copy.deepcopy(disabled)
                del baseline['compute']
                baseline_agent = agents[name].create(
                    23, batch['observations'], batch['actions'], baseline
                )
                disabled_agent = agents[name].create(
                    23, batch['observations'], batch['actions'], disabled
                )
                _tree_assert_equal(
                    self, baseline_agent.network.params, disabled_agent.network.params
                )
                _tree_assert_equal(
                    self,
                    baseline_agent.network.model_state,
                    disabled_agent.network.model_state,
                )
                baseline_action = baseline_agent.sample_actions(
                    batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(7)
                )
                disabled_action = disabled_agent.sample_actions(
                    batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(7)
                )
                np.testing.assert_array_equal(
                    np.asarray(baseline_action), np.asarray(disabled_action)
                )
                baseline_updated, baseline_info = baseline_agent.update(batch)
                disabled_updated, disabled_info = disabled_agent.update(batch)
                _tree_assert_equal(
                    self,
                    baseline_updated.network.params,
                    disabled_updated.network.params,
                )
                _tree_assert_equal(
                    self,
                    baseline_updated.network.model_state,
                    disabled_updated.network.model_state,
                )
                self.assertEqual(set(baseline_info), set(disabled_info))
                for key in baseline_info:
                    np.testing.assert_array_equal(
                        np.asarray(baseline_info[key]), np.asarray(disabled_info[key])
                    )


class M14SingleStateSlotTest(unittest.TestCase):
    def test_each_canonical_slot_supports_feedforward(self):
        batch = _batch()
        for name in ALGORITHMS:
            config = _small_config(name)
            for slot_name in SLOT_DESCRIPTORS[name]:
                _enable(config, slot_name, topology='feedforward')
            agent = agents[name].create(
                29, batch['observations'], batch['actions'], config
            )
            updated, info = agent.update(batch)
            self.assertTrue(_finite_info(info))
            self.assertEqual(updated.network.model_state, {})
            report = _computation_slot_accounting(updated, config)
            self.assertEqual(set(report), set(SLOT_DESCRIPTORS[name]))
            self.assertTrue(all(slot['buffer_elements'] == 0 for slot in report.values()))
            self.assertTrue(all(slot['dense_macs'] > 0 for slot in report.values()))

    def test_each_canonical_slot_supports_single_state_k1(self):
        batch = _batch()
        for name in ALGORITHMS:
            for slot_name in SLOT_DESCRIPTORS[name]:
                with self.subTest(agent_name=name, slot=slot_name):
                    config = _small_config(name)
                    _enable(config, slot_name)
                    agent = agents[name].create(
                        31, batch['observations'], batch['actions'], config
                    )
                    updated, info = agent.update(batch)
                    self.assertTrue(_finite_info(info))
                    buffers = updated.network.model_state.get('buffers', {})
                    self.assertIn(f'modules_{slot_name}', buffers)
                    report = _computation_slot_accounting(updated, config)
                    self.assertEqual(set(report), {slot_name})
                    slot_report = report[slot_name]
                    for field in (
                        'topology', 'primitive', 'credit', 'trainable_params',
                        'buffer_elements', 'state_dim', 'iterations',
                        'h_cycles', 'l_cycles', 'total_update_executions',
                        'unique_dense_layers', 'executed_dense_layers',
                        'sequential_depth', 'dense_macs',
                    ):
                        self.assertIn(field, slot_report)
                    expected_dim = 6 if name == 'qrl' and slot_name in {'value', 'dynamics'} else 8
                    self.assertEqual(slot_report['state_dim'], expected_dim)
                    self.assertGreater(slot_report['dense_macs'], 0)

    def test_qrl_iqe_and_mrn_phi_paths_keep_operator_and_bypass(self):
        batch = _batch()
        for quasimetric_type in ('iqe', 'mrn'):
            config = _small_config('qrl')
            config.quasimetric_type = quasimetric_type
            _enable(config, 'value')
            agent = agents['qrl'].create(37, batch['observations'], batch['actions'], config)
            value = agent.network.select('value')(
                batch['observations'], batch['value_goals']
            )
            phi_state = jnp.ones((3, 6), dtype=jnp.float32)
            phi_goal = jnp.zeros((3, 6), dtype=jnp.float32)
            bypass = agent.network.select('value')(
                phi_state, phi_goal, is_phi=True
            )
            self.assertEqual(value.shape, (3,))
            self.assertEqual(bypass.shape, (3,))
            self.assertTrue(np.all(np.isfinite(np.asarray(value))))
            self.assertTrue(np.all(np.isfinite(np.asarray(bypass))))


class M14TwoStateSmokeTest(unittest.TestCase):
    def test_major_roles_support_two_state_h2l1_full_bptt(self):
        batch = _batch()
        roles = (
            ('gcbc', 'actor'),
            ('gciql', 'critic'),
            ('gcivl', 'value'),
            ('qrl', 'value'),
            ('qrl', 'dynamics'),
        )
        for name, slot_name in roles:
            with self.subTest(agent_name=name, slot=slot_name):
                config = _small_config(name)
                _enable(config, slot_name, topology='two_state', credit='full_bptt')
                agent = agents[name].create(
                    41, batch['observations'], batch['actions'], config
                )
                updated, info = agent.update(batch)
                self.assertTrue(_finite_info(info))
                report = _computation_slot_accounting(updated, config)[slot_name]
                self.assertEqual(report['topology'], 'two_state')
                self.assertEqual(report['h_cycles'], 2)
                self.assertEqual(report['l_cycles'], 1)
                self.assertEqual(report['total_update_executions'], 4)
                self.assertGreater(report['dense_macs'], 0)
                self.assertIn(f'modules_{slot_name}', updated.network.model_state['buffers'])


class M14TargetAndCheckpointTest(unittest.TestCase):
    def test_recurrent_target_params_and_buffers_are_equal_then_params_only_polyak(self):
        batch = _batch()
        for name, online, target in (
            ('gciql', 'critic', 'target_critic'),
            ('gcivl', 'value', 'target_value'),
        ):
            with self.subTest(agent_name=name):
                config = _small_config(name)
                _enable(config, online)
                agent = agents[name].create(
                    43, batch['observations'], batch['actions'], config
                )
                _tree_assert_equal(
                    self,
                    agent.network.params[f'modules_{online}'],
                    agent.network.params[f'modules_{target}'],
                )
                _tree_assert_equal(
                    self,
                    agent.network.model_state['buffers'][f'modules_{online}'],
                    agent.network.model_state['buffers'][f'modules_{target}'],
                )
                if name == 'gciql':
                    online_output = agent.network.select('critic')(
                        batch['observations'], batch['value_goals'], batch['actions']
                    )
                    target_output = agent.network.select('target_critic')(
                        batch['observations'], batch['value_goals'], batch['actions']
                    )
                else:
                    online_output = agent.network.select('value')(
                        batch['observations'], batch['value_goals']
                    )
                    target_output = agent.network.select('target_value')(
                        batch['observations'], batch['value_goals']
                    )
                _tree_assert_equal(self, online_output, target_output)
                old_target_params = copy.deepcopy(agent.network.params[f'modules_{target}'])
                old_online_buffers = copy.deepcopy(
                    agent.network.model_state['buffers'][f'modules_{online}']
                )
                old_target_buffers = copy.deepcopy(
                    agent.network.model_state['buffers'][f'modules_{target}']
                )
                updated, info = agent.update(batch)
                self.assertTrue(_finite_info(info))
                expected = jax.tree_util.tree_map(
                    lambda p, tp: config['tau'] * p + (1 - config['tau']) * tp,
                    updated.network.params[f'modules_{online}'],
                    old_target_params,
                )
                _tree_assert_allclose(
                    self, expected, updated.network.params[f'modules_{target}']
                )
                _tree_assert_equal(
                    self,
                    old_online_buffers,
                    updated.network.model_state['buffers'][f'modules_{online}'],
                )
                _tree_assert_equal(
                    self,
                    old_target_buffers,
                    updated.network.model_state['buffers'][f'modules_{target}'],
                )

    def test_recurrent_roundtrip_restores_params_and_model_state(self):
        batch = _batch()
        for name in ALGORITHMS:
            with self.subTest(agent_name=name):
                config = _small_config(name)
                for slot_name in SLOT_DESCRIPTORS[name]:
                    _enable(config, slot_name)
                agent, _ = agents[name].create(
                    47, batch['observations'], batch['actions'], config
                ).update(batch)
                with tempfile.TemporaryDirectory(prefix='m14_roundtrip_') as directory:
                    path = f'{directory}/params.pkl'
                    with open(path, 'wb') as handle:
                        pickle.dump(
                            {'agent': flax.serialization.to_state_dict(agent)}, handle
                        )
                    restored = restore_agent_from_checkpoint(agent, path)
                _tree_assert_equal(self, agent.network.params, restored.network.params)
                _tree_assert_equal(
                    self, agent.network.model_state, restored.network.model_state
                )
                action = agent.sample_actions(
                    batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(5)
                )
                restored_action = restored.sample_actions(
                    batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(5)
                )
                np.testing.assert_array_equal(np.asarray(action), np.asarray(restored_action))


class M14RuntimeAccountingTest(unittest.TestCase):
    def test_descriptor_runtime_uses_latent_dim_not_actor_width(self):
        config = _small_config('qrl')
        for slot_name in config.compute:
            _enable(config, slot_name)
        extras = _computation_runtime_extras(config)
        self.assertEqual(extras['slot_descriptors']['value']['role'], 'value')
        self.assertEqual(extras['slot_descriptors']['dynamics']['role'], 'dynamics')
        self.assertEqual(extras['slot_descriptors']['value']['state_dim'], 6)
        self.assertEqual(extras['slot_descriptors']['dynamics']['state_dim'], 6)
        self.assertEqual(extras['slot_descriptors']['actor']['state_dim'], 8)
        self.assertEqual(extras['slot_descriptors']['value']['hidden_dims'], [8, 8, 6])
        self.assertEqual(extras['slot_descriptors']['dynamics']['hidden_dims'], [8, 8, 6])

    def test_new_agent_accounting_is_explicitly_not_applicable_to_legacy_actor_audit(self):
        for name in ALGORITHMS:
            config = _small_config(name)
            for slot_name in config.compute:
                _enable(config, slot_name)
            agent = agents[name].create(
                53, _batch()['observations'], _batch()['actions'], config
            )
            generic = _computation_slot_accounting(agent, config)
            audit = _accounting_consistency_audit({}, generic, config)
            self.assertEqual(audit['status'], 'not_applicable')
            self.assertEqual(set(audit['generic_slots']), set(generic))


if __name__ == '__main__':
    unittest.main()
