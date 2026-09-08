"""Correctness tests for the reusable relation-free Cube structured path."""

import copy
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.gciql import GCIQLAgent
from impls.computation.accounting import (
    gciql_architecture_accounting,
    relation_free_structured_body_accounting,
)
from impls.computation.factory import ComputationSpec, make_computation_core
from impls.experiment import prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.computation.slots import validate_compute_slots


ROOT = __import__('pathlib').Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M20B_cube_entity_mixer_scaling/study.yaml'


def _cube_spec(num_cubes, *, action_semantics='none'):
    return ComputationSpec.from_mapping({
        'primitive': 'mlp',
        'structure': 'cube_tokens',
        'block': 'mlp_mixer',
        'topology': 'feedforward',
        'parameter_sharing': 'shared',
        'credit': 'direct',
        'input_semantics': 'goal_pair',
        'action_semantics': action_semantics,
        'structure_kwargs': {
            'num_cubes': num_cubes,
            'robot_dim': 19,
            'cube_feature_dim': 9,
            'token_dim': 128,
            'robot_hidden_dim': 128,
            'slot_identity_embedding': False,
        },
        'block_kwargs': {
            'num_blocks': 2,
            'token_hidden_dim': 64,
            'channel_hidden_dim': 256,
            'tm_mode': 'none',
        },
        'readout': 'mean_context',
        'readout_kwargs': {'output_dim': 512},
        'relation_mode': 'legacy_none',
        'relation_kwargs': {},
        'relation_augmenter': 'none',
        'relation_augmenter_kwargs': {},
    })


def _goal_pair(num_cubes, batch_size=4, action_dim=0):
    observation_dim = 19 + 9 * num_cubes
    width = 2 * observation_dim + action_dim
    return jnp.arange(batch_size * width, dtype=jnp.float32).reshape(batch_size, width) / 101.0


def _tree_changed(left, right):
    return any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


class RelationFreeCubeFactoryTest(unittest.TestCase):
    def test_dynamic_entity_count_and_relation_absence(self):
        for num_cubes in (1, 2, 3):
            with self.subTest(num_cubes=num_cubes):
                spec = _cube_spec(num_cubes)
                core = make_computation_core(
                    spec,
                    hidden_dims=(512, 512, 512),
                    activate_final=True,
                    layer_norm=True,
                )
                inputs = _goal_pair(num_cubes)
                variables = core.init(jax.random.PRNGKey(num_cubes), inputs)
                with mock.patch(
                    'impls.representation.manipulation.build_cube_relations',
                    side_effect=AssertionError('relation builder must not execute'),
                ):
                    output = core.apply(variables, inputs)
                self.assertEqual(output.representation.shape, (4, 512))
                self.assertNotIn('relations', output.auxiliary)
                self.assertNotIn('relation_mask', output.auxiliary)
                self.assertNotIn('relation_augmenter', variables['params'])
                self.assertNotIn('cube_slot_embedding', variables['params']['adapter'])
                for block_name in ('blocks_0', 'blocks_1'):
                    block = variables['params']['core']['topology']['primitive'][block_name]
                    self.assertNotIn('tm_weights', block)
                    self.assertEqual(block['token_dense2']['kernel'].shape[-1], num_cubes)
                report = relation_free_structured_body_accounting(
                    variables['params'],
                    spec.structure_kwargs,
                    block_kwargs=spec.block_kwargs,
                )
                self.assertEqual(report['num_tokens'], num_cubes)
                self.assertEqual(report['relation_params'], 0)
                self.assertEqual(report['relation_macs'], 0)
                self.assertEqual(report['structured_sequential_depth'], 10)

    def test_critic_action_stays_in_robot_context(self):
        spec = _cube_spec(2, action_semantics='robot_context')
        core = make_computation_core(
            spec, hidden_dims=(512, 512, 512), activate_final=True, layer_norm=True,
        )
        inputs = _goal_pair(2, action_dim=5)
        variables = core.init(jax.random.PRNGKey(22), inputs)
        first = core.apply(variables, inputs)
        changed = inputs.at[:, -5:].set(7.0)
        second = core.apply(variables, changed)
        np.testing.assert_array_equal(
            np.asarray(first.auxiliary['computed_tokens']),
            np.asarray(second.auxiliary['computed_tokens']),
        )
        self.assertFalse(np.array_equal(
            np.asarray(first.representation), np.asarray(second.representation),
        ))

    def test_schema_accepts_only_relation_free_cube_recipe(self):
        spec = _cube_spec(2)
        slot = {
            'enabled': True,
            'primitive': spec.primitive,
            'structure': spec.structure,
            'structure_kwargs': dict(spec.structure_kwargs),
            'block': spec.block,
            'block_kwargs': dict(spec.block_kwargs),
            'topology': spec.topology,
            'parameter_sharing': spec.parameter_sharing,
            'credit': spec.credit,
            'relation_mode': spec.relation_mode,
            'relation_kwargs': {},
            'relation_augmenter': spec.relation_augmenter,
            'relation_augmenter_kwargs': {},
            'readout': spec.readout,
            'readout_kwargs': {'output_dim': 512},
        }
        validate_compute_slots('gciql', {'compute': {'actor': slot, 'value': slot, 'critic': slot}})
        invalid = copy.deepcopy(slot)
        invalid['relation_kwargs'] = {'num_relation_types': 3}
        with self.assertRaises(ValueError):
            validate_compute_slots('gciql', {'compute': {'actor': invalid}})


class RelationFreeCubeGCIQLTest(unittest.TestCase):
    def test_all_gciql_slots_are_structured_and_target_updates(self):
        _, configuration = prepare_run_design(STUDY, 'M20B-double-S002')
        config = _make_config(_parse_args(['--agent', 'gciql']), configuration=configuration)
        observations = jnp.zeros((4, 37), dtype=jnp.float32)
        actions = jnp.zeros((4, 5), dtype=jnp.float32)
        agent = GCIQLAgent.create(0, observations, actions, config)
        batch = {
            'observations': observations,
            'next_observations': observations + 0.01,
            'actions': actions,
            'value_goals': observations + 0.02,
            'actor_goals': observations + 0.03,
            'rewards': jnp.asarray([-1.0, -1.0, 0.0, -1.0]),
            'masks': jnp.asarray([1.0, 1.0, 0.0, 1.0]),
        }
        before = agent.network.params['modules_critic']
        updated, info = agent.update(batch)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(value))) for value in info.values()))
        self.assertTrue(_tree_changed(before, updated.network.params['modules_critic']))
        self.assertEqual(updated.sample_actions(
            observations[:1], batch['actor_goals'][:1], seed=jax.random.PRNGKey(7),
        ).shape, (1, 5))

        report = _computation_slot_accounting(updated, config)
        self.assertEqual(set(report), {'actor', 'value', 'critic'})
        for slot_name, item in report.items():
            with self.subTest(slot=slot_name):
                self.assertEqual(item['structure'], 'cube_tokens')
                self.assertEqual(item['num_tokens'], 2)
                self.assertEqual(item['relation_mode'], 'legacy_none')
                self.assertEqual(item['relation_augmenter'], 'none')
                self.assertEqual(item['relation_params'], 0)
                self.assertEqual(item['relation_macs'], 0)
                self.assertEqual(item['block_depth_L'], 2)
                self.assertEqual(item['readout'], 'mean_context')
        architecture = gciql_architecture_accounting(updated.network.params, config, report)
        for slot_name in ('actor', 'value', 'critic'):
            self.assertEqual(architecture['slots'][slot_name]['relation_params'], 0)
            self.assertEqual(architecture['slots'][slot_name]['relation_macs'], 0)


if __name__ == '__main__':
    unittest.main()
