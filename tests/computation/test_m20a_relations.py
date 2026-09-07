"""Focused unit coverage for the M20A first-class relation path."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from impls.computation.factory import ComputationSpec, make_computation_core
from impls.computation.readouts import HybridContextQueryReadout
from impls.computation.relation import RelationAugmenter
from impls.representation.interfaces import StructuredRepresentation
from impls.representation.manipulation import parse_cube_observation, parse_scene_observation
from impls.representation.puzzle import PuzzleTokenAdapter, parse_puzzle_observation
from impls.representation.relations import (
    CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
    PUZZLE_4X4_SHUFFLE_PERMUTATION,
    build_cube_relations,
    build_puzzle_relations,
    build_scene_relations,
    relation_edge_counts,
)


def _puzzle_spec(mode):
    return ComputationSpec.from_mapping({
        'primitive': 'mlp',
        'structure': 'puzzle_tokens',
        'structure_kwargs': {
            'num_buttons': 16, 'robot_dim': 19, 'button_feature_dim': 4,
            'token_dim': 128, 'robot_hidden_dim': 128, 'index_embedding': True,
        },
        'block': 'mlp_mixer',
        'block_kwargs': {
            'num_blocks': 2, 'token_hidden_dim': 64,
            'channel_hidden_dim': 256, 'tm_mode': 'none',
        },
        'topology': 'feedforward',
        'credit': 'direct',
        'input_semantics': 'goal_pair',
        'action_semantics': 'none',
        'relation_mode': mode,
        'relation_kwargs': {
            'num_relation_types': 1, 'rows': 4, 'cols': 4,
            'shuffle_permutation': list(PUZZLE_4X4_SHUFFLE_PERMUTATION),
        },
        'relation_augmenter': 'relation_mlp',
        'relation_augmenter_kwargs': {
            'relation_hidden_dim': 256, 'activation': 'gelu',
            'first_use_bias': False, 'second_use_bias': False,
            'normalization': 'none', 'dropout': 'none', 'output_dim': 128,
        },
        'readout': 'hybrid_context_query',
        'readout_kwargs': {'output_dim': 8, 'query_dim': 128},
    })


class M20ARelationUnitTest(unittest.TestCase):
    def test_structured_representation_appends_relation_fields(self):
        rep = StructuredRepresentation('tokens', 'context', 'mask', 'auxiliary')
        self.assertEqual(rep.tokens, 'tokens')
        self.assertEqual(rep.context, 'context')
        self.assertEqual(rep.mask, 'mask')
        self.assertEqual(rep.auxiliary, 'auxiliary')
        self.assertIsNone(rep.relations)
        self.assertIsNone(rep.relation_mask)

    def test_strict_entity_parsers(self):
        robot, buttons = parse_puzzle_observation(jnp.zeros((2, 83)), num_buttons=16)
        self.assertEqual(robot.shape, (2, 19))
        self.assertEqual(buttons.shape, (2, 16, 4))
        cube_robot, cubes = parse_cube_observation(jnp.zeros((2, 46)), num_cubes=3)
        self.assertEqual(cube_robot.shape, (2, 19))
        self.assertEqual(cubes.shape, (2, 3, 9))
        scene = parse_scene_observation(jnp.zeros((2, 40)))
        self.assertEqual([part.shape for part in scene], [(2, 19), (2, 9), (2, 2, 4), (2, 2), (2, 2)])
        with self.assertRaises(ValueError):
            parse_cube_observation(jnp.zeros((2, 45)), num_cubes=3)

    def test_puzzle_scene_and_cube_controls(self):
        puzzle = build_puzzle_relations(
            jnp.zeros((2, 83)), jnp.zeros((2, 83)), mode='correct', rows=4, cols=4,
            shuffle_permutation=PUZZLE_4X4_SHUFFLE_PERMUTATION,
        )
        shuffled_puzzle = build_puzzle_relations(
            jnp.zeros((2, 83)), jnp.zeros((2, 83)), mode='shuffled', rows=4, cols=4,
            shuffle_permutation=PUZZLE_4X4_SHUFFLE_PERMUTATION,
        )
        self.assertEqual(puzzle.shape, (2, 16, 16, 1))
        self.assertTrue(np.all(np.asarray(relation_edge_counts(puzzle)) == 64))
        self.assertTrue(np.array_equal(relation_edge_counts(puzzle), relation_edge_counts(shuffled_puzzle)))
        self.assertFalse(np.array_equal(np.asarray(puzzle), np.asarray(shuffled_puzzle)))

        scene = build_scene_relations(jnp.zeros((2, 40)), jnp.zeros((2, 40)), mode='correct')
        scene_shuffled = build_scene_relations(jnp.zeros((2, 40)), jnp.zeros((2, 40)), mode='shuffled')
        self.assertEqual(scene.shape, (2, 5, 5, 1))
        self.assertTrue(np.all(np.asarray(relation_edge_counts(scene)) == 2))
        self.assertTrue(np.array_equal(relation_edge_counts(scene), relation_edge_counts(scene_shuffled)))

        state = np.zeros((1, 3, 9), dtype=np.float32)
        goal = np.zeros((1, 3, 9), dtype=np.float32)
        state[0, 1, 2] = 0.4  # z=0.04 after canonical /10 scaler.
        cube = build_cube_relations(
            state, goal, mode='correct',
            current_support_epsilon_xy=0.02, current_support_epsilon_z=0.01,
            goal_support_epsilon_xy=0.02, goal_support_epsilon_z=0.01,
            conflict_radius=0.04,
        )
        cube_shuffled = build_cube_relations(
            state, goal, mode='shuffled',
            current_support_epsilon_xy=0.02, current_support_epsilon_z=0.01,
            goal_support_epsilon_xy=0.02, goal_support_epsilon_z=0.01,
            conflict_radius=0.04,
            shuffle_derangement=CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
        )
        self.assertEqual(cube.shape, (1, 3, 3, 3))
        self.assertEqual(float(cube[0, 0, 1, 0]), 1.0)
        self.assertEqual(float(cube[0, 1, 0, 0]), 0.0)
        self.assertTrue(np.array_equal(relation_edge_counts(cube), relation_edge_counts(cube_shuffled)))
        self.assertTrue(np.all(build_cube_relations(
            state, goal, mode='zero',
            current_support_epsilon_xy=None, current_support_epsilon_z=None,
            goal_support_epsilon_xy=None, goal_support_epsilon_z=None,
            conflict_radius=None,
        ) == 0))

    def test_relation_augmenter_direction_mask_and_zero_identity(self):
        tokens = jnp.asarray([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
        zero = jnp.zeros((1, 3, 3, 1))
        module = RelationAugmenter(token_dim=2, relation_hidden_dim=256)
        variables = module.init(jax.random.PRNGKey(1), tokens, zero)
        np.testing.assert_array_equal(np.asarray(module.apply(variables, tokens, zero)), np.asarray(tokens))
        relations = np.zeros((1, 3, 3, 1), dtype=np.float32)
        relations[0, 0, 1, 0] = 1.0
        features, _ = RelationAugmenter.aggregate_features(tokens, relations)
        features = np.asarray(features)
        np.testing.assert_array_equal(features[0, 0, :2], np.asarray(tokens)[0, 1])
        np.testing.assert_array_equal(features[0, 1, 2:], np.asarray(tokens)[0, 0])
        masked, _ = RelationAugmenter.aggregate_features(
            tokens, relations, relation_mask=np.zeros((1, 3, 3), dtype=bool)
        )
        np.testing.assert_array_equal(np.asarray(masked), np.zeros_like(features))

    def test_hybrid_readout_mask_and_permutation(self):
        tokens = jnp.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        context = jnp.asarray([[0.5, -0.5, 1.0]])
        mask = jnp.asarray([[True, False, True]])
        module = HybridContextQueryReadout(output_dim=7, token_dim=2, query_dim=128)
        variables = module.init(jax.random.PRNGKey(2), tokens, context=context, mask=mask)
        output = module.apply(variables, tokens, context=context, mask=mask)
        attention = module.apply(
            variables, tokens, context=context, mask=mask,
            method=HybridContextQueryReadout.attention_weights,
        )
        self.assertEqual(output.shape, (1, 7))
        self.assertEqual(float(attention[0, 1]), 0.0)
        self.assertAlmostEqual(float(attention.sum()), 1.0, places=6)
        permutation = np.asarray([2, 0, 1])
        permuted = module.apply(
            variables, tokens[:, permutation], context=context, mask=mask[:, permutation]
        )
        np.testing.assert_allclose(np.asarray(output), np.asarray(permuted), rtol=1e-6, atol=1e-6)

    def test_m20_factory_parameter_parity_and_legacy_adapter_parity(self):
        x = jnp.zeros((2, 166), dtype=jnp.float32)
        cores = [
            make_computation_core(_puzzle_spec(mode), hidden_dims=(8,), activate_final=True)
            for mode in ('zero', 'correct', 'shuffled')
        ]
        variables = [core.init(jax.random.PRNGKey(3), x) for core in cores]
        reference = flatten_dict(variables[0]['params'])
        for candidate in variables[1:]:
            current = flatten_dict(candidate['params'])
            self.assertEqual(set(reference), set(current))
            for key in reference:
                np.testing.assert_array_equal(np.asarray(reference[key]), np.asarray(current[key]))
        diagnostic = cores[1].apply(variables[1], x, method='relation_diagnostic')
        self.assertEqual(diagnostic['relations'].shape, (2, 16, 16, 1))

        legacy = PuzzleTokenAdapter(num_buttons=16)
        explicit = PuzzleTokenAdapter(num_buttons=16, relation_mode='legacy_none')
        legacy_vars = legacy.init(jax.random.PRNGKey(4), jnp.zeros((2, 166)))
        explicit_vars = explicit.init(jax.random.PRNGKey(4), jnp.zeros((2, 166)))
        self.assertEqual(set(flatten_dict(legacy_vars['params'])), set(flatten_dict(explicit_vars['params'])))
        legacy_output = legacy.apply(legacy_vars, jnp.zeros((2, 166)))
        explicit_output = explicit.apply(explicit_vars, jnp.zeros((2, 166)))
        np.testing.assert_array_equal(np.asarray(legacy_output.tokens), np.asarray(explicit_output.tokens))
        np.testing.assert_array_equal(np.asarray(legacy_output.context), np.asarray(explicit_output.context))
        self.assertIsNone(legacy_output.relations)


if __name__ == '__main__':
    unittest.main()
