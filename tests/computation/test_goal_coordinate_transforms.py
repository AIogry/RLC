"""V2 strict schema, direction, distance and setup/forward boundaries."""

import copy
import json
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from impls.networks.goal_conditioning import (
    goal_conditioning_layout, make_goal_conditioner, resolve_goal_conditioning,
)
from impls.representation.goal_coordinate_transforms import (
    generate_goal_coordinate_transform, resolve_goal_coordinate_transform,
)
from impls.representation.puzzle_algebra import apply_gf2_map, build_toggle_matrix
from impls.representation.puzzle_conditioning import extract_button_bits
from tests.reference.goal_conditioning import batch, goal_config, observations, role


class GoalCoordinateTransformsTest(unittest.TestCase):
    def test_direction_inverse_and_single_press_updates(self):
        config = goal_config(role('operation', 'P'), role('operation', 'B'))
        plan = resolve_goal_conditioning(config)
        p = config['transforms']['P']['permutation']
        matrix = np.asarray(config['transforms']['B']['matrix'], dtype=np.uint8)
        x = np.eye(9, dtype=np.uint8)
        current = observations(apply_gf2_map(build_toggle_matrix(3, 3), x))
        goal = observations(np.zeros_like(x))
        for conditioner, expected in ((plan.actor, x[:, p]), (plan.value_side, matrix.T)):
            prepared = conditioner.prepare(current, goal)
            np.testing.assert_array_equal(extract_button_bits(prepared.goals, num_buttons=9), expected)
            recovered = conditioner.transform.inverse()(expected)
            np.testing.assert_array_equal(recovered, x)
            # One press changes coordinates by P e_i / B e_i, independent of s.
            x0 = np.broadcast_to(np.arange(9, dtype=np.uint8) % 2, (9, 9))
            start = observations(apply_gf2_map(build_toggle_matrix(3, 3), x0))
            end = observations(apply_gf2_map(build_toggle_matrix(3, 3), x0 ^ x))
            z0 = extract_button_bits(conditioner.prepare(start, goal).goals, num_buttons=9)
            z1 = extract_button_bits(conditioner.prepare(end, goal).goals, num_buttons=9)
            np.testing.assert_array_equal(z0 ^ z1, expected)
        identity = resolve_goal_coordinate_transform({'kind': 'identity'}, num_coordinates=9)
        np.testing.assert_array_equal(identity(x), x)
        np.testing.assert_array_equal(np.sum(x[:, p], axis=-1), np.sum(x, axis=-1))

    def test_distance_uses_unencoded_operation_and_keeps_raw_state(self):
        config = goal_config(role('operation', 'B', 'exact_press_fraction'))
        plan = resolve_goal_conditioning(config)
        x = np.zeros((2, 3, 9), dtype=np.uint8)
        x[..., 1] = 1
        current = observations(apply_gf2_map(build_toggle_matrix(3, 3), x), 3)
        goal = observations(np.zeros_like(x), 9)
        before = np.asarray(current).copy()
        prepared = plan.actor.prepare(current, goal)
        np.testing.assert_array_equal(current, before)
        self.assertEqual(prepared.token_aux.shape, (2, 3, 9, 1))
        np.testing.assert_allclose(prepared.token_aux, 1 / 9)
        encoded = extract_button_bits(prepared.goals, num_buttons=9)
        np.testing.assert_array_equal(np.sum(encoded, axis=-1), 2)
        np.testing.assert_array_equal(prepared.goals[..., :19], 0)
        buttons = np.asarray(prepared.goals[..., 19:]).reshape(2, 3, 9, 4)
        np.testing.assert_array_equal(buttons[..., 2:], 0)
        same_board = plan.actor.prepare(current, observations(
            apply_gf2_map(build_toggle_matrix(3, 3), x), 99))
        np.testing.assert_array_equal(same_board.token_aux, 0)

    def test_residual_distance_and_unbatched_shapes(self):
        data = batch()
        for distance in ('zero', 'exact_press_fraction'):
            plan = resolve_goal_conditioning(goal_config(role('residual', distance_feature=distance)))
            for unbatched in (False, True):
                s, g = data['observations'], data['actor_goals']
                if unbatched:
                    s, g = s[0], g[0]
                output = plan.actor.prepare(s, g)
                d = extract_button_bits(s, num_buttons=9) ^ extract_button_bits(g, num_buttons=9)
                np.testing.assert_array_equal(extract_button_bits(output.goals, num_buttons=9), d)
                self.assertEqual(output.token_aux.shape, (*s.shape[:-1], 9, 1))
                if distance == 'zero':
                    np.testing.assert_array_equal(output.token_aux, 0)

    def test_residual_zero_never_constructs_or_applies_oracle(self):
        config = goal_config(rows=4, cols=4)
        config['transforms'] = {}
        with mock.patch('impls.networks.goal_conditioning.operator_metadata', side_effect=AssertionError('oracle setup')), \
             mock.patch('impls.representation.puzzle_conditioning.apply_gf2_map', side_effect=AssertionError('oracle forward')):
            plan = resolve_goal_conditioning(config)
            current = observations(np.zeros((2, 16), dtype=np.uint8))
            output = jax.jit(plan.actor.prepare)(current, current)
            np.testing.assert_array_equal(output.token_aux, 0)
            self.assertIsNone(plan.actor.operation_inverse)
        config['roles']['actor']['distance_feature'] = 'exact_press_fraction'
        with self.assertRaisesRegex(ValueError, 'full-rank'):
            resolve_goal_conditioning(config)

    def test_forward_uses_no_setup_algebra_or_randomness(self):
        plan = resolve_goal_conditioning(goal_config(role('operation', 'B', 'exact_press_fraction')))
        data = batch()
        with mock.patch('impls.networks.goal_conditioning.operator_metadata', side_effect=AssertionError('setup')), \
             mock.patch('impls.representation.puzzle_algebra.gf2_inverse', side_effect=AssertionError('inverse')), \
             mock.patch('impls.representation.goal_coordinate_transforms.gf2_rank', side_effect=AssertionError('rank')), \
             mock.patch('impls.representation.goal_coordinate_transforms.semantic_hash', side_effect=AssertionError('hash')), \
             mock.patch('numpy.random.default_rng', side_effect=AssertionError('RNG')), \
             mock.patch('builtins.open', side_effect=AssertionError('file')):
            eager = plan.actor.prepare(data['observations'], data['actor_goals'])
            compiled = jax.jit(plan.actor.prepare)(data['observations'], data['actor_goals'])
            for a, b in zip(eager, compiled):
                np.testing.assert_array_equal(a, b)

    def test_layout_does_no_transform_or_operator_work(self):
        config = goal_config(role('operation', 'B', 'exact_press_fraction'))
        with mock.patch('impls.networks.goal_conditioning.operator_metadata', side_effect=AssertionError('operator')), \
             mock.patch('impls.networks.goal_conditioning.resolve_goal_coordinate_transform', side_effect=AssertionError('transform')):
            layout = goal_conditioning_layout(config, dataset_class='PuzzleBoardGCDataset')
        self.assertEqual(layout['num_buttons'], 9)
        self.assertNotIn('roles', layout)
        self.assertNotIn('transforms', layout)

    def test_strict_config_failures(self):
        bad_configs = []
        def changed(mutator):
            config = goal_config()
            mutator(config)
            bad_configs.append(config)
        changed(lambda c: c.update(mode='residual'))
        changed(lambda c: c.update(unknown=True))
        changed(lambda c: c.update(schema_version=True))
        changed(lambda c: c.update(input_schema='flat'))
        changed(lambda c: c.update(robot_dim=18))
        changed(lambda c: c.update(rows=4))
        changed(lambda c: c['roles'].update(critic=role()))
        changed(lambda c: c['roles']['actor'].update(coordinate='board'))
        changed(lambda c: c['roles']['actor'].update(transform_id='missing'))
        changed(lambda c: c['roles']['actor'].update(transform_id='P'))
        changed(lambda c: c['roles']['actor'].update(distance_feature='popcount_y'))
        changed(lambda c: c['roles']['actor'].update(extra=1))
        changed(lambda c: c.update(transforms=None))
        changed(lambda c: c['transforms'].update(identity=c['transforms']['P']))
        for config in bad_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_goal_conditioning(config)
        with self.assertRaisesRegex(ValueError, 'board-equality'):
            resolve_goal_conditioning(goal_config(), dataset_class='GCDataset')
        with self.assertRaisesRegex(ValueError, 'make_goal_conditioners'):
            make_goal_conditioner(goal_config())

    def test_malformed_matrix_permutation_and_raw_input(self):
        malformed = [
            {'kind': 'permutation', 'permutation': [0, 0, 2]},
            {'kind': 'permutation', 'permutation': [0, True, 2]},
            {'kind': 'permutation', 'permutation': [0, 1]},
            {'kind': 'gf2_linear', 'matrix': [[1, 0], [0, 1]]},
            {'kind': 'gf2_linear', 'matrix': np.zeros((3, 3)).tolist()},
            {'kind': 'gf2_linear', 'matrix': np.full((3, 3), np.nan).tolist()},
            {'kind': 'gf2_linear', 'matrix': (np.eye(3) * 2).tolist()},
            {'kind': 'identity', 'transform_seed': 3},
            {'kind': 'gf2_linear', 'path': '/tmp/mutable.json'},
        ]
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                resolve_goal_coordinate_transform(payload, num_coordinates=3)
        for kind in ('identity', 'permutation', 'gf2_linear'):
            transform = resolve_goal_coordinate_transform(
                generate_goal_coordinate_transform(kind=kind, num_coordinates=3, transform_seed=9),
                num_coordinates=3,
            )
            for bits in ([1, 0, 2], [1, float('nan'), 0], [1, 0]):
                with self.assertRaises(ValueError):
                    transform(bits)
        plan = resolve_goal_conditioning(goal_config())
        data = batch()
        for invalid in (data['observations'].at[0, 19:21].set(1),
                        data['observations'].at[0, 19].set(0.5),
                        data['observations'].at[0, 0].set(jnp.nan),
                        data['observations'][:, :-1]):
            with self.assertRaises(ValueError):
                plan.actor.prepare(invalid, data['actor_goals'])

    def test_json_resolution_hash_and_private_generation_rng(self):
        plan = resolve_goal_conditioning(goal_config(role('operation', 'P')))
        again = resolve_goal_conditioning(json.loads(json.dumps(plan.to_config())))
        self.assertEqual(plan, again)
        state = np.random.get_state()
        for kind in ('identity', 'permutation', 'gf2_linear'):
            first = generate_goal_coordinate_transform(kind=kind, num_coordinates=9, transform_seed=26)
            second = generate_goal_coordinate_transform(kind=kind, num_coordinates=9, transform_seed=26)
            self.assertEqual(first, second)
            changed = copy.deepcopy(first)
            changed['provenance']['transform_seed'] = 999
            changed['provenance']['generator'] = 'different_source'
            self.assertEqual(resolve_goal_coordinate_transform(changed, num_coordinates=9).content_sha256,
                             first['content_sha256'])
        for a, b in zip(state, np.random.get_state()):
            np.testing.assert_array_equal(a, b)
        with mock.patch('impls.representation.goal_coordinate_transforms.gf2_rank', return_value=0):
            with self.assertRaisesRegex(ValueError, '2 attempts'):
                generate_goal_coordinate_transform(kind='gf2_linear', num_coordinates=9,
                                                   transform_seed=26, max_attempts=2)


if __name__ == '__main__':
    unittest.main()
