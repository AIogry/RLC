"""Network-level and GCIQL parity gates for Puzzle goal conditioning."""

import copy
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from impls.agents import agent_configs, agents
from impls.main import _computation_runtime_extras, _make_config, _parse_args
from impls.networks.common import GCValue
from impls.networks.goal_conditioning import (
    make_goal_conditioner,
    validate_goal_conditioning_config,
)
from impls.representation.puzzle_conditioning import condition_puzzle_goal


def _goal_config(mode, rows=3, cols=3):
    policy = 'preserve' if mode == 'canonical' else 'zero'
    return {
        'domain': 'puzzle',
        'mode': mode,
        'rows': rows,
        'cols': cols,
        'num_buttons': rows * cols,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'robot_goal_policy': policy,
        'button_goal_transient_policy': policy,
    }


def _observations(bits, offset=0.0):
    bits = np.asarray(bits, dtype=np.uint8)
    leading = bits.shape[:-1]
    num_buttons = bits.shape[-1]
    robot = np.broadcast_to(
        np.arange(19, dtype=np.float32) + offset,
        (*leading, 19),
    ).copy()
    buttons = np.zeros((*leading, num_buttons, 4), dtype=np.float32)
    buttons[..., 0] = 1 - bits
    buttons[..., 1] = bits
    buttons[..., 2] = offset + 0.25
    buttons[..., 3] = -offset - 0.5
    return jnp.asarray(
        np.concatenate([robot, buttons.reshape(*leading, -1)], axis=-1)
    )


def _batch():
    bits = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 0, 0],
    ], dtype=np.uint8)
    next_bits = bits.copy()
    next_bits[:, [0, 1, 3]] ^= 1
    goal_bits = np.asarray([
        [1, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 1],
    ], dtype=np.uint8)
    return {
        'observations': _observations(bits, 1),
        'next_observations': _observations(next_bits, 2),
        'actions': jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10,
        'value_goals': _observations(goal_bits, 3),
        'actor_goals': _observations(goal_bits[::-1].copy(), 4),
        'rewards': jnp.asarray([-1.0, 0.0, -1.0]),
        'masks': jnp.asarray([1.0, 0.0, 1.0]),
    }


def _config(mode=None, *, structured=True):
    config = copy.deepcopy(agent_configs['gciql']())
    config.actor_hidden_dims = (8,)
    config.value_hidden_dims = (8,)
    config.batch_size = 3
    if mode is not None:
        config.goal_conditioning = _goal_config(mode)
        config.dataset_class = (
            'GCDataset' if mode == 'canonical' else 'PuzzleBoardGCDataset'
        )
    if structured:
        structure_kwargs = {
            'num_buttons': 9,
            'robot_dim': 19,
            'button_feature_dim': 4,
            'token_dim': 7,
            'robot_hidden_dim': 8,
            'token_mlp_hidden_dim': 5,
            'channel_mlp_hidden_dim': 11,
            'num_mixer_blocks': 2,
            'index_embedding': True,
            'readout': 'mean',
            'tm_mode': 'none',
        }
        for slot_name in ('actor', 'value', 'critic'):
            slot = config.compute[slot_name]
            slot.enabled = True
            slot.primitive = 'mlp'
            slot.structure = 'puzzle_tokens'
            slot.block = 'mlp_mixer'
            slot.topology = 'feedforward'
            slot.credit = 'direct'
            slot.structure_kwargs = copy.deepcopy(structure_kwargs)
    return config


def _assert_tree_equal(testcase, left, right):
    left_flat = flatten_dict(left)
    right_flat = flatten_dict(right)
    testcase.assertEqual(set(left_flat), set(right_flat))
    for key in left_flat:
        np.testing.assert_array_equal(np.asarray(left_flat[key]), np.asarray(right_flat[key]))


class GCIQLPuzzleGoalConditioningTest(unittest.TestCase):
    def test_three_modes_have_identical_trainable_parameter_trees_and_counts(self):
        batch = _batch()
        agents_by_mode = {
            mode: agents['gciql'].create(
                24005, batch['observations'], batch['actions'], _config(mode)
            )
            for mode in ('board', 'residual', 'oracle_operation')
        }
        reference = agents_by_mode['board'].network.params
        reference_flat = flatten_dict(reference)
        reference_shapes = {key: value.shape for key, value in reference_flat.items()}
        reference_count = sum(value.size for value in reference_flat.values())
        reference_opt_structure = jax.tree_util.tree_structure(
            agents_by_mode['board'].network.opt_state
        )
        reference_opt_leaves = jax.tree_util.tree_leaves(
            agents_by_mode['board'].network.opt_state
        )
        for mode in ('residual', 'oracle_operation'):
            candidate = agents_by_mode[mode].network.params
            candidate_flat = flatten_dict(candidate)
            self.assertEqual(
                {key: value.shape for key, value in candidate_flat.items()},
                reference_shapes,
            )
            self.assertEqual(sum(value.size for value in candidate_flat.values()), reference_count)
            _assert_tree_equal(self, candidate, reference)
            candidate_opt = agents_by_mode[mode].network.opt_state
            self.assertEqual(
                jax.tree_util.tree_structure(candidate_opt), reference_opt_structure
            )
            for actual, expected in zip(
                jax.tree_util.tree_leaves(candidate_opt), reference_opt_leaves
            ):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            self.assertFalse(agents_by_mode[mode].network.model_state)
        self.assertFalse(agents_by_mode['board'].network.model_state)
        self.assertFalse(any('condition' in '/'.join(key) for key in reference_flat))

    def test_explicit_canonical_matches_none_forward_loss_update_and_rng(self):
        batch = _batch()
        historical = agents['gciql'].create(
            24006, batch['observations'], batch['actions'], _config(None)
        )
        canonical = agents['gciql'].create(
            24006, batch['observations'], batch['actions'], _config('canonical')
        )
        _assert_tree_equal(self, historical.network.params, canonical.network.params)
        historical_loss, historical_info = historical.total_loss(
            batch, historical.network.params, rng=jax.random.PRNGKey(9)
        )
        canonical_loss, canonical_info = canonical.total_loss(
            batch, canonical.network.params, rng=jax.random.PRNGKey(9)
        )
        np.testing.assert_array_equal(historical_loss, canonical_loss)
        _assert_tree_equal(self, historical_info, canonical_info)
        historical_updated, historical_update_info = historical.update(batch)
        canonical_updated, canonical_update_info = canonical.update(batch)
        _assert_tree_equal(
            self, historical_updated.network.params, canonical_updated.network.params
        )
        _assert_tree_equal(self, historical_update_info, canonical_update_info)
        np.testing.assert_array_equal(historical_updated.rng, canonical_updated.rng)

    def test_flat_network_recomputes_state_dependent_condition_per_call(self):
        batch = _batch()
        for mode in ('residual', 'oracle_operation'):
            with self.subTest(mode=mode):
                config = _goal_config(mode)
                conditioner = make_goal_conditioner(config)
                conditioned_value = GCValue(
                    hidden_dims=(8,),
                    layer_norm=False,
                    ensemble=False,
                    goal_conditioner=conditioner,
                )
                plain_value = GCValue(
                    hidden_dims=(8,), layer_norm=False, ensemble=False
                )
                variables = conditioned_value.init(
                    jax.random.PRNGKey(24007),
                    batch['observations'],
                    batch['value_goals'],
                )
                plain_variables = plain_value.init(
                    jax.random.PRNGKey(24007),
                    batch['observations'],
                    batch['value_goals'],
                )
                _assert_tree_equal(self, variables['params'], plain_variables['params'])
                first_goal = conditioner(batch['observations'], batch['value_goals'])
                next_goal = conditioner(
                    batch['next_observations'], batch['value_goals']
                )
                self.assertFalse(np.array_equal(np.asarray(first_goal), np.asarray(next_goal)))
                actual_first = conditioned_value.apply(
                    variables, batch['observations'], batch['value_goals']
                )
                actual_next = conditioned_value.apply(
                    variables, batch['next_observations'], batch['value_goals']
                )
                expected_first = plain_value.apply(
                    plain_variables, batch['observations'], first_goal
                )
                expected_next = plain_value.apply(
                    plain_variables, batch['next_observations'], next_goal
                )
                np.testing.assert_array_equal(actual_first, expected_first)
                np.testing.assert_array_equal(actual_next, expected_next)

    def test_critic_action_is_appended_after_conditioning(self):
        batch = _batch()
        for mode in ('board', 'residual', 'oracle_operation'):
            with self.subTest(mode=mode):
                conditioner = make_goal_conditioner(_goal_config(mode))
                conditioned_critic = GCValue(
                    hidden_dims=(8,),
                    layer_norm=False,
                    ensemble=False,
                    goal_conditioner=conditioner,
                )
                plain_critic = GCValue(
                    hidden_dims=(8,), layer_norm=False, ensemble=False
                )
                variables = conditioned_critic.init(
                    jax.random.PRNGKey(24008),
                    batch['observations'],
                    batch['value_goals'],
                    batch['actions'],
                )
                plain_variables = plain_critic.init(
                    jax.random.PRNGKey(24008),
                    batch['observations'],
                    batch['value_goals'],
                    batch['actions'],
                )
                conditioned_goal = conditioner(
                    batch['observations'], batch['value_goals']
                )
                for actions in (batch['actions'], batch['actions'] + 7):
                    actual = conditioned_critic.apply(
                        variables,
                        batch['observations'],
                        batch['value_goals'],
                        actions,
                    )
                    expected = plain_critic.apply(
                        plain_variables,
                        batch['observations'],
                        conditioned_goal,
                        actions,
                    )
                    np.testing.assert_array_equal(actual, expected)

    def test_config_layout_rank_dataset_and_runtime_provenance_validation(self):
        with self.assertRaisesRegex(ValueError, 'full-rank'):
            validate_goal_conditioning_config(
                _goal_config('oracle_operation', 4, 4),
                dataset_class='PuzzleBoardGCDataset',
            )
        mismatch = _config('board')
        mismatch.compute.actor.structure_kwargs.num_buttons = 8
        with self.assertRaisesRegex(ValueError, 'conflicts with structured slot'):
            validate_goal_conditioning_config(
                mismatch.goal_conditioning,
                compute_slots=mismatch.compute,
                dataset_class=mismatch.dataset_class,
            )
        with self.assertRaisesRegex(ValueError, 'board-equality'):
            validate_goal_conditioning_config(
                _goal_config('residual'), dataset_class='GCDataset'
            )
        hidden = _config(None)
        hidden.compute.actor.structure_kwargs.goal_conditioning_mode = 'residual'
        with self.assertRaisesRegex(ValueError, 'shared top-level network semantic'):
            validate_goal_conditioning_config(
                None,
                compute_slots=hidden.compute,
                dataset_class=hidden.dataset_class,
            )
        flat = _config('board', structured=False)
        batch = _batch()
        agent = agents['gciql'].create(
            24009, batch['observations'], batch['actions'], flat
        )
        self.assertEqual(
            agent.sample_actions(
                batch['observations'][:1],
                batch['actor_goals'][:1],
                seed=jax.random.PRNGKey(10),
            ).shape,
            (1, 2),
        )
        runtime = _computation_runtime_extras(_config('oracle_operation'))
        expected_goal_metadata = {
            **_goal_config('oracle_operation'),
            'operator_matrix_source': 'canonical_puzzle_toggle_rule',
            'operator_orientation': 'M[target,source]',
            'operator_rank': 9,
            'operator_full_rank': True,
        }
        self.assertEqual(runtime['goal_conditioning'], expected_goal_metadata)
        self.assertEqual(runtime['goal_success_semantics'], 'board_equality')
        self.assertEqual(runtime['dataset_class'], 'PuzzleBoardGCDataset')

    def test_main_resolution_keeps_one_top_level_shared_semantic(self):
        source = _config('residual')
        configuration = type('Configuration', (), {
            'data': {
                'algorithm': 'gciql',
                'agent_overrides': {
                    'dataset_class': 'PuzzleBoardGCDataset',
                    'goal_conditioning': dict(source.goal_conditioning),
                    'compute': {
                        slot_name: dict(source.compute[slot_name])
                        for slot_name in ('actor', 'value', 'critic')
                    },
                },
            }
        })()
        resolved = _make_config(
            _parse_args(['--agent', 'gciql']), configuration=configuration
        )
        self.assertEqual(resolved.goal_conditioning.mode, 'residual')
        for slot_name in ('actor', 'value', 'critic'):
            self.assertNotIn('goal_conditioning', resolved.compute[slot_name])

    def test_network_result_matches_explicit_manual_conditioning(self):
        batch = _batch()
        conditioner = make_goal_conditioner(_goal_config('board'))
        conditioned = GCValue(
            hidden_dims=(8,), layer_norm=False, ensemble=False,
            goal_conditioner=conditioner,
        )
        plain = GCValue(hidden_dims=(8,), layer_norm=False, ensemble=False)
        conditioned_vars = conditioned.init(
            jax.random.PRNGKey(24010), batch['observations'], batch['value_goals']
        )
        plain_vars = plain.init(
            jax.random.PRNGKey(24010), batch['observations'], batch['value_goals']
        )
        manual_goal = condition_puzzle_goal(
            batch['observations'],
            batch['value_goals'],
            mode='board',
            rows=3,
            cols=3,
            num_buttons=9,
        )
        np.testing.assert_array_equal(
            conditioned.apply(
                conditioned_vars, batch['observations'], batch['value_goals']
            ),
            plain.apply(plain_vars, batch['observations'], manual_goal),
        )

    def test_oracle_network_forward_never_runs_setup_algebra(self):
        batch = _batch()
        conditioner = make_goal_conditioner(_goal_config('oracle_operation'))
        value = GCValue(
            hidden_dims=(8,),
            layer_norm=False,
            ensemble=False,
            goal_conditioner=conditioner,
        )
        variables = value.init(
            jax.random.PRNGKey(24011),
            batch['observations'],
            batch['value_goals'],
        )
        with mock.patch(
            'impls.networks.goal_conditioning.operator_metadata',
            side_effect=AssertionError('forward requested setup-time algebra'),
        ), mock.patch(
            'impls.representation.puzzle_conditioning.operator_metadata',
            side_effect=AssertionError('forward requested setup-time algebra'),
        ):
            output = value.apply(
                variables, batch['observations'], batch['value_goals']
            )
        self.assertTrue(np.all(np.isfinite(np.asarray(output))))


if __name__ == '__main__':
    unittest.main()
