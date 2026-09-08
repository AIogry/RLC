"""Synthetic integration coverage for the M20A-D diagnostic-only path.

The test deliberately uses a small synthetic Cube-shaped batch.  It does not
read a source run, a real checkpoint, or a dataset artifact, so it remains
safe and reusable in CI while exercising the same restored-agent API used by
the formal diagnostic tool.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.gciql import GCIQLAgent
from tools.m20a_relation_utilization import _gradient_metrics, _pytree_fingerprint


def _slot(output_dim=8):
    return {
        'enabled': True,
        'primitive': 'mlp',
        'structure': 'cube_tokens',
        'structure_kwargs': {
            'num_cubes': 3,
            'robot_dim': 19,
            'cube_feature_dim': 9,
            'token_dim': 128,
            'robot_hidden_dim': 128,
            'slot_identity_embedding': False,
        },
        'block': 'mlp_mixer',
        'block_kwargs': {
            'num_blocks': 2,
            'token_hidden_dim': 64,
            'channel_hidden_dim': 256,
            'tm_mode': 'none',
        },
        'topology': 'feedforward',
        'credit': 'direct',
        'relation_mode': 'correct',
        'relation_kwargs': {
            'num_relation_types': 3,
            'shuffle_derangement': [1, 2, 0],
            'current_support_epsilon_xy': 0.02,
            'current_support_epsilon_z': 0.01,
            'goal_support_epsilon_xy': 0.02,
            'goal_support_epsilon_z': 0.01,
            'goal_conflict_radius': 0.04,
            'threshold_status': 'FROZEN_USER_PHASE2',
            'cube_height': 0.04,
            'xyz_scaler': 10.0,
        },
        'relation_augmenter': 'relation_mlp',
        'relation_augmenter_kwargs': {
            'relation_hidden_dim': 256,
            'activation': 'gelu',
            'first_use_bias': False,
            'second_use_bias': False,
            'normalization': 'none',
            'dropout': 'none',
            'output_dim': 128,
        },
        'readout': 'hybrid_context_query',
        'readout_kwargs': {'output_dim': output_dim, 'query_dim': 128},
    }


def _config():
    return {
        'agent_name': 'gciql',
        'lr': 3e-4,
        'batch_size': 3,
        'actor_hidden_dims': (8, 8),
        'value_hidden_dims': (8, 8),
        'layer_norm': True,
        'discount': 0.99,
        'tau': 0.005,
        'expectile': 0.9,
        'actor_loss': 'ddpgbc',
        'alpha': 1.0,
        'const_std': True,
        'discrete': False,
        'encoder': None,
        'dataset_class': 'GCDataset',
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 1.0,
        'actor_p_randomgoal': 0.0,
        'actor_geom_sample': False,
        'gc_negative': True,
        'p_aug': 0.0,
        'frame_stack': None,
        'compute': {
            'actor': _slot(),
            'value': _slot(),
            'critic': _slot(),
        },
    }


def _cube_pair():
    """Create active and inactive relation examples in canonical scaled XYZ."""

    state = np.zeros((3, 3, 9), dtype=np.float32)
    goal = np.zeros_like(state)

    # Current support is active for sample 0: cube 1 is 0.04 above cube 0.
    state[0, 0, :3] = (0.0, 0.0, 0.0)
    state[0, 1, :3] = (0.0, 0.0, 0.4)
    state[0, 2, :3] = (2.0, 2.0, 2.0)
    # Goal support is active for sample 2, with state/goal positions separated
    # enough to avoid accidental goal-conflict edges.
    state[2, :, :3] = ((10.0, 10.0, 10.0), (20.0, 20.0, 20.0), (30.0, 30.0, 30.0))
    goal[2, 0, :3] = (0.0, 0.0, 0.0)
    goal[2, 1, :3] = (0.0, 0.0, 0.4)
    goal[2, 2, :3] = (2.0, 2.0, 2.0)

    # Sample 1 is a genuine inactive negative control: all pairwise distances
    # exceed the frozen support/conflict thresholds.
    state[1, :, :3] = ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0))
    goal[1, :, :3] = ((10.0, 10.0, 10.0), (20.0, 20.0, 20.0), (30.0, 30.0, 30.0))

    observations = np.concatenate(
        [np.zeros((3, 19), dtype=np.float32), state.reshape(3, -1)], axis=-1,
    )
    goals = np.concatenate(
        [np.zeros((3, 19), dtype=np.float32), goal.reshape(3, -1)], axis=-1,
    )
    return jnp.asarray(observations), jnp.asarray(goals)


def _as_numpy_trace(trace):
    return {key: np.asarray(value) if value is not None else None for key, value in trace.items()}


class M20ARelationUtilizationIntegrationTest(unittest.TestCase):
    def test_trace_interventions_ensemble_and_gradients_are_read_only(self):
        observations, goals = _cube_pair()
        actions = jnp.zeros((3, 3), dtype=jnp.float32)
        config = _config()
        agent = GCIQLAgent.create(23, observations, actions, config)
        batch = {
            'observations': observations,
            'next_observations': observations,
            'actions': actions,
            'actor_goals': goals,
            'value_goals': goals,
            'rewards': jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float32),
            'masks': jnp.ones((3,), dtype=jnp.float32),
        }

        before = {
            'params': _pytree_fingerprint(agent.network.params),
            'model_state': _pytree_fingerprint(agent.network.model_state),
            'opt_state': _pytree_fingerprint(agent.network.opt_state),
            'rng': _pytree_fingerprint(agent.rng),
            'step': int(agent.network.step),
        }

        actor_trace = _as_numpy_trace(agent.network(
            observations, goals, name='actor', method='relation_utilization_trace',
        ))
        actor_normal = np.asarray(agent.network.select('actor')(
            observations, goals, temperature=0.0,
        ).mode())
        np.testing.assert_allclose(actor_trace['action_means'], actor_normal, atol=1e-6, rtol=1e-6)
        self.assertEqual(actor_trace['relations_original'].shape, (3, 3, 3, 3))

        actor_zero = _as_numpy_trace(agent.network(
            observations,
            goals,
            name='actor',
            method='relation_utilization_trace',
            relation_override=jnp.zeros_like(actor_trace['relations_original']),
        ))
        np.testing.assert_array_equal(
            actor_zero['tokens_post_relation'], actor_zero['tokens_pre_relation'],
        )
        np.testing.assert_array_equal(actor_zero['relations_used'], np.zeros_like(actor_trace['relations_original']))

        actor_drop_goal = _as_numpy_trace(agent.network(
            observations,
            goals,
            name='actor',
            method='relation_utilization_trace',
            relation_channel_mask=jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32),
        ))
        self.assertEqual(actor_drop_goal['relations_used'][..., 1].sum(), 0.0)

        value_trace = _as_numpy_trace(agent.network(
            observations, goals, name='value', method='relation_utilization_trace',
        ))
        value_normal = np.asarray(agent.network.select('value')(observations, goals))
        np.testing.assert_allclose(value_trace['values'], value_normal, atol=1e-6, rtol=1e-6)

        critic_trace = _as_numpy_trace(agent.network(
            observations,
            goals,
            actions,
            name='critic',
            method='relation_utilization_trace',
        ))
        critic_normal = np.asarray(agent.network.select('critic')(observations, goals, actions))
        np.testing.assert_allclose(critic_trace['values'], critic_normal, atol=1e-6, rtol=1e-6)
        self.assertEqual(critic_trace['relations_original'].shape, (2, 3, 3, 3, 3))
        critic_zero = _as_numpy_trace(agent.network(
            observations,
            goals,
            actions,
            name='critic',
            method='relation_utilization_trace',
            # A vmap'd critic receives one relation tensor per ensemble member;
            # the same zero control is broadcast to both members.
            relation_override=jnp.zeros_like(critic_trace['relations_original'][0]),
        ))
        np.testing.assert_array_equal(
            critic_zero['tokens_post_relation'], critic_zero['tokens_pre_relation'],
        )
        critic_drop_goal = _as_numpy_trace(agent.network(
            observations,
            goals,
            actions,
            name='critic',
            method='relation_utilization_trace',
            relation_channel_mask=jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32),
        ))
        self.assertEqual(critic_drop_goal['relations_used'][..., 1].sum(), 0.0)

        # The inactive sample must remain unchanged under Correct-versus-Zero
        # and channel ablation; this is a negative control on intervention
        # routing, independent of whether random initialized weights amplify an
        # active relation on this synthetic model.
        inactive = np.asarray(actor_trace['relations_original']).sum(axis=(1, 2, 3)) == 0
        self.assertTrue(np.array_equal(inactive, np.asarray([False, True, False])))
        np.testing.assert_array_equal(
            actor_trace['action_means'][inactive], actor_zero['action_means'][inactive],
        )
        np.testing.assert_array_equal(
            actor_trace['tokens_post_core'][inactive], actor_zero['tokens_post_core'][inactive],
        )
        np.testing.assert_array_equal(
            actor_trace['action_means'][inactive], actor_drop_goal['action_means'][inactive],
        )

        gradient_result = _gradient_metrics(agent, batch)
        self.assertEqual(set(gradient_result['metrics']), {'actor', 'value', 'critic'})
        self.assertTrue(all(
            row['finite'] and row['relation_param_count'] > 0
            for row in gradient_result['metrics'].values()
        ))
        self.assertEqual(gradient_result['state_before'], gradient_result['state_after'])

        after = {
            'params': _pytree_fingerprint(agent.network.params),
            'model_state': _pytree_fingerprint(agent.network.model_state),
            'opt_state': _pytree_fingerprint(agent.network.opt_state),
            'rng': _pytree_fingerprint(agent.rng),
            'step': int(agent.network.step),
        }
        self.assertEqual(before, after)

    def test_override_shape_and_dtype_are_strict(self):
        observations, goals = _cube_pair()
        agent = GCIQLAgent.create(29, observations, jnp.zeros((3, 3), dtype=jnp.float32), _config())
        with self.assertRaises(ValueError):
            agent.network(
                observations,
                goals,
                name='actor',
                method='relation_utilization_trace',
                relation_override=jnp.zeros((3, 3, 3), dtype=jnp.float32),
            )
        with self.assertRaises(TypeError):
            agent.network(
                observations,
                goals,
                name='actor',
                method='relation_utilization_trace',
                relation_override=np.zeros((3, 3, 3, 3), dtype=np.float64),
            )


if __name__ == '__main__':
    unittest.main()
