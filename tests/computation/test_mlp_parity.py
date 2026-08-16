"""Parity tests for the HIQL actor and value computation-slot migrations.

The tests deliberately compare semantic parameter groups instead of relying on
Flax's generated ``Dense_0`` paths.  The computation wrapper changes the
scope of the actor body, while the actor readouts and all non-actor modules
remain semantically unchanged.
"""

from collections.abc import Mapping
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.hiql import HIQLAgent
from impls.computation.accounting import count_parameters, make_parameter_report
from impls.computation.factory import ComputationSpec, make_computation_core, resolve_slot_spec
from impls.computation.interfaces import ComputationOutput
from impls.computation.primitives.mlp import MLP
from impls.networks.common import GCActor
from tests.reference.ogbench_mlp import ReferenceMLP


ATOL = 1e-6
ACTOR_NAMES = ('low_actor', 'high_actor')
VALUE_NAMES = ('value', 'target_value')


def _path_get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _leaf_items(tree, prefix=()):
    if isinstance(tree, Mapping):
        for key in sorted(tree):
            yield from _leaf_items(tree[key], prefix + (str(key),))
    else:
        yield prefix, tree


def _actor_body_path(actor_params, computation_enabled):
    """Return the body path implied by the public actor/computation contract."""

    if not computation_enabled:
        return ('actor_net',)

    actor_net = actor_params.get('actor_net')
    if not isinstance(actor_net, Mapping) or set(actor_net) != {'topology'}:
        raise AssertionError(f'Unexpected computation actor wrapper tree: {actor_net}')
    topology = actor_net['topology']
    if not isinstance(topology, Mapping) or set(topology) != {'primitive'}:
        raise AssertionError(f'Unexpected computation topology tree: {topology}')
    return ('actor_net', 'topology', 'primitive')


def _value_body_path(value_params, computation_enabled):
    """Return the value body path implied by the computation wrapper."""

    if not computation_enabled:
        return ('value_net',)
    value_net = value_params.get('value_net')
    if not isinstance(value_net, Mapping) or set(value_net) != {'core'}:
        raise AssertionError(f'Unexpected computation value tree: {value_net}')
    core = value_net['core']
    if not isinstance(core, Mapping) or set(core) != {'topology'}:
        raise AssertionError(f'Unexpected value core tree: {core}')
    topology = core['topology']
    if not isinstance(topology, Mapping) or set(topology) != {'primitive'}:
        raise AssertionError(f'Unexpected value topology tree: {topology}')
    return ('value_net', 'core', 'topology', 'primitive')


def _legacy_value_readout_key(value_params):
    dense_keys = [key for key in value_params['value_net'] if str(key).startswith('Dense_')]
    if not dense_keys:
        raise AssertionError(f'No legacy value Dense readout found: {value_params}')
    return sorted(dense_keys)[-1]


def _semantic_value_leaves(value_params, computation_enabled):
    """Flatten value params as body and scalar-readout semantic groups."""

    result = {}
    if computation_enabled:
        body = _path_get(value_params, _value_body_path(value_params, True))
        readout_items = _leaf_items(value_params['value_readout'])
    else:
        readout_key = _legacy_value_readout_key(value_params)
        body = {key: value for key, value in value_params['value_net'].items() if key != readout_key}
        readout_items = _leaf_items(value_params['value_net'][readout_key])
    for relative_path, leaf in _leaf_items(body):
        result[('body',) + relative_path] = leaf

    for relative_path, leaf in readout_items:
        result[('readout',) + relative_path] = leaf
    return result


def _semantic_actor_leaves(params, computation_slots):
    """Flatten params under stable semantic names.

Actor body leaves are exposed as ``(<actor>, 'body', ...)`` regardless of
whether the body is wrapped.  Readouts retain their direct module names.
    """

    result = {}
    for root_key, root_value in params.items():
        if not root_key.startswith('modules_') or root_key[8:] not in ACTOR_NAMES:
            for relative_path, leaf in _leaf_items(root_value):
                result[(str(root_key),) + relative_path] = leaf
            continue

        actor_name = root_key[8:]
        body_path = _actor_body_path(root_value, computation_slots.get(actor_name, False))
        body = _path_get(root_value, body_path)
        for relative_path, leaf in _leaf_items(body):
            result[(actor_name, 'body') + relative_path] = leaf

        for module_key, module_value in root_value.items():
            if module_key == 'actor_net':
                continue
            for relative_path, leaf in _leaf_items(module_value):
                result[(actor_name, str(module_key)) + relative_path] = leaf
    return result


def _graft_semantic_params(old_params, new_params, computation_slots, value_enabled=False):
    """Copy legacy parameters into computation-wrapped semantic subtrees."""

    old_root = flax.core.unfreeze(old_params)
    new_root = flax.core.unfreeze(new_params)
    special_roots = {f'modules_{name}' for name in ACTOR_NAMES + VALUE_NAMES}

    for root_key, value in old_root.items():
        if root_key not in special_roots:
            new_root[root_key] = value

    for actor_name in ACTOR_NAMES:
        root_key = f'modules_{actor_name}'
        old_actor = old_root[root_key]
        new_actor = new_root[root_key]
        if computation_slots.get(actor_name, False):
            body_path = _actor_body_path(new_actor, True)
            _path_set(new_actor, body_path, old_actor['actor_net'])
        else:
            new_actor['actor_net'] = old_actor['actor_net']
        for module_key, value in old_actor.items():
            if module_key != 'actor_net':
                new_actor[module_key] = value
        new_root[root_key] = new_actor

    for value_name in VALUE_NAMES:
        root_key = f'modules_{value_name}'
        if not value_enabled:
            new_root[root_key] = old_root[root_key]
            continue
        old_value = old_root[root_key]
        new_value = new_root[root_key]
        readout_key = _legacy_value_readout_key(old_value)
        old_body = {key: value for key, value in old_value['value_net'].items() if key != readout_key}
        _path_set(new_value, _value_body_path(new_value, True), old_body)
        new_value['value_readout'] = old_value['value_net'][readout_key]
        new_root[root_key] = new_value

    # HIQL's create path stores the parameter tree as a mutable dict after
    # installing target_value.  Keep that container type here so the existing
    # Optax state remains the same pytree as the grafted parameters.
    return new_root


def _graft_semantic_actor_params(old_params, new_params, computation_slots):
    return _graft_semantic_params(old_params, new_params, computation_slots, value_enabled=False)


def _path_set(tree, path, value):
    cursor = tree
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _assert_semantic_allclose(testcase, old_tree, new_tree, computation_slots, context):
    old_items = _semantic_actor_leaves(old_tree, {'low_actor': False, 'high_actor': False})
    new_items = _semantic_actor_leaves(new_tree, computation_slots)
    testcase.assertEqual(set(old_items), set(new_items), msg=f'{context}: semantic parameter labels differ')

    max_error = 0.0
    for label in sorted(old_items):
        old_value = old_items[label]
        new_value = new_items[label]
        if old_value is None or new_value is None:
            testcase.assertIsNone(old_value, msg=f'{context}: {label} expected None')
            testcase.assertIsNone(new_value, msg=f'{context}: {label} expected None')
            continue
        old_array = np.asarray(old_value)
        new_array = np.asarray(new_value)
        testcase.assertEqual(old_array.shape, new_array.shape, msg=f'{context}: {label} shape differs')
        error = float(np.max(np.abs(old_array - new_array))) if old_array.size else 0.0
        max_error = max(max_error, error)
        testcase.assertTrue(
            np.allclose(old_array, new_array, rtol=0.0, atol=ATOL),
            msg=f'{context}: {label} max_abs_error={error}',
        )
    return max_error


def _semantic_params(params, actor_slots, value_enabled):
    """Flatten all agent params under actor/value semantic labels."""

    result = _semantic_actor_leaves(params, actor_slots)
    for value_name in VALUE_NAMES:
        root_key = f'modules_{value_name}'
        for label in [label for label in result if label[0] == root_key]:
            del result[label]
        for relative_path, leaf in _semantic_value_leaves(params[root_key], value_enabled).items():
            result[(root_key,) + relative_path] = leaf
    return result


def _assert_full_semantic_allclose(testcase, old_tree, new_tree, actor_slots, value_enabled, context):
    old_items = _semantic_params(old_tree, {'low_actor': False, 'high_actor': False}, False)
    new_items = _semantic_params(new_tree, actor_slots, value_enabled)
    testcase.assertEqual(set(old_items), set(new_items), msg=f'{context}: semantic parameter labels differ')
    max_error = 0.0
    for label in sorted(old_items):
        old_value = old_items[label]
        new_value = new_items[label]
        if old_value is None or new_value is None:
            testcase.assertIsNone(old_value, msg=f'{context}: {label} expected None')
            testcase.assertIsNone(new_value, msg=f'{context}: {label} expected None')
            continue
        old_array = np.asarray(old_value)
        new_array = np.asarray(new_value)
        testcase.assertEqual(old_array.shape, new_array.shape, msg=f'{context}: {label} shape differs')
        error = float(np.max(np.abs(old_array - new_array))) if old_array.size else 0.0
        max_error = max(max_error, error)
        testcase.assertTrue(
            np.allclose(old_array, new_array, rtol=0.0, atol=ATOL),
            msg=f'{context}: {label} max_abs_error={error}',
        )
    return max_error


def _assert_info_allclose(testcase, old_info, new_info, context, skip_keys=()):
    old_keys = set(old_info) - set(skip_keys)
    new_keys = set(new_info) - set(skip_keys)
    testcase.assertEqual(old_keys, new_keys, msg=f'{context}: info keys differ')
    max_error = 0.0
    for key in sorted(old_keys):
        old_value = np.asarray(old_info[key])
        new_value = np.asarray(new_info[key])
        error = float(np.max(np.abs(old_value - new_value))) if old_value.size else 0.0
        max_error = max(max_error, error)
        testcase.assertTrue(
            np.allclose(old_value, new_value, rtol=0.0, atol=ATOL),
            msg=f'{context}: {key} max_abs_error={error}',
        )
    return max_error


def _assert_actor_gradient_parity(testcase, old_grads, new_grads, computation_slots, actor_name, context):
    _assert_semantic_allclose(testcase, old_grads, new_grads, computation_slots, context)
    old_items = _semantic_actor_leaves(old_grads, {'low_actor': False, 'high_actor': False})
    new_items = _semantic_actor_leaves(new_grads, computation_slots)
    actor_labels = [label for label in old_items if label[0] == actor_name]
    testcase.assertTrue(actor_labels, msg=f'{context}: no trainable {actor_name} leaves found')
    for label, value in old_items.items():
        if label[0] == actor_name:
            continue
        if value is None:
            testcase.assertIsNone(new_items[label], msg=f'{context}: {label} unexpectedly has a gradient')
        else:
            max_error = float(np.max(np.abs(np.asarray(value)))) if np.asarray(value).size else 0.0
            testcase.assertLessEqual(max_error, ATOL, msg=f'{context}: non-{actor_name} gradient at {label} is {max_error}')


def _small_config(low_enabled=False, high_enabled=False, value_enabled=False):
    slot = lambda enabled: dict(enabled=enabled, primitive='mlp', topology='feedforward', credit='direct')
    return dict(
        agent_name='hiql', lr=3e-4, batch_size=4,
        actor_hidden_dims=(6, 6), value_hidden_dims=(6, 6), layer_norm=True,
        discount=0.99, tau=0.005, expectile=0.7, low_alpha=3.0, high_alpha=3.0,
        subgoal_steps=2, rep_dim=3, low_actor_rep_grad=False, const_std=True,
        discrete=False, encoder=None,
        compute={
            'low_actor': slot(low_enabled),
            'high_actor': slot(high_enabled),
            'value': slot(value_enabled),
        },
    )


def _make_agents(low_enabled=False, high_enabled=False, value_enabled=False, seed=10):
    ex_observations = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 5.0
    ex_actions = jnp.arange(4, dtype=jnp.float32).reshape(2, 2) / 7.0
    old_agent = HIQLAgent.create(seed, ex_observations, ex_actions, _small_config(False, False))
    new_agent = HIQLAgent.create(
        seed, ex_observations, ex_actions, _small_config(low_enabled, high_enabled, value_enabled),
    )
    slots = {'low_actor': low_enabled, 'high_actor': high_enabled}
    new_params = _graft_semantic_params(
        old_agent.network.params, new_agent.network.params, slots, value_enabled=value_enabled,
    )
    new_agent = new_agent.replace(network=new_agent.network.replace(params=new_params))
    return old_agent, new_agent, slots


def _synthetic_batch(step=0, batch_size=3):
    offset = jnp.float32(step) * 0.03125
    base = jnp.arange(batch_size * 4, dtype=jnp.float32).reshape(batch_size, 4) / 17.0 + offset
    next_observations = jnp.flip(base, axis=0) + 0.07
    low_goals = base * 0.5 + 0.11
    high_goals = base * 0.25 - 0.13
    high_targets = jnp.roll(base, 1, axis=0) * 0.75 + 0.19
    actions = jnp.arange(batch_size * 2, dtype=jnp.float32).reshape(batch_size, 2) / 13.0 - 0.2
    return {
        'observations': base,
        'next_observations': next_observations,
        'low_actor_goals': low_goals,
        'actions': actions,
        'high_actor_goals': high_goals,
        'high_actor_targets': high_targets,
        'value_goals': base * 0.33 + 0.05,
        'rewards': jnp.linspace(-0.2, 0.3, batch_size, dtype=jnp.float32) + offset,
        'masks': jnp.asarray((jnp.arange(batch_size) % 2) == 0, dtype=jnp.float32),
    }


class MLPParityTest(unittest.TestCase):
    def test_primitive_matches_source_reference_with_fixed_parameters(self):
        x = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 7.0
        reference = ReferenceMLP((7, 4), activate_final=True, layer_norm=True)
        computation = MLP((7, 4), activate_final=True, layer_norm=True)
        params = reference.init(jax.random.PRNGKey(0), x)['params']
        expected = reference.apply({'params': params}, x)
        actual = computation.apply({'params': params}, x)
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))

    def test_feedforward_core_has_no_fake_state_and_supports_sequence_shape(self):
        x = jnp.ones((2, 3, 5), dtype=jnp.float32)
        core = make_computation_core(ComputationSpec(), hidden_dims=(7, 5), activate_final=True)
        variables = core.init(jax.random.PRNGKey(1), x)
        output = core.apply(variables, x)
        self.assertIsInstance(output, ComputationOutput)
        self.assertIsNone(output.state)
        self.assertEqual(output.representation.shape, (2, 3, 5))

    def test_policy_distribution_matches_after_body_parameter_graft(self):
        observations = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 10.0
        goals = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 9.0
        old = GCActor((7, 6), action_dim=2, const_std=True)
        new = GCActor((7, 6), action_dim=2, const_std=True, computation_spec=ComputationSpec())
        old_params = old.init(jax.random.PRNGKey(2), observations, goals)['params']
        new_params = new.init(jax.random.PRNGKey(3), observations, goals)['params']
        self.assertEqual(_actor_body_path(old_params, False), ('actor_net',))
        self.assertEqual(_actor_body_path(new_params, True), ('actor_net', 'topology', 'primitive'))
        _path_set(new_params := flax.core.unfreeze(new_params), _actor_body_path(new_params, True), old_params['actor_net'])
        for key in ('mean_net', 'log_stds'):
            if key in old_params:
                new_params[key] = old_params[key]
        new_params = flax.core.freeze(new_params)

        old_dist = old.apply({'params': old_params}, observations, goals)
        new_dist = new.apply({'params': new_params}, observations, goals)
        np.testing.assert_array_equal(np.asarray(old_dist.mode()), np.asarray(new_dist.mode()))
        np.testing.assert_array_equal(np.asarray(old_dist.scale_diag), np.asarray(new_dist.scale_diag))
        actions = jnp.zeros((4, 2), dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(old_dist.log_prob(actions)), np.asarray(new_dist.log_prob(actions)))

    def test_computation_wrapper_adds_no_trainable_parameters(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True)
        old_items = _semantic_actor_leaves(old_agent.network.params, {'low_actor': False, 'high_actor': False})
        new_items = _semantic_actor_leaves(new_agent.network.params, slots)
        self.assertEqual(set(old_items), set(new_items))
        for label in old_items:
            self.assertEqual(np.asarray(old_items[label]).shape, np.asarray(new_items[label]).shape, msg=str(label))


class SlotSpecResolutionTest(unittest.TestCase):
    def test_disabled_or_missing_slot_resolves_to_none(self):
        self.assertIsNone(resolve_slot_spec({}, 'low_actor'))
        self.assertIsNone(resolve_slot_spec({'compute': {'low_actor': {'enabled': False}}}, 'low_actor'))

    def test_enabled_slot_resolves_to_computation_spec(self):
        spec = resolve_slot_spec({'compute': {'low_actor': {
            'enabled': True, 'primitive': 'mlp', 'topology': 'feedforward', 'credit': 'direct',
        }}}, 'low_actor')
        self.assertEqual(spec, ComputationSpec('mlp', 'feedforward', 'direct'))


class HIQLActorParityTest(unittest.TestCase):
    def test_low_actor_loss_info_and_full_trainable_subtree_gradient_match(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True)
        batch = _synthetic_batch()
        old_loss, old_info = old_agent.low_actor_loss(batch, old_agent.network.params)
        new_loss, new_info = new_agent.low_actor_loss(batch, new_agent.network.params)
        np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
        _assert_info_allclose(self, old_info, new_info, 'low actor loss info')

        old_grads = jax.grad(lambda params: old_agent.low_actor_loss(batch, params)[0])(old_agent.network.params)
        new_grads = jax.grad(lambda params: new_agent.low_actor_loss(batch, params)[0])(new_agent.network.params)
        _assert_actor_gradient_parity(self, old_grads, new_grads, slots, 'low_actor', 'low actor full gradient')

    def test_high_actor_loss_distribution_info_and_full_trainable_subtree_gradient_match(self):
        old_agent, new_agent, slots = _make_agents(high_enabled=True)
        batch = _synthetic_batch()
        old_params = old_agent.network.params
        new_params = new_agent.network.params

        old_dist = old_agent.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=old_params,
        )
        new_dist = new_agent.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=new_params,
        )
        old_target = old_agent.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1), params=old_params,
        )
        new_target = new_agent.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1), params=new_params,
        )
        for name, old_value, new_value in (
            ('mode', old_dist.mode(), new_dist.mode()),
            ('scale_diag', old_dist.scale_diag, new_dist.scale_diag),
            ('target', old_target, new_target),
            ('log_prob', old_dist.log_prob(old_target), new_dist.log_prob(new_target)),
        ):
            np.testing.assert_allclose(np.asarray(old_value), np.asarray(new_value), rtol=0, atol=ATOL, err_msg=name)

        old_loss, old_info = old_agent.high_actor_loss(batch, old_params)
        new_loss, new_info = new_agent.high_actor_loss(batch, new_params)
        np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
        _assert_info_allclose(self, old_info, new_info, 'high actor loss info')

        old_grads = jax.grad(lambda params: old_agent.high_actor_loss(batch, params)[0])(old_params)
        new_grads = jax.grad(lambda params: new_agent.high_actor_loss(batch, params)[0])(new_params)
        _assert_actor_gradient_parity(self, old_grads, new_grads, slots, 'high_actor', 'high actor full gradient')

    def test_integrated_total_loss_matches_with_legacy_value_and_two_computation_actors(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True)
        batch = _synthetic_batch()
        old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
        new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
        np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
        _assert_info_allclose(self, old_info, new_info, 'integrated total loss')
        for prefix in ('value/', 'low_actor/', 'high_actor/'):
            self.assertTrue(any(key.startswith(prefix) for key in old_info), msg=prefix)

    def test_one_real_optimizer_update_matches_semantically(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True)
        batch = _synthetic_batch()
        old_initial = old_agent.network.params
        new_initial = new_agent.network.params
        old_agent, old_info = old_agent.update(batch)
        new_agent, new_info = new_agent.update(batch)
        _assert_info_allclose(self, old_info, new_info, 'one-step update info')
        _assert_semantic_allclose(self, old_agent.network.params, new_agent.network.params, slots, 'one-step updated params')

        old_before = _semantic_actor_leaves(old_initial, {'low_actor': False, 'high_actor': False})
        old_after = _semantic_actor_leaves(old_agent.network.params, {'low_actor': False, 'high_actor': False})
        new_before = _semantic_actor_leaves(new_initial, slots)
        new_after = _semantic_actor_leaves(new_agent.network.params, slots)
        for label in old_before:
            if label[0] not in ACTOR_NAMES:
                continue
            old_delta = np.asarray(old_after[label]) - np.asarray(old_before[label])
            new_delta = np.asarray(new_after[label]) - np.asarray(new_before[label])
            np.testing.assert_allclose(old_delta, new_delta, rtol=0, atol=ATOL, err_msg=str(label))

    def test_deterministic_n_step_regression_for_ten_and_twenty_updates(self):
        for num_steps in (10, 20):
            with self.subTest(num_steps=num_steps):
                old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True, seed=20)
                old_initial = old_agent.network.params
                new_initial = new_agent.network.params
                for step in range(num_steps):
                    batch = _synthetic_batch(step=step)
                    old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
                    new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
                    np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
                    _assert_info_allclose(self, old_info, new_info, f'N={num_steps} step={step} loss info')
                    old_agent, old_update_info = old_agent.update(batch)
                    new_agent, new_update_info = new_agent.update(batch)
                    _assert_info_allclose(self, old_update_info, new_update_info, f'N={num_steps} step={step} update info')
                    _assert_semantic_allclose(
                        self, old_agent.network.params, new_agent.network.params, slots,
                        f'N={num_steps} step={step} params',
                    )

                old_before = _semantic_actor_leaves(old_initial, {'low_actor': False, 'high_actor': False})
                old_after = _semantic_actor_leaves(old_agent.network.params, {'low_actor': False, 'high_actor': False})
                new_before = _semantic_actor_leaves(new_initial, slots)
                new_after = _semantic_actor_leaves(new_agent.network.params, slots)
                for label in old_before:
                    if label[0] not in ACTOR_NAMES:
                        continue
                    old_delta = np.asarray(old_after[label]) - np.asarray(old_before[label])
                    new_delta = np.asarray(new_after[label]) - np.asarray(new_before[label])
                    np.testing.assert_allclose(old_delta, new_delta, rtol=0, atol=ATOL, err_msg=str(label))


class HIQLValueParityTest(unittest.TestCase):
    def _assert_value_module_allclose(self, old_tree, new_tree, value_enabled, context):
        old_items = _semantic_value_leaves(old_tree, False)
        new_items = _semantic_value_leaves(new_tree, value_enabled)
        self.assertEqual(set(old_items), set(new_items), msg=f'{context}: semantic labels differ')
        max_error = 0.0
        for label in sorted(old_items):
            old_value = np.asarray(old_items[label])
            new_value = np.asarray(new_items[label])
            error = float(np.max(np.abs(old_value - new_value))) if old_value.size else 0.0
            max_error = max(max_error, error)
            np.testing.assert_allclose(old_value, new_value, rtol=0, atol=ATOL, err_msg=f'{context}: {label}')
        return max_error

    def _assert_zero_tree(self, tree, context):
        max_error = 0.0
        for path, value in _leaf_items(tree):
            if value is None:
                continue
            error = float(np.max(np.abs(np.asarray(value)))) if np.asarray(value).size else 0.0
            max_error = max(max_error, error)
            self.assertLessEqual(error, ATOL, msg=f'{context}: {path} max_abs={error}')
        return max_error

    def test_value_forward_matches_both_ensemble_members_and_parameter_count(self):
        old_agent, new_agent, slots = _make_agents(value_enabled=True)
        batch = _synthetic_batch()
        old_value = old_agent.network.select('value')(batch['observations'], batch['value_goals'])
        new_value = new_agent.network.select('value')(batch['observations'], batch['value_goals'])
        old_target = old_agent.network.select('target_value')(batch['observations'], batch['value_goals'])
        new_target = new_agent.network.select('target_value')(batch['observations'], batch['value_goals'])
        for member in range(2):
            np.testing.assert_allclose(np.asarray(old_value[member]), np.asarray(new_value[member]), rtol=0, atol=ATOL)
            np.testing.assert_allclose(np.asarray(old_target[member]), np.asarray(new_target[member]), rtol=0, atol=ATOL)
        np.testing.assert_allclose(np.asarray(jnp.minimum(*old_value)), np.asarray(jnp.minimum(*new_value)), rtol=0, atol=ATOL)
        np.testing.assert_allclose(np.asarray(jnp.mean(old_value, axis=0)), np.asarray(jnp.mean(new_value, axis=0)), rtol=0, atol=ATOL)

        old_value_params = old_agent.network.params['modules_value']
        new_value_params = new_agent.network.params['modules_value']
        self._assert_value_module_allclose(old_value_params, new_value_params, True, 'value forward params')
        self.assertEqual(count_parameters(old_value_params), count_parameters(new_value_params))
        new_body_kernel = _path_get(new_value_params, _value_body_path(new_value_params, True))['Dense_0']['kernel']
        self.assertEqual(new_body_kernel.shape[0], 2)
        self.assertFalse(np.array_equal(np.asarray(new_body_kernel[0]), np.asarray(new_body_kernel[1])))
        new_readout_kernel = new_value_params['value_readout']['kernel']
        self.assertEqual(new_readout_kernel.shape[0], 2)
        self.assertFalse(np.array_equal(np.asarray(new_readout_kernel[0]), np.asarray(new_readout_kernel[1])))
        self.assertEqual(
            count_parameters(old_agent.network.params['modules_target_value']),
            count_parameters(new_agent.network.params['modules_target_value']),
        )

    def test_value_loss_and_full_mapped_gradient_match_with_target_zero_gradient(self):
        old_agent, new_agent, slots = _make_agents(value_enabled=True)
        batch = _synthetic_batch()
        old_loss, old_info = old_agent.value_loss(batch, old_agent.network.params)
        new_loss, new_info = new_agent.value_loss(batch, new_agent.network.params)
        np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
        _assert_info_allclose(self, old_info, new_info, 'value loss info')

        old_grads = jax.grad(lambda params: old_agent.value_loss(batch, params)[0])(old_agent.network.params)
        new_grads = jax.grad(lambda params: new_agent.value_loss(batch, params)[0])(new_agent.network.params)
        max_error = _assert_full_semantic_allclose(
            self, old_grads, new_grads, slots, True, 'value full gradient',
        )
        self.assertLessEqual(max_error, ATOL)
        self._assert_zero_tree(old_grads['modules_target_value'], 'legacy target value gradient')
        self._assert_zero_tree(new_grads['modules_target_value'], 'computation target value gradient')
        self._assert_zero_tree(old_grads['modules_high_actor'], 'legacy high actor value-loss gradient')
        self._assert_zero_tree(new_grads['modules_high_actor'], 'computation high actor value-loss gradient')
        self._assert_zero_tree(old_grads['modules_low_actor'], 'legacy low actor value-loss gradient')
        self._assert_zero_tree(new_grads['modules_low_actor'], 'computation low actor value-loss gradient')

    def test_polyak_target_update_matches_for_mapped_online_and_target_trees(self):
        old_agent, new_agent, slots = _make_agents(value_enabled=True)
        old_params = flax.core.unfreeze(old_agent.network.params)
        for root_key in ('modules_value', 'modules_target_value'):
            old_params[root_key] = jax.tree_util.tree_map(
                lambda value: value + (0.17 if root_key == 'modules_value' else -0.23),
                old_params[root_key],
            )
        new_params = _graft_semantic_params(old_params, new_agent.network.params, slots, value_enabled=True)
        old_agent = old_agent.replace(network=old_agent.network.replace(params=old_params))
        new_agent = new_agent.replace(network=new_agent.network.replace(params=new_params))

        old_network = old_agent.network.replace(params=flax.core.unfreeze(old_agent.network.params))
        new_network = new_agent.network.replace(params=flax.core.unfreeze(new_agent.network.params))
        old_agent.target_update(old_network, 'value')
        new_agent.target_update(new_network, 'value')
        max_error = self._assert_value_module_allclose(
            old_network.params['modules_target_value'], new_network.params['modules_target_value'], True,
            'Polyak target update',
        )
        self.assertLessEqual(max_error, ATOL)
        self.assertEqual(
            set(new_network.params['modules_target_value']),
            set(new_network.params['modules_value']),
        )

    def test_value_one_step_update_matches_online_target_and_legacy_actor_behavior(self):
        old_agent, new_agent, slots = _make_agents(value_enabled=True)
        batch = _synthetic_batch()
        old_initial = old_agent.network.params
        new_initial = new_agent.network.params
        old_agent, old_info = old_agent.update(batch)
        new_agent, new_info = new_agent.update(batch)
        _assert_info_allclose(self, old_info, new_info, 'value one-step update info', skip_keys=('grad/norm',))
        _assert_full_semantic_allclose(self, old_agent.network.params, new_agent.network.params, slots, True, 'value one-step params')

        old_before = _semantic_params(old_initial, {'low_actor': False, 'high_actor': False}, False)
        old_after = _semantic_params(old_agent.network.params, {'low_actor': False, 'high_actor': False}, False)
        new_before = _semantic_params(new_initial, slots, True)
        new_after = _semantic_params(new_agent.network.params, slots, True)
        for label in old_before:
            if label[0] not in ('modules_value', 'modules_target_value'):
                continue
            np.testing.assert_allclose(
                np.asarray(old_after[label]) - np.asarray(old_before[label]),
                np.asarray(new_after[label]) - np.asarray(new_before[label]),
                rtol=0, atol=ATOL, err_msg=str(label),
            )

    def test_value_n_step_regression_n10_and_n20(self):
        for num_steps in (10, 20):
            with self.subTest(num_steps=num_steps):
                old_agent, new_agent, slots = _make_agents(value_enabled=True, seed=30)
                for step in range(num_steps):
                    batch = _synthetic_batch(step=step)
                    old_loss, old_info = old_agent.value_loss(batch, old_agent.network.params)
                    new_loss, new_info = new_agent.value_loss(batch, new_agent.network.params)
                    np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
                    _assert_info_allclose(self, old_info, new_info, f'value N={num_steps} step={step} info')
                    old_agent, _ = old_agent.update(batch)
                    new_agent, _ = new_agent.update(batch)
                    _assert_full_semantic_allclose(
                        self, old_agent.network.params, new_agent.network.params, slots, True,
                        f'value N={num_steps} step={step} params',
                    )


class HIQLFullParityTest(unittest.TestCase):
    def test_three_slot_total_loss_and_full_gradient_match(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True, value_enabled=True)
        batch = _synthetic_batch()
        old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
        new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
        np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
        _assert_info_allclose(self, old_info, new_info, 'full total loss info')

        old_grads = jax.grad(lambda params: old_agent.total_loss(batch, params)[0])(old_agent.network.params)
        new_grads = jax.grad(lambda params: new_agent.total_loss(batch, params)[0])(new_agent.network.params)
        max_error = _assert_full_semantic_allclose(
            self, old_grads, new_grads, slots, True, 'full HIQL gradient',
        )
        self.assertLessEqual(max_error, ATOL)

    def test_three_slot_one_step_update_matches_all_mapped_deltas(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True, value_enabled=True)
        batch = _synthetic_batch()
        old_initial = old_agent.network.params
        new_initial = new_agent.network.params
        old_agent, old_info = old_agent.update(batch)
        new_agent, new_info = new_agent.update(batch)
        _assert_info_allclose(self, old_info, new_info, 'full one-step update info', skip_keys=('grad/norm',))
        _assert_full_semantic_allclose(self, old_agent.network.params, new_agent.network.params, slots, True, 'full one-step params')

        old_before = _semantic_params(old_initial, {'low_actor': False, 'high_actor': False}, False)
        old_after = _semantic_params(old_agent.network.params, {'low_actor': False, 'high_actor': False}, False)
        new_before = _semantic_params(new_initial, slots, True)
        new_after = _semantic_params(new_agent.network.params, slots, True)
        for label in old_before:
            np.testing.assert_allclose(
                np.asarray(old_after[label]) - np.asarray(old_before[label]),
                np.asarray(new_after[label]) - np.asarray(new_before[label]),
                rtol=0, atol=ATOL, err_msg=str(label),
            )

    def test_three_slot_n_step_regression_n10_and_n20(self):
        for num_steps in (10, 20):
            with self.subTest(num_steps=num_steps):
                old_agent, new_agent, slots = _make_agents(
                    low_enabled=True, high_enabled=True, value_enabled=True, seed=40,
                )
                for step in range(num_steps):
                    batch = _synthetic_batch(step=step)
                    old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
                    new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
                    np.testing.assert_allclose(np.asarray(old_loss), np.asarray(new_loss), rtol=0, atol=ATOL)
                    _assert_info_allclose(self, old_info, new_info, f'full N={num_steps} step={step} loss info')
                    old_agent, old_update_info = old_agent.update(batch)
                    new_agent, new_update_info = new_agent.update(batch)
                    _assert_info_allclose(
                        self, old_update_info, new_update_info,
                        f'full N={num_steps} step={step} update info', skip_keys=('grad/norm',),
                    )
                    _assert_full_semantic_allclose(
                        self, old_agent.network.params, new_agent.network.params, slots, True,
                        f'full N={num_steps} step={step} params',
                    )

    def test_three_slot_parameter_count_and_target_mirror(self):
        old_agent, new_agent, slots = _make_agents(low_enabled=True, high_enabled=True, value_enabled=True)
        slot_names = ('value', 'high_actor', 'low_actor')
        old_report = make_parameter_report(old_agent.network.params, slot_names=slot_names)
        new_report = make_parameter_report(new_agent.network.params, slot_names=slot_names)
        self.assertEqual(old_report.total, new_report.total)
        self.assertEqual(old_report.per_slot, new_report.per_slot)
        self.assertEqual(
            count_parameters(new_agent.network.params['modules_value']),
            count_parameters(new_agent.network.params['modules_target_value']),
        )
        self.assertIsNone(resolve_slot_spec(_small_config(value_enabled=True), 'target_value'))


class AccountingTest(unittest.TestCase):
    def test_reports_total_slot_and_core_parameters(self):
        core = make_computation_core(ComputationSpec(), hidden_dims=(7, 5), activate_final=True)
        params = core.init(jax.random.PRNGKey(4), jnp.ones((2, 3)))['params']
        core_count = count_parameters(params)
        report = make_parameter_report({'low_actor': params}, slot_names=('low_actor',), core_params={'low_actor': params})
        self.assertEqual(report.total, core_count)
        self.assertEqual(report.per_slot['low_actor'], core_count)
        self.assertEqual(report.per_core['low_actor'], core_count)


if __name__ == '__main__':
    unittest.main()
