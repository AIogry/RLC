"""Run the opt-in GPU N=1000 strict HIQL parity diagnostic.

This is deliberately separate from the daily N=20 CPU regression.  It does
not alter the model, computation implementation, dataset wrapper, or runtime
configuration; it only measures the existing semantic-parameter mapping over
real OGBench batches.
"""

import argparse
import copy
import json
import os
import time

import jax
import numpy as np

from impls.agents.hiql import HIQLAgent
from impls.utils.datasets import Dataset, HGCDataset
from impls.utils.env_utils import make_env_and_datasets, resolve_dataset_dir
from tests.computation.test_mlp_parity import (
    _graft_semantic_params,
    _semantic_params,
)


DATASET_NAME = 'antmaze-medium-navigate-v0'
REQUIRED_BATCH_KEYS = (
    'observations',
    'next_observations',
    'actions',
    'rewards',
    'masks',
    'value_goals',
    'high_actor_goals',
    'high_actor_targets',
    'low_actor_goals',
)
SLOTS = {'low_actor': True, 'high_actor': True, 'value': True}


def _runtime_config():
    """Use the same small real-parity configuration as the N=20 test."""
    from impls.agents.hiql import get_config

    config = get_config()
    config['batch_size'] = 8
    config['actor_hidden_dims'] = (6, 6)
    config['value_hidden_dims'] = (6, 6)
    config['rep_dim'] = 3
    config['subgoal_steps'] = 7
    config['p_aug'] = 0.0
    return config


def _set_slots(config, enabled):
    for name in ('low_actor', 'high_actor', 'value'):
        config['compute'][name]['enabled'] = enabled


def _copy_dataset(raw_dataset, seed):
    return Dataset.create(seed=seed, **dict(raw_dataset))


def _scalar(value):
    return float(np.asarray(value))


def _semantic_error(old_params, new_params):
    old_items = _semantic_params(old_params, {'low_actor': False, 'high_actor': False}, False)
    new_items = _semantic_params(new_params, SLOTS, True)
    if set(old_items) != set(new_items):
        missing = sorted(set(old_items) ^ set(new_items))
        raise AssertionError(f'semantic parameter labels differ: {missing[:5]}')
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


def _optimizer_state_error(old_state, new_state):
    """Compare Adam state by parameter semantics, not generated Flax paths."""
    old_scale, new_scale = old_state[0], new_state[0]
    count_error = float(np.max(np.abs(np.asarray(old_scale.count) - np.asarray(new_scale.count))))
    mu_error = _semantic_error(old_scale.mu, new_scale.mu)
    nu_error = _semantic_error(old_scale.nu, new_scale.nu)
    return max(count_error, mu_error, nu_error)


def _grouped_semantic_errors(old_params, new_params):
    old_items = _semantic_params(old_params, {'low_actor': False, 'high_actor': False}, False)
    new_items = _semantic_params(new_params, SLOTS, True)
    groups = {
        'online': lambda label: label[0] != 'modules_target_value',
        'target_value': lambda label: label[0] == 'modules_target_value',
    }
    result = {}
    for group, predicate in groups.items():
        errors = []
        for label, old_value in old_items.items():
            if not predicate(label) or old_value is None:
                continue
            errors.append(float(np.max(np.abs(np.asarray(old_value) - np.asarray(new_items[label])))))
        result[group] = max(errors, default=0.0)
    return result


def _native_step_zero(old_agent, new_agent, batch):
    old_loss, old_info = old_agent.total_loss(batch, old_agent.network.params)
    new_loss, new_info = new_agent.total_loss(batch, new_agent.network.params)
    native = {
        'legacy': {
            'total': _scalar(old_loss),
            'value': _scalar(old_info['value/value_loss']),
            'high_actor': _scalar(old_info['high_actor/actor_loss']),
            'low_actor': _scalar(old_info['low_actor/actor_loss']),
        },
        'computation': {
            'total': _scalar(new_loss),
            'value': _scalar(new_info['value/value_loss']),
            'high_actor': _scalar(new_info['high_actor/actor_loss']),
            'low_actor': _scalar(new_info['low_actor/actor_loss']),
        },
        'loss_abs_errors': {
            'total': abs(_scalar(old_loss) - _scalar(new_loss)),
            'value': abs(_scalar(old_info['value/value_loss']) - _scalar(new_info['value/value_loss'])),
            'high_actor': abs(_scalar(old_info['high_actor/actor_loss']) - _scalar(new_info['high_actor/actor_loss'])),
            'low_actor': abs(_scalar(old_info['low_actor/actor_loss']) - _scalar(new_info['low_actor/actor_loss'])),
        },
        'semantic_parameter_abs_error': _grouped_semantic_errors(
            old_agent.network.params,
            new_agent.network.params,
        ),
    }
    return native


def run(steps):
    devices = jax.devices()
    backend = jax.default_backend()
    print('jax.devices():', devices)
    print('jax.default_backend():', backend)
    if backend != 'gpu' or not any(device.platform == 'gpu' for device in devices):
        raise RuntimeError(
            f'GPU diagnostic requires a CUDA JAX backend; got backend={backend!r}, devices={devices!r}'
        )

    config = _runtime_config()
    old_config = copy.deepcopy(config)
    new_config = copy.deepcopy(config)
    _set_slots(old_config, False)
    _set_slots(new_config, True)

    env = None
    try:
        env, raw_train, _ = make_env_and_datasets(
            DATASET_NAME,
            seed=1234,
            dataset_seed=5678,
            dataset_dir=resolve_dataset_dir(),
        )
        old_dataset = HGCDataset(_copy_dataset(raw_train, 2026), old_config, rng=2026)
        new_dataset = HGCDataset(_copy_dataset(raw_train, 2026), new_config, rng=2026)
        init_dataset = HGCDataset(_copy_dataset(raw_train, 2026), copy.deepcopy(old_config), rng=2026)
        initial_batch = init_dataset.sample(2)

        old_agent = HIQLAgent.create(44, initial_batch['observations'], initial_batch['actions'], old_config)
        new_agent = HIQLAgent.create(44, initial_batch['observations'], initial_batch['actions'], new_config)
        native = _native_step_zero(old_agent, new_agent, initial_batch)

        new_params = _graft_semantic_params(
            old_agent.network.params,
            new_agent.network.params,
            SLOTS,
            value_enabled=True,
        )
        new_agent = new_agent.replace(network=new_agent.network.replace(params=new_params))

        initial_matched = {
            'semantic_parameter_abs_error': _grouped_semantic_errors(
                old_agent.network.params,
                new_agent.network.params,
            ),
            'agent_rng_abs_error': float(
                np.max(np.abs(np.asarray(old_agent.rng) - np.asarray(new_agent.rng)))
            ),
            'optimizer_state_abs_error': _optimizer_state_error(
                old_agent.network.opt_state,
                new_agent.network.opt_state,
            ),
        }
        if initial_matched['semantic_parameter_abs_error']['online'] != 0.0:
            raise AssertionError(f'online semantic graft is not exact: {initial_matched}')
        if initial_matched['semantic_parameter_abs_error']['target_value'] != 0.0:
            raise AssertionError(f'target semantic graft is not exact: {initial_matched}')
        if initial_matched['agent_rng_abs_error'] != 0.0:
            raise AssertionError(f'agent RNG mismatch: {initial_matched}')
        if initial_matched['optimizer_state_abs_error'] != 0.0:
            raise AssertionError(f'optimizer state mismatch: {initial_matched}')

        max_errors = {
            'total_loss': 0.0,
            'value_loss': 0.0,
            'high_actor_loss': 0.0,
            'low_actor_loss': 0.0,
            'semantic_online_parameters': 0.0,
            'target_value_parameters': 0.0,
            'agent_rng': 0.0,
            'optimizer_state': 0.0,
            'update_grad_max': 0.0,
            'update_grad_min': 0.0,
            'update_grad_norm': 0.0,
        }
        first_exact_divergence = None
        first_float32_divergence = None
        first_parameter_divergence = None
        start = time.time()
        for step in range(1, steps + 1):
            old_batch = old_dataset.sample(old_config['batch_size'])
            new_batch = new_dataset.sample(new_config['batch_size'])
            for key in REQUIRED_BATCH_KEYS:
                np.testing.assert_array_equal(old_batch[key], new_batch[key], err_msg=f'batch {step} {key}')

            old_loss, old_info = old_agent.total_loss(old_batch, old_agent.network.params)
            new_loss, new_info = new_agent.total_loss(new_batch, new_agent.network.params)
            losses = {
                'total_loss': abs(_scalar(old_loss) - _scalar(new_loss)),
                'value_loss': abs(_scalar(old_info['value/value_loss']) - _scalar(new_info['value/value_loss'])),
                'high_actor_loss': abs(_scalar(old_info['high_actor/actor_loss']) - _scalar(new_info['high_actor/actor_loss'])),
                'low_actor_loss': abs(_scalar(old_info['low_actor/actor_loss']) - _scalar(new_info['low_actor/actor_loss'])),
            }
            for key, error in losses.items():
                max_errors[key] = max(max_errors[key], error)

            old_agent, old_update_info = old_agent.update(old_batch)
            new_agent, new_update_info = new_agent.update(new_batch)
            parameter_errors = _grouped_semantic_errors(old_agent.network.params, new_agent.network.params)
            max_errors['semantic_online_parameters'] = max(
                max_errors['semantic_online_parameters'], parameter_errors['online']
            )
            max_errors['target_value_parameters'] = max(
                max_errors['target_value_parameters'], parameter_errors['target_value']
            )
            rng_error = float(np.max(np.abs(np.asarray(old_agent.rng) - np.asarray(new_agent.rng))))
            max_errors['agent_rng'] = max(max_errors['agent_rng'], rng_error)
            optimizer_error = _optimizer_state_error(old_agent.network.opt_state, new_agent.network.opt_state)
            max_errors['optimizer_state'] = max(max_errors['optimizer_state'], optimizer_error)
            for key in ('grad/max', 'grad/min', 'grad/norm'):
                error = abs(_scalar(old_update_info[key]) - _scalar(new_update_info[key]))
                metric_name = {'grad/max': 'update_grad_max', 'grad/min': 'update_grad_min', 'grad/norm': 'update_grad_norm'}[key]
                max_errors[metric_name] = max(max_errors[metric_name], error)

            step_loss_error = max(losses.values())
            step_parameter_error = max(parameter_errors.values())
            if first_exact_divergence is None and (step_loss_error > 0.0 or step_parameter_error > 0.0 or rng_error > 0.0):
                first_exact_divergence = step
            if first_float32_divergence is None and max(step_loss_error, step_parameter_error, rng_error) > 1e-6:
                first_float32_divergence = step
            if first_parameter_divergence is None and step_parameter_error > 1e-6:
                first_parameter_divergence = step

            if step == 1 or step % 100 == 0 or step == steps:
                elapsed = time.time() - start
                print(
                    f'step={step}/{steps} loss_error={step_loss_error:.9g} '
                    f'parameter_error={step_parameter_error:.9g} elapsed={elapsed:.1f}s'
                )

        return {
            'backend': backend,
            'devices': [str(device) for device in devices],
            'gpu_models': [device.device_kind for device in devices if device.platform == 'gpu'],
            'dataset': DATASET_NAME,
            'steps_requested': steps,
            'steps_completed': steps,
            'native_step_zero': native,
            'matched_initialization': initial_matched,
            'first_exact_divergence_step': first_exact_divergence,
            'first_float32_divergence_step': first_float32_divergence,
            'first_parameter_divergence_step': first_parameter_divergence,
            'maximum_absolute_errors': max_errors,
        }
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    result = run(args.steps)
    print('RESULT_JSON_BEGIN')
    print(json.dumps(result, indent=2, sort_keys=True))
    print('RESULT_JSON_END')


if __name__ == '__main__':
    main()
