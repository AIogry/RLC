"""Small OGBench-style HIQL training entry point for RLC."""

import argparse
import copy
import os
import time
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from .agents import agent_configs, agents, resolve_agent_class
from .computation.accounting import (
    actor_slot_accounting,
    computation_slot_accounting,
    count_non_trainable,
    count_parameters,
    hiql_policy_accounting,
)
from .utils.datasets import GCDataset, HGCDataset, MultiHGCDataset
from .utils.env_utils import make_env_and_datasets, resolve_dataset_dir
from .utils.evaluation import evaluate, extract_episode_success
from .utils.checkpointing import should_update_best
from .utils.flax_utils import (
    resolve_checkpoint,
    restore_module_from_checkpoint,
    restore_agent,
    restore_agent_from_checkpoint,
    save_agent,
    save_semantic_checkpoint,
    write_checkpoint_index,
)
from .utils.log_utils import CsvLogger
from .utils.reproducibility import derive_seed, seed_everything
from .experiment import (
    create_run_context,
    finalize_run,
    prepare_run_design,
    validate_source_run_dependency,
    update_runtime_metadata,
)
from .utils.checkpointing import (
    checkpoint_module_fingerprint,
    parameter_module_key,
    tree_fingerprint,
)


def _parse_args(argv=None):
    def _eval_tasks(value):
        normalized = str(value).lower()
        if normalized in {'all', 'none'}:
            return normalized
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError('eval_tasks must be positive, all, or none')
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=sorted(agents), default='hiql')
    parser.add_argument('--env_name', default='antmaze-medium-navigate-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', default=None, help='Legacy/debug artifact root.')
    parser.add_argument('--run_root', default='runs', help='Canonical experiment artifact root.')
    parser.add_argument(
        '--run_attempt', type=int, default=0,
        help='Explicit non-negative rerun instance; zero keeps the canonical path.',
    )
    parser.add_argument('--study', default=None, help='Path to a Study study.yaml.')
    parser.add_argument('--config', default=None, help='Path or ID of a Study configuration YAML.')
    parser.add_argument('--restore_path', default=None)
    parser.add_argument('--restore_epoch', type=int, default=None)
    parser.add_argument('--train_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--actor_loss', choices=('ddpgbc', 'awr'), default=None)
    parser.add_argument('--computation', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--eval_tasks', type=_eval_tasks, default=1)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--eval_temperature', type=float, default=0.0)
    parser.add_argument('--eval_gaussian', type=float, default=None)
    parser.add_argument('--video_episodes', type=int, default=0)
    parser.add_argument(
        '--save-best-checkpoint', '--save_best_checkpoint',
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--save-last-checkpoint', '--save_last_checkpoint',
        action=argparse.BooleanOptionalAction, default=True,
    )
    return parser.parse_args(argv)


def _merge_config(config, overrides):
    """Apply explicit Study agent_overrides without slug-based inference."""

    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and key in config and hasattr(config[key], 'items'):
            _merge_config(config[key], value)
        else:
            config[key] = copy.deepcopy(value)
    return config


def _make_config(args, configuration=None):
    config = agent_configs[args.agent]()
    # Reference CRL configs historically used ml-collections placeholders for
    # optional visual settings.  The first RLC runtime slice is state-based;
    # make those defaults explicit without changing the reference semantics.
    if config['encoder'] is None:
        config['encoder'] = None
    if config['frame_stack'] is None:
        config['frame_stack'] = None
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.latent_dim is not None and 'latent_dim' in config:
        config['latent_dim'] = args.latent_dim
    if args.actor_loss is not None and 'actor_loss' in config:
        config['actor_loss'] = args.actor_loss
    if args.width is not None:
        depth = 3 if args.depth is None else args.depth
        config['actor_hidden_dims'] = (args.width,) * depth
        config['value_hidden_dims'] = (args.width,) * depth
    elif args.depth is not None:
        config['actor_hidden_dims'] = (512,) * args.depth
        config['value_hidden_dims'] = (512,) * args.depth
    if args.agent == 'hiql':
        for slot in ('low_actor', 'high_actor', 'value'):
            config['compute'][slot]['enabled'] = bool(args.computation)
    elif args.agent == 'crl':
        for slot in ('actor', 'critic_state', 'critic_goal', 'value_state', 'value_goal'):
            config['compute'][slot]['enabled'] = bool(args.computation)
        if config['actor_loss'] != 'awr':
            for slot in ('value_state', 'value_goal'):
                config['compute'][slot]['enabled'] = False
    elif args.agent == 'coghp' and args.computation:
        raise ValueError('Vanilla CoGHP does not use --computation; use its official Mixer core.')
    if configuration is not None:
        if configuration.data.get('study_id') == 'M11B':
            from .experiment.m11b import m11b_agent_overrides

            overrides = m11b_agent_overrides(
                configuration.data['algorithm'],
                configuration.data['environment'],
                configuration.data['condition'],
            )
        else:
            overrides = configuration.data.get('agent_overrides', {})
        config = _merge_config(config, overrides)
        hidden_dims = tuple(config['actor_hidden_dims'])
        if (
            hidden_dims != (512, 512, 512)
            and not configuration.data.get('allow_noncanonical_actor_hidden_dims', False)
        ):
            raise ValueError(
                'M9 canonical actor hidden dims must remain (512, 512, 512), '
                f'got {hidden_dims!r}'
            )
        if config.get('actor_loss') != 'ddpgbc' and configuration.data.get('algorithm') == 'crl':
            raise ValueError('M9 CRL actor configurations must use actor_loss=ddpgbc.')
    return config


def _computation_runtime_extras(config):
    extras = {'resolved_actor_hidden_dims': list(config['actor_hidden_dims'])}
    slots = config.get('compute', {})
    single_state = {}
    two_state = {}
    for slot_name, slot in slots.items():
        if not slot.get('enabled', False):
            continue
        kwargs = slot.get('topology_kwargs', {})
        is_critic_branch = slot_name.startswith(('critic_', 'value_'))
        if slot.get('topology') == 'single_state':
            single_state[slot_name] = {
                'topology': 'single_state',
                'primitive': slot.get('primitive', 'mlp'),
                'credit': slot.get('credit', 'direct'),
                'state_dim': int(kwargs.get('state_dim', config['actor_hidden_dims'][-1])),
                'iterations': int(kwargs.get('iterations', 1)),
                'update_depth': int(kwargs.get('update_depth', 2)),
                'layer_norm': bool(kwargs.get(
                    'layer_norm',
                    config.get('layer_norm', False) if is_critic_branch else False,
                )),
                'update_activate_final': bool(kwargs.get(
                    'update_activate_final', not is_critic_branch,
                )),
                'residual': bool(kwargs.get('residual', False)),
                'input_injection': kwargs.get('input_injection', 'z_plus_x'),
                'state_init': kwargs.get('state_init', 'normal_buffer'),
                'state_init_std': float(kwargs.get('state_init_std', 1.0)),
                'parameter_sharing': slot.get(
                    'parameter_sharing', kwargs.get('parameter_sharing', 'shared')
                ),
            }
        elif slot.get('topology') == 'feedforward':
            extras.setdefault('feedforward', {})[slot_name] = {
                'topology': 'feedforward',
                'primitive': slot.get('primitive', 'mlp'),
                'block': slot.get('block', 'plain'),
                'block_kwargs': _jsonable(slot.get('block_kwargs', {})),
            }
        elif slot.get('topology') == 'two_state':
            h_cycles = int(kwargs.get('h_cycles', 2))
            l_cycles = int(kwargs.get('l_cycles', 1))
            two_state[slot_name] = {
                'topology': 'two_state',
                'primitive': slot.get('primitive', 'mlp'),
                'credit': slot.get('credit'),
                'state_dim': int(kwargs.get('state_dim', config['actor_hidden_dims'][-1])),
                'h_cycles': h_cycles,
                'l_cycles': l_cycles,
                'update_depth': int(kwargs.get('update_depth', 2)),
                'layer_norm': bool(kwargs.get(
                    'layer_norm',
                    config.get('layer_norm', False) if is_critic_branch else False,
                )),
                'update_activate_final': bool(kwargs.get(
                    'update_activate_final', not is_critic_branch,
                )),
                'h_update_executions': h_cycles,
                'l_update_executions': h_cycles * l_cycles,
                'total_update_executions': h_cycles * (l_cycles + 1),
                'input_injection': kwargs.get('input_injection', 'l_receives_x'),
                'state_init': kwargs.get('state_init', 'normal_buffer'),
                'state_init_std': float(kwargs.get('state_init_std', 1.0)),
            }
    if single_state:
        extras['single_state'] = single_state
    if two_state:
        extras['two_state'] = two_state
    return extras


# Backward-compatible private name used by earlier M9A diagnostics.
_single_state_runtime_extras = _computation_runtime_extras


def _actor_parameter_accounting(agent, config):
    """Report actor/core totals, buffers, schedules, and credit metadata."""

    if config['agent_name'] == 'hiql':
        slot_names = ('high_actor', 'low_actor')
    elif config['agent_name'] == 'crl':
        slot_names = ('actor',)
    else:
        return {}

    params = agent.network.params
    model_state = agent.network.model_state or {}
    buffers = model_state.get('buffers', {}) if hasattr(model_state, 'get') else {}
    report = {}
    for slot_name in slot_names:
        module = params.get(f'modules_{slot_name}', params.get(slot_name, {}))
        actor_net = module.get('actor_net', {}) if hasattr(module, 'get') else {}
        core = actor_net.get('topology', actor_net) if hasattr(actor_net, 'get') else actor_net
        buffer_module = buffers.get(f'modules_{slot_name}', buffers.get(slot_name, {})) if hasattr(buffers, 'get') else {}
        spec = config.get('compute', {}).get(slot_name, {})
        enabled = bool(spec.get('enabled', False))
        topology_kwargs = spec.get('topology_kwargs', {})
        iterations = int(topology_kwargs.get('iterations', 0)) if enabled and spec.get('topology') == 'single_state' else 0
        flat_module = flatten_dict(module)
        input_kernel = next(
            (
                value for path, value in flat_module.items()
                if path[-3:] == ('input_mapping', 'Dense_0', 'kernel')
                or path[-3:] == ('actor_net', 'Dense_0', 'kernel')
            ),
            None,
        )
        input_dim = int(input_kernel.shape[0]) if input_kernel is not None else None
        output_kernel = next(
            (
                value for path, value in flat_module.items()
                if path[-2:] in (('mean_net', 'kernel'), ('logit_net', 'kernel'))
            ),
            None,
        )
        output_dim = int(output_kernel.shape[1]) if output_kernel is not None else None
        hidden_dim = int(config['actor_hidden_dims'][-1])
        vanilla_params = None
        if input_dim is not None and output_dim is not None:
            vanilla_params = (
                input_dim * hidden_dim + hidden_dim
                + hidden_dim * hidden_dim + hidden_dim
                + hidden_dim * hidden_dim + hidden_dim
                + hidden_dim * output_dim + output_dim
            )
        topology = spec.get('topology') if enabled else None
        topology_kwargs = topology_kwargs if enabled else {}
        parameter_sharing = spec.get(
            'parameter_sharing', topology_kwargs.get('parameter_sharing', 'shared')
        )
        block = spec.get('block', 'plain') if enabled else 'plain'
        state_dim = (
            int(topology_kwargs.get('state_dim', hidden_dim))
            if topology in ('single_state', 'two_state') else None
        )
        input_mapping_params = count_parameters(core.get('input_mapping', {})) if hasattr(core, 'get') else 0
        h_update_params = count_parameters(core.get('h_update', {})) if hasattr(core, 'get') else 0
        l_update_params = count_parameters(core.get('l_update', {})) if hasattr(core, 'get') else 0
        h_cycles = int(topology_kwargs.get('h_cycles', 0)) if topology == 'two_state' else 0
        l_cycles = int(topology_kwargs.get('l_cycles', 0)) if topology == 'two_state' else 0
        h_update_executions = h_cycles if topology == 'two_state' else 0
        l_update_executions = h_cycles * l_cycles if topology == 'two_state' else 0
        total_update_executions = h_update_executions + l_update_executions
        report[slot_name] = {
            'topology': topology,
            'primitive': spec.get('primitive') if enabled else None,
            'credit': spec.get('credit') if enabled else None,
            'trainable_params': count_parameters(module),
            'core_trainable_params': count_parameters(core),
            'input_mapping_params': input_mapping_params,
            'h_update_params': h_update_params,
            'l_update_params': l_update_params,
            'baseline_actor_trainable_params': vanilla_params,
            'vanilla_actor_trainable_params': vanilla_params,
            'buffer_elements': count_non_trainable(buffer_module),
            'state_dim': state_dim,
            'iterations': iterations,
            'shared_update_executions': iterations,
            'h_cycles': h_cycles or None,
            'l_cycles': l_cycles or None,
            'h_update_executions': h_update_executions,
            'l_update_executions': l_update_executions,
            'total_update_executions': total_update_executions,
            'state_init': topology_kwargs.get('state_init') if topology in ('single_state', 'two_state') else None,
            'state_init_std': float(topology_kwargs.get('state_init_std', 1.0)) if topology in ('single_state', 'two_state') else None,
            'parameter_sharing': parameter_sharing if topology == 'single_state' else None,
            'block': block if topology == 'feedforward' else None,
        }
        report[slot_name].update(actor_slot_accounting(
            module,
            buffer_module,
            topology=topology,
            iterations=iterations,
            topology_kwargs=topology_kwargs,
            parameter_sharing=parameter_sharing,
            block=block,
        ))
    if config['agent_name'] == 'hiql':
        policy_audit = hiql_policy_accounting(
            params,
            buffers,
            slot_specs=config.get('compute', {}),
        )
        for slot_name in slot_names:
            report[slot_name].update(policy_audit['slots'][slot_name])
        report['policy'] = {
            key: value
            for key, value in policy_audit.items()
            if key != 'slots'
        }
    return report


def _computation_slot_accounting(agent, config):
    """Account every enabled actor/critic/value computation slot.

    ``actor_parameter_accounting`` is retained for backwards compatibility;
    this generic report adds the same auditable schema for CRL bilinear
    branches without changing the legacy actor metadata shape.
    """

    params = agent.network.params
    model_state = agent.network.model_state or {}
    buffers = model_state.get('buffers', {}) if hasattr(model_state, 'get') else {}
    compute = config.get('compute', {})
    slot_paths = {
        'actor': (('modules_actor',), ('actor_net', 'topology')),
        'high_actor': (('modules_high_actor',), ('actor_net', 'topology')),
        'low_actor': (('modules_low_actor',), ('actor_net', 'topology')),
        'critic_state': (('modules_critic', 'phi'), ('core', 'topology')),
        'critic_goal': (('modules_critic', 'psi'), ('core', 'topology')),
        'value_state': (('modules_value', 'phi'), ('core', 'topology')),
        'value_goal': (('modules_value', 'psi'), ('core', 'topology')),
    }

    def path_get(tree, path):
        for key in path:
            if not hasattr(tree, 'get') or key not in tree:
                raise KeyError(f'Missing parameter path component {key!r} in {path!r}')
            tree = tree[key]
        return tree

    report = {}
    for slot_name, spec in compute.items():
        if not spec.get('enabled', False):
            continue
        if slot_name not in slot_paths:
            raise ValueError(f'Unsupported computation slot for accounting: {slot_name!r}')
        module_path, core_path = slot_paths[slot_name]
        topology = spec.get('topology')
        kwargs = dict(spec.get('topology_kwargs', {}))
        kwargs.setdefault(
            'parameter_sharing', spec.get('parameter_sharing', 'shared')
        )
        kwargs.setdefault('block', spec.get('block', 'plain'))
        hidden_dim = int(
            config['value_hidden_dims'][-1]
            if slot_name.startswith(('critic_', 'value_'))
            else config['actor_hidden_dims'][-1]
        )
        if topology in ('single_state', 'two_state'):
            kwargs.setdefault('state_dim', hidden_dim)
            kwargs.setdefault('update_depth', 2)
            # Primitive semantics are caller-owned, not M11A factors.  CRL
            # critic/value branches replace vanilla LayerNorm MLPs with a
            # final latent Dense, while actor slots preserve legacy M9 MLPs.
            is_critic_branch = slot_name.startswith(('critic_', 'value_'))
            kwargs.setdefault(
                'layer_norm',
                bool(config.get('layer_norm', False)) if is_critic_branch else False,
            )
            kwargs.setdefault('update_activate_final', not is_critic_branch)
            if topology == 'single_state':
                kwargs.setdefault('iterations', 1)
            else:
                kwargs.setdefault('h_cycles', 2)
                kwargs.setdefault('l_cycles', 1)
        module = path_get(params, module_path)
        buffer_module = path_get(buffers, module_path) if buffers else {}
        report[slot_name] = computation_slot_accounting(
            module,
            buffer_module,
            slot_name=slot_name,
            topology=topology,
            primitive=spec.get('primitive', 'mlp'),
            credit=spec.get('credit', 'direct'),
            topology_kwargs=kwargs,
            core_path=core_path,
        )
    return report


_ACCOUNTING_CONSISTENCY_FIELDS = (
    'topology',
    'trainable_params',
    'buffer_elements',
    'state_dim',
    'iterations',
)


def _accounting_consistency_audit(legacy, generic, config):
    """Require legacy and generic accounting to agree on shared invariants.

    The legacy HIQL report remains the compatibility surface used by existing
    manifests, while the generic report is the slot-complete accounting path.
    This audit compares the two reports after agent initialization so a future
    topology/module-path drift fails before the first optimizer update.
    """

    mismatches = []
    checked_slots = []
    for slot_name, spec in config.get('compute', {}).items():
        if not spec.get('enabled', False):
            continue
        checked_slots.append(slot_name)
        if slot_name not in legacy:
            mismatches.append(f'{slot_name}: missing from actor_parameter_accounting')
            continue
        if slot_name not in generic:
            mismatches.append(f'{slot_name}: missing from computation_slot_accounting')
            continue
        legacy_slot = legacy[slot_name]
        generic_slot = generic[slot_name]
        for field in _ACCOUNTING_CONSISTENCY_FIELDS:
            left = legacy_slot.get(field)
            right = generic_slot.get(field)
            # The old report encoded non-recurrent/two-state iterations as 0;
            # the generic schema uses None when no SingleState iteration count
            # exists.  Both mean that the field is not applicable.
            if field == 'iterations' and legacy_slot.get('topology') != 'single_state':
                left = 0 if left is None else left
                right = 0 if right is None else right
            if left != right:
                mismatches.append(
                    f'{slot_name}.{field}: legacy={left!r} generic={right!r}'
                )
    if mismatches:
        raise ValueError('Accounting consistency audit failed: ' + '; '.join(mismatches))
    return {
        'status': 'pass',
        'checked_slots': checked_slots,
        'fields': list(_ACCOUNTING_CONSISTENCY_FIELDS),
        'mismatches': [],
    }


def _as_float_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.size == 1:
            result[key] = float(array)
    return result


def _jsonable(value):
    """Convert ConfigDict/numpy containers into stable JSON values."""

    if isinstance(value, Mapping) or hasattr(value, 'items'):
        items = value.items()
        return {
            str(key): _jsonable(item)
            for key, item in sorted(items, key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolved_compute_snapshot(config):
    """Return every resolved computation slot in stable JSON form."""

    compute = config.get('compute', {}) if config is not None else {}
    if compute is None:
        return {}
    return _jsonable(compute)


def _loss_metric(update_info):
    """Return the algorithm-specific total from the shared update info."""
    keys = (
        'critic/contrastive_loss', 'value/contrastive_loss', 'actor/actor_loss',
        'value/value_loss', 'high_actor/actor_loss', 'low_actor/actor_loss',
        'actor/q_loss', 'contrastive_loss', 'actor_loss', 'q_loss',
    )
    return sum(update_info[key] for key in keys if key in update_info)


def _evaluate_tasks(agent, env, config, args, eval_seed):
    if args.eval_tasks == 'none':
        return {}
    task_infos = getattr(env.unwrapped, 'task_infos', None)
    if task_infos is None:
        task_ids = [None]
    else:
        task_count = (
            len(task_infos)
            if args.eval_tasks in (None, 'all')
            else min(args.eval_tasks, len(task_infos))
        )
        task_ids = list(range(1, task_count + 1))

    metrics = {}
    success_values = []
    for task_id in task_ids:
        stats, _, _ = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=args.eval_episodes,
            num_video_episodes=args.video_episodes,
            eval_temperature=args.eval_temperature,
            eval_gaussian=args.eval_gaussian,
            seed=derive_seed(eval_seed, task_id or 0),
        )
        name = 'default' if task_id is None else task_infos[task_id - 1]['task_name']
        for key, value in stats.items():
            if key.endswith('success') or key == 'success':
                success_values.append(float(value))
                metrics[f'evaluation/{name}_{key}'] = float(value)
        if any(key.endswith('success') or key == 'success' for key in stats):
            # Keep the legacy aggregate fields, while routing the canonical
            # success interpretation through the shared ambiguity check used
            # by post-hoc episode evaluation.
            extract_episode_success(stats)
    if success_values:
        metrics['evaluation/overall_success'] = float(np.mean(success_values))
    return metrics


def _validate_restored_checkpoint(agent, restored, observations, goals, actions=None, seed_step=0):
    key = jax.random.PRNGKey(derive_seed(seed_step, 17))
    if agent.config.get('agent_name') == 'coghp':
        # CoGHP's public policy API receives one unbatched observation; the
        # dataset example batch used by the shared trainer is (1, D).
        observations = observations[0]
        goals = goals[0]
    before_action = agent.sample_actions(observations, goals, seed=key)
    after_action = restored.sample_actions(observations, goals, seed=key)
    np.testing.assert_array_equal(np.asarray(before_action), np.asarray(after_action))
    # ModuleDict stores selected modules under ``modules_<name>`` in the
    # parameter tree.  CRL's DDPG+BC configuration has no V module, so use its
    # legacy bilinear critic for the value probe instead.
    module_name = 'value' if 'modules_value' in agent.network.params else 'critic'
    if module_name == 'critic':
        before_value = agent.network.select(module_name)(observations, goals, actions)
        after_value = restored.network.select(module_name)(observations, goals, actions)
    else:
        before_value = agent.network.select(module_name)(observations, goals)
        after_value = restored.network.select(module_name)(observations, goals)
    np.testing.assert_array_equal(np.asarray(before_value), np.asarray(after_value))
    print('Checkpoint save/load probe: PASS (same action/value)')
    if agent.network.model_state:
        before_state = jax.tree_util.tree_leaves(agent.network.model_state)
        after_state = jax.tree_util.tree_leaves(restored.network.model_state)
        if len(before_state) != len(after_state):
            raise AssertionError('Checkpoint model_state leaf count changed during restore')
        for before_leaf, after_leaf in zip(before_state, after_state):
            np.testing.assert_array_equal(np.asarray(before_leaf), np.asarray(after_leaf))
        print('Checkpoint buffer/model_state probe: PASS')


def _validate_checkpoint(agent, save_dir, step, observations, goals, actions=None):
    restored = restore_agent(agent, save_dir, step)
    _validate_restored_checkpoint(
        agent, restored, observations, goals, actions=actions, seed_step=step
    )


def _validate_checkpoint_file(agent, checkpoint_path, observations, goals, actions=None, seed_step=0):
    restored = restore_agent_from_checkpoint(agent, checkpoint_path)
    _validate_restored_checkpoint(
        agent, restored, observations, goals, actions=actions, seed_step=seed_step
    )


def _update_for_training_mode(agent, batch, config):
    """Dispatch a generic training mode without study-specific branches."""

    training_mode = config.get('training_mode', 'joint')
    if training_mode == 'critic_only':
        return agent.critic_only_update(batch)
    if training_mode in {'joint', 'policy_extraction'}:
        return agent.update(batch)
    raise ValueError(f'Unsupported training_mode: {training_mode!r}')


def _frozen_dependency_records(dependencies):
    return {
        name: {
            key: value
            for key, value in dependency.items()
            if key not in {'checkpoint_metadata'}
        }
        for name, dependency in dependencies.items()
    }


def _assert_frozen_dependencies(agent, dependencies, *, checkpoint_path=None):
    """Fail loudly if a structurally frozen dependency changes."""

    for name, dependency in dependencies.items():
        module_name = dependency['module']
        module_key = parameter_module_key(agent.network.params, module_name)
        actual = tree_fingerprint(agent.network.params[module_key])
        expected = dependency['module_fingerprint']
        if actual != expected:
            raise ValueError(
                f'Frozen dependency {name!r} changed in memory: '
                f'expected={expected}, actual={actual}'
            )
        if checkpoint_path is not None:
            checkpoint_actual = checkpoint_module_fingerprint(
                checkpoint_path, module_name,
            )
            if checkpoint_actual != expected:
                raise ValueError(
                    f'Frozen dependency {name!r} changed in checkpoint: '
                    f'expected={expected}, actual={checkpoint_actual}'
                )


def _checkpoint_metadata(run_context, args, compute_snapshot, dataset_dir, *, role, step,
                         best_step, metric_value, selected_from_training_evaluation):
    metadata = {
        'environment': args.env_name,
        'dataset_dir': str(dataset_dir),
        'computation': bool(args.computation),
        'compute_slots': compute_snapshot,
        'study_id': run_context.metadata['study_id'],
        'config_id': run_context.metadata['config_id'],
        'config_slug': run_context.metadata['config_slug'],
        'seed': args.seed,
        'training_seed': args.seed,
        'git_commit': run_context.metadata['git_commit'],
        'selection_metric': 'evaluation/overall_success',
        'selection_metric_value': metric_value,
        'best_step': best_step,
        'train_steps': args.train_steps,
        'evaluation_protocol_at_selection': {
            'eval_tasks': args.eval_tasks,
            'eval_episodes': args.eval_episodes,
            'eval_temperature': args.eval_temperature,
            'eval_gaussian': args.eval_gaussian,
        },
        'selected_from_training_evaluation': bool(selected_from_training_evaluation),
        'checkpoint_role': role,
        'checkpoint_step': int(step),
    }
    if run_context.metadata.get('frozen_dependencies'):
        metadata['frozen_dependencies'] = run_context.metadata['frozen_dependencies']
    return metadata


def run(args):
    seed_everything(args.seed)
    if (args.study is None) != (args.config is None):
        raise ValueError('--study and --config must be supplied together')
    study = configuration = None
    if args.study is not None:
        study, configuration = prepare_run_design(args.study, args.config)
        if configuration.data.get('algorithm') != args.agent:
            raise ValueError(
                f'Study configuration algorithm {configuration.data.get("algorithm")!r} '
                f'does not match --agent={args.agent!r}'
            )
        if args.computation:
            raise ValueError('Canonical Study configurations must control computation slots via agent_overrides, not --computation.')
        if args.width is not None or args.depth is not None:
            raise ValueError('Canonical Study runs do not allow --width or --depth architecture overrides.')
        override_actor_loss = configuration.data.get('agent_overrides', {}).get('actor_loss')
        if args.actor_loss is not None and override_actor_loss is not None and args.actor_loss != override_actor_loss:
            raise ValueError(
                f'--actor_loss={args.actor_loss!r} conflicts with Study agent_overrides '
                f'actor_loss={override_actor_loss!r}'
            )
    config = _make_config(args, configuration=configuration)
    dataset_dir = resolve_dataset_dir()
    if configuration is not None and configuration.data.get('executable', True) is False:
        raise ValueError(
            f'{configuration.config_id} is a planned/non-executable configuration; '
            'no scientific run was started.'
        )
    artifact_root = args.save_dir or args.run_root
    dependency_records = {}
    if configuration is not None:
        dependency_specs = configuration.data.get('dependencies', {})
        if isinstance(dependency_specs, Mapping):
            for dependency_name in dependency_specs:
                dependency_records[dependency_name] = validate_source_run_dependency(
                    study,
                    configuration,
                    dependency_name,
                    seed=args.seed,
                    run_root=artifact_root,
                    resolved_agent=config,
                )
    compute_snapshot = _resolved_compute_snapshot(config)
    try:
        ogbench_module = os.path.abspath(__import__('ogbench').__file__)
    except (ImportError, AttributeError):
        ogbench_module = None
    resolved_runtime_config = {
        'launcher': vars(args),
        'agent': config,
        'dataset_root': dataset_dir,
        'environment': args.env_name,
        'training_seed': args.seed,
        'run_attempt': args.run_attempt,
        'training_mode': config.get('training_mode', 'joint'),
        'runtime_variant': config.get('runtime_variant', 'canonical'),
        'dependencies': dependency_records,
    }
    runtime_extras = _computation_runtime_extras(config)
    runtime_extras.update({
        'training_mode': config.get('training_mode', 'joint'),
        'runtime_variant': config.get('runtime_variant', 'canonical'),
        'seed_streams': {
            'actor_seed': int(args.seed),
            'dataset_seed': derive_seed(args.seed, 1),
            'train_data_rng_seed': derive_seed(args.seed, 11),
            'evaluation_seed': derive_seed(args.seed, 4),
            'sampling_protocol': 'explicit_derived_seed_v1',
        },
    })
    if dependency_records:
        runtime_extras['frozen_dependencies'] = _frozen_dependency_records(
            dependency_records,
        )
    if configuration is not None and configuration.data.get('study_id') == 'M11B':
        runtime_extras.update({
            'semantic_condition': configuration.data['semantic_condition'],
            'semantic_label': configuration.data.get('semantic_label', configuration.data['semantic_condition']),
            'm11b_condition': configuration.data['condition'],
            'm11b_canonical_source': '/home/eai/Research/offline-rl/docs/ALGORITHM_HYPERPARAMETERS.md',
            'm11b_environment_reference': configuration.data['environment'],
        })
    run_context = create_run_context(
        study=study,
        configuration=configuration,
        run_root=artifact_root,
        legacy_root=args.save_dir or os.path.join(args.run_root, 'legacy'),
        algorithm=args.agent,
        environment=args.env_name,
        seed=args.seed,
        dataset_dir=dataset_dir,
        computation=args.computation,
        compute_slots=compute_snapshot,
        resolved_config=resolved_runtime_config,
        repo_root=Path(__file__).resolve().parents[1],
        ogbench_module=ogbench_module,
        runtime_extras=runtime_extras,
        run_attempt=args.run_attempt,
    )
    run_dir = str(run_context.run_dir)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    env = None
    train_logger = None
    eval_logger = None
    failure_reason = None
    terminal_status = 'completed'
    best_metric = None
    best_step = None
    best_record = None
    last_record = None
    last_eval_metric = None
    update_runtime_metadata(run_dir, {
        'checkpoint_lifecycle': {
            'selection_metric': 'evaluation/overall_success',
            'selection_rule': 'strict_greater_than_keep_earlier_tie',
            'save_best_checkpoint': bool(args.save_best_checkpoint),
            'save_last_checkpoint': bool(args.save_last_checkpoint),
            'evaluation_protocol_at_selection': {
                'eval_tasks': args.eval_tasks,
                'eval_episodes': args.eval_episodes,
                'eval_temperature': args.eval_temperature,
                'eval_gaussian': args.eval_gaussian,
            },
        },
    })
    try:
        env, raw_train, raw_val = make_env_and_datasets(
            args.env_name,
            frame_stack=config['frame_stack'],
            seed=derive_seed(args.seed, 3),
            dataset_seed=derive_seed(args.seed, 1),
            dataset_dir=dataset_dir,
        )
        dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
            'MultiHGCDataset': MultiHGCDataset,
        }[config['dataset_class']]
        train_dataset = dataset_class(raw_train, config, rng=derive_seed(args.seed, 11))
        val_dataset = dataset_class(raw_val, config, rng=derive_seed(args.seed, 12)) if raw_val is not None else None

        example_batch = train_dataset.sample(1)
        if config['discrete']:
            example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)
        agent_class = resolve_agent_class(
            config['agent_name'], config.get('runtime_variant', 'canonical'),
        )
        agent = agent_class.create(args.seed, example_batch['observations'], example_batch['actions'], config)
        if dependency_records:
            for dependency in dependency_records.values():
                agent = restore_module_from_checkpoint(
                    agent, dependency['checkpoint_path'], dependency['module'],
                )
            _assert_frozen_dependencies(agent, dependency_records)
            for dependency in dependency_records.values():
                dependency['target_module_fingerprint_before'] = tree_fingerprint(
                    agent.network.params[
                        parameter_module_key(agent.network.params, dependency['module'])
                    ],
                )
            # Keep the in-memory RunContext and every subsequent semantic
            # checkpoint aligned with the post-restore invariant baseline.
            run_context.metadata['frozen_dependencies'] = _frozen_dependency_records(
                dependency_records,
            )
        accounting = _actor_parameter_accounting(agent, config)
        run_context.metadata['actor_parameter_accounting'] = accounting
        slot_accounting = _computation_slot_accounting(agent, config)
        run_context.metadata['computation_slot_accounting'] = slot_accounting
        accounting_consistency = _accounting_consistency_audit(
            accounting, slot_accounting, config,
        )
        run_context.metadata['accounting_consistency'] = accounting_consistency
        update_runtime_metadata(run_dir, {
            'actor_parameter_accounting': accounting,
            'computation_slot_accounting': slot_accounting,
            'accounting_consistency': accounting_consistency,
            'frozen_dependencies': _frozen_dependency_records(dependency_records),
        })
        if args.restore_path is not None:
            if args.restore_epoch is None:
                raise ValueError('--restore_epoch is required with --restore_path')
            agent = restore_agent(agent, args.restore_path, args.restore_epoch)

        train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
        eval_logger = CsvLogger(os.path.join(run_dir, 'eval.csv'))
        first_time = last_time = time.time()
        for step in range(1, args.train_steps + 1):
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = _update_for_training_mode(agent, batch, config)
            if not all(np.all(np.isfinite(np.asarray(value))) for value in update_info.values()):
                raise FloatingPointError(f'Non-finite update at step {step}: {update_info}')
            if dependency_records and (
                step % args.save_interval == 0 or step == args.train_steps
            ):
                _assert_frozen_dependencies(agent, dependency_records)

            if step % args.log_interval == 0 or step == args.train_steps:
                metrics = {f'training/{key}': value for key, value in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config['batch_size'], evaluation=True)
                    if config.get('training_mode', 'joint') == 'critic_only':
                        _, val_info = agent.critic_only_loss(val_batch, grad_params=None)
                    else:
                        _, val_info = agent.total_loss(val_batch, grad_params=None)
                    metrics.update({f'validation/{key}': value for key, value in val_info.items()})
                now = time.time()
                metrics['time/interval_seconds'] = (now - last_time) / max(args.log_interval, 1)
                metrics['time/total_seconds'] = now - first_time
                last_time = now
                train_logger.log(metrics, step)
                print(f'step={step} loss={float(_loss_metric(update_info)):.6f}')

            if args.eval_tasks != 'none' and (
                step % args.eval_interval == 0 or step == args.train_steps
            ):
                eval_metrics = _evaluate_tasks(
                    agent,
                    env,
                    config,
                    args,
                    derive_seed(args.seed, 4, step),
                )
                if eval_metrics:
                    eval_logger.log(eval_metrics, step)
                    print('evaluation:', _as_float_metrics(eval_metrics))
                    metric = eval_metrics.get('evaluation/overall_success')
                    if metric is not None and np.isfinite(float(metric)):
                        last_eval_metric = float(metric)
                        if args.save_best_checkpoint and should_update_best(metric, best_metric):
                            best_metric = float(metric)
                            best_step = step
                            best_record = save_semantic_checkpoint(
                                agent,
                                run_dir,
                                'best',
                                step,
                                _checkpoint_metadata(
                                    run_context,
                                    args,
                                    compute_snapshot,
                                    dataset_dir,
                                    role='best',
                                    step=step,
                                    best_step=best_step,
                                    metric_value=best_metric,
                                    selected_from_training_evaluation=True,
                                ),
                            )
                            if dependency_records:
                                _assert_frozen_dependencies(
                                    agent,
                                    dependency_records,
                                    checkpoint_path=best_record['path']
                                    if os.path.isabs(best_record['path'])
                                    else os.path.join(run_dir, best_record['path']),
                                )
                            write_checkpoint_index(
                                run_dir,
                                best=best_record,
                                last=last_record,
                            )
                            best_checkpoint = resolve_checkpoint(run_dir, 'best')
                            goal_key = 'high_actor_goals' if 'high_actor_goals' in example_batch else 'actor_goals'
                            _validate_checkpoint_file(
                                agent,
                                best_checkpoint['checkpoint_path'],
                                example_batch['observations'],
                                example_batch[goal_key],
                                example_batch.get('actions'),
                                seed_step=step,
                            )

            if step % args.save_interval == 0 or step == args.train_steps:
                checkpoint_path = save_agent(
                    agent,
                    checkpoints_dir,
                    step,
                    checkpoint_metadata={
                        'environment': args.env_name,
                        'dataset_dir': dataset_dir,
                        'computation': bool(args.computation),
                        'compute_slots': compute_snapshot,
                        'study_id': run_context.metadata['study_id'],
                        'config_id': run_context.metadata['config_id'],
                        'config_slug': run_context.metadata['config_slug'],
                        'git_commit': run_context.metadata['git_commit'],
                        'seed': args.seed,
                        'training_seed': args.seed,
                        'checkpoint_role': 'numeric',
                        'checkpoint_step': step,
                        'train_steps': args.train_steps,
                        'training_mode': config.get('training_mode', 'joint'),
                        'runtime_variant': config.get('runtime_variant', 'canonical'),
                        'frozen_dependencies': _frozen_dependency_records(dependency_records),
                    },
                )
                if dependency_records:
                    _assert_frozen_dependencies(
                        agent, dependency_records, checkpoint_path=checkpoint_path,
                    )
                del checkpoint_path
                goal_key = 'high_actor_goals' if 'high_actor_goals' in example_batch else 'actor_goals'
                _validate_checkpoint(
                    agent,
                    checkpoints_dir,
                    step,
                    example_batch['observations'],
                    example_batch[goal_key],
                    example_batch.get('actions'),
                )
        if args.save_last_checkpoint:
            last_record = save_semantic_checkpoint(
                agent,
                run_dir,
                'last',
                args.train_steps,
                _checkpoint_metadata(
                    run_context,
                    args,
                    compute_snapshot,
                    dataset_dir,
                    role='last',
                    step=args.train_steps,
                    best_step=best_step,
                    metric_value=last_eval_metric,
                    selected_from_training_evaluation=False,
                ),
            )
            if dependency_records:
                last_checkpoint_path = os.path.join(run_dir, last_record['path'])
                _assert_frozen_dependencies(
                    agent, dependency_records, checkpoint_path=last_checkpoint_path,
                )
            write_checkpoint_index(
                run_dir,
                best=best_record,
                last=last_record,
            )
            last_checkpoint = resolve_checkpoint(run_dir, 'last')
            goal_key = 'high_actor_goals' if 'high_actor_goals' in example_batch else 'actor_goals'
            _validate_checkpoint_file(
                agent,
                last_checkpoint['checkpoint_path'],
                example_batch['observations'],
                example_batch[goal_key],
                example_batch.get('actions'),
                seed_step=args.train_steps,
            )
            write_checkpoint_index(
                run_dir,
                best=best_record,
                last=last_record,
            )
    except KeyboardInterrupt as error:
        terminal_status = 'aborted'
        failure_reason = f'{type(error).__name__}: {error}'
        raise
    except BaseException as error:
        terminal_status = 'failed'
        failure_reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        if train_logger is not None:
            train_logger.close()
        if eval_logger is not None:
            eval_logger.close()
        if env is not None:
            env.close()
        finalize_run(
            run_dir,
            terminal_status,
            failure_reason=failure_reason,
        )
    return run_dir


def main(argv=None):
    return run(_parse_args(argv))


if __name__ == '__main__':
    main()
