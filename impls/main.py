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

from .agents import agent_configs, agents
from .computation.accounting import (
    count_non_trainable,
    count_parameters,
    hiql_policy_accounting,
)
from .utils.datasets import GCDataset, HGCDataset, MultiHGCDataset
from .utils.env_utils import make_env_and_datasets, resolve_dataset_dir
from .utils.evaluation import evaluate, extract_episode_success
from .utils.flax_utils import restore_agent, save_agent
from .utils.log_utils import CsvLogger
from .utils.reproducibility import derive_seed, seed_everything
from .experiment import (
    create_run_context,
    finalize_run,
    prepare_run_design,
    update_runtime_metadata,
)


def _parse_args(argv=None):
    def _eval_tasks(value):
        if str(value).lower() in {'all', 'none'}:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError('eval_tasks must be positive or all')
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=sorted(agents), default='hiql')
    parser.add_argument('--env_name', default='antmaze-medium-navigate-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', default=None, help='Legacy/debug artifact root.')
    parser.add_argument('--run_root', default='runs', help='Canonical experiment artifact root.')
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
        overrides = configuration.data.get('agent_overrides', {})
        config = _merge_config(config, overrides)
        hidden_dims = tuple(config['actor_hidden_dims'])
        if hidden_dims != (512, 512, 512):
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
        if slot.get('topology') == 'single_state':
            single_state[slot_name] = {
                'topology': 'single_state',
                'primitive': slot.get('primitive', 'mlp'),
                'credit': slot.get('credit', 'direct'),
                'state_dim': int(kwargs.get('state_dim', config['actor_hidden_dims'][-1])),
                'iterations': int(kwargs.get('iterations', 1)),
                'residual': bool(kwargs.get('residual', False)),
                'state_init': kwargs.get('state_init', 'normal_buffer'),
                'state_init_std': float(kwargs.get('state_init_std', 1.0)),
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
        }
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
    )
    return sum(update_info[key] for key in keys if key in update_info)


def _evaluate_tasks(agent, env, config, args, eval_seed):
    task_infos = getattr(env.unwrapped, 'task_infos', None)
    if task_infos is None:
        task_ids = [None]
    else:
        task_count = len(task_infos) if args.eval_tasks is None else min(args.eval_tasks, len(task_infos))
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


def _validate_checkpoint(agent, save_dir, step, observations, goals, actions=None):
    restored = restore_agent(agent, save_dir, step)
    key = jax.random.PRNGKey(derive_seed(step, 17))
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
    compute_snapshot = _resolved_compute_snapshot(config)
    try:
        ogbench_module = os.path.abspath(__import__('ogbench').__file__)
    except (ImportError, AttributeError):
        ogbench_module = None
    artifact_root = args.save_dir or args.run_root
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
        resolved_config={'launcher': vars(args), 'agent': config},
        repo_root=Path(__file__).resolve().parents[1],
        ogbench_module=ogbench_module,
        runtime_extras=_computation_runtime_extras(config),
    )
    run_dir = str(run_context.run_dir)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    env = None
    train_logger = None
    eval_logger = None
    failure_reason = None
    terminal_status = 'completed'
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
        agent_class = agents[config['agent_name']]
        agent = agent_class.create(args.seed, example_batch['observations'], example_batch['actions'], config)
        accounting = _actor_parameter_accounting(agent, config)
        run_context.metadata['actor_parameter_accounting'] = accounting
        update_runtime_metadata(run_dir, {'actor_parameter_accounting': accounting})
        if args.restore_path is not None:
            if args.restore_epoch is None:
                raise ValueError('--restore_epoch is required with --restore_path')
            agent = restore_agent(agent, args.restore_path, args.restore_epoch)

        train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
        eval_logger = CsvLogger(os.path.join(run_dir, 'eval.csv'))
        first_time = last_time = time.time()
        for step in range(1, args.train_steps + 1):
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)
            if not all(np.all(np.isfinite(np.asarray(value))) for value in update_info.values()):
                raise FloatingPointError(f'Non-finite update at step {step}: {update_info}')

            if step % args.log_interval == 0 or step == args.train_steps:
                metrics = {f'training/{key}': value for key, value in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config['batch_size'], evaluation=True)
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    metrics.update({f'validation/{key}': value for key, value in val_info.items()})
                now = time.time()
                metrics['time/interval_seconds'] = (now - last_time) / max(args.log_interval, 1)
                metrics['time/total_seconds'] = now - first_time
                last_time = now
                train_logger.log(metrics, step)
                print(f'step={step} loss={float(_loss_metric(update_info)):.6f}')

            if step % args.eval_interval == 0 or step == args.train_steps:
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
                    },
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
