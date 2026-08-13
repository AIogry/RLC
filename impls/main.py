"""Small OGBench-style HIQL training entry point for RLC."""

import argparse
import json
import os
import time

import jax
import numpy as np

from .agents import agent_configs, agents
from .utils.datasets import GCDataset, HGCDataset
from .utils.env_utils import make_env_and_datasets, resolve_dataset_dir
from .utils.evaluation import evaluate
from .utils.flax_utils import restore_agent, save_agent
from .utils.log_utils import CsvLogger, get_exp_name
from .utils.reproducibility import derive_seed, seed_everything


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=sorted(agents), default='hiql')
    parser.add_argument('--env_name', default='antmaze-medium-navigate-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', default='exp')
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
    parser.add_argument('--eval_tasks', type=int, default=1)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--eval_temperature', type=float, default=0.0)
    parser.add_argument('--eval_gaussian', type=float, default=None)
    parser.add_argument('--video_episodes', type=int, default=0)
    return parser.parse_args(argv)


def _make_config(args):
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
        config['compute']['actor']['enabled'] = bool(args.computation)
    return config


def _as_float_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.size == 1:
            result[key] = float(array)
    return result


def _loss_metric(update_info):
    """Return the algorithm-specific total from the shared update info."""
    keys = ('critic/contrastive_loss', 'value/contrastive_loss', 'actor/actor_loss')
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
    if success_values:
        metrics['evaluation/overall_success'] = float(np.mean(success_values))
    return metrics


def _validate_checkpoint(agent, save_dir, step, observations, goals, actions=None):
    restored = restore_agent(agent, save_dir, step)
    key = jax.random.PRNGKey(derive_seed(step, 17))
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


def run(args):
    seed_everything(args.seed)
    config = _make_config(args)
    dataset_dir = resolve_dataset_dir()
    env, raw_train, raw_val = make_env_and_datasets(
        args.env_name,
        frame_stack=config['frame_stack'],
        seed=derive_seed(args.seed, 3),
        dataset_seed=derive_seed(args.seed, 1),
        dataset_dir=dataset_dir,
    )
    dataset_class = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset}[config['dataset_class']]
    train_dataset = dataset_class(raw_train, config, rng=derive_seed(args.seed, 11))
    val_dataset = dataset_class(raw_val, config, rng=derive_seed(args.seed, 12)) if raw_val is not None else None

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(args.seed, example_batch['observations'], example_batch['actions'], config)
    if args.restore_path is not None:
        if args.restore_epoch is None:
            raise ValueError('--restore_epoch is required with --restore_path')
        agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    run_dir = os.path.join(args.save_dir, get_exp_name(args.seed))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'runtime_metadata.json'), 'w') as file:
        json.dump(
            {
                'agent': args.agent,
                'environment': args.env_name,
                'dataset_dir': dataset_dir,
                'ogbench_module': os.path.abspath(__import__('ogbench').__file__),
                'seed': args.seed,
                'computation': bool(args.computation),
            },
            file,
            indent=2,
        )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(run_dir, 'eval.csv'))
    first_time = last_time = time.time()
    try:
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
                    run_dir,
                    step,
                    checkpoint_metadata={
                        'environment': args.env_name,
                        'dataset_dir': dataset_dir,
                        'computation': bool(args.computation),
                    },
                )
                del checkpoint_path
                goal_key = 'high_actor_goals' if 'high_actor_goals' in example_batch else 'actor_goals'
                _validate_checkpoint(
                    agent,
                    run_dir,
                    step,
                    example_batch['observations'],
                    example_batch[goal_key],
                    example_batch.get('actions'),
                )
    finally:
        train_logger.close()
        eval_logger.close()
        env.close()
    return run_dir


def main(argv=None):
    return run(_parse_args(argv))


if __name__ == '__main__':
    main()
