"""Run the bounded, real-data lifecycle smoke for every M22 configuration.

The smoke intentionally lives outside ``impls`` and is parameterized by a
Study and dataset root.  It constructs each canonical agent, samples one real
batch, performs two updates, evaluates one episode per task, and performs one
save/restore parity check for each algorithm.  It never creates formal M22
Run directories.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.agents import resolve_agent_class  # noqa: E402
from impls.computation.accounting import count_parameters  # noqa: E402
from impls.experiment import load_study, prepare_run_design  # noqa: E402
from impls.main import _evaluate_tasks, _make_config, _update_for_training_mode  # noqa: E402
from impls.utils.datasets import GCDataset, HGCDataset, MultiHGCDataset  # noqa: E402
from impls.utils.env_utils import make_env_and_datasets  # noqa: E402
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent  # noqa: E402
from impls.utils.reproducibility import derive_seed  # noqa: E402


STUDY_ID = 'M22'
ALGORITHMS = ('gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql')
ENVIRONMENTS = (
    'puzzle-3x3-play-v0',
    'puzzle-4x4-play-v0',
    'puzzle-4x5-play-v0',
    'puzzle-4x6-play-v0',
)
DEFAULT_STUDY = REPO_ROOT / 'experiments/M22_puzzle_baselines_unified_rlc/study.yaml'
DEFAULT_DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
DEFAULT_OUTPUT_ROOT = Path('/tmp/m22_puzzle_baseline_smoke')


def _resolved_args(algorithm: str) -> SimpleNamespace:
    return SimpleNamespace(
        agent=algorithm,
        batch_size=None,
        latent_dim=None,
        actor_loss=None,
        width=None,
        depth=None,
        computation=False,
    )


def _finite_tree(tree) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def _trees_differ(before, after) -> bool:
    leaves_before = jax.tree_util.tree_leaves(before)
    leaves_after = jax.tree_util.tree_leaves(after)
    return len(leaves_before) == len(leaves_after) and any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(leaves_before, leaves_after)
    )


def _dataset_class(config):
    return {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'MultiHGCDataset': MultiHGCDataset,
    }[config['dataset_class']]


def _smoke_args():
    return SimpleNamespace(
        eval_tasks='all',
        eval_episodes=1,
        video_episodes=0,
        eval_temperature=0.0,
        eval_gaussian=None,
    )


def _goal_for_action(batch, algorithm):
    return batch['high_actor_goals'] if algorithm == 'hiql' else batch['actor_goals']


def _check_action(agent, batch, algorithm, seed):
    observations = batch['observations'][:1]
    goals = _goal_for_action(batch, algorithm)[:1]
    action = agent.sample_actions(
        observations,
        goals,
        seed=jax.random.PRNGKey(seed),
        temperature=0.0,
    )
    array = np.asarray(action)
    if array.ndim != 2 or array.shape[0] != 1:
        raise AssertionError(f'Unexpected action shape: {array.shape}')
    if not np.all(np.isfinite(array)):
        raise AssertionError('Action contains non-finite values')
    if np.any(array < -1.000001) or np.any(array > 1.000001):
        raise AssertionError('Continuous action is outside [-1, 1]')
    return {'shape': list(array.shape), 'min': float(array.min()), 'max': float(array.max())}


def _run_config(configuration, env, raw_train, raw_val, *, output_root, checkpoint_representative):
    algorithm = configuration.data['algorithm']
    environment_name = configuration.data['environment']
    seed = int(configuration.data.get('smoke_seed', 0))
    config = _make_config(_resolved_args(algorithm), configuration)
    smoke_config = copy.deepcopy(config)
    smoke_config['batch_size'] = 4
    dataset_type = _dataset_class(smoke_config)
    train_dataset = dataset_type(
        raw_train,
        smoke_config,
        rng=derive_seed(seed, 11),
    )
    val_dataset = None if raw_val is None else dataset_type(
        raw_val,
        smoke_config,
        rng=derive_seed(seed, 12),
    )
    batch = train_dataset.sample(4)
    if smoke_config['discrete']:
        batch['actions'] = np.full_like(batch['actions'], env.action_space.n - 1)
    agent_class = resolve_agent_class(smoke_config['agent_name'], smoke_config.get('runtime_variant', 'canonical'))
    agent = agent_class.create(
        seed,
        batch['observations'],
        batch['actions'],
        smoke_config,
    )
    before_params = agent.network.params
    if not _finite_tree(before_params):
        raise AssertionError('Initial parameter tree is non-finite')
    updates = []
    for _ in range(2):
        agent, update_info = _update_for_training_mode(agent, batch, smoke_config)
        if not _finite_tree(update_info):
            raise AssertionError(f'Non-finite update info: {update_info}')
        updates.append({key: float(np.asarray(value)) for key, value in update_info.items() if np.asarray(value).size == 1})
    if not _trees_differ(before_params, agent.network.params):
        raise AssertionError('Parameters did not change after two optimizer updates')
    action = _check_action(agent, batch, algorithm, derive_seed(seed, 31))
    evaluation = _evaluate_tasks(
        agent,
        env,
        smoke_config,
        _smoke_args(),
        derive_seed(seed, 41),
    )
    if 'evaluation/overall_success' not in evaluation:
        raise AssertionError('Tiny evaluation did not produce evaluation/overall_success')
    overall_success = float(evaluation['evaluation/overall_success'])
    if not math.isfinite(overall_success) or not 0.0 <= overall_success <= 1.0:
        raise AssertionError(f'Invalid tiny evaluation success: {overall_success}')

    checkpoint_status = 'not_selected'
    if checkpoint_representative:
        checkpoint_dir = output_root / 'checkpoints' / algorithm
        checkpoint_path = save_agent(
            agent,
            checkpoint_dir,
            2,
            checkpoint_metadata={
                'smoke': True,
                'study_id': STUDY_ID,
                'config_id': configuration.config_id,
                'environment': environment_name,
                'algorithm': algorithm,
            },
        )
        restored = restore_agent_from_checkpoint(agent, checkpoint_path)
        observations = batch['observations'][:1]
        goals = _goal_for_action(batch, algorithm)[:1]
        key = jax.random.PRNGKey(derive_seed(seed, 51))
        before_action = np.asarray(agent.sample_actions(observations, goals, seed=key, temperature=0.0))
        after_action = np.asarray(restored.sample_actions(observations, goals, seed=key, temperature=0.0))
        np.testing.assert_array_equal(before_action, after_action)
        checkpoint_status = 'pass'

    return {
        'config_id': configuration.config_id,
        'algorithm': algorithm,
        'environment': environment_name,
        'status': 'pass',
        'dataset_class': smoke_config['dataset_class'],
        'observation_dim': int(batch['observations'].shape[-1]),
        'action_dim': int(batch['actions'].shape[-1]),
        'batch_size': 4,
        'optimizer_updates': 2,
        'parameters_changed': True,
        'trainable_parameter_count': count_parameters(agent.network.params),
        'action': action,
        'evaluation_task_count': len(getattr(env.unwrapped, 'task_infos', ()) or ()),
        'evaluation_episodes_per_task': 1,
        'evaluation_overall_success': overall_success,
        'checkpoint_smoke': checkpoint_status,
        'update_scalars': updates,
    }


def run_smoke(*, study_path=DEFAULT_STUDY, dataset_root=DEFAULT_DATASET_ROOT, output_root=DEFAULT_OUTPUT_ROOT):
    study = load_study(study_path)
    configurations = [
        prepare_run_design(study_path, path)[1]
        for path in sorted((study.path.parent / 'configs').glob('*.yaml'))
    ]
    grouped = {environment: [] for environment in ENVIRONMENTS}
    for configuration in configurations:
        grouped[configuration.data['environment']].append(configuration)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for environment in ENVIRONMENTS:
        env = raw_train = raw_val = None
        try:
            env, raw_train, raw_val = make_env_and_datasets(
                environment,
                frame_stack=None,
                seed=derive_seed(0, 3),
                dataset_seed=derive_seed(0, 1),
                dataset_dir=dataset_root,
            )
            for configuration in sorted(grouped[environment], key=lambda item: item.config_id):
                try:
                    records.append(_run_config(
                        configuration,
                        env=env,
                        raw_train=raw_train,
                        raw_val=raw_val,
                        output_root=output_root,
                        checkpoint_representative=configuration.data['algorithm'] in ALGORITHMS and configuration.config_id.endswith('P3X3'),
                    ))
                except Exception as error:  # report all config failures in one invocation
                    records.append({
                        'config_id': configuration.config_id,
                        'algorithm': configuration.data['algorithm'],
                        'environment': environment,
                        'status': 'fail',
                        'error': f'{type(error).__name__}: {error}',
                    })
        finally:
            if env is not None:
                env.close()
            del raw_train, raw_val, env
            gc.collect()
    report = {
        'schema': 'm22_real_data_smoke_v1',
        'study_id': STUDY_ID,
        'study_path': str(Path(study_path).resolve()),
        'dataset_root': str(Path(dataset_root).resolve()),
        'formal_training_started': False,
        'config_count': len(records),
        'expected_config_count': 24,
        'passed_config_count': sum(record['status'] == 'pass' for record in records),
        'checkpoint_representatives': list(ALGORITHMS),
        'records': records,
        'status': 'pass' if len(records) == 24 and all(record['status'] == 'pass' for record in records) else 'fail',
    }
    (output_root / 'smoke_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    report = run_smoke(
        study_path=args.study,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(
        f'M22 SMOKE: {report["status"].upper()} '
        f'configs={report["passed_config_count"]}/{report["expected_config_count"]}'
    )
    for record in report['records']:
        if record['status'] != 'pass':
            print(f'  - {record["config_id"]}: {record.get("error", "failed")}')
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
