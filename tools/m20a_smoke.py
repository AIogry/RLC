"""Tiny real-data M20A smoke: two GCIQL updates per task, under /tmp only.

This is intentionally not a Study run.  It does not construct a RunContext,
does not write under the formal M20A run root, and must never be interpreted
as a performance experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from impls.agents.gciql import GCIQLAgent
from impls.experiment import prepare_run_design
from impls.main import _make_config, _parse_args
from impls.utils.datasets import GCDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent


STUDY_DEFAULT = 'experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml'
DATASET_ROOT_DEFAULT = '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
SMOKE_ROOT_DEFAULT = '/tmp/m20a_phase1_smoke'
CUBE_CANDIDATE = {
    'current_support_epsilon_xy': 0.02,
    'current_support_epsilon_z': 0.010,
    'goal_support_epsilon_xy': 0.02,
    'goal_support_epsilon_z': 0.010,
    'goal_conflict_radius': 0.04,
}
TASKS = {
    'puzzle': ('M20A-PUZZLE-C-M', 'puzzle-4x4-play-v0'),
    'cube': ('M20A-CUBE-C-M', 'cube-triple-play-v0'),
    'scene': ('M20A-SCENE-C-M', 'scene-play-v0'),
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
    temporary.replace(path)


def _config(study_path, config_id, *, cube_candidate):
    _, configuration = prepare_run_design(study_path, config_id)
    config = _make_config(_parse_args(['--agent', 'gciql']), configuration=configuration)
    if cube_candidate:
        for slot in config['compute'].values():
            if slot.get('structure') == 'cube_tokens':
                for key, value in CUBE_CANDIDATE.items():
                    slot['relation_kwargs'][key] = value
                slot['relation_phase2_blocked'] = False
        config['m20a_phase2_blocked_skeleton'] = False
    return config


def _tree_equal(left, right):
    flat_left = flatten_dict(left)
    flat_right = flatten_dict(right)
    if set(flat_left) != set(flat_right):
        return False
    return all(np.array_equal(np.asarray(flat_left[key]), np.asarray(flat_right[key])) for key in flat_left)


def _tree_changed(before, after):
    flat_before = flatten_dict(before)
    flat_after = flatten_dict(after)
    return any(
        not np.array_equal(np.asarray(flat_before[key]), np.asarray(flat_after[key]))
        for key in flat_before
    )


def _tree_allclose(left, right, *, rtol=1e-6, atol=1e-7):
    flat_left = flatten_dict(left)
    flat_right = flatten_dict(right)
    if set(flat_left) != set(flat_right):
        return False
    return all(
        np.allclose(
            np.asarray(flat_left[key]), np.asarray(flat_right[key]),
            rtol=rtol, atol=atol,
        )
        for key in flat_left
    )


def _target_update_valid(before_params, after_params, tau):
    old_target = before_params['modules_target_critic']
    new_online = after_params['modules_critic']
    expected = jax.tree_util.tree_map(
        lambda online, target: online * tau + target * (1.0 - tau),
        new_online,
        old_target,
    )
    # The update itself is JIT compiled; a separately evaluated NumPy/JAX
    # expression can differ by one float32 rounding unit, hence tolerance is
    # used only for this algebraic smoke invariant (not init parity).
    return _tree_allclose(expected, after_params['modules_target_critic'])


def run_task(task_name, config_id, environment, *, study_path, dataset_root, smoke_root):
    config = _config(study_path, config_id, cube_candidate=(task_name == 'cube'))
    env, raw_train, _ = make_env_and_datasets(
        environment,
        frame_stack=config['frame_stack'],
        seed=0,
        dataset_seed=0,
        dataset_dir=dataset_root,
    )
    try:
        dataset = GCDataset(raw_train, config, rng=np.random.default_rng(0))
        example = dataset.sample(2)
        agent = GCIQLAgent.create(0, example['observations'], example['actions'], config)
        initial_params = agent.network.params
        info_records = []
        target_update_checks = []
        relation_shape = None
        for _ in range(2):
            batch = dataset.sample(config['batch_size'])
            # ``target_update`` updates the fresh TrainState's parameter
            # mapping in place.  Materialize a leaf-wise copy so this smoke
            # comparison retains the genuine pre-update target values.
            before_update = jax.tree_util.tree_map(
                lambda value: np.array(value), agent.network.params
            )
            agent, info = agent.update(batch)
            target_update_checks.append(
                _target_update_valid(before_update, agent.network.params, config['tau'])
            )
            info_records.append({key: float(np.asarray(value)) for key, value in info.items()})
            relation = agent.network.select('actor')(
                batch['observations'][:2], batch['actor_goals'][:2], method='relation_diagnostic'
            )['relations']
            relation_shape = list(np.asarray(relation).shape)
        finite = all(
            np.isfinite(value)
            for record in info_records for value in record.values()
        )
        gradients_finite = all(
            np.isfinite(record[key])
            for record in info_records
            for key in ('grad/max', 'grad/min', 'grad/norm')
        )
        task_root = Path(smoke_root) / task_name
        checkpoint_path = save_agent(agent, task_root, 2)
        restored = restore_agent_from_checkpoint(agent, checkpoint_path)
        checkpoint_roundtrip = _tree_equal(agent.network.params, restored.network.params)
        return {
            'configuration_id': config_id,
            'environment': environment,
            'updates': 2,
            'all_losses_finite': bool(finite),
            'gradients_finite': bool(gradients_finite),
            'parameters_changed': bool(_tree_changed(initial_params, agent.network.params)),
            'target_update_valid': bool(all(target_update_checks)),
            'checkpoint_roundtrip': bool(checkpoint_roundtrip),
            'relation_shape': relation_shape,
            'last_update_info': info_records[-1],
            'evaluation_episode': 'not_run; smoke correctness is not a scientific performance result',
            'formal_run_root_used': False,
        }
    finally:
        env.close()
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=STUDY_DEFAULT)
    parser.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    parser.add_argument('--smoke-root', default=SMOKE_ROOT_DEFAULT)
    args = parser.parse_args(argv)
    root = Path(args.smoke_root).resolve()
    # Guard the caller from accidentally pointing this utility at a formal
    # Study root; /tmp is the required Phase-1 smoke location.
    if not str(root).startswith('/tmp/'):
        raise ValueError(f'M20A smoke root must be under /tmp, got {root}')
    result = {
        'schema': 'm20a_phase1_tiny_real_data_smoke_v1',
        'formal_training_started': False,
        'formal_m20a_run_artifact_created': False,
        'smoke_root': str(root),
        'tasks': {},
    }
    for task_name, (config_id, environment) in TASKS.items():
        print(f'[M20A smoke] {task_name}: two real GCIQL updates under {root}', flush=True)
        result['tasks'][task_name] = run_task(
            task_name,
            config_id,
            environment,
            study_path=args.study,
            dataset_root=args.dataset_root,
            smoke_root=root,
        )
    _write_json(root / 'm20a_phase1_smoke.json', result)
    print(f'M20A tiny smoke written: {root / "m20a_phase1_smoke.json"}')


if __name__ == '__main__':
    main()
