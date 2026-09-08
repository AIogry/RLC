"""Non-executing M20B audit, launch doctor, and real-data smoke runner.

The tool owns reusable source/data/configuration checks for the Cube
Entity-Structured Mixer transfer screen.  The normal doctor never creates a
formal run directory and never starts training.  ``--smoke`` is deliberately
isolated under a caller-selected diagnostic root and performs only two update
steps per one of the six seed-0 configurations.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from numpy.lib import format as npy_format

from impls.agents.gciql import GCIQLAgent
from impls.computation.accounting import gciql_architecture_accounting
from impls.experiment import (
    load_configuration,
    load_study,
    make_run_path,
    prepare_run_design,
)
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.representation.manipulation import CubeTokenAdapter
from impls.utils.datasets import Dataset, GCDataset
from impls.utils.evaluation import evaluate_episodes
from impls.utils.flax_utils import restore_agent_from_checkpoint, save_agent
from impls.utils.reproducibility import derive_seed


STUDY_ID = 'M20B'
STUDY_DEFAULT = 'experiments/M20B_cube_entity_mixer_scaling/study.yaml'
AUDIT_DEFAULT = 'docs/9-8/M20B_source_audit.json'
REPORT_DEFAULT = 'docs/9-8/M20B_doctor.json'
SMOKE_DEFAULT = '/tmp/m20b_cube_entity_mixer_smoke'
RUN_ROOT_DEFAULT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'
DATASET_ROOT_DEFAULT = os.environ.get('OGBENCH_DATASET_DIR', '')

ENV_SPECS = {
    'cube-single-play-v0': {
        'num_cubes': 1,
        'obs_dim': 28,
        'train_episodes': 1000,
        'val_episodes': 100,
        'tasks': [
            'task1_horizontal', 'task2_vertical1', 'task3_vertical2',
            'task4_diagonal1', 'task5_diagonal2',
        ],
    },
    'cube-double-play-v0': {
        'num_cubes': 2,
        'obs_dim': 37,
        'train_episodes': 1000,
        'val_episodes': 100,
        'tasks': [
            'task1_single_pnp', 'task2_double_pnp1', 'task3_double_pnp2',
            'task4_swap', 'task5_stack',
        ],
    },
    'cube-triple-play-v0': {
        'num_cubes': 3,
        'obs_dim': 46,
        'train_episodes': 3000,
        'val_episodes': 300,
        'tasks': [
            'task1_single_pnp', 'task2_triple_pnp', 'task3_pnp_from_stack',
            'task4_cycle', 'task5_stack',
        ],
    },
}

EXPECTED_CONFIG_IDS = {
    'M20B-single-B000', 'M20B-single-S002',
    'M20B-double-B000', 'M20B-double-S002',
    'M20B-triple-B000', 'M20B-triple-S002',
}
CONDITION_FOR_CONFIG = {
    config_id: config_id.rsplit('-', 1)[-1]
    for config_id in EXPECTED_CONFIG_IDS
}
BASE_AGENT_KEYS = (
    'actor_loss', 'alpha', 'actor_hidden_dims', 'value_hidden_dims',
    'layer_norm', 'lr', 'batch_size', 'discount', 'expectile', 'tau',
    'const_std', 'discrete', 'encoder', 'dataset_class',
    'value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal',
    'value_geom_sample', 'actor_p_curgoal', 'actor_p_trajgoal',
    'actor_p_randomgoal', 'actor_geom_sample', 'gc_negative', 'p_aug',
    'frame_stack',
)
GOAL_SAMPLING_KEYS = (
    'value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal',
    'value_geom_sample', 'actor_p_curgoal', 'actor_p_trajgoal',
    'actor_p_randomgoal', 'actor_geom_sample', 'gc_negative',
)
PROTOCOL_KEYS = {
    'train_steps': 1_000_000,
    'batch_size': 1024,
    'log_interval': 5000,
    'eval_interval': 100_000,
    'eval_tasks': 'all',
    'eval_episodes': 50,
    'eval_temperature': 0.0,
    'eval_gaussian': None,
    'video_episodes': 0,
    'save_interval': 100_000,
    'save_best_checkpoint': True,
    'save_last_checkpoint': True,
    'primary_endpoint': 'final@1M',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(_jsonable(value), file, indent=2, sort_keys=True)
        file.write('\n')
    temporary.replace(path)


def _source_provenance():
    observed_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, check=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        capture_output=True, check=True, text=True,
    ).stdout.splitlines()
    return {
        'head': observed_head,
        'worktree_clean': not status,
        'dirty_files': status,
        'git_policy': 'read_only_audit',
    }


def _agent_args():
    return _parse_args(['--agent', 'gciql'])


def _resolved_configuration(study_path, config_id):
    _, configuration = prepare_run_design(study_path, config_id)
    return configuration, _make_config(_agent_args(), configuration=configuration)


def _configuration_paths(study):
    return sorted((Path(study.path).parent / 'configs').glob('*.yaml'))


def _check_protocol(protocol, label):
    for key, expected in PROTOCOL_KEYS.items():
        _require(protocol.get(key) == expected, f'{label}: protocol {key}={expected!r}')
    auc = protocol.get('auc', {})
    _require(
        auc.get('checkpoints') == list(range(100_000, 1_000_001, 100_000)),
        f'{label}: AUC checkpoints',
    )
    _require(auc.get('interval') == [100_000, 1_000_000], f'{label}: AUC interval')
    _require(
        auc.get('rule') == 'trapezoidal_area_divided_by_900000',
        f'{label}: AUC rule',
    )


def _check_base_agent(agent, label):
    _require(agent.get('alpha') == 1.0, f'{label}: alpha must be explicit 1.0')
    expected = {
        'actor_loss': 'ddpgbc',
        'actor_hidden_dims': [512, 512, 512],
        'value_hidden_dims': [512, 512, 512],
        'layer_norm': True,
        'lr': 0.0003,
        'batch_size': 1024,
        'discount': 0.99,
        'expectile': 0.9,
        'tau': 0.005,
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
    }
    for key, value in expected.items():
        _require(agent.get(key) == value, f'{label}: agent {key}={value!r}')


def _check_relation_free_slot(slot, *, num_cubes, label):
    _require(slot.get('enabled') is True, f'{label}: S002 slot must be enabled')
    for key, expected in (
        ('structure', 'cube_tokens'), ('primitive', 'mlp'),
        ('block', 'mlp_mixer'), ('topology', 'feedforward'),
        ('parameter_sharing', 'shared'), ('credit', 'direct'),
        ('relation_mode', 'legacy_none'), ('relation_augmenter', 'none'),
        ('readout', 'mean_context'),
    ):
        _require(slot.get(key) == expected, f'{label}: {key}={expected!r}')
    _require(slot.get('topology_kwargs', {}) == {}, f'{label}: topology_kwargs must be empty')
    _require(slot.get('relation_kwargs', {}) == {}, f'{label}: relation_kwargs must be empty')
    _require(
        slot.get('relation_augmenter_kwargs', {}) == {},
        f'{label}: relation_augmenter_kwargs must be empty',
    )
    structure = slot.get('structure_kwargs', {})
    _require(
        set(structure) == {
            'num_cubes', 'robot_dim', 'cube_feature_dim', 'token_dim',
            'robot_hidden_dim', 'slot_identity_embedding',
        },
        f'{label}: exact relation-free structure keys',
    )
    _require(structure.get('num_cubes') == num_cubes, f'{label}: num_cubes')
    for key, expected in (
        ('robot_dim', 19), ('cube_feature_dim', 9),
        ('token_dim', 128), ('robot_hidden_dim', 128),
    ):
        _require(structure.get(key) == expected, f'{label}: structure {key}')
    _require(structure.get('slot_identity_embedding') is False, f'{label}: slot identity')
    block = slot.get('block_kwargs', {})
    _require(
        set(block) == {'num_blocks', 'token_hidden_dim', 'channel_hidden_dim', 'tm_mode'},
        f'{label}: exact Mixer keys',
    )
    for key, expected in (
        ('num_blocks', 2), ('token_hidden_dim', 64),
        ('channel_hidden_dim', 256), ('tm_mode', 'none'),
    ):
        _require(block.get(key) == expected, f'{label}: Mixer {key}')
    readout_kwargs = slot.get('readout_kwargs', {})
    _require(set(readout_kwargs) <= {'output_dim'}, f'{label}: readout keys')
    _require(readout_kwargs.get('output_dim') == 512, f'{label}: readout output_dim')


def check_study_schema(study_path):
    study = load_study(study_path)
    _require(study.study_id == STUDY_ID, f'Expected study_id={STUDY_ID!r}')
    _require(study.data.get('study_type') == 'seed0_screening', 'Study type must be seed0_screening')
    _require(study.data.get('seeds') == [0], 'Study must contain exactly seed 0')
    _require(
        study.data.get('environments') == list(ENV_SPECS),
        f'Study environments must be {list(ENV_SPECS)!r}',
    )
    metadata = study.data.get('metadata', {})
    for key, expected in (
        ('primary_contrast', 'cube-double S002 minus B000'),
        ('relation_study', False), ('parameter_matched', False),
        ('entity_count_causal_scaling', False),
    ):
        _require(metadata.get(key) == expected, f'Study metadata {key}')
    fixed = study.data.get('fixed_design', {})
    _require(fixed.get('algorithm') == 'gciql', 'Study algorithm must be gciql')
    _require(fixed.get('alpha') == 1.0, 'Fixed design alpha must be 1.0')
    _require(fixed.get('training_seed') == 0, 'Fixed design training seed')
    _require(study.data.get('protocol', {}).get('eval_episodes') == 50, 'Study eval episodes')

    paths = _configuration_paths(study)
    _require(len(paths) == 6, f'Expected exactly six config files, found {len(paths)}')
    configurations = {}
    for path in paths:
        configuration = load_configuration(study, path)
        data = configuration.data
        config_id = configuration.config_id
        _require(config_id in EXPECTED_CONFIG_IDS, f'Unexpected config_id {config_id!r}')
        _require(config_id not in configurations, f'Duplicate config_id {config_id!r}')
        configurations[config_id] = data
        environment = data.get('environment')
        _require(environment in ENV_SPECS, f'{config_id}: unsupported environment')
        _require(data.get('training_seed') == 0, f'{config_id}: training seed')
        _require(data.get('executable') is True, f'{config_id}: executable flag')
        _require(data.get('factors', {}).get('alpha') == 1.0, f'{config_id}: factor alpha')
        _check_protocol(data.get('frozen_protocol', {}), config_id)
        agent = data.get('agent_overrides', {})
        _check_base_agent(agent, config_id)
        compute = agent.get('compute', {})
        _require(set(compute) == {'actor', 'value', 'critic'}, f'{config_id}: exact GCIQL slots')
        condition = CONDITION_FOR_CONFIG[config_id]
        _require(data.get('condition_id') == condition, f'{config_id}: condition id')
        if condition == 'B000':
            for slot_name, slot in compute.items():
                _require(slot.get('enabled') is False, f'{config_id}.{slot_name}: B000 enabled')
                _require(slot.get('structure', 'vector') == 'vector', f'{config_id}.{slot_name}: B000 structure')
        else:
            for slot_name, slot in compute.items():
                _check_relation_free_slot(
                    slot,
                    num_cubes=ENV_SPECS[environment]['num_cubes'],
                    label=f'{config_id}.{slot_name}',
                )

    _require(set(configurations) == EXPECTED_CONFIG_IDS, 'Configuration ID set mismatch')
    pair_audit = {}
    for environment in ENV_SPECS:
        pair = [
            data for data in configurations.values()
            if data.get('environment') == environment
        ]
        _require(len(pair) == 2, f'{environment}: expected B000/S002 pair')
        base_views = []
        protocol_views = []
        for data in pair:
            agent = copy.deepcopy(_jsonable(data['agent_overrides']))
            agent.pop('compute', None)
            base_views.append(agent)
            protocol_views.append(_jsonable(data['frozen_protocol']))
        _require(base_views[0] == base_views[1], f'{environment}: non-architecture agent protocol drift')
        _require(protocol_views[0] == protocol_views[1], f'{environment}: eval protocol drift')
        pair_audit[environment] = {
            'dataset_identity_same_by_environment_key': True,
            'dataset_sampling_seed_same': True,
            'goal_sampling_same': True,
            'evaluation_seed_schedule_same': True,
            'alpha_same': True,
            'optimizer_protocol_same': True,
            'eval_protocol_same': True,
            'only_intended_difference': 'compute architecture package',
        }
    return {
        'study_id': study.study_id,
        'config_count': len(configurations),
        'config_ids': sorted(configurations),
        'environments': list(ENV_SPECS),
        'within_environment_protocol_identity': pair_audit,
    }


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _npz_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith('.npy'):
                continue
            with archive.open(member) as file:
                version = npy_format.read_magic(file)
                if version == (1, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_1_0(file)
                elif version == (2, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_2_0(file)
                elif version == (3, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_3_0(file)
                else:  # pragma: no cover - NumPy currently supports these versions.
                    raise ValueError(f'Unsupported NPY header version {version!r}: {member}')
            result[Path(member).stem] = {
                'shape': list(shape),
                'dtype': np.dtype(dtype).str,
                'fortran_order': bool(fortran_order),
            }
    return result


def _inspect_dataset_file(path, *, expected_episodes):
    path = Path(path)
    _require(path.is_file(), f'Missing dataset file: {path}')
    headers = _npz_headers(path)
    for key in ('observations', 'actions', 'terminals'):
        _require(key in headers, f'{path}: missing array {key}')
    with np.load(path, allow_pickle=False) as arrays:
        terminals = np.asarray(arrays['terminals'])
        terminal_count = int(np.count_nonzero(terminals > 0))
        transition_count = int(terminals.shape[0])
        last_terminal = int(np.flatnonzero(terminals > 0)[-1]) if terminal_count else None
    _require(terminal_count == expected_episodes, f'{path}: episode count')
    _require(last_terminal == transition_count - 1, f'{path}: final terminal marker')
    return {
        'path': str(path.resolve()),
        'sha256': _sha256_file(path),
        'bytes': int(path.stat().st_size),
        'arrays': headers,
        'episodes': terminal_count,
        'transitions': transition_count,
        'last_terminal_index': last_terminal,
    }


def _make_env(environment, *, dataset_root):
    import ogbench

    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.filterwarnings(
            'ignore',
            message=".*Box (low|high)'s precision lowered.*",
            category=UserWarning,
        )
        env = ogbench.make_env_and_datasets(
            environment,
            dataset_dir=str(dataset_root),
            env_only=True,
        )
    return env


def _environment_audit(environment, *, dataset_root):
    expected = ENV_SPECS[environment]
    train_path = Path(dataset_root) / f'{environment}.npz'
    val_path = Path(dataset_root) / f'{environment}-val.npz'
    train = _inspect_dataset_file(train_path, expected_episodes=expected['train_episodes'])
    validation = _inspect_dataset_file(val_path, expected_episodes=expected['val_episodes'])
    env = _make_env(environment, dataset_root=dataset_root)
    try:
        observation, _ = env.reset(seed=20_020)
        task_infos = list(getattr(env.unwrapped, 'task_infos', ()))
        task_names = [str(item['task_name']) for item in task_infos]
        observed_obs_dim = int(np.asarray(observation).shape[-1])
        action_shape = list(getattr(env.action_space, 'shape', ()))
        _require(observed_obs_dim == expected['obs_dim'], f'{environment}: observation dimension')
        _require(task_names == expected['tasks'], f'{environment}: task names')
        _require(action_shape == [5], f'{environment}: action shape')
    finally:
        env.close()
    return {
        'environment': environment,
        'num_cubes': expected['num_cubes'],
        'observation_dim': observed_obs_dim,
        'action_shape': action_shape,
        'task_names': task_names,
        'train': train,
        'validation': validation,
    }


def _cube_reset_permutation_audit():
    from ogbench.manipspace.envs.cube_env import CubeEnv

    source_path = Path(inspect.getsourcefile(CubeEnv)).resolve()
    lines = source_path.read_text().splitlines()
    required_fragments = {
        'permutation_draw': 'permutation = self.np_random.permutation(self._num_cubes)',
        'initial_pair_indexing': "init_xyzs = self.cur_task_info['init_xyzs'].copy()[permutation]",
        'goal_pair_indexing': "goal_xyzs = self.cur_task_info['goal_xyzs'].copy()[permutation]",
    }
    line_refs = {}
    for name, fragment in required_fragments.items():
        matches = [index + 1 for index, line in enumerate(lines) if fragment in line]
        _require(matches, f'CubeEnv source audit missing {fragment!r}')
        line_refs[name] = matches
    source_text = '\n'.join(lines)
    forbidden_canonicalization = (
        'argsort(', 'sort(', 'lexsort(', 'DeepSets', 'PermutationInvariant',
    )
    return {
        'source_file': str(source_path),
        'source_sha256': _sha256_file(source_path),
        'line_references': line_refs,
        'same_permutation_applied_to_init_and_goal': True,
        'cube_i_goal_cube_i_pairing_preserved': True,
        'slot_identity_embedding': False,
        'standard_mlp_mixer_not_permutation_equivariant': True,
        'canonicalization_fragments_absent': {
            fragment: fragment not in source_text
            for fragment in forbidden_canonicalization
        },
    }


def audit_source_and_data(*, dataset_root, study_path):
    study = load_study(study_path)
    environments = {
        environment: _environment_audit(environment, dataset_root=dataset_root)
        for environment in study.data['environments']
    }
    return {
        'audit_version': 'M20B-source-audit-v1',
        'study_id': STUDY_ID,
        'source': _source_provenance(),
        'dataset_root': str(Path(dataset_root).resolve()),
        'environments': environments,
        'cube_reset_permutation': _cube_reset_permutation_audit(),
        'relation_free_design': {
            'structure': 'cube_tokens',
            'relation_mode': 'legacy_none',
            'relation_augmenter': 'none',
            'relation_kwargs': {},
            'relation_augmenter_kwargs': {},
            'relation_params': 0,
            'relation_macs': 0,
        },
    }


def _path_get(tree, path):
    for key in path:
        if not hasattr(tree, 'get') or key not in tree:
            return None
        tree = tree[key]
    return tree


def _tree_paths(tree):
    return [tuple(path) for path in flatten_dict(tree).keys()]


def _contains_path_fragment(paths, fragment):
    return [path for path in paths if fragment in '/'.join(map(str, path))]


def _adapter_relation_contract(*, num_cubes, slot_name):
    """Exercise the source-audited Cube adapter's relation-free output."""

    action_semantics = 'robot_context' if slot_name == 'critic' else 'none'
    adapter = CubeTokenAdapter(
        num_cubes=num_cubes,
        robot_dim=19,
        cube_feature_dim=9,
        token_dim=128,
        robot_hidden_dim=128,
        slot_identity_embedding=False,
        input_semantics='goal_pair',
        action_semantics=action_semantics,
        layer_norm=True,
        relation_mode='legacy_none',
        relation_kwargs={},
    )
    observation_dim = 19 + 9 * num_cubes
    input_width = 2 * observation_dim + (5 if action_semantics == 'robot_context' else 0)
    inputs = jnp.zeros((2, input_width), dtype=jnp.float32)
    variables = adapter.init(jax.random.PRNGKey(31 + num_cubes), inputs)
    representation = adapter.apply(variables, inputs)
    _require(representation.relations is None, f'{slot_name}: adapter relations must be None')
    return {
        'relations_is_none': True,
        'tokens_shape': list(np.asarray(representation.tokens).shape),
        'context_shape': list(np.asarray(representation.context).shape),
    }


def _runtime_parameter_audit(agent, config, *, environment, condition):
    reports = _computation_slot_accounting(agent, config)
    architecture = gciql_architecture_accounting(agent.network.params, config, reports)
    params = agent.network.params
    paths = _tree_paths(params)
    obs_dim = ENV_SPECS[environment]['obs_dim']
    observations = jnp.zeros((2, obs_dim), dtype=jnp.float32)
    goals = jnp.zeros_like(observations)
    actions = jnp.zeros((2, 5), dtype=jnp.float32)
    actor_distribution = agent.network.select('actor')(observations, goals)
    actor_action = np.asarray(actor_distribution.mode())
    unbatched_distribution = agent.network.select('actor')(observations[0], goals[0])
    unbatched_action = np.asarray(unbatched_distribution.mode())
    value = np.asarray(agent.network.select('value')(observations, goals))
    q1, q2 = agent.network.select('critic')(observations, goals, actions)
    target_q1, target_q2 = agent.network.select('target_critic')(observations, goals, actions)
    all_forward_finite = all(
        np.all(np.isfinite(value))
        for value in (actor_action, unbatched_action, value, np.asarray(q1), np.asarray(q2), np.asarray(target_q1), np.asarray(target_q2))
    )
    _require(all_forward_finite, f'{environment}/{condition}: non-finite forward')
    _require(unbatched_action.shape == (5,), f'{environment}/{condition}: unbatched action shape')
    _require(np.all(np.abs(unbatched_action) <= 1.0 + 1e-6), f'{environment}/{condition}: unclipped action')

    if condition == 'B000':
        _require(not reports, f'{environment}/B000: disabled slots produced accounting')
        return {
            'condition': condition,
            'forward_finite': True,
            'unbatched_action_shape': list(unbatched_action.shape),
            'structured_slots': {},
            'architecture': architecture,
            'relation_absent': True,
        }

    expected_n = ENV_SPECS[environment]['num_cubes']
    adapter_contract = {
        slot_name: _adapter_relation_contract(num_cubes=expected_n, slot_name=slot_name)
        for slot_name in ('actor', 'value', 'critic')
    }
    structured = {}
    banned_runtime_fragments = (
        'relation_augmenter', 'relations', 'tm_weights', 'cube_slot_embedding',
    )
    absent_paths = {
        fragment: not _contains_path_fragment(paths, fragment)
        for fragment in banned_runtime_fragments
    }
    _require(all(absent_paths.values()), f'{environment}/{condition}: relation/slot parameters present')
    for slot_name in ('actor', 'value', 'critic'):
        item = reports[slot_name]
        for key, expected in (
            ('structure', 'cube_tokens'), ('relation_mode', 'legacy_none'),
            ('relation_augmenter', 'none'), ('relation_params', 0),
            ('relation_parameters', 0), ('relation_augmenter_params', 0),
            ('relation_macs', 0), ('relation_dense_macs', 0),
            ('relation_projection_dense_macs', 0),
            ('executed_dense_relation_aggregation_cost', 0),
            ('num_tokens', expected_n), ('num_cubes', expected_n),
            ('token_dim', 128), ('token_hidden_dim', 64),
            ('channel_hidden_dim', 256), ('block_depth_L', 2),
            ('tm_mode', 'none'), ('block_type', 'mlp_mixer'),
            ('readout', 'mean_context'), ('index_embedding', False),
        ):
            _require(item.get(key) == expected, f'{environment}/{condition}/{slot_name}: {key}')
        _require(item.get('cross_entity_interaction') is (expected_n > 1), f'{environment}/{condition}: cross-entity flag')
        if slot_name == 'actor':
            body_path = ('modules_actor', 'actor_net')
        else:
            body_path = (f'modules_{slot_name}', 'value_net', 'core')
        body = _path_get(params, body_path)
        _require(body is not None, f'{environment}/{condition}/{slot_name}: body path')
        _require('relation_augmenter' not in body, f'{environment}/{condition}/{slot_name}: augmenter tree')
        token_kernel = _path_get(
            body,
            ('core', 'topology', 'primitive', 'blocks_0', 'token_dense2', 'kernel')
            if slot_name == 'actor'
            else ('core', 'topology', 'primitive', 'blocks_0', 'token_dense2', 'kernel'),
        )
        _require(getattr(token_kernel, 'shape', ())[-1] == expected_n, f'{environment}/{condition}/{slot_name}: Mixer T')
        structured[slot_name] = {
            'num_tokens': int(item['num_tokens']),
            'mixer_token_dense2_kernel_shape': list(token_kernel.shape),
            'relation_params': 0,
            'relation_macs': 0,
            'relation_augmenter_absent': True,
            'slot_identity_embedding_absent': True,
            'forward_finite': True,
        }
    return {
        'condition': condition,
        'forward_finite': True,
        'unbatched_action_shape': list(unbatched_action.shape),
        'structured_slots': structured,
        'adapter_relation_contract': adapter_contract,
        'architecture': architecture,
        'relation_absent': absent_paths,
    }


def _runtime_initialize_audit(study_path):
    result = {}
    for config_id in sorted(EXPECTED_CONFIG_IDS):
        configuration, config = _resolved_configuration(study_path, config_id)
        environment = configuration.data['environment']
        obs_dim = ENV_SPECS[environment]['obs_dim']
        observations = jnp.zeros((2, obs_dim), dtype=jnp.float32)
        actions = jnp.zeros((2, 5), dtype=jnp.float32)
        agent = GCIQLAgent.create(0, observations, actions, config)
        result[config_id] = _runtime_parameter_audit(
            agent,
            config,
            environment=environment,
            condition=configuration.data['condition_id'],
        )
    return result


def paired_sampling_audit(study_path, *, dataset_root, sample_count=1024, seed=20_020):
    result = {}
    for environment in ENV_SPECS:
        path = Path(dataset_root) / f'{environment}.npz'
        with np.load(path, allow_pickle=False) as arrays:
            raw = {
                key: arrays[key]
                for key in ('observations', 'actions', 'terminals')
            }
            dataset = Dataset.create(freeze=True, **raw)
        short_name = environment.removeprefix('cube-').removesuffix('-play-v0')
        b000_configuration, b000_config = _resolved_configuration(
            study_path,
            f'M20B-{short_name}-B000',
        )
        s002_configuration, s002_config = _resolved_configuration(
            study_path,
            f'M20B-{short_name}-S002',
        )
        _require(
            b000_configuration.data['environment'] == s002_configuration.data['environment'],
            f'{environment}: paired environment identity',
        )
        left = GCDataset(dataset, b000_config, rng=np.random.default_rng(seed))
        right = GCDataset(dataset, s002_config, rng=np.random.default_rng(seed))
        left_batch, left_trace = left.sample(
            sample_count,
            rng=np.random.default_rng(seed),
            return_sampling_trace=True,
        )
        right_batch, right_trace = right.sample(
            sample_count,
            rng=np.random.default_rng(seed),
            return_sampling_trace=True,
        )
        trace_equal = all(
            np.array_equal(left_trace[key], right_trace[key])
            for key in left_trace
        )
        compared_keys = sorted(set(left_batch) & set(right_batch))
        batch_arrays_equal = all(
            np.array_equal(np.asarray(left_batch[key]), np.asarray(right_batch[key]))
            for key in compared_keys
        )
        _require(trace_equal and batch_arrays_equal, f'{environment}: paired sampling mismatch')
        result[environment] = {
            'sample_count': int(sample_count),
            'seed': int(seed),
            'trace_ids_equal': trace_equal,
            'batch_arrays_equal': batch_arrays_equal,
            'trace_keys': sorted(left_trace),
            'compared_batch_keys': compared_keys,
            'dataset_identity_same_by_path': True,
        }
        del left, right, dataset, raw
    return result


def _formal_artifact_audit(study_path, run_root):
    study = load_study(study_path)
    study_root = Path(run_root) / STUDY_ID
    _require(not study_root.exists(), f'Existing M20B formal artifact: {study_root}')
    paths = []
    for config_id in sorted(EXPECTED_CONFIG_IDS):
        configuration = load_configuration(study, config_id)
        environment = configuration.data['environment']
        run_path = make_run_path(
            run_root,
            STUDY_ID,
            config_id,
            configuration.slug,
            environment,
            0,
        )
        _require(run_path not in paths, f'Duplicate planned run path: {run_path}')
        _require(not run_path.exists(), f'Existing M20B planned artifact: {run_path}')
        paths.append(run_path)
    return {
        'formal_study_root_absent': True,
        'checked_study_root': str(study_root),
        'planned_run_count': len(paths),
        'planned_run_paths': [str(path) for path in paths],
        'completed': 0,
        'failed': 0,
        'running': 0,
        'remaining': len(paths),
    }


def _finite_tree(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def _stable_label_seed(label):
    """Map a textual configuration label to a reproducible uint32 seed."""

    digest = hashlib.sha256(str(label).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little', signed=False)


def _tree_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left_leaves, right_leaves))


def _tree_allclose(left, right, *, rtol=1e-6, atol=1e-6):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
        for a, b in zip(left_leaves, right_leaves)
    )


def _tree_changed(left, right):
    return not _tree_equal(left, right)


def _load_smoke_dataset(environment, dataset_root):
    import ogbench

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, train_data, _ = ogbench.make_env_and_datasets(
            environment,
            dataset_dir=str(dataset_root),
            compact_dataset=True,
        )
    return Dataset.create(seed=20_020, **train_data)


def _smoke_one(configuration, config, dataset, env, smoke_dir, *, batch_size=8):
    environment = configuration.data['environment']
    obs_dim = ENV_SPECS[environment]['obs_dim']
    example_observations = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    example_actions = jnp.zeros((batch_size, 5), dtype=jnp.float32)
    agent = GCIQLAgent.create(0, example_observations, example_actions, config)
    initial_params = jax.tree_util.tree_map(lambda value: jnp.array(value), agent.network.params)
    expected_target = jax.tree_util.tree_map(
        lambda value: jnp.array(value),
        agent.network.params['modules_target_critic'],
    )
    gcdataset = GCDataset(
        dataset,
        config,
        rng=np.random.default_rng(derive_seed(20_020, _stable_label_seed(configuration.config_id))),
    )
    update_infos = []
    updated = agent
    for update_index in range(2):
        batch = gcdataset.sample(
            batch_size,
            rng=np.random.default_rng(
                derive_seed(20_020, _stable_label_seed(configuration.config_id), update_index)
            ),
        )
        updated, info = updated.update(batch)
        expected_target = jax.tree_util.tree_map(
            lambda online, old: config['tau'] * online + (1.0 - config['tau']) * old,
            updated.network.params['modules_critic'],
            expected_target,
        )
        update_infos.append({
            key: float(np.asarray(value))
            for key, value in info.items()
            if np.asarray(value).size == 1
        })
    _require(all(np.isfinite(value) for item in update_infos for value in item.values()), f'{configuration.config_id}: non-finite smoke info')
    _require(_tree_changed(initial_params, updated.network.params), f'{configuration.config_id}: params unchanged')
    target_update_valid = _tree_allclose(
        expected_target,
        updated.network.params['modules_target_critic'],
    )
    _require(target_update_valid, f'{configuration.config_id}: target update mismatch')
    observations = jnp.zeros((2, obs_dim), dtype=jnp.float32)
    goals = jnp.zeros_like(observations)
    actions = jnp.zeros((2, 5), dtype=jnp.float32)
    target_outputs = updated.network.select('target_critic')(observations, goals, actions)
    _require(all(np.all(np.isfinite(np.asarray(value))) for value in target_outputs), f'{configuration.config_id}: target critic non-finite')

    checkpoint_dir = Path(smoke_dir) / configuration.config_id / 'checkpoints'
    checkpoint_path = save_agent(updated, checkpoint_dir, 2, checkpoint_metadata={
        'study_id': STUDY_ID,
        'config_id': configuration.config_id,
        'formal_training_started': False,
        'smoke_updates': 2,
    })
    restored = restore_agent_from_checkpoint(agent, checkpoint_path)
    restored_action = np.asarray(
        restored.network.select('actor')(observations, goals).mode()
    )
    updated_action = np.asarray(
        updated.network.select('actor')(observations, goals).mode()
    )
    checkpoint_roundtrip = np.array_equal(restored_action, updated_action) and _tree_equal(
        restored.network.params,
        updated.network.params,
    )
    _require(checkpoint_roundtrip, f'{configuration.config_id}: checkpoint roundtrip mismatch')

    task_results = {}
    for task_id, task_name in enumerate(ENV_SPECS[environment]['tasks'], start=1):
        try:
            records = evaluate_episodes(
                updated,
                env,
                task_id=task_id,
                task_name=task_name,
                config=config,
                evaluation_seed=derive_seed(20_020, 4),
                episode_indices=[0],
                eval_temperature=0.0,
                eval_gaussian=None,
            )
            task_results[task_name] = {
                'episodes': len(records),
                'success': records[0].get('success') if records else None,
                'evaluation_completed': True,
            }
        except Exception as error:  # Evaluation is explicitly optional when impractical.
            task_results[task_name] = {
                'evaluation_completed': False,
                'skipped_reason': f'{type(error).__name__}: {error}',
            }
    return {
        'environment': environment,
        'condition': configuration.data['condition_id'],
        'updates': 2,
        'batch_size': batch_size,
        'all_losses_finite': True,
        'gradients_finite': all(
            np.isfinite(item.get(key, np.nan))
            for item in update_infos
            for key in ('grad/max', 'grad/min', 'grad/norm')
        ),
        'parameters_changed': True,
        'target_update_valid': True,
        'target_critic_forward_finite': True,
        'checkpoint_roundtrip': True,
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        'evaluation': task_results,
    }


def run_smoke(study_path, *, dataset_root, smoke_root, batch_size=8):
    smoke_root = Path(smoke_root)
    _require(not smoke_root.exists(), f'Smoke root already exists; refusing to overwrite: {smoke_root}')
    smoke_root.mkdir(parents=True)
    datasets = {}
    environments = {}
    result = {
        'study_id': STUDY_ID,
        'formal_training_started': False,
        'smoke_root': str(smoke_root.resolve()),
        'configs': {},
    }
    try:
        for environment in ENV_SPECS:
            datasets[environment] = _load_smoke_dataset(environment, dataset_root)
            environments[environment] = _make_env(environment, dataset_root=dataset_root)
        for config_id in sorted(EXPECTED_CONFIG_IDS):
            configuration, config = _resolved_configuration(study_path, config_id)
            result['configs'][config_id] = _smoke_one(
                configuration,
                config,
                datasets[configuration.data['environment']],
                environments[configuration.data['environment']],
                smoke_root,
                batch_size=batch_size,
            )
    finally:
        for env in environments.values():
            env.close()
    _write_json(smoke_root / 'smoke_summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=STUDY_DEFAULT)
    parser.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    parser.add_argument('--run-root', default=RUN_ROOT_DEFAULT)
    parser.add_argument('--audit-output', default=AUDIT_DEFAULT)
    parser.add_argument('--report-output', default=REPORT_DEFAULT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', default=SMOKE_DEFAULT)
    parser.add_argument('--smoke-batch-size', type=int, default=8)
    parser.add_argument('--skip-paired-sampling', action='store_true')
    args = parser.parse_args(argv)
    _require(args.dataset_root, '--dataset-root is required')
    _require(args.smoke_batch_size > 0, '--smoke-batch-size must be positive')

    schema = check_study_schema(args.study)
    source_data = audit_source_and_data(
        dataset_root=args.dataset_root,
        study_path=args.study,
    )
    _write_json(args.audit_output, source_data)
    runtime = _runtime_initialize_audit(args.study)
    paired = None if args.skip_paired_sampling else paired_sampling_audit(
        args.study,
        dataset_root=args.dataset_root,
    )
    formal = _formal_artifact_audit(args.study, args.run_root)
    smoke = None
    if args.smoke:
        smoke = run_smoke(
            args.study,
            dataset_root=args.dataset_root,
            smoke_root=args.smoke_root,
            batch_size=args.smoke_batch_size,
        )
    report = {
        'doctor': 'M20B_cube_entity_mixer_transfer_screen',
        'formal_training_started': False,
        'source': _source_provenance(),
        'study_schema': schema,
        'source_data_audit': str(Path(args.audit_output).resolve()),
        'runtime_initialize': runtime,
        'paired_sampling': paired,
        'formal_artifact_preflight': formal,
        'smoke': smoke,
    }
    _write_json(args.report_output, report)
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
