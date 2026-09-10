"""Non-executing scientific and runtime preflight for the M24A Study.

The doctor validates the exact six-cell matrix, resolved GCIQL semantics,
historical M16D operating-point parity, parameter invariance, real-data raw
sampling pairing, dataset identity, and output-path absence.  It never creates
a Run directory, performs an optimizer update, starts training, assigns a GPU,
or mutates historical artifacts.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from impls.agents import agents
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.management import jsonable
from impls.main import _computation_runtime_extras, _make_config, _parse_args
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.puzzle_datasets import PuzzleBoardGCDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    REPO_ROOT
    / 'experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml'
)
DEFAULT_DATASET_ROOT = Path(
    '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
)
DEFAULT_RUN_ROOT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
STUDY_ID = 'M24A'
SLOT_NAMES = ('actor', 'value', 'critic')

ENVIRONMENTS = {
    'puzzle-4x5-play-v0': {
        'rows': 4,
        'cols': 5,
        'num_buttons': 20,
        'observation_dim': 99,
        'parameter_count': 1_678_586,
    },
    'puzzle-4x6-play-v0': {
        'rows': 4,
        'cols': 6,
        'num_buttons': 24,
        'observation_dim': 115,
        'parameter_count': 1_687_850,
    },
}

CONDITIONS = {
    'G1': {'mode': 'board', 'coordinate': 'target_board'},
    'G2': {'mode': 'residual', 'coordinate': 'state_goal_residual'},
    'G4': {
        'mode': 'oracle_operation',
        'coordinate': 'exact_remaining_press_parity',
    },
}

EXPECTED_CONFIGS = {
    'M24A-C001': ('puzzle-4x5-play-v0', 'G1'),
    'M24A-C002': ('puzzle-4x6-play-v0', 'G1'),
    'M24A-C003': ('puzzle-4x5-play-v0', 'G2'),
    'M24A-C004': ('puzzle-4x6-play-v0', 'G2'),
    'M24A-C005': ('puzzle-4x5-play-v0', 'G4'),
    'M24A-C006': ('puzzle-4x6-play-v0', 'G4'),
}

FROZEN_AGENT_FIELDS = {
    'actor_hidden_dims': [512, 512, 512],
    'value_hidden_dims': [512, 512, 512],
    'layer_norm': True,
    'lr': 0.0003,
    'batch_size': 1024,
    'discount': 0.99,
    'expectile': 0.9,
    'tau': 0.005,
    'actor_loss': 'ddpgbc',
    'alpha': 0.4,
    'const_std': True,
    'discrete': False,
    'encoder': None,
    'dataset_class': 'PuzzleBoardGCDataset',
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

FROZEN_PROTOCOL = {
    'train_steps': 1_000_000,
    'batch_size': 1024,
    'log_interval': 5_000,
    'eval_interval': 100_000,
    'eval_tasks': 'all',
    'eval_task_count': 5,
    'eval_episodes': 20,
    'eval_temperature': 0.0,
    'eval_gaussian': None,
    'video_episodes': 0,
    'save_interval': 100_000,
    'save_best_checkpoint': True,
    'save_last_checkpoint': True,
    'selection_metric': 'evaluation/overall_success',
    'selection_rule': 'strict_greater_than_keep_earlier_tie',
    'primary_endpoint': 'final@1M',
    'no_early_stopping_for_poor_performance': True,
    'formal_training_started': False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
    return jsonable(left) == jsonable(right)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _first_difference(left, right, path=()):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                return path + (str(key),), left.get(key), right.get(key)
            difference = _first_difference(
                left[key], right[key], path + (str(key),)
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path + ('length',), len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item, right_item, path + (str(index),)
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, left, right
    return None


def _require_equal_mapping(label, left, right):
    difference = _first_difference(jsonable(left), jsonable(right))
    if difference is not None:
        field = '.'.join(difference[0]) or '<root>'
        raise ValueError(
            f'{label} differs at {field}: left={difference[1]!r}, '
            f'right={difference[2]!r}'
        )


def _agent_args():
    return _parse_args(['--agent', 'gciql'])


def _resolved(study_path, config_path):
    _, configuration = prepare_run_design(study_path, config_path)
    config = _make_config(_agent_args(), configuration=configuration)
    return configuration, config


def _check_study(study):
    _require(study.study_id == STUDY_ID, f'Expected study_id={STUDY_ID!r}')
    _require(
        study.data.get('research_program') == 'M24 — Operation-Aligned Goal Learning',
        'M24A research program name is not frozen',
    )
    _require(
        study.data.get('primary_factors')
        == ['goal_conditioning_mode', 'environment'],
        'M24A primary factor declaration mismatch',
    )
    _require(study.data.get('algorithms') == ['gciql'], 'M24A must use GCIQL')
    _require(
        study.data.get('placements') == ['actor+value+critic'],
        'M24A placement mismatch',
    )
    _require(
        study.data.get('environments') == list(ENVIRONMENTS),
        'M24A environment order/matrix mismatch',
    )
    _require(study.data.get('seeds') == [0], 'M24A screening must use seed 0')
    _require(
        study.data.get('conditions') == list(CONDITIONS),
        'M24A condition order must be G1/G2/G4',
    )
    _require(
        study.data.get('primary_metric')
        == 'analysis/final_at_1m_hard_task_mean',
        'M24A scientific primary metric mismatch',
    )
    fixed = study.data.get('fixed_design', {})
    for field, expected in (
        ('algorithm', 'gciql'),
        ('dataset_class', 'PuzzleBoardGCDataset'),
        ('goal_success_semantics', 'board_equality'),
        ('alpha', 0.4),
        ('train_steps', 1_000_000),
        ('batch_size', 1024),
        ('frame_stack', None),
    ):
        _require(_same(fixed.get(field), expected), f'fixed_design.{field} mismatch')
    alpha = study.data.get('alpha_policy', {})
    _require(_same(alpha.get('value'), 0.4), 'alpha_policy.value must be 0.4')
    _require(alpha.get('no_alpha_sweep') is True, 'M24A must prohibit alpha sweep')
    protocol = study.data.get('protocol', {})
    for field, expected in FROZEN_PROTOCOL.items():
        _require(_same(protocol.get(field), expected), f'protocol.{field} mismatch')
    _require(
        protocol.get('expected_evaluation_steps')
        == list(range(100_000, 1_000_001, 100_000)),
        'M24A evaluation grid mismatch',
    )
    endpoints = study.data.get('scientific_endpoints', {})
    _require(
        endpoints.get('infrastructure_checkpoint_selection_metric')
        == 'evaluation/overall_success',
        'M24A checkpoint selection metric mismatch',
    )
    _require(
        endpoints.get('primary_derived_metric', {}).get('name') == 'HardTaskMean',
        'M24A must declare HardTaskMean',
    )
    comparisons = study.data.get('comparison_policy', {})
    _require(
        [item.get('contrast') for item in comparisons.get('clean_primary_comparisons', [])]
        == ['G2_minus_G1', 'G4_minus_G2'],
        'M24A clean contrasts mismatch',
    )
    _require(
        comparisons.get('historical_G0', {}).get('role')
        == 'descriptive_reference_only',
        'Historical G0 must remain descriptive only',
    )
    diagnosis = study.data.get('final_diagnosis', {})
    for field, expected in (
        ('checkpoint_role', 'last'),
        ('checkpoint_step', 1_000_000),
        ('task_ids', [1, 2, 3, 4, 5]),
        ('episodes_per_task', 50),
        ('evaluation_seed', 20260909),
        ('eval_temperature', 0.0),
        ('eval_gaussian', None),
    ):
        _require(_same(diagnosis.get(field), expected), f'final_diagnosis.{field} mismatch')
    execution = study.data.get('execution', {})
    _require(execution.get('physical_gpu') is None, 'Physical GPU must remain unset')
    _require(
        execution.get('source_requirement') == 'clean_detached_frozen_worktree',
        'M24A must require a clean detached source',
    )
    _require(
        execution.get('concurrency', {}).get('planned_jobs_per_gpu') == 2,
        'M24A planned jobs_per_gpu mismatch',
    )
    _require(
        execution.get('concurrency', {}).get('gate')
        == 'exact_workload_concurrent_gpu_smoke_required_before_formal_launch',
        'M24A concurrent smoke gate is missing',
    )
    acceptance = study.data.get('acceptance', {})
    _require(acceptance.get('expected_formal_runs') == 6, 'M24A must declare 6 Runs')
    _require(
        acceptance.get('no_automatic_formal_launch') is True,
        'M24A launch must remain user-controlled',
    )


def _check_slot(config_id, slot_name, slot, num_buttons):
    for field, expected in (
        ('enabled', True),
        ('primitive', 'mlp'),
        ('structure', 'puzzle_tokens'),
        ('topology', 'feedforward'),
        ('block', 'mlp_mixer'),
        ('credit', 'direct'),
    ):
        _require(
            slot.get(field) == expected,
            f'{config_id}: {slot_name}.{field} expected {expected!r}',
        )
    _require(
        slot.get('relation_mode', 'legacy_none') == 'legacy_none',
        f'{config_id}: relation augmentation is out of scope',
    )
    _require(
        'topology_kwargs' not in slot,
        f'{config_id}: recurrent topology is out of scope',
    )
    _require(
        not ({'goal_conditioning', 'goal_conditioning_mode'} & set(slot)),
        f'{config_id}: goal semantics must not be slot-local',
    )
    structure = slot.get('structure_kwargs', {})
    for field, expected in (
        ('num_buttons', num_buttons),
        ('token_dim', 128),
        ('robot_hidden_dim', 128),
        ('token_mlp_hidden_dim', 64),
        ('channel_mlp_hidden_dim', 256),
        ('num_mixer_blocks', 2),
        ('index_embedding', True),
        ('readout', 'mean'),
        ('tm_mode', 'none'),
    ):
        _require(
            structure.get(field) == expected,
            f'{config_id}: {slot_name}.structure_kwargs.{field} mismatch',
        )
    _require(slot.get('readout') == 'mean', f'{config_id}: readout must resolve to mean')


def _historical_agent(study, environment):
    record = study.data['architecture_anchor']['resolved_artifacts'][environment]
    path = Path(record['path'])
    _require(path.is_file(), f'Missing M16D resolved artifact: {path}')
    _require(
        _sha256(path) == record['resolved_config_sha256'],
        f'M16D resolved artifact SHA256 mismatch: {path}',
    )
    with path.open() as file:
        payload = json.load(file)
    _require(
        payload.get('resolved_config_fingerprint')
        == record['resolved_config_fingerprint'],
        f'M16D embedded config fingerprint mismatch: {environment}',
    )
    return payload['algorithm_config']['agent']


def _check_configuration(study, configuration, config, expected):
    config_id = configuration.config_id
    environment, condition_id = expected
    condition = CONDITIONS[condition_id]
    spec = ENVIRONMENTS[environment]
    data = configuration.data
    _require('seed' not in data, f'{config_id}: seed belongs to a Run, not a config')
    for field, expected_value in (
        ('algorithm', 'gciql'),
        ('environment', environment),
        ('protocol_stage', 'formal'),
        ('placement', 'actor+value+critic'),
        ('executable', True),
        ('condition_id', condition_id),
    ):
        _require(
            data.get(field) == expected_value,
            f'{config_id}: {field} expected {expected_value!r}',
        )
    factors = data.get('factors', {})
    for field, expected_value in (
        ('environment', environment),
        ('condition', condition_id),
        ('goal_conditioning_mode', condition['mode']),
        ('goal_coordinate', condition['coordinate']),
        ('goal_success_semantics', 'board_equality'),
        ('structure', 'puzzle_tokens'),
        ('topology', 'feedforward'),
        ('block', 'mlp_mixer'),
        ('num_buttons', spec['num_buttons']),
        ('num_mixer_blocks', 2),
        ('alpha', 0.4),
    ):
        _require(
            _same(factors.get(field), expected_value),
            f'{config_id}: factors.{field} mismatch',
        )
    overrides = data.get('agent_overrides', {})
    for field, expected_value in FROZEN_AGENT_FIELDS.items():
        _require(
            _same(overrides.get(field), expected_value),
            f'{config_id}: agent_overrides.{field} mismatch',
        )
        _require(
            _same(config.get(field), expected_value),
            f'{config_id}: resolved agent.{field} mismatch',
        )
    goal = config.get('goal_conditioning', {})
    for field, expected_value in (
        ('domain', 'puzzle'),
        ('mode', condition['mode']),
        ('rows', spec['rows']),
        ('cols', spec['cols']),
        ('num_buttons', spec['num_buttons']),
        ('robot_dim', 19),
        ('button_feature_dim', 4),
        ('robot_goal_policy', 'zero'),
        ('button_goal_transient_policy', 'zero'),
    ):
        _require(goal.get(field) == expected_value, f'{config_id}: goal_conditioning.{field} mismatch')
    _require(set(config.get('compute', {})) == set(SLOT_NAMES), f'{config_id}: slot set mismatch')
    for slot_name in SLOT_NAMES:
        _check_slot(config_id, slot_name, config['compute'][slot_name], spec['num_buttons'])
    runtime = _computation_runtime_extras(config)
    runtime_goal = runtime.get('goal_conditioning', {})
    _require(runtime_goal.get('mode') == condition['mode'], f'{config_id}: runtime mode mismatch')
    _require(runtime_goal.get('operator_rank') == spec['num_buttons'], f'{config_id}: operator rank mismatch')
    _require(runtime_goal.get('operator_full_rank') is True, f'{config_id}: operator must be full rank')
    _require(runtime.get('goal_success_semantics') == 'board_equality', f'{config_id}: runtime success semantics mismatch')

    candidate_common = copy.deepcopy(jsonable(config))
    candidate_common.pop('goal_conditioning')
    candidate_common['dataset_class'] = 'GCDataset'
    historical = _historical_agent(study, environment)
    _require_equal_mapping(
        f'{config_id} common runtime versus actual M16D attempt-001',
        candidate_common,
        historical,
    )


def _valid_observations(num_buttons, batch_size=2):
    observations = np.zeros(
        (batch_size, 19 + 4 * num_buttons), dtype=np.float32
    )
    buttons = observations[:, 19:].reshape(batch_size, num_buttons, 4)
    buttons[..., 0] = 1.0
    return observations


def _array_trees_equal(left, right):
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(left_item), np.asarray(right_item))
        for left_item, right_item in zip(left_leaves, right_leaves)
    )


def _parameter_audit(configs):
    report = {}
    for environment, spec in ENVIRONMENTS.items():
        agents_by_condition = {}
        observations = _valid_observations(spec['num_buttons'])
        actions = np.zeros((2, 5), dtype=np.float32)
        for condition_id in CONDITIONS:
            config_id = next(
                config_id
                for config_id, cell in EXPECTED_CONFIGS.items()
                if cell == (environment, condition_id)
            )
            agent = agents['gciql'].create(
                24024, observations, actions, configs[config_id]
            )
            _require(not agent.network.model_state, f'{config_id}: unexpected model state')
            for module_name in ('actor', 'value', 'critic', 'target_critic'):
                conditioner = agent.network.model_def.modules[module_name].goal_conditioner
                _require(conditioner is not None, f'{config_id}: {module_name} lacks conditioner')
                _require(
                    conditioner.mode == CONDITIONS[condition_id]['mode'],
                    f'{config_id}: {module_name} conditioner mismatch',
                )
            agents_by_condition[condition_id] = agent

        reference = agents_by_condition['G1']
        reference_flat = flatten_dict(reference.network.params)
        schema = {path: value.shape for path, value in reference_flat.items()}
        count = sum(value.size for value in reference_flat.values())
        _require(
            count == spec['parameter_count'],
            f'{environment}: expected {spec["parameter_count"]} params, got {count}',
        )
        opt_structure = jax.tree_util.tree_structure(reference.network.opt_state)
        opt_leaves = jax.tree_util.tree_leaves(reference.network.opt_state)
        for condition_id in ('G2', 'G4'):
            candidate = agents_by_condition[condition_id]
            candidate_flat = flatten_dict(candidate.network.params)
            _require(
                {path: value.shape for path, value in candidate_flat.items()} == schema,
                f'{environment}: {condition_id} parameter schema mismatch',
            )
            _require(
                _array_trees_equal(candidate.network.params, reference.network.params),
                f'{environment}: {condition_id} initialized params mismatch',
            )
            _require(
                jax.tree_util.tree_structure(candidate.network.opt_state) == opt_structure,
                f'{environment}: {condition_id} optimizer structure mismatch',
            )
            _require(
                all(
                    np.array_equal(np.asarray(left), np.asarray(right))
                    for left, right in zip(
                        jax.tree_util.tree_leaves(candidate.network.opt_state),
                        opt_leaves,
                    )
                ),
                f'{environment}: {condition_id} optimizer state mismatch',
            )
        report[environment] = {
            'trainable_leaves': len(reference_flat),
            'trainable_parameters': count,
            'conditions_equal': True,
        }
        del agents_by_condition, reference, reference_flat
        gc.collect()
    return report


def _pair_fingerprint(sampled_sequence):
    digest = hashlib.sha256()
    for batch, trace in sampled_sequence:
        for group in (trace, batch):
            for key in sorted(group):
                array = np.ascontiguousarray(np.asarray(group[key]))
                digest.update(key.encode())
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def _real_data_pairing_audit(dataset_root, configs):
    report = {}
    for environment_index, (environment, spec) in enumerate(ENVIRONMENTS.items()):
        env = None
        try:
            env, raw_train, _ = make_env_and_datasets(
                environment,
                seed=24100 + environment_index,
                dataset_seed=24200 + environment_index,
                dataset_dir=dataset_root,
            )
            datasets = {}
            for condition_id in CONDITIONS:
                config_id = next(
                    config_id
                    for config_id, cell in EXPECTED_CONFIGS.items()
                    if cell == (environment, condition_id)
                )
                datasets[condition_id] = PuzzleBoardGCDataset(
                    raw_train,
                    configs[config_id],
                    rng=24300 + environment_index,
                )
            sequences = {condition_id: [] for condition_id in CONDITIONS}
            for _ in range(3):
                for condition_id, dataset in datasets.items():
                    sequences[condition_id].append(
                        dataset.sample(32, return_sampling_trace=True)
                    )
            reference = sequences['G1']
            for condition_id in ('G2', 'G4'):
                for call_index, (left, right) in enumerate(
                    zip(reference, sequences[condition_id])
                ):
                    _require(
                        _array_trees_equal(left, right),
                        f'{environment}: raw sample mismatch for {condition_id} '
                        f'at call {call_index}',
                    )
                _require(
                    jsonable(datasets[condition_id].rng.bit_generator.state)
                    == jsonable(datasets['G1'].rng.bit_generator.state),
                    f'{environment}: RNG state mismatch for {condition_id}',
                )
            report[environment] = {
                'calls': 3,
                'batch_size': 32,
                'transition_value_actor_pairs_equal': True,
                'raw_batches_equal': True,
                'rng_states_equal': True,
                'paired_sequence_sha256': _pair_fingerprint(reference),
                'num_buttons': spec['num_buttons'],
            }
        finally:
            if env is not None:
                env.close()
            gc.collect()
    return report


def validate(study_path, dataset_root, run_root):
    """Run all non-mutating M24A gates and return a JSON-ready report."""

    study = load_study(study_path)
    _check_study(study)
    dataset_root = Path(dataset_root)
    run_root = Path(run_root)
    study_run_root = run_root / STUDY_ID
    _require(
        not study_run_root.exists(),
        'M24A formal output namespace must be absent before launch: '
        f'{study_run_root}',
    )
    missing = [
        dataset_root / filename
        for environment in ENVIRONMENTS
        for filename in (f'{environment}.npz', f'{environment}-val.npz')
        if not (dataset_root / filename).is_file()
    ]
    _require(
        not missing,
        'Missing M24A dataset files:\n' + '\n'.join(f'  {path}' for path in missing),
    )
    config_dir = Path(study.path).parent / 'configs'
    config_paths = sorted(config_dir.glob('*.yaml'))
    _require(
        [path.stem for path in config_paths] == list(EXPECTED_CONFIGS),
        'M24A must contain exactly C001 through C006',
    )

    configs = {}
    run_dirs = []
    seen_run_dirs = set()
    for config_path in config_paths:
        configuration, config = _resolved(study.path, config_path)
        config_id = configuration.config_id
        _check_configuration(
            study, configuration, config, EXPECTED_CONFIGS[config_id]
        )
        run_dir = make_run_path(
            run_root,
            study.study_id,
            config_id,
            configuration.slug,
            configuration.data['environment'],
            0,
            run_attempt=0,
        )
        _require(run_dir not in seen_run_dirs, f'Duplicate Run path: {run_dir}')
        _require(
            not run_dir.exists(),
            f'M24A output path already exists; refusing overwrite: {run_dir}',
        )
        seen_run_dirs.add(run_dir)
        run_dirs.append(str(run_dir))
        configs[config_id] = config

    parameter_report = _parameter_audit(configs)
    pairing_report = _real_data_pairing_audit(dataset_root, configs)
    return {
        'status': 'PASS',
        'study_id': study.study_id,
        'configs': list(EXPECTED_CONFIGS),
        'matrix': {
            config_id: {
                'environment': cell[0],
                'condition': cell[1],
                'mode': CONDITIONS[cell[1]]['mode'],
            }
            for config_id, cell in EXPECTED_CONFIGS.items()
        },
        'seeds': [0],
        'expected_formal_runs': 6,
        'protocol': jsonable(study.data['protocol']),
        'parameter_invariance': parameter_report,
        'raw_sampling_pairing': pairing_report,
        'run_root': str(run_root.resolve()),
        'study_run_root': str(study_run_root.resolve()),
        'run_paths_absent': True,
        'run_dirs': run_dirs,
        'physical_gpu': None,
        'planned_jobs_per_gpu_after_smoke': 2,
        'concurrent_gpu_smoke_completed': False,
        'formal_training_started': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        '--json-output',
        type=Path,
        default=None,
        help='Optional path for a reusable preflight report; omitted by default.',
    )
    args = parser.parse_args(argv)
    try:
        report = validate(args.study, args.dataset_root, args.run_root)
    except Exception as error:
        print(f'M24A PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2

    print('M24A PREFLIGHT: PASS')
    print(
        'configs=6; environments=2; conditions=3; seeds=[0]; '
        'planned_runs=6; formal_training_started=False'
    )
    print('config_id\tenvironment\tcondition\tmode\trun_path_absent')
    for config_id, cell in report['matrix'].items():
        print(
            f'{config_id}\t{cell["environment"]}\t{cell["condition"]}\t'
            f'{cell["mode"]}\ttrue'
        )
    for environment, parameter in report['parameter_invariance'].items():
        pairing = report['raw_sampling_pairing'][environment]
        print(
            f'{environment}: params={parameter["trainable_parameters"]}; '
            f'leaves={parameter["trainable_leaves"]}; '
            f'paired_calls={pairing["calls"]}; '
            f'pair_sha256={pairing["paired_sequence_sha256"]}'
        )
    print(
        'Concurrent GPU workload smoke: REQUIRED, NOT RUN. '
        'Physical GPU assignment: UNSET.'
    )
    print('Formal training was not started. Git and launch remain user-controlled.')
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n'
        )
        print(f'JSON report: {args.json_output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
