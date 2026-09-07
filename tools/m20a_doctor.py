"""Non-executing M20A Phase-2 freeze and launch-readiness doctor.

The doctor deliberately never invokes the formal study launcher, RunContext,
or a checkpoint under the M20A run root.  It validates the user-frozen,
executable 18-cell design, the Phase-1 relation audit, correctness gates, and
the small real-data smoke artifact produced separately under ``/tmp``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
import warnings
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from impls.agents.gciql import GCIQLAgent
from impls.computation.accounting import gciql_architecture_accounting
from impls.computation.readouts import HybridContextQueryReadout
from impls.computation.relation import RelationAugmenter
from impls.experiment import load_configuration, load_study, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.representation.interfaces import StructuredRepresentation
from impls.representation.manipulation import (
    parse_cube_observation,
    parse_scene_observation,
)
from impls.representation.puzzle import parse_puzzle_observation
from impls.representation.relations import (
    CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
    PUZZLE_4X4_SHUFFLE_PERMUTATION,
    build_cube_relations,
    build_puzzle_relations,
    build_scene_relations,
    cube_correct_relations,
    puzzle_shuffle_diagnostics,
    puzzle_toggle_relation,
    relation_edge_counts,
    scene_controls_relation,
)


STUDY_ID = 'M20A'
STUDY_DEFAULT = 'experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml'
AUDIT_DEFAULT = 'docs/9-8/M20A_phase1_relation_audit.json'
SMOKE_DEFAULT = '/tmp/m20a_phase1_smoke/m20a_phase1_smoke.json'
REPORT_DEFAULT = 'docs/9-8/M20A_phase2_doctor.json'
RUN_ROOT_DEFAULT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'
AUDIT_SEED = 20_020
AUDIT_SAMPLES = 100_000
CUBE_FROZEN = {
    'current_support_epsilon_xy': 0.02,
    'current_support_epsilon_z': 0.010,
    'goal_support_epsilon_xy': 0.02,
    'goal_support_epsilon_z': 0.010,
    'goal_conflict_radius': 0.04,
}
TASK_SPECS = {
    'PUZZLE': {'env': 'puzzle-4x4-play-v0', 'obs_dim': 83, 'relation_types': 1},
    'CUBE': {'env': 'cube-triple-play-v0', 'obs_dim': 46, 'relation_types': 3},
    'SCENE': {'env': 'scene-play-v0', 'obs_dim': 40, 'relation_types': 1},
}
MODES = (('Z', 'zero'), ('C', 'correct'), ('S', 'shuffled'))
READOUTS = (('M', 'mean_context'), ('Q', 'hybrid_context_query'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    path = Path(path)
    _require(path.is_file(), f'Missing required artifact: {path}')
    with path.open() as file:
        value = json.load(file)
    _require(isinstance(value, dict), f'Expected JSON object: {path}')
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
    temporary.replace(path)


def _config_id(task, mode_short, readout_short):
    return f'M20A-{task}-{mode_short}-{readout_short}'


def _agent_args():
    return _parse_args(['--agent', 'gciql'])


def _resolved_config(study_path, config_id):
    _, configuration = prepare_run_design(study_path, config_id)
    return _make_config(_agent_args(), configuration=configuration)


def _expected_ids():
    return {
        _config_id(task, mode_short, readout_short)
        for task in TASK_SPECS
        for mode_short, _ in MODES
        for readout_short, _ in READOUTS
    }


def check_study_schema(study_path):
    study = load_study(study_path)
    _require(study.study_id == STUDY_ID, f'Expected study_id={STUDY_ID}')
    _require(study.data.get('seeds') == [0], 'M20A requires exactly training seed 0')
    _require(study.data.get('alpha_policy', {}).get('value') == 1.0, 'Study alpha must be 1.0')
    readiness = study.data.get('phase2_readiness', {})
    _require(readiness.get('formal_executable_runs') == 18, 'Study must record 18 executable Phase-2 runs')
    _require(readiness.get('blocked_phase2_skeletons') == 0, 'Study must record zero blocked Phase-2 runs')
    _require(readiness.get('user_phase2_freeze_recorded') is True, 'User Phase-2 freeze must be recorded')
    frozen_thresholds = study.data.get('frozen_user_decisions', {}).get('cube_relation_thresholds', {})
    for key, expected in CUBE_FROZEN.items():
        _require(frozen_thresholds.get(key) == expected, f'Study frozen Cube threshold {key}')
    _require(frozen_thresholds.get('threshold_status') == 'FROZEN_USER_PHASE2', 'Study Cube threshold status')
    _require(
        frozen_thresholds.get('support_geometry_shared_between_current_and_goal') is True,
        'Study must record shared current/goal support geometry',
    )
    config_paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    _require(len(config_paths) == 18, f'M20A requires 18 configs, found {len(config_paths)}')
    seen = set()
    summary = {}
    for path in config_paths:
        configuration = load_configuration(study, path)
        data = configuration.data
        seen.add(configuration.config_id)
        _require(data.get('protocol_stage') == 'phase2_formal', f'{configuration.config_id}: wrong stage')
        _require(data.get('executable') is True, f'{configuration.config_id}: must be executable')
        _require(not data.get('blocked_by'), f'{configuration.config_id}: stale block metadata')
        _require(data.get('training_seed') == 0, f'{configuration.config_id}: training_seed must be 0')
        _require(data.get('factors', {}).get('alpha') == 1.0, f'{configuration.config_id}: factors alpha')
        _require(data.get('agent_overrides', {}).get('alpha') == 1.0, f'{configuration.config_id}: runtime alpha')
        protocol = data.get('frozen_protocol', {})
        for key, expected in (
            ('train_steps', 1_000_000), ('batch_size', 1024),
            ('eval_interval', 100_000), ('eval_episodes', 50),
            ('eval_tasks', 'all'), ('eval_temperature', 0.0),
            ('save_interval', 100_000), ('save_best_checkpoint', True),
            ('save_last_checkpoint', True), ('primary_endpoint', 'final@1M'),
        ):
            _require(protocol.get(key) == expected, f'{configuration.config_id}: frozen protocol {key}')
        resolved = _resolved_config(study_path, configuration.config_id)
        _require(resolved.get('alpha') == 1.0, f'{configuration.config_id}: resolved alpha')
        slots = resolved.get('compute', {})
        _require(set(slots) >= {'actor', 'value', 'critic'}, f'{configuration.config_id}: missing slots')
        slot_summary = {}
        for slot_name in ('actor', 'value', 'critic'):
            slot = slots[slot_name]
            _require(slot.get('enabled') is True, f'{configuration.config_id}.{slot_name}: disabled')
            for key, expected in (
                ('structure', {'PUZZLE': 'puzzle_tokens', 'CUBE': 'cube_tokens', 'SCENE': 'scene_tokens'}[configuration.config_id.split('-')[1]]),
                ('block', 'mlp_mixer'), ('topology', 'feedforward'), ('credit', 'direct'),
                ('relation_augmenter', 'relation_mlp'),
            ):
                _require(slot.get(key) == expected, f'{configuration.config_id}.{slot_name}: {key}')
            block = slot.get('block_kwargs', {})
            for key, expected in (
                ('num_blocks', 2), ('token_hidden_dim', 64),
                ('channel_hidden_dim', 256), ('tm_mode', 'none'),
            ):
                _require(
                    block.get(key) == expected,
                    f'{configuration.config_id}.{slot_name}: frozen Mixer {key}',
                )
            augmenter = slot.get('relation_augmenter_kwargs', {})
            for key, expected in (
                ('relation_hidden_dim', 256), ('activation', 'gelu'),
                ('first_use_bias', False), ('second_use_bias', False),
                ('normalization', 'none'), ('dropout', 'none'), ('output_dim', 128),
            ):
                _require(augmenter.get(key) == expected, f'{configuration.config_id}.{slot_name}: {key}')
            readout = slot.get('readout')
            _require(readout in {'mean_context', 'hybrid_context_query'}, 'invalid M20A readout')
            if readout == 'hybrid_context_query':
                _require(slot.get('readout_kwargs', {}).get('query_dim') == 128, 'query_dim must be 128')
            if configuration.config_id.split('-')[1] == 'CUBE':
                relation = slot.get('relation_kwargs', {})
                _require(relation.get('threshold_status') == 'FROZEN_USER_PHASE2', 'Cube threshold status')
                for key, expected in CUBE_FROZEN.items():
                    _require(
                        relation.get(key) == expected,
                        f'{configuration.config_id}.{slot_name}: frozen {key}',
                    )
                protocol_thresholds = protocol.get('cube_relation_thresholds', {})
                _require(protocol.get('cube_threshold_status') == 'FROZEN_USER_PHASE2', 'Cube protocol threshold status')
                for key, expected in CUBE_FROZEN.items():
                    _require(
                        protocol_thresholds.get(key) == expected,
                        f'{configuration.config_id}: frozen protocol {key}',
                    )
            slot_summary[slot_name] = {
                'structure': slot['structure'], 'relation_mode': slot['relation_mode'],
                'readout': readout,
            }
        summary[configuration.config_id] = slot_summary
    _require(seen == _expected_ids(), f'M20A config IDs differ: {sorted(seen)!r}')
    return {
        'config_count': len(config_paths),
        'config_ids': sorted(seen),
        'resolved_slot_summary': summary,
        'official_alpha_audit': {
            'hard_frozen_alpha': 1.0,
            'historical_authority': study.data['alpha_policy'].get('source'),
            'local_official_reference_status': study.data['alpha_policy'].get('local_official_reference_status'),
        },
        'phase2_freeze': {
            'formal_executable_runs': readiness['formal_executable_runs'],
            'frozen_cube_thresholds': CUBE_FROZEN,
        },
    }


def check_environment_creation():
    import ogbench

    observed = {}
    for task, spec in TASK_SPECS.items():
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
            env = ogbench.make_env_and_datasets(spec['env'], env_only=True)
            try:
                observation, _ = env.reset(seed=AUDIT_SEED)
                observed_dim = int(np.asarray(observation).shape[-1])
                _require(observed_dim == spec['obs_dim'], f'{task}: observation dim {observed_dim}')
                observed[task] = {'environment': spec['env'], 'observation_dim': observed_dim}
            finally:
                env.close()
    return observed


def check_entity_parsers_and_static_oracles():
    puzzle_robot, puzzle_buttons = parse_puzzle_observation(
        np.zeros((2, 83), dtype=np.float32), num_buttons=16
    )
    cube_robot, cube_tokens = parse_cube_observation(
        np.zeros((2, 46), dtype=np.float32), num_cubes=3
    )
    scene = parse_scene_observation(np.zeros((2, 40), dtype=np.float32))
    _require(puzzle_robot.shape == (2, 19) and puzzle_buttons.shape == (2, 16, 4), 'Puzzle parser shape')
    _require(cube_robot.shape == (2, 19) and cube_tokens.shape == (2, 3, 9), 'Cube parser shape')
    _require([part.shape for part in scene] == [(2, 19), (2, 9), (2, 2, 4), (2, 2), (2, 2)], 'Scene parser shape')

    puzzle = np.asarray(puzzle_toggle_relation(4, 4))[:, :, 0]
    degree = puzzle.sum(axis=1).astype(int)
    _require(sorted(np.unique(degree).tolist()) == [3, 4, 5], 'Puzzle degree classes')
    _require(int(np.sum(degree == 3)) == 4, 'Puzzle corner degree count')
    _require(int(np.sum(degree == 4)) == 8, 'Puzzle edge degree count')
    _require(int(np.sum(degree == 5)) == 4, 'Puzzle interior degree count')
    diagnostics = puzzle_shuffle_diagnostics(4, 4, PUZZLE_4X4_SHUFFLE_PERMUTATION)
    _require(diagnostics['differing_adjacency_entries'] > 0, 'Puzzle shuffle is an automorphism')

    scene_correct = np.asarray(scene_controls_relation(mode='correct'))[:, :, 0]
    scene_shuffled = np.asarray(scene_controls_relation(mode='shuffled'))[:, :, 0]
    _require(scene_correct[1, 3] == 1 and scene_correct[2, 4] == 1, 'Scene correct map')
    _require(scene_shuffled[1, 4] == 1 and scene_shuffled[2, 3] == 1, 'Scene shuffled map')
    _require(scene_correct.sum() == scene_shuffled.sum() == 2, 'Scene control edge parity')
    return {
        'parser_shapes': {
            'puzzle': {'robot': list(puzzle_robot.shape), 'tokens': list(puzzle_buttons.shape)},
            'cube': {'robot': list(cube_robot.shape), 'tokens': list(cube_tokens.shape)},
            'scene': [list(part.shape) for part in scene],
        },
        'puzzle': diagnostics | {'degree_vector': degree.tolist()},
        'scene': {'correct_edge_count': int(scene_correct.sum()), 'shuffled_edge_count': int(scene_shuffled.sum())},
    }


def _cube_feature(positions):
    values = np.zeros((1, 3, 9), dtype=np.float32)
    values[0, :, :3] = np.asarray(positions, dtype=np.float32) * 10.0
    return values


def _cube_relation(state_positions, goal_positions, mode='correct'):
    thresholds = {
        **CUBE_FROZEN,
        'conflict_radius': CUBE_FROZEN['goal_conflict_radius'],
    }
    del thresholds['goal_conflict_radius']
    return np.asarray(build_cube_relations(
        _cube_feature(state_positions), _cube_feature(goal_positions), mode=mode,
        shuffle_derangement=CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
        cube_height=0.04, xyz_scaler=10.0, **thresholds,
    ))


def check_cube_synthetic_oracles():
    far_goal = ((1.0, 1.0, 0.0), (1.2, 1.0, 0.0), (1.4, 1.0, 0.0))
    state_stack = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.04), (0.5, 0.5, 0.0))
    case_a = _cube_relation(state_stack, far_goal)
    _require(case_a[0, 0, 1, 0] == 1 and case_a[0, 1, 0, 0] == 0, 'Cube current support direction')

    state_far = ((-1.0, -1.0, 0.0), (-1.2, -1.0, 0.0), (-1.4, -1.0, 0.0))
    goal_stack = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.04), (0.5, 0.5, 0.0))
    case_b = _cube_relation(state_far, goal_stack)
    _require(case_b[0, 0, 1, 1] == 1 and case_b[0, 1, 0, 1] == 0, 'Cube goal support direction')

    state_conflict = ((0.0, 0.0, 0.0), (0.7, 0.7, 0.0), (0.9, 0.9, 0.0))
    goal_conflict = ((-0.5, -0.5, 0.0), (0.0, 0.0, 0.0), (1.4, 1.4, 0.0))
    case_c = _cube_relation(state_conflict, goal_conflict)
    _require(case_c[0, 0, 1, 2] == 1 and case_c[0, 1, 0, 2] == 0, 'Cube goal conflict direction')

    separated = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.2), (1.0, 1.0, 0.4))
    case_d = _cube_relation(separated, ((-1.0, -1.0, 0.0), (-1.5, -1.5, 0.2), (-2.0, -2.0, 0.4)))
    _require(np.all(case_d == 0), 'Separated Cube case must have no relation')
    shuffled = _cube_relation(state_stack, far_goal, mode='shuffled')
    _require(np.array_equal(relation_edge_counts(case_a), relation_edge_counts(shuffled)), 'Cube shuffled count parity')
    _require(not np.array_equal(case_a, shuffled), 'Cube shuffled endpoints did not change')
    return {
        'frozen_thresholds_used_for_synthetic_tests': CUBE_FROZEN,
        'case_edges': {
            'current_support': int(case_a.sum()),
            'goal_support': int(case_b.sum()),
            'goal_conflict': int(case_c.sum()),
            'separated': int(case_d.sum()),
        },
    }


def check_relation_augmenter_and_readout():
    tokens = jnp.asarray([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    zero = jnp.zeros((1, 3, 3, 1), dtype=jnp.float32)
    module = RelationAugmenter(token_dim=2, relation_hidden_dim=256)
    variables = module.init(jax.random.PRNGKey(7), tokens, zero)
    output = module.apply(variables, tokens, zero)
    _require(np.array_equal(np.asarray(output), np.asarray(tokens)), 'Zero relation must be exact identity')

    directed = np.zeros((1, 3, 3, 1), dtype=np.float32)
    directed[0, 0, 1, 0] = 1.0
    features, _ = RelationAugmenter.aggregate_features(tokens, directed)
    features = np.asarray(features)
    # First D entries are outgoing type 0; second D entries incoming type 0.
    np.testing.assert_array_equal(features[0, 0, :2], np.asarray(tokens)[0, 1])
    np.testing.assert_array_equal(features[0, 1, 2:], np.asarray(tokens)[0, 0])
    masked_features, _ = RelationAugmenter.aggregate_features(
        tokens, directed, mask=np.asarray([[True, False, True]])
    )
    _require(np.array_equal(np.asarray(masked_features), np.zeros_like(features)), 'Entity mask relation semantics')
    relation_mask = np.zeros((1, 3, 3), dtype=bool)
    masked_features, _ = RelationAugmenter.aggregate_features(
        tokens, directed, relation_mask=relation_mask
    )
    _require(np.array_equal(np.asarray(masked_features), np.zeros_like(features)), 'Relation mask semantics')

    readout = HybridContextQueryReadout(output_dim=5, token_dim=2, query_dim=128)
    readout_tokens = jnp.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    context = jnp.asarray([[0.5, -0.5, 1.0]])
    mask = jnp.asarray([[True, False, True]])
    readout_vars = readout.init(jax.random.PRNGKey(9), readout_tokens, context=context, mask=mask)
    readout_output = readout.apply(readout_vars, readout_tokens, context=context, mask=mask)
    attention = readout.apply(
        readout_vars, readout_tokens, context=context, mask=mask,
        method=HybridContextQueryReadout.attention_weights,
    )
    _require(readout_output.shape == (1, 5), 'Hybrid readout output shape')
    _require(np.allclose(np.asarray(attention)[0, 1], 0.0), 'Hybrid invalid attention weight')
    _require(np.allclose(np.asarray(attention).sum(axis=-1), 1.0), 'Hybrid attention normalization')
    permutation = np.asarray([2, 0, 1])
    permuted = readout.apply(
        readout_vars, readout_tokens[:, permutation], context=context, mask=mask[:, permutation]
    )
    np.testing.assert_allclose(np.asarray(readout_output), np.asarray(permuted), rtol=1e-6, atol=1e-6)
    return {
        'zero_exact_identity': True,
        'direction_test': 'outgoing source=0 receives H_target=1; incoming target=1 receives H_source=0',
        'mask_tests': True,
        'hybrid_mask_and_permutation_tests': True,
    }


def _synthetic_agent_inputs(task, *, relation_goals=False):
    spec = TASK_SPECS[task]
    observations = np.zeros((2, spec['obs_dim']), dtype=np.float32)
    actions = np.zeros((2, 5), dtype=np.float32)
    if task == 'CUBE' and relation_goals:
        # Current slots are deliberately unstacked. Actor goals have a stack,
        # whereas value goals are separated; this makes goal-dependent wiring
        # observable through the actual GCIQL modules.
        observations[:, 19:46] = _cube_feature(((0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)))[0].reshape(-1)
    return jnp.asarray(observations), jnp.asarray(actions)


def _flat_parameter_map(params):
    return {
        tuple(path): np.asarray(value)
        for path, value in flatten_dict(params).items()
    }


def _exact_parameter_parity(reference, candidate, label):
    left = _flat_parameter_map(reference.network.params)
    right = _flat_parameter_map(candidate.network.params)
    _require(set(left) == set(right), f'{label}: parameter paths differ')
    for path in left:
        _require(left[path].shape == right[path].shape, f'{label}: shape differs at {path!r}')
        if not np.array_equal(left[path], right[path]):
            raise ValueError(f'{label}: initial value differs at {path!r}')
    return {'parameter_paths': len(left), 'parameter_scalars': int(sum(value.size for value in left.values()))}


def check_initialization_and_accounting(study_path):
    result = {}
    for task in TASK_SPECS:
        result[task] = {}
        for readout_short, _ in READOUTS:
            created = {}
            observations, actions = _synthetic_agent_inputs(task)
            for mode_short, _ in MODES:
                config_id = _config_id(task, mode_short, readout_short)
                config = _resolved_config(study_path, config_id)
                created[mode_short] = GCIQLAgent.create(0, observations, actions, config)
            c_parity = _exact_parameter_parity(created['Z'], created['C'], f'{task}/{readout_short} Z/C')
            s_parity = _exact_parameter_parity(created['Z'], created['S'], f'{task}/{readout_short} Z/S')
            # The generic accounting must expose dense relation reduction cost
            # separately from edge sparsity.  Z/C/S are nominally matched.
            config = _resolved_config(study_path, _config_id(task, 'Z', readout_short))
            accounting = _computation_slot_accounting(created['Z'], config)
            architecture = gciql_architecture_accounting(created['Z'].network.params, config, accounting)
            slot_checks = {}
            for slot_name, item in accounting.items():
                _require(item['relation_augmenter_params'] > 0, f'{task}/{readout_short}/{slot_name}: relation params')
                _require(item['executed_dense_relation_aggregation_cost'] > 0, f'{task}/{readout_short}/{slot_name}: dense aggregation')
                _require(item['relation_sparsity_not_hardware_saving'] is True, f'{task}/{readout_short}/{slot_name}: sparse claim')
                slot_checks[slot_name] = {
                    'params': item['trainable_params'],
                    'relation_augmenter_params': item['relation_augmenter_params'],
                    'relation_dense_aggregation_cost': item['executed_dense_relation_aggregation_cost'],
                    'readout': item['readout'],
                    'query_params': item['query_params'],
                }
            result[task][readout_short] = {
                'zero_correct': c_parity,
                'zero_shuffled': s_parity,
                'accounting': slot_checks,
                'architecture_total_params': architecture['total_trainable_params'],
                'architecture_total_dense_macs': architecture['total_dense_macs'],
            }
            del created
    return result


def _cube_goal(positions):
    value = np.zeros((2, 46), dtype=np.float32)
    value[:, 19:46] = _cube_feature(positions)[0].reshape(-1)
    return jnp.asarray(value)


def check_call_specific_goal_wiring(study_path):
    config = _resolved_config(study_path, 'M20A-CUBE-C-M')
    observations, actions = _synthetic_agent_inputs('CUBE', relation_goals=True)
    agent = GCIQLAgent.create(0, observations, actions, config)
    actor_goals = _cube_goal(((0.0, 0.0, 0.0), (0.0, 0.0, 0.04), (0.7, 0.7, 0.0)))
    value_goals = _cube_goal(((0.0, 0.0, 0.0), (0.4, 0.4, 0.0), (0.8, 0.8, 0.0)))
    actor_relation = agent.network.select('actor')(
        observations, actor_goals, method='relation_diagnostic'
    )['relations']
    value_relation = agent.network.select('value')(
        observations, value_goals, method='relation_diagnostic'
    )['relations']
    critic_value_relation = agent.network.select('critic')(
        observations, value_goals, actions, method='relation_diagnostic'
    )['relations']
    critic_actor_relation = agent.network.select('critic')(
        observations, actor_goals, actions, method='relation_diagnostic'
    )['relations']
    target_value_relation = agent.network.select('target_critic')(
        observations, value_goals, actions, method='relation_diagnostic'
    )['relations']
    evaluation_relation = agent.network.select('actor')(
        observations, value_goals, method='relation_diagnostic'
    )['relations']
    actor_relation = np.asarray(actor_relation)
    value_relation = np.asarray(value_relation)
    critic_value_relation = np.asarray(critic_value_relation)
    critic_actor_relation = np.asarray(critic_actor_relation)
    target_value_relation = np.asarray(target_value_relation)
    evaluation_relation = np.asarray(evaluation_relation)
    _require(not np.array_equal(actor_relation, value_relation), 'Actor and value goals did not alter actual relation tensors')
    _require(np.array_equal(critic_value_relation[0], value_relation), 'Critic TD must use value goals')
    _require(np.array_equal(target_value_relation[0], value_relation), 'Next value/target critic must use value goals')
    _require(np.array_equal(critic_actor_relation[0], actor_relation), 'Actor-loss critic must use actor goals')
    _require(np.array_equal(evaluation_relation, value_relation), 'Evaluation actor must use passed evaluation goal')
    return {
        'actor_relation_shape': list(actor_relation.shape),
        'value_relation_shape': list(value_relation.shape),
        'critic_relation_shape': list(critic_value_relation.shape),
        'actor_value_relations_differ': True,
        'value_and_critic_td_match': True,
        'actor_goal_critic_match': True,
        'evaluation_passed_goal_match': True,
    }


def check_audit_artifact(path):
    artifact = _read_json(path)
    _require(artifact.get('audit_seed') == AUDIT_SEED, 'audit seed mismatch')
    _require(artifact.get('audit_sample_count') == AUDIT_SAMPLES, 'audit sample count mismatch')
    source = artifact.get('source', {})
    _require(
        source.get('source_status') == 'git_status_not_read_due_to_user_only_git_policy',
        'Git/source status must truthfully record the user-only Git policy',
    )
    task_result = {}
    for task in ('puzzle', 'cube', 'scene'):
        item = artifact.get('tasks', {}).get(task, {})
        _require(item.get('environment') == TASK_SPECS[task.upper()]['env'], f'audit {task} environment')
        sampling = item.get('gcdataset_sampling', {})
        _require(sampling.get('method', '').startswith('GCDataset.sample('), f'audit {task} direct sample method')
        _require(sampling.get('sample_count') == AUDIT_SAMPLES, f'audit {task} sample count')
        paired = sampling.get('same_seed_mode_paired_sampling', {})
        _require(paired.get('all_trace_equal') is True, f'audit {task} paired trace')
        _require(paired.get('all_goal_arrays_equal') is True, f'audit {task} paired goals')
        for distribution in ('actor_goal', 'value_goal', 'td_next_value_goal'):
            controls = item.get('distributions', {}).get(distribution, {}).get('controls', {})
            _require(controls.get('zero_tensor_all_zero') is True, f'audit {task}/{distribution} zero')
            for control in controls.get('per_type', {}).values():
                _require(control.get('correct_shuffled_edge_count_equal_all_samples') is True, f'audit {task}/{distribution} edge parity')
        task_result[task] = {
            'dataset_sha256': item.get('dataset_identity', {}).get('sha256'),
            'paired_sampling': paired,
        }
    cube = artifact['tasks']['cube'].get('cube_geometry_audit', {})
    _require(
        cube.get('required_statement') == 'CUBE RELATION THRESHOLDS NOT FROZEN',
        'Phase-1 audit must preserve its pre-freeze Cube threshold statement',
    )
    fraction = cube.get('shuffle_semantic_validity', {}).get('real_batch_fraction', 0.0)
    _require(float(fraction) > 0.0, 'Cube shuffled control has no real-batch semantic change')
    task_result['cube']['real_batch_correct_shuffled_difference_fraction'] = float(fraction)
    return task_result


def check_sweep_readiness(study_path):
    from tools import sweep

    jobs = sweep._jobs(study_path, '/tmp/m20a_doctor_nonexecuting_run_root')
    _require(len(jobs) == 18, f'Sweep should see 18 formal jobs, got {len(jobs)}')
    _require(all(job['status'] == 'planned' for job in jobs), 'Formal Phase-2 job is not planned')
    _require(all(job['executable'] is True for job in jobs), 'Formal executable flag drift')
    return {'jobs': len(jobs), 'planned': sum(job['status'] == 'planned' for job in jobs)}


def check_no_conflicting_formal_artifacts(run_root):
    path = Path(run_root) / STUDY_ID
    _require(not path.exists(), f'Formal M20A run artifact exists: {path}')
    return {'formal_m20a_run_root_absent': True, 'checked_path': str(path)}


def check_expected_source_commit(expected_source_commit):
    """Verify frozen-worktree provenance only when a commit is supplied."""

    if expected_source_commit is None:
        return {'checked': False, 'reason': 'expected source commit not supplied before commit'}
    observed = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, check=True, text=True,
    ).stdout.strip()
    _require(observed == expected_source_commit, f'Source commit mismatch: {observed} != {expected_source_commit}')
    cleanliness = subprocess.run(
        ['git', 'status', '--porcelain'], capture_output=True, check=True, text=True,
    ).stdout
    _require(not cleanliness, 'Frozen worktree is dirty')
    return {'checked': True, 'expected_source_commit': expected_source_commit, 'observed_source_commit': observed, 'worktree_clean': True}


def check_smoke_artifact(path):
    item = _read_json(path)
    _require(item.get('formal_training_started') is False, 'Smoke artifact must not claim formal training')
    for task in ('puzzle', 'cube', 'scene'):
        entry = item.get('tasks', {}).get(task, {})
        _require(entry.get('updates') == 2, f'Smoke {task}: expected two updates')
        _require(entry.get('all_losses_finite') is True, f'Smoke {task}: finite losses')
        _require(entry.get('gradients_finite') is True, f'Smoke {task}: finite gradients')
        _require(entry.get('parameters_changed') is True, f'Smoke {task}: parameters did not change')
        _require(entry.get('target_update_valid') is True, f'Smoke {task}: target update')
        _require(entry.get('checkpoint_roundtrip') is True, f'Smoke {task}: checkpoint roundtrip')
    return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=STUDY_DEFAULT)
    parser.add_argument('--audit', default=AUDIT_DEFAULT)
    parser.add_argument('--smoke', default=SMOKE_DEFAULT)
    parser.add_argument('--run-root', default=RUN_ROOT_DEFAULT)
    parser.add_argument('--report', default=REPORT_DEFAULT)
    parser.add_argument('--expected-source-commit')
    parser.add_argument('--skip-smoke-artifact', action='store_true')
    args = parser.parse_args(argv)
    report = {
        'doctor': 'm20a_phase2_freeze',
        'phase1_git_provenance': 'Phase-1 audit retains its truthful user-only Git source-status record.',
        'source_commit': check_expected_source_commit(args.expected_source_commit),
        'study': check_study_schema(args.study),
        'environments': check_environment_creation(),
        'entity_and_static_oracles': check_entity_parsers_and_static_oracles(),
        'cube_synthetic_oracles': check_cube_synthetic_oracles(),
        'relation_augmenter_and_readout': check_relation_augmenter_and_readout(),
        'initialization_and_accounting': check_initialization_and_accounting(args.study),
        'call_specific_goal_wiring': check_call_specific_goal_wiring(args.study),
        'audit_artifact': check_audit_artifact(args.audit),
        'sweep_readiness': check_sweep_readiness(args.study),
        'formal_artifact_guard': check_no_conflicting_formal_artifacts(args.run_root),
    }
    if not args.skip_smoke_artifact:
        report['tiny_real_data_smoke'] = check_smoke_artifact(args.smoke)
    _write_json(args.report, report)
    print('M20A PHASE-2 FREEZE: PASS')
    print('FORMAL TRAINING: READY')


if __name__ == '__main__':
    main()
