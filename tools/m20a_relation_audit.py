"""Run the M20A Phase-1 real GCIQL relation-prevalence audit.

This tool is intentionally an audit, not a training launcher.  It invokes
the production :meth:`GCDataset.sample` pipeline exactly once per treatment
replica, writes only the requested Phase-1 audit artifacts, and never creates
a RunContext, checkpoint, or formal M20A run directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Keep audit execution usable on a host without a visible CUDA device.  An
# explicitly supplied user setting always wins.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np

from impls.main import _make_config, _parse_args
from impls.representation.manipulation import parse_cube_observation
from impls.representation.relations import (
    CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
    PUZZLE_4X4_SHUFFLE_PERMUTATION,
    SCENE_ENTITY_ORDER,
    build_cube_relations,
    build_puzzle_relations,
    build_scene_relations,
    puzzle_shuffle_diagnostics,
    relation_edge_counts,
)
from impls.utils.datasets import GCDataset
from impls.utils.env_utils import make_env_and_datasets


AUDIT_SEED = 20_020
AUDIT_SAMPLE_COUNT = 100_000
DATASET_ROOT_DEFAULT = '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
STUDY_PATH_DEFAULT = 'experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml'
OUTPUT_DEFAULT = 'docs/9-8/M20A_phase1_relation_audit.json'
MARKDOWN_DEFAULT = 'docs/9-8/M20A_phase1_relation_audit.md'

# These are deliberately diagnostic candidates only.  They are not stored as
# a formal Cube configuration value and are never selected by RL performance.
CUBE_SUPPORT_CANDIDATES = (
    {'epsilon_xy': 0.01, 'epsilon_z': 0.005},
    {'epsilon_xy': 0.02, 'epsilon_z': 0.010},
    {'epsilon_xy': 0.03, 'epsilon_z': 0.020},
)
CUBE_CONFLICT_CANDIDATES = (0.02, 0.04, 0.06)
CUBE_AUDIT_REFERENCE = {
    'current_support_epsilon_xy': 0.02,
    'current_support_epsilon_z': 0.010,
    'goal_support_epsilon_xy': 0.02,
    'goal_support_epsilon_z': 0.010,
    'goal_conflict_radius': 0.04,
}

TASKS = {
    'puzzle': {
        'environment': 'puzzle-4x4-play-v0',
        'config_id': 'M20A-PUZZLE-C-M',
        'relation_names': ('toggle',),
    },
    'cube': {
        'environment': 'cube-triple-play-v0',
        'config_id': 'M20A-CUBE-C-M',
        'relation_names': ('current_support', 'goal_support', 'goal_conflict'),
    },
    'scene': {
        'environment': 'scene-play-v0',
        'config_id': 'M20A-SCENE-C-M',
        'relation_names': ('controls',),
    },
}


def _jsonable(value):
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_identity(dataset_root, environment):
    path = Path(dataset_root).resolve() / f'{environment}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Missing M20A train dataset: {path}')
    stat = path.stat()
    return {
        'path': str(path),
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'sha256': _sha256_file(path),
        'identity_kind': 'full_file_sha256_plus_stat',
    }


def _percentile_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError('Cannot summarize an empty audit vector')
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'max': float(np.max(values)),
    }


def relation_metrics(relations, relation_names):
    """Return per-type prevalence with explicit dense-tensor denominators."""

    relations = np.asarray(relations, dtype=np.float64)
    if relations.ndim != 4:
        raise ValueError(f'Expected [B,T,T,K] relation tensor, got {relations.shape}')
    batch_size, num_entities, targets, num_types = relations.shape
    if num_entities != targets or len(relation_names) != num_types:
        raise ValueError(
            'Relation metric shape/name mismatch: '
            f'{relations.shape} vs {relation_names!r}'
        )
    edge_counts = np.asarray(relation_edge_counts(relations))
    result = {
        'batch_size': int(batch_size),
        'num_entities': int(num_entities),
        'num_relation_types': int(num_types),
        'global_activation_density': float(np.mean(relations)),
        'per_type': {},
    }
    for index, name in enumerate(relation_names):
        active = edge_counts[:, index]
        outgoing = relations[:, :, :, index].sum(axis=2)
        incoming = relations[:, :, :, index].sum(axis=1)
        result['per_type'][name] = {
            'active_edges_per_sample': _percentile_summary(active),
            'fraction_samples_with_at_least_one_edge': float(np.mean(active >= 1)),
            'global_activation_density': float(np.mean(relations[:, :, :, index])),
            'per_entity_outgoing_degree_mean': [float(value) for value in outgoing.mean(axis=0)],
            'per_entity_incoming_degree_mean': [float(value) for value in incoming.mean(axis=0)],
        }
    return result


def control_metrics(correct, shuffled, zero, relation_names):
    correct = np.asarray(correct)
    shuffled = np.asarray(shuffled)
    zero = np.asarray(zero)
    if correct.shape != shuffled.shape or correct.shape != zero.shape:
        raise ValueError('Correct/Shuffled/Zero shape mismatch in control audit')
    correct_counts = np.asarray(relation_edge_counts(correct))
    shuffled_counts = np.asarray(relation_edge_counts(shuffled))
    zero_counts = np.asarray(relation_edge_counts(zero))
    per_type = {}
    for index, name in enumerate(relation_names):
        equal = correct_counts[:, index] == shuffled_counts[:, index]
        per_type[name] = {
            'correct_shuffled_edge_count_equal_fraction': float(np.mean(equal)),
            'correct_shuffled_edge_count_equal_all_samples': bool(np.all(equal)),
            'zero_edge_count_all_zero': bool(np.all(zero_counts[:, index] == 0)),
        }
    per_sample_different = np.any(correct != shuffled, axis=(1, 2, 3))
    return {
        'per_type': per_type,
        'correct_vs_shuffled_tensor_different_fraction': float(np.mean(per_sample_different)),
        'correct_vs_shuffled_tensor_different_any': bool(np.any(per_sample_different)),
        'zero_tensor_all_zero': bool(np.all(zero == 0)),
    }


def _sampling_trace_equal(left, right):
    return {
        key: bool(np.array_equal(left[key], right[key]))
        for key in ('transition_indices', 'value_goal_indices', 'actor_goal_indices')
    }


def _batch_fields_equal(left, right):
    return {
        key: bool(np.array_equal(np.asarray(left[key]), np.asarray(right[key])))
        for key in ('observations', 'next_observations', 'actor_goals', 'value_goals')
    }


def _direct_paired_sample(dataset, config):
    """Call actual GCDataset.sample three times with independently reset RNGs."""

    replicas = []
    for _ in ('zero', 'correct', 'shuffled'):
        wrapper = GCDataset(dataset, config, rng=np.random.default_rng(AUDIT_SEED))
        replicas.append(wrapper.sample(AUDIT_SAMPLE_COUNT, return_sampling_trace=True))
    (zero_batch, zero_trace), (correct_batch, correct_trace), (shuffled_batch, shuffled_trace) = replicas
    paired = {
        'zero_vs_correct_trace': _sampling_trace_equal(zero_trace, correct_trace),
        'zero_vs_shuffled_trace': _sampling_trace_equal(zero_trace, shuffled_trace),
        'zero_vs_correct_batch_fields': _batch_fields_equal(zero_batch, correct_batch),
        'zero_vs_shuffled_batch_fields': _batch_fields_equal(zero_batch, shuffled_batch),
    }
    paired['all_trace_equal'] = bool(all(
        value for group in (
            paired['zero_vs_correct_trace'], paired['zero_vs_shuffled_trace']
        ) for value in group.values()
    ))
    paired['all_goal_arrays_equal'] = bool(all(
        value for group in (
            paired['zero_vs_correct_batch_fields'], paired['zero_vs_shuffled_batch_fields']
        ) for value in group.values()
    ))
    # The batches are equal by construction.  Retain only one real production
    # sample for all relation analysis, then promptly let duplicate arrays go.
    del zero_batch, zero_trace, shuffled_batch, shuffled_trace, replicas
    gc.collect()
    return correct_batch, correct_trace, paired


def _puzzle_relations(state, goal, mode):
    return np.asarray(build_puzzle_relations(
        state, goal, mode=mode, rows=4, cols=4,
        shuffle_permutation=PUZZLE_4X4_SHUFFLE_PERMUTATION,
    ))


def _scene_relations(state, goal, mode):
    return np.asarray(build_scene_relations(state, goal, mode=mode))


def _cube_parts(state, goal):
    _, state_cubes = parse_cube_observation(
        state, num_cubes=3, robot_dim=19, cube_feature_dim=9
    )
    _, goal_cubes = parse_cube_observation(
        goal, num_cubes=3, robot_dim=19, cube_feature_dim=9
    )
    return np.asarray(state_cubes), np.asarray(goal_cubes)


def _cube_relations(state, goal, mode, thresholds=CUBE_AUDIT_REFERENCE):
    state_cubes, goal_cubes = _cube_parts(state, goal)
    return np.asarray(build_cube_relations(
        state_cubes,
        goal_cubes,
        mode=mode,
        current_support_epsilon_xy=thresholds.get('current_support_epsilon_xy'),
        current_support_epsilon_z=thresholds.get('current_support_epsilon_z'),
        goal_support_epsilon_xy=thresholds.get('goal_support_epsilon_xy'),
        goal_support_epsilon_z=thresholds.get('goal_support_epsilon_z'),
        conflict_radius=thresholds.get('goal_conflict_radius'),
        shuffle_derangement=CUBE_TRIPLE_SHUFFLE_DERANGEMENT,
        cube_height=0.04,
        xyz_scaler=10.0,
    ))


def _pair_geometry(cubes):
    """Describe all ordered distinct cube pairs in physical XYZ units."""

    xyz = np.asarray(cubes, dtype=np.float64)[..., :3] / 10.0
    source = xyz[:, :, None, :]
    target = xyz[:, None, :, :]
    delta_xy = np.linalg.norm(source[..., :2] - target[..., :2], axis=-1)
    vertical_delta = target[..., 2] - source[..., 2]
    nonself = np.broadcast_to(
        ~np.eye(xyz.shape[1], dtype=bool)[None, :, :], delta_xy.shape
    )
    return {
        'ordered_nonself_pair_count': int(np.sum(nonself) * xyz.shape[0]),
        'delta_xy': _percentile_summary(delta_xy[nonself]),
        'vertical_delta_z': _percentile_summary(vertical_delta[nonself]),
        'fraction_upper_pairs': float(np.mean(vertical_delta[nonself] > 0.0)),
    }


def _cube_sensitivity(state, goal, relation_names):
    support_rows = []
    for candidate in CUBE_SUPPORT_CANDIDATES:
        thresholds = {
            'current_support_epsilon_xy': candidate['epsilon_xy'],
            'current_support_epsilon_z': candidate['epsilon_z'],
            'goal_support_epsilon_xy': candidate['epsilon_xy'],
            'goal_support_epsilon_z': candidate['epsilon_z'],
            'goal_conflict_radius': CUBE_AUDIT_REFERENCE['goal_conflict_radius'],
        }
        metrics = relation_metrics(
            _cube_relations(state, goal, 'correct', thresholds), relation_names
        )
        support_rows.append({
            **candidate,
            'current_support': metrics['per_type']['current_support'],
            'goal_support': metrics['per_type']['goal_support'],
        })
    conflict_rows = []
    for radius in CUBE_CONFLICT_CANDIDATES:
        thresholds = dict(CUBE_AUDIT_REFERENCE, goal_conflict_radius=radius)
        metrics = relation_metrics(
            _cube_relations(state, goal, 'correct', thresholds), relation_names
        )
        conflict_rows.append({
            'goal_conflict_radius': radius,
            'goal_conflict': metrics['per_type']['goal_conflict'],
        })
    state_cubes, goal_cubes = _cube_parts(state, goal)
    return {
        'state_pair_geometry': _pair_geometry(state_cubes),
        'goal_pair_geometry': _pair_geometry(goal_cubes),
        'support_candidates': support_rows,
        'conflict_candidates': conflict_rows,
    }


def _relations_for_distribution(task_name, state, goal):
    task = TASKS[task_name]
    if task_name == 'puzzle':
        builder = _puzzle_relations
    elif task_name == 'cube':
        builder = _cube_relations
    elif task_name == 'scene':
        builder = _scene_relations
    else:  # pragma: no cover - module-level TASKS fixes the domain.
        raise ValueError(task_name)
    return {
        mode: builder(state, goal, mode)
        for mode in ('zero', 'correct', 'shuffled')
    }, task['relation_names']


def _sampling_config(config):
    keys = (
        'value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal',
        'value_geom_sample', 'actor_p_curgoal', 'actor_p_trajgoal',
        'actor_p_randomgoal', 'actor_geom_sample', 'discount', 'gc_negative',
        'p_aug', 'frame_stack', 'dataset_class', 'alpha',
    )
    return {key: _jsonable(config.get(key)) for key in keys}


def _source_metadata():
    try:
        import ogbench
        ogbench_file = str(Path(ogbench.__file__).resolve())
    except (ImportError, AttributeError):  # pragma: no cover - env creation catches this too.
        ogbench_file = None
    try:
        ogbench_version = importlib.metadata.version('ogbench')
    except importlib.metadata.PackageNotFoundError:
        ogbench_version = 'installed-local-source-version-unavailable'
    return {
        'source_status': 'git_status_not_read_due_to_user_only_git_policy',
        'actual_starting_head': 'not independently read by Codex; user-only Git policy',
        'ogbench_module': ogbench_file,
        'ogbench_distribution_version': ogbench_version,
        'source_semantics_audited': {
            'puzzle': 'PuzzleEnv.post_step toggles self plus in-bounds cardinal neighbors.',
            'cube': 'CubeEnv observation is robot(19)+three cube(9); same slot pairs state and goal after reset role permutation.',
            'scene': 'Scene order is cube, button_0, button_1, drawer, window; button_0 controls drawer and button_1 controls window.',
        },
    }


def audit_task(task_name, *, study_path, dataset_root):
    task = TASKS[task_name]
    from impls.experiment import prepare_run_design

    study, configuration = prepare_run_design(study_path, task['config_id'])
    args = _parse_args(['--agent', 'gciql'])
    config = _make_config(args, configuration=configuration)
    environment = task['environment']
    identity = _dataset_identity(dataset_root, environment)
    env, train_dataset, _ = make_env_and_datasets(
        environment,
        frame_stack=config['frame_stack'],
        seed=AUDIT_SEED,
        dataset_seed=AUDIT_SEED,
        dataset_dir=dataset_root,
    )
    try:
        batch, trace, paired_sampling = _direct_paired_sample(train_dataset, config)
        distributions = {
            'actor_goal': (batch['observations'], batch['actor_goals']),
            'value_goal': (batch['observations'], batch['value_goals']),
            'td_next_value_goal': (batch['next_observations'], batch['value_goals']),
        }
        distribution_results = {}
        cube_sensitivity = {}
        for distribution_name, (state, goal) in distributions.items():
            relation_sets, relation_names = _relations_for_distribution(task_name, state, goal)
            distribution_results[distribution_name] = {
                'correct': relation_metrics(relation_sets['correct'], relation_names),
                'shuffled': relation_metrics(relation_sets['shuffled'], relation_names),
                'zero': relation_metrics(relation_sets['zero'], relation_names),
                'controls': control_metrics(
                    relation_sets['correct'], relation_sets['shuffled'],
                    relation_sets['zero'], relation_names,
                ),
            }
            if task_name == 'cube':
                cube_sensitivity[distribution_name] = _cube_sensitivity(
                    state, goal, relation_names
                )
        task_result = {
            'environment': environment,
            'configuration_id_for_sampling': configuration.config_id,
            'dataset_identity': identity,
            'gcdataset_sampling': {
                'method': 'GCDataset.sample(batch_size=100000, return_sampling_trace=True)',
                'audit_seed': AUDIT_SEED,
                'sample_count': AUDIT_SAMPLE_COUNT,
                'resolved_gciql_goal_sampling': _sampling_config(config),
                'same_seed_mode_paired_sampling': paired_sampling,
                'trace_fingerprint': {
                    key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                    for key, value in trace.items()
                },
            },
            'distributions': distribution_results,
        }
        if task_name == 'puzzle':
            matrix = _puzzle_relations(batch['observations'][:1], batch['actor_goals'][:1], 'correct')[0]
            task_result['puzzle_topology'] = {
                'correct_relation_matrix': matrix[:, :, 0].astype(int).tolist(),
                'edge_list_source_to_target': [
                    [int(source), int(target)]
                    for source, target in np.argwhere(matrix[:, :, 0] > 0)
                ],
                'outgoing_toggle_degree_vector': matrix[:, :, 0].sum(axis=1).astype(int).tolist(),
                'expected_degree_classes': {'corner': 3, 'edge_noncorner': 4, 'interior': 5},
                'shuffle': puzzle_shuffle_diagnostics(
                    4, 4, PUZZLE_4X4_SHUFFLE_PERMUTATION
                ),
            }
        if task_name == 'scene':
            task_result['scene_topology'] = {
                'entity_order': list(SCENE_ENTITY_ORDER),
                'correct_edges_source_to_target': [['button_0', 'drawer'], ['button_1', 'window']],
                'shuffled_edges_source_to_target': [['button_0', 'window'], ['button_1', 'drawer']],
                'source_audit_evidence': 'SceneEnv._apply_button_states: button_0 lock/unlock drawer; button_1 lock/unlock window.',
            }
        if task_name == 'cube':
            task_result['cube_geometry_audit'] = {
                'threshold_status': 'UNFROZEN_PHASE1',
                'required_statement': 'CUBE RELATION THRESHOLDS NOT FROZEN',
                'candidate_provenance': {
                    'xy': '0.02 is the audited stack-alignment source constant; 0.01/0.03 are non-performance sensitivity brackets.',
                    'z': '0.04 cube height is simulator geometry; 0.005/0.010/0.020 are non-performance vertical-tolerance sensitivity candidates.',
                    'conflict': '0.04 is audited Cube success-radius source constant; 0.02/0.06 are non-performance sensitivity brackets.',
                },
                'audit_reference_candidate_only': CUBE_AUDIT_REFERENCE,
                'sensitivity_by_distribution': cube_sensitivity,
                'shuffle_semantic_validity': {
                    'endpoint_derangement': list(CUBE_TRIPLE_SHUFFLE_DERANGEMENT),
                    'token_features_not_permuted': True,
                    'slot_identity_embedding': False,
                    'semantic_alignment_gate': (
                        'valid_if_real_batch_correct_vs_shuffled_tensor_different_fraction_gt_zero'
                    ),
                    'real_batch_fraction': distribution_results['value_goal']['controls'][
                        'correct_vs_shuffled_tensor_different_fraction'
                    ],
                },
            }
        return task_result
    finally:
        close = getattr(env, 'close', None)
        if close is not None:
            close()
        gc.collect()


def _brief_metric(task):
    actor = task['distributions']['actor_goal']['correct']['per_type']
    value = task['distributions']['value_goal']['correct']['per_type']
    rows = []
    for name in actor:
        rows.append(
            f'| {name} | {actor[name]["active_edges_per_sample"]["mean"]:.6f} | '
            f'{value[name]["active_edges_per_sample"]["mean"]:.6f} | '
            f'{actor[name]["fraction_samples_with_at_least_one_edge"]:.6f} | '
            f'{value[name]["fraction_samples_with_at_least_one_edge"]:.6f} |'
        )
    return '\n'.join(rows)


def render_markdown(artifact):
    lines = [
        '# M20A Phase-1 relation prevalence audit',
        '',
        'This artifact is a data/semantic audit only. It contains no formal RL training result.',
        '',
        f'- Audit seed: `{artifact["audit_seed"]}`',
        f'- Samples per environment: `{artifact["audit_sample_count"]}` via real `GCDataset.sample()`',
        f'- Dataset root: `{artifact["dataset_root"]}`',
        f'- Source status: `{artifact["source"]["source_status"]}`',
        '',
    ]
    for task_name, task in artifact['tasks'].items():
        lines.extend([
            f'## {task_name}',
            '',
            f'- Environment: `{task["environment"]}`',
            f'- Dataset SHA-256: `{task["dataset_identity"]["sha256"]}`',
            f'- Same-seed Z/C/S trace parity: `{task["gcdataset_sampling"]["same_seed_mode_paired_sampling"]["all_trace_equal"]}`',
            '',
            '| Relation | actor mean edges/sample | value mean edges/sample | actor fraction ≥1 | value fraction ≥1 |',
            '| --- | ---: | ---: | ---: | ---: |',
            _brief_metric(task),
            '',
        ])
        if task_name == 'puzzle':
            shuffle = task['puzzle_topology']['shuffle']
            lines.extend([
                f'- Fixed shuffle permutation: `{shuffle["permutation"]}`',
                f'- Correct/Shuffled differing adjacency entries: `{shuffle["differing_adjacency_entries"]}`',
                '',
            ])
        if task_name == 'scene':
            lines.extend([
                '- Correct controls: `button_0 → drawer`, `button_1 → window`.',
                '- Shuffled controls swap only those targets.',
                '',
            ])
        if task_name == 'cube':
            cube = task['cube_geometry_audit']
            lines.extend([
                f'**{cube["required_statement"]}**',
                '',
                f'- Audit-only reference candidate: `{cube["audit_reference_candidate_only"]}`',
                f'- Value-goal real-batch Correct/Shuffled tensor-difference fraction: `{cube["shuffle_semantic_validity"]["real_batch_fraction"]:.6f}`',
                '',
            ])
    lines.extend([
        '## USER DECISIONS REQUIRED BEFORE PHASE 2',
        '',
        '- `current_support_epsilon_xy`',
        '- `current_support_epsilon_z`',
        '- `goal_support_epsilon_xy`',
        '- `goal_support_epsilon_z`',
        '- `goal_conflict_radius`',
        '',
        'The complete machine-readable sensitivity tables are in the paired JSON artifact. No candidate has been frozen by this audit.',
        '',
    ])
    return '\n'.join(lines)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(_jsonable(value), file, indent=2, sort_keys=True)
        file.write('\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=STUDY_PATH_DEFAULT)
    parser.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    parser.add_argument('--output', default=OUTPUT_DEFAULT)
    parser.add_argument('--markdown-output', default=MARKDOWN_DEFAULT)
    parser.add_argument('--sample-count', type=int, default=AUDIT_SAMPLE_COUNT)
    args = parser.parse_args(argv)
    if args.sample_count != AUDIT_SAMPLE_COUNT:
        raise ValueError(
            f'M20A Phase-1 audit is frozen at sample_count={AUDIT_SAMPLE_COUNT}, '
            f'got {args.sample_count}'
        )
    artifact = {
        'artifact_schema': 'm20a_phase1_relation_audit_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'audit_seed': AUDIT_SEED,
        'audit_sample_count': AUDIT_SAMPLE_COUNT,
        'dataset_root': str(Path(args.dataset_root).resolve()),
        'source': _source_metadata(),
        'tasks': {},
    }
    for task_name in ('puzzle', 'cube', 'scene'):
        print(f'[M20A audit] sampling {task_name}: {AUDIT_SAMPLE_COUNT} real GCDataset entries', flush=True)
        artifact['tasks'][task_name] = audit_task(
            task_name,
            study_path=args.study,
            dataset_root=args.dataset_root,
        )
    write_json(args.output, artifact)
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(artifact) + '\n')
    print(f'M20A relation audit written: {Path(args.output).resolve()}')
    print(f'M20A relation audit markdown written: {markdown_path.resolve()}')
    return artifact


if __name__ == '__main__':
    main()
