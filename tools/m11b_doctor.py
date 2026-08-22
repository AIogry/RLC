"""M11B static doctor, real-shape smoke checks, and non-executing dry-run.

This tool intentionally never launches ``impls.main`` as a training worker.
The runtime probes construct agents, perform one finite update, exercise action
inference, and test checkpoint serialization in temporary directories only.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import agents
from impls.experiment import (
    config_fingerprint,
    load_study,
    make_run_path,
    prepare_run_design,
)
from impls.experiment.m11b import (
    ALL_ENVIRONMENTS,
    ANCHOR_ENVIRONMENT,
    CANONICAL_SOURCE,
    CRL_CONDITIONS,
    HIQL_CONDITIONS,
    NEW_ENVIRONMENTS,
    PROTOCOL,
    aggregate_factorial_rows,
    canonical_hyperparameters,
    computation_slots,
    config_specs,
    condition_label,
    m11b_agent_overrides,
    normalized_eval_auc,
    resolved_fingerprint_payload,
    spec_by_id,
)
from impls.main import (
    _accounting_consistency_audit,
    _actor_parameter_accounting,
    _computation_slot_accounting,
    _make_config,
    _parse_args,
    _resolved_compute_snapshot,
)
from impls.utils.checkpointing import sha256_file
from impls.utils.flax_utils import restore_agent, save_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / 'experiments' / 'M11B_cross_task_computation' / 'study.yaml'
DEFAULT_DATASET = Path(os.environ.get('OGBENCH_DATASET_DIR', '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'))
DEFAULT_RUN_ROOT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')


def _underlying_env_id(environment: str) -> str:
    for suffix in ('-navigate-v0', '-stitch-v0'):
        if environment.endswith(suffix):
            return environment[: -len(suffix)] + '-v0'
    raise ValueError(f'Unsupported M11B dataset reference: {environment!r}')


def _finite_tree(value) -> bool:
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(value))


def _dataset_header(dataset_root: Path, environment: str) -> dict:
    result = {'environment': environment}
    for split in ('train', 'val'):
        path = dataset_root / (f'{environment}.npz' if split == 'train' else f'{environment}-val.npz')
        result[f'{split}_path'] = str(path)
        result[f'{split}_exists'] = path.is_file()
        if not path.is_file():
            continue
        with np.load(path, mmap_mode='r', allow_pickle=False) as data:
            result[f'{split}_keys'] = sorted(data.files)
            result[f'{split}_observation_shape'] = list(data['observations'].shape)
            result[f'{split}_action_shape'] = list(data['actions'].shape)
            result[f'{split}_terminal_shape'] = list(data['terminals'].shape)
    return result


def validate_environment_references(study, dataset_root: Path) -> list[dict]:
    import gymnasium
    import ogbench  # noqa: F401  # registration is the evidence we need.

    result = []
    for requested in study.data['environments']:
        underlying = _underlying_env_id(requested)
        row = _dataset_header(dataset_root, requested)
        row['requested_id'] = requested
        row['underlying_ogbench_env'] = underlying
        try:
            spec = gymnasium.spec(underlying)
            row['environment_registered'] = True
            row['max_episode_steps'] = spec.max_episode_steps
        except Exception as error:
            row['environment_registered'] = False
            row['environment_error'] = str(error)
        row['resolved_canonical_id'] = requested if row['train_exists'] and row['val_exists'] else None
        row['valid'] = bool(
            row['environment_registered'] and row['train_exists'] and row['val_exists']
        )
        result.append(row)
    return result


def validate_structure(study):
    expected = config_specs()
    config_paths = sorted((study.path.parent / 'configs').glob('M11B-C*.yaml'))
    loaded = [prepare_run_design(study.path, path)[1] for path in config_paths]
    by_id = {configuration.config_id: configuration for configuration in loaded}
    errors = []
    if len(config_paths) != 34:
        errors.append(f'expected 34 config files, found {len(config_paths)}')
    if len(by_id) != len(loaded):
        errors.append('config IDs are not unique')
    if [row['config_id'] for row in expected] != sorted(by_id):
        errors.append('config ID set/order does not match permanent M11B scheme')
    semantic_keys = []
    rows = []
    for row in expected:
        configuration = by_id.get(row['config_id'])
        if configuration is None:
            errors.append(f'missing {row["config_id"]}')
            continue
        data = configuration.data
        if data.get('environment') != row['environment']:
            errors.append(f'{row["config_id"]}: environment mismatch')
        if data.get('algorithm') != row['algorithm'] or data.get('condition') != row['condition']:
            errors.append(f'{row["config_id"]}: algorithm/condition mismatch')
        if data.get('semantic_condition') != row['semantic_condition']:
            errors.append(f'{row["config_id"]}: semantic condition mismatch')
        expected_label = row['semantic_label']
        if data.get('semantic_label') != expected_label:
            errors.append(f'{row["config_id"]}: semantic label mismatch')
        semantic_keys.append(data.get('semantic_label'))
        rows.append({**row, 'configuration': configuration})
    if len(semantic_keys) != len(set(semantic_keys)):
        errors.append('environment-scoped semantic labels are not unique')
    if tuple(study.data['environments']) != ALL_ENVIRONMENTS:
        errors.append('study environment order/set does not match M11B references')
    if study.data['seeds'] != [0]:
        errors.append(f'expected seed [0], found {study.data["seeds"]!r}')
    if study.data.get('primary_metric') != 'evaluation/overall_success':
        errors.append('primary metric is not evaluation/overall_success')
    if study.data.get('protocol') != {
        key: value for key, value in PROTOCOL.items()
        if key not in {'secondary_endpoints', 'auc_checkpoints', 'auc_interval', 'auc_rule'}
    }:
        # The YAML protocol intentionally stores the AUC fields nested, so
        # validate those fields separately below rather than requiring a
        # byte-identical representation.
        protocol = study.data.get('protocol', {})
        for key in ('train_steps', 'batch_size', 'log_interval', 'eval_interval', 'eval_episodes', 'save_interval'):
            if protocol.get(key) != PROTOCOL[key]:
                errors.append(f'protocol {key} mismatch')
        if protocol.get('eval_tasks') != 'all' or protocol.get('eval_temperature') != 0.0:
            errors.append('evaluation protocol mismatch')
        auc = protocol.get('auc', {})
        if auc.get('checkpoints') != list(PROTOCOL['auc_checkpoints']):
            errors.append('AUC checkpoints mismatch')
        if auc.get('interval') != PROTOCOL['auc_interval'] or auc.get('rule') != PROTOCOL['auc_rule']:
            errors.append('AUC interval/rule mismatch')
    return rows, errors


def _resolved_configs(study, rows):
    errors = []
    resolved = []
    for row in rows:
        configuration = row['configuration']
        args = _parse_args(['--agent', row['algorithm']])
        config = _make_config(args, configuration=configuration)
        expected = canonical_hyperparameters(row['algorithm'], row['environment'])
        for key, value in expected.items():
            actual = config[key]
            expected_value = tuple(value) if key.endswith('_dims') else value
            actual_value = tuple(actual) if key.endswith('_dims') else actual
            if actual_value != expected_value:
                errors.append(f'{row["config_id"]}: canonical {key} mismatch')
        expected_slots = computation_slots(row['algorithm'], row['condition'])
        actual_slots = _resolved_compute_snapshot(config)
        if actual_slots != expected_slots:
            errors.append(f'{row["config_id"]}: computation slot mismatch')
        encoded = json.dumps(actual_slots, sort_keys=True)
        if 'two_state' in encoded or 'H2L' in encoded or 'h_cycles' in encoded or 'l_cycles' in encoded:
            errors.append(f'{row["config_id"]}: TwoState/H2L leakage')
        for slot_name, slot in actual_slots.items():
            if slot.get('enabled') and slot.get('topology') == 'single_state':
                kwargs = slot.get('topology_kwargs', {})
                if kwargs.get('iterations') != 4 or kwargs.get('residual') is not False:
                    errors.append(f'{row["config_id"]}: SingleState K4 non-residual mismatch')
                if kwargs.get('state_dim') != 512 or kwargs.get('input_injection') != 'z_plus_x':
                    errors.append(f'{row["config_id"]}: SingleState frozen spec mismatch')
        resolved.append((row, config))
    return resolved, errors


def _synthetic_batch(algorithm, observations, actions):
    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)
    goals = jnp.flip(observations, axis=0)
    if algorithm == 'crl':
        return {
            'observations': observations,
            'value_goals': goals,
            'actor_goals': goals,
            'actions': actions,
        }
    return {
        'observations': observations,
        'next_observations': jnp.roll(observations, 1, axis=0),
        'low_actor_goals': goals,
        'actions': actions,
        'high_actor_goals': goals,
        'high_actor_targets': jnp.roll(goals, 1, axis=0),
        'value_goals': goals,
        'rewards': jnp.zeros((len(observations),), dtype=jnp.float32),
        'masks': jnp.ones((len(observations),), dtype=jnp.float32),
    }


def runtime_smoke(study, dataset_root: Path, resolved, *, max_probes=None):
    """Run real-dimension probes for FF and fully-computed conditions.

    The single-slot conditions are statically resolved for all 34 rows.  The
    real runtime matrix uses baseline and both-slots-enabled representatives
    for each algorithm/environment pair, which covers both FF and SS bodies
    without constructing four redundant 512-wide agents per cell.
    """

    import gymnasium
    import ogbench  # noqa: F401

    by_key = {(row['algorithm'], row['environment'], row['condition']): (row, config) for row, config in resolved}
    probes = []
    for environment in study.data['environments']:
        # The AntMaze-Large anchor intentionally contains only two fresh
        # baselines; its complete CRL/HIQL factorial is historical M11A/M9
        # context and is not re-run in M11B.
        if environment == ANCHOR_ENVIRONMENT:
            continue
        if not (dataset_root / f'{environment}.npz').is_file():
            continue
        with np.load(dataset_root / f'{environment}.npz', mmap_mode='r', allow_pickle=False) as data:
            observations = np.asarray(data['observations'][:2], dtype=np.float32)
            actions = np.asarray(data['actions'][:2], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]
        for algorithm, full_condition in (('crl', 'actor_critic_ss'), ('hiql', 'high_low_ss')):
            for condition in ('baseline', full_condition):
                if max_probes is not None and len(probes) >= max_probes:
                    return probes
                row, config = by_key[(algorithm, environment, condition)]
                env = gymnasium.make(_underlying_env_id(environment))
                try:
                    env_observation, _ = env.reset(seed=0)
                    if tuple(env.observation_space.shape) != tuple(observations.shape[1:]):
                        raise AssertionError(
                            f'{environment}: dataset observation {observations.shape[1:]} != '
                            f'env {env.observation_space.shape}'
                        )
                    if tuple(env.action_space.shape) != tuple(actions.shape[1:]):
                        raise AssertionError(
                            f'{environment}: dataset action {actions.shape[1:]} != '
                            f'env {env.action_space.shape}'
                        )
                    del env_observation
                    agent = agents[algorithm].create(
                        0, jnp.asarray(observations), jnp.asarray(actions), config
                    )
                    goals = jnp.asarray(np.asarray(observations[::-1], dtype=np.float32))
                    sampled = agent.sample_actions(
                        jnp.asarray(observations), goals, seed=jax.random.PRNGKey(17), temperature=0.0
                    )
                    if tuple(sampled.shape) != tuple(actions.shape):
                        raise AssertionError(f'{row["config_id"]}: action shape mismatch {sampled.shape}')
                    if not _finite_tree(sampled):
                        raise AssertionError(f'{row["config_id"]}: non-finite sampled action')
                    action_array = np.asarray(sampled)
                    if np.any(action_array < env.action_space.low - 1e-5) or np.any(action_array > env.action_space.high + 1e-5):
                        raise AssertionError(f'{row["config_id"]}: action bound violation')
                    updated, info = agent.update(_synthetic_batch(algorithm, observations, actions))
                    if not _finite_tree(info):
                        raise AssertionError(f'{row["config_id"]}: non-finite update info')
                    probes.append({
                        'config_id': row['config_id'],
                        'environment': environment,
                        'algorithm': algorithm,
                        'condition': condition,
                        'observation_shape': list(observations.shape[1:]),
                        'action_shape': list(actions.shape[1:]),
                        'one_finite_update': True,
                        'deterministic_action': True,
                        'action_bounds': True,
                        'updated_step': int(updated.network.step),
                    })
                finally:
                    env.close()
    return probes


def runtime_accounting_smoke(study, dataset_root: Path, resolved):
    """Exercise HIQL high-only, low-only, and joint accounting on real shapes."""

    dataset_environment = next(
        (
            environment for environment in NEW_ENVIRONMENTS
            if (dataset_root / f'{environment}.npz').is_file()
        ),
        None,
    )
    if dataset_environment is None:
        raise AssertionError('No M11B new-environment dataset is available for accounting smoke')
    with np.load(
        dataset_root / f'{dataset_environment}.npz',
        mmap_mode='r',
        allow_pickle=False,
    ) as data:
        observations = np.asarray(data['observations'][:2], dtype=np.float32)
        actions = np.asarray(data['actions'][:2], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]

    by_key = {
        (row['algorithm'], row['environment'], row['condition']): (row, config)
        for row, config in resolved
    }
    probes = []
    for condition in ('high_ss', 'low_ss', 'high_low_ss'):
        row, config = by_key[('hiql', dataset_environment, condition)]
        agent = agents['hiql'].create(
            0, jnp.asarray(observations), jnp.asarray(actions), config,
        )
        legacy = _actor_parameter_accounting(agent, config)
        generic = _computation_slot_accounting(agent, config)
        consistency = _accounting_consistency_audit(legacy, generic, config)
        enabled_slots = [
            slot_name for slot_name in ('high_actor', 'low_actor')
            if config['compute'][slot_name]['enabled']
        ]
        if condition == 'high_ss' and enabled_slots != ['high_actor']:
            raise AssertionError(f'{row["config_id"]}: unexpected high-only slots {enabled_slots}')
        if condition == 'low_ss' and enabled_slots != ['low_actor']:
            raise AssertionError(f'{row["config_id"]}: unexpected low-only slots {enabled_slots}')
        if condition == 'high_low_ss' and enabled_slots != ['high_actor', 'low_actor']:
            raise AssertionError(f'{row["config_id"]}: unexpected joint slots {enabled_slots}')
        for slot_name in enabled_slots:
            spec = config['compute'][slot_name]
            kwargs = spec['topology_kwargs']
            if (
                spec['topology'] != 'single_state'
                or kwargs.get('iterations') != 4
                or kwargs.get('state_dim') != 512
                or kwargs.get('residual') is not False
                or kwargs.get('input_injection') != 'z_plus_x'
            ):
                raise AssertionError(f'{row["config_id"]}: SingleState spec mismatch for {slot_name}')
        probes.append({
            'config_id': row['config_id'],
            'environment': dataset_environment,
            'condition': condition,
            'enabled_slots': enabled_slots,
            'resolved_single_state': {
                slot_name: {
                    'topology': config['compute'][slot_name]['topology'],
                    'iterations': config['compute'][slot_name]['topology_kwargs']['iterations'],
                    'state_dim': config['compute'][slot_name]['topology_kwargs']['state_dim'],
                    'residual': config['compute'][slot_name]['topology_kwargs']['residual'],
                    'input_injection': config['compute'][slot_name]['topology_kwargs']['input_injection'],
                }
                for slot_name in enabled_slots
            },
            'legacy_accounting': {
                slot_name: {
                    field: legacy[slot_name].get(field)
                    for field in ('topology', 'trainable_params', 'buffer_elements', 'state_dim', 'iterations')
                }
                for slot_name in enabled_slots
            },
            'generic_accounting': {
                slot_name: {
                    field: generic[slot_name].get(field)
                    for field in ('topology', 'trainable_params', 'buffer_elements', 'state_dim', 'iterations')
                }
                for slot_name in enabled_slots
            },
            'consistency': consistency,
        })
    return probes


def checkpoint_smoke(resolved):
    checks = []
    for algorithm, condition in (('crl', 'actor_critic_ss'), ('hiql', 'high_low_ss')):
        row, config = next(
            (
                item for item in resolved
                if item[0]['algorithm'] == algorithm
                and item[0]['environment'] in NEW_ENVIRONMENTS
                and item[0]['condition'] == condition
            ),
            (None, None),
        )
        if row is None:
            continue
        observation_dim = 4
        action_dim = 2
        observations = jnp.zeros((2, observation_dim), dtype=jnp.float32)
        actions = jnp.zeros((2, action_dim), dtype=jnp.float32)
        agent = agents[algorithm].create(0, observations, actions, config)
        with tempfile.TemporaryDirectory(prefix='m11b_checkpoint_') as directory:
            path = save_agent(agent, directory, 1)
            restored = restore_agent(agent, directory, 1)
            before = agent.sample_actions(observations, observations, seed=jax.random.PRNGKey(1), temperature=0.0)
            after = restored.sample_actions(observations, observations, seed=jax.random.PRNGKey(1), temperature=0.0)
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
            checks.append({'algorithm': algorithm, 'condition': condition, 'save_restore': True, 'sha256': sha256_file(path)})
    return checks


def synthetic_checks():
    auc = normalized_eval_auc({step: step / 1_000_000 for step in PROTOCOL['auc_checkpoints']})
    # The first checkpoint is 100k, so the linear sequence is 0.1..1.0
    # across the closed [100k, 1M] interval; its trapezoidal mean is 0.55.
    expected_auc = (0.1 + 1.0) / 2.0
    if not np.isclose(auc, expected_auc):
        raise AssertionError(f'AUC synthetic check failed: {auc} != {expected_auc}')
    rows = []
    for algorithm, conditions in (('crl', CRL_CONDITIONS), ('hiql', HIQL_CONDITIONS)):
        values = dict(zip(conditions, (0.2, 0.4, 0.5, 0.8)))
        for condition, value in values.items():
            rows.append({'algorithm': algorithm, 'environment': ANCHOR_ENVIRONMENT, 'condition': condition, 'final_success': value})
    aggregated = aggregate_factorial_rows(rows)
    if len(aggregated) != 2 or any(not np.isclose(row['interaction'], 0.1) for row in aggregated):
        raise AssertionError(f'factorial aggregation synthetic check failed: {aggregated}')
    return {'auc': auc, 'factorial_rows': aggregated}


def doctor(study_path: Path, dataset_root: Path, *, source_commit: str | None = None, max_runtime_probes=None):
    study = load_study(study_path)
    rows, structure_errors = validate_structure(study)
    resolved, resolution_errors = _resolved_configs(study, rows)
    environment_rows = validate_environment_references(study, dataset_root)
    runtime_errors = []
    runtime_probes = []
    runtime_accounting_probes = []
    checkpoint_checks = []
    synthetic = {}
    try:
        synthetic = synthetic_checks()
    except Exception as error:
        runtime_errors.append(f'synthetic checks: {error}')
    if not structure_errors and not resolution_errors:
        try:
            runtime_probes = runtime_smoke(study, dataset_root, resolved, max_probes=max_runtime_probes)
        except Exception as error:
            runtime_errors.append(f'real runtime smoke: {type(error).__name__}: {error}')
        try:
            runtime_accounting_probes = runtime_accounting_smoke(study, dataset_root, resolved)
        except Exception as error:
            runtime_errors.append(
                f'runtime accounting smoke: {type(error).__name__}: {error}'
            )
        try:
            checkpoint_checks = checkpoint_smoke(resolved)
        except Exception as error:
            runtime_errors.append(f'checkpoint smoke: {type(error).__name__}: {error}')
    dataset_errors = [
        f'{row["requested_id"]}: missing canonical train/val dataset or environment'
        for row in environment_rows if not row['valid']
    ]
    errors = structure_errors + resolution_errors + dataset_errors + runtime_errors
    report = {
        'study_id': study.study_id,
        'canonical_source': CANONICAL_SOURCE,
        'planned_configs': len(rows),
        'structure_pass': not structure_errors,
        'resolution_pass': not resolution_errors,
        'environment_references': environment_rows,
        'runtime_probes': runtime_probes,
        'runtime_accounting_probes': runtime_accounting_probes,
        'checkpoint_checks': checkpoint_checks,
        'synthetic_checks': synthetic,
        'errors': errors,
        'go': not errors and len(rows) == 34,
        'formal_training_started': False,
    }
    # Fingerprints are computed for all rows even when the external Stitch
    # dataset is absent, so missing data cannot silently change the design.
    fingerprints = []
    for row, config in resolved:
        payload = resolved_fingerprint_payload(
            spec=row,
            resolved_agent=config,
            dataset_root=str(dataset_root.resolve()),
            seed=0,
            source_commit=source_commit or '<manual-user-supplied>',
        )
        fingerprints.append({
            'config_id': row['config_id'],
            'fingerprint': config_fingerprint(payload),
            'semantic_condition': row['semantic_condition'],
        })
    report['fingerprints'] = fingerprints
    return report


def dry_run(study_path: Path, dataset_root: Path, run_root: Path, gpus: str, source_commit: str | None):
    study = load_study(study_path)
    rows, errors = validate_structure(study)
    environment_rows = validate_environment_references(study, dataset_root)
    missing = [row['requested_id'] for row in environment_rows if not row['valid']]
    gpu_ids = [item.strip() for item in gpus.split(',') if item.strip()]
    print(f'source_commit: {source_commit or "<manual-user-supplied>"}')
    print(f'study_id: {study.study_id}')
    print(f'planned_configs: {len(rows)}')
    print('training_protocol: train_steps=1000000 batch_size=1024 log_interval=5000 eval_interval=100000 save_interval=100000')
    print('evaluation_protocol: tasks=all episodes=20 temperature=0 gaussian=none video=false primary=last@1M secondary=normalized_eval_auc,best_success,best_step,last3_mean')
    print(f'dataset_root: {dataset_root}')
    if missing:
        print(f'dataset_preflight: NO-GO missing={",".join(missing)}')
    else:
        print('dataset_preflight: PASS')
    for index, row in enumerate(rows):
        gpu = gpu_ids[index % len(gpu_ids)] if gpu_ids else '<pending>'
        run_dir = make_run_path(run_root, study.study_id, row['config_id'], row['slug'], row['environment'], 0)
        print(
            f'[PLANNED] {row["config_id"]} {row["semantic_condition"]} '
            f'algorithm={row["algorithm"]} environment={row["environment"]} seed=0 '
            f'GPU={gpu} train_steps=1000000 eval=allx20 run_dir={run_dir}'
        )
    if errors:
        print('structure_preflight: NO-GO')
        for error in errors:
            print(f'ERROR: {error}')
    else:
        print('structure_preflight: PASS')
    print('formal_training_started: false')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--source-commit', default=os.environ.get('RLC_SOURCE_COMMIT'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-runtime-probes', type=int, default=None)
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run(args.study, args.dataset_root, args.run_root, args.gpus, args.source_commit)
    report = doctor(
        args.study,
        args.dataset_root,
        source_commit=args.source_commit,
        max_runtime_probes=args.max_runtime_probes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f'M11B doctor: {"34/34 PASS" if report["go"] else "NO-GO"}')
    return 0 if report['go'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
