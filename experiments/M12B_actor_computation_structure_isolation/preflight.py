"""M12B architecture, provenance, accounting and dry-run preflight.

This module validates design and existing artifacts only.  It never creates a
formal M12B run and never starts training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents.crl_policy_extractor import CRLPolicyExtractorAgent
from impls.computation.accounting import actor_slot_accounting
from impls.experiment import (
    load_study,
    make_run_path,
    prepare_run_design,
    validate_source_run_dependency,
)
from impls.main import _make_config, _parse_args
from impls.utils.checkpointing import (
    checkpoint_module_fingerprint,
    parameter_module_key,
    resolve_checkpoint,
    tree_fingerprint,
)
from impls.utils.datasets import Dataset, GCDataset


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = Path(__file__).resolve().parent / 'study.yaml'
DEFAULT_RUN_ROOT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
DEFAULT_DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
EXPECTED_CONFIGS = tuple(f'M12B-C{index:03d}' for index in range(1, 8))
EXPECTED_NEW_CONDITIONS = ('B002', 'B003', 'B005', 'B006', 'B007', 'B008', 'B009')
SEEDS = (0, 1, 2)
ENVIRONMENT = 'antmaze-large-navigate-v0'


def _jsonable(value):
    if hasattr(value, 'items'):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def _load_json(path):
    with Path(path).open() as file:
        return json.load(file)


def _source_study_path(study, declaration):
    path = Path(declaration['source_study_path'])
    if not path.is_absolute():
        path = study.path.parent / path
    return path


def _actor_module(agent):
    key = parameter_module_key(agent.network.params, 'actor')
    module = agent.network.params[key]
    buffers = (agent.network.model_state or {}).get('buffers', {})
    return key, module, buffers.get(key, {})


def _make_agent(config, seed=0, obs_dim=29, action_dim=8):
    observations = jnp.zeros((2, obs_dim), dtype=jnp.float32)
    actions = jnp.zeros((2, action_dim), dtype=jnp.float32)
    return CRLPolicyExtractorAgent.create(seed, observations, actions, config)


def _same_tree(first, second):
    return tree_fingerprint(first) == tree_fingerprint(second)


def _zero_tree(tree):
    return all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree_util.tree_leaves(tree))


def _anchor_record(study, anchor_name, run_root, errors):
    anchor = study.data['external_anchors'][anchor_name]
    source_study, source_configuration = prepare_run_design(
        _source_study_path(study, anchor), anchor['source_config_id']
    )
    if source_study.study_id != anchor['source_study_id']:
        errors.append(f'{anchor_name}: source study identity mismatch')
    records = []
    for seed in SEEDS:
        run_dir = make_run_path(
            run_root,
            source_study.study_id,
            source_configuration.config_id,
            source_configuration.slug,
            source_configuration.data['environment'],
            seed,
            run_attempt=int(anchor['source_run_attempt']),
        )
        metadata_path = run_dir / 'runtime_metadata.json'
        resolved_path = run_dir / 'resolved_config.json'
        if not metadata_path.is_file() or not resolved_path.is_file():
            errors.append(f'{anchor_name} seed={seed}: missing metadata at {run_dir}')
            continue
        metadata = _load_json(metadata_path)
        resolved = _load_json(resolved_path)
        expected_identity = {
            'status': 'completed',
            'study_id': source_study.study_id,
            'config_id': source_configuration.config_id,
            'environment': ENVIRONMENT,
            'seed': seed,
            'run_attempt': int(anchor['source_run_attempt']),
        }
        for key, expected in expected_identity.items():
            if metadata.get(key) != expected:
                errors.append(
                    f'{anchor_name} seed={seed}: {key}={metadata.get(key)!r}, '
                    f'expected {expected!r}'
                )
        try:
            checkpoint = resolve_checkpoint(run_dir, 'last')
        except Exception as error:
            errors.append(f'{anchor_name} seed={seed}: invalid last checkpoint: {error}')
            continue
        if checkpoint['checkpoint_step'] != 1_000_000:
            errors.append(f'{anchor_name} seed={seed}: last checkpoint is not @1M')
        protocol = metadata.get('training_protocol', {})
        for key, expected in {
            'train_steps': 1_000_000,
            'batch_size': 1024,
            'eval_interval': 100_000,
            'eval_tasks': 'all',
            'eval_episodes': 20,
            'eval_temperature': 0.0,
            'eval_gaussian': None,
            'video_episodes': 0,
            'save_best_checkpoint': True,
            'save_last_checkpoint': True,
        }.items():
            if protocol.get(key) != expected:
                errors.append(
                    f'{anchor_name} seed={seed}: protocol {key}={protocol.get(key)!r}'
                )
        agent = resolved.get('algorithm_config', {}).get('agent', {})
        if agent.get('agent_name') != 'crl' or agent.get('actor_loss') != 'ddpgbc':
            errors.append(f'{anchor_name} seed={seed}: not CRL DDPG+BC')
        actor_slot = agent.get('compute', {}).get('actor', {})
        if anchor_name == 'B001':
            if actor_slot.get('enabled') or actor_slot.get('topology') != 'feedforward':
                errors.append(f'{anchor_name} seed={seed}: not canonical FeedForward')
        else:
            if actor_slot.get('topology') != 'single_state':
                errors.append(f'{anchor_name} seed={seed}: not SingleState')
            kwargs = actor_slot.get('topology_kwargs', {})
            if kwargs.get('iterations') != 4 or kwargs.get('residual') is not False:
                errors.append(f'{anchor_name} seed={seed}: K4 actor spec mismatch')
            if actor_slot.get('parameter_sharing', 'shared') != 'shared':
                errors.append(f'{anchor_name} seed={seed}: anchor is not shared')
        dependency = metadata.get('frozen_dependencies', {}).get('frozen_critic', {})
        source_critic_study, source_critic_config = prepare_run_design(
            _source_study_path(study, study.data['fixed_design']['frozen_critic']),
            study.data['fixed_design']['frozen_critic']['source_config_id'],
        )
        critic_dir = make_run_path(
            run_root,
            source_critic_study.study_id,
            source_critic_config.config_id,
            source_critic_config.slug,
            ENVIRONMENT,
            seed,
            run_attempt=0,
        )
        critic_checkpoint = resolve_checkpoint(critic_dir, 'last')
        expected_sha = critic_checkpoint['checkpoint_sha256']
        expected_fp = checkpoint_module_fingerprint(critic_checkpoint['checkpoint_path'], 'critic')
        for key, expected in {
            'source_config_id': source_critic_config.config_id,
            'source_seed': seed,
            'source_run_attempt': 0,
            'checkpoint_role': 'last',
            'checkpoint_step': 1_000_000,
            'checkpoint_sha256': expected_sha,
            'module_fingerprint': expected_fp,
        }.items():
            if dependency.get(key) != expected:
                errors.append(f'{anchor_name} seed={seed}: frozen critic {key} mismatch')
        records.append({
            'condition_id': anchor['condition_id'],
            'structure': anchor['structure'],
            'state_init': anchor.get('state_init'),
            'parameter_sharing': anchor.get('parameter_sharing'),
            'seed': seed,
            'source_study': source_study.study_id,
            'source_config': anchor['source_config_id'],
            'source_attempt': int(anchor['source_run_attempt']),
            'source_commit': metadata.get('git_commit'),
            'source_checkpoint': checkpoint['checkpoint_path'],
            'critic_sha': expected_sha,
            'critic_fingerprint': expected_fp,
        })
    return records


def _check_initialization_invariants(study, configurations, errors):
    configs = {item.config_id: item for item in configurations}
    _, m12a_c003 = prepare_run_design(
        _source_study_path(study, study.data['external_anchors']['B004']),
        study.data['external_anchors']['B004']['source_config_id'],
    )
    pairs = (
        ('B002/B003', configs['M12B-C001'], configs['M12B-C002'], False),
        ('B004/B005', m12a_c003, configs['M12B-C003'], False),
        ('B006/B007', configs['M12B-C004'], configs['M12B-C005'], False),
        ('B002/B004 normal', configs['M12B-C001'], m12a_c003, True),
        ('B003/B005 zero', configs['M12B-C002'], configs['M12B-C003'], True),
    )
    args = _parse_args(['--agent', 'crl'])
    for label, left_configuration, right_configuration, buffers_equal in pairs:
        for seed in SEEDS:
            try:
                left = _make_agent(_make_config(args, configuration=left_configuration), seed)
                right = _make_agent(_make_config(args, configuration=right_configuration), seed)
                _, left_params, left_buffers = _actor_module(left)
                _, right_params, right_buffers = _actor_module(right)
                if not _same_tree(left_params, right_params):
                    errors.append(f'{label} seed={seed}: actor params differ')
                if buffers_equal and not _same_tree(left_buffers, right_buffers):
                    errors.append(f'{label} seed={seed}: buffers differ')
                if not buffers_equal and _same_tree(left_buffers, right_buffers):
                    errors.append(f'{label} seed={seed}: normal/zero buffers unexpectedly equal')
                if not buffers_equal and _zero_tree(right_buffers) is False and 'zero' in label:
                    errors.append(f'{label} seed={seed}: expected zero buffer')
            except Exception as error:
                errors.append(f'{label} seed={seed}: initialization check failed: {error}')


def _config_accounting(configuration, seed=0):
    args = _parse_args(['--agent', 'crl'])
    config = _make_config(args, configuration=configuration)
    agent = _make_agent(config, seed)
    _, module, buffers = _actor_module(agent)
    spec = config['compute']['actor']
    kwargs = dict(spec.get('topology_kwargs', {}))
    return actor_slot_accounting(
        module,
        buffers,
        topology=spec.get('topology'),
        iterations=int(kwargs.get('iterations', 0)),
        topology_kwargs=kwargs,
        parameter_sharing=spec.get('parameter_sharing', 'shared'),
        block=spec.get('block', 'plain'),
    )


def _check_accounting(study, configurations, errors):
    configs = {item.config_id: item for item in configurations}
    _, m12a_c002 = prepare_run_design(
        _source_study_path(study, study.data['external_anchors']['B001']),
        study.data['external_anchors']['B001']['source_config_id'],
    )
    _, m12a_c003 = prepare_run_design(
        _source_study_path(study, study.data['external_anchors']['B004']),
        study.data['external_anchors']['B004']['source_config_id'],
    )
    cases = (
        ('B001', m12a_c002), ('B002', configs['M12B-C001']),
        ('B003', configs['M12B-C002']), ('B004', m12a_c003),
        ('B005', configs['M12B-C003']), ('B006', configs['M12B-C004']),
        ('B007', configs['M12B-C005']), ('B008', configs['M12B-C006']),
        ('B009', configs['M12B-C007']),
    )
    expected = {
        'B001': (555520, 559624, 0, 3, 3, 553984, 558080),
        'B002': (555520, 559624, 512, 3, 3, 553984, 558080),
        'B003': (555520, 559624, 512, 3, 3, 553984, 558080),
        'B004': (555520, 559624, 512, 3, 9, 2126848, 2130944),
        'B005': (555520, 559624, 512, 3, 9, 2126848, 2130944),
        'B006': (2131456, 2135560, 512, 9, 9, 2126848, 2130944),
        'B007': (2131456, 2135560, 512, 9, 9, 2126848, 2130944),
        'B008': (2131456, 2135560, 0, 9, 9, 2126848, 2130944),
        'B009': (2131456, 2135560, 0, 9, 9, 2126848, 2130944),
    }
    for condition, configuration in cases:
        try:
            report = _config_accounting(configuration)
            observed = (
                report['core_trainable_params'], report['trainable_params'],
                report['buffer_elements'], report['unique_dense_layers'],
                report['executed_dense_layers'], report['actor_body_dense_macs'],
                report['full_actor_forward_dense_macs'],
            )
            if observed != expected[condition]:
                errors.append(f'{condition}: accounting {observed!r} != {expected[condition]!r}')
        except Exception as error:
            errors.append(f'{condition}: accounting failed: {error}')


def _check_paired_data_stream(configurations, errors):
    raw = {
        'observations': np.arange(100 * 5, dtype=np.float32).reshape(100, 5),
        'actions': np.arange(100 * 2, dtype=np.float32).reshape(100, 2),
        'terminals': np.array([0] * 99 + [1], dtype=np.float32),
    }
    args = _parse_args(['--agent', 'crl'])
    streams = []
    for configuration in configurations:
        config = _make_config(args, configuration=configuration)
        dataset = GCDataset(Dataset.create(**raw), config, rng=12345)
        streams.append(dataset)
    reference = None
    for stream in streams:
        fingerprints = []
        for _ in range(10):
            batch = stream.sample(4)
            fingerprints.append(tree_fingerprint(batch))
        if reference is None:
            reference = fingerprints
        elif fingerprints != reference:
            errors.append('paired dataset stream hash mismatch across configurations')


def validate_design(study_path=STUDY_PATH, run_root=DEFAULT_RUN_ROOT, dataset_root=DEFAULT_DATASET_ROOT):
    study = load_study(study_path)
    errors = []
    if tuple(study.data.get('seeds', ())) != SEEDS:
        errors.append(f'seeds must be {SEEDS!r}')
    if tuple(study.data.get('environments', ())) != (ENVIRONMENT,):
        errors.append(f'environment must be {ENVIRONMENT!r}')
    if tuple(study.data.get('new_conditions', ())) != EXPECTED_NEW_CONDITIONS:
        errors.append('new_conditions must contain exactly the seven active conditions')
    if study.data.get('new_formal_runs') != 21:
        errors.append('new_formal_runs must be exactly 21')
    forbidden = (
        ROOT / 'impls' / 'computation' / 'topologies' / 'untied_single_state.py',
        ROOT / 'impls' / 'computation' / 'topologies' / 'residual_stack.py',
        ROOT / 'impls' / 'experiment' / 'm12b.py',
    )
    for path in forbidden:
        if path.exists():
            errors.append(f'forbidden architecture file exists: {path}')
    for suffix in (f'{ENVIRONMENT}.npz', f'{ENVIRONMENT}-val.npz'):
        if not (Path(dataset_root) / suffix).is_file():
            errors.append(f'missing dataset file: {Path(dataset_root) / suffix}')

    configurations = []
    args = _parse_args(['--agent', 'crl'])
    for config_id in EXPECTED_CONFIGS:
        try:
            _, configuration = prepare_run_design(study_path, config_id)
            config = _make_config(args, configuration=configuration)
            configurations.append(configuration)
            if configuration.data.get('executable') is not True:
                errors.append(f'{config_id}: executable must be true')
            if configuration.data.get('condition_id') not in EXPECTED_NEW_CONDITIONS:
                errors.append(f'{config_id}: invalid active condition')
            if config['agent_name'] != 'crl' or config['actor_loss'] != 'ddpgbc':
                errors.append(f'{config_id}: CRL DDPG+BC mismatch')
            if tuple(config['value_hidden_dims']) != (512, 512, 512):
                errors.append(f'{config_id}: value hidden dims changed')
            topology = config['compute']['actor'].get('topology')
            block = config['compute']['actor'].get('block', 'plain')
            if topology not in ('single_state', 'feedforward'):
                errors.append(f'{config_id}: invalid topology {topology!r}')
            if block not in ('plain', 'residual'):
                errors.append(f'{config_id}: invalid block {block!r}')
        except Exception as error:
            errors.append(f'{config_id}: {type(error).__name__}: {error}')

    if len(configurations) == len(EXPECTED_CONFIGS):
        _check_initialization_invariants(study, configurations, errors)
        _check_accounting(study, configurations, errors)
        _check_paired_data_stream(configurations, errors)
        for configuration in configurations:
            for seed in SEEDS:
                try:
                    validate_source_run_dependency(
                        study, configuration, 'frozen_critic',
                        seed=seed, run_root=run_root,
                        resolved_agent=_make_config(args, configuration=configuration),
                    )
                except Exception as error:
                    errors.append(f'{configuration.config_id} seed={seed}: dependency: {error}')

    anchor_records = []
    for anchor_name in ('B001', 'B004'):
        try:
            anchor_records.extend(_anchor_record(study, anchor_name, run_root, errors))
        except Exception as error:
            errors.append(f'{anchor_name}: anchor validation failed: {error}')

    for configuration in configurations:
        for seed in SEEDS:
            path = make_run_path(
                run_root, study.study_id, configuration.config_id,
                configuration.slug, ENVIRONMENT, seed, run_attempt=0,
            )
            if path.exists():
                errors.append(f'new M12B formal path already exists: {path}')
    return {
        'study': study,
        'configurations': configurations,
        'anchor_records': anchor_records,
        'errors': errors,
    }


def print_dry_run(result, run_root):
    errors = result['errors']
    if errors:
        print(f'study_id: {result["study"].study_id}')
        for error in errors:
            print(f'ERROR: {error}')
        print('preflight: NO-GO')
        return 2
    print(f'study_id: {result["study"].study_id}')
    print(f'run_root: {run_root}')
    print(f'planned_runs: {len(result["configurations"]) * len(SEEDS)}')
    for configuration in result['configurations']:
        for seed in SEEDS:
            path = make_run_path(
                run_root, result['study'].study_id, configuration.config_id,
                configuration.slug, ENVIRONMENT, seed, run_attempt=0,
            )
            print(f'[PLANNED] {path}')
    print(f'external_anchor_runs: {len(result["anchor_records"])}')
    print(f'new_formal_runs: {len(result["configurations"]) * len(SEEDS)}')
    print('formal_training_started: false')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY_PATH)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args(argv)
    result = validate_design(args.study, args.run_root, args.dataset_root)
    raise SystemExit(print_dry_run(result, args.run_root))


if __name__ == '__main__':
    main()
