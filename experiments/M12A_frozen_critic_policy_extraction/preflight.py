"""Declarative M12A structure and dependency preflight.

This module never starts training.  It validates the small Study design and
reports whether Stage 2's source Runs are available for a user-requested run
root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from impls.experiment import (
    load_study,
    make_run_path,
    prepare_run_design,
    validate_source_run_dependency,
)
from impls.main import _make_config, _parse_args


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = Path(__file__).resolve().parent / 'study.yaml'
EXPECTED_CONFIGS = ('M12A-C001', 'M12A-C002', 'M12A-C003')
PRIMARY_ENVIRONMENT = 'antmaze-large-navigate-v0'


def validate_design(study_path=STUDY_PATH):
    study = load_study(study_path)
    errors = []
    if tuple(study.data.get('seeds', ())) != (0, 1, 2):
        errors.append(f'seeds must be [0, 1, 2], got {study.data.get("seeds")!r}')
    if tuple(study.data.get('environments', ())) != (PRIMARY_ENVIRONMENT,):
        errors.append('M12A-Core must contain only antmaze-large-navigate-v0')
    configs = []
    for config_id in EXPECTED_CONFIGS:
        try:
            _, configuration = prepare_run_design(study_path, config_id)
            config = _make_config(_parse_args(['--agent', 'crl']), configuration=configuration)
            configs.append((configuration, config))
        except Exception as error:
            errors.append(f'{config_id}: {type(error).__name__}: {error}')
    if len(configs) != 3:
        errors.append(f'expected 3 configurations, found {len(configs)}')
        return study, configs, errors
    c001, c002, c003 = configs
    if c001[0].data.get('stage') != 'critic_pretrain':
        errors.append('C001 must be critic_pretrain')
    if c002[0].data.get('stage') != 'policy_extraction' or c003[0].data.get('stage') != 'policy_extraction':
        errors.append('C002/C003 must be policy_extraction')
    for configuration, config in configs:
        if config['agent_name'] != 'crl' or config['actor_loss'] != 'ddpgbc':
            errors.append(f'{configuration.config_id}: CRL DDPG+BC identity mismatch')
        if tuple(config['actor_hidden_dims']) != (512, 512, 512):
            errors.append(f'{configuration.config_id}: actor hidden dims mismatch')
        if tuple(config['value_hidden_dims']) != (512, 512, 512):
            errors.append(f'{configuration.config_id}: value hidden dims mismatch')
        for slot_name in ('critic_state', 'critic_goal'):
            slot = config['compute'][slot_name]
            if slot.get('enabled') or slot.get('topology') != 'feedforward':
                errors.append(f'{configuration.config_id}: critic is not canonical FF')
    actor_slot = c003[1]['compute']['actor']
    kwargs = actor_slot.get('topology_kwargs', {})
    expected = {
        'enabled': True,
        'topology': 'single_state',
        'credit': 'direct',
        'iterations': 4,
        'residual': False,
        'input_injection': 'z_plus_x',
        'state_dim': 512,
        'state_init': 'normal_buffer',
        'state_init_std': 1.0,
        'update_depth': 2,
        'layer_norm': False,
        'update_activate_final': True,
    }
    observed = {
        'enabled': bool(actor_slot.get('enabled')),
        'topology': actor_slot.get('topology'),
        'credit': actor_slot.get('credit'),
        **kwargs,
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            errors.append(f'C003 actor {key}: expected {value!r}, got {observed.get(key)!r}')
    if Path(ROOT / 'impls/experiment/m12a.py').exists():
        errors.append('forbidden impls/experiment/m12a.py exists')
    return study, configs, errors


def print_dry_run(stage, run_root, study_path=STUDY_PATH):
    study, configs, errors = validate_design(study_path)
    print(f'study_id: {study.study_id}')
    print(f'stage: {stage}')
    print(f'seeds: {study.data.get("seeds")}')
    print(f'run_root: {Path(run_root).resolve()}')
    if errors:
        for error in errors:
            print(f'ERROR: {error}')
        print('preflight: NO-GO')
        return 1
    if stage == 1:
        selected = [configs[0][0]]
    elif stage == 2:
        selected = [configs[1][0], configs[2][0]]
        dependency_errors = []
        for configuration in selected:
            for seed in study.data['seeds']:
                for name in configuration.data.get('dependencies', {}):
                    try:
                        validate_source_run_dependency(
                            study,
                            configuration,
                            name,
                            seed=seed,
                            run_root=run_root,
                            resolved_agent=dict(configs[1 if configuration.config_id.endswith('002') else 2][1]),
                        )
                    except Exception as error:
                        dependency_errors.append(
                            f'{configuration.config_id} seed={seed}: {error}'
                        )
        if dependency_errors:
            print('dependency_preflight: NO-GO')
            for error in dependency_errors:
                print(f'ERROR: {error}')
            return 1
    else:
        raise ValueError(f'Unsupported stage: {stage}')
    jobs = []
    for configuration in selected:
        for seed in study.data['seeds']:
            jobs.append(make_run_path(
                run_root,
                study.study_id,
                configuration.config_id,
                configuration.slug,
                PRIMARY_ENVIRONMENT,
                seed,
            ))
    print(f'planned_runs: {len(jobs)}')
    for path in jobs:
        print(f'[PLANNED] {path}')
    print('formal_training_started: false')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--study', type=Path, default=STUDY_PATH)
    args = parser.parse_args(argv)
    return print_dry_run(args.stage, args.run_root, args.study)


if __name__ == '__main__':
    raise SystemExit(main())
