"""Non-executing preflight and architecture audit for the M16A Study.

This tool intentionally does not perform Git checks, create run directories,
or launch training.  It validates the complete 16-cell design and prints the
GPU assignment that the generic sweep would use after manual approval.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from impls.computation.accounting import gciql_architecture_accounting
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.agents import agents


ENV_SPECS = {
    'puzzle-3x3-play-v0': {'num_buttons': 9},
    'puzzle-4x4-play-v0': {'num_buttons': 16},
    'puzzle-4x5-play-v0': {'num_buttons': 20},
    'puzzle-4x6-play-v0': {'num_buttons': 24},
}
CONDITIONS = {
    'B000': {'structure': 'vector', 'blocks': 0},
    'S001': {'structure': 'puzzle_tokens', 'blocks': 1},
    'S002': {'structure': 'puzzle_tokens', 'blocks': 2},
    'S004': {'structure': 'puzzle_tokens', 'blocks': 4},
}
FROZEN_AGENT_FIELDS = (
    'lr', 'batch_size', 'actor_hidden_dims', 'value_hidden_dims',
    'layer_norm', 'discount', 'tau', 'expectile', 'actor_loss', 'alpha',
    'const_std', 'discrete', 'encoder', 'dataset_class',
    'value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal',
    'value_geom_sample', 'actor_p_curgoal', 'actor_p_trajgoal',
    'actor_p_randomgoal', 'actor_geom_sample', 'gc_negative', 'p_aug',
    'frame_stack',
)


def _parse_gpus(value):
    gpus = [item.strip() for item in str(value).split(',') if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('--gpus must contain one or more unique device IDs')
    return gpus


def _agent_args():
    # Reuse the canonical parser defaults without invoking the training entry
    # point.  The Study configuration supplies all M16A values explicitly.
    return _parse_args(['--agent', 'gciql'])


def _expected_dataset_paths(dataset_root, environment):
    return [
        Path(dataset_root) / f'{environment}.npz',
        Path(dataset_root) / f'{environment}-val.npz',
    ]


def _structure_summary(config, architecture):
    slots = architecture['slots']
    return {
        'slots': slots,
        'actor': {
            key: slots['actor'].get(key)
            for key in ('structure', 'num_mixer_blocks', 'token_dim',
                        'token_hidden_dim', 'channel_hidden_dim',
                        'trainable_params', 'total_dense_macs',
                        'sequential_depth')
        },
        'value': {
            key: slots['value'].get(key)
            for key in ('structure', 'num_mixer_blocks', 'token_dim',
                        'token_hidden_dim', 'channel_hidden_dim',
                        'trainable_params', 'total_dense_macs',
                        'sequential_depth')
        },
        'critic': {
            key: slots['critic'].get(key)
            for key in ('structure', 'num_mixer_blocks', 'token_dim',
                        'token_hidden_dim', 'channel_hidden_dim',
                        'trainable_params', 'total_dense_macs',
                        'sequential_depth')
        },
        'total_trainable_params': architecture['total_trainable_params'],
        'total_dense_macs': architecture['total_dense_macs'],
        'agent_name': config['agent_name'],
    }


def validate(study_path, dataset_root, run_root, gpus):
    study, _ = prepare_run_design(study_path, next(iter(sorted(
        (Path(study_path).parent / 'configs').glob('*.yaml')
    ))))
    if study.study_id != 'M16A':
        raise ValueError(f'Expected M16A study, got {study.study_id!r}')
    expected_envs = list(ENV_SPECS)
    if study.data['environments'] != expected_envs:
        raise ValueError(
            f'Environment order/matrix mismatch: {study.data["environments"]!r}'
        )
    if study.data['algorithms'] != ['gciql'] or study.data['seeds'] != [0]:
        raise ValueError('M16A must contain exactly gciql and seed [0]')
    if study.data.get('protocol', {}).get('formal_training_started') is not False:
        raise ValueError('formal_training_started must remain false before manual launch')

    config_paths = sorted((Path(study_path).parent / 'configs').glob('*.yaml'))
    if len(config_paths) != 16:
        raise ValueError(f'M16A requires exactly 16 configs, found {len(config_paths)}')
    configurations = [prepare_run_design(study_path, path)[1] for path in config_paths]
    seen = set()
    rows = []
    resolved_reference = None
    args = _agent_args()

    for configuration in configurations:
        data = configuration.data
        environment = data.get('environment')
        condition = data.get('condition_id')
        if environment not in ENV_SPECS:
            raise ValueError(f'{configuration.config_id}: unknown environment {environment!r}')
        if condition not in CONDITIONS:
            raise ValueError(f'{configuration.config_id}: unknown condition {condition!r}')
        key = (environment, condition)
        if key in seen:
            raise ValueError(f'Duplicate environment/condition cell: {key!r}')
        seen.add(key)
        expected = CONDITIONS[condition]
        expected_buttons = ENV_SPECS[environment]['num_buttons']
        overrides = data.get('agent_overrides', {})
        compute = overrides.get('compute', {})
        for slot_name in ('actor', 'value', 'critic'):
            slot = compute.get(slot_name, {})
            if bool(slot.get('enabled')) != (condition != 'B000'):
                raise ValueError(f'{configuration.config_id}: {slot_name} enabled mismatch')
            if slot.get('structure', 'vector') != expected['structure']:
                raise ValueError(f'{configuration.config_id}: {slot_name} structure mismatch')
            if condition != 'B000':
                kwargs = slot.get('structure_kwargs', {})
                for field, required in (
                    ('num_buttons', expected_buttons), ('token_dim', 128),
                    ('robot_hidden_dim', 128), ('token_mlp_hidden_dim', 64),
                    ('channel_mlp_hidden_dim', 256),
                    ('num_mixer_blocks', expected['blocks']),
                ):
                    if kwargs.get(field) != required:
                        raise ValueError(
                            f'{configuration.config_id}: {slot_name}.{field} '
                            f'expected {required!r}, got {kwargs.get(field)!r}'
                        )
                if slot.get('topology') != 'feedforward' or slot.get('block') != 'mlp_mixer':
                    raise ValueError(f'{configuration.config_id}: structured topology/block mismatch')
                if slot.get('credit') != 'direct':
                    raise ValueError(f'{configuration.config_id}: structured credit must be direct')
                if kwargs.get('tm_mode') != 'none':
                    raise ValueError(f'{configuration.config_id}: recurrence-like tm_mode is not allowed')
        if any(
            marker in json.dumps(data, sort_keys=True).lower()
            for marker in ('hrm', 'recurrent', 'two_state', 'single_state')
        ):
            raise ValueError(f'{configuration.config_id}: forbidden recurrence/HRM marker')

        config = _make_config(args, configuration=configuration)
        observed = {field: config[field] for field in FROZEN_AGENT_FIELDS}
        if resolved_reference is None:
            resolved_reference = observed
        elif observed != resolved_reference:
            raise ValueError(
                f'{configuration.config_id}: resolved algorithm hyperparameters differ '
                f'from reference: {observed!r} != {resolved_reference!r}'
            )

        obs_dim = 19 + expected_buttons * 4
        observations = np.zeros((2, obs_dim), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        agent = agents['gciql'].create(0, observations, actions, config)
        slot_accounting = _computation_slot_accounting(agent, config)
        architecture = gciql_architecture_accounting(
            agent.network.params, config, slot_accounting,
        )
        if condition == 'B000':
            if slot_accounting:
                raise ValueError(f'{configuration.config_id}: Flat baseline has enabled slots')
        elif set(slot_accounting) != {'actor', 'value', 'critic'}:
            raise ValueError(f'{configuration.config_id}: structured GCIQL must enable all slots')
        architecture_summary = _structure_summary(config, architecture)
        run_dir = make_run_path(
            run_root, study.study_id, configuration.config_id, configuration.slug,
            environment, 0, run_attempt=0,
        )
        rows.append({
            'config_id': configuration.config_id,
            'environment': environment,
            'condition_id': condition,
            'seed': 0,
            'gpu': gpus[(len(rows)) % len(gpus)],
            'num_buttons': expected_buttons,
            'run_dir': str(run_dir),
            'run_dir_exists': run_dir.exists(),
            'architecture': architecture_summary,
        })
        if run_dir.exists():
            raise ValueError(f'Output path already exists; refusing overwrite: {run_dir}')
        del agent, config, slot_accounting, architecture
        gc.collect()

    if seen != {(environment, condition) for environment in expected_envs for condition in CONDITIONS}:
        raise ValueError('M16A does not cover every environment/condition cell')
    return {
        'status': 'PASS',
        'study_id': study.study_id,
        'expected_runs': 16,
        'observed_configs': len(configurations),
        'environments': expected_envs,
        'conditions': list(CONDITIONS),
        'seeds': [0],
        'gpus': gpus,
        'dataset_root': str(Path(dataset_root).resolve()),
        'run_root': str(Path(run_root).resolve()),
        'dataset_files': {
            environment: [str(path) for path in _expected_dataset_paths(dataset_root, environment)]
            for environment in expected_envs
        },
        'frozen_agent_fields': resolved_reference,
        'jobs': rows,
        'formal_training_started': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M16A_puzzle_mixer_depth_scaling/study.yaml')
    parser.add_argument('--dataset-root', default='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--json-output', default=None)
    args = parser.parse_args(argv)
    try:
        dataset_root = Path(args.dataset_root)
        missing = [
            path for environment in ENV_SPECS
            for path in _expected_dataset_paths(dataset_root, environment)
            if not path.is_file()
        ]
        if missing:
            raise ValueError('Missing dataset files:\n' + '\n'.join(f'  {path}' for path in missing))
        report = validate(args.study, dataset_root, args.run_root, _parse_gpus(args.gpus))
    except Exception as error:
        print(f'M16A PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2
    print('M16A PREFLIGHT: PASS')
    print(f'16 jobs; environments={report["environments"]}; conditions={report["conditions"]}; seeds=[0]')
    print('config_id\tenvironment\tcondition\tbuttons\tGPU\tparams\tMACs\tdepth(actor/value/critic)\trun_dir')
    for job in report['jobs']:
        slots = job['architecture']['slots']
        depth = '/'.join(str(slots[name]['sequential_depth']) for name in ('actor', 'value', 'critic'))
        print(
            f'{job["config_id"]}\t{job["environment"]}\t{job["condition_id"]}\t'
            f'{job["num_buttons"]}\t{job["gpu"]}\t{job["architecture"]["total_trainable_params"]}\t'
            f'{job["architecture"]["total_dense_macs"]}\t{depth}\t{job["run_dir"]}'
        )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'JSON report: {output}')
    print('Formal training was not started. Manual launch remains required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
