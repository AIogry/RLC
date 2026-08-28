"""Non-executing preflight for the M16B Puzzle alpha=1.0 Study.

The doctor checks the complete 8-cell B000/S002 matrix, verifies that alpha is
explicitly present in every executable configuration, instantiates the
canonical GCIQL networks, and refuses existing output paths. It never runs
training, performs Git operations, or mutates run artifacts.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from impls.agents import agents
from impls.computation.accounting import gciql_architecture_accounting
from impls.experiment import make_run_path, load_study, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args


ENV_SPECS = {
    'puzzle-3x3-play-v0': {'num_buttons': 9},
    'puzzle-4x4-play-v0': {'num_buttons': 16},
    'puzzle-4x5-play-v0': {'num_buttons': 20},
    'puzzle-4x6-play-v0': {'num_buttons': 24},
}
CONDITIONS = {
    'B000': {'structure': 'vector', 'blocks': 0},
    'S002': {'structure': 'puzzle_tokens', 'blocks': 2},
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
    return _parse_args(['--agent', 'gciql'])


def _expected_dataset_paths(dataset_root, environment):
    return [
        Path(dataset_root) / f'{environment}.npz',
        Path(dataset_root) / f'{environment}-val.npz',
    ]


def _structure_summary(architecture):
    slots = architecture['slots']
    return {
        'total_trainable_params': architecture['total_trainable_params'],
        'total_dense_macs': architecture['total_dense_macs'],
        'depth_actor_value_critic': [
            slots[name].get('sequential_depth')
            for name in ('actor', 'value', 'critic')
        ],
    }


def validate(study_path, dataset_root, run_root, gpus):
    study = load_study(study_path)
    if study.study_id != 'M16B':
        raise ValueError(f'Expected M16B study, got {study.study_id!r}')
    if study.data['environments'] != list(ENV_SPECS):
        raise ValueError(
            f'Environment order/matrix mismatch: {study.data["environments"]!r}'
        )
    if study.data.get('conditions') != list(CONDITIONS):
        raise ValueError(
            f'Condition matrix mismatch: {study.data.get("conditions")!r}'
        )
    if study.data['algorithms'] != ['gciql'] or study.data['seeds'] != [0]:
        raise ValueError('M16B must contain exactly gciql and seed [0]')
    if study.data.get('protocol', {}).get('formal_training_started') is not False:
        raise ValueError('formal_training_started must remain false before manual launch')

    fixed_design = study.data.get('fixed_design', {})
    policy = study.data.get('alpha_policy', {})
    if fixed_design.get('alpha') != 1.0:
        raise ValueError('M16B fixed_design.alpha must be exactly 1.0')
    if policy.get('mode') != 'explicit_per_study' or policy.get('value') != 1.0:
        raise ValueError('M16B alpha_policy must explicitly declare value=1.0')
    if policy.get('runtime_authority') != 'configuration.agent_overrides.alpha':
        raise ValueError('M16B alpha_policy runtime authority is not explicit config')

    config_paths = sorted((Path(study_path).parent / 'configs').glob('*.yaml'))
    expected_count = len(ENV_SPECS) * len(CONDITIONS)
    if len(config_paths) != expected_count:
        raise ValueError(
            f'M16B requires exactly {expected_count} configs, found {len(config_paths)}'
        )

    dataset_root = Path(dataset_root)
    missing = [
        path
        for environment in ENV_SPECS
        for path in _expected_dataset_paths(dataset_root, environment)
        if not path.is_file()
    ]
    if missing:
        raise ValueError('Missing dataset files:\n' + '\n'.join(f'  {path}' for path in missing))

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

        factors = data.get('factors', {})
        if factors.get('alpha') != 1.0:
            raise ValueError(f'{configuration.config_id}: factors.alpha must be 1.0')
        overrides = data.get('agent_overrides', {})
        if overrides.get('alpha') != 1.0:
            raise ValueError(
                f'{configuration.config_id}: agent_overrides.alpha must be explicit 1.0'
            )
        config = _make_config(args, configuration=configuration)
        if config['alpha'] != 1.0:
            raise ValueError(
                f'{configuration.config_id}: resolved runtime alpha is {config["alpha"]!r}'
            )
        observed = {field: config[field] for field in FROZEN_AGENT_FIELDS}
        if resolved_reference is None:
            resolved_reference = observed
        elif observed != resolved_reference:
            raise ValueError(
                f'{configuration.config_id}: resolved algorithm hyperparameters differ '
                f'from reference: {observed!r} != {resolved_reference!r}'
            )

        expected = CONDITIONS[condition]
        expected_buttons = ENV_SPECS[environment]['num_buttons']
        compute = overrides.get('compute', {})
        for slot_name in ('actor', 'value', 'critic'):
            slot = compute.get(slot_name, {})
            if bool(slot.get('enabled')) != (condition != 'B000'):
                raise ValueError(f'{configuration.config_id}: {slot_name} enabled mismatch')
            if slot.get('structure', 'vector') != expected['structure']:
                raise ValueError(f'{configuration.config_id}: {slot_name} structure mismatch')
            if condition == 'S002':
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
                if slot.get('credit') != 'direct' or kwargs.get('tm_mode') != 'none':
                    raise ValueError(f'{configuration.config_id}: structured credit/tm mismatch')

        observations = np.zeros((2, 19 + expected_buttons * 4), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        agent = agents['gciql'].create(0, observations, actions, config)
        slot_accounting = _computation_slot_accounting(agent, config)
        if condition == 'B000' and slot_accounting:
            raise ValueError(f'{configuration.config_id}: Flat baseline has enabled slots')
        if condition == 'S002' and set(slot_accounting) != {'actor', 'value', 'critic'}:
            raise ValueError(f'{configuration.config_id}: structured slots are incomplete')
        architecture = gciql_architecture_accounting(
            agent.network.params, config, slot_accounting,
        )
        run_dir = make_run_path(
            run_root, study.study_id, configuration.config_id, configuration.slug,
            environment, 0, run_attempt=0,
        )
        if run_dir.exists():
            raise ValueError(f'Output path already exists; refusing overwrite: {run_dir}')
        rows.append({
            'config_id': configuration.config_id,
            'environment': environment,
            'condition': condition,
            'alpha': config['alpha'],
            'gpu': gpus[len(rows) % len(gpus)],
            'run_dir': str(run_dir),
            'architecture': _structure_summary(architecture),
        })
        del agent, config, slot_accounting, architecture
        gc.collect()

    expected_cells = {
        (environment, condition)
        for environment in ENV_SPECS
        for condition in CONDITIONS
    }
    if seen != expected_cells:
        raise ValueError(f'M16B matrix is incomplete: missing={expected_cells - seen}')
    return {
        'status': 'PASS',
        'study_id': study.study_id,
        'expected_runs': expected_count,
        'environments': list(ENV_SPECS),
        'conditions': list(CONDITIONS),
        'seeds': [0],
        'alpha': 1.0,
        'gpus': gpus,
        'dataset_root': str(dataset_root.resolve()),
        'run_root': str(Path(run_root).resolve()),
        'frozen_agent_fields': resolved_reference,
        'jobs': rows,
        'formal_training_started': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M16B_puzzle_alpha_correction/study.yaml')
    parser.add_argument('--dataset-root', default='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--json-output', default=None)
    args = parser.parse_args(argv)
    try:
        report = validate(args.study, args.dataset_root, args.run_root, _parse_gpus(args.gpus))
    except Exception as error:
        print(f'M16B PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2
    print('M16B PREFLIGHT: PASS')
    print(
        f'{report["expected_runs"]} jobs; environments={len(report["environments"])}; '
        f'conditions={report["conditions"]}; seeds=[0]; alpha=1.0'
    )
    print('config_id\tenvironment\tcondition\talpha\tGPU\tparams\tMACs\tdepth A/V/C\trun_dir')
    for job in report['jobs']:
        architecture = job['architecture']
        print(
            f'{job["config_id"]}\t{job["environment"]}\t{job["condition"]}\t'
            f'{job["alpha"]}\t{job["gpu"]}\t{architecture["total_trainable_params"]}\t'
            f'{architecture["total_dense_macs"]}\t'
            f'{"/".join(map(str, architecture["depth_actor_value_critic"]))}\t'
            f'{job["run_dir"]}'
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
