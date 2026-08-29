"""Non-executing preflight for the M16C Puzzle-4x4 S002 alpha sweep.

The doctor validates the four new alpha configurations, their reusable M16A/
M16B S002 anchors, dataset availability, network construction, and canonical
output paths. It never launches training, performs Git operations, or mutates
run artifacts.
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

from impls.agents import agents
from impls.computation.accounting import gciql_architecture_accounting
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args


ENVIRONMENT = 'puzzle-4x4-play-v0'
NUM_BUTTONS = 16
EXPECTED_ALPHAS = (0.1, 0.2, 0.5, 0.7)
EXPECTED_ANCHORS = {
    'alpha_0p3': {
        'alpha': 0.3,
        'study_id': 'M16A',
        'config_id': 'M16A-4x4-S002',
    },
    'alpha_1p0': {
        'alpha': 1.0,
        'study_id': 'M16B',
        'config_id': 'M16B-4x4-S002',
    },
}
FIXED_AGENT_FIELDS = (
    'lr', 'batch_size', 'actor_hidden_dims', 'value_hidden_dims',
    'layer_norm', 'discount', 'tau', 'expectile', 'actor_loss',
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


def _expected_dataset_paths(dataset_root):
    return [
        Path(dataset_root) / f'{ENVIRONMENT}.npz',
        Path(dataset_root) / f'{ENVIRONMENT}-val.npz',
    ]


def _as_float(value, label):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be numeric, got {value!r}') from error


def _same_float(left, right):
    return abs(float(left) - float(right)) <= 1e-12


def _read_anchor(name, spec):
    run_dir = Path(spec.get('source_run', ''))
    if not run_dir.is_dir():
        raise ValueError(f'Anchor {name}: source_run is unavailable: {run_dir}')
    metadata_path = run_dir / 'runtime_metadata.json'
    resolved_path = run_dir / 'resolved_config.json'
    eval_path = run_dir / 'eval.csv'
    if not metadata_path.is_file() or not resolved_path.is_file() or not eval_path.is_file():
        raise ValueError(f'Anchor {name}: missing metadata, resolved config, or eval.csv')
    metadata = json.loads(metadata_path.read_text())
    resolved = json.loads(resolved_path.read_text())
    expected = EXPECTED_ANCHORS[name]
    if metadata.get('status') != 'completed':
        raise ValueError(f'Anchor {name}: status is {metadata.get("status")!r}, expected completed')
    if metadata.get('study_id') != expected['study_id']:
        raise ValueError(f'Anchor {name}: study_id mismatch')
    if metadata.get('config_id') != expected['config_id']:
        raise ValueError(f'Anchor {name}: config_id mismatch')
    if metadata.get('environment') != ENVIRONMENT or int(metadata.get('seed', -1)) != 0:
        raise ValueError(f'Anchor {name}: environment/seed mismatch')
    agent = resolved.get('algorithm_config', {}).get('agent', {})
    if not _same_float(agent.get('alpha'), expected['alpha']):
        raise ValueError(f'Anchor {name}: resolved alpha mismatch')
    with eval_path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    steps = [int(row['step']) for row in rows]
    expected_steps = list(range(100_000, 1_000_001, 100_000))
    if steps != expected_steps:
        raise ValueError(f'Anchor {name}: incomplete evaluation steps {steps!r}')
    return {
        'name': name,
        'alpha': expected['alpha'],
        'run_dir': str(run_dir),
        'git_commit': metadata.get('git_commit'),
        'resolved_config_fingerprint': metadata.get('resolved_config_fingerprint'),
    }


def _validate_structure(configuration):
    data = configuration.data
    if data.get('environment') != ENVIRONMENT:
        raise ValueError(f'{configuration.config_id}: environment mismatch')
    if data.get('condition_id') != 'S002':
        raise ValueError(f'{configuration.config_id}: condition_id must be S002')
    if data.get('algorithm') != 'gciql' or data.get('placement') != 'actor+value+critic':
        raise ValueError(f'{configuration.config_id}: algorithm/placement mismatch')
    overrides = data.get('agent_overrides', {})
    if overrides.get('actor_loss') != 'ddpgbc':
        raise ValueError(f'{configuration.config_id}: actor_loss must be ddpgbc')
    compute = overrides.get('compute', {})
    for slot_name in ('actor', 'value', 'critic'):
        slot = compute.get(slot_name, {})
        if not slot.get('enabled'):
            raise ValueError(f'{configuration.config_id}: {slot_name} must be enabled')
        if (
            slot.get('primitive') != 'mlp'
            or slot.get('topology') != 'feedforward'
            or slot.get('block') != 'mlp_mixer'
            or slot.get('credit') != 'direct'
            or slot.get('structure') != 'puzzle_tokens'
        ):
            raise ValueError(f'{configuration.config_id}: {slot_name} computation mismatch')
        kwargs = slot.get('structure_kwargs', {})
        expected_kwargs = {
            'num_buttons': NUM_BUTTONS,
            'token_dim': 128,
            'robot_hidden_dim': 128,
            'token_mlp_hidden_dim': 64,
            'channel_mlp_hidden_dim': 256,
            'num_mixer_blocks': 2,
            'index_embedding': True,
            'readout': 'mean',
            'tm_mode': 'none',
        }
        for field, expected in expected_kwargs.items():
            if kwargs.get(field) != expected:
                raise ValueError(
                    f'{configuration.config_id}: {slot_name}.{field} '
                    f'expected {expected!r}, got {kwargs.get(field)!r}'
                )


def validate(study_path, dataset_root, run_root, gpus):
    study = load_study(study_path)
    if study.study_id != 'M16C':
        raise ValueError(f'Expected M16C study, got {study.study_id!r}')
    if study.data['environments'] != [ENVIRONMENT]:
        raise ValueError(f'M16C environments mismatch: {study.data["environments"]!r}')
    if study.data.get('conditions') != ['S002']:
        raise ValueError(f'M16C conditions mismatch: {study.data.get("conditions")!r}')
    if study.data['algorithms'] != ['gciql'] or study.data['seeds'] != [0]:
        raise ValueError('M16C must contain exactly gciql and seed [0]')
    if study.data.get('protocol', {}).get('formal_training_started') is not False:
        raise ValueError('formal_training_started must remain false before manual launch')

    policy = study.data.get('alpha_policy', {})
    observed_new_values = tuple(_as_float(value, 'alpha_policy.new_scanned_values') for value in policy.get('new_scanned_values', []))
    if observed_new_values != EXPECTED_ALPHAS:
        raise ValueError(
            f'alpha_policy.new_scanned_values mismatch: {observed_new_values!r}'
        )
    if policy.get('mode') != 'explicit_per_configuration_sweep':
        raise ValueError('M16C alpha policy must be explicit_per_configuration_sweep')
    if policy.get('runtime_authority') != 'configuration.agent_overrides.alpha':
        raise ValueError('M16C alpha runtime authority is not explicit config')

    dataset_root = Path(dataset_root)
    missing = [path for path in _expected_dataset_paths(dataset_root) if not path.is_file()]
    if missing:
        raise ValueError('Missing dataset files:\n' + '\n'.join(f'  {path}' for path in missing))

    anchor_specs = study.data.get('reusable_anchor_runs', {})
    if set(anchor_specs) != set(EXPECTED_ANCHORS):
        raise ValueError('M16C anchor declaration mismatch')
    anchors = [_read_anchor(name, anchor_specs[name]) for name in EXPECTED_ANCHORS]

    config_paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    if len(config_paths) != len(EXPECTED_ALPHAS):
        raise ValueError(f'M16C requires {len(EXPECTED_ALPHAS)} configs, found {len(config_paths)}')
    configurations = [prepare_run_design(study.path, path)[1] for path in config_paths]
    seen_alphas = set()
    rows = []
    fixed_reference = None
    args = _agent_args()

    for configuration in configurations:
        _validate_structure(configuration)
        factors = configuration.data.get('factors', {})
        overrides = configuration.data.get('agent_overrides', {})
        factor_alpha = _as_float(factors.get('alpha'), f'{configuration.config_id}.factors.alpha')
        override_alpha = _as_float(overrides.get('alpha'), f'{configuration.config_id}.agent_overrides.alpha')
        if not _same_float(factor_alpha, override_alpha):
            raise ValueError(f'{configuration.config_id}: factor/override alpha mismatch')
        if factor_alpha not in EXPECTED_ALPHAS:
            raise ValueError(f'{configuration.config_id}: unexpected alpha {factor_alpha}')
        if factor_alpha in seen_alphas:
            raise ValueError(f'{configuration.config_id}: duplicate alpha {factor_alpha}')
        seen_alphas.add(factor_alpha)

        config = _make_config(args, configuration=configuration)
        if not _same_float(config['alpha'], factor_alpha):
            raise ValueError(
                f'{configuration.config_id}: resolved alpha {config["alpha"]!r} '
                f'does not match {factor_alpha!r}'
            )
        fixed_fields = {field: config[field] for field in FIXED_AGENT_FIELDS}
        if fixed_reference is None:
            fixed_reference = fixed_fields
        elif fixed_fields != fixed_reference:
            raise ValueError(
                f'{configuration.config_id}: non-alpha agent settings differ from reference'
            )

        observations = np.zeros((2, 19 + NUM_BUTTONS * 4), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        agent = agents['gciql'].create(0, observations, actions, config)
        slot_accounting = _computation_slot_accounting(agent, config)
        if set(slot_accounting) != {'actor', 'value', 'critic'}:
            raise ValueError(f'{configuration.config_id}: structured slots are incomplete')
        architecture = gciql_architecture_accounting(
            agent.network.params, config, slot_accounting,
        )
        run_dir = make_run_path(
            run_root, study.study_id, configuration.config_id, configuration.slug,
            ENVIRONMENT, 0, run_attempt=0,
        )
        if run_dir.exists():
            raise ValueError(f'Output path already exists; refusing overwrite: {run_dir}')
        rows.append({
            'config_id': configuration.config_id,
            'alpha': factor_alpha,
            'gpu': gpus[len(rows) % len(gpus)],
            'run_dir': str(run_dir),
            'total_trainable_params': architecture['total_trainable_params'],
            'total_dense_macs': architecture['total_dense_macs'],
            'depth_actor_value_critic': [
                architecture['slots'][slot]['sequential_depth']
                for slot in ('actor', 'value', 'critic')
            ],
        })
        del agent, config, slot_accounting, architecture
        gc.collect()

    if tuple(sorted(seen_alphas)) != EXPECTED_ALPHAS:
        raise ValueError(f'M16C alpha matrix incomplete: {sorted(seen_alphas)!r}')
    return {
        'status': 'PASS',
        'study_id': study.study_id,
        'expected_runs': len(EXPECTED_ALPHAS),
        'environment': ENVIRONMENT,
        'condition': 'S002',
        'seeds': [0],
        'new_alphas': list(EXPECTED_ALPHAS),
        'anchors': anchors,
        'gpus': gpus,
        'dataset_root': str(dataset_root.resolve()),
        'run_root': str(Path(run_root).resolve()),
        'fixed_agent_fields_excluding_alpha': fixed_reference,
        'jobs': rows,
        'formal_training_started': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml')
    parser.add_argument('--dataset-root', default='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--json-output', default=None)
    args = parser.parse_args(argv)
    try:
        report = validate(args.study, args.dataset_root, args.run_root, _parse_gpus(args.gpus))
    except Exception as error:
        print(f'M16C PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2
    print('M16C PREFLIGHT: PASS')
    print(
        f'{report["expected_runs"]} jobs; environment={report["environment"]}; '
        f'condition=S002; seeds=[0]; new_alphas={report["new_alphas"]}'
    )
    print('config_id\talpha\tGPU\tparams\tMACs\tdepth A/V/C\trun_dir')
    for job in report['jobs']:
        print(
            f'{job["config_id"]}\t{job["alpha"]}\t{job["gpu"]}\t'
            f'{job["total_trainable_params"]}\t{job["total_dense_macs"]}\t'
            f'{"/".join(map(str, job["depth_actor_value_critic"]))}\t{job["run_dir"]}'
        )
    print('Validated reusable anchors: ' + ', '.join(
        f'{item["name"]}=alpha{item["alpha"]}' for item in report['anchors']
    ))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'JSON report: {output}')
    print('Formal training was not started. Manual launch remains required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
