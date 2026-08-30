"""Non-executing launch gate for M18 fixed-parameter recurrent scaling.

The doctor validates the scientific matrix and the *resolved* GCIQL runtime
configuration.  It constructs networks only for preflight/accounting; it does
not create a run directory, launch training, modify checkpoints, or invoke
Git.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from impls.agents import agents
from impls.computation.accounting import gciql_architecture_accounting
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args


STUDY_ID = 'M18'
ENVIRONMENT = 'puzzle-4x4-play-v0'
NUM_BUTTONS = 16
BLOCK_DEPTH = 2
K_VALUES = (1, 2, 4, 8)
ALPHA = 0.4
SLOT_NAMES = ('actor', 'value', 'critic')

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


def _expected_dataset_paths(dataset_root):
    return [
        Path(dataset_root) / f'{ENVIRONMENT}.npz',
        Path(dataset_root) / f'{ENVIRONMENT}-val.npz',
    ]


def _same_float(left, right):
    return abs(float(left) - float(right)) <= 1e-12


def _require_equal(config_id, label, actual, expected):
    if actual != expected:
        raise ValueError(
            f'{config_id}: {label} expected {expected!r}, got {actual!r}'
        )


def _validate_slot(config_id, slot_name, slot, expected_k):
    _require_equal(config_id, f'{slot_name}.enabled', slot.get('enabled'), True)
    for field, expected in (
        ('primitive', 'mlp'),
        ('structure', 'puzzle_tokens'),
        ('block', 'mlp_mixer'),
        ('topology', 'single_state'),
        ('parameter_sharing', 'shared'),
        ('credit', 'direct'),
        ('readout', 'mean_context'),
    ):
        _require_equal(config_id, f'{slot_name}.{field}', slot.get(field), expected)

    structure = slot.get('structure_kwargs', {})
    for field, expected in (
        ('num_buttons', NUM_BUTTONS),
        ('robot_dim', 19),
        ('button_feature_dim', 4),
        ('token_dim', 128),
        ('robot_hidden_dim', 128),
        ('index_embedding', True),
    ):
        _require_equal(config_id, f'{slot_name}.structure_kwargs.{field}', structure.get(field), expected)

    block = slot.get('block_kwargs', {})
    for field, expected in (
        ('num_blocks', BLOCK_DEPTH),
        ('token_hidden_dim', 64),
        ('channel_hidden_dim', 256),
        ('tm_mode', 'none'),
    ):
        _require_equal(config_id, f'{slot_name}.block_kwargs.{field}', block.get(field), expected)

    topology = slot.get('topology_kwargs', {})
    for field, expected in (
        ('iterations', expected_k),
        ('input_mapping', 'identity'),
        ('state_dim', 128),
        ('state_init', 'zero_buffer'),
        ('state_init_std', 1.0),
        ('input_injection', 'z_plus_x'),
        ('residual', False),
        ('parameter_sharing', 'shared'),
    ):
        actual = topology.get(field)
        if field == 'state_init_std':
            if not _same_float(actual, expected):
                raise ValueError(
                    f'{config_id}: {slot_name}.topology_kwargs.{field} '
                    f'expected {expected!r}, got {actual!r}'
                )
        else:
            _require_equal(config_id, f'{slot_name}.topology_kwargs.{field}', actual, expected)
    _require_equal(
        config_id,
        f'{slot_name}.readout_kwargs.output_dim',
        slot.get('readout_kwargs', {}).get('output_dim'),
        512,
    )


def _walk_keys(tree, path=()):
    if not hasattr(tree, 'items'):
        return [path]
    leaves = []
    for key, value in tree.items():
        leaves.extend(_walk_keys(value, path + (str(key),)))
    return leaves


def _assert_shared_parameter_tree(config_id, slot_name, slot_params):
    for path in _walk_keys(slot_params):
        if any(part.startswith('update_block_') or part.startswith('update_modules_') for part in path):
            raise ValueError(
                f'{config_id}: {slot_name} contains untied recurrent update subtree {path!r}'
            )
    if not any('update_module' in path for path in _walk_keys(slot_params)):
        raise ValueError(f'{config_id}: {slot_name} has no shared update_module subtree')


def _slot_params(params, slot_name):
    key = f'modules_{slot_name}'
    if key not in params:
        raise ValueError(f'Missing GCIQL parameter slot: {key}')
    return params[key]


def _study_checks(study):
    if study.study_id != STUDY_ID:
        raise ValueError(f'Expected {STUDY_ID} study, got {study.study_id!r}')
    if study.data.get('algorithms') != ['gciql']:
        raise ValueError('M18 must contain only algorithm=gciql')
    if study.data.get('environments') != [ENVIRONMENT]:
        raise ValueError(f'M18 environment matrix mismatch: {study.data.get("environments")!r}')
    if study.data.get('seeds') != [0]:
        raise ValueError('M18 must contain only seed=0')
    if study.data.get('placements') != ['actor+value+critic']:
        raise ValueError('M18 placement must be actor+value+critic')
    if study.data.get('primary_factors') != ['recurrent_compute_budget_K']:
        raise ValueError('M18 must expose only recurrent_compute_budget_K as its primary factor')
    matrix = study.data.get('matrix', {})
    if matrix.get('block_depth_L') != BLOCK_DEPTH:
        raise ValueError('M18 matrix.block_depth_L must be 2')
    if tuple(matrix.get('recurrent_compute_budget_K', ())) != K_VALUES:
        raise ValueError(f'M18 K matrix must be {K_VALUES!r}')
    if study.data.get('fixed_design', {}).get('alpha') != ALPHA:
        raise ValueError('M18 fixed_design.alpha must be 0.4')
    alpha_provenance = study.data.get('alpha_provenance', {})
    if alpha_provenance.get('runtime_authority') != 'configuration.agent_overrides.alpha':
        raise ValueError('M18 alpha runtime authority must be configuration.agent_overrides.alpha')
    if not _same_float(alpha_provenance.get('value'), ALPHA):
        raise ValueError('M18 alpha provenance value must be 0.4')
    if study.data.get('protocol', {}).get('formal_training_started') is not False:
        raise ValueError('formal_training_started must remain false before a user manual launch')


def _runtime_checks(config_id, config, expected_k):
    if not _same_float(config['alpha'], ALPHA):
        raise ValueError(f'{config_id}: resolved runtime alpha is {config["alpha"]!r}, not {ALPHA}')
    if config.get('actor_loss') != 'ddpgbc':
        raise ValueError(f'{config_id}: resolved actor_loss must be ddpgbc')
    slots = config.get('compute', {})
    if set(slots) != set(SLOT_NAMES):
        raise ValueError(f'{config_id}: resolved GCIQL slots mismatch: {sorted(slots)!r}')
    for slot_name in SLOT_NAMES:
        slot = slots[slot_name]
        _validate_slot(config_id, slot_name, slot, expected_k)


def _job_architecture(config_id, agent, config, expected_k):
    slot_accounting = _computation_slot_accounting(agent, config)
    if set(slot_accounting) != set(SLOT_NAMES):
        raise ValueError(f'{config_id}: missing structured accounting slots')
    architecture = gciql_architecture_accounting(agent.network.params, config, slot_accounting)
    actor = slot_accounting['actor']
    for slot_name, report in slot_accounting.items():
        _assert_shared_parameter_tree(config_id, slot_name, _slot_params(agent.network.params, slot_name))
        if report.get('block_depth_L') != BLOCK_DEPTH:
            raise ValueError(f'{config_id}: {slot_name} accounting L mismatch')
        if report.get('iterations_K') != expected_k:
            raise ValueError(f'{config_id}: {slot_name} accounting K mismatch')
        if report.get('state_dim') != 128 or report.get('input_mapping') != 'identity':
            raise ValueError(f'{config_id}: {slot_name} state/mapping accounting mismatch')
        if report.get('state_init') != 'zero_buffer' or report.get('residual') is not False:
            raise ValueError(f'{config_id}: {slot_name} state-init/residual accounting mismatch')
        if report.get('parameter_sharing') != 'shared':
            raise ValueError(f'{config_id}: {slot_name} parameter sharing accounting mismatch')
        physical_unique = int(report.get('unique_mixer_layers', -1))
        physical_executed = int(report.get('executed_mixer_layers', -1))
        if physical_unique <= 0 or physical_executed != physical_unique * expected_k:
            raise ValueError(f'{config_id}: {slot_name} executed Mixer accounting mismatch')
    if actor.get('unique_mixer_layers') != BLOCK_DEPTH:
        raise ValueError(f'{config_id}: actor must own exactly {BLOCK_DEPTH} unique Mixer layers')
    if actor.get('executed_mixer_layers') != BLOCK_DEPTH * expected_k:
        raise ValueError(f'{config_id}: actor executed Mixer layers must equal 2K')
    return {
        'total_trainable_params': int(architecture['total_trainable_params']),
        'total_dense_macs': int(architecture['total_dense_macs']),
        'actor_trainable_params': int(actor['trainable_params']),
        'actor_body_dense_macs': int(actor['structured_body_dense_macs']),
        'actor_unique_mixer_layers': int(actor['unique_mixer_layers']),
        'actor_executed_mixer_layers': int(actor['executed_mixer_layers']),
        'actor_executed_depth': int(actor['executed_sequential_depth']),
        'actor_buffer_elements': int(actor['buffer_elements']),
        'slot_trainable_params': {
            slot_name: int(slot_accounting[slot_name]['trainable_params'])
            for slot_name in SLOT_NAMES
        },
        'slot_buffer_elements': {
            slot_name: int(slot_accounting[slot_name]['buffer_elements'])
            for slot_name in SLOT_NAMES
        },
        'slot_unique_mixer_layers_physical': {
            slot_name: int(slot_accounting[slot_name]['unique_mixer_layers'])
            for slot_name in SLOT_NAMES
        },
        'slot_executed_mixer_layers_physical': {
            slot_name: int(slot_accounting[slot_name]['executed_mixer_layers'])
            for slot_name in SLOT_NAMES
        },
    }


def validate(study_path, dataset_root, run_root, gpus):
    study = load_study(study_path)
    _study_checks(study)
    dataset_root = Path(dataset_root)
    missing = [path for path in _expected_dataset_paths(dataset_root) if not path.is_file()]
    if missing:
        raise ValueError('Missing dataset files:\n' + '\n'.join(f'  {path}' for path in missing))

    config_paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    if len(config_paths) != len(K_VALUES):
        raise ValueError(f'M18 requires exactly {len(K_VALUES)} configs, found {len(config_paths)}')

    args = _agent_args()
    seen_k = set()
    rows = []
    frozen_reference = None
    parameter_reference = None
    run_paths = set()
    for config_path in config_paths:
        _, configuration = prepare_run_design(study.path, config_path)
        data = configuration.data
        config_id = configuration.config_id
        if data.get('algorithm') != 'gciql' or data.get('environment') != ENVIRONMENT:
            raise ValueError(f'{config_id}: algorithm/environment mismatch')
        if data.get('placement') != 'actor+value+critic' or data.get('protocol_stage') != 'formal':
            raise ValueError(f'{config_id}: placement/protocol_stage mismatch')
        if data.get('executable') is not True:
            raise ValueError(f'{config_id}: executable must be true')
        factors = data.get('factors', {})
        expected_k = factors.get('recurrent_compute_budget_K')
        if expected_k not in K_VALUES:
            raise ValueError(f'{config_id}: invalid recurrent_compute_budget_K={expected_k!r}')
        if factors.get('K') != expected_k or factors.get('block_depth_L') != BLOCK_DEPTH:
            raise ValueError(f'{config_id}: factor K/L aliases are inconsistent')
        if not _same_float(factors.get('alpha'), ALPHA):
            raise ValueError(f'{config_id}: factors.alpha must be {ALPHA}')
        if expected_k in seen_k:
            raise ValueError(f'{config_id}: duplicate K={expected_k}')
        seen_k.add(expected_k)

        overrides = data.get('agent_overrides', {})
        if not _same_float(overrides.get('alpha'), ALPHA):
            raise ValueError(f'{config_id}: agent_overrides.alpha must explicitly be {ALPHA}')
        if overrides.get('actor_loss') != 'ddpgbc':
            raise ValueError(f'{config_id}: agent_overrides.actor_loss must be ddpgbc')
        for slot_name in SLOT_NAMES:
            _validate_slot(config_id, slot_name, overrides.get('compute', {}).get(slot_name, {}), expected_k)

        config = _make_config(args, configuration=configuration)
        _runtime_checks(config_id, config, expected_k)
        frozen = {field: config[field] for field in FROZEN_AGENT_FIELDS}
        if frozen_reference is None:
            frozen_reference = frozen
        elif frozen != frozen_reference:
            raise ValueError(f'{config_id}: non-K resolved GCIQL settings differ from reference')

        observations = np.zeros((2, 19 + NUM_BUTTONS * 4), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        agent = agents['gciql'].create(0, observations, actions, config)
        architecture = _job_architecture(config_id, agent, config, expected_k)
        invariant = {
            'total_trainable_params': architecture['total_trainable_params'],
            'actor_trainable_params': architecture['actor_trainable_params'],
            'slot_trainable_params': architecture['slot_trainable_params'],
            'slot_buffer_elements': architecture['slot_buffer_elements'],
            'slot_unique_mixer_layers_physical': architecture['slot_unique_mixer_layers_physical'],
        }
        if parameter_reference is None:
            parameter_reference = invariant
        elif invariant != parameter_reference:
            raise ValueError(
                f'{config_id}: trainable parameter/buffer/unique-layer invariance over K failed'
            )

        run_dir = make_run_path(
            run_root, study.study_id, config_id, configuration.slug, ENVIRONMENT, 0,
            run_attempt=0,
        )
        if run_dir in run_paths:
            raise ValueError(f'{config_id}: duplicate output path {run_dir}')
        run_paths.add(run_dir)
        if run_dir.exists():
            raise ValueError(f'Output path already exists; refusing overwrite: {run_dir}')
        rows.append({
            'config_id': config_id,
            'environment': ENVIRONMENT,
            'L': BLOCK_DEPTH,
            'K': expected_k,
            'alpha': float(config['alpha']),
            'gpu': gpus[len(rows) % len(gpus)],
            'run_dir': str(run_dir),
            **architecture,
        })
        del agent, config
        gc.collect()

    if tuple(sorted(seen_k)) != K_VALUES:
        raise ValueError(f'M18 K matrix incomplete: {sorted(seen_k)!r}')
    rows.sort(key=lambda row: row['K'])
    previous_macs = None
    for row in rows:
        if row['actor_unique_mixer_layers'] != BLOCK_DEPTH:
            raise ValueError('Actor unique Mixer layer count is not fixed at L=2')
        if row['actor_executed_mixer_layers'] != BLOCK_DEPTH * row['K']:
            raise ValueError('Actor executed Mixer layer count is not 2K')
        if previous_macs is not None and row['actor_body_dense_macs'] <= previous_macs:
            raise ValueError('Actor executed body MACs must grow strictly with K')
        previous_macs = row['actor_body_dense_macs']
    return {
        'status': 'PASS',
        'study_id': study.study_id,
        'expected_runs': len(K_VALUES),
        'environment': ENVIRONMENT,
        'L': BLOCK_DEPTH,
        'K_values': list(K_VALUES),
        'alpha': ALPHA,
        'seed': 0,
        'gpus': list(gpus),
        'dataset_root': str(dataset_root.resolve()),
        'run_root': str(Path(run_root).resolve()),
        'frozen_agent_fields': frozen_reference,
        'jobs': rows,
        'formal_training_started': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M18_puzzle_recurrent_compute_scaling/study.yaml')
    parser.add_argument('--dataset-root', default='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--json-output', default=None)
    args = parser.parse_args(argv)
    try:
        report = validate(args.study, args.dataset_root, args.run_root, _parse_gpus(args.gpus))
    except Exception as error:
        print(f'M18 PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2
    print('M18 PREFLIGHT: PASS')
    print(
        f'{report["expected_runs"]} jobs; environment={report["environment"]}; '
        f'L={report["L"]}; K={report["K_values"]}; alpha={report["alpha"]}; seed=0'
    )
    print('config_id\tenv\tL\tK\talpha\tGPU\tparams(total/actor)\tactor MAC\tunique/executed Mixer\tdepth\tbuffer\trun_dir')
    for job in report['jobs']:
        print(
            f'{job["config_id"]}\t{job["environment"]}\t{job["L"]}\t{job["K"]}\t'
            f'{job["alpha"]}\t{job["gpu"]}\t'
            f'{job["total_trainable_params"]}/{job["actor_trainable_params"]}\t'
            f'{job["actor_body_dense_macs"]}\t'
            f'{job["actor_unique_mixer_layers"]}/{job["actor_executed_mixer_layers"]}\t'
            f'{job["actor_executed_depth"]}\t{job["actor_buffer_elements"]}\t{job["run_dir"]}'
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
