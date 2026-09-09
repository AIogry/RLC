"""Non-executing preflight and optional real-data smoke for M16D.

The preflight validates the four-cell M16D Study against the historical M16B
S002 configuration.  It resolves the actual GCIQL runtime configuration and
constructs each network, but never creates a Run directory, starts formal
training, performs Git operations, or writes experiment artifacts.

``--smoke`` additionally performs two optimizer updates and one tiny
evaluation episode per selected configuration using the real Puzzle dataset.
This is an infrastructure check only; it is not a formal M16D run.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from impls.agents import agents
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.management import jsonable
from impls.main import _computation_slot_accounting, _make_config, _parse_args
from impls.utils.datasets import GCDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = 'M16D'
ANCHOR_STUDY_ID = 'M16B'
SLOT_NAMES = ('actor', 'value', 'critic')

ENVIRONMENTS = {
    'puzzle-3x3-play-v0': {'tag': 'P3X3', 'num_buttons': 9, 'observation_dim': 55},
    'puzzle-4x4-play-v0': {'tag': 'P4X4', 'num_buttons': 16, 'observation_dim': 83},
    'puzzle-4x5-play-v0': {'tag': 'P4X5', 'num_buttons': 20, 'observation_dim': 99},
    'puzzle-4x6-play-v0': {'tag': 'P4X6', 'num_buttons': 24, 'observation_dim': 115},
}

EXPECTED_CONFIGS = {
    environment: f'M16D-{spec["tag"]}-S002-A04'
    for environment, spec in ENVIRONMENTS.items()
}
ANCHOR_CONFIGS = {
    'puzzle-3x3-play-v0': 'M16B-3x3-S002',
    'puzzle-4x4-play-v0': 'M16B-4x4-S002',
    'puzzle-4x5-play-v0': 'M16B-4x5-S002',
    'puzzle-4x6-play-v0': 'M16B-4x6-S002',
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


def _agent_args():
    return _parse_args(['--agent', 'gciql'])


def _same_float(left: Any, right: Any) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _expected_dataset_paths(dataset_root: Path, environment: str):
    return (
        dataset_root / f'{environment}.npz',
        dataset_root / f'{environment}-val.npz',
    )


def _first_difference(left: Any, right: Any, path=()):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            if key not in left or key not in right:
                return path + (str(key),), left.get(key), right.get(key)
            difference = _first_difference(left[key], right[key], path + (str(key),))
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path + ('length',), len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, path + (str(index),))
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, left, right
    return None


def _resolved_config(study_path: Path, config_path: Path):
    _, configuration = prepare_run_design(study_path, config_path)
    return configuration, _make_config(_agent_args(), configuration=configuration)


def _check_study(study, anchor_study):
    _require(study.study_id == STUDY_ID, f'Expected study_id={STUDY_ID!r}, got {study.study_id!r}')
    _require(study.data.get('algorithms') == ['gciql'], 'M16D must contain only algorithm=gciql')
    _require(
        study.data.get('placements') == ['actor+value+critic'],
        'M16D placement must be actor+value+critic',
    )
    _require(
        study.data.get('environments') == list(ENVIRONMENTS),
        f'M16D environment matrix mismatch: {study.data.get("environments")!r}',
    )
    _require(study.data.get('seeds') == [0], 'M16D must contain only seed=0')
    _require(study.data.get('conditions') == ['S002'], 'M16D must contain only condition S002')
    _require(study.data.get('fixed_design', {}).get('alpha') == 0.4, 'M16D fixed_design.alpha must be 0.4')
    _require(study.data.get('alpha_policy', {}).get('value') == 0.4, 'M16D alpha_policy.value must be 0.4')
    _require(
        study.data.get('alpha_policy', {}).get('runtime_authority')
        == 'configuration.agent_overrides.alpha',
        'M16D alpha must be controlled by configuration.agent_overrides.alpha',
    )
    provenance = study.data.get('alpha_provenance', {})
    _require(
        provenance.get('status') == 'empirically_motivated_operating_point_not_alpha_optimization',
        'M16D alpha provenance must not present 0.4 as an optimum',
    )
    _require(
        study.data.get('protocol', {}).get('formal_training_started') is False,
        'M16D formal_training_started must remain false before manual launch',
    )
    _require(study.data.get('acceptance', {}).get('expected_formal_runs') == 4, 'M16D must declare 4 formal runs')

    anchor_data = anchor_study.data
    fixed_design = study.data.get('fixed_design', {})
    anchor_fixed_design = anchor_data.get('fixed_design', {})
    _require(
        {key: value for key, value in fixed_design.items() if key != 'alpha'}
        == {key: value for key, value in anchor_fixed_design.items() if key != 'alpha'},
        'M16D fixed_design differs from M16B beyond alpha',
    )
    _require(
        jsonable(study.data.get('protocol', {})) == jsonable(anchor_data.get('protocol', {})),
        'M16D protocol differs from M16B protocol',
    )


def _check_slot(config_id: str, slot_name: str, slot: Mapping[str, Any], num_buttons: int):
    _require(slot.get('enabled') is True, f'{config_id}: {slot_name}.enabled must be true')
    for field, expected in (
        ('primitive', 'mlp'),
        ('structure', 'puzzle_tokens'),
        ('topology', 'feedforward'),
        ('block', 'mlp_mixer'),
        ('credit', 'direct'),
    ):
        _require(
            slot.get(field) == expected,
            f'{config_id}: {slot_name}.{field} expected {expected!r}, got {slot.get(field)!r}',
        )
    _require(
        slot.get('relation_mode', 'legacy_none') == 'legacy_none',
        f'{config_id}: {slot_name} must be relation-free',
    )
    _require('topology_kwargs' not in slot, f'{config_id}: {slot_name} must not define recurrent topology_kwargs')
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
            f'{config_id}: {slot_name}.structure_kwargs.{field} '
            f'expected {expected!r}, got {structure.get(field)!r}',
        )


def _check_runtime(config_id: str, config, num_buttons: int):
    _require(_same_float(config['alpha'], 0.4), f'{config_id}: resolved alpha is not 0.4')
    _require(config.get('actor_loss') == 'ddpgbc', f'{config_id}: actor_loss must be ddpgbc')
    slots = config.get('compute', {})
    _require(set(slots) == set(SLOT_NAMES), f'{config_id}: resolved GCIQL slots mismatch')
    for slot_name in SLOT_NAMES:
        slot = slots[slot_name]
        _check_slot(config_id, slot_name, slot, num_buttons)
        _require(slot.get('readout') == 'mean', f'{config_id}: {slot_name}.readout must be mean')
        _require(slot.get('topology') == 'feedforward', f'{config_id}: {slot_name} must be feedforward')


def _anchor_runtime_parity(config_id: str, candidate, anchor):
    candidate_json = jsonable(candidate)
    anchor_json = jsonable(anchor)
    candidate_json = copy.deepcopy(candidate_json)
    candidate_json['alpha'] = anchor_json['alpha']
    difference = _first_difference(candidate_json, anchor_json)
    if difference is not None:
        raise ValueError(
            f'{config_id}: resolved runtime differs from M16B after alpha normalization '
            f'at {".".join(difference[0])}: candidate={difference[1]!r}, '
            f'anchor={difference[2]!r}'
        )


def validate(study_path, anchor_study_path, dataset_root, run_root, gpu='0'):
    """Validate M16D and return a structured, non-mutating preflight report."""

    study = load_study(study_path)
    anchor_study = load_study(anchor_study_path)
    _require(anchor_study.study_id == ANCHOR_STUDY_ID, 'M16D anchor Study must be M16B')
    _check_study(study, anchor_study)

    dataset_root = Path(dataset_root)
    missing = [
        path
        for environment in ENVIRONMENTS
        for path in _expected_dataset_paths(dataset_root, environment)
        if not path.is_file()
    ]
    if missing:
        raise ValueError('Missing dataset files:\n' + '\n'.join(f'  {path}' for path in missing))

    config_dir = Path(study.path).parent / 'configs'
    config_paths = sorted(config_dir.glob('*.yaml'))
    expected_paths = {f'{config_id}.yaml' for config_id in EXPECTED_CONFIGS.values()}
    _require(
        {path.name for path in config_paths} == expected_paths,
        f'M16D config matrix mismatch: observed={[path.name for path in config_paths]!r}',
    )

    rows = []
    seen = set()
    args = _agent_args()
    for config_path in config_paths:
        configuration, config = _resolved_config(study.path, config_path)
        data = configuration.data
        config_id = configuration.config_id
        environment = data.get('environment')
        _require(environment in ENVIRONMENTS, f'{config_id}: unknown environment {environment!r}')
        expected_config_id = EXPECTED_CONFIGS[environment]
        _require(config_id == expected_config_id, f'Expected {expected_config_id}, got {config_id}')
        _require(data.get('condition_id') == 'S002', f'{config_id}: condition must be S002')
        _require(data.get('protocol_stage') == 'formal', f'{config_id}: protocol_stage must be formal')
        _require(data.get('placement') == 'actor+value+critic', f'{config_id}: placement mismatch')
        _require(data.get('executable') is True, f'{config_id}: executable must be true')

        spec = ENVIRONMENTS[environment]
        factors = data.get('factors', {})
        for field, expected in (
            ('environment', environment),
            ('condition', 'S002'),
            ('structure', 'puzzle_tokens'),
            ('topology', 'feedforward'),
            ('block', 'mlp_mixer'),
            ('num_buttons', spec['num_buttons']),
            ('num_mixer_blocks', 2),
        ):
            _require(factors.get(field) == expected, f'{config_id}: factors.{field} mismatch')
        _require(_same_float(factors.get('alpha'), 0.4), f'{config_id}: factors.alpha must be 0.4')

        overrides = data.get('agent_overrides', {})
        _require(_same_float(overrides.get('alpha'), 0.4), f'{config_id}: agent_overrides.alpha must be 0.4')
        _require(overrides.get('actor_loss') == 'ddpgbc', f'{config_id}: agent_overrides.actor_loss mismatch')
        compute_overrides = overrides.get('compute', {})
        _require(set(compute_overrides) == set(SLOT_NAMES), f'{config_id}: config slot set mismatch')
        for slot_name in SLOT_NAMES:
            _check_slot(config_id, slot_name, compute_overrides[slot_name], spec['num_buttons'])

        _check_runtime(config_id, config, spec['num_buttons'])
        for field in FROZEN_AGENT_FIELDS:
            if field == 'alpha':
                continue
            anchor_path = Path(anchor_study.path).parent / 'configs' / f'{ANCHOR_CONFIGS[environment]}.yaml'
            _, anchor_config = _resolved_config(anchor_study.path, anchor_path)
            _require(
                jsonable(config[field]) == jsonable(anchor_config[field]),
                f'{config_id}: resolved {field} differs from M16B S002',
            )
        anchor_path = Path(anchor_study.path).parent / 'configs' / f'{ANCHOR_CONFIGS[environment]}.yaml'
        _, anchor_config = _resolved_config(anchor_study.path, anchor_path)
        _anchor_runtime_parity(config_id, config, anchor_config)

        observations = np.zeros((2, spec['observation_dim']), dtype=np.float32)
        actions = np.zeros((2, 5), dtype=np.float32)
        agent = agents['gciql'].create(0, observations, actions, config)
        accounting = _computation_slot_accounting(agent, config)
        _require(set(accounting) == set(SLOT_NAMES), f'{config_id}: structured accounting slots mismatch')
        run_dir = make_run_path(
            run_root,
            study.study_id,
            config_id,
            configuration.slug,
            environment,
            0,
            run_attempt=0,
        )
        _require(run_dir not in seen, f'{config_id}: duplicate run path {run_dir}')
        _require(not run_dir.exists(), f'Output path already exists; refusing overwrite: {run_dir}')
        seen.add(run_dir)
        rows.append({
            'config_id': config_id,
            'environment': environment,
            'num_buttons': spec['num_buttons'],
            'observation_dim': spec['observation_dim'],
            'alpha': float(config['alpha']),
            'seed': 0,
            'run_dir': str(run_dir),
            'structured_slots': sorted(accounting),
        })
        del agent, accounting, config, anchor_config
        gc.collect()

    expected_cells = set(EXPECTED_CONFIGS)
    _require(
        {row['environment'] for row in rows} == expected_cells,
        'M16D environment matrix is incomplete',
    )
    return {
        'status': 'PASS',
        'study_id': STUDY_ID,
        'anchor_study_id': ANCHOR_STUDY_ID,
        'anchor_study_path': str(Path(anchor_study_path).resolve()),
        'expected_formal_runs': 4,
        'configs': [row['config_id'] for row in rows],
        'environments': list(ENVIRONMENTS),
        'observation_dims': {env: spec['observation_dim'] for env, spec in ENVIRONMENTS.items()},
        'seeds': [0],
        'alpha': 0.4,
        'mixer_depth': 2,
        'readout': 'mean',
        'placement': 'actor+value+critic',
        'gpu_policy': {'physical_gpu': str(gpu), 'jobs_per_gpu': 2},
        'dataset_root': str(dataset_root.resolve()),
        'run_root': str(Path(run_root).resolve()),
        'training_protocol': jsonable(study.data['protocol']),
        'jobs': rows,
        'formal_training_started': False,
    }


def run_smoke(study_path, dataset_root, report, selected_config_ids=None):
    """Run two updates and one tiny evaluation per selected M16D config."""

    selected = set(selected_config_ids or report['configs'])
    unknown = selected - set(report['configs'])
    if unknown:
        raise ValueError(f'Unknown M16D smoke config(s): {sorted(unknown)!r}')
    results = []
    for row in report['jobs']:
        if row['config_id'] not in selected:
            continue
        config_path = Path(study_path).parent / 'configs' / f'{row["config_id"]}.yaml'
        configuration, config = _resolved_config(Path(study_path), config_path)
        env = None
        raw_train = None
        dataset = None
        agent = None
        try:
            env, raw_train, _ = make_env_and_datasets(
                row['environment'],
                seed=230400 + row['num_buttons'],
                dataset_seed=230500 + row['num_buttons'],
                dataset_dir=dataset_root,
            )
            dataset = GCDataset(raw_train, config, rng=230600 + row['num_buttons'])
            batch = dataset.sample(config['batch_size'])
            agent = agents['gciql'].create(
                230700 + row['num_buttons'],
                batch['observations'],
                batch['actions'],
                config,
            )
            update_metrics = []
            for _ in range(2):
                agent, info = agent.update(batch)
                if not all(np.all(np.isfinite(np.asarray(value))) for value in info.values()):
                    raise ValueError(f'{row["config_id"]}: non-finite optimizer update metric')
                update_metrics.append(sorted(info))
            stats, _, renders = evaluate(
                agent,
                env,
                task_id=1,
                config=config,
                num_eval_episodes=1,
                num_video_episodes=0,
                eval_temperature=0.0,
                eval_gaussian=None,
                seed=230800 + row['num_buttons'],
            )
            success_keys = [key for key in stats if key == 'success' or key.endswith('success')]
            if not success_keys or renders:
                raise ValueError(f'{row["config_id"]}: tiny evaluation contract failed')
            result = {
                'config_id': row['config_id'],
                'environment': row['environment'],
                'updates': 2,
                'finite_updates': True,
                'eval_success_keys': sorted(success_keys),
                'eval_stats': {key: float(value) for key, value in stats.items() if key in success_keys},
            }
            results.append(result)
            print(
                f'SMOKE PASS {row["config_id"]}: updates=2; '
                f'eval_success={result["eval_stats"]}',
            )
        finally:
            if env is not None:
                env.close()
            del agent, dataset, raw_train, config
            gc.collect()
    return results


def _parse_config_ids(values):
    if not values:
        return None
    result = set()
    for value in values:
        result.update(item.strip() for item in value.split(',') if item.strip())
    return result or None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=str(REPO_ROOT / 'experiments/M16D_puzzle_mixer_alpha04_completion/study.yaml'))
    parser.add_argument('--anchor-study', default=str(REPO_ROOT / 'experiments/M16B_puzzle_alpha_correction/study.yaml'))
    parser.add_argument('--dataset-root', default='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--smoke', action='store_true', help='Run the non-formal real-data two-update smoke.')
    parser.add_argument('--config-id', action='append', help='Restrict --smoke to one or more config IDs.')
    parser.add_argument('--json-output', default=None, help='Optional reusable machine-readable preflight report.')
    args = parser.parse_args(argv)
    try:
        report = validate(
            args.study,
            args.anchor_study,
            args.dataset_root,
            args.run_root,
            gpu=args.gpu,
        )
        if args.smoke:
            report['smoke'] = run_smoke(
                args.study,
                args.dataset_root,
                report,
                selected_config_ids=_parse_config_ids(args.config_id),
            )
    except Exception as error:
        print(f'M16D PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2

    print('M16D PREFLIGHT: PASS')
    print(
        f'configs={len(report["configs"])}; seeds={report["seeds"]}; '
        f'planned_runs={report["expected_formal_runs"]}; alpha={report["alpha"]}; '
        f'Mixer-L={report["mixer_depth"]}; readout={report["readout"]}'
    )
    print('config_id\tenvironment\tN\tD\talpha\tseed\trun_dir')
    for job in report['jobs']:
        print(
            f'{job["config_id"]}\t{job["environment"]}\t{job["num_buttons"]}\t'
            f'{job["observation_dim"]}\t{job["alpha"]}\t{job["seed"]}\t{job["run_dir"]}'
        )
    if args.smoke:
        print(f'real_data_smoke_configs={len(report.get("smoke", []))}; formal_training_started=False')
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'JSON report: {output}')
    print('Formal training was not started. Manual launch remains required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
