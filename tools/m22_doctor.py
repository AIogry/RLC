"""Validate the declarative M22 Puzzle baseline Study before launch.

This is intentionally a campaign-preparation doctor.  It validates only the
M22 Study, its audited provenance, its actual RLC config resolution, the
dataset hashes, and the deterministic Run design.  It does not inspect or
mutate other milestone namespaces and it never creates a Run directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.computation import resolve_slot_spec  # noqa: E402
from impls.experiment import (  # noqa: E402
    ExperimentError,
    load_study,
    make_run_path,
    prepare_run_design,
)
from impls.experiment.management import jsonable  # noqa: E402
from impls.main import _make_config  # noqa: E402


STUDY_ID = 'M22'
CAMPAIGN_NAME = 'M22 — OGBench Puzzle Baselines in Unified RLC'
ALGORITHMS = ('gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql')
ENVIRONMENTS = (
    'puzzle-3x3-play-v0',
    'puzzle-4x4-play-v0',
    'puzzle-4x5-play-v0',
    'puzzle-4x6-play-v0',
)
SEEDS = (0, 1, 2)
CONFIG_IDS = {
    f'M22-{algorithm.upper()}-P{environment.split("-")[1].replace("x", "X").upper()}'
    for algorithm in ALGORITHMS
    for environment in ENVIRONMENTS
}
OFFICIAL_OVERRIDES = {
    'gcbc': {},
    'gcivl': {'alpha': 10.0},
    'gciql': {'alpha': 1.0},
    'qrl': {'alpha': 0.3},
    'crl': {'alpha': 3.0},
    'hiql': {'high_alpha': 3.0, 'low_alpha': 3.0, 'subgoal_steps': 10},
}
IMPLEMENTATION_SEMANTICS = {
    'gcbc': ('upstream_semantic_match', None),
    'gcivl': ('rlc_variant_documented', 'post-gradient target Polyak update'),
    'gciql': ('rlc_variant_documented', 'post-gradient target Polyak update'),
    'qrl': ('upstream_semantic_match', None),
    'crl': ('upstream_semantic_match', None),
    'hiql': ('upstream_semantic_match', None),
}
SLOT_NAMES = {
    'gcbc': ('actor',),
    'gcivl': ('actor', 'value'),
    'gciql': ('actor', 'value', 'critic'),
    'qrl': ('actor', 'value', 'dynamics'),
    'crl': ('actor', 'critic_state', 'critic_goal', 'value_state', 'value_goal'),
    'hiql': ('low_actor', 'high_actor', 'value'),
}
PROTOCOL = {
    'train_steps': 1_000_000,
    'batch_size': 1024,
    'log_interval': 5_000,
    'eval_interval': 100_000,
    'eval_tasks': 'all',
    'eval_episodes': 50,
    'eval_temperature': 0.0,
    'eval_gaussian': None,
    'video_episodes': 0,
    'save_interval': 100_000,
    'save_best_checkpoint': True,
    'save_last_checkpoint': True,
}
DEFAULT_STUDY = REPO_ROOT / 'experiments/M22_puzzle_baselines_unified_rlc/study.yaml'
DEFAULT_SEMANTIC_AUDIT = REPO_ROOT / 'docs/9-9/canonical_semantics_audit.json'
DEFAULT_BASELINE_AUDIT = REPO_ROOT / 'docs/9-9/M22_upstream_baseline_audit.json'
DEFAULT_DATASET_AUDIT = REPO_ROOT / 'docs/9-9/M22_dataset_audit.json'
DEFAULT_DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
DEFAULT_RUN_PARENT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _same(actual: Any, expected: Any) -> bool:
    """Compare JSON/YAML values after converting ConfigDict-like values."""

    return jsonable(actual) == jsonable(expected)


def _semantic_gate_status(artifact: Mapping[str, Any]) -> str | None:
    # The correction artifact uses ``status`` while the baseline audit also
    # exposes the more specific ``semantic_gate_status`` spelling.
    return artifact.get('semantic_gate', artifact.get('semantic_gate_status', artifact.get('status')))


def _expected_config_ids() -> set[str]:
    return set(CONFIG_IDS)


def _config_id(algorithm: str, environment: str) -> str:
    puzzle = environment.split('-')[1].replace('x', 'X').upper()
    return f'M22-{algorithm.upper()}-P{puzzle}'


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_args(algorithm: str) -> SimpleNamespace:
    return SimpleNamespace(
        agent=algorithm,
        batch_size=None,
        latent_dim=None,
        actor_loss=None,
        width=None,
        depth=None,
        computation=False,
    )


def _check_audits(
    semantic_path: Path,
    baseline_path: Path,
    dataset_audit_path: Path,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        semantic = _read_json(semantic_path)
        baseline = _read_json(baseline_path)
        dataset = _read_json(dataset_audit_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f'audit artifact read failure: {error}')
        return {}, {}, {}
    for label, artifact in (
        ('semantic audit', semantic),
        ('baseline audit', baseline),
        ('dataset audit', dataset),
    ):
        if artifact.get('status') != 'pass':
            errors.append(f'{label} status is not pass: {artifact.get("status")!r}')
    if _semantic_gate_status(semantic) != 'pass':
        errors.append(
            f'semantic gate is not pass: {_semantic_gate_status(semantic)!r}'
        )
    if baseline.get('semantic_gate_status') != 'pass':
        errors.append(
            'baseline audit semantic_gate_status is not pass: '
            f'{baseline.get("semantic_gate_status")!r}'
        )
    if semantic.get('material_mismatches', []) not in ([], None):
        errors.append('semantic audit contains material_mismatches')
    if semantic.get('byte_for_byte_upstream_implementation_equivalence') is not False:
        errors.append('M22 must not claim byte-for-byte upstream equivalence')
    if baseline.get('official_hyperparameters_authoritative') is not True:
        errors.append('baseline audit does not mark official hyperparameters authoritative')
    if baseline.get('formal_training_started') is not False:
        errors.append('baseline audit says formal training has already started')
    observed = baseline.get('baseline_provenance', {})
    for algorithm in ALGORITHMS:
        expected_semantics, expected_difference = IMPLEMENTATION_SEMANTICS[algorithm]
        record = observed.get(algorithm, {})
        expected = {
            'hyperparameter_provenance': 'official_ogbench',
            'implementation_semantics': expected_semantics,
            'documented_difference': expected_difference,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                errors.append(
                    f'baseline provenance mismatch {algorithm}.{key}: '
                    f'expected={value!r}, observed={record.get(key)!r}'
                )
    if dataset.get('status') != 'pass' or dataset.get('failures'):
        errors.append('dataset audit is not a clean pass')
    return semantic, baseline, dataset


def _check_study_and_configs(
    study_path: Path,
    baseline: Mapping[str, Any],
    errors: list[str],
) -> tuple[Any, list[Any]]:
    try:
        study = load_study(study_path)
    except (ExperimentError, OSError, ValueError) as error:
        errors.append(f'Study load failure: {error}')
        return None, []
    data = study.data
    if study.study_id != STUDY_ID:
        errors.append(f'Study ID mismatch: {study.study_id!r}')
    if data.get('name') != CAMPAIGN_NAME:
        errors.append(f'Study name mismatch: {data.get("name")!r}')
    if tuple(data.get('algorithms', ())) != ALGORITHMS:
        errors.append(f'algorithm set/order mismatch: {data.get("algorithms")!r}')
    if tuple(data.get('environments', ())) != ENVIRONMENTS:
        errors.append(f'environment set/order mismatch: {data.get("environments")!r}')
    if tuple(data.get('seeds', ())) != SEEDS:
        errors.append(f'seed set/order mismatch: {data.get("seeds")!r}')
    protocol = data.get('protocol', {})
    if not isinstance(protocol, Mapping):
        errors.append('Study protocol must be a mapping')
    else:
        for key, expected in PROTOCOL.items():
            if not _same(protocol.get(key), expected):
                errors.append(
                    f'protocol mismatch {key}: expected={expected!r}, '
                    f'observed={protocol.get(key)!r}'
                )
    if data.get('primary_metric') != 'evaluation/overall_success':
        errors.append('Study primary_metric must be evaluation/overall_success')

    config_dir = study.path.parent / 'configs'
    paths = sorted(config_dir.glob('*.yaml'))
    expected_ids = _expected_config_ids()
    observed_ids = {path.stem for path in paths}
    if len(paths) != 24:
        errors.append(f'config count is {len(paths)}, expected 24')
    if observed_ids != expected_ids:
        errors.append(
            f'config IDs mismatch: missing={sorted(expected_ids - observed_ids)}, '
            f'extra={sorted(observed_ids - expected_ids)}'
        )
    configurations = []
    for path in paths:
        try:
            _, configuration = prepare_run_design(study_path, path)
        except (ExperimentError, OSError, ValueError) as error:
            errors.append(f'configuration load failure {path.name}: {error}')
            continue
        configurations.append(configuration)
        data = configuration.data
        algorithm = data.get('algorithm')
        environment = data.get('environment')
        if algorithm not in ALGORITHMS:
            errors.append(f'{path.name}: unsupported algorithm {algorithm!r}')
            continue
        if environment not in ENVIRONMENTS:
            errors.append(f'{path.name}: unsupported environment {environment!r}')
            continue
        expected_id = _config_id(algorithm, environment)
        if data.get('config_id') != expected_id:
            errors.append(
                f'{path.name}: config_id does not match algorithm/environment: '
                f'{data.get("config_id")!r} != {expected_id!r}'
            )
        if 'seed' in data:
            errors.append(f'{path.name}: seed must not be stored in a config')
        if data.get('hyperparameter_provenance') != 'official_ogbench':
            errors.append(f'{path.name}: wrong hyperparameter provenance')
        expected_semantics, expected_difference = IMPLEMENTATION_SEMANTICS[algorithm]
        if data.get('implementation_semantics') != expected_semantics:
            errors.append(f'{path.name}: wrong implementation semantics')
        if data.get('documented_difference') != expected_difference:
            errors.append(f'{path.name}: wrong documented difference')
        factors = data.get('factors', {})
        if factors.get('algorithm') != algorithm or factors.get('environment') != environment:
            errors.append(f'{path.name}: factors do not identify the config')
        expected_override = OFFICIAL_OVERRIDES[algorithm]
        overrides = data.get('agent_overrides', {})
        for key, value in expected_override.items():
            if not _same(overrides.get(key), value):
                errors.append(
                    f'{path.name}: override {key} expected {value!r}, '
                    f'observed {overrides.get(key)!r}'
                )
        if algorithm == 'gcbc' and any(key in overrides for key in OFFICIAL_OVERRIDES['gcivl']):
            errors.append(f'{path.name}: GCBC must not receive an alpha override')
    combinations = {(c.data.get('algorithm'), c.data.get('environment')) for c in configurations}
    expected_combinations = {(algorithm, environment) for algorithm in ALGORITHMS for environment in ENVIRONMENTS}
    if combinations != expected_combinations:
        errors.append('algorithm/environment factorial is not exactly 6 x 4')
    return study, configurations


def _check_resolved_configs(
    configurations: list[Any],
    baseline: Mapping[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    resolved_records = []
    upstream_defaults = baseline.get('upstream_agent_defaults', {})
    for configuration in configurations:
        data = configuration.data
        algorithm = data.get('algorithm')
        try:
            resolved = jsonable(_make_config(_resolved_args(algorithm), configuration))
        except (Exception,) as error:  # pragma: no cover - message is the useful result
            errors.append(f'{configuration.config_id}: actual config resolution failed: {error}')
            continue
        resolved_records.append(resolved)
        if resolved.get('agent_name') != algorithm:
            errors.append(f'{configuration.config_id}: resolved agent_name mismatch')
        expected = copy.deepcopy(upstream_defaults.get(algorithm, {}))
        expected.update(OFFICIAL_OVERRIDES[algorithm])
        if not expected:
            errors.append(f'{configuration.config_id}: missing audited defaults for {algorithm}')
        for key, expected_value in expected.items():
            if key not in resolved:
                errors.append(f'{configuration.config_id}: resolved config lacks {key}')
            elif not _same(resolved[key], expected_value):
                errors.append(
                    f'{configuration.config_id}: resolved {key} mismatch: '
                    f'expected={expected_value!r}, observed={resolved[key]!r}'
                )
        compute = resolved.get('compute', {})
        for slot_name in SLOT_NAMES.get(algorithm, ()):
            slot = compute.get(slot_name, {})
            if slot.get('enabled') is not False:
                errors.append(f'{configuration.config_id}: compute.{slot_name} is enabled')
            try:
                if resolve_slot_spec(resolved, slot_name) is not None:
                    errors.append(f'{configuration.config_id}: compute.{slot_name} resolves to an active spec')
            except Exception as error:
                errors.append(f'{configuration.config_id}: compute.{slot_name} resolution failed: {error}')
        metadata = data.get('resolved_defaults', {})
        if metadata.get('agent_name') != algorithm:
            errors.append(f'{configuration.config_id}: resolved_defaults snapshot agent mismatch')
        for key, expected_value in OFFICIAL_OVERRIDES[algorithm].items():
            if not _same(metadata.get(key), expected_value):
                errors.append(f'{configuration.config_id}: resolved_defaults snapshot {key} mismatch')
    return resolved_records


def _check_datasets(
    dataset_audit: Mapping[str, Any],
    dataset_root: Path,
    errors: list[str],
) -> dict[str, dict[str, str]]:
    observed = {}
    environments = dataset_audit.get('environments', {})
    for environment in ENVIRONMENTS:
        record = environments.get(environment, {})
        observed[environment] = {}
        for split, suffix in (('train', '.npz'), ('validation', '-val.npz')):
            expected = record.get(split, {})
            path = dataset_root / f'{environment}{suffix}'
            if not path.is_file():
                errors.append(f'missing dataset file: {path}')
                continue
            expected_size = expected.get('file_size_bytes')
            if expected_size is not None and path.stat().st_size != expected_size:
                errors.append(
                    f'dataset size mismatch {path}: expected={expected_size}, '
                    f'observed={path.stat().st_size}'
                )
            expected_sha = expected.get('sha256')
            if not expected_sha:
                errors.append(f'missing stored dataset SHA for {path}')
                continue
            actual_sha = _sha256(path)
            observed[environment][split] = actual_sha
            if actual_sha != expected_sha:
                errors.append(
                    f'dataset SHA mismatch {path}: expected={expected_sha}, '
                    f'observed={actual_sha}'
                )
    return observed


def _check_run_design(study: Any, configurations: list[Any], run_root: Path, errors: list[str]) -> list[dict[str, Any]]:
    identities = []
    identity_keys = set()
    expected_combinations = {
        (algorithm, environment, seed)
        for algorithm in ALGORITHMS
        for environment in ENVIRONMENTS
        for seed in SEEDS
    }
    for configuration in configurations:
        algorithm = configuration.data.get('algorithm')
        environment = configuration.data.get('environment')
        for seed in study.data.get('seeds', ()):
            key = (study.study_id, configuration.config_id, algorithm, environment, int(seed))
            if key in identity_keys:
                errors.append(f'duplicate run identity: {key!r}')
            identity_keys.add(key)
            run_dir = make_run_path(
                run_root,
                study.study_id,
                configuration.config_id,
                configuration.slug,
                environment,
                int(seed),
            )
            identities.append({
                'study_id': study.study_id,
                'config_id': configuration.config_id,
                'algorithm': algorithm,
                'environment': environment,
                'seed': int(seed),
                'run_dir': str(run_dir),
            })
    observed_combinations = {(row['algorithm'], row['environment'], row['seed']) for row in identities}
    if observed_combinations != expected_combinations:
        errors.append(
            'expanded factorial mismatch: '
            f'missing={sorted(expected_combinations - observed_combinations)}, '
            f'extra={sorted(observed_combinations - expected_combinations)}'
        )
    if len(identities) != 72 or len(identity_keys) != 72:
        errors.append(f'formal run expansion is {len(identities)} unique runs, expected 72')
    formal_root = run_root / STUDY_ID
    if formal_root.exists() and any(formal_root.iterdir()):
        errors.append(
            f'formal M22 namespace already contains artifacts; refusing to inspect/delete: {formal_root}'
        )
    return identities


def validate_m22(
    *,
    study_path: Path = DEFAULT_STUDY,
    semantic_audit_path: Path = DEFAULT_SEMANTIC_AUDIT,
    baseline_audit_path: Path = DEFAULT_BASELINE_AUDIT,
    dataset_audit_path: Path = DEFAULT_DATASET_AUDIT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    run_root: Path = DEFAULT_RUN_PARENT,
) -> dict[str, Any]:
    """Return a structured M22 preflight report without mutating the repo."""

    errors: list[str] = []
    semantic, baseline, dataset_audit = _check_audits(
        semantic_audit_path, baseline_audit_path, dataset_audit_path, errors
    )
    study, configurations = _check_study_and_configs(study_path, baseline, errors)
    resolved_records = []
    identities = []
    dataset_records = {}
    if study is not None:
        resolved_records = _check_resolved_configs(configurations, baseline, errors)
        dataset_records = _check_datasets(dataset_audit, dataset_root, errors)
        identities = _check_run_design(study, configurations, run_root, errors)
    return {
        'status': 'pass' if not errors else 'fail',
        'errors': errors,
        'semantic_gate': _semantic_gate_status(semantic),
        'study_id': STUDY_ID,
        'config_count': len(configurations),
        'seed_count': len(SEEDS),
        'formal_run_count': len(identities),
        'algorithms': list(ALGORITHMS),
        'environments': list(ENVIRONMENTS),
        'seeds': list(SEEDS),
        'resolved_config_count': len(resolved_records),
        'dataset_records': dataset_records,
        'gpu_policy': {'physical_gpu': '1 only', 'jobs_per_gpu': 2},
        'formal_training_started': False,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--semantic-audit', type=Path, default=DEFAULT_SEMANTIC_AUDIT)
    parser.add_argument('--baseline-audit', type=Path, default=DEFAULT_BASELINE_AUDIT)
    parser.add_argument('--dataset-audit', type=Path, default=DEFAULT_DATASET_AUDIT)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_PARENT)
    parser.add_argument('--gpus', default='1', help='Expected launcher GPU policy; doctor accepts only physical GPU 1.')
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    report = validate_m22(
        study_path=args.study,
        semantic_audit_path=args.semantic_audit,
        baseline_audit_path=args.baseline_audit,
        dataset_audit_path=args.dataset_audit,
        dataset_root=args.dataset_root,
        run_root=args.run_root,
    )
    if args.gpus != '1' or args.jobs_per_gpu != 2:
        report['errors'].append(
            f'GPU policy mismatch: expected physical GPU 1 with jobs_per_gpu=2, '
            f'observed gpus={args.gpus!r}, jobs_per_gpu={args.jobs_per_gpu!r}'
        )
        report['status'] = 'fail'
    if report['status'] == 'pass':
        print('M22 PREFLIGHT: PASS')
    else:
        print('M22 PREFLIGHT: FAIL')
        for error in report['errors']:
            print(f'  - {error}')
    print(f'Study configs: {report["config_count"]}')
    print(f'Seeds: {report["seed_count"]}')
    print(f'Formal runs: {report["formal_run_count"]}')
    print('\nAlgorithms:\n6/6' if set(report['algorithms']) == set(ALGORITHMS) else '\nAlgorithms:\nFAIL')
    print('Environments:\n4/4' if set(report['environments']) == set(ENVIRONMENTS) else 'Environments:\nFAIL')
    print(f'Semantic gate:\n{"PASS" if report["semantic_gate"] == "pass" else "FAIL"}')
    print(f'Official hyperparameters:\n{"PASS" if report["status"] == "pass" else "FAIL"}')
    print(f'Dataset provenance:\n{"PASS" if not any("dataset" in e for e in report["errors"]) else "FAIL"}')
    print(f'Canonical computation disabled:\n{"PASS" if report["status"] == "pass" else "FAIL"}')
    print('GPU policy:\nphysical GPU1 only\njobs_per_gpu=2')
    print('Formal training:\nNOT STARTED')
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
