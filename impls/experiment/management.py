"""File-based Study -> Configuration -> Run management.

This module deliberately keeps experiment state in ordinary YAML, JSON and
CSV files.  It does not know how an agent trains and does not add scientific
semantics to the computation layer.
"""

import csv
import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import numpy as np
import yaml

from ..utils.checkpointing import (
    checkpoint_module_fingerprint,
    resolve_checkpoint,
)


_STUDY_FIELDS = (
    'study_id',
    'name',
    'question',
    'primary_factors',
    'fixed_design',
    'deferred_factors',
    'algorithms',
    'placements',
    'environments',
    'seeds',
    'primary_metric',
)
_SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_RESERVED_SLUG_PARTS = {'final', 'new', 'best', 'v2', 'try2'}
_MANIFEST_FIELDS = (
    'study_id',
    'config_id',
    'slug',
    'algorithm',
    'placement',
    'topology',
    'block',
    'iterations',
    'residual',
    'schedule',
    'credit',
    'state_dim',
    'h_cycles',
    'l_cycles',
    'h_update_executions',
    'l_update_executions',
    'total_update_executions',
    'trainable_params',
    'core_trainable_params',
    'buffer_elements',
    'environment',
    'seed',
    'git_commit',
    'run_attempt',
    'status',
    'run_dir',
    'final_success',
    'best_success',
    'best_step',
)


class ExperimentError(ValueError):
    """Raised when a study, configuration or run identity is invalid."""


@dataclass(frozen=True)
class Study:
    path: Path
    data: dict[str, Any]

    @property
    def study_id(self):
        return self.data['study_id']


@dataclass(frozen=True)
class Configuration:
    path: Path
    data: dict[str, Any]

    @property
    def study_id(self):
        return self.data['study_id']

    @property
    def config_id(self):
        return self.data['config_id']

    @property
    def slug(self):
        return self.data['slug']


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    metadata: dict[str, Any]
    study: Study | None = None
    configuration: Configuration | None = None


def jsonable(value):
    """Convert ConfigDict/JAX/numpy/path values to deterministic JSON values."""

    if isinstance(value, Mapping) or hasattr(value, 'items'):
        return {
            str(key): jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_fingerprint(value):
    """Return a stable SHA-256 fingerprint for a resolved configuration.

    The fingerprint intentionally excludes filesystem ordering and Python
    mapping insertion order.  It is a provenance guard for a Run identity;
    it is not a replacement for the Git commit, dataset identity, seed, or
    training protocol recorded alongside it.
    """

    payload = json.dumps(
        jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _training_protocol(resolved_config):
    """Extract launcher-level protocol fields for runtime provenance."""

    resolved_config = resolved_config or {}
    launcher = resolved_config.get('launcher', {})
    agent_config = resolved_config.get('agent', {})
    if not isinstance(launcher, Mapping):
        launcher = {}
    if not isinstance(agent_config, Mapping):
        agent_config = {}
    protocol = {
        key: launcher.get(key)
        for key in (
            'train_steps',
            'batch_size',
            'eval_interval',
            'eval_tasks',
            'eval_episodes',
            'save_interval',
            'eval_temperature',
            'eval_gaussian',
            'video_episodes',
            'save_best_checkpoint',
            'save_last_checkpoint',
        )
        if key in launcher
    }
    if protocol.get('batch_size') is None and 'batch_size' in agent_config:
        protocol['batch_size'] = agent_config['batch_size']
    return jsonable(protocol)


def _read_yaml(path):
    path = Path(path).resolve()
    with path.open() as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ExperimentError(f'Expected a YAML mapping in {path}')
    return path, data


def _validate_component(value, label):
    if not isinstance(value, str) or not value or not _SAFE_COMPONENT.fullmatch(value):
        raise ExperimentError(f'{label} must be a simple stable path component: {value!r}')


def _validate_list(data, key):
    if not isinstance(data.get(key), list) or not data[key]:
        raise ExperimentError(f'Study field {key!r} must be a non-empty list')


def load_study(path):
    path, data = _read_yaml(path)
    missing = [key for key in _STUDY_FIELDS if key not in data]
    if missing:
        raise ExperimentError(f'Study {path} is missing fields: {missing}')
    _validate_component(data['study_id'], 'study_id')
    if not isinstance(data['fixed_design'], dict):
        raise ExperimentError('fixed_design must be a mapping')
    for key in ('primary_factors', 'deferred_factors', 'algorithms', 'placements', 'environments', 'seeds'):
        _validate_list(data, key)
    if not isinstance(data['primary_metric'], str) or not data['primary_metric']:
        raise ExperimentError('primary_metric must be a non-empty string')
    return Study(path=path, data=data)


def load_configuration(study, config_ref):
    """Load a configuration by path or by file stem under ``study/configs``."""

    if not isinstance(study, Study):
        study = load_study(study)
    config_path = Path(config_ref)
    if not config_path.exists():
        config_path = study.path.parent / 'configs' / config_path
    if config_path.suffix == '':
        config_path = config_path.with_suffix('.yaml')
    config_path, data = _read_yaml(config_path)
    for key in ('study_id', 'config_id', 'slug', 'factors'):
        if key not in data:
            raise ExperimentError(f'Configuration {config_path} is missing field {key!r}')
    if data['study_id'] != study.study_id:
        raise ExperimentError(
            f'Configuration study_id {data["study_id"]!r} does not match {study.study_id!r}'
        )
    _validate_component(data['config_id'], 'config_id')
    _validate_component(data['slug'], 'slug')
    if any(part.lower() in _RESERVED_SLUG_PARTS for part in re.split(r'[-_.]+', data['slug'])):
        raise ExperimentError(f'Configuration slug uses a reserved mutable name: {data["slug"]!r}')
    if 'seed' in data:
        raise ExperimentError('Configuration must not contain seed; seed belongs to a Run')
    if not isinstance(data['factors'], dict):
        raise ExperimentError('Configuration factors must be a mapping')
    if 'environment' in data:
        _validate_component(data['environment'], 'environment')
        if data['environment'] not in study.data['environments']:
            raise ExperimentError(
                f'Configuration environment {data["environment"]!r} is not declared by '
                f'Study {study.study_id}: {study.data["environments"]!r}'
            )
    return Configuration(path=config_path, data=data)


def prepare_run_design(study_path, config_ref):
    study = load_study(study_path)
    configuration = load_configuration(study, config_ref)
    return study, configuration


def resolve_run_dependency(
    study,
    configuration,
    dependency_name,
    *,
    seed,
    run_root,
):
    """Resolve a declarative source Run dependency without study-specific logic."""

    if not isinstance(study, Study):
        study = load_study(study)
    if not isinstance(configuration, Configuration):
        configuration = load_configuration(study, configuration)
    dependencies = configuration.data.get('dependencies', {})
    dependency = dependencies.get(dependency_name) if isinstance(dependencies, Mapping) else None
    if not isinstance(dependency, Mapping):
        raise ExperimentError(
            f'{configuration.config_id} has no dependency {dependency_name!r}'
        )
    source_config_id = dependency.get('source_config_id')
    if not isinstance(source_config_id, str) or not source_config_id:
        raise ExperimentError(
            f'{configuration.config_id}: dependency {dependency_name!r} requires source_config_id'
        )
    source_configuration = load_configuration(study, source_config_id)
    seed_policy = dependency.get('seed_policy', 'same_seed')
    if seed_policy == 'same_seed':
        source_seed = int(seed)
    elif seed_policy == 'explicit_seed':
        if 'source_seed' not in dependency:
            raise ExperimentError(
                f'{configuration.config_id}: explicit_seed dependency requires source_seed'
            )
        source_seed = int(dependency['source_seed'])
    else:
        raise ExperimentError(f'Unsupported dependency seed_policy: {seed_policy!r}')
    source_environment = source_configuration.data.get(
        'environment', configuration.data.get('environment')
    )
    source_attempt = int(dependency.get('source_run_attempt', 0))
    source_run_dir = make_run_path(
        run_root,
        study.study_id,
        source_configuration.config_id,
        source_configuration.slug,
        source_environment,
        source_seed,
        run_attempt=source_attempt,
    )
    return {
        'dependency_name': dependency_name,
        'module': dependency.get('module'),
        'source_config_id': source_configuration.config_id,
        'source_slug': source_configuration.slug,
        'source_environment': source_environment,
        'source_seed': source_seed,
        'source_run_attempt': source_attempt,
        'source_run_dir': str(source_run_dir.resolve()),
        'checkpoint_role': dependency.get('checkpoint_role'),
        'checkpoint_step': dependency.get('checkpoint_step'),
        'seed_policy': seed_policy,
    }


def validate_source_run_dependency(
    study,
    configuration,
    dependency_name,
    *,
    seed,
    run_root,
    resolved_agent=None,
):
    """Validate a source Run and its semantic checkpoint before Run creation."""

    dependency = resolve_run_dependency(
        study,
        configuration,
        dependency_name,
        seed=seed,
        run_root=run_root,
    )
    if dependency['checkpoint_role'] != 'last':
        raise ExperimentError(
            f'{dependency_name}: only checkpoint_role=last is permitted, '
            f'got {dependency["checkpoint_role"]!r}'
        )
    try:
        checkpoint_step = int(dependency['checkpoint_step'])
    except (TypeError, ValueError) as error:
        raise ExperimentError(
            f'{dependency_name}: checkpoint_step must be 1000000, '
            f'got {dependency.get("checkpoint_step")!r}'
        ) from error
    if checkpoint_step != 1_000_000:
        raise ExperimentError(
            f'{dependency_name}: checkpoint_step must be 1000000, '
            f'got {dependency["checkpoint_step"]!r}'
        )
    target_environment = configuration.data.get('environment')
    if target_environment != dependency['source_environment']:
        raise ExperimentError(
            f'{dependency_name}: source environment {dependency["source_environment"]!r} '
            f'does not match target {target_environment!r}'
        )
    source_configuration = load_configuration(study, dependency['source_config_id'])
    target_algorithm = configuration.data.get('algorithm')
    source_algorithm = source_configuration.data.get('algorithm')
    if target_algorithm != source_algorithm:
        raise ExperimentError(
            f'{dependency_name}: source algorithm {source_algorithm!r} '
            f'does not match target {target_algorithm!r}'
        )
    source_run_dir = Path(dependency['source_run_dir'])
    metadata_path = source_run_dir / 'runtime_metadata.json'
    if not metadata_path.is_file():
        raise ExperimentError(f'Missing source runtime metadata: {metadata_path}')
    with metadata_path.open() as file:
        source_metadata = json.load(file)
    if source_metadata.get('status') != 'completed':
        raise ExperimentError(
            f'Source run is not completed: {source_metadata.get("status")!r}'
        )
    expected_metadata = {
        'config_id': dependency['source_config_id'],
        'environment': dependency['source_environment'],
        'algorithm': source_algorithm,
        'seed': dependency['source_seed'],
        'run_attempt': dependency['source_run_attempt'],
    }
    for key, expected in expected_metadata.items():
        observed = source_metadata.get(key)
        if observed != expected:
            raise ExperimentError(
                f'Source metadata mismatch for {key}: expected={expected!r}, observed={observed!r}'
            )
    checkpoint = resolve_checkpoint(source_run_dir, 'last')
    if checkpoint['checkpoint_role'] != 'last' or checkpoint['checkpoint_step'] != 1_000_000:
        raise ExperimentError(
            'Source checkpoint is not the required last@1M checkpoint: '
            f'{checkpoint["checkpoint_role"]}@{checkpoint["checkpoint_step"]}'
        )
    module_name = dependency.get('module')
    if not isinstance(module_name, str) or not module_name:
        raise ExperimentError(f'{dependency_name}: module must be a non-empty string')
    resolved_path = source_run_dir / 'resolved_config.json'
    if not resolved_path.is_file():
        raise ExperimentError(f'Missing source resolved config: {resolved_path}')
    with resolved_path.open() as file:
        source_resolved = json.load(file)
    source_agent = source_resolved.get('algorithm_config', {}).get('agent', {})
    target_agent = jsonable(resolved_agent or {})
    if resolved_agent is not None:
        if source_agent.get('agent_name') != target_agent.get('agent_name'):
            raise ExperimentError('Source and target base agent identities are incompatible')
        ignored = {'compute', 'training_mode', 'runtime_variant', 'dependencies'}
        for key in set(source_agent) | set(target_agent):
            if key in ignored:
                continue
            if source_agent.get(key) != target_agent.get(key):
                raise ExperimentError(f'Source/target agent mismatch for {key!r}')
    source_compute = source_agent.get('compute', {})
    target_compute = target_agent.get('compute', {})
    module_slots = {
        key for key in set(source_compute) | set(target_compute)
        if key == module_name or key.startswith(f'{module_name}_')
    }
    for slot_name in module_slots:
        if resolved_agent is not None and source_compute.get(slot_name) != target_compute.get(slot_name):
            raise ExperimentError(f'Source/target computation mismatch for {slot_name!r}')
        source_slot = source_compute.get(slot_name, {})
        if source_slot.get('enabled', False) or source_slot.get('topology') != 'feedforward':
            raise ExperimentError(
                f'Source module {module_name!r} is not feedforward: {slot_name}'
            )
    module_fingerprint = checkpoint_module_fingerprint(
        checkpoint['checkpoint_path'], module_name,
    )
    return {
        **dependency,
        'checkpoint_path': checkpoint['checkpoint_path'],
        'checkpoint_sha256': checkpoint['checkpoint_sha256'],
        'checkpoint_metadata': checkpoint['checkpoint_metadata'],
        'module': module_name,
        'module_fingerprint': module_fingerprint,
        'source_status': source_metadata.get('status'),
    }


def make_run_path(run_root, study_id, config_id, slug, environment, seed, run_attempt=0):
    """Return a stable path for one Run without creating it.

    Attempt zero is the historical canonical path.  A positive attempt is an
    explicit, generic rerun instance and never reuses the canonical path.
    """

    for value, label in (
        (study_id, 'study_id'),
        (config_id, 'config_id'),
        (slug, 'slug'),
        (environment, 'environment'),
    ):
        _validate_component(value, label)
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ExperimentError(f'seed must be a non-negative integer: {seed!r}')
    if not isinstance(run_attempt, (int, np.integer)) or int(run_attempt) < 0:
        raise ExperimentError(
            f'run_attempt must be a non-negative integer: {run_attempt!r}'
        )
    seed_component = f'seed_{int(seed):03d}'
    if int(run_attempt):
        seed_component += f'__attempt_{int(run_attempt):03d}'
    return (
        Path(run_root)
        / study_id
        / f'{config_id}__{slug}'
        / environment
        / seed_component
    )


def git_metadata(repo_root=None):
    """Return stable commit/dirty metadata, including safe missing-git fallbacks."""

    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=all'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {'git_commit': commit, 'git_dirty': bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': None, 'git_dirty': None}


def _jax_metadata():
    try:
        import jax

        return {
            'jax_backend': jax.default_backend(),
            'jax_device_descriptions': [str(device) for device in jax.devices()],
        }
    except Exception as error:  # pragma: no cover - only for incomplete environments.
        return {'jax_backend': None, 'jax_device_descriptions': [], 'jax_error': str(error)}


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _legacy_run_path(root, seed):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(root) / f'sd{int(seed):03d}_{timestamp}'


def _metadata_for_run(
    *,
    run_dir,
    study,
    configuration,
    algorithm,
    environment,
    seed,
    dataset_dir,
    computation,
    compute_slots,
    repo_root,
    ogbench_module=None,
    run_attempt=0,
):
    git_info = git_metadata(repo_root)
    metadata = {
        # M8 compatibility fields.
        'agent': algorithm,
        'environment': environment,
        'dataset_dir': os.path.abspath(dataset_dir) if dataset_dir is not None else None,
        'ogbench_module': ogbench_module,
        'seed': int(seed),
        'run_attempt': int(run_attempt),
        'computation': bool(computation),
        'compute_slots': jsonable(compute_slots or {}),
        # M9 experiment identity/provenance.
        'study_id': study.study_id if study is not None else None,
        'config_id': configuration.config_id if configuration is not None else None,
        'config_slug': configuration.slug if configuration is not None else None,
        'algorithm': algorithm,
        'git_commit': git_info['git_commit'],
        'git_dirty': git_info['git_dirty'],
        'start_time': _utc_now(),
        'hostname': socket.gethostname(),
        'dataset_identity': {
            'name': environment,
            'path': os.path.abspath(dataset_dir) if dataset_dir is not None else None,
        },
        'run_dir': str(Path(run_dir).resolve()),
        'status': 'running',
    }
    metadata.update(_jax_metadata())
    return metadata


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(jsonable(data), file, indent=2, sort_keys=True)
        file.write('\n')


def create_run_context(
    *,
    study=None,
    configuration=None,
    run_root='runs',
    legacy_root=None,
    algorithm,
    environment,
    seed,
    dataset_dir=None,
    computation=False,
    compute_slots=None,
    resolved_config=None,
    repo_root=None,
    ogbench_module=None,
    runtime_extras=None,
    run_attempt=0,
):
    """Create a run directory and its initial metadata, failing if it exists."""

    if study is not None and not isinstance(study, Study):
        study = load_study(study)
    if configuration is not None and not isinstance(configuration, Configuration):
        configuration = load_configuration(study, configuration)
    if (study is None) != (configuration is None):
        raise ExperimentError('study and configuration must be supplied together')
    if study is None:
        run_dir = _legacy_run_path(legacy_root or run_root, seed)
    else:
        run_dir = make_run_path(
            run_root,
            study.study_id,
            configuration.config_id,
            configuration.slug,
            environment,
            seed,
            run_attempt=run_attempt,
        )
    run_dir = run_dir.resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    (run_dir / 'checkpoints').mkdir()

    metadata = _metadata_for_run(
        run_dir=run_dir,
        study=study,
        configuration=configuration,
        algorithm=algorithm,
        environment=environment,
        seed=seed,
        dataset_dir=dataset_dir,
        computation=computation,
        compute_slots=compute_slots,
        repo_root=repo_root,
        ogbench_module=ogbench_module,
        run_attempt=run_attempt,
    )
    if runtime_extras:
        metadata.update(jsonable(runtime_extras))
    resolved_payload = {
        'study': None if study is None else study.data,
        'configuration': None if configuration is None else configuration.data,
        'algorithm_config': resolved_config or {},
    }
    resolved_fingerprint = config_fingerprint(resolved_payload)
    metadata.update({
        'resolved_config_fingerprint': resolved_fingerprint,
        'training_protocol': _training_protocol(resolved_config),
    })
    _write_json(run_dir / 'runtime_metadata.json', metadata)
    _write_json(
        run_dir / 'resolved_config.json',
        resolved_payload | {'resolved_config_fingerprint': resolved_fingerprint},
    )
    return RunContext(run_dir=run_dir, metadata=metadata, study=study, configuration=configuration)


def update_runtime_metadata(run_dir, updates):
    """Merge runtime-only fields after agent initialization."""

    path = Path(run_dir) / 'runtime_metadata.json'
    with path.open() as file:
        metadata = json.load(file)
    metadata.update(jsonable(updates))
    _write_json(path, metadata)
    return metadata


def _float_or_none(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_eval_csv(eval_path, status='completed'):
    """Summarize explicit success columns without guessing missing metrics."""

    summary = {
        'status': status,
        'final_success': None,
        'best_success': None,
        'best_step': None,
        'success_column': None,
    }
    eval_path = Path(eval_path)
    if not eval_path.exists():
        return summary
    with eval_path.open(newline='') as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        candidates = ('evaluation/overall_success', 'overall_success', 'success')
        success_column = next((name for name in candidates if name in fields), None)
        if success_column is None:
            return summary
        values = []
        for row in reader:
            value = _float_or_none(row.get(success_column))
            step = _float_or_none(row.get('step'))
            if value is not None:
                values.append((value, int(step) if step is not None else None))
    if not values:
        return summary | {'success_column': success_column}
    best_value, best_step = max(values, key=lambda item: item[0])
    return {
        'status': status,
        'final_success': values[-1][0],
        'best_success': best_value,
        'best_step': best_step,
        'success_column': success_column,
    }


def finalize_run(run_dir, status, failure_reason=None):
    """Close the run lifecycle while retaining partial artifacts on failure."""

    if status not in {'completed', 'failed', 'aborted', 'invalid'}:
        raise ExperimentError(f'Invalid terminal run status: {status!r}')
    run_dir = Path(run_dir)
    metadata_path = run_dir / 'runtime_metadata.json'
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open() as file:
            metadata = json.load(file)
    metadata['status'] = status
    metadata['end_time'] = _utc_now()
    if failure_reason is not None:
        metadata['failure_reason'] = str(failure_reason)
        _write_json(
            run_dir / 'failure.json',
            {'status': status, 'failure_reason': str(failure_reason), 'time': metadata['end_time']},
        )
    _write_json(metadata_path, metadata)
    summary = summarize_eval_csv(run_dir / 'eval.csv', status=status)
    checkpoint_index_path = run_dir / 'checkpoints' / 'index.json'
    if checkpoint_index_path.exists():
        with checkpoint_index_path.open() as file:
            checkpoint_index = json.load(file)
        best = checkpoint_index.get('best') or {}
        if best:
            summary['best_success'] = _float_or_none(best.get('metric'))
            summary['best_step'] = int(best['step'])
    _write_json(run_dir / 'summary.json', summary)
    return summary


def _factor_value(configuration, key):
    data = configuration.data if configuration is not None else {}
    factors = data.get('factors', {})
    if key in data:
        return data[key]
    if key in factors:
        return factors[key]
    aliases = {
        'iterations': ('internal_iterations', 'internal_iterations_K', 'K'),
    }
    for alias in aliases.get(key, ()):
        if alias in factors:
            return factors[alias]
    return None


def _display_factor(configuration, key):
    value = _factor_value(configuration, key)
    return '' if value is None else value


def _manifest_row(study, configuration, *, environment='', seed='', metadata=None, summary=None, run_dir=''):
    metadata = metadata or {}
    summary = summary or {}
    accounting = metadata.get('actor_parameter_accounting', {})
    accounting_rows = [value for value in accounting.values() if isinstance(value, Mapping)]

    def account_value(key, *, aggregate=False):
        values = [value.get(key) for value in accounting_rows if value.get(key) is not None]
        if not values:
            return ''
        if aggregate and all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            return sum(values)
        return values[0]

    override_compute = configuration.data.get('agent_overrides', {}).get('compute', {}) if configuration else {}
    enabled_specs = [
        value for value in override_compute.values()
        if isinstance(value, Mapping) and value.get('enabled', False)
    ]
    planned_kwargs = enabled_specs[0].get('topology_kwargs', {}) if enabled_specs else {}
    planned_h_cycles = _display_factor(configuration, 'h_cycles') or planned_kwargs.get('h_cycles', '')
    planned_l_cycles = _display_factor(configuration, 'l_cycles') or planned_kwargs.get('l_cycles', '')
    planned_state_dim = planned_kwargs.get('state_dim', '')
    planned_h_executions = (
        int(planned_h_cycles) if planned_h_cycles not in ('', None) else ''
    )
    planned_l_executions = (
        planned_h_executions * int(planned_l_cycles)
        if planned_h_executions != '' and planned_l_cycles not in ('', None)
        else ''
    )

    return {
        'study_id': metadata.get('study_id', study.study_id),
        'config_id': metadata.get('config_id', configuration.config_id if configuration else ''),
        'slug': metadata.get('config_slug', configuration.slug if configuration else ''),
        'algorithm': metadata.get('algorithm', _display_factor(configuration, 'algorithm')),
        'placement': metadata.get('placement', _display_factor(configuration, 'placement')),
        'topology': metadata.get('topology', _display_factor(configuration, 'topology')),
        'block': metadata.get('block', _display_factor(configuration, 'block')),
        'iterations': metadata.get('iterations', _display_factor(configuration, 'iterations')),
        'residual': metadata.get('residual', _display_factor(configuration, 'residual')),
        'schedule': metadata.get('schedule', _display_factor(configuration, 'schedule')),
        'credit': metadata.get('credit', account_value('credit') or _display_factor(configuration, 'credit')),
        'state_dim': metadata.get('state_dim', account_value('state_dim') or planned_state_dim),
        'h_cycles': metadata.get('h_cycles', _display_factor(configuration, 'h_cycles') or account_value('h_cycles') or planned_h_cycles),
        'l_cycles': metadata.get('l_cycles', _display_factor(configuration, 'l_cycles') or account_value('l_cycles') or planned_l_cycles),
        'h_update_executions': metadata.get('h_update_executions', account_value('h_update_executions', aggregate=True) or planned_h_executions),
        'l_update_executions': metadata.get('l_update_executions', account_value('l_update_executions', aggregate=True) or planned_l_executions),
        'total_update_executions': metadata.get('total_update_executions', account_value('total_update_executions', aggregate=True) or (planned_h_executions + planned_l_executions if planned_h_executions != '' and planned_l_executions != '' else '')),
        'trainable_params': metadata.get('trainable_params', account_value('trainable_params', aggregate=True)),
        'core_trainable_params': metadata.get('core_trainable_params', account_value('core_trainable_params', aggregate=True)),
        'buffer_elements': metadata.get('buffer_elements', account_value('buffer_elements', aggregate=True)),
        'environment': metadata.get('environment', environment),
        'seed': metadata.get('seed', seed),
        'run_attempt': metadata.get('run_attempt', 0),
        'git_commit': metadata.get('git_commit', ''),
        'status': metadata.get('status', 'planned'),
        'run_dir': run_dir,
        'final_success': summary.get('final_success'),
        'best_success': summary.get('best_success'),
        'best_step': summary.get('best_step'),
    }


def _load_run_metadata(run_root, study_id):
    root = Path(run_root) / study_id
    if not root.exists():
        return []
    result = []
    for metadata_path in sorted(root.rglob('runtime_metadata.json')):
        with metadata_path.open() as file:
            metadata = json.load(file)
        result.append((metadata_path.parent, metadata))
    return result


def write_manifest(study_path, run_root='runs', output_path=None, repo_root=None):
    """Write planned and observed runs for one Study to ``manifest.csv``."""

    study = load_study(study_path)
    config_dir = study.path.parent / 'configs'
    configurations = [
        load_configuration(study, path)
        for path in sorted(config_dir.glob('*.yaml'))
    ]
    if not configurations:
        raise ExperimentError(f'No configuration YAML files found in {config_dir}')
    # The repository root is two levels above experiments/<study>/study.yaml.
    # Callers may override it for a separately staged fixture.
    repo_root = Path(repo_root or study.path.parents[2]).resolve()
    rows = {}
    for configuration in configurations:
        environments = (
            [configuration.data['environment']]
            if 'environment' in configuration.data
            else study.data['environments']
        )
        for environment in environments:
            for seed in study.data['seeds']:
                run_path = make_run_path(
                    run_root,
                    study.study_id,
                    configuration.config_id,
                    configuration.slug,
                    environment,
                    seed,
                )
                relative_run = os.path.relpath(run_path, repo_root)
                row = _manifest_row(
                    study,
                    configuration,
                    environment=environment,
                    seed=seed,
                    run_dir=relative_run,
                )
                rows[(configuration.config_id, environment, int(seed))] = row

    for run_path, metadata in _load_run_metadata(run_root, study.study_id):
        config_id = metadata.get('config_id')
        configuration = next(
            (item for item in configurations if item.config_id == config_id), None
        )
        if configuration is None:
            continue
        environment = metadata.get('environment', '')
        seed = int(metadata.get('seed', 0))
        summary_path = run_path / 'summary.json'
        summary = {}
        if summary_path.exists():
            with summary_path.open() as file:
                summary = json.load(file)
        key = (configuration.config_id, environment, seed)
        rows[key] = _manifest_row(
            study,
            configuration,
            environment=environment,
            seed=seed,
            metadata=metadata,
            summary=summary,
            run_dir=os.path.relpath(run_path, repo_root),
        )

    output_path = Path(output_path or study.path.parent / 'manifest.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: (item['config_id'], item['environment'], str(item['seed']))):
            writer.writerow(row)
    return output_path


def aggregate_manifest(manifest_path, output_path=None, metric='final_success'):
    """Aggregate numeric metric values by ``config_id + environment``."""

    manifest_path = Path(manifest_path)
    with manifest_path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    groups = {}
    for row in rows:
        # Failed/aborted attempts may contain partial evaluation artifacts;
        # they are provenance, never primary scientific results.  Synthetic
        # manifests without a status remain backward-compatible.
        if row.get('status') and row.get('status') != 'completed':
            continue
        value = _float_or_none(row.get(metric))
        if value is None:
            continue
        key = (row.get('config_id', ''), row.get('environment', ''))
        groups.setdefault(key, {'slug': row.get('slug', ''), 'values': []})['values'].append(value)
    output_path = Path(output_path or manifest_path.parent / 'aggregated.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('config_id', 'slug', 'environment', 'metric', 'count', 'mean', 'std')
    with output_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (config_id, environment), group in sorted(groups.items()):
            values = group['values']
            writer.writerow({
                'config_id': config_id,
                'slug': group['slug'],
                'environment': environment,
                'metric': metric,
                'count': len(values),
                'mean': mean(values),
                'std': pstdev(values) if len(values) > 1 else 0.0,
            })
    return output_path
