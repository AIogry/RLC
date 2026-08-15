"""File-based Study -> Configuration -> Run management.

This module deliberately keeps experiment state in ordinary YAML, JSON and
CSV files.  It does not know how an agent trains and does not add scientific
semantics to the computation layer.
"""

import csv
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
    'environment',
    'seed',
    'git_commit',
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
    return Configuration(path=config_path, data=data)


def prepare_run_design(study_path, config_ref):
    study = load_study(study_path)
    configuration = load_configuration(study, config_ref)
    return study, configuration


def make_run_path(run_root, study_id, config_id, slug, environment, seed):
    """Return the stable canonical path for one Run without creating it."""

    for value, label in (
        (study_id, 'study_id'),
        (config_id, 'config_id'),
        (slug, 'slug'),
        (environment, 'environment'),
    ):
        _validate_component(value, label)
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ExperimentError(f'seed must be a non-negative integer: {seed!r}')
    return (
        Path(run_root)
        / study_id
        / f'{config_id}__{slug}'
        / environment
        / f'seed_{int(seed):03d}'
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
):
    git_info = git_metadata(repo_root)
    metadata = {
        # M8 compatibility fields.
        'agent': algorithm,
        'environment': environment,
        'dataset_dir': os.path.abspath(dataset_dir) if dataset_dir is not None else None,
        'ogbench_module': ogbench_module,
        'seed': int(seed),
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
    )
    _write_json(run_dir / 'runtime_metadata.json', metadata)
    _write_json(
        run_dir / 'resolved_config.json',
        {
            'study': None if study is None else study.data,
            'configuration': None if configuration is None else configuration.data,
            'algorithm_config': resolved_config or {},
        },
    )
    return RunContext(run_dir=run_dir, metadata=metadata, study=study, configuration=configuration)


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
        'environment': metadata.get('environment', environment),
        'seed': metadata.get('seed', seed),
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
        for environment in study.data['environments']:
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
