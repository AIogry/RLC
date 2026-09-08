"""Evaluation-only M20A-D relation-utilization diagnosis.

This tool restores numeric checkpoints from one already-trained M20A source
run and evaluates Correct-versus-Zero and channel-ablation interventions on
one fixed ``GCDataset.sample`` batch.  It never calls an agent update, writes
under the source run, or constructs a second model for the counterfactual.

The output root is an independent diagnostic campaign root.  Existing formal
artifacts are never overwritten: an existing fixed batch is validated by its
fingerprint and an existing checkpoint output is a collision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
from flax.traverse_util import flatten_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.agents.gciql import GCIQLAgent
from impls.experiment import (
    config_fingerprint,
    make_run_path,
    prepare_run_design,
)
from impls.experiment.reevaluation import _resolved_agent_config
from impls.utils.checkpointing import sha256_file
from impls.utils.datasets import GCDataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.flax_utils import restore_agent_from_checkpoint


DIAGNOSTIC_ID = 'M20AD'
DIAGNOSTIC_SCHEMA = 'm20a_relation_utilization_v1'
STUDY_ID = 'M20A'
DEFAULT_SOURCE_CONFIG_ID = 'M20A-CUBE-C-Q'
# Backwards-compatible alias for callers that used the pre-prompt5 default.
SOURCE_CONFIG_ID = DEFAULT_SOURCE_CONFIG_ID
ENVIRONMENT = 'cube-triple-play-v0'
SOURCE_TRAINING_SEED = 0
EXPECTED_FIXED_BATCH_FINGERPRINT = (
    'ccf4c1e32bad61016fd4b0f3a4baeea254bcf506d8a19f0dd0d08a8e3bf19ed9'
)
RELATION_NAMES = ('current_support', 'goal_support', 'goal_conflict')
RELATION_CHANNELS = len(RELATION_NAMES)
DEFAULT_STUDY = 'experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml'
DEFAULT_SOURCE_RUN_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'
DEFAULT_DATASET_ROOT = '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench'
DEFAULT_OUTPUT_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'
DEFAULT_CHECKPOINT_STEPS = (100_000, 200_000, 300_000, 'latest')
DEFAULT_BATCH_SIZE = 4096
DEFAULT_DIAGNOSTIC_SEED = 20_022
ATOL = 1e-7
RTOL = 1e-6
EPSILON = 1e-8
_NUMERIC_CHECKPOINT = re.compile(r'^params_(\d+)\.pkl$')
_CHECKPOINT_DISCOVERY_CODE_PATH = 'tools/m20a_relation_utilization.py:discover_checkpoint_artifacts'
_CHECKPOINT_DISCOVERY_PATTERNS = (
    'checkpoints/params_*.pkl',
    'checkpoints/best/*',
    'checkpoints/last/*',
)
_REQUIRED_BATCH_FIELDS = frozenset({
    'sample_id',
    'transition_indices',
    'actor_goal_indices',
    'value_goal_indices',
    'observations',
    'next_observations',
    'actions',
    'actor_goals',
    'value_goals',
    'rewards',
    'masks',
})


class M20ADiagnosticError(ValueError):
    """Raised when an M20A-D provenance or correctness gate fails."""


def _jsonable(value):
    if isinstance(value, Mapping) or hasattr(value, 'items'):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def _read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise M20ADiagnosticError(f'Cannot read JSON artifact {path}: {error}') from error
    if not isinstance(value, Mapping):
        raise M20ADiagnosticError(f'Expected JSON mapping in {path}')
    return dict(value)


def _write_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite diagnostic artifact: {path}')
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + '\n')


def _array_fingerprint(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(str(name).encode('utf-8'))
        digest.update(str(array.dtype).encode('utf-8'))
        digest.update(repr(tuple(array.shape)).encode('utf-8'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _pytree_fingerprint(tree):
    """Fingerprint arbitrary JAX/Flax pytrees without mutating them."""

    digest = hashlib.sha256()
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, value in path_leaves:
        path_repr = repr(tuple(str(item) for item in path))
        digest.update(path_repr.encode('utf-8'))
        if value is None:
            digest.update(b'None')
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode('utf-8'))
        digest.update(repr(tuple(array.shape)).encode('utf-8'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_info():
    try:
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=all'],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        return {'head': head, 'dirty': bool(status)}
    except (OSError, subprocess.CalledProcessError) as error:
        return {'head': None, 'dirty': None, 'error': str(error)}


def _stable_sha256(path):
    """Hash twice and require unchanged stat/hash continuity."""

    path = Path(path)
    if not path.is_file():
        raise M20ADiagnosticError(f'Missing checkpoint file: {path}')
    stat_before = path.stat()
    first = sha256_file(path)
    stat_mid = path.stat()
    second = sha256_file(path)
    stat_after = path.stat()
    if first != second or stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
        raise M20ADiagnosticError(f'Checkpoint changed while hashing: {path}')
    return second, _file_stat(path)


def _file_stat(path):
    stat = Path(path).stat()
    return {
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'inode': int(stat.st_ino),
    }


def _audit_file_artifact(path):
    """Return stable identity information for a non-checkpoint artifact."""

    path = Path(path).resolve()
    if not path.is_file():
        return {'path': str(path), 'status': 'missing'}
    stat_before = _file_stat(path)
    sha = sha256_file(path)
    stat_after = _file_stat(path)
    if stat_before != stat_after:
        raise M20ADiagnosticError(f'Artifact changed while hashing: {path}')
    return {
        'path': str(path),
        'status': 'ok',
        'sha256': sha,
        'file_stat': stat_after,
    }


def _file_snapshot(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        result[str(path.relative_to(root))] = _file_stat(path)
    return result


def _assert_source_snapshot(provenance, *, label):
    current = _file_snapshot(provenance['source_run_path'])
    if current != provenance['source_file_snapshot']:
        raise M20ADiagnosticError(
            f'Source run changed {label}; refusing to continue: '
            f'expected_files={len(provenance["source_file_snapshot"])}, '
            f'observed_files={len(current)}'
        )
    return current


def _under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _require_equal(actual, expected, label):
    if actual != expected:
        raise M20ADiagnosticError(f'{label}: expected {expected!r}, got {actual!r}')


def _dataset_identity(dataset_root, *, environment=ENVIRONMENT):
    dataset_root = Path(dataset_root).resolve()
    path = dataset_root / f'{environment}.npz'
    if not path.is_file():
        raise M20ADiagnosticError(f'Missing source train dataset: {path}')
    stat = path.stat()
    sha = sha256_file(path)
    return {
        'path': str(path),
        'root': str(dataset_root),
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'sha256': sha,
        'identity_kind': 'full_file_sha256_plus_stat',
    }


def _validate_slot_config(agent_config, slot_name, *, expected_readout):
    compute = agent_config.get('compute', {})
    slot = compute.get(slot_name) if isinstance(compute, Mapping) else None
    if not isinstance(slot, Mapping):
        raise M20ADiagnosticError(f'Source resolved config has no compute.{slot_name} mapping')
    for key, expected in (
        ('enabled', True),
        ('structure', 'cube_tokens'),
        ('relation_mode', 'correct'),
        ('relation_augmenter', 'relation_mlp'),
        ('readout', expected_readout),
        ('topology', 'feedforward'),
        ('block', 'mlp_mixer'),
        ('credit', 'direct'),
    ):
        _require_equal(slot.get(key), expected, f'source compute.{slot_name}.{key}')
    if expected_readout == 'hybrid_context_query':
        _require_equal(
            slot.get('readout_kwargs', {}).get('query_dim'), 128,
            f'source compute.{slot_name}.readout_kwargs.query_dim',
        )
    elif expected_readout == 'mean_context':
        if slot.get('readout_kwargs', {}).get('query_dim') is not None:
            raise M20ADiagnosticError(
                f'source compute.{slot_name}.mean_context must not define query_dim'
            )
    else:
        raise M20ADiagnosticError(f'Unsupported M20A readout: {expected_readout!r}')
    relation_kwargs = slot.get('relation_kwargs', {})
    for key, expected in (
        ('num_relation_types', 3),
        ('threshold_status', 'FROZEN_USER_PHASE2'),
        ('current_support_epsilon_xy', 0.02),
        ('current_support_epsilon_z', 0.01),
        ('goal_support_epsilon_xy', 0.02),
        ('goal_support_epsilon_z', 0.01),
        ('goal_conflict_radius', 0.04),
    ):
        _require_equal(
            relation_kwargs.get(key), expected,
            f'source compute.{slot_name}.relation_kwargs.{key}',
        )
    return dict(slot)


def _campaign_namespace(config_id):
    """Return an output namespace without overwriting the historical Q record."""

    if config_id == DEFAULT_SOURCE_CONFIG_ID:
        return Path(DIAGNOSTIC_ID)
    return Path(DIAGNOSTIC_ID) / 'campaigns' / config_id


def _campaign_id(config_id):
    if config_id == 'M20A-CUBE-C-M':
        return 'M20A-D-M'
    if config_id == DEFAULT_SOURCE_CONFIG_ID:
        return 'M20A-D-Q'
    return f'M20A-D-{config_id}'


def resolve_authoritative_source_run(
    study_path=DEFAULT_STUDY,
    *,
    run_root=DEFAULT_SOURCE_RUN_ROOT,
    config_id=SOURCE_CONFIG_ID,
):
    """Resolve a source run through the Study/config path authority.

    The returned path is deliberately constructed by the same resolver used
    by the training launcher.  This keeps a diagnostic from silently accepting
    a hand-written path with a missing configuration slug.
    """

    study, configuration = prepare_run_design(study_path, config_id)
    environment = configuration.data.get('environment')
    training_seed = configuration.data.get('training_seed')
    if environment is None or training_seed is None:
        raise M20ADiagnosticError(
            f'Configuration {config_id!r} must define environment and training_seed'
        )
    return make_run_path(
        run_root,
        study.study_id,
        configuration.config_id,
        configuration.slug,
        environment,
        int(training_seed),
    ).resolve()


def _source_run_root(source_run_path):
    """Return the canonical ``runs`` root for a resolved source path."""

    source_run_path = Path(source_run_path).resolve()
    # .../runs/<study>/<config>__<slug>/<environment>/seed_000
    return source_run_path.parents[3]


def _observe_source_process(source_run_path, *, source_config_id=SOURCE_CONFIG_ID):
    """Observe source-training processes without modifying process state."""

    source_run_path = str(Path(source_run_path).resolve())
    markers = ('impls/main.py', 'tools/run.py', 'tools/sweep.py', 'train_')
    try:
        result = subprocess.run(
            ['ps', '-eo', 'pid=,args='],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            'state': 'observation_failed',
            'matches': [],
            'error': f'{type(error).__name__}: {error}',
        }

    matches = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2:
            continue
        try:
            pid = int(pieces[0])
        except ValueError:
            continue
        command = pieces[1]
        if pid == os.getpid():
            continue
        source_match = source_run_path in command
        command_match = (
            source_config_id in command
            and any(marker in command for marker in markers)
        )
        if source_match or command_match:
            matches.append({'pid': pid, 'command': command})
    return {
        'state': 'running' if matches else 'stopped',
        'matches': matches,
    }


def _read_last_training_step(source_run_path):
    """Read the actual final training step recorded by ``train.csv``."""

    train_path = Path(source_run_path).resolve() / 'train.csv'
    if not train_path.is_file():
        return {
            'path': str(train_path),
            'status': 'missing',
            'last_step': None,
            'row_count': 0,
        }
    steps = []
    try:
        with train_path.open(newline='') as file:
            for row in csv.DictReader(file):
                raw_step = row.get('step')
                if raw_step in (None, ''):
                    continue
                steps.append(int(raw_step))
    except (OSError, ValueError, csv.Error) as error:
        raise M20ADiagnosticError(
            f'Cannot read artifact-derived training step from {train_path}: '
            f'{type(error).__name__}: {error}'
        ) from error
    if any(next_step < step for step, next_step in zip(steps, steps[1:])):
        raise M20ADiagnosticError(
            f'train.csv steps are not monotonic in {train_path}: {steps!r}'
        )
    artifact_identity = _audit_file_artifact(train_path)
    return {
        'path': str(train_path),
        'status': 'ok',
        'last_step': None if not steps else int(steps[-1]),
        'row_count': len(steps),
        'sha256': artifact_identity['sha256'],
        'file_stat': artifact_identity['file_stat'],
    }


def _audit_csv_artifact(path):
    """Record lightweight identity and shape information for a CSV artifact."""

    path = Path(path).resolve()
    if not path.is_file():
        return {'path': str(path), 'status': 'missing'}
    try:
        with path.open(newline='') as file:
            reader = csv.reader(file)
            fields = next(reader, [])
            row_count = sum(1 for _ in reader)
        return {
            'path': str(path),
            'status': 'ok',
            'fields': fields,
            'row_count': int(row_count),
            'sha256': sha256_file(path),
            'file_stat': _file_stat(path),
        }
    except (OSError, csv.Error) as error:
        raise M20ADiagnosticError(
            f'Cannot audit CSV artifact {path}: {type(error).__name__}: {error}'
        ) from error


def _checkpoint_identity_error(path, error):
    return f'{type(error).__name__}: {error} (discovery={_CHECKPOINT_DISCOVERY_CODE_PATH}; path={path})'


def _validate_embedded_checkpoint_metadata(
    path,
    payload,
    *,
    filename_step,
    expected_metadata,
    expected_role,
):
    if not isinstance(payload, Mapping):
        raise M20ADiagnosticError(
            f'checkpoint payload must be a mapping, got {type(payload).__name__}'
        )
    if not isinstance(payload.get('agent'), Mapping):
        raise M20ADiagnosticError('checkpoint payload has no agent mapping')
    metadata = payload.get('checkpoint_metadata')
    if not isinstance(metadata, Mapping):
        raise M20ADiagnosticError('checkpoint payload has no checkpoint_metadata mapping')
    try:
        embedded_step = int(metadata.get('checkpoint_step'))
    except (TypeError, ValueError) as error:
        raise M20ADiagnosticError(
            f'checkpoint_metadata.checkpoint_step is not an integer: '
            f'{metadata.get("checkpoint_step")!r}'
        ) from error
    _require_equal(embedded_step, int(filename_step), f'{path} filename/metadata checkpoint step')
    if expected_role is not None:
        _require_equal(
            metadata.get('checkpoint_role'),
            expected_role,
            f'{path} checkpoint_metadata.checkpoint_role',
        )
    for key, expected in expected_metadata.items():
        if expected is None:
            continue
        _require_equal(metadata.get(key), expected, f'{path} checkpoint_metadata.{key}')
    return dict(metadata), embedded_step


def _discover_checkpoint_artifacts(
    source_run,
    *,
    expected_metadata=None,
    observed_process_state='stopped',
):
    """Audit numeric and semantic checkpoint artifacts without restoring them."""

    source_run = Path(source_run).resolve()
    checkpoint_dir = source_run / 'checkpoints'
    audit = {
        'discovery_code_path': _CHECKPOINT_DISCOVERY_CODE_PATH,
        'patterns': list(_CHECKPOINT_DISCOVERY_PATTERNS),
        'numeric_checkpoints': [],
        'semantic_artifacts': [],
    }
    if not checkpoint_dir.is_dir():
        return audit

    numeric_paths = sorted(
        path for path in checkpoint_dir.rglob('params_*.pkl') if path.is_file()
    )
    numeric_path_set = set(numeric_paths)
    for path in numeric_paths:
        match = _NUMERIC_CHECKPOINT.fullmatch(path.name)
        if match is None:
            continue
        filename_step = int(match.group(1))
        relative = path.relative_to(checkpoint_dir)
        if len(relative.parts) == 1:
            role = 'numeric'
            selection_kind = 'direct_numeric'
            selection_priority = 0
        elif relative.parts[0] in {'best', 'last'}:
            role = relative.parts[0]
            selection_kind = f'semantic_{role}_numeric'
            selection_priority = 1 if role == 'last' else 2
        else:
            role = relative.parts[0]
            selection_kind = f'nested_{role}_numeric'
            selection_priority = 3
        record = {
            'step': filename_step,
            'filename_step': filename_step,
            'embedded_checkpoint_step': None,
            'path': str(path),
            'relative_path': str(relative),
            'checkpoint_role': role,
            'selection_kind': selection_kind,
            'selection_priority': selection_priority,
        }
        try:
            stat = path.stat()
            if stat.st_size <= 0:
                raise M20ADiagnosticError('checkpoint file is empty')
            if observed_process_state != 'stopped':
                raise M20ADiagnosticError(
                    f'source process state is {observed_process_state!r}; '
                    'checkpoint is not stable while training may be active'
                )
            sha_before, stat_before = _stable_sha256(path)
            with path.open('rb') as file:
                payload = pickle.load(file)
            expected_role = role if role in {'numeric', 'best', 'last'} else None
            metadata, embedded_step = _validate_embedded_checkpoint_metadata(
                path,
                payload,
                filename_step=filename_step,
                expected_metadata=expected_metadata or {},
                expected_role=expected_role,
            )
            sha_after, stat_after = _stable_sha256(path)
            _require_equal(sha_after, sha_before, f'{path} SHA256 during pickle audit')
            _require_equal(stat_after, stat_before, f'{path} stat during pickle audit')
            record.update({
                'status': 'stable_numeric_checkpoint',
                'sha256': sha_after,
                'file_stat': stat_after,
                'embedded_checkpoint_step': embedded_step,
                'checkpoint_metadata': metadata,
            })
        except Exception as error:
            record.update({
                'status': 'invalid_numeric_checkpoint',
                'error': _checkpoint_identity_error(path, error),
            })
            try:
                record['file_stat'] = _file_stat(path)
            except OSError:
                pass
        audit['numeric_checkpoints'].append(record)

    for role in ('best', 'last'):
        role_dir = checkpoint_dir / role
        if not role_dir.is_dir():
            continue
        for path in sorted(item for item in role_dir.rglob('*') if item.is_file()):
            if path in numeric_path_set:
                continue
            record = {
                'path': str(path),
                'relative_path': str(path.relative_to(checkpoint_dir)),
                'checkpoint_role': role,
            }
            try:
                sha, stat = _stable_sha256(path)
                record.update({
                    'status': 'stable_semantic_artifact',
                    'sha256': sha,
                    'file_stat': stat,
                })
            except (OSError, M20ADiagnosticError) as error:
                record.update({
                    'status': 'unreadable_semantic_artifact',
                    'error': _checkpoint_identity_error(path, error),
                })
            audit['semantic_artifacts'].append(record)
    return audit


def validate_source_run(
    source_run,
    *,
    study_path=DEFAULT_STUDY,
    dataset_root=None,
    source_config=SOURCE_CONFIG_ID,
):
    """Read and validate source metadata/config without restoring a model."""

    source_run_path = Path(source_run).resolve()
    runtime_metadata_path = source_run_path / 'runtime_metadata.json'
    resolved_config_path = source_run_path / 'resolved_config.json'
    metadata = _read_json(runtime_metadata_path)
    resolved = _read_json(resolved_config_path)
    if metadata.get('status') not in {'running', 'completed', 'aborted', 'failed'}:
        raise M20ADiagnosticError(
            f'Unsupported source run status: {metadata.get("status")!r}'
        )
    _require_equal(metadata.get('git_dirty'), False, 'source runtime_metadata.git_dirty')

    study, configuration = prepare_run_design(study_path, source_config)
    source_study_id = study.study_id
    source_config_id = configuration.config_id
    source_environment = configuration.data.get('environment')
    source_training_seed = int(configuration.data.get('training_seed'))
    source_readout = configuration.data.get('factors', {}).get('readout')
    if source_readout not in {'mean_context', 'hybrid_context_query'}:
        raise M20ADiagnosticError(
            f'Unsupported source readout for {source_config_id}: {source_readout!r}'
        )
    path_parts = source_run_path.parts
    if len(path_parts) < 5 or '__' not in path_parts[-3] or not path_parts[-1].startswith('seed_'):
        raise M20ADiagnosticError(f'Not a canonical M20A source run path: {source_run_path}')
    # Canonical layout is ``.../runs/<study>/<config>__<slug>/<env>/<seed>``.
    # With the trailing seed/env/config/study components, study is ``[-4]``.
    path_study = path_parts[-4]
    path_config, path_slug = path_parts[-3].split('__', 1)
    path_environment = path_parts[-2]
    try:
        path_seed = int(path_parts[-1].removeprefix('seed_'))
    except ValueError as error:
        raise M20ADiagnosticError(f'Invalid source seed path: {source_run_path}') from error
    for key, actual, expected in (
        ('study', path_study, source_study_id),
        ('config', path_config, source_config_id),
        ('environment', path_environment, source_environment),
        ('seed', path_seed, source_training_seed),
    ):
        _require_equal(actual, expected, f'source path {key}')
    authoritative_path = make_run_path(
        _source_run_root(source_run_path),
        study.study_id,
        source_config_id,
        configuration.slug,
        source_environment,
        source_training_seed,
    ).resolve()
    _require_equal(
        source_run_path,
        authoritative_path,
        'authoritative resolver source run path',
    )
    for key, expected in (
        ('study_id', source_study_id),
        ('config_id', source_config_id),
        ('config_slug', path_slug),
        ('environment', source_environment),
        ('seed', source_training_seed),
        ('algorithm', 'gciql'),
    ):
        _require_equal(metadata.get(key), expected, f'source runtime_metadata.{key}')
    _require_equal(
        Path(metadata.get('run_dir', source_run_path)).resolve(),
        source_run_path,
        'source runtime_metadata.run_dir',
    )

    payload = {
        'study': resolved.get('study'),
        'configuration': resolved.get('configuration'),
        'algorithm_config': resolved.get('algorithm_config', {}),
    }
    calculated_fingerprint = config_fingerprint(payload)
    _require_equal(
        resolved.get('resolved_config_fingerprint'),
        calculated_fingerprint,
        'source resolved_config fingerprint',
    )
    _require_equal(
        metadata.get('resolved_config_fingerprint'),
        calculated_fingerprint,
        'source runtime metadata fingerprint',
    )

    _require_equal(configuration.slug, path_slug, 'local/source config slug')
    _require_equal(configuration.data.get('environment'), source_environment, 'local source environment')
    agent_config = _resolved_agent_config(resolved)
    _require_equal(agent_config.get('agent_name'), 'gciql', 'source agent_name')
    _require_equal(agent_config.get('alpha'), 1.0, 'source GCIQL alpha')
    _require_equal(agent_config.get('actor_loss'), 'ddpgbc', 'source actor_loss')
    _require_equal(agent_config.get('dataset_class'), 'GCDataset', 'source dataset_class')
    for slot_name in ('actor', 'value', 'critic'):
        _validate_slot_config(
            agent_config, slot_name, expected_readout=source_readout,
        )

    resolved_dataset_root = Path(
        dataset_root or metadata.get('dataset_dir') or DEFAULT_DATASET_ROOT
    ).resolve()
    metadata_dataset_root = metadata.get('dataset_dir')
    if metadata_dataset_root is not None:
        _require_equal(
            Path(metadata_dataset_root).resolve(), resolved_dataset_root,
            'source dataset directory',
        )
    dataset_identity = _dataset_identity(
        resolved_dataset_root, environment=source_environment,
    )

    index_path = source_run_path / 'checkpoints' / 'index.json'
    checkpoint_index = _read_json(index_path) if index_path.is_file() else None
    process_observation = _observe_source_process(
        source_run_path, source_config_id=source_config_id,
    )
    training_step = _read_last_training_step(source_run_path)
    expected_checkpoint_metadata = {
        'environment': source_environment,
        'study_id': source_study_id,
        'config_id': source_config_id,
        'config_slug': path_slug,
        'git_commit': metadata.get('git_commit'),
        'seed': source_training_seed,
        'training_seed': source_training_seed,
    }
    checkpoint_audit = _discover_checkpoint_artifacts(
        source_run_path,
        expected_metadata=expected_checkpoint_metadata,
        observed_process_state=process_observation['state'],
    )
    numeric = checkpoint_audit['numeric_checkpoints']
    semantic_last_fallback = discover_semantic_last_fallback(
        source_run_path, checkpoint_index, numeric_records=numeric,
    )
    failure_path = source_run_path / 'failure.json'
    failure_artifact = _read_json(failure_path) if failure_path.is_file() else None
    if metadata.get('termination_context') is not None:
        termination_context = metadata['termination_context']
    elif failure_artifact is not None:
        termination_context = {
            'artifact': str(failure_path),
            'status': failure_artifact.get('status'),
            'failure_reason': failure_artifact.get('failure_reason'),
            'time': failure_artifact.get('time'),
        }
    else:
        termination_context = 'not_recorded'
    snapshot = _file_snapshot(source_run_path)
    return {
        'source_run_path': str(source_run_path),
        'source_metadata': metadata,
        'resolved_config': resolved,
        'agent_config': agent_config,
        'source_status': metadata.get('status'),
        'recorded_runtime_status': metadata.get('status'),
        'observed_process_state': process_observation['state'],
        'source_process_observation': process_observation,
        'termination_context': termination_context,
        'observed_last_training_step': training_step['last_step'],
        'training_step_artifact': training_step,
        'source_study_id': source_study_id,
        'source_config_id': source_config_id,
        'source_config_slug': path_slug,
        'source_environment': source_environment,
        'source_training_seed': source_training_seed,
        'source_readout': source_readout,
        'diagnostic_campaign_id': _campaign_id(source_config_id),
        'diagnostic_namespace': str(_campaign_namespace(source_config_id)),
        'source_git_commit': metadata.get('git_commit'),
        'source_resolved_config_fingerprint': calculated_fingerprint,
        'source_path_resolution': {
            'resolver': 'prepare_run_design + make_run_path',
            'study_path': str(Path(study_path).resolve()),
            'run_root': str(_source_run_root(source_run_path)),
            'authoritative_path': str(authoritative_path),
            'observed_path': str(source_run_path),
            'matches': True,
        },
        'dataset_identity': dataset_identity,
        'dataset_root': str(resolved_dataset_root),
        'runtime_metadata_artifact': _audit_file_artifact(runtime_metadata_path),
        'resolved_config_artifact': _audit_file_artifact(resolved_config_path),
        'train_artifact': training_step,
        'eval_artifact': _audit_csv_artifact(source_run_path / 'eval.csv'),
        'failure_artifact': failure_artifact,
        'checkpoint_index_path': None if checkpoint_index is None else str(index_path),
        'checkpoint_index': checkpoint_index,
        'checkpoint_discovery_code_path': _CHECKPOINT_DISCOVERY_CODE_PATH,
        'checkpoint_audit': checkpoint_audit,
        'numeric_checkpoints': numeric,
        'latest_recoverable_checkpoint_step': max(
            (
                int(item['step']) for item in numeric
                if item.get('status') == 'stable_numeric_checkpoint'
            ),
            default=None,
        ),
        'semantic_last_fallback': semantic_last_fallback,
        'source_file_snapshot': snapshot,
        'local_git': _git_info(),
    }


def discover_numeric_checkpoints(source_run):
    """Return audited numeric checkpoints, including semantic directories."""

    return _discover_checkpoint_artifacts(source_run)['numeric_checkpoints']


def discover_semantic_last_fallback(source_run, checkpoint_index, *, numeric_records):
    """Lock semantic ``last`` only when no numeric checkpoint exists.

    M20A-D prefers immutable numeric artifacts.  The source checkpoint
    lifecycle permits a semantic ``last`` fallback only when the source has no
    numeric artifact at all; its indexed metadata and stable SHA are then
    recorded explicitly so a moving training-time path is never followed.
    """

    if numeric_records or not isinstance(checkpoint_index, Mapping):
        return None
    entry = checkpoint_index.get('last')
    if not isinstance(entry, Mapping) or not isinstance(entry.get('path'), str):
        return None
    source_run = Path(source_run).resolve()
    checkpoint_path = (source_run / entry['path']).resolve()
    if not _under(checkpoint_path, source_run):
        raise M20ADiagnosticError(
            f'Semantic last checkpoint escapes source run: {checkpoint_path}'
        )
    metadata_path = entry.get('metadata_path')
    if metadata_path is None:
        metadata_path = 'checkpoints/last/checkpoint.json'
    metadata_path = (source_run / metadata_path).resolve()
    if not _under(metadata_path, source_run) or not metadata_path.is_file():
        return {
            'status': 'semantic_last_missing_metadata',
            'path': str(checkpoint_path),
            'metadata_path': str(metadata_path),
            'error': 'semantic last metadata is missing',
        }
    if not checkpoint_path.is_file():
        return {
            'status': 'semantic_last_missing_file',
            'path': str(checkpoint_path),
            'metadata_path': str(metadata_path),
            'error': 'semantic last checkpoint file is missing',
        }
    metadata = _read_json(metadata_path)
    if metadata.get('checkpoint_role') not in (None, 'last'):
        raise M20ADiagnosticError(
            f'Semantic fallback metadata role mismatch: {metadata.get("checkpoint_role")!r}'
        )
    step = metadata.get('checkpoint_step', entry.get('step'))
    try:
        step = int(step)
    except (TypeError, ValueError) as error:
        raise M20ADiagnosticError(
            f'Semantic fallback has no integer checkpoint_step: {metadata_path}'
        ) from error
    sha, stat = _stable_sha256(checkpoint_path)
    indexed_sha = entry.get('sha256')
    metadata_sha = metadata.get('checkpoint_sha256')
    for label, expected in (
        ('checkpoint index SHA256', indexed_sha),
        ('semantic metadata SHA256', metadata_sha),
    ):
        if expected is not None and expected != sha:
            raise M20ADiagnosticError(
                f'{label} mismatch for semantic last: expected={expected!r}, observed={sha!r}'
            )
    return {
        'step': step,
        'path': str(checkpoint_path),
        'status': 'stable_semantic_last_sha_locked',
        'selection_kind': 'semantic_last_fallback',
        'sha256': sha,
        'file_stat': stat,
        'metadata_path': str(metadata_path),
        'checkpoint_metadata': metadata,
    }


def parse_checkpoint_selectors(value):
    if value is None:
        return list(DEFAULT_CHECKPOINT_STEPS)
    selectors = []
    for item in str(value).split(','):
        item = item.strip().lower()
        if not item:
            continue
        if item == 'latest':
            selectors.append('latest')
        elif item.isdigit() and int(item) >= 0:
            selectors.append(int(item))
        else:
            raise M20ADiagnosticError(
                f'Invalid checkpoint selector {item!r}; expected numeric step or latest'
            )
    if not selectors:
        raise M20ADiagnosticError('At least one checkpoint selector is required')
    if len(set(selectors)) != len(selectors):
        raise M20ADiagnosticError(f'Repeated checkpoint selector: {selectors!r}')
    return selectors


def select_checkpoints(numeric_records, selectors):
    by_step = {}
    for item in numeric_records:
        if item.get('status') not in {
            'stable_numeric_checkpoint',
            # Kept for compatibility with pre-prompt4 in-memory callers.
            'stable_hash_locked',
        }:
            continue
        step = int(item['step'])
        current = by_step.get(step)
        if current is None or (
            int(item.get('selection_priority', 99)), str(item.get('path'))
        ) < (
            int(current.get('selection_priority', 99)), str(current.get('path'))
        ):
            by_step[step] = item
    latest_step = max(by_step) if by_step else None
    selected = []
    missing = []
    aliases = {}
    for selector in selectors:
        step = latest_step if selector == 'latest' else int(selector)
        if step is None or step not in by_step:
            missing.append({'selector': selector, 'resolved_step': step})
            continue
        aliases.setdefault(step, []).append(selector)
    for step in sorted(aliases):
        record = dict(by_step[step])
        record['selector_aliases'] = aliases[step]
        record['selection'] = 'latest_stable_numeric' if 'latest' in aliases[step] else 'explicit_numeric'
        selected.append(record)
    return {
        'requested_selectors': selectors,
        'latest_stable_numeric_step': latest_step,
        'selected': selected,
        'missing': missing,
        'discovered_numeric_steps': sorted(by_step),
    }


def add_semantic_last_fallback(checkpoint_plan, fallback):
    """Resolve ``latest`` to a stable semantic last only in the no-numeric case."""

    if checkpoint_plan['discovered_numeric_steps'] or fallback is None:
        checkpoint_plan['semantic_last_fallback'] = None
        return checkpoint_plan
    checkpoint_plan['semantic_last_fallback'] = fallback
    latest_requested = 'latest' in checkpoint_plan['requested_selectors']
    if latest_requested and fallback.get('status') == 'stable_semantic_last_sha_locked':
        checkpoint_plan['selected'].append({
            **fallback,
            'selector_aliases': ['latest'],
            'selection': 'semantic_last_fallback_stable_sha',
        })
        checkpoint_plan['selected'].sort(key=lambda item: int(item['step']))
        checkpoint_plan['missing'] = [
            item for item in checkpoint_plan['missing']
            if item.get('selector') != 'latest'
        ]
    return checkpoint_plan


def _fixed_batch_paths(output_root, batch_size, diagnostic_seed):
    root = (
        Path(output_root).resolve() / DIAGNOSTIC_ID / 'fixed_batch'
        / f'cube_triple_N{int(batch_size)}_seed{int(diagnostic_seed)}'
    )
    return root, root / 'fixed_batch.npz', root / 'fixed_batch_metadata.json'


def _batch_config_metadata(provenance):
    config = provenance['agent_config']
    return {
        'actor_goal_sampling': {
            key: _jsonable(config.get(key))
            for key in ('actor_p_curgoal', 'actor_p_trajgoal', 'actor_p_randomgoal', 'actor_geom_sample')
        },
        'value_goal_sampling': {
            key: _jsonable(config.get(key))
            for key in ('value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal', 'value_geom_sample')
        },
        'discount': _jsonable(config.get('discount')),
        'gc_negative': _jsonable(config.get('gc_negative')),
        'p_aug': _jsonable(config.get('p_aug')),
        'frame_stack': _jsonable(config.get('frame_stack')),
    }


def _validate_batch_arrays(arrays, batch_size):
    if set(arrays) != _REQUIRED_BATCH_FIELDS:
        raise M20ADiagnosticError(
            f'Fixed batch fields mismatch: expected={sorted(_REQUIRED_BATCH_FIELDS)}, '
            f'observed={sorted(arrays)}'
        )
    for name, array in arrays.items():
        if np.asarray(array).ndim < 1 or np.asarray(array).shape[0] != int(batch_size):
            raise M20ADiagnosticError(
                f'Fixed batch field {name!r} does not have leading dimension {batch_size}: '
                f'{np.asarray(array).shape}'
            )


def _load_fixed_batch(batch_path, metadata_path, *, provenance, batch_size, diagnostic_seed):
    if not Path(batch_path).is_file() or not Path(metadata_path).is_file():
        raise M20ADiagnosticError(
            f'Fixed batch pair is incomplete: {batch_path}, {metadata_path}'
        )
    metadata = _read_json(metadata_path)
    _require_equal(metadata.get('diagnostic_id'), DIAGNOSTIC_ID, 'fixed batch diagnostic_id')
    _require_equal(
        metadata.get('environment'), provenance['source_environment'],
        'fixed batch environment',
    )
    _require_equal(metadata.get('batch_size'), int(batch_size), 'fixed batch size')
    _require_equal(metadata.get('diagnostic_seed'), int(diagnostic_seed), 'fixed batch seed')
    _require_equal(
        metadata.get('dataset_identity', {}).get('sha256'),
        provenance['dataset_identity']['sha256'],
        'fixed batch dataset fingerprint',
    )
    _require_equal(
        metadata.get('batch_fingerprint_sha256'),
        EXPECTED_FIXED_BATCH_FINGERPRINT,
        'fixed batch frozen expected fingerprint',
    )
    _require_equal(
        metadata.get('sampling_semantics'),
        _batch_config_metadata(provenance),
        'fixed batch sampling semantics',
    )
    with np.load(batch_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    _validate_batch_arrays(arrays, batch_size)
    fingerprint = _array_fingerprint(arrays)
    _require_equal(metadata.get('batch_fingerprint_sha256'), fingerprint, 'fixed batch fingerprint')
    return arrays, metadata


def create_or_load_fixed_batch(
    output_root,
    provenance,
    *,
    batch_size=DEFAULT_BATCH_SIZE,
    diagnostic_seed=DEFAULT_DIAGNOSTIC_SEED,
):
    """Create exactly one fixed batch through the production GCDataset API."""

    root, batch_path, metadata_path = _fixed_batch_paths(
        output_root, batch_size, diagnostic_seed,
    )
    if batch_path.exists() or metadata_path.exists():
        return _load_fixed_batch(
            batch_path,
            metadata_path,
            provenance=provenance,
            batch_size=batch_size,
            diagnostic_seed=diagnostic_seed,
        ) + ({'root': str(root), 'created': False},)

    root.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env, raw_train, _ = make_env_and_datasets(
            provenance['source_environment'],
            frame_stack=provenance['agent_config'].get('frame_stack'),
            seed=diagnostic_seed,
            dataset_seed=diagnostic_seed,
            dataset_dir=provenance['dataset_root'],
        )
        dataset = GCDataset(
            raw_train,
            provenance['agent_config'],
            rng=np.random.default_rng(int(diagnostic_seed)),
        )
        batch, trace = dataset.sample(
            int(batch_size),
            evaluation=True,
            rng=np.random.default_rng(int(diagnostic_seed)),
            return_sampling_trace=True,
        )
        arrays = {
            'sample_id': np.arange(int(batch_size), dtype=np.int64),
            'transition_indices': np.asarray(trace['transition_indices'], dtype=np.int64),
            'actor_goal_indices': np.asarray(trace['actor_goal_indices'], dtype=np.int64),
            'value_goal_indices': np.asarray(trace['value_goal_indices'], dtype=np.int64),
        }
        for name in (
            'observations', 'next_observations', 'actions', 'actor_goals',
            'value_goals', 'rewards', 'masks',
        ):
            arrays[name] = np.asarray(batch[name])
        _validate_batch_arrays(arrays, batch_size)
        fingerprint = _array_fingerprint(arrays)
        metadata = {
            'diagnostic_id': DIAGNOSTIC_ID,
            'artifact_schema': f'{DIAGNOSTIC_SCHEMA}_fixed_batch_v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'environment': provenance['source_environment'],
            'dataset_class': 'GCDataset',
            'dataset_identity': provenance['dataset_identity'],
            'dataset_path': provenance['dataset_identity']['path'],
            'fixed_batch_path': str(batch_path),
            'fixed_batch_metadata_path': str(metadata_path),
            'diagnostic_seed': int(diagnostic_seed),
            'sampling_rng': 'numpy.random.default_rng(diagnostic_seed)',
            'batch_size': int(batch_size),
            'source_study': provenance['source_study_id'],
            'source_config': provenance['source_config_id'],
            'source_run_path': provenance['source_run_path'],
            'source_training_seed': provenance['source_training_seed'],
            'source_git_commit': provenance['source_git_commit'],
            'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
            'actor_goal_sampling': _batch_config_metadata(provenance)['actor_goal_sampling'],
            'value_goal_sampling': _batch_config_metadata(provenance)['value_goal_sampling'],
            'sampling_semantics': _batch_config_metadata(provenance),
            'sampling_method': 'GCDataset.sample(batch_size, evaluation=True, return_sampling_trace=True)',
            'trace_fields': [
                'sample_id', 'transition_indices', 'actor_goal_indices', 'value_goal_indices',
            ],
            'arrays': {
                name: {'shape': list(np.asarray(array).shape), 'dtype': str(np.asarray(array).dtype)}
                for name, array in sorted(arrays.items())
            },
            'batch_fingerprint_sha256': fingerprint,
        }
        with batch_path.open('wb') as file:
            np.savez_compressed(file, **arrays)
        _write_json(metadata_path, metadata)
        return arrays, metadata, {'root': str(root), 'created': True}
    except BaseException:
        # Leave an incomplete directory as a visible collision rather than
        # deleting an artifact that may help diagnose an interrupted run.
        raise
    finally:
        if env is not None:
            env.close()


def verify_fixed_batch_reproducibility(
    provenance,
    arrays,
    *,
    batch_size=DEFAULT_BATCH_SIZE,
    diagnostic_seed=DEFAULT_DIAGNOSTIC_SEED,
):
    """Re-sample the locked batch and require exact field-by-field identity."""

    env = None
    try:
        env, raw_train, _ = make_env_and_datasets(
            provenance['source_environment'],
            frame_stack=provenance['agent_config'].get('frame_stack'),
            seed=diagnostic_seed,
            dataset_seed=diagnostic_seed,
            dataset_dir=provenance['dataset_root'],
        )
        dataset = GCDataset(
            raw_train,
            provenance['agent_config'],
            rng=np.random.default_rng(int(diagnostic_seed)),
        )
        batch, trace = dataset.sample(
            int(batch_size),
            evaluation=True,
            rng=np.random.default_rng(int(diagnostic_seed)),
            return_sampling_trace=True,
        )
        resampled = {
            'sample_id': np.arange(int(batch_size), dtype=np.int64),
            'transition_indices': np.asarray(trace['transition_indices'], dtype=np.int64),
            'actor_goal_indices': np.asarray(trace['actor_goal_indices'], dtype=np.int64),
            'value_goal_indices': np.asarray(trace['value_goal_indices'], dtype=np.int64),
        }
        for name in (
            'observations', 'next_observations', 'actions', 'actor_goals',
            'value_goals', 'rewards', 'masks',
        ):
            resampled[name] = np.asarray(batch[name])
        _validate_batch_arrays(resampled, batch_size)
        mismatches = []
        for name in sorted(_REQUIRED_BATCH_FIELDS):
            expected = np.asarray(arrays[name])
            observed = np.asarray(resampled[name])
            if expected.dtype != observed.dtype or expected.shape != observed.shape or not np.array_equal(expected, observed):
                mismatches.append({
                    'field': name,
                    'expected_dtype': str(expected.dtype),
                    'observed_dtype': str(observed.dtype),
                    'expected_shape': list(expected.shape),
                    'observed_shape': list(observed.shape),
                })
        if mismatches:
            raise M20ADiagnosticError(
                f'Fixed batch deterministic re-sampling mismatch: {mismatches}'
            )
        return {
            'status': 'pass',
            'method': 'fresh GCDataset.sample with identical seeds and config',
            'fields_checked': sorted(_REQUIRED_BATCH_FIELDS),
            'batch_fingerprint_sha256': _array_fingerprint(resampled),
        }
    finally:
        if env is not None:
            env.close()


def verify_and_record_existing_fixed_batch(
    output_root,
    provenance,
    *,
    batch_size=DEFAULT_BATCH_SIZE,
    diagnostic_seed=DEFAULT_DIAGNOSTIC_SEED,
):
    """Revalidate an existing fixed batch and persist the reproducibility proof."""

    _, batch_path, metadata_path = _fixed_batch_paths(
        output_root, batch_size, diagnostic_seed,
    )
    arrays, metadata = _load_fixed_batch(
        batch_path,
        metadata_path,
        provenance=provenance,
        batch_size=batch_size,
        diagnostic_seed=diagnostic_seed,
    )
    result = verify_fixed_batch_reproducibility(
        provenance,
        arrays,
        batch_size=batch_size,
        diagnostic_seed=diagnostic_seed,
    )
    metadata = dict(metadata)
    metadata['reproducibility_check'] = result
    _write_json(metadata_path, metadata, overwrite=True)
    return result


def _relation_base(trace, *, ensemble):
    relations = np.asarray(trace['relations_original'])
    if ensemble:
        if relations.ndim != 5 or relations.shape[0] != 2:
            raise M20ADiagnosticError(f'Unexpected critic relation shape: {relations.shape}')
        if not np.array_equal(relations[0], relations[1]):
            raise M20ADiagnosticError('Critic ensemble members built different relation tensors')
        return relations[0]
    if relations.ndim != 4:
        raise M20ADiagnosticError(f'Unexpected relation shape: {relations.shape}')
    return relations


def _activity_groups(relations):
    relations = np.asarray(relations)
    counts = np.sum(relations, axis=(1, 2))
    any_active = np.any(counts > 0, axis=-1)
    return counts, {
        'inactive': ~any_active,
        'any_relation_active': any_active,
        'current_support_active': counts[:, 0] > 0,
        'goal_support_active': counts[:, 1] > 0,
        'goal_conflict_active': counts[:, 2] > 0,
    }


def _rms(values, axes):
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.mean(np.square(values), axis=axes))


def _l2(values, axes):
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.sum(np.square(values), axis=axes))


def _collapse_metric_diff(left, right, *, ensemble, per_sample_axes):
    diff = np.asarray(left) - np.asarray(right)
    axes = (0,) + tuple(axis + 1 for axis in per_sample_axes) if ensemble else per_sample_axes
    return _rms(diff, axes)


def _hidden_metrics(correct, zero, *, ensemble):
    h0 = np.asarray(correct['tokens_pre_relation'])
    relation_delta = np.asarray(correct['tokens_post_relation']) - h0
    if ensemble:
        relation_axes = (0, 2, 3)
        norm_axes = (0, 2, 3)
        core_axes = (0, 2, 3)
        readout_axes = (0, 2)
    else:
        relation_axes = (1, 2)
        norm_axes = (1, 2)
        core_axes = (1, 2)
        readout_axes = (1,)
    delta_rel = _rms(relation_delta, relation_axes)
    h0_norm = _l2(h0, norm_axes)
    core_diff = np.asarray(correct['tokens_post_core']) - np.asarray(zero['tokens_post_core'])
    core_zero = np.asarray(zero['tokens_post_core'])
    delta_mix = _rms(core_diff, core_axes)
    core_zero_norm = _l2(core_zero, core_axes)
    readout_diff = np.asarray(correct['readout_vector']) - np.asarray(zero['readout_vector'])
    readout_zero = np.asarray(zero['readout_vector'])
    delta_readout = _rms(readout_diff, readout_axes)
    readout_zero_norm = _l2(readout_zero, readout_axes)
    if correct.get('attention_weights') is None or zero.get('attention_weights') is None:
        attention_tv = None
        attention_status = 'not_applicable_mean_readout'
    elif ensemble:
        attention_diff = np.asarray(correct['attention_weights']) - np.asarray(zero['attention_weights'])
        attention_tv = 0.5 * np.mean(np.sum(np.abs(attention_diff), axis=-1), axis=0)
        attention_status = 'applicable_hybrid_context_query'
    else:
        attention_diff = np.asarray(correct['attention_weights']) - np.asarray(zero['attention_weights'])
        attention_tv = 0.5 * np.sum(np.abs(attention_diff), axis=-1)
        attention_status = 'applicable_hybrid_context_query'
    propagation = np.full_like(delta_mix, np.nan, dtype=np.float64)
    valid = delta_rel > ATOL
    propagation[valid] = delta_mix[valid] / (delta_rel[valid] + EPSILON)
    return {
        'delta_rel': delta_rel,
        'relative_relation_update': delta_rel / (h0_norm + EPSILON),
        'delta_mix': delta_mix,
        'relative_core_difference': _l2(core_diff, core_axes) / (core_zero_norm + EPSILON),
        'propagation_ratio': propagation,
        'delta_readout': delta_readout,
        'relative_readout_difference': _l2(readout_diff, readout_axes) / (readout_zero_norm + EPSILON),
        'attention_tv': attention_tv,
        'attention_status': attention_status,
    }


def _output_metrics(correct, zero, slot, *, ensemble):
    result = {}
    if slot == 'actor':
        mean_r = np.asarray(correct['action_means'])
        mean_0 = np.asarray(zero['action_means'])
        action_r = np.clip(mean_r, -1.0, 1.0)
        action_0 = np.clip(mean_0, -1.0, 1.0)
        mean_diff = mean_r - mean_0
        action_diff = action_r - action_0
        result.update({
            'delta_mean': _rms(mean_diff, (1,)),
            'actor_mean_l2': _l2(mean_diff, (1,)),
            'delta_action': _rms(action_diff, (1,)),
            'actor_clipped_action_l2': _l2(action_diff, (1,)),
            'actor_max_abs_difference': np.max(np.abs(mean_diff), axis=-1),
            'mu_R': mean_r,
            'mu_0': mean_0,
            'action_R': action_r,
            'action_0': action_0,
        })
    elif slot == 'value':
        value_r = np.asarray(correct['values'])
        value_0 = np.asarray(zero['values'])
        result.update({
            'delta_value': np.abs(value_r - value_0),
            'relative_value_difference': np.abs(value_r - value_0) / (np.abs(value_0) + EPSILON),
            'value_R': value_r,
            'value_0': value_0,
        })
    elif slot == 'critic':
        q_r = np.asarray(correct['values'])
        q_0 = np.asarray(zero['values'])
        if q_r.shape[0] != 2 or q_0.shape != q_r.shape:
            raise M20ADiagnosticError(f'Unexpected critic scalar output shape: {q_r.shape}, {q_0.shape}')
        qmin_r = np.minimum(q_r[0], q_r[1])
        qmin_0 = np.minimum(q_0[0], q_0[1])
        result.update({
            'delta_q1': np.abs(q_r[0] - q_0[0]),
            'delta_q2': np.abs(q_r[1] - q_0[1]),
            'delta_qmin': np.abs(qmin_r - qmin_0),
            'Q1_R': q_r[0],
            'Q2_R': q_r[1],
            'Qmin_R': qmin_r,
            'Q1_0': q_0[0],
            'Q2_0': q_0[1],
            'Qmin_0': qmin_0,
        })
    else:  # pragma: no cover - fixed diagnostic slot set.
        raise M20ADiagnosticError(f'Unknown diagnostic slot: {slot!r}')
    return result


def _primary_metrics(correct, zero, slot, *, ensemble):
    result = _hidden_metrics(correct, zero, ensemble=ensemble)
    result.update(_output_metrics(correct, zero, slot, ensemble=ensemble))
    return result


def _channel_metrics(all_trace, drop_trace, slot, *, ensemble):
    result = {}
    if ensemble:
        hidden_axes = (0, 2, 3)
        readout_axes = (0, 2)
    else:
        hidden_axes = (1, 2)
        readout_axes = (1,)
    if all_trace.get('attention_weights') is None or drop_trace.get('attention_weights') is None:
        result['channel_delta_attention'] = None
        result['attention_status'] = 'not_applicable_mean_readout'
    elif ensemble:
        attention_diff = np.asarray(all_trace['attention_weights']) - np.asarray(drop_trace['attention_weights'])
        result['channel_delta_attention'] = 0.5 * np.mean(np.sum(np.abs(attention_diff), axis=-1), axis=0)
        result['attention_status'] = 'applicable_hybrid_context_query'
    else:
        attention_diff = np.asarray(all_trace['attention_weights']) - np.asarray(drop_trace['attention_weights'])
        result['channel_delta_attention'] = 0.5 * np.sum(np.abs(attention_diff), axis=-1)
        result['attention_status'] = 'applicable_hybrid_context_query'
    rel_diff = np.asarray(all_trace['tokens_post_relation']) - np.asarray(drop_trace['tokens_post_relation'])
    mix_diff = np.asarray(all_trace['tokens_post_core']) - np.asarray(drop_trace['tokens_post_core'])
    readout_diff = np.asarray(all_trace['readout_vector']) - np.asarray(drop_trace['readout_vector'])
    result.update({
        'channel_delta_rel': _rms(rel_diff, hidden_axes),
        'channel_delta_mix': _rms(mix_diff, hidden_axes),
        'channel_delta_readout': _rms(readout_diff, readout_axes),
    })
    if slot == 'actor':
        result['channel_output_effect'] = _rms(
            np.clip(np.asarray(all_trace['action_means']), -1.0, 1.0)
            - np.clip(np.asarray(drop_trace['action_means']), -1.0, 1.0),
            (1,),
        )
    elif slot == 'value':
        result['channel_output_effect'] = np.abs(
            np.asarray(all_trace['values']) - np.asarray(drop_trace['values'])
        )
    else:
        q_all = np.asarray(all_trace['values'])
        q_drop = np.asarray(drop_trace['values'])
        result['channel_output_effect'] = np.abs(
            np.minimum(q_all[0], q_all[1]) - np.minimum(q_drop[0], q_drop[1])
        )
    return result


def _select_batch(array, mask, *, ensemble):
    array = np.asarray(array)
    mask = np.asarray(mask, dtype=bool)
    return array[:, mask, ...] if ensemble else array[mask, ...]


def _max_abs_group(left, right, mask, *, ensemble):
    if not np.any(mask):
        return None
    left_selected = _select_batch(left, mask, ensemble=ensemble)
    right_selected = _select_batch(right, mask, ensemble=ensemble)
    difference = np.abs(left_selected - right_selected)
    return float(np.max(difference)) if difference.size else 0.0


def _negative_control(correct, zero, groups, *, ensemble, label):
    checks = {}
    fields = (
        'tokens_post_relation', 'tokens_post_core', 'readout_vector',
        'attention_weights', 'action_means' if 'action_means' in correct else 'values',
    )
    for group_name, mask in groups.items():
        max_values = {}
        for field in fields:
            if correct.get(field) is None:
                continue
            max_values[field] = _max_abs_group(
                correct[field], zero[field], mask, ensemble=ensemble,
            )
        observed = [value for value in max_values.values() if value is not None]
        max_abs = max(observed) if observed else None
        checks[group_name] = {
            'count': int(np.sum(mask)),
            'max_abs_by_field': max_values,
            'max_abs': max_abs,
            'status': 'not_observed' if not np.any(mask) else (
                'pass' if max_abs is None or max_abs <= ATOL + RTOL else 'fail'
            ),
        }
        if checks[group_name]['status'] == 'fail':
            raise M20ADiagnosticError(
                f'{label} negative control failed for {group_name}: {max_values}'
            )
    return checks


def _aggregate(values, mask):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    selected = values[mask]
    finite = selected[np.isfinite(selected)]
    result = {
        'count': int(selected.size),
        'finite_count': int(finite.size),
        'fraction_gt_numerical_tolerance': None,
    }
    if finite.size == 0:
        result.update({key: None for key in ('mean', 'std', 'median', 'p10', 'p25', 'p75', 'p90', 'max')})
        return result
    result.update({
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
        'median': float(np.median(finite)),
        'p10': float(np.percentile(finite, 10)),
        'p25': float(np.percentile(finite, 25)),
        'p75': float(np.percentile(finite, 75)),
        'p90': float(np.percentile(finite, 90)),
        'max': float(np.max(finite)),
        'fraction_gt_numerical_tolerance': float(np.mean(finite > ATOL)),
    })
    return result


def _metric_summaries(metrics, groups):
    return {
        group_name: {
            metric_name: _aggregate(values, mask)
            for metric_name, values in metrics.items()
            if np.asarray(values).ndim == 1 and np.asarray(values).shape[0] == mask.shape[0]
        }
        for group_name, mask in groups.items()
    }


def _coverage(relations, groups):
    counts, _ = _activity_groups(relations)
    result = {
        'total_edge_count': int(np.sum(counts)),
        'active_fraction': float(np.mean(np.any(counts > 0, axis=-1))),
        'per_channel': {},
    }
    for index, name in enumerate(RELATION_NAMES):
        channel = counts[:, index]
        result['per_channel'][name] = {
            'edge_count': _aggregate(channel, np.ones(channel.shape[0], dtype=bool)),
            'active_fraction': float(np.mean(channel > 0)),
        }
    result['groups'] = {
        name: {'count': int(np.sum(mask)), 'fraction': float(np.mean(mask))}
        for name, mask in groups.items()
    }
    return result


def _relation_params(module_params):
    flat = flatten_dict(module_params)
    paths = [
        path for path in flat
        if 'relation_augmenter' in path
        or 'relation_dense1' in path
        or 'relation_dense2' in path
    ]
    if not paths:
        raise M20ADiagnosticError('No RelationAugmenter parameter subtree found')
    return [flat[path] for path in paths], [tuple(path) for path in paths]


def _norms(values):
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    return {
        'l1': float(sum(np.sum(np.abs(value)) for value in arrays)),
        'l2': float(np.sqrt(sum(np.sum(np.square(value)) for value in arrays))),
        'parameter_count': int(sum(value.size for value in arrays)),
    }


def _module_subtree(tree, module_name):
    key = module_name if module_name in tree else f'modules_{module_name}'
    if key not in tree:
        raise M20ADiagnosticError(f'Parameter tree has no module {module_name!r}')
    return tree[key]


def _gradient_metrics(agent, batch):
    before = {
        'params': _pytree_fingerprint(agent.network.params),
        'model_state': _pytree_fingerprint(agent.network.model_state),
        'opt_state': _pytree_fingerprint(agent.network.opt_state),
        'rng': _pytree_fingerprint(agent.rng),
        'step': int(agent.network.step),
    }
    gradients = {}
    loss_fns = {
        'actor': lambda params: agent.actor_loss(batch, params)[0],
        'value': lambda params: agent.value_loss(batch, params)[0],
        'critic': lambda params: agent.critic_loss(batch, params)[0],
    }
    for slot_name, loss_fn in loss_fns.items():
        grads = jax.grad(loss_fn)(agent.network.params)
        module_params = _module_subtree(agent.network.params, slot_name)
        module_grads = _module_subtree(grads, slot_name)
        relation_params, relation_paths = _relation_params(module_params)
        relation_grad_flat = flatten_dict(module_grads)
        relation_grads = []
        for path in relation_paths:
            if path not in relation_grad_flat:
                raise M20ADiagnosticError(
                    f'Gradient tree lost RelationAugmenter path {slot_name}/{path}'
                )
            relation_grads.append(relation_grad_flat[path])
        relation_norms = _norms(relation_params)
        relation_grad_norms = _norms(relation_grads)
        full_norms = _norms(jax.tree_util.tree_leaves(module_grads))
        gradients[slot_name] = {
            'loss_name': f'{slot_name}_loss',
            'relation_param_l1': relation_norms['l1'],
            'relation_param_l2': relation_norms['l2'],
            'relation_param_count': relation_norms['parameter_count'],
            'relation_grad_l1': relation_grad_norms['l1'],
            'relation_grad_l2': relation_grad_norms['l2'],
            'normalized_relation_grad': relation_grad_norms['l2'] / (relation_norms['l2'] + EPSILON),
            'full_module_grad_l2': full_norms['l2'],
            'relation_grad_fraction': relation_grad_norms['l2'] / (full_norms['l2'] + EPSILON),
            'relation_param_paths': [list(path) for path in relation_paths],
            'finite': bool(all(np.all(np.isfinite(np.asarray(value))) for value in relation_grads)),
        }
        if not gradients[slot_name]['finite']:
            raise M20ADiagnosticError(f'Non-finite {slot_name} RelationAugmenter gradient')
    after = {
        'params': _pytree_fingerprint(agent.network.params),
        'model_state': _pytree_fingerprint(agent.network.model_state),
        'opt_state': _pytree_fingerprint(agent.network.opt_state),
        'rng': _pytree_fingerprint(agent.rng),
        'step': int(agent.network.step),
    }
    if before != after:
        raise M20ADiagnosticError(
            f'Gradient diagnostic changed agent state: before={before}, after={after}'
        )
    return {'metrics': gradients, 'state_before': before, 'state_after': after}


def _trace_call(agent, slot, observations, goals, actions=None, *, relation_override=None, relation_channel_mask=None):
    args = [observations, goals]
    if actions is not None:
        args.append(actions)
    return agent.network(
        *args,
        name=slot,
        method='relation_utilization_trace',
        relation_override=relation_override,
        relation_channel_mask=relation_channel_mask,
    )


def _zero_relation(trace, *, ensemble):
    return np.zeros_like(_relation_base(trace, ensemble=ensemble))


def _trace_arrays(trace):
    return {
        key: None if value is None else np.asarray(value)
        for key, value in trace.items()
    }


def _normal_forward_parity(agent, batch, traces):
    actor_normal = np.asarray(agent.network.select('actor')(
        batch['observations'], batch['actor_goals'], temperature=0.0,
    ).mode())
    value_normal = np.asarray(agent.network.select('value')(
        batch['observations'], batch['value_goals'],
    ))
    critic_normal = np.asarray(agent.network.select('critic')(
        batch['observations'], batch['value_goals'], batch['actions'],
    ))
    actor_error = float(np.max(np.abs(actor_normal - traces['actor']['all']['action_means'])))
    value_error = float(np.max(np.abs(value_normal - traces['value']['all']['values'])))
    critic_error = float(np.max(np.abs(critic_normal - traces['critic_value']['all']['values'])))
    result = {
        'actor_max_abs_error': actor_error,
        'value_max_abs_error': value_error,
        'critic_max_abs_error': critic_error,
        'tolerance': {'atol': ATOL, 'rtol': RTOL},
        'status': 'pass' if all(
            error <= ATOL + RTOL * max(1.0, float(np.max(np.abs(reference))))
            for error, reference in (
                (actor_error, actor_normal),
                (value_error, value_normal),
                (critic_error, critic_normal),
            )
        ) else 'fail',
    }
    if result['status'] != 'pass':
        raise M20ADiagnosticError(f'Normal-forward parity failed: {result}')
    return result


def _parameter_learning(agent):
    result = {}
    for slot_name in ('actor', 'value', 'critic'):
        params = _module_subtree(agent.network.params, slot_name)
        values, paths = _relation_params(params)
        norms = _norms(values)
        result[slot_name] = {
            'relation_param_l1': norms['l1'],
            'relation_param_l2': norms['l2'],
            'relation_param_count': norms['parameter_count'],
            'initial_parameter_displacement': None,
            'initial_parameter_displacement_status': 'unavailable',
            'initial_parameter_displacement_reason': (
                'The source checkpoint does not contain source initialization, and same-seed '
                'reconstruction was not independently parity-verified.'
            ),
            'relation_param_paths': [list(path) for path in paths],
        }
    return result


def _add_npz_array(destination, name, value):
    if value is None:
        return
    destination[name] = np.asarray(value)


def _save_per_sample_artifact(path, arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite per-sample artifact: {path}')
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _metric_arrays_for_artifact(destination, prefix, metrics):
    for key, value in metrics.items():
        if key in {'mu_R', 'mu_0', 'action_R', 'action_0', 'value_R', 'value_0', 'Q1_R', 'Q2_R', 'Qmin_R', 'Q1_0', 'Q2_0', 'Qmin_0'}:
            _add_npz_array(destination, f'{prefix}_{key}', value)
        elif np.asarray(value).ndim == 1:
            _add_npz_array(destination, f'{prefix}_{key}', value)


def _run_checkpoint(job, provenance, batch, batch_metadata, output_root, *, diagnostic_code):
    step = int(job['step'])
    output_dir = (
        Path(output_root).resolve()
        / provenance['diagnostic_namespace']
        / 'relation_utilization'
        / f'checkpoint_{step}'
    )
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint diagnostic output: {output_dir}')
    _assert_source_snapshot(provenance, label='before checkpoint execution')
    planned_sha = job['sha256']
    source_hash_before, stat_before = _stable_sha256(job['path'])
    _require_equal(source_hash_before, planned_sha, f'checkpoint {step} pre-restore SHA256')
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / 'm20ad_metadata.json'
    metadata = {
        'status': 'running',
        'diagnostic_id': provenance['diagnostic_campaign_id'],
        'diagnostic_namespace': provenance['diagnostic_namespace'],
        'artifact_schema': DIAGNOSTIC_SCHEMA,
        'diagnostic_code_head': diagnostic_code.get('head'),
        'diagnostic_code_worktree_dirty': diagnostic_code.get('dirty'),
        'source_study': provenance['source_study_id'],
        'source_config': provenance['source_config_id'],
        'source_run_path': provenance['source_run_path'],
        'source_run_status_at_diagnostic': provenance['source_status'],
        'recorded_runtime_status': provenance['recorded_runtime_status'],
        'observed_process_state': provenance['observed_process_state'],
        'observed_last_training_step': provenance['observed_last_training_step'],
        'termination_context': provenance['termination_context'],
        'source_runtime_metadata_artifact': provenance.get('runtime_metadata_artifact'),
        'source_resolved_config_artifact': provenance.get('resolved_config_artifact'),
        'source_train_artifact': provenance.get('train_artifact'),
        'source_eval_artifact': provenance.get('eval_artifact'),
        'source_failure_artifact': provenance.get('failure_artifact'),
        'source_git_commit': provenance['source_git_commit'],
        'source_readout': provenance['source_readout'],
        'attention_status': (
            'not_applicable_mean_readout'
            if provenance['source_readout'] == 'mean_context'
            else 'applicable_hybrid_context_query'
        ),
        'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'source_checkpoint_step': step,
        'source_checkpoint_filename_step': job.get('filename_step', step),
        'source_checkpoint_embedded_step': job.get('embedded_checkpoint_step', step),
        'source_checkpoint_role': job.get('checkpoint_role'),
        'source_checkpoint_selection_kind': job.get('selection_kind'),
        'source_checkpoint_embedded_metadata': job.get('checkpoint_metadata'),
        'source_checkpoint_selector_aliases': job['selector_aliases'],
        'source_checkpoint_selection': job['selection'],
        'source_checkpoint_path': job['path'],
        'source_checkpoint_sha256': planned_sha,
        'source_checkpoint_hash_before': source_hash_before,
        'source_checkpoint_file_stat_before': stat_before,
        'fixed_batch_metadata_path': str(batch_metadata['metadata_path']),
        'fixed_batch_path': str(batch_metadata['batch_path']),
        'fixed_batch_fingerprint_sha256': batch_metadata['batch_fingerprint_sha256'],
        'fixed_batch_source_config': batch_metadata.get('source_config'),
        'fixed_batch_source_run_path': batch_metadata.get('source_run_path'),
        'batch_size': int(batch['sample_id'].shape[0]),
        'diagnostic_seed': int(batch_metadata['diagnostic_seed']),
        'interventions': {
            'primary': 'same checkpoint/input/parameters; Correct relation vs zeros_like(Correct relation)',
            'channel_masks': {
                'ALL': [1, 1, 1],
                'DROP_CURRENT': [0, 1, 1],
                'DROP_GOAL_SUPPORT': [1, 0, 1],
                'DROP_GOAL_CONFLICT': [1, 1, 0],
                'ZERO': [0, 0, 0],
            },
        },
        'atol': ATOL,
        'rtol': RTOL,
        'epsilon': EPSILON,
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
        'parameter_displacement_status': 'unavailable',
    }
    _write_json(metadata_path, metadata, overwrite=True)
    try:
        jax_batch = {
            key: jax.numpy.asarray(value)
            for key, value in batch.items()
            if key in _REQUIRED_BATCH_FIELDS
            and key not in {'sample_id', 'transition_indices', 'actor_goal_indices', 'value_goal_indices'}
        }
        initial_agent = GCIQLAgent.create(
            provenance['source_training_seed'],
            jax_batch['observations'][:1],
            jax_batch['actions'][:1],
            provenance['agent_config'],
        )
        restored = restore_agent_from_checkpoint(initial_agent, job['path'])
        with Path(job['path']).open('rb') as file:
            payload = pickle.load(file)
        checkpoint_metadata = payload.get('checkpoint_metadata') or {}
        if not isinstance(checkpoint_metadata, Mapping):
            raise M20ADiagnosticError(
                f'Checkpoint {step} has no structured checkpoint_metadata mapping'
            )
        for key, expected in (
            ('environment', provenance['source_environment']),
            ('study_id', provenance['source_study_id']),
            ('config_id', provenance['source_config_id']),
            ('config_slug', provenance['source_config_slug']),
            ('git_commit', provenance['source_git_commit']),
            ('seed', provenance['source_training_seed']),
            ('checkpoint_step', step),
        ):
            _require_equal(checkpoint_metadata.get(key), expected, f'checkpoint metadata {key}')
        parameter_count = int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(restored.network.params)))
        initial_parameter_count = int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(initial_agent.network.params)))
        if parameter_count != initial_parameter_count:
            raise M20ADiagnosticError(
                f'Checkpoint restore changed parameter count: {initial_parameter_count} -> {parameter_count}'
            )
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(restored.network.params)):
            raise M20ADiagnosticError('Restored checkpoint has non-finite parameters')
        state_before = {
            'network_params_fingerprint': _pytree_fingerprint(restored.network.params),
            'network_model_state_fingerprint': _pytree_fingerprint(restored.network.model_state),
            'optimizer_state_fingerprint': _pytree_fingerprint(restored.network.opt_state),
            'agent_rng_fingerprint': _pytree_fingerprint(restored.rng),
            'network_step': int(restored.network.step),
        }

        slot_inputs = {
            'actor': (jax_batch['observations'], jax_batch['actor_goals'], None, False),
            'value': (jax_batch['observations'], jax_batch['value_goals'], None, False),
            'critic_value': (jax_batch['observations'], jax_batch['value_goals'], jax_batch['actions'], True),
            'critic_actor': (jax_batch['observations'], jax_batch['actor_goals'], None, True),
        }
        # The first Correct actor pass supplies the fixed actor-loss action
        # reference.  No sampled action enters this diagnostic.
        actor_all = _trace_arrays(_trace_call(
            restored, 'actor', *slot_inputs['actor'][:2],
        ))
        actor_relation = _relation_base(actor_all, ensemble=False)
        actor_zero = _trace_arrays(_trace_call(
            restored, 'actor', *slot_inputs['actor'][:2],
            relation_override=np.zeros_like(actor_relation),
        ))
        value_all = _trace_arrays(_trace_call(
            restored, 'value', *slot_inputs['value'][:2],
        ))
        value_relation = _relation_base(value_all, ensemble=False)
        value_zero = _trace_arrays(_trace_call(
            restored, 'value', *slot_inputs['value'][:2],
            relation_override=np.zeros_like(value_relation),
        ))
        critic_value_all = _trace_arrays(_trace_call(
            restored, 'critic', *slot_inputs['critic_value'][:3],
        ))
        critic_value_relation = _relation_base(critic_value_all, ensemble=True)
        critic_value_zero = _trace_arrays(_trace_call(
            restored, 'critic', *slot_inputs['critic_value'][:3],
            relation_override=np.zeros_like(critic_value_relation),
        ))
        actor_reference_action = np.clip(actor_all['action_means'], -1.0, 1.0)
        critic_actor_all = _trace_arrays(_trace_call(
            restored, 'critic', jax_batch['observations'], jax_batch['actor_goals'],
            jax.numpy.asarray(actor_reference_action),
        ))
        critic_actor_relation = _relation_base(critic_actor_all, ensemble=True)
        critic_actor_zero = _trace_arrays(_trace_call(
            restored, 'critic', jax_batch['observations'], jax_batch['actor_goals'],
            jax.numpy.asarray(actor_reference_action),
            relation_override=np.zeros_like(critic_actor_relation),
        ))

        traces = {
            'actor': {'all': actor_all, 'zero': actor_zero},
            'value': {'all': value_all, 'zero': value_zero},
            'critic_value': {'all': critic_value_all, 'zero': critic_value_zero},
            'critic_actor': {'all': critic_actor_all, 'zero': critic_actor_zero},
        }
        relations = {
            'actor': actor_relation,
            'value': value_relation,
            'critic_value': critic_value_relation,
            'critic_actor': critic_actor_relation,
        }
        groups = {}
        coverage = {}
        for input_name, relation in relations.items():
            _, groups[input_name] = _activity_groups(relation)
            coverage[input_name] = _coverage(relation, groups[input_name])
            _negative_control(
                traces[input_name]['all'], traces[input_name]['zero'],
                {'inactive': groups[input_name]['inactive']},
                ensemble=input_name.startswith('critic'), label=f'{input_name} Correct/Zero',
            )

        primary = {
            'actor': _primary_metrics(actor_all, actor_zero, 'actor', ensemble=False),
            'value': _primary_metrics(value_all, value_zero, 'value', ensemble=False),
            'critic_value': _primary_metrics(critic_value_all, critic_value_zero, 'critic', ensemble=True),
            'critic_actor': _primary_metrics(critic_actor_all, critic_actor_zero, 'critic', ensemble=True),
        }

        channel_masks = {
            'current_support': np.asarray([0, 1, 1], dtype=np.float32),
            'goal_support': np.asarray([1, 0, 1], dtype=np.float32),
            'goal_conflict': np.asarray([1, 1, 0], dtype=np.float32),
        }
        channel_traces = {}
        channel_results = {}
        channel_negative_controls = {}
        for input_name, slot_name, input_values, ensemble in (
            ('actor', 'actor', slot_inputs['actor'], False),
            ('value', 'value', slot_inputs['value'], False),
            ('critic_value', 'critic', slot_inputs['critic_value'], True),
            ('critic_actor', 'critic', slot_inputs['critic_actor'], True),
        ):
            channel_traces[input_name] = {}
            channel_results[input_name] = {}
            channel_negative_controls[input_name] = {}
            for channel_name, channel_mask in channel_masks.items():
                args = list(input_values[:3])
                if slot_name == 'critic' and args[2] is None:
                    # The critic-actor path must use the same fixed actor
                    # action as its Correct/Zero intervention pair.
                    args[2] = jax.numpy.asarray(actor_reference_action)
                dropped = _trace_arrays(_trace_call(
                    restored, slot_name, *args, relation_channel_mask=channel_mask,
                ))
                channel_traces[input_name][channel_name] = dropped
                channel_result = _channel_metrics(
                    traces[input_name]['all'], dropped, slot_name, ensemble=ensemble,
                )
                channel_results[input_name][channel_name] = channel_result
                inactive_mask = groups[input_name][f'{channel_name}_active'] == 0
                channel_negative_controls[input_name][channel_name] = _negative_control(
                    traces[input_name]['all'], dropped,
                    {'inactive_channel': inactive_mask},
                    ensemble=ensemble,
                    label=f'{input_name} {channel_name} ALL/DROP',
                )

        parity = _normal_forward_parity(
            restored,
            jax_batch,
            {
                'actor': {'all': actor_all},
                'value': {'all': value_all},
                'critic_value': {'all': critic_value_all},
            },
        )
        gradients = _gradient_metrics(restored, jax_batch)
        parameter_learning = _parameter_learning(restored)

        per_sample = {'sample_id': np.asarray(batch['sample_id'], dtype=np.int64)}
        for input_name, relation in relations.items():
            counts, group_values = _activity_groups(relation)
            _add_npz_array(per_sample, f'{input_name}_relation_counts', counts)
            for group_name, mask in group_values.items():
                _add_npz_array(per_sample, f'{input_name}_group_{group_name}', mask.astype(np.uint8))
        for input_name, slot_name, ensemble in (
            ('actor', 'actor', False), ('value', 'value', False),
            ('critic_value', 'critic', True), ('critic_actor', 'critic', True),
        ):
            metrics = primary[input_name]
            _metric_arrays_for_artifact(per_sample, input_name, metrics)
            _add_npz_array(per_sample, f'{input_name}_attention_R', traces[input_name]['all']['attention_weights'])
            _add_npz_array(per_sample, f'{input_name}_attention_0', traces[input_name]['zero']['attention_weights'])
        for input_name, channels in channel_results.items():
            for channel_name, metrics in channels.items():
                for metric_name, values in metrics.items():
                    _add_npz_array(per_sample, f'{input_name}_channel_{channel_name}_{metric_name}', values)
        per_sample_path = output_dir / 'per_sample_metrics.npz'
        _save_per_sample_artifact(per_sample_path, per_sample)

        source_hash_after, stat_after = _stable_sha256(job['path'])
        _require_equal(source_hash_after, source_hash_before, f'checkpoint {step} post-diagnostic SHA256')
        _require_equal(stat_after, stat_before, f'checkpoint {step} post-diagnostic file stat')
        source_snapshot_after = _assert_source_snapshot(provenance, label='after checkpoint execution')
        state_after = {
            'network_params_fingerprint': _pytree_fingerprint(restored.network.params),
            'network_model_state_fingerprint': _pytree_fingerprint(restored.network.model_state),
            'optimizer_state_fingerprint': _pytree_fingerprint(restored.network.opt_state),
            'agent_rng_fingerprint': _pytree_fingerprint(restored.rng),
            'network_step': int(restored.network.step),
        }
        if state_before != state_after:
            raise M20ADiagnosticError(
                f'Forward/gradient diagnostics changed restored network state: {state_before} -> {state_after}'
            )

        summary = {
            'status': 'completed',
            'diagnostic_id': provenance['diagnostic_campaign_id'],
            'diagnostic_namespace': provenance['diagnostic_namespace'],
            'artifact_schema': DIAGNOSTIC_SCHEMA,
            'source_config': provenance['source_config_id'],
            'source_readout': provenance['source_readout'],
            'attention_status': (
                'not_applicable_mean_readout'
                if provenance['source_readout'] == 'mean_context'
                else 'applicable_hybrid_context_query'
            ),
            'source_checkpoint_step': step,
            'source_checkpoint_sha256': source_hash_after,
            'source_checkpoint_hash_before': source_hash_before,
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_file_stat_before': stat_before,
            'source_checkpoint_file_stat_after': stat_after,
            'source_checkpoint_immutable': True,
            'source_run_immutable': source_snapshot_after == provenance['source_file_snapshot'],
            'parameter_count': parameter_count,
            'network_step_before': state_before['network_step'],
            'network_step_after': state_after['network_step'],
            'optimizer_updates': 0,
            'evaluation_only': True,
            'finetuning': False,
            'fixed_batch_fingerprint_sha256': batch_metadata['batch_fingerprint_sha256'],
            'fixed_batch_source_config': batch_metadata.get('source_config'),
            'fixed_batch_source_run_path': batch_metadata.get('source_run_path'),
            'attention_status_by_input': {
                input_name: primary[input_name].get('attention_status')
                for input_name in primary
            },
            'channel_attention_status_by_input': {
                input_name: {
                    channel_name: metrics.get('attention_status')
                    for channel_name, metrics in channels.items()
                }
                for input_name, channels in channel_results.items()
            },
            'normal_forward_parity': parity,
            'coverage': coverage,
            'primary_metrics': {
                input_name: _metric_summaries(primary[input_name], groups[input_name])
                for input_name in primary
            },
            'negative_controls': {
                input_name: _negative_control(
                    traces[input_name]['all'], traces[input_name]['zero'],
                    {'inactive': groups[input_name]['inactive']},
                    ensemble=input_name.startswith('critic'), label=f'{input_name} Correct/Zero',
                )
                for input_name in traces
            },
            'channel_ablation': {
                input_name: {
                    channel_name: {
                        'metrics': _metric_summaries(
                            metrics,
                            {'any_relation_active': groups[input_name]['any_relation_active']},
                        )['any_relation_active'],
                        'active_channel_metrics': _metric_summaries(
                            metrics,
                            {f'{channel_name}_active': groups[input_name][f'{channel_name}_active']},
                        )[f'{channel_name}_active'],
                    }
                    for channel_name, metrics in channels.items()
                }
                for input_name, channels in channel_results.items()
            },
            'channel_negative_controls': channel_negative_controls,
            'parameter_learning': parameter_learning,
            'gradient_analysis': gradients,
            'state_integrity': {
                'before': state_before,
                'after': state_after,
                'unchanged': state_before == state_after,
            },
            'artifact_paths': {
                'per_sample_metrics': str(per_sample_path),
                'metadata': str(metadata_path),
            },
        }
        _write_json(output_dir / 'diagnostic_summary.json', summary)
        metadata.update({
            'status': 'completed',
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_file_stat_after': stat_after,
            'source_checkpoint_immutable': True,
            'source_run_immutable': True,
            'network_step_before': state_before['network_step'],
            'network_step_after': state_after['network_step'],
            'network_params_fingerprint_before': state_before['network_params_fingerprint'],
            'network_params_fingerprint_after': state_after['network_params_fingerprint'],
            'optimizer_state_unchanged': state_before['optimizer_state_fingerprint'] == state_after['optimizer_state_fingerprint'],
            'agent_rng_unchanged': state_before['agent_rng_fingerprint'] == state_after['agent_rng_fingerprint'],
            'artifact_paths': summary['artifact_paths'],
        })
        _write_json(metadata_path, metadata, overwrite=True)
        return summary
    except BaseException as error:
        try:
            after_sha, after_stat = _stable_sha256(job['path'])
            metadata['source_checkpoint_hash_after'] = after_sha
            metadata['source_checkpoint_file_stat_after'] = after_stat
            metadata['source_checkpoint_immutable'] = after_sha == source_hash_before and after_stat == stat_before
        except BaseException as hash_error:
            metadata['source_checkpoint_hash_after_error'] = f'{type(hash_error).__name__}: {hash_error}'
        try:
            source_snapshot_after = _file_snapshot(provenance['source_run_path'])
            metadata['source_run_immutable'] = source_snapshot_after == provenance['source_file_snapshot']
        except BaseException as snapshot_error:
            metadata['source_run_immutable_error'] = (
                f'{type(snapshot_error).__name__}: {snapshot_error}'
            )
        if 'state_before' in locals():
            metadata['network_state_before_failure'] = state_before
        metadata.update({
            'status': 'failed',
            'failure_reason': f'{type(error).__name__}: {error}',
            'optimizer_updates': 0,
            'evaluation_only': True,
            'finetuning': False,
        })
        _write_json(metadata_path, metadata, overwrite=True)
        raise


def plan_diagnostic(
    *,
    source_run,
    study_path,
    source_config=SOURCE_CONFIG_ID,
    dataset_root,
    output_root,
    selectors,
    batch_size,
    diagnostic_seed,
):
    provenance = validate_source_run(
        source_run,
        study_path=study_path,
        dataset_root=dataset_root,
        source_config=source_config,
    )
    output_root = Path(output_root).resolve()
    if _under(output_root, provenance['source_run_path']) or _under(provenance['source_run_path'], output_root):
        raise M20ADiagnosticError(
            f'Diagnostic output root must be independent of source run: {output_root}'
        )
    checkpoint_plan = select_checkpoints(provenance['numeric_checkpoints'], selectors)
    checkpoint_plan = add_semantic_last_fallback(
        checkpoint_plan, provenance.get('semantic_last_fallback'),
    )
    batch_root, batch_path, batch_metadata_path = _fixed_batch_paths(
        output_root, batch_size, diagnostic_seed,
    )
    if batch_path.exists() != batch_metadata_path.exists():
        batch_status = 'collision_incomplete'
    elif batch_path.is_file() and batch_metadata_path.is_file():
        batch_status = 'existing_validate'
    else:
        batch_status = 'planned_create'
    jobs = []
    for record in checkpoint_plan['selected']:
        output_dir = (
            output_root
            / provenance['diagnostic_namespace']
            / 'relation_utilization'
            / f'checkpoint_{int(record["step"])}'
        )
        jobs.append({
            **record,
            'output_dir': str(output_dir),
            'status': 'output_exists' if output_dir.exists() else 'planned',
        })
    return {
        'diagnostic_id': provenance['diagnostic_campaign_id'],
        'source': provenance,
        'checkpoint_plan': checkpoint_plan,
        'batch_plan': {
            'root': str(batch_root),
            'path': str(batch_path),
            'metadata_path': str(batch_metadata_path),
            'batch_size': int(batch_size),
            'diagnostic_seed': int(diagnostic_seed),
            'status': batch_status,
        },
        'jobs': jobs,
        'output_root': str(output_root),
    }


def _write_execution_manifest(path, plan, *, status, summaries=None, error=None, overwrite=False):
    source = plan['source']
    payload = {
        'status': status,
        'diagnostic_id': source['diagnostic_campaign_id'],
        'diagnostic_namespace': source['diagnostic_namespace'],
        'artifact_schema': f'{DIAGNOSTIC_SCHEMA}_campaign_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_study': source['source_study_id'],
        'source_config': source['source_config_id'],
        'source_run_path': source['source_run_path'],
        'source_run_status': source['source_status'],
        'recorded_runtime_status': source['recorded_runtime_status'],
        'observed_process_state': source['observed_process_state'],
        'observed_last_training_step': source['observed_last_training_step'],
        'termination_context': source['termination_context'],
        'source_failure_artifact': source.get('failure_artifact'),
        'source_train_artifact': source.get('train_artifact'),
        'source_eval_artifact': source.get('eval_artifact'),
        'source_runtime_metadata_path': str(
            Path(source['source_run_path']) / 'runtime_metadata.json'
        ),
        'source_resolved_config_path': str(
            Path(source['source_run_path']) / 'resolved_config.json'
        ),
        'source_path_resolution': source['source_path_resolution'],
        'source_git_commit': source['source_git_commit'],
        'source_resolved_config_fingerprint': source['source_resolved_config_fingerprint'],
        'source_readout': source['source_readout'],
        'attention_status': (
            'not_applicable_mean_readout'
            if source['source_readout'] == 'mean_context'
            else 'applicable_hybrid_context_query'
        ),
        'checkpoint_discovery_code_path': source['checkpoint_discovery_code_path'],
        'checkpoint_audit': source['checkpoint_audit'],
        'latest_recoverable_checkpoint_step': source['latest_recoverable_checkpoint_step'],
        'source_run_immutable_after_fixed_batch': source.get(
            'source_run_immutable_after_fixed_batch'
        ),
        'source_checkpoint_plan': plan['checkpoint_plan'],
        'batch_plan': plan['batch_plan'],
        'jobs': [
            {key: value for key, value in job.items() if key != 'path'}
            for job in plan['jobs']
        ],
        'summaries': summaries or [],
        'missing_requested_checkpoints': plan['checkpoint_plan']['missing'],
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
        'error': error,
    }
    _write_json(path, payload, overwrite=overwrite)


def execute_plan(plan):
    if plan['batch_plan']['status'] == 'collision_incomplete':
        raise M20ADiagnosticError('Fixed batch output has only one of .npz/metadata; refusing repair')
    output_exists = [job for job in plan['jobs'] if job['status'] == 'output_exists']
    if output_exists:
        raise FileExistsError(
            'Diagnostic output collision: ' + ', '.join(job['output_dir'] for job in output_exists)
        )
    summary_dir = (
        Path(plan['output_root'])
        / plan['source']['diagnostic_namespace']
        / 'summary'
    )
    manifest_path = summary_dir / 'execution_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite execution manifest: {manifest_path}')
    batch_arrays, batch_metadata, batch_location = create_or_load_fixed_batch(
        plan['output_root'], plan['source'],
        batch_size=plan['batch_plan']['batch_size'],
        diagnostic_seed=plan['batch_plan']['diagnostic_seed'],
    )
    batch_metadata = dict(batch_metadata)
    batch_metadata.update({
        'path': str(_fixed_batch_paths(
            plan['output_root'], plan['batch_plan']['batch_size'], plan['batch_plan']['diagnostic_seed'],
        )[1]),
        'metadata_path': str(_fixed_batch_paths(
            plan['output_root'], plan['batch_plan']['batch_size'], plan['batch_plan']['diagnostic_seed'],
        )[2]),
        'batch_path': str(_fixed_batch_paths(
            plan['output_root'], plan['batch_plan']['batch_size'], plan['batch_plan']['diagnostic_seed'],
        )[1]),
        'batch_fingerprint_sha256': batch_metadata['batch_fingerprint_sha256'],
    })
    reproducibility = verify_fixed_batch_reproducibility(
        plan['source'],
        batch_arrays,
        batch_size=plan['batch_plan']['batch_size'],
        diagnostic_seed=plan['batch_plan']['diagnostic_seed'],
    )
    batch_metadata['reproducibility_check'] = reproducibility
    _write_json(batch_metadata['metadata_path'], batch_metadata, overwrite=True)
    plan['batch_plan']['status'] = (
        'created_and_reproducibility_checked'
        if batch_location['created'] else 'reused_and_reproducibility_checked'
    )
    source_snapshot_after_batch = _assert_source_snapshot(
        plan['source'], label='after fixed batch freeze'
    )
    plan['source']['source_run_immutable_after_fixed_batch'] = (
        source_snapshot_after_batch == plan['source']['source_file_snapshot']
    )
    diagnostic_code = _git_info()
    summary_dir.mkdir(parents=True, exist_ok=False)
    if not plan['jobs']:
        _write_execution_manifest(
            manifest_path,
            plan,
            status='blocked_missing_stable_numeric_checkpoint',
            error=(
                'No stable requested numeric checkpoint is available; '
                'fixed batch was frozen, but no model restore was attempted'
            ),
        )
        return []
    _write_execution_manifest(manifest_path, plan, status='running')
    summaries = []
    try:
        for job in plan['jobs']:
            summaries.append(_run_checkpoint(
                job,
                plan['source'],
                batch_arrays,
                batch_metadata,
                plan['output_root'],
                diagnostic_code=diagnostic_code,
            ))
        _write_execution_manifest(manifest_path, plan, status='completed', summaries=summaries, overwrite=True)
        return summaries
    except BaseException as error:
        _write_execution_manifest(
            manifest_path, plan, status='failed', summaries=summaries,
            error=f'{type(error).__name__}: {error}', overwrite=True,
        )
        raise


def _print_plan(plan):
    source = plan['source']
    checkpoint_plan = plan['checkpoint_plan']
    print(
        f'{source["diagnostic_campaign_id"]} source study/config: '
        f'{source["source_study_id"]}/{source["source_config_id"]}'
    )
    print(f'source run: {source["source_run_path"]}')
    print(f'authoritative source run: {source["source_path_resolution"]["authoritative_path"]}')
    print(f'recorded runtime status: {source["recorded_runtime_status"]}')
    print(f'observed process state: {source["observed_process_state"]}')
    print(f'observed last training step: {source["observed_last_training_step"]}')
    print(f'source readout: {source["source_readout"]}')
    print(
        'attention status: '
        f'{"not_applicable_mean_readout" if source["source_readout"] == "mean_context" else "applicable_hybrid_context_query"}'
    )
    print(f'termination context: {source["termination_context"]}')
    print(f'source commit: {source["source_git_commit"]}')
    print(f'resolved config fingerprint: {source["source_resolved_config_fingerprint"]}')
    print(f'dataset: {source["dataset_identity"]["path"]}')
    print(f'dataset SHA256: {source["dataset_identity"]["sha256"]}')
    print(f'checkpoint discovery code: {source["checkpoint_discovery_code_path"]}')
    for record in source['checkpoint_audit']['numeric_checkpoints']:
        print(
            f'checkpoint audit: path={record["path"]} '
            f'filename_step={record.get("filename_step")} '
            f'embedded_step={record.get("embedded_checkpoint_step")} '
            f'status={record["status"]} sha256={record.get("sha256")} '
            f'error={record.get("error")}'
        )
    print(f'discovered numeric checkpoints: {checkpoint_plan["discovered_numeric_steps"]}')
    print(f'latest stable numeric: {checkpoint_plan["latest_stable_numeric_step"]}')
    print(f'latest recoverable checkpoint: {source["latest_recoverable_checkpoint_step"]}')
    print(
        'semantic last fallback: '
        f'{checkpoint_plan.get("semantic_last_fallback")}'
    )
    print(f'requested selectors: {checkpoint_plan["requested_selectors"]}')
    print(f'missing selectors: {checkpoint_plan["missing"]}')
    print(
        f'fixed batch: N={plan["batch_plan"]["batch_size"]} '
        f'seed={plan["batch_plan"]["diagnostic_seed"]} '
        f'status={plan["batch_plan"]["status"]} '
        f'path={plan["batch_plan"]["path"]}'
    )
    for job in plan['jobs']:
        print(
            f'[{job["status"]}] checkpoint={job["step"]} '
            f'sha256={job.get("sha256")} output={job["output_dir"]}'
        )
    if not plan['jobs']:
        print('formal execution status: blocked_missing_stable_numeric_checkpoint')
    else:
        print(f'planned formal jobs: {len(plan["jobs"])}')


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-run',
        default=None,
        help='Optional explicit source path; it is checked against make_run_path.',
    )
    parser.add_argument('--study', default=DEFAULT_STUDY)
    parser.add_argument(
        '--source-config',
        default=DEFAULT_SOURCE_CONFIG_ID,
        help='Authoritative study configuration ID; default preserves the Q campaign.',
    )
    parser.add_argument('--checkpoint-steps', default=','.join(map(str, DEFAULT_CHECKPOINT_STEPS)))
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--diagnostic-seed', type=int, default=DEFAULT_DIAGNOSTIC_SEED)
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.execute == args.dry_run:
        parser.error('exactly one of --execute or --dry-run is required')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if args.diagnostic_seed < 0:
        parser.error('--diagnostic-seed must be non-negative')
    return args


def main(argv=None):
    args = _args(argv)
    try:
        selectors = parse_checkpoint_selectors(args.checkpoint_steps)
        source_run = args.source_run or str(
            resolve_authoritative_source_run(
                args.study, config_id=args.source_config,
            )
        )
        plan = plan_diagnostic(
            source_run=source_run,
            study_path=args.study,
            source_config=args.source_config,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            selectors=selectors,
            batch_size=args.batch_size,
            diagnostic_seed=args.diagnostic_seed,
        )
        _print_plan(plan)
        if args.dry_run:
            if plan['batch_plan']['status'] == 'collision_incomplete':
                return 2
            if any(job['status'] == 'output_exists' for job in plan['jobs']):
                return 2
            return 0
        summaries = execute_plan(plan)
        if not plan['jobs']:
            print(
                f'{_campaign_id(args.source_config)} formal execution blocked after fixed-batch freeze: '
                'blocked_missing_stable_numeric_checkpoint',
                file=sys.stderr,
            )
            return 2
        print(
            f'{_campaign_id(args.source_config)} formal execution completed '
            'with optimizer_updates=0'
        )
        return 0
    except (FileExistsError, FileNotFoundError, OSError, M20ADiagnosticError, ValueError) as error:
        print(f'{_campaign_id(args.source_config)}: FAIL: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
