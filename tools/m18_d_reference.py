"""Immutable M18-D reference-artifact contract for the final supplements.

The original D1/D2-D4 runs selected semantic ``best`` checkpoints at the time
they were executed.  Later diagnostics must not re-resolve a mutable current
``best`` pointer.  This module instead reads those completed artifacts, checks
their mutual provenance, and restores only the exact checkpoint path/SHA they
recorded.

It is intentionally diagnostic-only and never writes source runs, checkpoint
indices, or semantic checkpoint metadata.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from impls.experiment import make_run_path
from impls.experiment.management import config_fingerprint, jsonable
from impls.experiment.reevaluation import (
    ReevaluationError,
    _resolved_agent_config,
    _resolved_payload,
    _split_config_identity,
)
from impls.utils.checkpointing import sha256_file
from tools import m18_cross_k_eval as d1
from tools import m18_trace_diagnostics as trace


STUDY_ID = 'M18'
ENVIRONMENT = 'puzzle-4x4-play-v0'
DIAGNOSTIC_ID_D1 = 'M18-D1'
DIAGNOSTIC_ID_TRACE = 'M18-D234'
LOCKED_REFERENCE_SELECTOR = 'locked-reference'
DEFAULT_REFERENCE_BATCH_SIZE = 1024
DEFAULT_REFERENCE_DIAGNOSTIC_SEED = 18018
REQUIRED_BATCH_FIELDS = {
    'sample_id', 'sample_indices', 'observations', 'actor_goals', 'value_goals', 'dataset_actions',
}


def _read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReevaluationError(f'Cannot read JSON artifact {path}: {error}') from error
    if not isinstance(value, Mapping):
        raise ReevaluationError(f'Expected JSON mapping in {path}')
    return dict(value)


def _m18d_root(reference_diagnostics_root):
    root = Path(reference_diagnostics_root).resolve()
    return root if root.name == 'M18D' else root / 'M18D'


def _same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def _under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _require_equal(observed, expected, label):
    if observed != expected:
        raise ReevaluationError(f'{label}: expected {expected!r}, got {observed!r}')


def _require_int(value, label):
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ReevaluationError(f'{label} must be an integer, got {value!r}') from error


def stable_checkpoint_sha256(path):
    """Hash a checkpoint twice and reject a source being modified concurrently."""

    return d1._stable_checkpoint_sha256(path)


def _find_exactly_one(paths, *, label):
    paths = list(paths)
    if len(paths) != 1:
        raise ReevaluationError(f'{label}: expected exactly one artifact, found {len(paths)}: {paths!r}')
    return paths[0]


def _d1_reference(root, train_k):
    candidates = []
    search_root = root / 'cross_k' / 'checkpoint_best'
    if not search_root.is_dir():
        raise ReevaluationError(f'Missing D1 reference root: {search_root}')
    for summary_path in sorted(search_root.rglob('summary.json')):
        summary = _read_json(summary_path)
        if (
            summary.get('status') == 'completed'
            and summary.get('diagnostic_id') == DIAGNOSTIC_ID_D1
            and _require_int(summary.get('K_train'), f'{summary_path}.K_train') == int(train_k)
            and _require_int(summary.get('K_actor_test'), f'{summary_path}.K_actor_test') == int(train_k)
        ):
            candidates.append((summary_path, summary))
    summary_path, summary = _find_exactly_one(candidates, label=f'D1 self-depth K{int(train_k)}')
    metadata_path = summary_path.with_name('m18d_metadata.json')
    metadata = _read_json(metadata_path)
    if metadata.get('status') != 'completed' or metadata.get('diagnostic_id') != DIAGNOSTIC_ID_D1:
        raise ReevaluationError(f'D1 metadata is not a completed M18-D1 artifact: {metadata_path}')
    _require_equal(summary.get('checkpoint_role'), 'best', f'{summary_path}.checkpoint_role')
    _require_equal(metadata.get('source_checkpoint_role'), 'best', f'{metadata_path}.source_checkpoint_role')
    for key, metadata_key in (
        ('checkpoint_sha256', 'source_checkpoint_sha256'),
        ('checkpoint_step', 'source_checkpoint_step'),
        ('source_config_id', 'source_config_id'),
    ):
        if summary.get(key) is not None:
            _require_equal(summary.get(key), metadata.get(metadata_key), f'D1 summary/metadata {key}')
    return {
        'summary_path': str(summary_path),
        'metadata_path': str(metadata_path),
        'summary': summary,
        'metadata': metadata,
    }


def _trace_reference(root, train_k, *, reference_batch_size, reference_diagnostic_seed):
    batch_name = f'fixed_batch_N{int(reference_batch_size)}_seed{int(reference_diagnostic_seed)}'
    trace_root = root / 'trace' / 'checkpoint_best' / batch_name
    metadata_path = trace_root / f'trainK{int(train_k)}' / 'maxTraceK8' / 'm18d_metadata.json'
    metadata = _read_json(metadata_path)
    if metadata.get('status') != 'completed' or metadata.get('diagnostic_id') != DIAGNOSTIC_ID_TRACE:
        raise ReevaluationError(f'Trace metadata is not a completed M18-D234 artifact: {metadata_path}')
    _require_equal(_require_int(metadata.get('K_train'), f'{metadata_path}.K_train'), int(train_k), 'Trace K_train')
    _require_equal(metadata.get('source_checkpoint_role'), 'best', f'{metadata_path}.source_checkpoint_role')
    actor_path = metadata_path.with_name('actor_metrics.npz')
    if not actor_path.is_file():
        raise ReevaluationError(f'Missing D3 actor action artifact: {actor_path}')
    return {
        'trace_root': str(trace_root),
        'metadata_path': str(metadata_path),
        'metadata': metadata,
        'actor_metrics_path': str(actor_path),
    }


def _load_fixed_batch(trace_root, *, reference_batch_size, reference_diagnostic_seed):
    trace_root = Path(trace_root)
    batch_path = trace_root / 'fixed_batch.npz'
    metadata_path = trace_root / 'fixed_batch_metadata.json'
    metadata = _read_json(metadata_path)
    _require_equal(_require_int(metadata.get('batch_size'), f'{metadata_path}.batch_size'), int(reference_batch_size), 'Fixed batch size')
    _require_equal(
        _require_int(metadata.get('diagnostic_seed'), f'{metadata_path}.diagnostic_seed'),
        int(reference_diagnostic_seed),
        'Fixed batch diagnostic seed',
    )
    _require_equal(metadata.get('environment'), ENVIRONMENT, f'{metadata_path}.environment')
    _require_equal(
        metadata.get('goal_semantics_for_actor_value_critic_trace'),
        'actor_goals',
        f'{metadata_path}.goal semantics',
    )
    if not batch_path.is_file():
        raise ReevaluationError(f'Missing fixed batch artifact: {batch_path}')
    with np.load(batch_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    if set(arrays) != REQUIRED_BATCH_FIELDS:
        raise ReevaluationError(f'Fixed batch fields mismatch: {sorted(arrays)!r}')
    n = int(reference_batch_size)
    if any(value.ndim < 1 or value.shape[0] != n for value in arrays.values()):
        raise ReevaluationError('Fixed batch arrays do not share the required leading dimension')
    fingerprint = trace._array_fingerprint(arrays)
    _require_equal(metadata.get('batch_fingerprint_sha256'), fingerprint, 'Fixed batch fingerprint')
    return arrays, {
        'path': str(batch_path),
        'metadata_path': str(metadata_path),
        'metadata': metadata,
        'fingerprint': fingerprint,
    }


def load_reference_contract(
    reference_diagnostics_root,
    *,
    train_ks=(4, 8),
    reference_batch_size=DEFAULT_REFERENCE_BATCH_SIZE,
    reference_diagnostic_seed=DEFAULT_REFERENCE_DIAGNOSTIC_SEED,
):
    """Read and cross-validate the original D1/D234 checkpoint identities.

    This function deliberately never calls ``resolve_checkpoint(..., 'best')``.
    A D1/D234 SHA mismatch raises before either D5 or D6 can be formally
    planned, preventing a mixed-checkpoint interpretation chain.
    """

    train_ks = tuple(sorted({int(value) for value in train_ks}))
    if train_ks != (4, 8):
        raise ReevaluationError(f'M18-D supplement requires exactly train K=(4, 8), got {train_ks!r}')
    root = _m18d_root(reference_diagnostics_root)
    if not root.is_dir():
        raise ReevaluationError(f'Missing M18D reference diagnostics root: {root}')
    references = {}
    common_batch = None
    for train_k in train_ks:
        d1_reference = _d1_reference(root, train_k)
        trace_reference = _trace_reference(
            root,
            train_k,
            reference_batch_size=reference_batch_size,
            reference_diagnostic_seed=reference_diagnostic_seed,
        )
        d1_summary = d1_reference['summary']
        d1_metadata = d1_reference['metadata']
        trace_metadata = trace_reference['metadata']
        d1_sha = d1_summary.get('checkpoint_sha256')
        trace_sha = trace_metadata.get('source_checkpoint_sha256')
        if not isinstance(d1_sha, str) or not d1_sha:
            raise ReevaluationError(f'D1 reference has no checkpoint SHA256: {d1_reference["summary_path"]}')
        _require_equal(trace_sha, d1_sha, f'K{train_k} D1/D234 checkpoint SHA256')
        d1_step = _require_int(d1_summary.get('checkpoint_step'), f'K{train_k} D1 checkpoint_step')
        trace_step = _require_int(trace_metadata.get('source_checkpoint_step'), f'K{train_k} trace checkpoint_step')
        _require_equal(trace_step, d1_step, f'K{train_k} D1/D234 checkpoint step')
        d1_path = d1_metadata.get('source_checkpoint_path')
        trace_path = trace_metadata.get('source_checkpoint_path')
        if not isinstance(d1_path, str) or not isinstance(trace_path, str):
            raise ReevaluationError(f'K{train_k} reference metadata is missing source_checkpoint_path')
        if not _same_path(d1_path, trace_path):
            raise ReevaluationError(f'K{train_k} D1/D234 source checkpoint paths differ: {d1_path!r} vs {trace_path!r}')
        d1_run = d1_metadata.get('source_run_dir')
        trace_run = trace_metadata.get('source_run_dir')
        if not isinstance(d1_run, str) or not isinstance(trace_run, str) or not _same_path(d1_run, trace_run):
            raise ReevaluationError(f'K{train_k} D1/D234 source run paths differ')
        d1_config = d1_metadata.get('source_config_id')
        trace_config = trace_metadata.get('source_config_id')
        _require_equal(trace_config, d1_config, f'K{train_k} D1/D234 source config')
        actual_sha = stable_checkpoint_sha256(trace_path)
        _require_equal(actual_sha, d1_sha, f'K{train_k} locked checkpoint SHA256')
        arrays, batch = _load_fixed_batch(
            trace_reference['trace_root'],
            reference_batch_size=reference_batch_size,
            reference_diagnostic_seed=reference_diagnostic_seed,
        )
        if common_batch is None:
            common_batch = batch
        else:
            _require_equal(batch['fingerprint'], common_batch['fingerprint'], 'K4/K8 fixed batch fingerprint')
            if not _same_path(batch['path'], common_batch['path']):
                raise ReevaluationError('K4/K8 trace artifacts do not reference one fixed_batch.npz')
            # Keep a direct ordering guard even if fingerprints have already
            # matched, so callers can rely on sample_id semantics explicitly.
            with np.load(common_batch['path'], allow_pickle=False) as loaded:
                _require_equal(
                    bool(np.array_equal(arrays['sample_id'], np.asarray(loaded['sample_id']))),
                    True,
                    'K4/K8 fixed batch sample_id ordering',
                )
        references[train_k] = {
            'K_train': int(train_k),
            'source_config_id': str(trace_config),
            'source_config_slug': str(trace_metadata.get('source_config_slug')),
            'source_run_dir': str(Path(trace_run).resolve()),
            'checkpoint_role': 'best',
            'checkpoint_step': int(trace_step),
            'checkpoint_path': str(Path(trace_path).resolve()),
            'checkpoint_sha256': str(d1_sha),
            'd1_summary_path': d1_reference['summary_path'],
            'd1_metadata_path': d1_reference['metadata_path'],
            'trace_metadata_path': trace_reference['metadata_path'],
            'actor_metrics_path': trace_reference['actor_metrics_path'],
        }
    assert common_batch is not None
    return {
        'reference_m18d_root': str(root),
        'reference_batch_size': int(reference_batch_size),
        'reference_diagnostic_seed': int(reference_diagnostic_seed),
        'fixed_batch_path': common_batch['path'],
        'fixed_batch_metadata_path': common_batch['metadata_path'],
        'fixed_batch_fingerprint_sha256': common_batch['fingerprint'],
        'fixed_batch_metadata': common_batch['metadata'],
        'references': references,
    }


def load_fixed_batch_from_contract(contract, *, max_samples=None):
    """Load the exact stored D234 batch, optionally taking a prefix for smoke."""

    batch_path = Path(contract['fixed_batch_path'])
    with np.load(batch_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    if set(arrays) != REQUIRED_BATCH_FIELDS:
        raise ReevaluationError(f'Fixed batch fields mismatch: {sorted(arrays)!r}')
    fingerprint = trace._array_fingerprint(arrays)
    _require_equal(fingerprint, contract['fixed_batch_fingerprint_sha256'], 'Contract fixed batch fingerprint')
    total = int(arrays['sample_id'].shape[0])
    if max_samples is None:
        count = total
    else:
        count = _require_int(max_samples, 'max_samples')
        if not 1 <= count <= total:
            raise ReevaluationError(f'max_samples must be in [1, {total}], got {count}')
    return {name: value[:count].copy() for name, value in arrays.items()}, int(count)


def _validate_locked_checkpoint_payload(path, *, source_metadata, reference):
    try:
        with Path(path).open('rb') as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError) as error:
        raise ReevaluationError(f'Cannot load locked source checkpoint {path}: {error}') from error
    if not isinstance(payload, Mapping) or 'agent' not in payload:
        raise ReevaluationError(f'Locked checkpoint is not a serialized RLC agent: {path}')
    checkpoint_metadata = payload.get('checkpoint_metadata') or {}
    if not isinstance(checkpoint_metadata, Mapping):
        raise ReevaluationError(f'Locked checkpoint has malformed checkpoint_metadata: {path}')
    for key in ('environment', 'study_id', 'config_id', 'config_slug', 'git_commit'):
        _require_equal(checkpoint_metadata.get(key), source_metadata.get(key), f'Locked checkpoint metadata {key}')
    _require_equal(_require_int(checkpoint_metadata.get('seed'), 'Locked checkpoint metadata seed'), int(source_metadata.get('seed')), 'Locked checkpoint seed')
    _require_equal(checkpoint_metadata.get('checkpoint_role'), 'best', 'Locked checkpoint role')
    _require_equal(
        _require_int(checkpoint_metadata.get('checkpoint_step'), 'Locked checkpoint metadata step'),
        int(reference['checkpoint_step']),
        'Locked checkpoint step',
    )
    return jsonable(checkpoint_metadata)


def locked_provenance(contract, source_run_root, train_k):
    """Return source provenance whose checkpoint is the artifact-locked path.

    Unlike the older D1/D234 planner this does *not* read the current semantic
    checkpoint index.  The currently published best may therefore change
    without silently affecting D5/D6.
    """

    train_k = int(train_k)
    try:
        reference = dict(contract['references'][train_k])
    except (KeyError, TypeError) as error:
        raise ReevaluationError(f'No locked M18-D reference for K={train_k}') from error
    source_run_dir = Path(reference['source_run_dir']).resolve()
    source_root = Path(source_run_root).resolve()
    expected_path = make_run_path(
        source_root,
        STUDY_ID,
        reference['source_config_id'],
        reference['source_config_slug'],
        ENVIRONMENT,
        0,
        run_attempt=0,
    ).resolve()
    if source_run_dir != expected_path or not _under(source_run_dir, source_root):
        raise ReevaluationError(
            'Reference source run does not equal the canonical path under --source-run-root: '
            f'reference={source_run_dir}, expected={expected_path}'
        )
    metadata_path = source_run_dir / 'runtime_metadata.json'
    resolved_path = source_run_dir / 'resolved_config.json'
    source_metadata = _read_json(metadata_path)
    resolved = _read_json(resolved_path)
    if source_metadata.get('status') not in {'completed', 'running'}:
        raise ReevaluationError(f'Locked source run has unsupported status: {source_metadata.get("status")!r}')
    _require_equal(source_metadata.get('git_dirty'), False, 'Locked source git_dirty')
    study_id, config_id, config_slug, environment, training_seed = _split_config_identity(source_run_dir)
    _require_equal(study_id, STUDY_ID, 'Locked source study ID')
    _require_equal(environment, ENVIRONMENT, 'Locked source environment')
    _require_equal(config_id, reference['source_config_id'], 'Locked source config ID')
    _require_equal(config_slug, reference['source_config_slug'], 'Locked source config slug')
    for key, expected in (
        ('study_id', study_id),
        ('config_id', config_id),
        ('config_slug', config_slug),
        ('environment', environment),
        ('seed', training_seed),
    ):
        _require_equal(source_metadata.get(key), expected, f'Locked source metadata {key}')
    if source_metadata.get('run_dir') and not _same_path(source_metadata['run_dir'], source_run_dir):
        raise ReevaluationError('Locked source runtime_metadata.run_dir mismatches the source path')
    stored_fingerprint = source_metadata.get('resolved_config_fingerprint')
    _require_equal(resolved.get('resolved_config_fingerprint'), stored_fingerprint, 'Resolved config fingerprint')
    _require_equal(config_fingerprint(_resolved_payload(resolved)), stored_fingerprint, 'Calculated resolved config fingerprint')
    config = _resolved_agent_config(resolved)
    d1.validate_m18_agent_config(config, train_k, label=f'locked K{train_k} source config')
    _require_equal(d1._uniform_train_k(config, label=f'locked K{train_k} source config'), train_k, 'Locked source K_train')
    checkpoint_path = Path(reference['checkpoint_path']).resolve()
    if not checkpoint_path.is_file() or not _under(checkpoint_path, source_run_dir):
        raise ReevaluationError(f'Locked checkpoint is missing or outside its source run: {checkpoint_path}')
    checkpoint_sha = stable_checkpoint_sha256(checkpoint_path)
    _require_equal(checkpoint_sha, reference['checkpoint_sha256'], f'Locked K{train_k} checkpoint SHA256')
    checkpoint_metadata = _validate_locked_checkpoint_payload(
        checkpoint_path,
        source_metadata=source_metadata,
        reference=reference,
    )
    return {
        'source_run_dir': str(source_run_dir),
        'source_study_id': study_id,
        'source_config_id': config_id,
        'source_config_slug': config_slug,
        'source_environment': environment,
        'source_training_seed': int(training_seed),
        'source_git_commit': source_metadata.get('git_commit'),
        'source_git_dirty': source_metadata.get('git_dirty'),
        'source_run_status_at_validation': source_metadata.get('status'),
        'source_resolved_config_fingerprint': stored_fingerprint,
        'source_metadata': source_metadata,
        'resolved_config': resolved,
        'checkpoint_step': int(reference['checkpoint_step']),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': str(checkpoint_sha),
        'checkpoint_metadata': checkpoint_metadata,
        'requested_checkpoint_selector': {'selector': LOCKED_REFERENCE_SELECTOR},
        'resolved_checkpoint_role': 'best',
        'resolved_checkpoint_step': int(reference['checkpoint_step']),
        'reference_d1_summary_path': reference['d1_summary_path'],
        'reference_d1_metadata_path': reference['d1_metadata_path'],
        'reference_trace_metadata_path': reference['trace_metadata_path'],
        'reference_actor_metrics_path': reference['actor_metrics_path'],
        'reference_checkpoint_identity': {
            'role': 'best',
            'step': int(reference['checkpoint_step']),
            'sha256': str(checkpoint_sha),
            'path': str(checkpoint_path),
        },
    }
