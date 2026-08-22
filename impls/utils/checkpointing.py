"""Pure-file helpers for semantic checkpoint provenance and selection."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from flax.traverse_util import flatten_dict

def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(_jsonable(value), file, indent=2, sort_keys=True)
        file.write('\n')


def _read_json(path):
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ValueError(f'Expected JSON mapping: {path}')
    return dict(value)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(tree):
    """Return a deterministic fingerprint for a parameter/state subtree."""

    digest = hashlib.sha256()
    flat = flatten_dict(tree) if isinstance(tree, Mapping) else {(): tree}
    for path, value in sorted(flat.items(), key=lambda item: tuple(map(str, item[0]))):
        array = np.asarray(value)
        digest.update(repr(tuple(map(str, path))).encode('utf-8'))
        digest.update(str(array.dtype).encode('utf-8'))
        digest.update(repr(tuple(array.shape)).encode('utf-8'))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def parameter_module_key(params, module_name):
    """Resolve a logical module name against ModuleDict parameter naming."""

    for candidate in (module_name, f'modules_{module_name}'):
        if isinstance(params, Mapping) and candidate in params:
            return candidate
    raise ValueError(f'Parameter tree does not contain module {module_name!r}')


def checkpoint_module_fingerprint(checkpoint_path, module_name):
    """Fingerprint one module in a serialized full-agent checkpoint."""

    with Path(checkpoint_path).open('rb') as file:
        payload = pickle.load(file)
    try:
        params = payload['agent']['network']['params']
        module = params[parameter_module_key(params, module_name)]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f'Checkpoint does not contain network.params[{module_name!r}]'
        ) from error
    return tree_fingerprint(module)


def normalize_checkpoint_selector(selector):
    """Normalize ``best``, ``last`` and explicit numeric selectors."""

    if isinstance(selector, Mapping):
        name = selector.get('selector')
        if name == 'step':
            if 'step' not in selector:
                raise ValueError('checkpoint selector=step requires step')
            return {'selector': 'step', 'step': int(selector['step'])}
        if name in {'best', 'last'}:
            extra = set(selector) - {'selector'}
            if extra:
                raise ValueError(f'checkpoint selector {name!r} has unexpected fields: {sorted(extra)}')
            return {'selector': name}
        raise ValueError(f'Unsupported checkpoint selector: {name!r}')
    if isinstance(selector, bool):
        raise ValueError('Boolean is not a checkpoint selector')
    if isinstance(selector, int):
        return {'selector': 'step', 'step': int(selector)}
    if isinstance(selector, str):
        if selector in {'best', 'last'}:
            return {'selector': selector}
        if selector.isdigit():
            return {'selector': 'step', 'step': int(selector)}
    raise ValueError(f'Unsupported checkpoint selector: {selector!r}')


def should_update_best(new_metric, best_metric):
    """Return whether a training-time metric strictly improves the best."""

    return best_metric is None or float(new_metric) > float(best_metric)


def write_checkpoint_metadata(path, metadata):
    _write_json(path, metadata)


def write_checkpoint_index(run_dir, *, best=None, last=None, selection_metric='evaluation/overall_success'):
    """Write the stable run-level semantic checkpoint index."""

    def index_entry(record):
        if record is None:
            return None
        return {
            'step': int(record['checkpoint_step']),
            'metric': record.get('selection_metric_value'),
            'path': record['path'],
            'sha256': record['sha256'],
            'metadata_path': record['metadata_path'],
        }

    payload = {
        'schema_version': 1,
        'selection_metric': selection_metric,
        'best': index_entry(best),
        'last': index_entry(last),
        'best_step': None if best is None else int(best['checkpoint_step']),
        'last_step': None if last is None else int(last['checkpoint_step']),
        'best_equals_last': bool(
            best is not None
            and last is not None
            and int(best['checkpoint_step']) == int(last['checkpoint_step'])
        ),
    }
    path = Path(run_dir) / 'checkpoints' / 'index.json'
    _write_json(path, payload)
    return path


def _run_dir(run_dir):
    run_dir = Path(run_dir).resolve()
    return run_dir.parent if run_dir.name == 'checkpoints' else run_dir


def _semantic_record(run_dir, role, index, checkpoint_path):
    metadata_path = _run_dir(run_dir) / index[role].get(
        'metadata_path', f'checkpoints/{role}/checkpoint.json'
    )
    metadata = _read_json(metadata_path)
    expected_sha = metadata.get('checkpoint_sha256') or index[role].get('sha256')
    actual_sha = sha256_file(checkpoint_path)
    if expected_sha != actual_sha:
        raise ValueError(
            f'Checkpoint SHA256 mismatch for {role}: expected={expected_sha!r}, actual={actual_sha!r}'
        )
    if metadata.get('checkpoint_role') != role:
        raise ValueError(
            f'Checkpoint metadata role mismatch: expected={role!r}, '
            f'observed={metadata.get("checkpoint_role")!r}'
        )
    return {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_step': int(metadata['checkpoint_step']),
        'checkpoint_role': role,
        'checkpoint_sha256': actual_sha,
        'checkpoint_metadata': metadata,
    }


def resolve_checkpoint(run_dir, selector='last', *, load_metadata=True):
    """Resolve a semantic or explicit checkpoint without reading eval.csv."""

    run_dir = _run_dir(run_dir)
    normalized = normalize_checkpoint_selector(selector)
    if normalized['selector'] in {'best', 'last'}:
        index_path = run_dir / 'checkpoints' / 'index.json'
        if not index_path.is_file():
            raise FileNotFoundError(
                f'Checkpoint selector {normalized["selector"]!r} requires {index_path}'
            )
        index = _read_json(index_path)
        role = normalized['selector']
        entry = index.get(role)
        if not isinstance(entry, Mapping):
            raise FileNotFoundError(
                f'No saved {role} checkpoint is recorded in {index_path}'
            )
        relative_path = entry.get('path')
        if not isinstance(relative_path, str):
            raise ValueError(f'Checkpoint index has no path for {role}: {index_path}')
        checkpoint_path = (run_dir / relative_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Missing {role} checkpoint: {checkpoint_path}')
        return _semantic_record(run_dir, role, index, checkpoint_path)

    step = normalized['step']
    checkpoint_path = run_dir / 'checkpoints' / f'params_{step}.pkl'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Missing numeric checkpoint: {checkpoint_path}')
    metadata = {}
    if load_metadata:
        with checkpoint_path.open('rb') as file:
            payload = pickle.load(file)
        if not isinstance(payload, Mapping) or 'agent' not in payload:
            raise ValueError(f'Checkpoint is not a serialized RLC agent: {checkpoint_path}')
        metadata = payload.get('checkpoint_metadata') or {}
    return {
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_step': int(step),
        'checkpoint_role': metadata.get('checkpoint_role', 'explicit'),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'checkpoint_metadata': _jsonable(metadata),
    }
