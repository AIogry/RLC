"""Deterministic immutable banks for checkpoint-only diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from impls.utils.datasets import GCDataset
from impls.utils.reproducibility import derive_seed

BANK_SCHEMA_VERSION = 1


def _update_hash(digest, key, value):
    array = np.asarray(value)
    digest.update(str(key).encode())
    digest.update(str(array.dtype).encode())
    digest.update(repr(tuple(array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())


def arrays_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        _update_hash(digest, key, arrays[key])
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class BankArtifact:
    root: Path
    arrays: dict[str, np.ndarray]
    manifest: dict


def save_bank(root, arrays, manifest, sample_rows=()):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite immutable bank: {root}')
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    normalized = {str(k): np.asarray(v) for k, v in arrays.items()}
    lengths = {int(v.shape[0]) for v in normalized.values() if v.ndim > 0}
    if len(lengths) > 1:
        raise ValueError(f'Bank arrays have inconsistent first dimensions: {lengths}')
    np.savez_compressed(root / 'bank.npz', **normalized)
    result = dict(manifest)
    result.update({
        'schema_version': BANK_SCHEMA_VERSION,
        'bank_hash': arrays_hash(normalized),
        'sample_count': next(iter(lengths), 0),
        'array_keys': sorted(normalized),
        'immutable': True,
    })
    with (root / 'manifest.json').open('w') as file:
        json.dump(_jsonable(result), file, indent=2, sort_keys=True)
        file.write('\n')
    rows = list(sample_rows)
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (root / 'sample_index.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
    else:
        (root / 'sample_index.csv').write_text('index\n')
    return BankArtifact(root, normalized, result)


def load_bank(root, verify_hash=True):
    root = Path(root)
    with (root / 'manifest.json').open() as file:
        manifest = json.load(file)
    with np.load(root / 'bank.npz', allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if verify_hash and arrays_hash(arrays) != manifest.get('bank_hash'):
        raise ValueError(f'Bank hash mismatch: {root}')
    lengths = {int(v.shape[0]) for v in arrays.values() if v.ndim > 0}
    if len(lengths) > 1 or manifest.get('sample_count') != next(iter(lengths), 0):
        raise ValueError(f'Bank sample count mismatch: {root}')
    return BankArtifact(root, arrays, manifest)


class _TracingGCDataset(GCDataset):
    def __post_init__(self):
        super().__post_init__()
        self.goal_trace = []

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, rng):
        result = super().sample_goals(
            idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, rng
        )
        self.goal_trace.append(np.asarray(result, dtype=np.int64).copy())
        return result


def build_training_support_bank(
    dataset, config, *, seed, batches, batch_size, environment, dataset_root,
    source_commit, provenance=None,
):
    """Use the exact Stage-2 GCDataset actor-goal sampling semantics."""
    gc_dataset = _TracingGCDataset(
        dataset=dataset, config=config, rng=derive_seed(seed, 11)
    )
    names = (
        'observations', 'actor_goals', 'value_goals', 'actions',
        'dataset_indices', 'actor_goal_indices', 'value_goal_indices', 'batch_index',
    )
    collected = {name: [] for name in names}
    rows = []
    rng = gc_dataset.rng
    for batch_index in range(int(batches)):
        before = len(gc_dataset.goal_trace)
        idxs = gc_dataset.dataset.get_random_idxs(batch_size, rng=rng)
        batch = gc_dataset.sample(batch_size, idxs=idxs, rng=rng)
        goals = gc_dataset.goal_trace[before:]
        if len(goals) != 2:
            raise AssertionError('Expected value and actor goal traces')
        value_goal_idxs, actor_goal_idxs = goals
        collected['observations'].append(np.asarray(batch['observations']))
        collected['actor_goals'].append(np.asarray(batch['actor_goals']))
        collected['value_goals'].append(np.asarray(batch['value_goals']))
        collected['actions'].append(np.asarray(batch['actions']))
        collected['dataset_indices'].append(np.asarray(idxs, dtype=np.int64))
        collected['actor_goal_indices'].append(actor_goal_idxs)
        collected['value_goal_indices'].append(value_goal_idxs)
        collected['batch_index'].append(np.full(batch_size, batch_index, dtype=np.int64))
        rows.extend({
            'sample_index': batch_index * batch_size + i,
            'batch_index': batch_index,
            'dataset_index': int(idxs[i]),
            'actor_goal_index': int(actor_goal_idxs[i]),
            'value_goal_index': int(value_goal_idxs[i]),
        } for i in range(batch_size))
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    manifest = {
        'bank_type': 'B_T',
        'environment': environment,
        'critic_seed': int(seed),
        'dataset_root': str(dataset_root),
        'sampling_seed': int(derive_seed(seed, 11)),
        'batch_size': int(batch_size),
        'batch_count': int(batches),
        'goal_protocol': 'exact_GCDataset_actor_goal_sampling',
        'state_identity': 'dataset_indices',
        'source_commit': source_commit,
        'provenance': provenance or {},
    }
    return arrays, manifest, rows


def build_eval_goal_bank(training_bank, *, eval_goals, task_names, environment,
                         source_commit, dataset_root, evaluation_seed, provenance=None):
    state = training_bank.arrays['observations']
    action = training_bank.arrays['actions']
    state_indices = training_bank.arrays['dataset_indices']
    task_ids = sorted(int(k) for k in eval_goals)
    state_arrays, action_arrays, goal_arrays, task_arrays = [], [], [], []
    rows = []
    for task_id in task_ids:
        goal = np.asarray(eval_goals[task_id])
        state_arrays.append(state)
        action_arrays.append(action)
        goal_arrays.append(np.repeat(goal[None, :], len(state), axis=0))
        task_arrays.append(np.full(len(state), task_id, dtype=np.int64))
        for i, source_index in enumerate(state_indices):
            rows.append({
                'sample_index': len(rows), 'source_training_sample': i,
                'dataset_index': int(source_index), 'task_id': task_id,
                'task_name': task_names[task_id],
            })
    arrays = {
        'observations': np.concatenate(state_arrays),
        'eval_goals': np.concatenate(goal_arrays),
        'actions': np.concatenate(action_arrays),
        'dataset_indices': np.tile(state_indices, len(task_ids)),
        'task_id': np.concatenate(task_arrays),
    }
    manifest = {
        'bank_type': 'B_DE',
        'environment': environment,
        'dataset_root': str(dataset_root),
        'source_commit': source_commit,
        'evaluation_seed': int(evaluation_seed),
        'goal_protocol': 'formal_eval_task_reset_goal',
        'state_identity': 'same_dataset_indices_as_B_T',
        'parent_training_bank_hash': training_bank.manifest['bank_hash'],
        'task_balance': {str(task_id): len(state) for task_id in task_ids},
        'task_names': {str(k): v for k, v in task_names.items()},
        'provenance': provenance or {},
    }
    return arrays, manifest, rows

