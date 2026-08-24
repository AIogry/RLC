"""Deterministic dataset-state support proximity proxy."""

from __future__ import annotations

import numpy as np

from .banks import save_bank


def build_support_reference(observations, *, max_states=50000, epsilon=1e-6,
                            environment, dataset_root, source_commit):
    observations = np.asarray(observations)
    if observations.ndim != 2 or len(observations) == 0:
        raise ValueError(f'Expected non-empty [N,D] observations, got {observations.shape}')
    count = min(int(max_states), len(observations))
    indices = np.unique(np.floor(np.linspace(0, len(observations) - 1, count)).astype(np.int64))
    mean = observations.mean(axis=0).astype(np.float32)
    std = np.maximum(observations.std(axis=0), float(epsilon)).astype(np.float32)
    arrays = {'reference_observations': observations[indices]}
    manifest = {
        'bank_type': 'support_reference',
        'environment': environment,
        'dataset_root': str(dataset_root),
        'source_commit': source_commit,
        'reference_selection': 'evenly_spaced_sorted_dataset_indices',
        'reference_indices': indices.tolist(),
        'reference_count': int(len(indices)),
        'dataset_count': int(len(observations)),
        'normalization_mean': mean.tolist(),
        'normalization_std': std.tolist(),
        'normalization_epsilon': float(epsilon),
        'metric': 'nearest_standardized_euclidean_distance',
        'interpretation': 'dataset-state proximity proxy, not true OOD',
    }
    return arrays, manifest


def persist_support_reference(root, observations, **kwargs):
    arrays, manifest = build_support_reference(observations, **kwargs)
    return save_bank(root, arrays, manifest)


def support_distance(observations, reference_bank, *, query_chunk_size=4096):
    arrays = reference_bank.arrays if hasattr(reference_bank, 'arrays') else reference_bank
    manifest = reference_bank.manifest if hasattr(reference_bank, 'manifest') else {}
    reference = np.asarray(arrays['reference_observations'], dtype=np.float32)
    mean = np.asarray(manifest['normalization_mean'], dtype=np.float32)
    std = np.asarray(manifest['normalization_std'], dtype=np.float32)
    queries = np.asarray(observations, dtype=np.float32)
    if queries.ndim != 2 or reference.ndim != 2 or queries.shape[1] != reference.shape[1]:
        raise ValueError(f'Incompatible support shapes: {queries.shape} vs {reference.shape}')
    reference = (reference - mean) / std
    distances = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), int(query_chunk_size)):
        stop = min(start + int(query_chunk_size), len(queries))
        query = (queries[start:stop] - mean) / std
        distances[start:stop] = np.linalg.norm(
            query[:, None, :] - reference[None, :, :], axis=-1
        ).min(axis=1)
    return distances

