"""Non-oracle diagnostics for M25 Puzzle control coordinates.

All algebra in this module is derived from observed event vectors or the
learned flow.  It intentionally does not import Puzzle's true operation
matrix, inverse, solver, or physical button identities.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ...representation.puzzle_effects import pack_raw_effect_signature


def _binary_array(value, *, name, ndim=None):
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f'{name} must have rank {ndim}, got shape {array.shape}')
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f'{name} must contain exact 0/1 values')
    return array.astype(np.uint8, copy=False)


def gf2_rank(matrix):
    """Return exact row rank over GF(2), independent of Puzzle ground truth."""

    work = _binary_array(matrix, name='GF(2) matrix', ndim=2).copy()
    rows, cols = work.shape
    pivot_row = 0
    for col in range(cols):
        candidates = np.flatnonzero(work[pivot_row:, col])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            work[[pivot_row, selected]] = work[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and work[row, col]:
                work[row] ^= work[pivot_row]
        pivot_row += 1
        if pivot_row == rows:
            break
    return int(pivot_row)


def gf2_apply(matrix, vectors):
    """Apply A[out,input] to one or more row-vector inputs over GF(2)."""

    matrix = _binary_array(matrix, name='GF(2) matrix', ndim=2)
    vectors = _binary_array(vectors, name='GF(2) vectors')
    if vectors.shape[-1] != matrix.shape[1]:
        raise ValueError(
            f'GF(2) input dimension {vectors.shape[-1]} != {matrix.shape[1]}'
        )
    return ((vectors.astype(np.uint64) @ matrix.T.astype(np.uint64)) & 1).astype(
        np.uint8
    )


def _summary(values):
    values = np.asarray(values)
    if values.size == 0:
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'min': int(values.min()),
        'max': int(values.max()),
    }


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            'count': 0,
            'p0': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p90': None,
            'p95': None,
            'p100': None,
        }
    names = ('p0', 'p25', 'p50', 'p75', 'p90', 'p95', 'p100')
    quantiles = np.quantile(values, (0, 0.25, 0.5, 0.75, 0.9, 0.95, 1))
    return {'count': int(values.size)} | {
        name: float(value) for name, value in zip(names, quantiles)
    }


def _record_effect(record):
    start = np.asarray(record.start_board, dtype=np.uint8)
    end = np.asarray(record.end_board, dtype=np.uint8)
    return np.bitwise_xor(start, end)


def unique_observed_effects(event_index):
    """Return unique observed effects and stable keys in first-seen order."""

    vectors = {}
    for record in event_index.records:
        effect = _record_effect(record)
        key = record.raw_effect_signature or pack_raw_effect_signature(effect)
        vectors.setdefault(key, effect)
    keys = tuple(vectors)
    if keys:
        matrix = np.stack([vectors[key] for key in keys]).astype(np.uint8)
    else:
        matrix = np.empty((0, event_index.num_bits), dtype=np.uint8)
    return keys, matrix


def audit_effect_event_index(event_index, *, window_scales=(1, 2, 4, 8, 16, 32)):
    """Audit data identifiability, context coverage, and future windows."""

    scales = tuple(int(scale) for scale in window_scales)
    if any(scale <= 0 for scale in scales):
        raise ValueError(f'window_scales must be positive, got {window_scales!r}')
    per_episode = np.zeros(len(event_index.episode_bounds), dtype=np.int64)
    frequencies = Counter()
    contexts = defaultdict(set)
    episodes = defaultdict(set)
    weights = []
    previous_gaps = []
    next_gaps = []
    for record in event_index.records:
        per_episode[record.episode_id] += 1
        effect = _record_effect(record)
        key = record.raw_effect_signature or pack_raw_effect_signature(effect)
        frequencies[key] += 1
        contexts[key].add(record.start_board)
        episodes[key].add(record.episode_id)
        weights.append(record.raw_effect_hamming_weight)
        if record.previous_event_gap is not None:
            previous_gaps.append(record.previous_event_gap)
        if record.next_event_gap is not None:
            next_gaps.append(record.next_event_gap)

    effect_keys, unique_effects = unique_observed_effects(event_index)
    rank = gf2_rank(unique_effects)
    weight_frequency = Counter(int(weight) for weight in weights)
    num_events = len(event_index)
    window_coverage = {}
    for scale in scales:
        pre_count = sum(record.pre_event_steps >= scale for record in event_index.records)
        post_count = sum(record.post_event_steps >= scale for record in event_index.records)
        both_count = sum(
            record.pre_event_steps >= scale and record.post_event_steps >= scale
            for record in event_index.records
        )
        denominator = max(num_events, 1)
        window_coverage[str(scale)] = {
            'pre_event_count': int(pre_count),
            'pre_event_fraction': float(pre_count / denominator),
            'post_event_count': int(post_count),
            'post_event_fraction': float(post_count / denominator),
            'symmetric_count': int(both_count),
            'symmetric_fraction': float(both_count / denominator),
        }

    return {
        'num_bits': int(event_index.num_bits),
        'number_of_dataset_steps': int(event_index.num_steps),
        'number_of_episodes': int(len(event_index.episode_bounds)),
        'number_of_task_state_events': int(num_events),
        'events_per_episode': per_episode.tolist(),
        'events_per_episode_summary': _summary(per_episode),
        'number_of_unique_raw_xor_effects': int(len(effect_keys)),
        'raw_effect_frequencies': {
            key: int(frequencies[key]) for key in sorted(frequencies)
        },
        'raw_effect_hamming_weight_distribution': {
            str(weight): int(weight_frequency[weight]) for weight in sorted(weight_frequency)
        },
        'observed_effect_gf2_rank': int(rank),
        'number_of_independent_effect_vectors': int(rank),
        'observed_effects_span_full_task_space': bool(rank == event_index.num_bits),
        'distinct_start_contexts_per_raw_effect': {
            key: int(len(contexts[key])) for key in sorted(contexts)
        },
        'distinct_episodes_per_raw_effect': {
            key: int(len(episodes[key])) for key in sorted(episodes)
        },
        'effects_with_cross_episode_coverage': int(
            sum(len(episode_ids) > 1 for episode_ids in episodes.values())
        ),
        'all_effects_have_cross_episode_coverage': bool(
            episodes and all(len(episode_ids) > 1 for episode_ids in episodes.values())
        ),
        'previous_event_gap_quantiles': _quantiles(previous_gaps),
        'next_event_gap_quantiles': _quantiles(next_gaps),
        'candidate_future_window_scale_coverage': window_coverage,
    }


def effective_matrix_diagnostics(matrix, *, encode_fn=None, test_inputs=None):
    """Audit learned A and, optionally, exact H(x)=Ax agreement."""

    matrix = _binary_array(matrix, name='effective matrix', ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Effective matrix must be square, got {matrix.shape}')
    rank = gf2_rank(matrix)
    result = {
        'effective_matrix_rank': rank,
        'effective_matrix_full_rank': bool(rank == matrix.shape[0]),
        'effective_matrix_density': float(matrix.mean()),
        'effective_matrix_row_hamming_weights': matrix.sum(axis=1).astype(int).tolist(),
        'effective_matrix_column_hamming_weights': matrix.sum(axis=0).astype(int).tolist(),
    }
    if test_inputs is not None:
        if encode_fn is None:
            raise ValueError('encode_fn is required with test_inputs')
        test_inputs = _binary_array(test_inputs, name='matrix agreement inputs')
        predicted = gf2_apply(matrix, test_inputs)
        actual = _binary_array(
            np.asarray(encode_fn(test_inputs.astype(np.float32))),
            name='flow matrix agreement outputs',
        )
        exact = np.all(predicted == actual, axis=-1)
        result.update({
            'matrix_forward_agreement_count': int(exact.sum()),
            'matrix_forward_agreement_total': int(exact.size),
            'matrix_forward_agreement_rate': float(exact.mean()) if exact.size else None,
            'matrix_forward_exact': bool(np.all(exact)),
        })
    return result


def axis_assignment_diagnostics(event_index, effective_matrix):
    """Map every unique observed effect to its learned latent delta/axis."""

    keys, effects = unique_observed_effects(event_index)
    latent = gf2_apply(effective_matrix, effects)
    distances = latent.sum(axis=-1)
    assignments = {}
    axis_occupancy = Counter()
    axis_event_frequency = Counter()
    latent_owners = defaultdict(list)
    raw_frequencies = Counter(
        record.raw_effect_signature or pack_raw_effect_signature(_record_effect(record))
        for record in event_index.records
    )
    for key, delta, distance in zip(keys, latent, distances):
        latent_key = tuple(int(bit) for bit in delta)
        latent_owners[latent_key].append(key)
        axis = int(np.argmax(delta)) if distance == 1 else None
        if axis is not None:
            axis_occupancy[axis] += 1
            axis_event_frequency[axis] += raw_frequencies[key]
        assignments[key] = {
            'latent_delta': list(latent_key),
            'latent_hamming_distance': int(distance),
            'learned_axis': axis,
            'event_frequency': int(raw_frequencies[key]),
        }
    collisions = {
        ''.join(str(bit) for bit in latent_key): owners
        for latent_key, owners in latent_owners.items()
        if len(owners) > 1
    }
    one_hot = int(np.sum(distances == 1))
    return {
        'number_of_unique_observed_raw_effects': int(len(keys)),
        'number_mapped_to_exactly_one_latent_axis': one_hot,
        'number_of_distinct_latent_axes_used': int(len(axis_occupancy)),
        'all_unique_effects_axis_local': bool(one_hot == len(keys)),
        'one_hot_axis_collision': bool(any(count > 1 for count in axis_occupancy.values())),
        'latent_delta_collision_count': int(len(collisions)),
        'latent_delta_collisions': collisions,
        'axis_occupancy_by_unique_effect': {
            str(axis): int(axis_occupancy[axis]) for axis in sorted(axis_occupancy)
        },
        'axis_occupancy_by_event_frequency': {
            str(axis): int(axis_event_frequency[axis])
            for axis in sorted(axis_event_frequency)
        },
        'effect_axis_assignments': assignments,
    }


def encode_event_latent_deltas(event_index, encode_fn, *, batch_size=4096):
    """Encode all exact event endpoints in bounded deterministic batches."""

    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    start, end = event_index.endpoint_arrays()
    chunks = []
    for offset in range(0, len(start), int(batch_size)):
        encoded_start = _binary_array(
            np.asarray(encode_fn(start[offset:offset + batch_size].astype(np.float32))),
            name='encoded event start',
        )
        encoded_end = _binary_array(
            np.asarray(encode_fn(end[offset:offset + batch_size].astype(np.float32))),
            name='encoded event end',
        )
        chunks.append(np.bitwise_xor(encoded_start, encoded_end))
    if not chunks:
        return np.empty((0, event_index.num_bits), dtype=np.uint8)
    return np.concatenate(chunks, axis=0)


def same_effect_consistency_diagnostics(event_index, latent_deltas):
    """Verify structural same-effect and cross-episode latent consistency."""

    latent_deltas = _binary_array(latent_deltas, name='latent event deltas', ndim=2)
    if len(latent_deltas) != len(event_index):
        raise ValueError('One latent delta is required for every event record')
    grouped = defaultdict(list)
    for record, delta in zip(event_index.records, latent_deltas):
        key = record.raw_effect_signature or pack_raw_effect_signature(_record_effect(record))
        grouped[key].append((record.episode_id, tuple(int(bit) for bit in delta)))
    group_consistency = {}
    cross_episode_groups = 0
    cross_episode_consistent = 0
    for key, values in grouped.items():
        deltas = {delta for _, delta in values}
        episode_ids = {episode_id for episode_id, _ in values}
        consistent = len(deltas) == 1
        if len(episode_ids) > 1:
            cross_episode_groups += 1
            cross_episode_consistent += int(consistent)
        group_consistency[key] = {
            'number_of_events': len(values),
            'number_of_episodes': len(episode_ids),
            'number_of_distinct_latent_deltas': len(deltas),
            'consistent': bool(consistent),
        }
    consistent_groups = sum(item['consistent'] for item in group_consistency.values())
    return {
        'same_effect_latent_delta_consistency': bool(
            all(item['consistent'] for item in group_consistency.values())
        ),
        'same_effect_consistent_group_fraction': (
            float(consistent_groups / len(group_consistency)) if group_consistency else None
        ),
        'cross_episode_effect_group_count': int(cross_episode_groups),
        'cross_episode_consistent_group_count': int(cross_episode_consistent),
        'cross_episode_consistency': bool(
            cross_episode_consistent == cross_episode_groups
        ),
        'same_effect_groups': {
            key: group_consistency[key] for key in sorted(group_consistency)
        },
    }


def goal_mask_diagnostics(state_boards, goal_boards, goal_mask_fn):
    """Report H(s) XOR H(g) weights and exact repeated-call reproducibility."""

    state_boards = _binary_array(state_boards, name='goal-mask states')
    goal_boards = _binary_array(goal_boards, name='goal-mask goals')
    if state_boards.shape != goal_boards.shape:
        raise ValueError('Goal-mask state and goal arrays must have the same shape')
    first = _binary_array(
        np.asarray(goal_mask_fn(state_boards.astype(np.float32), goal_boards.astype(np.float32))),
        name='goal masks',
    )
    second = _binary_array(
        np.asarray(goal_mask_fn(state_boards.astype(np.float32), goal_boards.astype(np.float32))),
        name='repeated goal masks',
    )
    weights = first.sum(axis=-1)
    return {
        'goal_mask_count': int(len(first)),
        'goal_mask_hamming_weight_mean': float(weights.mean()) if len(weights) else None,
        'goal_mask_hamming_weight_median': float(np.median(weights)) if len(weights) else None,
        'goal_mask_hamming_weight_max': int(weights.max()) if len(weights) else None,
        'goal_mask_exact_reproducibility': bool(np.array_equal(first, second)),
    }


def evaluate_control_coordinates(
    event_index,
    *,
    encode_fn,
    inverse_fn,
    goal_mask_fn,
    effective_matrix,
    batch_size=4096,
    diagnostic_seed=0,
    matrix_test_states=256,
    goal_pair_count=1024,
    window_scales=(1, 2, 4, 8, 16, 32),
):
    """Run the complete non-oracle M25 Stage-1 evaluation bundle."""

    if len(event_index) == 0:
        raise ValueError('Control-coordinate evaluation requires at least one event')
    if batch_size <= 0 or matrix_test_states <= 0 or goal_pair_count <= 0:
        raise ValueError(
            'batch_size, matrix_test_states, and goal_pair_count must be positive'
        )
    rng = np.random.default_rng(int(diagnostic_seed))
    latent_deltas = encode_event_latent_deltas(
        event_index, encode_fn, batch_size=batch_size
    )
    distances = latent_deltas.sum(axis=-1)
    random_states = rng.integers(
        0, 2, size=(int(matrix_test_states), event_index.num_bits), dtype=np.uint8
    )
    encoded_random = _binary_array(
        np.asarray(encode_fn(random_states.astype(np.float32))),
        name='random encoded states',
    )
    recovered_random = _binary_array(
        np.asarray(inverse_fn(encoded_random.astype(np.float32))),
        name='random inverse states',
    )
    random_other = rng.integers(
        0, 2, size=random_states.shape, dtype=np.uint8
    )
    encoded_xor = _binary_array(
        np.asarray(encode_fn(np.bitwise_xor(random_states, random_other).astype(np.float32))),
        name='encoded XOR states',
    )
    separate_xor = np.bitwise_xor(
        encoded_random,
        _binary_array(
            np.asarray(encode_fn(random_other.astype(np.float32))),
            name='encoded second random states',
        ),
    )
    zero = np.zeros((1, event_index.num_bits), dtype=np.float32)
    encoded_zero = _binary_array(np.asarray(encode_fn(zero)), name='encoded zero')

    starts, ends = event_index.endpoint_arrays()
    pair_count = min(int(goal_pair_count), len(starts))
    pair_indices = rng.integers(0, len(starts), size=pair_count)
    goal_indices = rng.integers(0, len(ends), size=pair_count)

    return {
        'event_audit': audit_effect_event_index(
            event_index, window_scales=window_scales
        ),
        'hard_event_metrics': {
            'hard_axis_success_rate': float(np.mean(distances == 1)),
            'mean_hard_latent_event_distance': float(distances.mean()),
            'median_hard_latent_event_distance': float(np.median(distances)),
            'fraction_hard_latent_event_distance_gt_one': float(
                np.mean(distances > 1)
            ),
            'maximum_hard_latent_event_distance': int(distances.max()),
        },
        'axis_assignment': axis_assignment_diagnostics(
            event_index, effective_matrix
        ),
        'same_effect_consistency': same_effect_consistency_diagnostics(
            event_index, latent_deltas
        ),
        'effective_matrix': effective_matrix_diagnostics(
            effective_matrix, encode_fn=encode_fn, test_inputs=random_states
        ),
        'goal_mask': goal_mask_diagnostics(
            starts[pair_indices], ends[goal_indices], goal_mask_fn
        ),
        'structural_invariants': {
            'forward_inverse_exact': bool(np.array_equal(random_states, recovered_random)),
            'gf2_linearity_exact': bool(np.array_equal(encoded_xor, separate_xor)),
            'zero_maps_to_zero': bool(np.all(encoded_zero == 0)),
            'forward_outputs_exactly_binary': True,
        },
    }


__all__ = [
    'audit_effect_event_index',
    'axis_assignment_diagnostics',
    'effective_matrix_diagnostics',
    'encode_event_latent_deltas',
    'evaluate_control_coordinates',
    'gf2_apply',
    'gf2_rank',
    'goal_mask_diagnostics',
    'same_effect_consistency_diagnostics',
    'unique_observed_effects',
]
