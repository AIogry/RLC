"""Generic immutable event indexing and deterministic event-pair sampling."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from numbers import Integral

import numpy as np


@dataclasses.dataclass(frozen=True)
class EffectEventRecord:
    """One adjacent nonzero task-state transition inside one episode."""

    episode_id: int
    transition_idx: int
    start_idx: int
    end_idx: int
    episode_start_idx: int
    episode_end_idx: int
    start_board: tuple[int, ...]
    end_board: tuple[int, ...]
    raw_effect_signature: str | None
    raw_effect_hamming_weight: int
    previous_event_gap: int | None
    next_event_gap: int | None
    pre_event_steps: int
    post_event_steps: int


@dataclasses.dataclass(frozen=True)
class EffectEventIndex:
    """Immutable event records plus their originating episode boundaries."""

    records: tuple[EffectEventRecord, ...]
    episode_bounds: tuple[tuple[int, int], ...]
    num_steps: int
    num_bits: int

    def __post_init__(self):
        if not isinstance(self.records, tuple) or not isinstance(self.episode_bounds, tuple):
            raise TypeError('EffectEventIndex records and episode_bounds must be tuples')
        if self.num_steps <= 0 or self.num_bits <= 0:
            raise ValueError('Event index dimensions must be positive')
        for record in self.records:
            if record.end_idx != record.start_idx + 1:
                raise ValueError('An event must use exactly adjacent observation endpoints')
            if record.transition_idx != record.start_idx:
                raise ValueError('transition_idx must equal the event start index')
            if not (
                record.episode_start_idx <= record.start_idx
                < record.end_idx <= record.episode_end_idx
            ):
                raise ValueError('Event record crosses or leaves its episode boundary')
            if record.start_board == record.end_board:
                raise ValueError('Zero-change records are forbidden in an event index')

    def __len__(self):
        return len(self.records)

    def endpoint_arrays(self, record_indices=None):
        """Materialize only binary endpoints, never diagnostic effect identities."""

        if record_indices is None:
            selected = self.records
        else:
            selected = tuple(self.records[int(index)] for index in record_indices)
        if not selected:
            empty = np.empty((0, self.num_bits), dtype=np.uint8)
            return empty, empty.copy()
        return (
            np.asarray([record.start_board for record in selected], dtype=np.uint8),
            np.asarray([record.end_board for record in selected], dtype=np.uint8),
        )


def episode_bounds_from_terminals(terminals):
    """Return inclusive episode observation bounds from canonical terminals."""

    terminals = np.asarray(terminals)
    if terminals.ndim != 1 or terminals.size == 0:
        raise ValueError(f'terminals must be a non-empty vector, got {terminals.shape}')
    if not np.all(np.isfinite(terminals)) or not np.all(
        (terminals == 0) | (terminals == 1)
    ):
        raise ValueError('terminals must contain exact 0/1 values')
    terminal_indices = np.flatnonzero(terminals == 1)
    if terminal_indices.size == 0 or terminal_indices[-1] != terminals.size - 1:
        raise ValueError('Canonical terminals must mark the final dataset observation')
    starts = np.concatenate([np.asarray([0]), terminal_indices[:-1] + 1])
    return tuple(
        (int(start), int(end)) for start, end in zip(starts, terminal_indices)
    )


def episode_terminals_from_compact_valids(valids):
    """Recover original observation endpoints from canonical compact ``valids``.

    OGBench's compact representation keeps every observation, marks indices
    with a valid ``observation[t] -> observation[t + 1]`` transition as one,
    and marks the final observation of each trajectory as zero.  Its compact
    ``terminals`` field has different, transition-oriented semantics and must
    not be used as observation episode bounds.
    """

    valids = np.asarray(valids)
    if valids.ndim != 1 or valids.size == 0:
        raise ValueError(f'valids must be a non-empty vector, got {valids.shape}')
    if not np.all(np.isfinite(valids)) or not np.all((valids == 0) | (valids == 1)):
        raise ValueError('valids must contain exact 0/1 values')
    if valids[-1] != 0:
        raise ValueError('Canonical compact valids must end at an episode boundary')
    return (valids == 0).astype(np.uint8)


def build_effect_event_index(
    boards,
    terminals,
    *,
    event_mask=None,
    effect_signature_fn: Callable | None = None,
):
    """Build adjacent event records while masking every episode boundary."""

    boards = np.asarray(boards)
    if boards.ndim != 2 or boards.shape[0] == 0 or boards.shape[1] == 0:
        raise ValueError(f'boards must have shape [steps, bits], got {boards.shape}')
    if not np.all((boards == 0) | (boards == 1)):
        raise ValueError('boards must contain exact 0/1 values')
    boards = boards.astype(np.uint8, copy=False)
    terminals = np.asarray(terminals)
    if terminals.shape != (len(boards),):
        raise ValueError(
            f'terminals shape {terminals.shape} does not match {len(boards)} boards'
        )
    episode_bounds = episode_bounds_from_terminals(terminals)
    if event_mask is None:
        event_mask = np.any(boards[:-1] != boards[1:], axis=-1)
    event_mask = np.asarray(event_mask, dtype=bool)
    if event_mask.shape != (len(boards) - 1,):
        raise ValueError(
            f'event_mask must have shape {(len(boards) - 1,)}, got {event_mask.shape}'
        )

    records = []
    for episode_id, (episode_start, episode_end) in enumerate(episode_bounds):
        transition_indices = [
            transition_idx
            for transition_idx in range(episode_start, episode_end)
            if event_mask[transition_idx]
        ]
        for event_offset, transition_idx in enumerate(transition_indices):
            start_board = boards[transition_idx]
            end_board = boards[transition_idx + 1]
            effect = np.bitwise_xor(start_board, end_board)
            if not np.any(effect):
                raise ValueError(
                    f'event_mask includes zero-change transition {transition_idx}'
                )
            previous_gap = (
                None
                if event_offset == 0
                else transition_idx - transition_indices[event_offset - 1]
            )
            next_gap = (
                None
                if event_offset + 1 == len(transition_indices)
                else transition_indices[event_offset + 1] - transition_idx
            )
            signature = (
                None
                if effect_signature_fn is None
                else effect_signature_fn(start_board, end_board)
            )
            records.append(EffectEventRecord(
                episode_id=episode_id,
                transition_idx=transition_idx,
                start_idx=transition_idx,
                end_idx=transition_idx + 1,
                episode_start_idx=episode_start,
                episode_end_idx=episode_end,
                start_board=tuple(int(bit) for bit in start_board),
                end_board=tuple(int(bit) for bit in end_board),
                raw_effect_signature=signature,
                raw_effect_hamming_weight=int(effect.sum()),
                previous_event_gap=previous_gap,
                next_event_gap=next_gap,
                pre_event_steps=transition_idx - episode_start,
                post_event_steps=episode_end - (transition_idx + 1),
            ))
    return EffectEventIndex(
        records=tuple(records),
        episode_bounds=episode_bounds,
        num_steps=len(boards),
        num_bits=boards.shape[1],
    )


@dataclasses.dataclass(frozen=True)
class EffectEventDataset:
    """Uniform Stage-1 sampler with explicit, independently owned NumPy RNG."""

    event_index: EffectEventIndex
    seed: int | None = None
    event_sampling_mode: str = 'uniform'
    include_record_indices: bool = False
    rng: np.random.Generator = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.event_sampling_mode != 'uniform':
            raise ValueError(
                'Only explicit uniform event sampling is implemented for M25 Stage 1'
            )
        if len(self.event_index) == 0:
            raise ValueError('Cannot sample an empty event index')
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError(
                f'seed must be a non-negative integer or None, got {self.seed!r}'
            )
        object.__setattr__(self, 'rng', np.random.default_rng(self.seed))

    def sample(self, batch_size, *, record_indices=None, rng=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size!r}')
        generator = self.rng if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError('rng must be an explicit numpy.random.Generator')
        if record_indices is None:
            record_indices = generator.integers(
                0, len(self.event_index), size=int(batch_size)
            )
        record_indices = np.asarray(record_indices, dtype=np.int64)
        if record_indices.shape != (int(batch_size),):
            raise ValueError(
                f'record_indices must have shape {(int(batch_size),)}, '
                f'got {record_indices.shape}'
            )
        if np.any(record_indices < 0) or np.any(record_indices >= len(self.event_index)):
            raise IndexError('record_indices contains an out-of-range event index')
        start, end = self.event_index.endpoint_arrays(record_indices)
        batch = {'start_board': start, 'end_board': end}
        if self.include_record_indices:
            batch['record_indices'] = record_indices.copy()
        return batch


__all__ = [
    'EffectEventDataset',
    'EffectEventIndex',
    'EffectEventRecord',
    'build_effect_event_index',
    'episode_bounds_from_terminals',
    'episode_terminals_from_compact_valids',
]
