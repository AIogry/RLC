"""Reusable controlled policy-goal replay for canonical Puzzle rollouts.

Puzzle task reset constructs ``info['goal']`` with an internal action-space
sample that is not governed solely by ``env.reset(seed=...)``.  This module
records one real reset per paired episode and reuses that complete goal vector
for every compared policy, without changing the environment's own task target
or training semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...utils.evaluation import COMMON_EPISODE_SEED_SCHEME, common_episode_seeds
from .layout import decode_board_bits


CONTROLLED_GOAL_REPLAY_SCHEMA_VERSION = 'puzzle_controlled_goal_replay_v1'
_ARCHIVE_NAME = 'records.npz'
_METADATA_NAME = 'manifest.json'
_ARRAY_FIELDS = (
    'paired_episode_id',
    'task_id',
    'episode_index',
    'evaluation_seed',
    'task_seed',
    'episode_seed',
    'actor_seed',
    'noise_seed',
    'goal',
    'board_goal',
    'initial_observation',
    'goal_fingerprint',
    'board_goal_fingerprint',
    'initial_observation_fingerprint',
)


class ControlledGoalReplayError(ValueError):
    """Raised when a controlled Puzzle replay plan or pairing is invalid."""


def array_fingerprint(value):
    """Return a dtype/shape/byte-sensitive SHA-256 fingerprint for an array."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ControlledGoalReplayError('Cannot fingerprint an object-dtype array')
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {'dtype': array.dtype.str, 'shape': list(array.shape)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    digest = hashlib.sha256()
    digest.update(b'puzzle-array-fingerprint-v1\0')
    digest.update(header)
    digest.update(b'\0')
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def paired_episode_id(task_id, episode_index):
    return f'task{int(task_id):02d}_ep{int(episode_index):03d}'


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path, value):
    with Path(path).open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write('\n')


def _json_read(path):
    try:
        with Path(path).open() as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledGoalReplayError(f'Cannot read controlled replay metadata: {path}') from error
    if not isinstance(value, Mapping):
        raise ControlledGoalReplayError(f'Controlled replay metadata must be a mapping: {path}')
    return dict(value)


def _records_fingerprint(arrays):
    digest = hashlib.sha256()
    digest.update(b'puzzle-controlled-goal-records-v1\0')
    for name in sorted(arrays):
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(array_fingerprint(arrays[name]).encode('ascii'))
        digest.update(b'\0')
    return digest.hexdigest()


def _base_environment(env):
    return getattr(env, 'unwrapped', env)


def _puzzle_shape(env):
    base = _base_environment(env)
    try:
        rows = int(base._num_rows)
        cols = int(base._num_cols)
    except (AttributeError, TypeError, ValueError) as error:
        raise ControlledGoalReplayError('Environment is not a canonical Puzzle environment') from error
    if rows <= 0 or cols <= 0:
        raise ControlledGoalReplayError('Puzzle grid dimensions must be positive')
    return rows, cols


def _task_infos(env):
    task_infos = getattr(_base_environment(env), 'task_infos', None)
    if not isinstance(task_infos, Sequence) or isinstance(task_infos, (str, bytes)):
        raise ControlledGoalReplayError('Canonical Puzzle environment has no task_infos sequence')
    return task_infos


def _board_goal(env, task_id, *, rows, cols):
    task_infos = _task_infos(env)
    if not 1 <= int(task_id) <= len(task_infos):
        raise ControlledGoalReplayError(
            f'Unknown Puzzle task_id={task_id}; available=1..{len(task_infos)}'
        )
    try:
        board = np.asarray(task_infos[int(task_id) - 1]['goal_button_states'])
    except (KeyError, TypeError) as error:
        raise ControlledGoalReplayError(f'Puzzle task {task_id} has no goal_button_states') from error
    if board.shape != (rows * cols,) or not np.all((board == 0) | (board == 1)):
        raise ControlledGoalReplayError(f'Puzzle task {task_id} has invalid board goal shape/content')
    return np.asarray(board, dtype=np.uint8).copy()


def _assert_goal_board(goal, board_goal, *, rows, cols, label):
    try:
        decoded = decode_board_bits(goal, rows=rows, cols=cols)
    except (TypeError, ValueError) as error:
        raise ControlledGoalReplayError(f'{label} is not a canonical Puzzle goal observation: {error}') from error
    if not np.array_equal(decoded, board_goal):
        raise ControlledGoalReplayError(f'{label} has a board goal different from the canonical task target')


def _normalized_task_ids(env, task_ids):
    task_ids = tuple(int(task_id) for task_id in task_ids)
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ControlledGoalReplayError('task_ids must be a non-empty sequence of unique task IDs')
    count = len(_task_infos(env))
    invalid = [task_id for task_id in task_ids if not 1 <= task_id <= count]
    if invalid:
        raise ControlledGoalReplayError(f'Unknown Puzzle task IDs {invalid}; available=1..{count}')
    return tuple(sorted(task_ids))


@dataclass(frozen=True)
class ControlledGoalRecord:
    paired_episode_id: str
    task_id: int
    episode_index: int
    evaluation_seed: int
    task_seed: int
    episode_seed: int
    actor_seed: int
    noise_seed: int
    goal: np.ndarray
    board_goal: np.ndarray
    initial_observation: np.ndarray
    goal_fingerprint: str
    board_goal_fingerprint: str
    initial_observation_fingerprint: str


class ControlledGoalReplay:
    """A persisted, environment-specific collection of paired Puzzle inputs."""

    def __init__(self, root, metadata, records):
        self.root = Path(root).resolve()
        self.metadata = dict(metadata)
        self.environment = str(self.metadata['environment'])
        self.rows = int(self.metadata['rows'])
        self.cols = int(self.metadata['cols'])
        self._records = tuple(records)
        self._by_key = {
            (record.task_id, record.episode_index): record for record in self._records
        }
        if len(self._by_key) != len(self._records):
            raise ControlledGoalReplayError('Controlled replay has duplicate paired episode keys')

    @property
    def record_count(self):
        return len(self._records)

    @classmethod
    def create(
        cls,
        root,
        env,
        *,
        environment,
        evaluation_seed,
        task_ids,
        episodes_per_task,
    ):
        """Capture one real reset goal and initial observation per paired episode."""

        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f'Controlled replay root is not empty: {root}')
        if int(episodes_per_task) <= 0:
            raise ControlledGoalReplayError('episodes_per_task must be positive')
        rows, cols = _puzzle_shape(env)
        task_ids = _normalized_task_ids(env, task_ids)
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for task_id in task_ids:
            board_goal = _board_goal(env, task_id, rows=rows, cols=cols)
            for episode_index in range(int(episodes_per_task)):
                seeds = common_episode_seeds(int(evaluation_seed), task_id, episode_index)
                observation, info = env.reset(
                    seed=seeds['episode_seed'],
                    options={'task_id': task_id, 'render_goal': False},
                )
                if not isinstance(info, Mapping) or 'goal' not in info:
                    raise ControlledGoalReplayError(
                        f'Reset lacks policy-facing info[goal] for '
                        f'{paired_episode_id(task_id, episode_index)}'
                    )
                goal = np.asarray(info['goal']).copy()
                initial_observation = np.asarray(observation).copy()
                _assert_goal_board(
                    goal,
                    board_goal,
                    rows=rows,
                    cols=cols,
                    label=f'Captured goal for {paired_episode_id(task_id, episode_index)}',
                )
                records.append(ControlledGoalRecord(
                    paired_episode_id=paired_episode_id(task_id, episode_index),
                    task_id=task_id,
                    episode_index=episode_index,
                    evaluation_seed=int(evaluation_seed),
                    task_seed=int(seeds['task_seed']),
                    episode_seed=int(seeds['episode_seed']),
                    actor_seed=int(seeds['actor_seed']),
                    noise_seed=int(seeds['noise_seed']),
                    goal=goal,
                    board_goal=board_goal.copy(),
                    initial_observation=initial_observation,
                    goal_fingerprint=array_fingerprint(goal),
                    board_goal_fingerprint=array_fingerprint(board_goal),
                    initial_observation_fingerprint=array_fingerprint(initial_observation),
                ))
        arrays = _records_to_arrays(records)
        archive_path = root / _ARCHIVE_NAME
        np.savez_compressed(archive_path, **arrays)
        metadata = {
            'schema_version': CONTROLLED_GOAL_REPLAY_SCHEMA_VERSION,
            'environment': str(environment),
            'rows': rows,
            'cols': cols,
            'task_ids': list(task_ids),
            'episodes_per_task': int(episodes_per_task),
            'evaluation_seed': int(evaluation_seed),
            'seed_scheme': COMMON_EPISODE_SEED_SCHEME,
            'record_count': len(records),
            'archive': _ARCHIVE_NAME,
            'archive_sha256': _sha256_file(archive_path),
            'records_fingerprint': _records_fingerprint(arrays),
            'goal_source': (
                'one real canonical Puzzle reset per paired episode; the complete '
                'emitted info[goal] is replayed byte-for-byte for every compared policy'
            ),
            'pairing_contract': {
                'goal_fingerprint': 'must equal the controlled record for every policy',
                'board_goal_fingerprint': 'must equal the controlled record for every policy',
                'initial_observation_fingerprint': 'must equal the controlled record for every policy',
            },
        }
        _json_write(root / _METADATA_NAME, metadata)
        return cls.load(root)

    @classmethod
    def load(cls, root):
        """Load and fully validate a persisted controlled replay plan."""

        root = Path(root).resolve()
        metadata = _json_read(root / _METADATA_NAME)
        if metadata.get('schema_version') != CONTROLLED_GOAL_REPLAY_SCHEMA_VERSION:
            raise ControlledGoalReplayError(
                f'Unsupported controlled replay schema: {metadata.get("schema_version")!r}'
            )
        archive_name = metadata.get('archive')
        if archive_name != _ARCHIVE_NAME:
            raise ControlledGoalReplayError(f'Unexpected controlled replay archive: {archive_name!r}')
        archive_path = root / archive_name
        if not archive_path.is_file():
            raise ControlledGoalReplayError(f'Missing controlled replay archive: {archive_path}')
        if metadata.get('archive_sha256') != _sha256_file(archive_path):
            raise ControlledGoalReplayError('Controlled replay archive SHA-256 mismatch')
        try:
            with np.load(archive_path, allow_pickle=False) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        except (OSError, ValueError) as error:
            raise ControlledGoalReplayError(f'Cannot load controlled replay archive: {archive_path}') from error
        if set(arrays) != set(_ARRAY_FIELDS):
            raise ControlledGoalReplayError(
                f'Controlled replay archive fields mismatch: {sorted(arrays)!r}'
            )
        if metadata.get('records_fingerprint') != _records_fingerprint(arrays):
            raise ControlledGoalReplayError('Controlled replay records fingerprint mismatch')
        records = _arrays_to_records(arrays)
        if int(metadata.get('record_count', -1)) != len(records):
            raise ControlledGoalReplayError('Controlled replay record count mismatch')
        rows, cols = int(metadata['rows']), int(metadata['cols'])
        task_ids = set()
        for record in records:
            expected_id = paired_episode_id(record.task_id, record.episode_index)
            if record.paired_episode_id != expected_id:
                raise ControlledGoalReplayError(
                    f'Invalid paired episode ID: {record.paired_episode_id!r}'
                )
            expected_seeds = common_episode_seeds(
                record.evaluation_seed, record.task_id, record.episode_index,
            )
            for field, expected in expected_seeds.items():
                if int(getattr(record, field)) != int(expected):
                    raise ControlledGoalReplayError(
                        f'Controlled replay seed mismatch for {record.paired_episode_id}: {field}'
                    )
            if record.goal_fingerprint != array_fingerprint(record.goal):
                raise ControlledGoalReplayError(f'Goal fingerprint mismatch: {record.paired_episode_id}')
            if record.board_goal_fingerprint != array_fingerprint(record.board_goal):
                raise ControlledGoalReplayError(f'Board goal fingerprint mismatch: {record.paired_episode_id}')
            if record.initial_observation_fingerprint != array_fingerprint(record.initial_observation):
                raise ControlledGoalReplayError(
                    f'Initial observation fingerprint mismatch: {record.paired_episode_id}'
                )
            _assert_goal_board(
                record.goal,
                record.board_goal,
                rows=rows,
                cols=cols,
                label=f'Persisted goal for {record.paired_episode_id}',
            )
            task_ids.add(record.task_id)
        if sorted(task_ids) != [int(task_id) for task_id in metadata.get('task_ids', ())]:
            raise ControlledGoalReplayError('Controlled replay task IDs mismatch')
        return cls(root, metadata, records)

    def record_for(self, task_id, episode_index):
        try:
            return self._by_key[(int(task_id), int(episode_index))]
        except KeyError as error:
            raise ControlledGoalReplayError(
                f'No controlled replay record for task={task_id}, episode={episode_index}'
            ) from error


def _records_to_arrays(records):
    if not records:
        raise ControlledGoalReplayError('Controlled replay must contain at least one record')
    return {
        'paired_episode_id': np.asarray([record.paired_episode_id for record in records], dtype='U32'),
        'task_id': np.asarray([record.task_id for record in records], dtype=np.int64),
        'episode_index': np.asarray([record.episode_index for record in records], dtype=np.int64),
        'evaluation_seed': np.asarray([record.evaluation_seed for record in records], dtype=np.int64),
        'task_seed': np.asarray([record.task_seed for record in records], dtype=np.int64),
        'episode_seed': np.asarray([record.episode_seed for record in records], dtype=np.int64),
        'actor_seed': np.asarray([record.actor_seed for record in records], dtype=np.int64),
        'noise_seed': np.asarray([record.noise_seed for record in records], dtype=np.int64),
        'goal': np.stack([record.goal for record in records]),
        'board_goal': np.stack([record.board_goal for record in records]),
        'initial_observation': np.stack([record.initial_observation for record in records]),
        'goal_fingerprint': np.asarray([record.goal_fingerprint for record in records], dtype='U64'),
        'board_goal_fingerprint': np.asarray(
            [record.board_goal_fingerprint for record in records], dtype='U64'
        ),
        'initial_observation_fingerprint': np.asarray(
            [record.initial_observation_fingerprint for record in records], dtype='U64'
        ),
    }


def _arrays_to_records(arrays):
    count = int(arrays['task_id'].shape[0])
    for name in _ARRAY_FIELDS:
        if arrays[name].shape[0] != count:
            raise ControlledGoalReplayError(f'Controlled replay leading dimension mismatch: {name}')
    records = []
    for values in zip(*(arrays[name] for name in _ARRAY_FIELDS), strict=True):
        (
            identifier,
            task_id,
            episode_index,
            evaluation_seed,
            task_seed,
            episode_seed,
            actor_seed,
            noise_seed,
            goal,
            board_goal,
            initial_observation,
            goal_fingerprint,
            board_goal_fingerprint,
            initial_observation_fingerprint,
        ) = values
        records.append(ControlledGoalRecord(
            paired_episode_id=str(identifier),
            task_id=int(task_id),
            episode_index=int(episode_index),
            evaluation_seed=int(evaluation_seed),
            task_seed=int(task_seed),
            episode_seed=int(episode_seed),
            actor_seed=int(actor_seed),
            noise_seed=int(noise_seed),
            goal=np.asarray(goal).copy(),
            board_goal=np.asarray(board_goal, dtype=np.uint8).copy(),
            initial_observation=np.asarray(initial_observation).copy(),
            goal_fingerprint=str(goal_fingerprint),
            board_goal_fingerprint=str(board_goal_fingerprint),
            initial_observation_fingerprint=str(initial_observation_fingerprint),
        ))
    return records


class ControlledGoalReplayEnv:
    """Environment proxy that replaces only policy-facing reset goal vectors."""

    def __init__(self, env, replay):
        if not isinstance(replay, ControlledGoalReplay):
            raise ControlledGoalReplayError('replay must be a ControlledGoalReplay')
        rows, cols = _puzzle_shape(env)
        if (rows, cols) != (replay.rows, replay.cols):
            raise ControlledGoalReplayError(
                f'Puzzle shape mismatch: environment={(rows, cols)}, replay={(replay.rows, replay.cols)}'
            )
        self.env = env
        self.replay = replay
        self._active_record = None
        self._reset_observed = False
        self._last_pairing = None

    @property
    def unwrapped(self):
        return _base_environment(self.env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    @contextmanager
    def paired_episode(self, task_id, episode_index):
        if self._active_record is not None:
            raise ControlledGoalReplayError('A controlled replay episode is already active')
        record = self.replay.record_for(task_id, episode_index)
        self._active_record = record
        self._reset_observed = False
        self._last_pairing = None
        try:
            yield record
            if not self._reset_observed:
                raise ControlledGoalReplayError(
                    f'Controlled replay did not reset {record.paired_episode_id}'
                )
        finally:
            self._active_record = None

    def reset(self, *args, **kwargs):
        record = self._active_record
        if record is None:
            raise ControlledGoalReplayError('Controlled replay reset requires an active paired_episode context')
        if args:
            raise ControlledGoalReplayError('Controlled replay requires keyword reset arguments')
        seed = kwargs.get('seed')
        if seed is None or int(seed) != record.episode_seed:
            raise ControlledGoalReplayError(
                f'Controlled replay seed mismatch for {record.paired_episode_id}: '
                f'expected={record.episode_seed}, observed={seed!r}'
            )
        options = dict(kwargs.get('options') or {})
        if int(options.get('task_id', -1)) != record.task_id:
            raise ControlledGoalReplayError(
                f'Controlled replay task mismatch for {record.paired_episode_id}: '
                f'observed={options.get("task_id")!r}'
            )
        observation, info = self.env.reset(*args, **kwargs)
        if not isinstance(info, Mapping) or 'goal' not in info:
            raise ControlledGoalReplayError(
                f'Environment reset lacks info[goal] for {record.paired_episode_id}'
            )
        rows, cols = self.replay.rows, self.replay.cols
        board_goal = _board_goal(self.env, record.task_id, rows=rows, cols=cols)
        actual_initial_fingerprint = array_fingerprint(observation)
        actual_board_fingerprint = array_fingerprint(board_goal)
        if actual_initial_fingerprint != record.initial_observation_fingerprint:
            raise ControlledGoalReplayError(
                f'Initial observation fingerprint mismatch for {record.paired_episode_id}: '
                f'expected={record.initial_observation_fingerprint}, '
                f'observed={actual_initial_fingerprint}'
            )
        if actual_board_fingerprint != record.board_goal_fingerprint:
            raise ControlledGoalReplayError(
                f'Board goal fingerprint mismatch for {record.paired_episode_id}: '
                f'expected={record.board_goal_fingerprint}, observed={actual_board_fingerprint}'
            )
        _assert_goal_board(
            info['goal'],
            record.board_goal,
            rows=rows,
            cols=cols,
            label=f'Native reset goal for {record.paired_episode_id}',
        )
        replayed_info = dict(info)
        replayed_info['goal'] = record.goal.copy()
        self._reset_observed = True
        self._last_pairing = {
            'paired_episode_id': record.paired_episode_id,
            'goal_fingerprint': record.goal_fingerprint,
            'board_goal_fingerprint': record.board_goal_fingerprint,
            'initial_observation_fingerprint': record.initial_observation_fingerprint,
        }
        return observation, replayed_info

    @property
    def last_pairing(self):
        return None if self._last_pairing is None else dict(self._last_pairing)


def verify_paired_fingerprint_groups(records, *, expected_members):
    """Enforce exact paired input equality across compared-policy episode rows."""

    expected_members = int(expected_members)
    if expected_members <= 0:
        raise ControlledGoalReplayError('expected_members must be positive')
    grouped = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ControlledGoalReplayError('Paired fingerprint record must be a mapping')
        try:
            key = (str(record['environment']), str(record['paired_episode_id']))
            values = (
                str(record['goal_fingerprint']),
                str(record['board_goal_fingerprint']),
                str(record['initial_observation_fingerprint']),
            )
        except KeyError as error:
            raise ControlledGoalReplayError(f'Paired fingerprint record lacks {error.args[0]!r}') from error
        if any(not value or value == 'null' for value in values):
            raise ControlledGoalReplayError(f'Paired fingerprint record has missing values: {key}')
        grouped.setdefault(key, []).append(values)
    for key, values in grouped.items():
        if len(values) != expected_members:
            raise ControlledGoalReplayError(
                f'Paired episode {key} has {len(values)} members; expected={expected_members}'
            )
        if len(set(values)) != 1:
            raise ControlledGoalReplayError(f'Paired input fingerprint mismatch for {key}')
    return {
        'paired_episode_groups': len(grouped),
        'expected_members_per_group': expected_members,
        'status': 'passed',
    }


__all__ = [
    'CONTROLLED_GOAL_REPLAY_SCHEMA_VERSION',
    'ControlledGoalRecord',
    'ControlledGoalReplay',
    'ControlledGoalReplayEnv',
    'ControlledGoalReplayError',
    'array_fingerprint',
    'paired_episode_id',
    'verify_paired_fingerprint_groups',
]
