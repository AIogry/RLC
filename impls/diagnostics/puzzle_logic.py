"""Diagnostic-only exact logical model for the audited OGBench Puzzle-4x4.

This module deliberately does not participate in environment construction,
training, policy inference, or dataset relabeling.  It turns the *audited*
state-observation convention of the current Puzzle environment into a small
finite transition system for M18-D5.  The implementation is guarded by a
real-environment parity audit before an exact shortest-press distance is used
in a rollout diagnostic.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np


PUZZLE4X4_ROWS = 4
PUZZLE4X4_COLS = 4
PUZZLE4X4_BUTTONS = PUZZLE4X4_ROWS * PUZZLE4X4_COLS
PUZZLE4X4_STATES = 2
ROBOT_DIM = 19
BUTTON_FEATURE_DIM = 4
STATE_FEATURE_DIM = 2


class PuzzleLogicalError(ValueError):
    """Raised when an observation or environment cannot support the oracle."""


def _as_binary_states(states, *, num_buttons=PUZZLE4X4_BUTTONS):
    values = np.asarray(states)
    if values.shape != (int(num_buttons),):
        raise PuzzleLogicalError(
            f'Expected logical button state shape {(int(num_buttons),)!r}, got {values.shape!r}'
        )
    if not np.all(np.isfinite(values)):
        raise PuzzleLogicalError('Logical button states contain non-finite values')
    values = values.astype(np.int8, copy=False)
    if not np.all((values == 0) | (values == 1)):
        raise PuzzleLogicalError('Logical button states must be binary')
    return values


def array_sha256(array):
    """Return a stable identifier for one raw observation/goal array."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('utf-8'))
    digest.update(repr(tuple(value.shape)).encode('utf-8'))
    digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass
class Puzzle4x4LogicalOracle:
    """Exact binary press graph for the audited Puzzle-4x4 implementation.

    Button IDs are canonical row-major IDs.  The affected-button masks are
    derived from the environment's audited ``_num_rows``/``_num_cols`` grid
    contract and then checked against actual ``PuzzleEnv.post_step`` events by
    :func:`audit_real_puzzle_environment` before D5 enables ``d*``.
    """

    rows: int = PUZZLE4X4_ROWS
    cols: int = PUZZLE4X4_COLS
    _distance_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if int(self.rows) != PUZZLE4X4_ROWS or int(self.cols) != PUZZLE4X4_COLS:
            raise PuzzleLogicalError(
                'M18-D5 exact oracle is intentionally restricted to audited Puzzle-4x4; '
                f'got rows={self.rows!r}, cols={self.cols!r}'
            )
        self.rows = int(self.rows)
        self.cols = int(self.cols)
        self._press_masks = tuple(self._build_press_mask(button_id) for button_id in range(self.num_buttons))
        if len(set(self._press_masks)) != self.num_buttons:
            raise PuzzleLogicalError('Puzzle-4x4 press masks are not uniquely identifiable by button ID')

    @property
    def num_buttons(self):
        return self.rows * self.cols

    @property
    def state_count(self):
        return 1 << self.num_buttons

    @property
    def press_masks(self):
        return self._press_masks

    @classmethod
    def from_environment(cls, env):
        """Construct only when the live environment exposes the audited shape."""

        base = getattr(env, 'unwrapped', env)
        values = {
            'rows': getattr(base, '_num_rows', None),
            'cols': getattr(base, '_num_cols', None),
            'buttons': getattr(base, '_num_buttons', None),
            'button_states': getattr(base, '_num_button_states', None),
        }
        if values != {
            'rows': PUZZLE4X4_ROWS,
            'cols': PUZZLE4X4_COLS,
            'buttons': PUZZLE4X4_BUTTONS,
            'button_states': PUZZLE4X4_STATES,
        }:
            raise PuzzleLogicalError(
                'Live environment does not match the audited binary Puzzle-4x4 contract: '
                f'{values!r}'
            )
        return cls(rows=int(values['rows']), cols=int(values['cols']))

    def _build_press_mask(self, button_id):
        button_id = int(button_id)
        if not 0 <= button_id < self.num_buttons:
            raise PuzzleLogicalError(f'Invalid Puzzle button ID: {button_id}')
        row, col = divmod(button_id, self.cols)
        mask = 0
        # This is the exact cardinal-neighbour pattern audited from the local
        # PuzzleEnv.post_step implementation.  It is parameterized by the live
        # environment grid dimensions rather than by a separate task table.
        for delta_row, delta_col in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + delta_row, col + delta_col
            if 0 <= next_row < self.rows and 0 <= next_col < self.cols:
                mask |= 1 << (next_row * self.cols + next_col)
        return int(mask)

    def extract_logical_state(self, observation):
        """Strictly recover binary button states from a state/goal observation."""

        value = np.asarray(observation)
        expected_dim = ROBOT_DIM + self.num_buttons * BUTTON_FEATURE_DIM
        if value.ndim < 1 or value.shape[-1] != expected_dim:
            raise PuzzleLogicalError(
                'Puzzle-4x4 observation must end in exactly '
                f'{expected_dim} features, got shape {value.shape!r}'
            )
        blocks = value[..., ROBOT_DIM:].reshape(*value.shape[:-1], self.num_buttons, BUTTON_FEATURE_DIM)
        one_hot = np.asarray(blocks[..., :STATE_FEATURE_DIM], dtype=np.float64)
        if not np.all(np.isfinite(one_hot)):
            raise PuzzleLogicalError('Puzzle button one-hot encoding contains non-finite values')
        # Standard OGBench states are exact one-hot vectors.  A tight tolerance
        # keeps the extractor robust to float32 serialization without guessing
        # a logical state from arbitrary continuous values.
        if not np.allclose(one_hot.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6):
            raise PuzzleLogicalError('Puzzle button state features do not sum to one')
        if not np.all((one_hot >= -1e-6) & (one_hot <= 1.0 + 1e-6)):
            raise PuzzleLogicalError('Puzzle button state features are outside one-hot bounds')
        decoded = np.argmax(one_hot, axis=-1).astype(np.int8)
        reconstructed = np.eye(PUZZLE4X4_STATES, dtype=np.float64)[decoded]
        if not np.allclose(one_hot, reconstructed, rtol=0.0, atol=1e-6):
            raise PuzzleLogicalError('Puzzle button state features are not uniquely one-hot encoded')
        return decoded

    def encode(self, states):
        states = _as_binary_states(states, num_buttons=self.num_buttons)
        code = 0
        for button_id, state in enumerate(states):
            code |= int(state) << button_id
        return int(code)

    def decode(self, code):
        code = int(code)
        if not 0 <= code < self.state_count:
            raise PuzzleLogicalError(f'Logical state code outside [0, {self.state_count}): {code}')
        return np.asarray([(code >> button_id) & 1 for button_id in range(self.num_buttons)], dtype=np.int8)

    def transition_code(self, code, pressed_button_id):
        code = int(code)
        pressed_button_id = int(pressed_button_id)
        if not 0 <= code < self.state_count:
            raise PuzzleLogicalError(f'Logical state code outside [0, {self.state_count}): {code}')
        if not 0 <= pressed_button_id < self.num_buttons:
            raise PuzzleLogicalError(f'Invalid Puzzle pressed button ID: {pressed_button_id}')
        return int(code ^ self.press_masks[pressed_button_id])

    def transition_states(self, states, pressed_button_id):
        return self.decode(self.transition_code(self.encode(states), pressed_button_id))

    def classify_observed_transition(self, previous_states, next_states):
        """Classify a state transition without inferring an unobserved intention.

        A changed logical configuration is authoritative from the state
        observation.  A button ID is emitted only when its single-press mask
        uniquely equals the observed XOR change.  Multi-press or otherwise
        unrecognized changes remain explicitly unidentified.
        """

        previous_code = self.encode(previous_states)
        next_code = self.encode(next_states)
        changed_mask = previous_code ^ next_code
        if changed_mask == 0:
            return {
                'logical_transition_event': False,
                'verified_single_press_event': False,
                'pressed_button_id': None,
                'press_event_kind': 'no_logical_transition',
                'changed_mask': 0,
            }
        candidates = [button_id for button_id, mask in enumerate(self.press_masks) if mask == changed_mask]
        if len(candidates) == 1:
            return {
                'logical_transition_event': True,
                'verified_single_press_event': True,
                'pressed_button_id': int(candidates[0]),
                'press_event_kind': 'verified_single_press',
                'changed_mask': int(changed_mask),
            }
        return {
            'logical_transition_event': True,
            'verified_single_press_event': False,
            'pressed_button_id': None,
            'press_event_kind': 'unidentified_logical_transition',
            'changed_mask': int(changed_mask),
        }

    def distances_to_goal_code(self, goal_code):
        """Return reverse-BFS shortest valid-press distances for one goal."""

        goal_code = int(goal_code)
        if not 0 <= goal_code < self.state_count:
            raise PuzzleLogicalError(f'Goal code outside [0, {self.state_count}): {goal_code}')
        cached = self._distance_cache.get(goal_code)
        if cached is not None:
            return cached
        distances = np.full(self.state_count, -1, dtype=np.int32)
        distances[goal_code] = 0
        queue = deque([goal_code])
        while queue:
            current = queue.popleft()
            next_distance = int(distances[current]) + 1
            for press_mask in self.press_masks:
                predecessor = current ^ press_mask
                if distances[predecessor] < 0:
                    distances[predecessor] = next_distance
                    queue.append(predecessor)
        self._distance_cache[goal_code] = distances
        return distances

    def distance(self, states, goal_states):
        distances = self.distances_to_goal_code(self.encode(goal_states))
        value = int(distances[self.encode(states)])
        return None if value < 0 else value

    def optimal_pressed_buttons(self, states, goal_states):
        state_code = self.encode(states)
        goal_code = self.encode(goal_states)
        distances = self.distances_to_goal_code(goal_code)
        distance = int(distances[state_code])
        if distance < 0:
            return ()
        if distance == 0:
            return ()
        return tuple(
            button_id
            for button_id in range(self.num_buttons)
            if int(distances[self.transition_code(state_code, button_id)]) == distance - 1
        )

    def validate_distance_invariants(self, goal_states, *, sample_limit=256):
        """Check exact-distance invariants on a deterministic reachable sample."""

        goal_states = _as_binary_states(goal_states, num_buttons=self.num_buttons)
        goal_code = self.encode(goal_states)
        distances = self.distances_to_goal_code(goal_code)
        if int(distances[goal_code]) != 0:
            raise PuzzleLogicalError('Exact distance invariant failed: d*(goal, goal) != 0')
        if np.any(distances < -1):
            raise PuzzleLogicalError('Exact distance invariant failed: invalid negative distance')
        reachable = np.flatnonzero(distances >= 0)
        if not len(reachable):
            raise PuzzleLogicalError('Exact distance invariant failed: goal has no reachable state')
        # The reverse BFS construction already establishes shortest paths for
        # every discovered state.  Verify explicit one-step optimality on a
        # deterministic spread of reachable non-goal states as an independent
        # executable guard against indexing/mask mistakes.
        non_goal = reachable[reachable != goal_code]
        if len(non_goal):
            indices = np.linspace(0, len(non_goal) - 1, min(int(sample_limit), len(non_goal)), dtype=np.int64)
            for state_code in non_goal[indices]:
                distance = int(distances[state_code])
                has_optimal_press = any(
                    int(distances[int(state_code) ^ press_mask]) == distance - 1
                    for press_mask in self.press_masks
                )
                if not has_optimal_press:
                    raise PuzzleLogicalError(
                        f'Exact distance invariant failed at state={int(state_code)}, goal={goal_code}'
                    )
        return {
            'goal_code': int(goal_code),
            'reachable_state_count': int(len(reachable)),
            'state_count': int(self.state_count),
            'goal_distance': int(distances[goal_code]),
            'nonnegative_reachable_distances': True,
            'optimal_press_monotonicity_checked': int(min(int(sample_limit), len(non_goal))),
        }


def audit_real_puzzle_environment(env, *, validation_seed=18018, transition_cases=16):
    """Validate extraction and transition parity against the live PuzzleEnv.

    The transition audit invokes the environment's own ``pre_step`` and
    ``post_step`` threshold-event code on controlled joint positions.  It does
    not monkeypatch environment behaviour, and callers reset before any policy
    rollout.  Failure is reported structurally so D5 can disable exact ``d*``
    rather than silently using a heuristic.
    """

    result = {
        'exact_shortest_distance_available': False,
        'environment_semantics_audit_passed': False,
        'transition_cases_requested': int(transition_cases),
        'transition_cases_passed': 0,
        'errors': [],
    }
    try:
        oracle = Puzzle4x4LogicalOracle.from_environment(env)
        base = getattr(env, 'unwrapped', env)
        rng = np.random.default_rng(int(validation_seed))
        task_goal_states = []
        for task_id in range(1, 6):
            observation, info = env.reset(
                seed=int(validation_seed) + task_id,
                options={'task_id': task_id, 'render_goal': False},
            )
            observed_state = oracle.extract_logical_state(observation)
            observed_goal = oracle.extract_logical_state(np.asarray(info['goal']))
            if not np.array_equal(observed_state, np.asarray(base._cur_button_states, dtype=np.int8)):
                raise PuzzleLogicalError(f'Observation extraction mismatch for task {task_id}')
            if not np.array_equal(observed_goal, np.asarray(base._target_button_states, dtype=np.int8)):
                raise PuzzleLogicalError(f'Goal extraction mismatch for task {task_id}')
            task_goal_states.append(observed_goal.copy())

        # Use the actual environment event condition: pre-step records the
        # official previous joint position, then crossing <= -0.02 makes
        # PuzzleEnv.post_step update its logical state.
        import mujoco

        for case_index in range(int(transition_cases)):
            task_id = case_index % 5 + 1
            env.reset(
                seed=int(validation_seed) + 100 + case_index,
                options={'task_id': task_id, 'render_goal': False},
            )
            states = rng.integers(0, 2, size=oracle.num_buttons, dtype=np.int8)
            base.set_state(base._data.qpos.copy(), base._data.qvel.copy(), states)
            button_id = int(rng.integers(0, oracle.num_buttons))
            before = np.asarray(base._cur_button_states, dtype=np.int8).copy()
            base.pre_step()
            base._data.joint(f'buttonbox_joint_{button_id}').qpos[0] = -0.03
            mujoco.mj_forward(base._model, base._data)
            base.post_step()
            after = np.asarray(base._cur_button_states, dtype=np.int8).copy()
            expected = oracle.transition_states(before, button_id)
            if not np.array_equal(after, expected):
                raise PuzzleLogicalError(
                    f'Real post_step parity failed for case={case_index}, button={button_id}'
                )
            observed_after = oracle.extract_logical_state(base.compute_observation())
            if not np.array_equal(after, observed_after):
                raise PuzzleLogicalError(f'Post-step observation extraction mismatch for case={case_index}')
            event = oracle.classify_observed_transition(before, after)
            if not event['verified_single_press_event'] or event['pressed_button_id'] != button_id:
                raise PuzzleLogicalError(
                    f'Pressed-button identification mismatch for case={case_index}, button={button_id}'
                )
            result['transition_cases_passed'] += 1

        invariants = [oracle.validate_distance_invariants(goal) for goal in task_goal_states]
        result.update({
            'exact_shortest_distance_available': True,
            'environment_semantics_audit_passed': True,
            'environment_class': type(base).__name__,
            'rows': oracle.rows,
            'cols': oracle.cols,
            'num_buttons': oracle.num_buttons,
            'num_button_states': PUZZLE4X4_STATES,
            'observation_dimension': ROBOT_DIM + oracle.num_buttons * BUTTON_FEATURE_DIM,
            'observation_state_encoding': 'button block[:2] strict binary one-hot',
            'goal_state_encoding': 'same standard state-observation encoding in reset info[goal]',
            'button_indexing': 'row-major: button_id = row * num_cols + col',
            'valid_press_transition': 'self plus in-bounds cardinal neighbours, binary toggle modulo 2',
            'physical_event_source': 'PuzzleEnv.post_step joint threshold crossing (> -0.02 to <= -0.02)',
            'logical_state_info_fields': ['prev_button_states', 'button_states'],
            'task_success_relation': 'all current logical button states equal target button states',
            'distance_invariants': invariants,
        })
    except BaseException as error:  # D5 must degrade safely instead of guessing d*.
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result
