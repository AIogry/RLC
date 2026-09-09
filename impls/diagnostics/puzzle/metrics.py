"""Post-hoc Puzzle episode metrics over retained canonical trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ...utils.evaluation import extract_episode_success
from .algebra import build_toggle_matrix, compute_dstar
from .events import identify_press_events


class PuzzleMetricError(ValueError):
    """Raised when a retained trajectory is incomplete or inconsistent."""


_REQUIRED_TRANSITION_FIELDS = ('observation', 'next_observation', 'action', 'reward', 'done', 'info')


def _trajectory_records(trajectory):
    if isinstance(trajectory, Mapping):
        missing = [field for field in _REQUIRED_TRANSITION_FIELDS if field not in trajectory]
        if missing:
            raise PuzzleMetricError(f'Trajectory is missing fields: {missing}')
        count = len(trajectory['observation'])
        records = []
        for index in range(count):
            record = {field: trajectory[field][index] for field in _REQUIRED_TRANSITION_FIELDS}
            for field in ('terminated', 'truncated'):
                if field in trajectory:
                    value = trajectory[field]
                    record[field] = value[-1] if isinstance(value, Sequence) else value
            records.append(record)
        return records
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise PuzzleMetricError('trajectory must be a mapping of lists or a sequence of transition mappings')
    records = []
    for index, record in enumerate(trajectory):
        if not isinstance(record, Mapping):
            raise PuzzleMetricError(f'Transition {index} is not a mapping')
        missing = [field for field in _REQUIRED_TRANSITION_FIELDS if field not in record]
        if missing:
            raise PuzzleMetricError(f'Transition {index} is missing fields: {missing}')
        records.append(dict(record))
    return records


def _validate_board(board, *, name, num_buttons):
    board = np.asarray(board)
    if board.shape != (num_buttons,):
        raise PuzzleMetricError(f'{name} must have shape {(num_buttons,)}, got {board.shape}')
    if not np.issubdtype(board.dtype, np.number) and board.dtype != np.bool_:
        raise PuzzleMetricError(f'{name} must be numeric binary data')
    if not np.all(np.isfinite(board)) or not np.all((board == 0) | (board == 1)):
        raise PuzzleMetricError(f'{name} must contain only finite 0/1 values')
    return board.astype(np.uint8)


def _optional_bool(records, field):
    value = records[-1].get(field)
    if value is None:
        return None
    return bool(value)


def _dstar_or_none(board, goal, matrix):
    return compute_dstar(board, goal, matrix)


def analyze_puzzle_episode(
    trajectory,
    *,
    goal_board,
    rows,
    cols,
    max_episode_steps=None,
):
    """Analyze one observation-only canonical Puzzle rollout.

    The returned mapping contains ``episode``, ``press_events``, and
    ``invariants``.  Unreachable GF(2) residuals are represented by null D*
    values and counted as invariant violations; no pseudo-distance is emitted.
    """

    records = _trajectory_records(trajectory)
    if not records:
        raise PuzzleMetricError('Cannot analyze an empty episode trajectory')
    rows, cols = int(rows), int(cols)
    num_buttons = rows * cols
    goal_board = _validate_board(goal_board, name='goal_board', num_buttons=num_buttons)
    matrix = build_toggle_matrix(rows, cols)
    if max_episode_steps is not None:
        max_episode_steps = int(max_episode_steps)
        if max_episode_steps <= 0:
            raise PuzzleMetricError('max_episode_steps must be positive or None')

    invariants = {
        'event_effect_consistency_failures': 0,
        'unreachable_residuals': 0,
        'malformed_board_observations': 0,
        'multi_source_events': 0,
    }
    press_events = []
    dstar_values = []
    single_events = 0
    progress_count = 0
    neutral_count = 0
    regress_count = 0
    progress_advantages = []
    event_steps = []

    for transition_index, record in enumerate(records):
        try:
            event = identify_press_events(
                record['observation'],
                record['next_observation'],
                rows=rows,
                cols=cols,
            )
        except (TypeError, ValueError) as error:
            invariants['malformed_board_observations'] += 1
            raise PuzzleMetricError(
                f'Malformed Puzzle transition at index {transition_index}: {error}'
            ) from error

        before_dstar = _dstar_or_none(event.board_before, goal_board, matrix)
        after_dstar = _dstar_or_none(event.board_after, goal_board, matrix)
        if before_dstar is None:
            invariants['unreachable_residuals'] += 1
        if after_dstar is None:
            invariants['unreachable_residuals'] += 1
        dstar_values.extend(value for value in (before_dstar, after_dstar) if value is not None)

        predicted_delta = np.zeros(num_buttons, dtype=np.uint8)
        effect_consistent = True
        if event.has_press:
            predicted_delta = (
                matrix.astype(np.int64) @ event.source_mask.astype(np.int64) % 2
            ).astype(np.uint8)
            effect_consistent = bool(np.array_equal(predicted_delta, event.board_delta))
        elif np.any(event.board_delta):
            effect_consistent = False
        if not effect_consistent:
            invariants['event_effect_consistency_failures'] += 1

        if not event.has_press:
            continue
        if event.is_multi_press:
            invariants['multi_source_events'] += 1
        step = transition_index + 1
        delta_dstar = (
            None
            if before_dstar is None or after_dstar is None
            else int(after_dstar - before_dstar)
        )
        if event.is_single_press:
            single_events += 1
            if delta_dstar is None:
                progress_class = 'invariant_violation'
                random_probability = None
                progress_advantage = None
            elif delta_dstar < 0:
                progress_count += 1
                progress_class = 'progress'
                random_probability = None if (rows, cols) == (4, 4) else float(before_dstar / num_buttons)
                progress_advantage = (
                    None if random_probability is None
                    else float(1.0 - random_probability)
                )
            elif delta_dstar == 0:
                neutral_count += 1
                progress_class = 'neutral'
                random_probability = None if (rows, cols) == (4, 4) else float(before_dstar / num_buttons)
                progress_advantage = (
                    None if random_probability is None
                    else float(-random_probability)
                )
            else:
                regress_count += 1
                progress_class = 'regress'
                random_probability = None if (rows, cols) == (4, 4) else float(before_dstar / num_buttons)
                progress_advantage = (
                    None if random_probability is None
                    else float(-random_probability)
                )
            if progress_advantage is not None:
                progress_advantages.append(progress_advantage)
        else:
            progress_class = 'invariant_violation' if delta_dstar is None else 'multi_source_descriptive'
            random_probability = None
            progress_advantage = None

        previous_step = event_steps[-1] if event_steps else None
        event_steps.append(step)
        press_events.append({
            'transition_index': transition_index,
            'step': step,
            'source_button_indices': list(event.source_indices),
            'source_mask': event.source_mask.copy(),
            'num_sources': event.num_sources,
            'event_type': event.event_type,
            'board_before': event.board_before.copy(),
            'board_after': event.board_after.copy(),
            'board_delta': event.board_delta.copy(),
            'predicted_board_delta': predicted_delta,
            'effect_consistent': effect_consistent,
            'goal_board': goal_board.copy(),
            'Dstar_before': before_dstar,
            'Dstar_after': after_dstar,
            'delta_Dstar': delta_dstar,
            'progress_class': progress_class,
            'random_progress_probability': random_probability,
            'progress_advantage': progress_advantage,
            'steps_since_previous_press': None if previous_step is None else step - previous_step,
            'remaining_episode_steps': (
                None if max_episode_steps is None else max_episode_steps - step
            ),
        })

    initial_board = identify_press_events(
        records[0]['observation'], records[0]['next_observation'], rows=rows, cols=cols
    ).board_before
    final_board = identify_press_events(
        records[-1]['observation'], records[-1]['next_observation'], rows=rows, cols=cols
    ).board_after
    initial_dstar = _dstar_or_none(initial_board, goal_board, matrix)
    final_dstar = _dstar_or_none(final_board, goal_board, matrix)
    if initial_dstar is None:
        invariants['unreachable_residuals'] += 1
    if final_dstar is None:
        invariants['unreachable_residuals'] += 1
    if initial_dstar is not None:
        dstar_values.append(initial_dstar)
    if final_dstar is not None:
        dstar_values.append(final_dstar)

    terminated = _optional_bool(records, 'terminated')
    truncated = _optional_bool(records, 'truncated')
    if truncated is None or max_episode_steps is None:
        horizon_exhausted = None
    else:
        horizon_exhausted = bool(truncated and len(records) >= max_episode_steps)
    success = float(extract_episode_success(records[-1]['info']))
    interval_values = [
        event['steps_since_previous_press']
        for event in press_events
        if event['steps_since_previous_press'] is not None
    ]
    progress_rate = None if single_events == 0 else float(progress_count / single_events)
    episode = {
        'success': success,
        'episode_length': len(records),
        'terminated': terminated,
        'truncated': truncated,
        'horizon_exhausted': horizon_exhausted,
        'max_episode_steps': max_episode_steps,
        'initial_Dstar': initial_dstar,
        'final_Dstar': final_dstar,
        'min_Dstar': min(dstar_values) if dstar_values else None,
        'num_press_events': len(press_events),
        'num_single_press_events': single_events,
        'num_multi_press_events': invariants['multi_source_events'],
        'num_source_presses': sum(event['num_sources'] for event in press_events),
        'num_unique_source_buttons': len({
            index for event in press_events for index in event['source_button_indices']
        }),
        'num_repeat_source_presses': (
            sum(event['num_sources'] for event in press_events)
            - len({index for event in press_events for index in event['source_button_indices']})
        ),
        'first_press_step': press_events[0]['step'] if press_events else None,
        'num_progress_presses': progress_count,
        'num_neutral_presses': neutral_count,
        'num_regress_presses': regress_count,
        'progress_rate': progress_rate,
        'mean_progress_advantage': (
            None if not progress_advantages else float(np.mean(progress_advantages))
        ),
        'mean_steps_between_press_events': None if len(press_events) < 2 else float(np.mean(interval_values)),
        'median_steps_between_press_events': None if len(press_events) < 2 else float(np.median(interval_values)),
        'max_steps_between_press_events': None if len(press_events) < 2 else int(max(interval_values)),
        'final_board_goal_consistent': bool(np.array_equal(final_board, goal_board)),
    }
    return {'episode': episode, 'press_events': press_events, 'invariants': invariants}


__all__ = ['PuzzleMetricError', 'analyze_puzzle_episode']
