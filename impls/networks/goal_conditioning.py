"""Shared network-input semantics for deterministic goal conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from ..representation.puzzle_algebra import operator_metadata
from ..representation.puzzle_conditioning import (
    PUZZLE_GOAL_CONDITIONING_MODES,
    condition_puzzle_goal,
)


_CONFIG_KEYS = frozenset({
    'domain',
    'mode',
    'rows',
    'cols',
    'num_buttons',
    'robot_dim',
    'button_feature_dim',
    'robot_goal_policy',
    'button_goal_transient_policy',
})
_BOARD_EQUALITY_MODES = frozenset({'board', 'residual', 'oracle_operation'})
_SLOT_LOCAL_GOAL_FIELDS = frozenset({
    'goal_conditioning',
    'goal_conditioning_mode',
})


def _positive_int(config: Mapping, name: str) -> int:
    if name not in config:
        raise ValueError(f'goal_conditioning requires {name!r}')
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(
            f'goal_conditioning.{name} must be a positive integer, got {value!r}'
        )
    return int(value)


def _validate_puzzle_slots(compute_slots, normalized) -> None:
    if compute_slots is None:
        return
    if not hasattr(compute_slots, 'items'):
        raise ValueError('compute slots must be a mapping for layout validation')
    expected = {
        'num_buttons': normalized['num_buttons'],
        'robot_dim': normalized['robot_dim'],
        'button_feature_dim': normalized['button_feature_dim'],
    }
    for slot_name, slot in compute_slots.items():
        if not hasattr(slot, 'get') or not slot.get('enabled', False):
            continue
        if slot.get('structure', 'vector') != 'puzzle_tokens':
            continue
        structure_kwargs = slot.get('structure_kwargs', {})
        if not hasattr(structure_kwargs, 'get'):
            raise ValueError(
                f'compute.{slot_name}.structure_kwargs must be a mapping'
            )
        actual = {
            'num_buttons': structure_kwargs.get('num_buttons'),
            'robot_dim': structure_kwargs.get('robot_dim', 19),
            'button_feature_dim': structure_kwargs.get('button_feature_dim', 4),
        }
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                raise ValueError(
                    'Puzzle goal-conditioning layout conflicts with structured slot: '
                    f'compute.{slot_name}.structure_kwargs.{field}={actual[field]!r}, '
                    f'goal_conditioning.{field}={expected_value!r}'
                )


def _reject_slot_local_goal_conditioning(compute_slots) -> None:
    """Keep scientific goal semantics out of computation-slot config."""

    if compute_slots is None:
        return
    if not hasattr(compute_slots, 'items'):
        raise ValueError('compute slots must be a mapping for layout validation')
    for slot_name, slot in compute_slots.items():
        if not hasattr(slot, 'get'):
            continue
        for container_name, container in (
            ('slot', slot),
            ('structure_kwargs', slot.get('structure_kwargs', {})),
            ('block_kwargs', slot.get('block_kwargs', {})),
        ):
            if not hasattr(container, 'keys'):
                continue
            forbidden = set(container) & _SLOT_LOCAL_GOAL_FIELDS
            if forbidden:
                raise ValueError(
                    'goal conditioning is one shared top-level network semantic; '
                    f'compute.{slot_name}.{container_name} contains forbidden fields '
                    f'{sorted(forbidden)!r}'
                )


def goal_conditioning_layout(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Validate shared Puzzle layout fields without constructing an operator."""

    _reject_slot_local_goal_conditioning(compute_slots)
    if config is None:
        if dataset_class == 'PuzzleBoardGCDataset':
            raise ValueError(
                'PuzzleBoardGCDataset requires an explicit goal_conditioning layout'
            )
        return None
    if not isinstance(config, Mapping) and not hasattr(config, 'items'):
        raise ValueError('goal_conditioning must be a mapping or None')
    unexpected = set(config) - _CONFIG_KEYS
    if unexpected:
        raise ValueError(
            f'Unsupported goal_conditioning fields: {sorted(unexpected)!r}'
        )
    if config.get('domain') != 'puzzle':
        raise ValueError(
            f"goal_conditioning.domain must be 'puzzle', got {config.get('domain')!r}"
        )
    mode = config.get('mode')
    if mode not in PUZZLE_GOAL_CONDITIONING_MODES:
        raise ValueError(
            f'Unsupported goal_conditioning.mode {mode!r}; '
            f'expected one of {sorted(PUZZLE_GOAL_CONDITIONING_MODES)!r}'
        )
    rows = _positive_int(config, 'rows')
    cols = _positive_int(config, 'cols')
    num_buttons = _positive_int(config, 'num_buttons')
    robot_dim = _positive_int(config, 'robot_dim')
    button_feature_dim = _positive_int(config, 'button_feature_dim')
    if rows * cols != num_buttons:
        raise ValueError(
            'goal_conditioning rows*cols must equal num_buttons: '
            f'{rows}*{cols} != {num_buttons}'
        )
    if robot_dim != 19:
        raise ValueError(
            f'goal_conditioning.robot_dim must match canonical Puzzle 19D, got {robot_dim}'
        )
    if button_feature_dim != 4:
        raise ValueError(
            'goal_conditioning.button_feature_dim must match canonical Puzzle 4D, '
            f'got {button_feature_dim}'
        )
    expected_policy = 'preserve' if mode == 'canonical' else 'zero'
    robot_goal_policy = config.get('robot_goal_policy', expected_policy)
    transient_policy = config.get(
        'button_goal_transient_policy', expected_policy
    )
    if robot_goal_policy != expected_policy:
        raise ValueError(
            'goal_conditioning.robot_goal_policy must be '
            f'{expected_policy!r} for mode={mode!r}'
        )
    if transient_policy != expected_policy:
        raise ValueError(
            'goal_conditioning.button_goal_transient_policy must be '
            f'{expected_policy!r} for mode={mode!r}'
        )
    if (
        mode in _BOARD_EQUALITY_MODES
        and dataset_class is not None
        and dataset_class != 'PuzzleBoardGCDataset'
    ):
        raise ValueError(
            f'goal_conditioning.mode={mode!r} requires '
            "dataset_class='PuzzleBoardGCDataset' for board-equality success semantics; "
            f'got {dataset_class!r}'
        )
    normalized = {
        'domain': 'puzzle',
        'mode': mode,
        'rows': rows,
        'cols': cols,
        'num_buttons': num_buttons,
        'robot_dim': robot_dim,
        'button_feature_dim': button_feature_dim,
        'robot_goal_policy': robot_goal_policy,
        'button_goal_transient_policy': transient_policy,
    }
    _validate_puzzle_slots(compute_slots, normalized)
    return normalized


def validate_goal_conditioning_config(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Validate one shared network config, including oracle invertibility."""

    normalized = goal_conditioning_layout(
        config,
        compute_slots=compute_slots,
        dataset_class=dataset_class,
    )
    if normalized is None:
        return None
    if normalized['mode'] == 'oracle_operation':
        metadata = operator_metadata(normalized['rows'], normalized['cols'])
        if not metadata['full_rank']:
            raise ValueError(
                'oracle_operation requires a unique full-rank GF(2) operator: '
                f'layout={normalized["rows"]}x{normalized["cols"]}, '
                f'rank={metadata["rank"]}, '
                f'num_buttons={normalized["num_buttons"]}'
            )
    return normalized


@dataclass(frozen=True)
class PuzzleGoalConditioner:
    """Hashable, parameter-free callable embedded in GC network modules."""

    mode: str
    rows: int
    cols: int
    num_buttons: int
    robot_dim: int
    button_feature_dim: int
    operation_inverse: tuple[tuple[int, ...], ...] | None = None

    def __call__(self, observations, goals):
        return condition_puzzle_goal(
            observations,
            goals,
            mode=self.mode,
            rows=self.rows,
            cols=self.cols,
            num_buttons=self.num_buttons,
            robot_dim=self.robot_dim,
            button_feature_dim=self.button_feature_dim,
            operation_inverse=self.operation_inverse,
        )


def make_goal_conditioner(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Build a parameter-free conditioner from shared network semantics."""

    normalized = validate_goal_conditioning_config(
        config,
        compute_slots=compute_slots,
        dataset_class=dataset_class,
    )
    if normalized is None:
        return None
    operation_inverse = None
    if normalized['mode'] == 'oracle_operation':
        inverse = operator_metadata(normalized['rows'], normalized['cols'])['inverse']
        operation_inverse = tuple(
            tuple(int(item) for item in row) for row in inverse
        )
    return PuzzleGoalConditioner(
        mode=normalized['mode'],
        rows=normalized['rows'],
        cols=normalized['cols'],
        num_buttons=normalized['num_buttons'],
        robot_dim=normalized['robot_dim'],
        button_feature_dim=normalized['button_feature_dim'],
        operation_inverse=operation_inverse,
    )


def goal_conditioning_runtime_metadata(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Return JSON-ready derived provenance without a parallel config system."""

    normalized = validate_goal_conditioning_config(
        config,
        compute_slots=compute_slots,
        dataset_class=dataset_class,
    )
    if normalized is None:
        return {}
    operator = operator_metadata(normalized['rows'], normalized['cols'])
    return {
        'goal_conditioning': {
            **normalized,
            'operator_matrix_source': operator['operator_matrix_source'],
            'operator_orientation': operator['operator_orientation'],
            'operator_rank': operator['rank'],
            'operator_full_rank': operator['full_rank'],
        },
        'dataset_class': dataset_class,
        'goal_success_semantics': (
            'board_equality'
            if dataset_class == 'PuzzleBoardGCDataset'
            else 'timestep_index_equality'
        ),
    }


__all__ = [
    'PuzzleGoalConditioner',
    'goal_conditioning_layout',
    'goal_conditioning_runtime_metadata',
    'make_goal_conditioner',
    'validate_goal_conditioning_config',
]
