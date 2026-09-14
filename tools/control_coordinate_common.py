"""Shared command-line glue for M25 control-coordinate tools."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from impls.representation.puzzle_effects import (
    detect_board_change_events,
    extract_dataset_boards,
    raw_effect_signature,
)
from impls.utils.effect_datasets import (
    build_effect_event_index,
    episode_terminals_from_compact_valids,
)
from impls.utils.env_utils import make_env_and_datasets, resolve_dataset_dir


_PUZZLE_ENVIRONMENT = re.compile(r'^puzzle-([1-9][0-9]*)x([1-9][0-9]*)-play-v0$')
PRIMARY_LAYOUTS = frozenset({(4, 5), (4, 6)})
CALIBRATION_LAYOUTS = frozenset({(3, 3)})


def puzzle_layout(environment):
    match = _PUZZLE_ENVIRONMENT.fullmatch(environment)
    if match is None:
        raise ValueError(
            'Expected a canonical state-based Puzzle play environment like '
            f'"puzzle-4x5-play-v0", got {environment!r}'
        )
    rows, cols = (int(value) for value in match.groups())
    return rows, cols, rows * cols


def validate_stage1_layout(environment, *, configured_num_bits=None):
    rows, cols, num_bits = puzzle_layout(environment)
    if configured_num_bits is not None and int(configured_num_bits) != num_bits:
        raise ValueError(
            f'Configured num_bits={configured_num_bits} does not match '
            f'{environment} dimension {num_bits}'
        )
    eligible = (rows, cols) in PRIMARY_LAYOUTS | CALIBRATION_LAYOUTS
    if not eligible:
        detail = '4x4 is singular and requires an out-of-scope quotient formulation' if (
            rows, cols
        ) == (4, 4) else 'layout is not declared for strict M25 Stage 1'
        raise ValueError(f'{environment} is not eligible: {detail}')
    return rows, cols, num_bits


def load_puzzle_event_index(
    environment,
    *,
    dataset_dir=None,
    dataset_seed=0,
    configured_num_bits=None,
    chunk_size=65_536,
):
    """Load canonical standard fields and construct the observation-only index."""

    rows, cols, num_bits = puzzle_layout(environment)
    if configured_num_bits is not None and int(configured_num_bits) != num_bits:
        raise ValueError(
            f'Configured num_bits={configured_num_bits} does not match '
            f'{environment} dimension {num_bits}'
        )
    env, dataset, _ = make_env_and_datasets(
        environment,
        seed=int(dataset_seed),
        dataset_seed=int(dataset_seed),
        dataset_dir=dataset_dir,
    )
    try:
        base_env = getattr(env, 'unwrapped', env)
        reported_rows = int(getattr(base_env, '_num_rows', 0))
        reported_cols = int(getattr(base_env, '_num_cols', 0))
        if (reported_rows, reported_cols) != (rows, cols):
            raise ValueError(
                'Environment-reported Puzzle layout does not match dataset name: '
                f'{(reported_rows, reported_cols)} != {(rows, cols)}'
            )
        missing = {'observations', 'actions', 'terminals'} - set(dataset.keys())
        if missing:
            raise ValueError(f'Canonical offline dataset is missing fields: {sorted(missing)}')
        if 'valids' not in dataset:
            raise ValueError(
                'Canonical compact offline dataset is missing its terminal-derived '
                'valid-transition mask'
            )
        if len(dataset['actions']) != len(dataset['observations']):
            raise ValueError('Dataset actions and observations must have identical lengths')
        observation_terminals = episode_terminals_from_compact_valids(
            dataset['valids']
        )
        boards = extract_dataset_boards(
            dataset['observations'],
            num_buttons=num_bits,
            chunk_size=int(chunk_size),
        )
        event_mask = detect_board_change_events(boards)
        event_index = build_effect_event_index(
            boards,
            observation_terminals,
            event_mask=event_mask,
            effect_signature_fn=raw_effect_signature,
        )
    finally:
        env.close()
    return event_index, {
        'environment': environment,
        'rows': rows,
        'cols': cols,
        'environment_reported_rows': reported_rows,
        'environment_reported_cols': reported_cols,
        'num_bits': num_bits,
        'dataset_dir': resolve_dataset_dir(dataset_dir),
        'standard_dataset_fields_required': ['observations', 'actions', 'terminals'],
        'episode_boundary_source': (
            'canonical_compact_valids_derived_only_from_standard_terminals'
        ),
        'compact_valids_used_as_model_input': False,
        'model_input_fields': ['start_board', 'end_board'],
        'actions_used_as_model_input': False,
        'diagnostic_effect_signature_used_as_model_input': False,
    }


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')


def json_print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


__all__ = [
    'CALIBRATION_LAYOUTS',
    'PRIMARY_LAYOUTS',
    'json_print',
    'json_write',
    'load_puzzle_event_index',
    'puzzle_layout',
    'validate_stage1_layout',
]
