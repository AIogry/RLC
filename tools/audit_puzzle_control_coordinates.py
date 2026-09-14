#!/usr/bin/env python3
"""Audit observation-derived Puzzle events without any oracle operation data."""

from __future__ import annotations

import argparse
from pathlib import Path

from impls.diagnostics.puzzle.control_coordinates import audit_effect_event_index
from tools.control_coordinate_common import (
    json_print,
    json_write,
    load_puzzle_event_index,
    puzzle_layout,
)


def _parser():
    parser = argparse.ArgumentParser(
        description='Audit M25 Puzzle event coverage from standard offline observations.'
    )
    parser.add_argument('--environment', required=True)
    parser.add_argument('--dataset-dir', type=Path, default=None)
    parser.add_argument('--num-bits', type=int, default=None)
    parser.add_argument('--dataset-seed', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=65_536)
    parser.add_argument(
        '--window-scales', type=int, nargs='+', default=(1, 2, 4, 8, 16, 32)
    )
    parser.add_argument('--output', type=Path, default=None)
    return parser


def run(args):
    rows, cols, inferred_bits = puzzle_layout(args.environment)
    event_index, provenance = load_puzzle_event_index(
        args.environment,
        dataset_dir=args.dataset_dir,
        dataset_seed=args.dataset_seed,
        configured_num_bits=args.num_bits,
        chunk_size=args.chunk_size,
    )
    audit = audit_effect_event_index(
        event_index, window_scales=tuple(args.window_scales)
    )
    result = {
        'schema_version': 1,
        'scientific_role': 'observation_only_stage1_data_identifiability_audit',
        'layout': {
            'rows': rows,
            'cols': cols,
            'num_bits': inferred_bits,
            'primary_m25_target': (rows, cols) in {(4, 5), (4, 6)},
            'strict_linear_stage1_eligible': (rows, cols) in {(3, 3), (4, 5), (4, 6)},
            'singular_4x4_excluded': (rows, cols) == (4, 4),
        },
        'provenance': provenance,
        'audit': audit,
        'oracle_operation_algebra_used': False,
    }
    if args.output is not None:
        json_write(args.output, result)
    json_print(result)
    return result


def main():
    run(_parser().parse_args())


if __name__ == '__main__':
    main()
