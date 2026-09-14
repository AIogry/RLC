#!/usr/bin/env python3
"""Evaluate one saved M25 Stage-1 Boolean coordinate checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from impls.agents.control_coordinate import ControlCoordinateAgent
from impls.diagnostics.puzzle.control_coordinates import evaluate_control_coordinates
from impls.experiment.management import jsonable
from impls.utils.checkpointing import resolve_checkpoint
from impls.utils.flax_utils import restore_agent_from_checkpoint
from tools.control_coordinate_common import (
    json_print,
    json_write,
    load_puzzle_event_index,
    validate_stage1_layout,
)


def _parser():
    parser = argparse.ArgumentParser(
        description='Run non-oracle diagnostics on a saved M25 coordinate flow.'
    )
    parser.add_argument('--run-dir', required=True, type=Path)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--checkpoint-role', choices=('best', 'last'), default='last')
    selector.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--dataset-dir', type=Path, default=None)
    parser.add_argument('--dataset-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--diagnostic-seed', type=int, default=None)
    parser.add_argument('--matrix-test-states', type=int, default=None)
    parser.add_argument('--goal-pair-count', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None)
    return parser


def _read_json(path):
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON mapping: {path}')
    return value


def run(args):
    run_dir = args.run_dir.resolve()
    metadata = _read_json(run_dir / 'runtime_metadata.json')
    if metadata.get('algorithm') != 'control_coordinate':
        raise ValueError('Source Run algorithm is not control_coordinate')
    resolved = _read_json(run_dir / 'resolved_config.json')
    algorithm_config = resolved.get('algorithm_config') or {}
    flow_config = dict(algorithm_config.get('agent') or {})
    event_config = dict(algorithm_config.get('event_data') or {})
    launcher = dict(algorithm_config.get('launcher') or {})
    if flow_config.get('agent_name', 'control_coordinate') != 'control_coordinate':
        raise ValueError('Run is not a standalone M25 control-coordinate run')
    environment = metadata['environment']
    validate_stage1_layout(
        environment, configured_num_bits=flow_config['num_bits']
    )
    selector = (
        {'selector': 'step', 'step': int(args.checkpoint_step)}
        if args.checkpoint_step is not None
        else args.checkpoint_role
    )
    checkpoint = resolve_checkpoint(run_dir, selector)
    agent = ControlCoordinateAgent.create(flow_config)
    agent = restore_agent_from_checkpoint(agent, checkpoint['checkpoint_path'])

    event_index, data_provenance = load_puzzle_event_index(
        environment,
        dataset_dir=args.dataset_dir or metadata.get('dataset_dir'),
        dataset_seed=args.dataset_seed,
        configured_num_bits=flow_config['num_bits'],
        chunk_size=int(event_config.get('extraction_chunk_size', 65_536)),
    )
    diagnostic_seed = (
        int(args.diagnostic_seed)
        if args.diagnostic_seed is not None
        else int(launcher.get('diagnostic_seed', 0))
    )
    matrix = np.asarray(agent.effective_matrix()).astype(np.uint8)
    diagnostics = evaluate_control_coordinates(
        event_index,
        encode_fn=agent.encode,
        inverse_fn=agent.inverse_encode,
        goal_mask_fn=agent.goal_mask,
        effective_matrix=matrix,
        batch_size=args.batch_size,
        diagnostic_seed=diagnostic_seed,
        matrix_test_states=(
            args.matrix_test_states
            if args.matrix_test_states is not None
            else int(launcher.get('matrix_test_states', 256))
        ),
        goal_pair_count=(
            args.goal_pair_count
            if args.goal_pair_count is not None
            else int(launcher.get('goal_pair_count', 1024))
        ),
        window_scales=tuple(event_config.get(
            'future_window_scales', (1, 2, 4, 8, 16, 32)
        )),
    )
    result = {
        'schema_version': 1,
        'scientific_role': 'M25_stage1_non_oracle_checkpoint_evaluation',
        'source_run': str(run_dir),
        'checkpoint': checkpoint,
        'configuration': jsonable(flow_config),
        'data_provenance': data_provenance,
        'diagnostics': diagnostics,
        'effective_matrix_binary': matrix.tolist(),
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
