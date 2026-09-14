#!/usr/bin/env python3
"""Synthetic architecture/optimization gate for the linear Boolean flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from impls.agents.control_coordinate import ControlCoordinateAgent
from impls.diagnostics.puzzle.control_coordinates import gf2_apply, gf2_rank
from tools.control_coordinate_common import json_print, json_write


def random_invertible_gf2_matrix(num_bits, rng):
    """Draw a random full-rank matrix without consulting any Puzzle algebra."""

    while True:
        candidate = rng.integers(
            0, 2, size=(num_bits, num_bits), dtype=np.uint8
        )
        if gf2_rank(candidate) == num_bits:
            return candidate


def _parser():
    parser = argparse.ArgumentParser(
        description='Train a Boolean flow toward a random synthetic GL(N,2) map.'
    )
    parser.add_argument('--num-bits', type=int, action='append', dest='dimensions')
    parser.add_argument('--num-layers', type=int, default=32)
    parser.add_argument('--permutation-seed', type=int, default=25_001)
    parser.add_argument('--coupling-logit-init-mean', type=float, default=-2.0)
    parser.add_argument('--coupling-logit-init-std', type=float, default=0.1)
    parser.add_argument('--binary-temperature', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=3e-3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=10_000)
    parser.add_argument('--train-size', type=int, default=8_192)
    parser.add_argument('--heldout-size', type=int, default=2_048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=1_000)
    parser.add_argument(
        '--include-full-basis-every-step',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Include all N identifying basis pairs in every supervised batch; '
            'remaining examples are sampled uniformly from random states.'
        ),
    )
    parser.add_argument('--require-exact', action='store_true')
    parser.add_argument('--output', type=Path, default=None)
    return parser


def run_dimension(args, num_bits):
    if num_bits < 2:
        raise ValueError('The Boolean-flow capacity gate requires N >= 2')
    if args.train_size < num_bits:
        raise ValueError('--train-size must be at least N so the basis can be included')
    if args.heldout_size <= 0:
        raise ValueError('--heldout-size must be positive')
    synthetic_data_seed = int(args.seed + 10_000 * num_bits)
    rng = np.random.default_rng(synthetic_data_seed)
    target_matrix = random_invertible_gf2_matrix(num_bits, rng)
    train_inputs = rng.integers(
        0, 2, size=(args.train_size, num_bits), dtype=np.uint8
    )
    # Include the full basis so the supervised set identifies the entire map.
    train_inputs[:num_bits] = np.eye(num_bits, dtype=np.uint8)
    train_targets = gf2_apply(target_matrix, train_inputs)
    basis_inputs = np.eye(num_bits, dtype=np.uint8)
    basis_targets = gf2_apply(target_matrix, basis_inputs)
    heldout_inputs = rng.integers(
        0, 2, size=(args.heldout_size, num_bits), dtype=np.uint8
    )
    heldout_targets = gf2_apply(target_matrix, heldout_inputs)
    config = {
        'num_bits': num_bits,
        'num_layers': args.num_layers,
        'permutation_seed': args.permutation_seed + num_bits,
        'coupling_logit_init_mean': args.coupling_logit_init_mean,
        'coupling_logit_init_std': args.coupling_logit_init_std,
        'binary_temperature': args.binary_temperature,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'train_steps': args.steps,
        'seed': args.seed + num_bits,
        'event_sampling_mode': 'uniform',
    }
    agent = ControlCoordinateAgent.create(config)
    last_info = None
    for step in range(1, args.steps + 1):
        if args.include_full_basis_every_step:
            if args.batch_size < num_bits:
                raise ValueError(
                    '--batch-size must be at least N when the full basis is included'
                )
            random_count = args.batch_size - num_bits
            indices = rng.integers(0, args.train_size, size=random_count)
            batch_inputs = np.concatenate([basis_inputs, train_inputs[indices]], axis=0)
            batch_targets = np.concatenate([basis_targets, train_targets[indices]], axis=0)
        else:
            indices = rng.integers(0, args.train_size, size=args.batch_size)
            batch_inputs = train_inputs[indices]
            batch_targets = train_targets[indices]
        agent, last_info = agent.supervised_update(
            batch_inputs, batch_targets
        )
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f'N={num_bits} step={step} '
                f'loss={float(last_info["loss/supervised_binary"]):.6f} '
                f'exact={float(last_info["metric/hard_exact_state_accuracy"]):.6f}',
                file=sys.stderr,
            )

    predictions = np.asarray(agent.encode(heldout_inputs.astype(np.float32))).astype(
        np.uint8
    )
    learned_matrix = np.asarray(agent.effective_matrix()).astype(np.uint8)
    exact_states = np.all(predictions == heldout_targets, axis=-1)
    matrix_exact = np.array_equal(learned_matrix, target_matrix)
    return {
        'num_bits': num_bits,
        'configuration': config,
        'preflight_base_seed': int(args.seed),
        'synthetic_data_seed': synthetic_data_seed,
        'train_size': int(args.train_size),
        'heldout_size': int(args.heldout_size),
        'full_identifying_basis_in_every_batch': bool(
            args.include_full_basis_every_step
        ),
        'target_matrix_rank': gf2_rank(target_matrix),
        'target_matrix_density': float(target_matrix.mean()),
        'target_matrix_binary': target_matrix.tolist(),
        'learned_matrix_rank': gf2_rank(learned_matrix),
        'learned_matrix_density': float(learned_matrix.mean()),
        'learned_matrix_binary': learned_matrix.tolist(),
        'heldout_exact_state_accuracy': float(exact_states.mean()),
        'heldout_bit_accuracy': float(np.mean(predictions == heldout_targets)),
        'target_matrix_exactly_recovered': bool(matrix_exact),
        'capacity_gate_passed': bool(matrix_exact and np.all(exact_states)),
        'final_supervised_loss': float(last_info['loss/supervised_binary']),
    }


def run(args):
    dimensions = tuple(dict.fromkeys(args.dimensions or (20, 24)))
    if args.steps <= 0 or args.batch_size <= 0 or args.log_interval <= 0:
        raise ValueError('--steps, --batch-size, and --log-interval must be positive')
    result = {
        'schema_version': 1,
        'scientific_role': 'architecture_optimization_preflight_not_m25_benchmark',
        'target_source': 'random_invertible_gf2_matrix',
        'puzzle_operation_matrix_used': False,
        'results': [run_dimension(args, num_bits) for num_bits in dimensions],
    }
    result['all_capacity_gates_passed'] = bool(
        all(item['capacity_gate_passed'] for item in result['results'])
    )
    if args.output is not None:
        json_write(args.output, result)
    json_print(result)
    if args.require_exact and not result['all_capacity_gates_passed']:
        raise SystemExit(1)
    return result


def main():
    run(_parser().parse_args())


if __name__ == '__main__':
    main()
