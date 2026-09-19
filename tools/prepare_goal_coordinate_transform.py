#!/usr/bin/env python3
"""Bounded, constrained transform preparation; never used during training.

The returned payload obeys the existing resolver contract. Selection evidence
is separate because acceptance criteria are preparation protocol, not runtime
transform fields. Draws use a private PCG64 Generator, never training RNG.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from impls.representation.goal_coordinate_transforms import resolve_goal_coordinate_transform
from impls.representation.puzzle_algebra import gf2_inverse, gf2_rank


def prepare_transform(*, kind, num_coordinates, transform_seed, max_attempts,
                      derangement=False, min_weight=None, max_weight=None):
    for key, value, minimum in (
        ('num_coordinates', num_coordinates, 1), ('transform_seed', transform_seed, 0),
        ('max_attempts', max_attempts, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    n = num_coordinates
    if kind not in ('permutation', 'gf2_linear'):
        raise ValueError('Preparation supports permutation or gf2_linear')
    if kind == 'permutation':
        if min_weight is not None or max_weight is not None:
            raise ValueError('Weight bounds apply only to gf2_linear')
        if derangement and n == 1:
            raise ValueError('A one-coordinate derangement is impossible')
    else:
        if derangement:
            raise ValueError('Derangement applies only to permutation')
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (min_weight, max_weight)):
            raise ValueError('Explicit integer min_weight and max_weight are required')
        if not 1 <= min_weight <= max_weight <= n:
            raise ValueError('Require 1 <= min_weight <= max_weight <= N')
    rng = np.random.Generator(np.random.PCG64(transform_seed))
    for attempt in range(1, max_attempts + 1):
        if kind == 'permutation':
            permutation = rng.permutation(n)
            if derangement and np.any(permutation == np.arange(n)):
                continue
            raw = {'kind': kind, 'permutation': permutation.tolist()}
        else:
            matrix = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
            weights = np.concatenate((matrix.sum(axis=0), matrix.sum(axis=1)))
            if np.any(weights < min_weight) or np.any(weights > max_weight) or gf2_rank(matrix) != n:
                continue
            raw = {'kind': kind, 'matrix': matrix.tolist()}
        raw['provenance'] = {
            'generator': 'constrained_goal_coordinate_transform', 'version': 1,
            'transform_seed': transform_seed,
        }
        transform = resolve_goal_coordinate_transform(raw, num_coordinates=n)
        inverse = gf2_inverse(transform.matrix)
        return {
            'payload': transform.to_config(),
            'selection': {
                'accepted_candidate_1based': attempt, 'max_attempts': max_attempts,
                'rng': 'numpy.Generator(PCG64)', 'numpy_version': np.__version__,
                'candidate_draw': 'permutation(N)' if kind == 'permutation' else 'integers(0,2,(N,N),dtype=uint8)',
                'derangement': derangement, 'min_weight': min_weight, 'max_weight': max_weight,
                'inverse_rank': gf2_rank(inverse),
                'inverse_row_weights': inverse.sum(axis=1).astype(int).tolist(),
                'inverse_column_weights': inverse.sum(axis=0).astype(int).tolist(),
                'inverse_is_model_input': False,
            },
        }
    raise ValueError(f'No acceptable {kind} in {max_attempts} candidates; do not substitute a seed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('permutation', 'gf2_linear'), required=True)
    parser.add_argument('--num-coordinates', type=int, required=True)
    parser.add_argument('--transform-seed', type=int, required=True)
    parser.add_argument('--max-attempts', type=int, required=True)
    parser.add_argument('--derangement', action='store_true')
    parser.add_argument('--min-weight', type=int)
    parser.add_argument('--max-weight', type=int)
    parser.add_argument('--output', type=Path, help='New JSON file; omit to print only')
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error('output already exists')
    result = prepare_transform(**{k: v for k, v in vars(args).items() if k != 'output'})
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            file.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
