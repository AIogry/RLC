#!/usr/bin/env python3
"""Generate one explicit, reproducible transform payload; never a Study."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impls.representation.goal_coordinate_transforms import generate_goal_coordinate_transform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('identity', 'permutation', 'gf2_linear'))
    parser.add_argument('--num-coordinates', '--N', dest='num_coordinates', type=int, required=True)
    parser.add_argument('--transform-seed', type=int, required=True)
    parser.add_argument('--max-attempts', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('output already exists; select a new payload path')
    payload = generate_goal_coordinate_transform(
        kind=args.kind, num_coordinates=args.num_coordinates,
        transform_seed=args.transform_seed, max_attempts=args.max_attempts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'content_sha256': payload['content_sha256'],
                      'rank': payload['rank'], 'kind': payload['kind']}))


if __name__ == '__main__':
    main()
