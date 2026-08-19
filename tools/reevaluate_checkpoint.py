#!/usr/bin/env python3
"""Run one provenance-checked checkpoint reevaluation.

This command is intentionally separate from ``impls.main``: it never enters
the training loop and writes only under the reevaluation root.
"""

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import (
    ReevaluationError,
    load_reevaluation_spec,
    run_checkpoint_reevaluation,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='Declarative reevaluation YAML spec.')
    parser.add_argument('--source-run-dir', required=True)
    parser.add_argument('--reeval-root', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--assigned-gpu', default=None)
    parser.add_argument('--repo-root', default=None)
    parser.add_argument('--episodes-per-task', type=int, default=None, help='Explicit smoke override.')
    parser.add_argument('--evaluation-seed', type=int, default=None, help='Explicit smoke override.')
    parser.add_argument('--eval-temperature', type=float, default=None, help='Explicit smoke override.')
    parser.add_argument('--eval-gaussian', type=float, default=None, help='Explicit smoke override.')
    return parser.parse_args(argv)


def _with_overrides(spec, args):
    if not any(value is not None for value in (
        args.episodes_per_task,
        args.evaluation_seed,
        args.eval_temperature,
        args.eval_gaussian,
    )):
        return spec
    spec = copy.deepcopy(spec)
    protocol = spec['protocol']
    if args.episodes_per_task is not None:
        if args.episodes_per_task <= 0:
            raise ValueError('--episodes-per-task must be positive')
        protocol['episodes_per_task'] = args.episodes_per_task
    if args.evaluation_seed is not None:
        protocol['evaluation_seed'] = args.evaluation_seed
    if args.eval_temperature is not None:
        protocol['eval_temperature'] = args.eval_temperature
    if args.eval_gaussian is not None:
        protocol['eval_gaussian'] = args.eval_gaussian
    return spec


def main(argv=None):
    args = _parse_args(argv)
    try:
        spec = _with_overrides(load_reevaluation_spec(args.spec), args)
        summary = run_checkpoint_reevaluation(
            args.source_run_dir,
            spec,
            reeval_root=args.reeval_root,
            resume=args.resume,
            assigned_gpu=args.assigned_gpu,
            repo_root=args.repo_root,
        )
    except (ReevaluationError, FileExistsError, OSError, ValueError) as error:
        print(f'checkpoint reevaluation failed: {error}', file=sys.stderr)
        return 2
    print(
        f'completed checkpoint reevaluation: '
        f'overall_success={summary.get("evaluation/overall_success")} '
        f'total_episodes={summary.get("total_episodes")}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
