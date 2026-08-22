"""Launch one stable Study Configuration + Environment + Seed Run.

The wrapper only supplies experiment identity.  Agent arguments after the
identity options are forwarded to ``impls.main`` unchanged.
"""

import argparse
import shlex
import subprocess
import sys

from impls.experiment import make_run_path, prepare_run_design


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-root', default='runs')
    parser.add_argument('--run-attempt', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    study, configuration = prepare_run_design(args.study, args.config)
    if configuration.data.get('executable', True) is False:
        raise SystemExit(
            f'{configuration.config_id} is a planned/non-executable configuration; '
            'no scientific run was started.'
        )
    run_path = make_run_path(
        args.run_root,
        study.study_id,
        configuration.config_id,
        configuration.slug,
        args.environment,
        args.seed,
        run_attempt=args.run_attempt,
    )
    command = [
        sys.executable,
        '-m',
        'impls.main',
        '--study',
        args.study,
        '--config',
        args.config,
        '--env_name',
        args.environment,
        '--seed',
        str(args.seed),
        '--run_root',
        args.run_root,
        '--run_attempt',
        str(args.run_attempt),
    ]
    if not any(item == '--agent' or item.startswith('--agent=') for item in forwarded):
        command.extend(['--agent', study.data['algorithms'][0] if configuration.data.get('algorithm') is None else configuration.data['algorithm']])
    command.extend(forwarded)
    print('run_dir:', run_path)
    print('command:', shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
