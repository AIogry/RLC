"""Small GPU worker for planned Study runs.

This is intentionally a local worker, not a distributed scheduler.  It is
safe by default: execution requires ``--execute`` and the training protocol
must be supplied explicitly after user confirmation.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from impls.experiment import load_study, make_run_path, prepare_run_design


def _jobs(study_path, run_root):
    study = load_study(study_path)
    config_dir = Path(study.path).parent / 'configs'
    jobs = []
    for config_path in sorted(config_dir.glob('*.yaml')):
        _, configuration = prepare_run_design(study_path, config_path)
        for environment in study.data['environments']:
            for seed in study.data['seeds']:
                run_dir = make_run_path(
                    run_root,
                    study.study_id,
                    configuration.config_id,
                    configuration.slug,
                    environment,
                    seed,
                )
                status = 'planned'
                metadata_path = run_dir / 'runtime_metadata.json'
                if metadata_path.exists():
                    try:
                        import json

                        with metadata_path.open() as file:
                            status = json.load(file).get('status', 'unknown')
                    except (OSError, ValueError):
                        status = 'invalid'
                jobs.append({
                    'study': study,
                    'configuration': configuration,
                    'environment': environment,
                    'seed': int(seed),
                    'run_dir': run_dir,
                    'status': status,
                })
    return jobs


def _command(job, run_root, extra_args):
    return [
        sys.executable,
        '-m',
        'impls.main',
        '--study',
        str(job['study'].path),
        '--config',
        str(job['configuration'].path),
        '--agent',
        job['configuration'].data['algorithm'],
        '--env_name',
        job['environment'],
        '--seed',
        str(job['seed']),
        '--run_root',
        str(run_root),
        *extra_args,
    ]


def _run_one(job, gpu, run_root, extra_args):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    command = _command(job, run_root, extra_args)
    print(f'[gpu={gpu}] start {job["configuration"].config_id} {job["environment"]} seed={job["seed"]}', flush=True)
    result = subprocess.run(command, env=env, check=False)
    print(f'[gpu={gpu}] exit={result.returncode} {job["run_dir"]}', flush=True)
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--gpus', default='0,1', help='Comma-separated CUDA device IDs.')
    parser.add_argument('--run-root', default='runs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Actually launch jobs; otherwise only print the plan.')
    parser.add_argument('--max-runs', type=int, default=None)
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print only the Study status summary; never launch jobs.',
    )
    args, extra_args = parser.parse_known_args(argv)

    jobs = _jobs(args.study, args.run_root)
    # Only untouched planned runs are eligible for automatic dispatch.  A
    # failed run is deliberately retained for diagnosis/restart rather than
    # silently being relaunched by a later sweep invocation; an invalid or
    # currently running run is likewise left untouched.
    pending = [job for job in jobs if job['status'] == 'planned']
    retained = [job for job in jobs if job['status'] != 'planned']
    if args.max_runs is not None:
        pending = pending[:args.max_runs]
    status_counts = {
        'planned': 0,
        'running': 0,
        'completed': 0,
        'failed': 0,
        'aborted': 0,
        'invalid': 0,
    }
    for job in jobs:
        status_counts[job['status']] = status_counts.get(job['status'], 0) + 1
    status_text = ' '.join(f'{key}={value}' for key, value in status_counts.items())
    print(
        f'total={len(jobs)} planned={status_counts["planned"]} '
        f'completed={status_counts["completed"]} failed={status_counts["failed"]} '
        f'running={status_counts["running"]} retained={len(retained)} '
        f'remaining={len(pending)} statuses: {status_text}'
    )
    if not args.summary_only:
        for job in pending:
            print(
                f'[PLANNED] {job["configuration"].config_id} '
                f'{job["configuration"].slug} {job["environment"]} '
                f'seed={job["seed"]} GPU=<pending> run_dir={job["run_dir"]}'
            )
    if args.summary_only or args.dry_run or not args.execute:
        if not args.dry_run:
            print('Execution disabled. Re-run with --execute after confirming the training protocol.')
        return 0

    gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if not gpus:
        raise SystemExit('--gpus must contain at least one device ID')
    failures = 0
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(_run_one, job, gpus[index % len(gpus)], args.run_root, extra_args): job
            for index, job in enumerate(pending)
        }
        for future in as_completed(futures):
            if future.result() != 0:
                failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
