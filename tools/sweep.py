"""Small GPU worker for planned Study runs.

This is intentionally a local worker, not a distributed scheduler.  It is
safe by default: execution requires ``--execute`` and the training protocol
must be supplied explicitly after user confirmation.
"""

import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path

from impls.experiment import (
    ExperimentError,
    load_study,
    make_run_path,
    prepare_run_design,
    validate_source_run_dependency,
)


def _parse_config_ids(value, option):
    if value is None:
        return None
    config_ids = {item.strip() for item in value.split(',') if item.strip()}
    if not config_ids:
        raise SystemExit(f'{option} must contain at least one config_id')
    return config_ids


def _parse_gpus(value):
    gpus = [gpu.strip() for gpu in value.split(',') if gpu.strip()]
    if not gpus:
        raise SystemExit('--gpus must contain at least one device ID')
    if len(set(gpus)) != len(gpus):
        raise SystemExit('--gpus must contain unique physical device IDs')
    return gpus


def _parse_jobs_per_gpu(value):
    try:
        jobs_per_gpu = int(value)
    except (TypeError, ValueError) as error:
        raise SystemExit('--jobs-per-gpu must be a positive integer') from error
    if jobs_per_gpu <= 0:
        raise SystemExit('--jobs-per-gpu must be a positive integer')
    return jobs_per_gpu


def _worker_slots(gpus, jobs_per_gpu=1):
    """Expand unique physical GPUs into explicit reusable worker slots."""

    gpus = _parse_gpus(','.join(str(gpu) for gpu in gpus))
    jobs_per_gpu = _parse_jobs_per_gpu(jobs_per_gpu)
    return [
        {'physical_gpu_id': gpu, 'worker_slot': slot}
        for gpu in gpus
        for slot in range(jobs_per_gpu)
    ]


def _validate_dataset(study_path, dataset_root, *, allow_missing=False):
    """Fail fast unless every Study environment has train and validation data."""

    study = load_study(study_path)
    dataset_root = Path(dataset_root)
    required = [
        dataset_root / f'{environment}.npz'
        for environment in study.data['environments']
    ] + [
        dataset_root / f'{environment}-val.npz'
        for environment in study.data['environments']
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = '\n'.join(f'  {path}' for path in missing)
        if allow_missing:
            print(
                f'Dataset preflight warning (dry-run only); missing files:\n{missing_text}',
                file=sys.stderr,
            )
            return required
        raise SystemExit(f'Dataset preflight failed; missing files:\n{missing_text}')
    return required


def _jobs(study_path, run_root, include_configs=None, exclude_configs=None, run_attempt=0):
    study = load_study(study_path)
    config_dir = Path(study.path).parent / 'configs'
    configurations = [
        prepare_run_design(study_path, config_path)[1]
        for config_path in sorted(config_dir.glob('*.yaml'))
    ]
    known_config_ids = {configuration.config_id for configuration in configurations}
    requested_config_ids = set(include_configs or ()) | set(exclude_configs or ())
    unknown = requested_config_ids - known_config_ids
    if unknown:
        unknown_text = ','.join(sorted(unknown))
        raise SystemExit(f'Unknown config_id(s) for {study.study_id}: {unknown_text}')
    if include_configs is not None and exclude_configs is not None:
        raise SystemExit('--configs and --exclude-configs are mutually exclusive')
    if include_configs is not None:
        configurations = [
            configuration
            for configuration in configurations
            if configuration.config_id in include_configs
        ]
    elif exclude_configs is not None:
        configurations = [
            configuration
            for configuration in configurations
            if configuration.config_id not in exclude_configs
        ]

    jobs = []
    for configuration in configurations:
        environments = (
            [configuration.data['environment']]
            if 'environment' in configuration.data
            else study.data['environments']
        )
        for environment in environments:
            for seed in study.data['seeds']:
                dependency_specs = configuration.data.get('dependencies', {})
                if isinstance(dependency_specs, dict):
                    for dependency_name in dependency_specs:
                        validate_source_run_dependency(
                            study,
                            configuration,
                            dependency_name,
                            seed=seed,
                            run_root=run_root,
                        )
                run_dir = make_run_path(
                    run_root,
                    study.study_id,
                    configuration.config_id,
                    configuration.slug,
                    environment,
                    seed,
                    run_attempt=run_attempt,
                )
                # Phase-1 studies may intentionally publish a complete
                # future factorial design while withholding permission to run
                # it.  Such a skeleton is never a pending job, even if no
                # runtime artifact exists at its deterministic path.
                executable = bool(configuration.data.get('executable', True))
                status = 'planned' if executable else 'blocked'
                metadata_path = run_dir / 'runtime_metadata.json'
                if executable and metadata_path.exists():
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
                    'run_attempt': int(run_attempt),
                    'run_dir': run_dir,
                    'status': status,
                    'executable': executable,
                    'blocked_by': configuration.data.get('blocked_by'),
                })
    return jobs


def _study_protocol_defaults(job):
    """Return generic Study protocol defaults for this Configuration stage."""

    protocol = job['study'].data.get('protocol', {})
    if not isinstance(protocol, Mapping):
        return {}
    section_name = job['configuration'].data.get(
        'protocol_stage', job['configuration'].data.get('stage')
    )
    if section_name and isinstance(protocol.get(section_name), Mapping):
        return protocol[section_name]
    return protocol


def _with_study_protocol_defaults(job, extra_args):
    """Fill omitted checkpoint flags from declarative Study protocol values."""

    result = list(extra_args)
    protocol = _study_protocol_defaults(job)
    option_names = {
        'save_best_checkpoint': (
            '--save-best-checkpoint', '--no-save-best-checkpoint',
            '--save_best_checkpoint', '--no_save_best_checkpoint',
        ),
        'save_last_checkpoint': (
            '--save-last-checkpoint', '--no-save-last-checkpoint',
            '--save_last_checkpoint', '--no_save_last_checkpoint',
        ),
    }
    for field, names in option_names.items():
        if field not in protocol:
            continue
        if any(str(argument).split('=', 1)[0] in names for argument in result):
            continue
        result.append(names[0] if bool(protocol[field]) else names[1])
    return result


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
        '--run_attempt',
        str(job['run_attempt']),
        '--run_root',
        str(run_root),
        *_with_study_protocol_defaults(job, extra_args),
    ]


def _run_one(job, gpu, run_root, extra_args, *, worker_slot=0, jobs_per_gpu=1):
    env = os.environ.copy()
    # Avoid JAX's process-wide default preallocation while retaining an
    # explicit caller choice for jobs that need a different memory policy.
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['RLC_ASSIGNED_PHYSICAL_GPU'] = str(gpu)
    env['RLC_WORKER_SLOT'] = str(worker_slot)
    env['RLC_JOBS_PER_GPU'] = str(jobs_per_gpu)
    command = _command(job, run_root, extra_args)
    print(
        f'[gpu={gpu} slot={worker_slot}] start '
        f'{job["configuration"].config_id} {job["environment"]} seed={job["seed"]}',
        flush=True,
    )
    result = subprocess.run(command, env=env, check=False)
    print(
        f'[gpu={gpu} slot={worker_slot}] exit={result.returncode} '
        f'{job["run_dir"]}',
        flush=True,
    )
    return result.returncode


def _dispatch_jobs(pending, gpus, run_root, extra_args, runner=None, jobs_per_gpu=1):
    """Run a dynamic queue with a configurable number of workers per GPU.

    ``runner`` retains its historical four-argument callback contract for
    tests and external callers.  The production runner additionally receives
    the explicit worker slot so resource assignment is visible in logs and
    child-process metadata environment variables.
    """

    worker_slots = _worker_slots(gpus, jobs_per_gpu)
    production_runner = runner is None
    runner = _run_one if production_runner else runner
    job_queue = queue.Queue()
    for job in pending:
        job_queue.put(job)

    def worker(worker_spec):
        gpu = worker_spec['physical_gpu_id']
        worker_slot = worker_spec['worker_slot']
        failures = 0
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return failures
            try:
                if production_runner:
                    result = runner(
                        job,
                        gpu,
                        run_root,
                        extra_args,
                        worker_slot=worker_slot,
                        jobs_per_gpu=jobs_per_gpu,
                    )
                else:
                    result = runner(job, gpu, run_root, extra_args)
                if result != 0:
                    failures += 1
            finally:
                job_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(worker_slots)) as executor:
        futures = [executor.submit(worker, worker_spec) for worker_spec in worker_slots]
        failures = sum(future.result() for future in futures)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--gpus', default='0,1', help='Comma-separated CUDA device IDs.')
    parser.add_argument(
        '--jobs-per-gpu', type=_parse_jobs_per_gpu, default=1,
        help='Number of concurrent worker processes allowed per physical GPU.',
    )
    parser.add_argument('--run-root', default='runs')
    parser.add_argument(
        '--run-attempt', type=int, default=0,
        help='Explicit non-negative rerun instance; zero keeps canonical paths.',
    )
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--configs', default=None, help='Comma-separated config_id allowlist.')
    parser.add_argument('--exclude-configs', default=None, help='Comma-separated config_id blocklist.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Actually launch jobs; otherwise only print the plan.')
    parser.add_argument('--max-runs', type=int, default=None)
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print only the Study status summary; never launch jobs.',
    )
    parser.add_argument(
        '--allow-missing-dataset',
        action='store_true',
        help='Allow missing dataset files for non-executing validation only.',
    )
    args, extra_args = parser.parse_known_args(argv)

    if args.configs is not None and args.exclude_configs is not None:
        raise SystemExit('--configs and --exclude-configs are mutually exclusive')
    include_configs = _parse_config_ids(args.configs, '--configs')
    exclude_configs = _parse_config_ids(args.exclude_configs, '--exclude-configs')
    if args.run_attempt < 0:
        raise SystemExit('--run-attempt must be non-negative')
    if args.dataset_root is not None:
        if args.allow_missing_dataset and args.execute and not args.dry_run:
            raise SystemExit('--allow-missing-dataset is only valid for dry-run/summary validation')
        _validate_dataset(
            args.study,
            args.dataset_root,
            allow_missing=args.allow_missing_dataset and not args.execute,
        )
    try:
        jobs = _jobs(
            args.study,
            args.run_root,
            include_configs=include_configs,
            exclude_configs=exclude_configs,
            run_attempt=args.run_attempt,
        )
    except ExperimentError as error:
        print(f'preflight: NO-GO: {error}', file=sys.stderr)
        return 2
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
        'blocked': 0,
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
    phase2_blocked = [
        job for job in jobs
        if (
            job['configuration'].data.get('protocol_stage') == 'phase2_skeleton'
            and not job['executable']
        )
    ]
    if phase2_blocked:
        # Exact, deliberately boring safety summary for M20A-style studies.
        # It is printed before any plan listing so a mistaken --execute is
        # visibly unable to create a formal training job.
        formal_executable = sum(1 for job in jobs if job['executable'])
        print(f'formal executable runs = {formal_executable}')
        print(f'blocked phase2 skeletons = {len(phase2_blocked)}')
    if not args.summary_only:
        for job in pending:
            semantic = job['configuration'].data.get('semantic_condition', job['configuration'].slug)
            print(
                f'[PLANNED] {job["configuration"].config_id} '
                f'{semantic} algorithm={job["configuration"].data.get("algorithm")} '
                f'{job["environment"]} '
                f'seed={job["seed"]} GPU=<pending> run_dir={job["run_dir"]}'
            )
    if args.summary_only or args.dry_run or not args.execute:
        if not args.dry_run:
            print('Execution disabled. Re-run with --execute after confirming the training protocol.')
        return 0

    if not pending:
        print('No executable planned jobs; blocked configurations were not dispatched.')
        return 0

    gpus = _parse_gpus(args.gpus)
    worker_slots = _worker_slots(gpus, args.jobs_per_gpu)
    print(
        'Worker slots: '
        + ', '.join(
            f'gpu={slot["physical_gpu_id"]}/slot={slot["worker_slot"]}'
            for slot in worker_slots
        )
    )
    failures = _dispatch_jobs(
        pending,
        gpus,
        args.run_root,
        extra_args,
        jobs_per_gpu=args.jobs_per_gpu,
    )
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
