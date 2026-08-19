#!/usr/bin/env python3
"""Plan or execute a generic checkpoint reevaluation campaign."""

from __future__ import annotations

import argparse
import copy
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import (
    ReevaluationError,
    aggregate_campaign,
    campaign_root,
    load_reevaluation_spec,
    protocol_fingerprint,
    validate_source_run,
)
from impls.experiment.reevaluation import _git_metadata, _read_json


def _parse_set(value, option, cast=str):
    if value is None:
        return None
    result = set()
    for item in value.split(','):
        item = item.strip()
        if item:
            try:
                result.add(cast(item))
            except ValueError as error:
                raise SystemExit(f'{option} contains an invalid value: {item!r}') from error
    if not result:
        raise SystemExit(f'{option} must contain at least one value')
    return result


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--source-run-root', default=None)
    parser.add_argument('--reeval-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations')
    parser.add_argument('--configs', default=None)
    parser.add_argument('--seeds', default=None)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-runs', type=int, default=None)
    return parser.parse_args(argv)


def _gpu_list(value):
    gpus = [item.strip() for item in value.split(',') if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise SystemExit('--gpus must contain unique physical GPU IDs')
    if any(not item.isdigit() for item in gpus):
        raise SystemExit('--gpus must contain numeric physical GPU IDs')
    return gpus


def _source_candidates(spec, source_run_root, config_filter, seed_filter, check_checkpoint_metadata=True):
    study_root = Path(source_run_root) / spec['source_study_id']
    if not study_root.exists():
        raise SystemExit(f'Source study root does not exist: {study_root}')
    candidates = []
    for metadata_path in sorted(study_root.rglob('runtime_metadata.json')):
        source_dir = metadata_path.parent
        try:
            metadata = _read_json(metadata_path)
            config_id = metadata.get('config_id')
            seed = int(metadata.get('seed'))
            environment = metadata.get('environment')
        except (OSError, TypeError, ValueError):
            candidates.append({'source_run_dir': source_dir, 'status': 'invalid', 'error': 'unreadable metadata'})
            continue
        if config_filter is not None and config_id not in config_filter:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        if environment not in spec['environments']:
            continue
        try:
            provenance = validate_source_run(
                source_dir,
                checkpoint_selector=spec['checkpoint'],
                expected_study_id=spec['source_study_id'],
                expected_environment=environment if len(spec['environments']) == 1 else None,
                check_checkpoint_metadata=check_checkpoint_metadata,
            )
            candidates.append({'source_run_dir': source_dir, 'status': None, 'provenance': provenance})
        except (ReevaluationError, OSError, ValueError) as error:
            candidates.append({'source_run_dir': source_dir, 'status': 'invalid', 'error': str(error)})

    if spec.get('configs') != 'all':
        allowed = set(spec.get('configs') or [])
        candidates = [
            item for item in candidates
            if item.get('provenance', {}).get('source_config_id') in allowed
            or item.get('status') == 'invalid'
        ]
    expected = {
        (config_id, environment, seed)
        for config_id in (
            sorted(spec.get('configs')) if spec.get('configs') != 'all' else
            sorted({item.get('provenance', {}).get('source_config_id') for item in candidates if item.get('provenance')})
        )
        for environment in spec['environments']
        for seed in spec['training_seeds']
    }
    keyed = {
        (
            item.get('provenance', {}).get('source_config_id'),
            item.get('provenance', {}).get('source_environment'),
            item.get('provenance', {}).get('source_training_seed'),
        )
        for item in candidates if item.get('provenance')
    }
    # Include missing expected identities as invalid planning rows so a dry
    # run cannot accidentally report an incomplete campaign as 33 planned.
    for config_id, environment, seed in sorted(expected - keyed):
        candidates.append({
            'source_run_dir': Path(source_run_root) / spec['source_study_id'] / f'{config_id}__missing' / environment / f'seed_{seed:03d}',
            'status': 'invalid',
            'error': 'expected source run was not discovered',
        })
    return sorted(
        candidates,
        key=lambda item: str(item['source_run_dir']),
    )


def _output_dir(spec, reeval_root, provenance):
    return (
        campaign_root(reeval_root, spec)
        / f'{provenance["source_config_id"]}__{provenance["source_config_slug"]}'
        / provenance['source_environment']
        / f'seed_{provenance["source_training_seed"]:03d}'
    )


def _status(item, spec, reeval_root):
    if item.get('status') == 'invalid':
        return 'invalid'
    provenance = item['provenance']
    output_dir = _output_dir(spec, reeval_root, provenance)
    metadata_path = output_dir / 'reevaluation_metadata.json'
    if not metadata_path.exists():
        return 'planned'
    try:
        metadata = _read_json(metadata_path)
        if metadata.get('checkpoint_sha256') != provenance['checkpoint_sha256']:
            return 'invalid'
        requested = metadata.get('requested_checkpoint_selector')
        if requested is not None:
            if requested != spec['checkpoint']:
                return 'invalid'
        elif spec['checkpoint']['selector'] == 'step' and metadata.get('checkpoint_step') != spec['checkpoint']['step']:
            return 'invalid'
        if metadata.get('reevaluation_protocol_fingerprint') != protocol_fingerprint(spec['protocol']):
            return 'invalid'
        return metadata.get('status', 'invalid')
    except (OSError, ValueError, TypeError):
        return 'invalid'


def _clean_worktree(repo_root):
    git = _git_metadata(repo_root)
    if git['git_dirty']:
        raise SystemExit(
            'Formal reevaluation execution requires a clean reevaluation worktree; '
            'use --dry-run or commit/stash changes first.'
        )


def _run_worker(job, gpu, args, spec_path, reeval_root):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name('reevaluate_checkpoint.py')),
        '--spec', str(spec_path),
        '--source-run-dir', str(job['provenance']['source_run_dir']),
        '--reeval-root', str(reeval_root),
        '--assigned-gpu', str(gpu),
    ]
    if args.resume:
        command.append('--resume')
    print(
        f'[gpu={gpu}] start {job["provenance"]["source_config_id"]} '
        f'{job["provenance"]["source_environment"]} '
        f'seed={job["provenance"]["source_training_seed"]}',
        flush=True,
    )
    result = subprocess.run(command, env=env, check=False)
    print(f'[gpu={gpu}] exit={result.returncode}', flush=True)
    return result.returncode


def _dispatch(jobs, gpus, args, spec_path, reeval_root):
    work = queue.Queue()
    for job in jobs:
        work.put(job)

    def worker(gpu):
        failures = 0
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return failures
            try:
                if _run_worker(job, gpu, args, spec_path, reeval_root) != 0:
                    failures += 1
            finally:
                work.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        return sum(future.result() for future in [executor.submit(worker, gpu) for gpu in gpus])


def main(argv=None):
    args = _args(argv)
    if args.dry_run == args.execute:
        raise SystemExit('Exactly one of --dry-run or --execute is required')
    spec = load_reevaluation_spec(args.spec)
    source_run_root = args.source_run_root or spec['source_run_root']
    config_filter = _parse_set(args.configs, '--configs')
    seed_filter = _parse_set(args.seeds, '--seeds', int)
    if seed_filter is None:
        seed_filter = set(spec['training_seeds'])
    if args.max_runs is not None and args.max_runs <= 0:
        raise SystemExit('--max-runs must be positive')
    candidates = _source_candidates(
        spec,
        source_run_root,
        config_filter,
        seed_filter,
        check_checkpoint_metadata=not args.dry_run,
    )
    for item in candidates:
        if item.get('provenance'):
            item['status'] = _status(item, spec, args.reeval_root)
    counts = {key: 0 for key in ('planned', 'completed', 'running', 'failed', 'aborted', 'invalid')}
    for item in candidates:
        counts[item['status']] = counts.get(item['status'], 0) + 1
    task_count = int(spec.get('expected_task_count', 0))
    episodes_per_task = int(spec['protocol']['episodes_per_task'])
    print(f'source runs       = {len(candidates)}')
    print(f'checkpoints       = {len(candidates)}')
    print(f'tasks/checkpoint  = {task_count}')
    print(f'episodes/task     = {episodes_per_task}')
    print(f'episodes/run      = {task_count * episodes_per_task}')
    print(f'total episodes    = {len(candidates) * task_count * episodes_per_task}')
    print('statuses: ' + ' '.join(f'{key}={counts[key]}' for key in counts))
    for item in candidates:
        if item['status'] == 'invalid':
            print(f'[INVALID] {item["source_run_dir"]}: {item.get("error", "invalid source")}', file=sys.stderr)
        elif args.dry_run:
            p = item['provenance']
            print(
                f'[PLANNED:{item["status"]}] {p["source_config_id"]} '
                f'{p["source_environment"]} seed={p["source_training_seed"]} '
                f'output={_output_dir(spec, args.reeval_root, p)}'
            )
    valid = [item for item in candidates if item.get('provenance')]
    pending_statuses = {'planned'} | ({'running', 'failed', 'aborted'} if args.resume else set())
    pending = [item for item in valid if item['status'] in pending_statuses]
    if args.max_runs is not None:
        pending = pending[:args.max_runs]
    if args.dry_run:
        return 0
    _clean_worktree(Path(__file__).resolve().parents[1])
    gpus = _gpu_list(args.gpus)
    failures = _dispatch(pending, gpus, args, Path(args.spec).resolve(), args.reeval_root)
    aggregate_campaign(
        spec,
        reeval_root=args.reeval_root,
        source_runs=[item['provenance'] for item in valid],
    )
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
