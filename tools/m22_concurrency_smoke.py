"""Run two short M22 canonical processes concurrently on one physical GPU.

This reusable probe exercises the existing Study scheduler with a bounded
update budget and a caller-selected pair of configurations.  It writes only
its own report under the supplied smoke root; it is never the formal M22
launcher and does not use the formal run namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import sweep  # noqa: E402


DEFAULT_STUDY = REPO_ROOT / 'experiments/M22_puzzle_baselines_unified_rlc/study.yaml'
DEFAULT_DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
DEFAULT_OUTPUT_ROOT = Path('/tmp/m22_puzzle_baseline_smoke/concurrency')
DEFAULT_CONFIGS = ('M22-CRL-P4X6', 'M22-HIQL-P4X6')


def _metadata(run_dir: Path) -> dict:
    path = run_dir / 'runtime_metadata.json'
    if not path.is_file():
        return {'status': 'missing', 'run_dir': str(run_dir)}
    try:
        with path.open() as file:
            value = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {'status': 'invalid', 'run_dir': str(run_dir), 'error': str(error)}
    return {
        'status': value.get('status'),
        'study_id': value.get('study_id'),
        'config_id': value.get('config_id'),
        'algorithm': value.get('algorithm'),
        'environment': value.get('environment'),
        'seed': value.get('seed'),
        'git_commit': value.get('git_commit'),
        'failure_reason': value.get('failure_reason'),
        'run_dir': str(run_dir),
    }


def run_concurrency_smoke(
    *,
    study_path=DEFAULT_STUDY,
    dataset_root=DEFAULT_DATASET_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    run_root=None,
    config_ids=DEFAULT_CONFIGS,
    gpu='1',
    jobs_per_gpu=2,
    train_steps=2_000,
):
    if str(gpu) != '1':
        raise ValueError('M22 concurrency smoke is restricted to physical GPU 1')
    if int(jobs_per_gpu) != 2:
        raise ValueError('M22 concurrency smoke requires jobs_per_gpu=2')
    config_ids = tuple(config_ids)
    if len(config_ids) != 2 or len(set(config_ids)) != 2:
        raise ValueError('Select exactly two distinct M22 config IDs')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(run_root or output_root / 'runs')
    os.environ['OGBENCH_DATASET_DIR'] = str(Path(dataset_root).resolve())
    # Match the documented formal launcher environment.  Without this, each
    # independent JAX process attempts a large allocator reservation and the
    # second process can emit avoidable RESOURCE_EXHAUSTED retries.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    jobs = sweep._jobs(
        study_path,
        run_root,
        include_configs=set(config_ids),
    )
    selected = []
    for config_id in config_ids:
        candidates = [job for job in jobs if job['configuration'].config_id == config_id]
        if not candidates:
            raise ValueError(f'No planned job for {config_id}')
        planned = [job for job in candidates if job['status'] == 'planned']
        if not planned:
            raise ValueError(f'{config_id} has no untouched planned seed in {run_root}')
        selected.append(sorted(planned, key=lambda job: job['seed'])[0])
    extra_args = [
        f'--train_steps={int(train_steps)}',
        '--batch_size=1024',
        '--log_interval=1000',
        f'--eval_interval={int(train_steps)}',
        '--eval_tasks=none',
        '--eval_episodes=1',
        '--eval_temperature=0',
        f'--save_interval={int(train_steps)}',
        '--no-save-best-checkpoint',
        '--no-save-last-checkpoint',
    ]
    log_root = output_root / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)

    def runner(job, assigned_gpu, assigned_run_root, command_args):
        child_env = os.environ.copy()
        child_env['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu)
        child_env['RLC_ASSIGNED_PHYSICAL_GPU'] = str(assigned_gpu)
        child_env['RLC_JOBS_PER_GPU'] = str(jobs_per_gpu)
        child_env['RLC_WORKER_SLOT'] = 'concurrency-smoke'
        log_path = log_root / f'{job["configuration"].config_id}.log'
        with log_path.open('w') as log_file:
            result = subprocess.run(
                sweep._command(job, assigned_run_root, command_args),
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode

    failure_count = sweep._dispatch_jobs(
        selected,
        [str(gpu)],
        str(run_root),
        extra_args,
        runner=runner,
        jobs_per_gpu=jobs_per_gpu,
    )
    records = [_metadata(job['run_dir']) for job in selected]
    forbidden_patterns = (
        re.compile(r'out[_ -]?of[_ -]?memory', re.IGNORECASE),
        re.compile(r'resource[_ -]?exhausted', re.IGNORECASE),
        re.compile(r'cuda_error_out_of_memory', re.IGNORECASE),
        re.compile(r'\b(?:nan|non[- ]finite)\b', re.IGNORECASE),
        re.compile(r'traceback \(most recent call last\)', re.IGNORECASE),
    )
    log_hits = {}
    for config_id in config_ids:
        log_path = log_root / f'{config_id}.log'
        text = log_path.read_text(errors='replace') if log_path.is_file() else ''
        hits = [line for line in text.splitlines() if any(pattern.search(line) for pattern in forbidden_patterns)]
        if hits:
            log_hits[config_id] = hits
    report = {
        'schema': 'm22_gpu_concurrency_smoke_v1',
        'study_id': 'M22',
        'configs': list(config_ids),
        'gpu_policy': {'physical_gpu': '1 only', 'jobs_per_gpu': 2},
        'train_steps': int(train_steps),
        'xla_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        'run_root': str(run_root.resolve()),
        'dataset_root': str(Path(dataset_root).resolve()),
        'scheduler_failures': int(failure_count),
        'records': records,
        'forbidden_log_hits': log_hits,
        'status': 'pass' if failure_count == 0 and not log_hits and all(record.get('status') == 'completed' for record in records) else 'fail',
        'performance_conclusion': False,
    }
    (output_root / 'concurrency_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--run-root', type=Path, default=None)
    parser.add_argument('--configs', default=','.join(DEFAULT_CONFIGS))
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    parser.add_argument('--train-steps', type=int, default=2_000)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    report = run_concurrency_smoke(
        study_path=args.study,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        run_root=args.run_root,
        config_ids=tuple(item.strip() for item in args.configs.split(',') if item.strip()),
        gpu=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        train_steps=args.train_steps,
    )
    print(
        f'M22 CONCURRENCY SMOKE: {report["status"].upper()} '
        f'gpu=1 jobs_per_gpu=2 configs={len(report["records"])}'
    )
    for record in report['records']:
        print(f'  {record.get("config_id")}: {record.get("status")} {record.get("run_dir")}')
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
