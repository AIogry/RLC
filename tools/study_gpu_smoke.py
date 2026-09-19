#!/usr/bin/env python3
"""Opt-in single/dual process production Study workload smoke on one GPU.

Without --execute this only prints a plan. Outputs must be new, external to
source and formal runs. No scheduler/Run lifecycle records are fabricated.
The bounded workload preserves full model/batch, samples real train/val data,
fresh-restores a checkpoint, and evaluates all tasks with fewer episodes.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def write_json(path, value):
    """Atomic phase/report writes inside this tool's newly reserved output."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_snapshot(gpu):
    def query(option, fields):
        output = subprocess.check_output(
            ['nvidia-smi', f'--{option}={fields}', '--format=csv,noheader,nounits'], text=True)
        return [[item.strip() for item in row] for row in csv.reader(output.splitlines()) if row]
    rows = query('query-gpu', 'index,uuid,memory.total,memory.used')
    selected = next((row for row in rows if row[0] == str(gpu)), None)
    if selected is None:
        raise ValueError(f'Physical GPU {gpu} not found')
    apps = query('query-compute-apps', 'gpu_uuid,pid,used_memory')
    return {'physical_gpu': int(gpu), 'uuid': selected[1], 'total_mib': float(selected[2]),
            'used_mib': float(selected[3]),
            'processes': [{'pid': int(pid), 'used_mib': float(memory)}
                          for uuid, pid, memory in apps if uuid == selected[1]]}


def host_snapshot(pids=()):
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    rss = {}
    for pid in pids:
        try:
            fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
            rss[str(pid)] = int(fields.get('VmRSS', '0 kB').split()[0]) / 1024
        except FileNotFoundError:
            pass
    return {'available_mib': int(meminfo['MemAvailable'].split()[0]) / 1024,
            'worker_rss_mib': rss, 'combined_worker_rss_mib': sum(rss.values())}


def child_environment(plan, worker_id):
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES=plan['gpu_snapshot']['uuid'],
                       RLC_ASSIGNED_PHYSICAL_GPU=str(plan['gpu']), RLC_WORKER_SLOT=str(worker_id),
                       RLC_JOBS_PER_GPU=str(plan['workers']), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    return environment


def worker_reports_valid(plan, reports, pids):
    if len(reports) != plan['workers'] or len(pids) != plan['workers']:
        return False
    for record, config, pid in zip(reports, plan['configs'], pids):
        expected = {'status': 'pass', 'source_sha': plan['source_sha'], 'pid': pid,
                    'config_id': config['config_id'], 'config_sha256': config['config_sha256'],
                    'physical_gpu': plan['gpu'], 'gpu_uuid': plan['gpu_snapshot']['uuid'],
                    'backend': 'gpu', 'batch_size': plan['batch_size_per_worker'],
                    'updates': plan['updates_per_worker'], 'checkpoint_fresh_restore': 'pass'}
        if any(record.get(k) != v for k, v in expected.items()):
            return False
    return True


def workload_overlap(reports):
    """Require overlapping workload intervals, not only idle GPU process presence."""
    result = {}
    for start, end in (('compile_warmup', 'training'), ('training', 'validation'), ('evaluation', 'done')):
        intervals = [r.get('phase_started_unix', {}) for r in reports]
        result[start] = bool(intervals and all(start in t and end in t for t in intervals)
                             and max(t[start] for t in intervals) < min(t[end] for t in intervals))
    return result


def make_plan(args):
    from tools.audit_goal_conditioning_study import resolved_configurations
    from impls.experiment.management import config_fingerprint, jsonable
    ids = args.configs.split(',')
    if len(ids) != args.workers or len(ids) != len(set(ids)):
        raise ValueError('Select exactly one distinct configuration per worker')
    if not 1 <= args.warmup < args.steps <= 1000:
        raise ValueError('Require 1 <= warmup < steps <= 1000 (engineering budget)')
    if not 1 <= args.eval_episodes <= 2:
        raise ValueError('Short smoke requires 1 or 2 episodes per task')
    if args.gpu < 0 or args.batch_size < 1:
        raise ValueError('GPU index and batch size must be valid')
    if not 0 < args.memory_limit_fraction <= 0.9:
        raise ValueError('Memory safety limit must be in (0, 0.9]')
    study, configurations = resolved_configurations(args.study, set(ids))
    declared_gpu = study.data['execution'].get('physical_gpu')
    if declared_gpu is not None and int(declared_gpu) != args.gpu:
        raise ValueError('GPU must match the active Study operational policy')
    by_id = {c.config_id: (c, cfg) for c, cfg in configurations}
    if len(study.data['seeds']) != 1:
        raise ValueError('Select a single-seed Study for this bounded probe')
    if len({c.data['environment'] for c, _ in configurations}) != 1:
        raise ValueError('Workers must use the same environment')
    output = args.output.resolve()
    formal = Path(study.data['execution']['run_root_parent']).resolve()
    source_roots = [Path(line[9:]).resolve() for line in read_git('worktree', 'list', '--porcelain').splitlines()
                    if line.startswith('worktree ')]
    if any(output.is_relative_to(p) for p in source_roots + [ROOT, formal]):
        raise ValueError('Engineering output must be outside source and formal run-root')
    if output.exists() or args.output.is_symlink():
        raise ValueError('Output exists; preserve it and select a new path')
    records = []
    for config_id in ids:
        c, cfg = by_id[config_id]
        if not c.data.get('executable', True):
            raise ValueError('Configuration is non-executable')
        if cfg['batch_size'] != args.batch_size or study.data['protocol']['batch_size'] != args.batch_size:
            raise ValueError('Full batch must match both config and Study; no downscaling')
        if cfg['dataset_class'] != 'PuzzleBoardGCDataset' or c.data['algorithm'] != 'gciql':
            raise ValueError('This goal-input workload probe currently supports PuzzleBoard GCIQL only')
        for suffix in ('', '-val'):
            if not (args.dataset_root / f'{c.data["environment"]}{suffix}.npz').is_file():
                raise FileNotFoundError('Both local train/val files are required; no downloads')
        records.append({'config_id': config_id, 'environment': c.data['environment'],
                        'config_sha256': config_fingerprint(cfg), 'resolved_agent': jsonable(cfg)})
    return {'engineering_only': True, 'execute_requested': args.execute, 'study_id': study.study_id,
            'study_path': str(study.path), 'study_sha256': config_fingerprint(study.data),
            'source_sha': read_git('rev-parse', 'HEAD'), 'source_branch': read_git('branch', '--show-current'),
            'source_dirty': bool(read_git('status', '--porcelain', '--untracked-files=all')),
            'seed': study.data['seeds'][0], 'gpu': args.gpu, 'workers': args.workers,
            'batch_size_per_worker': args.batch_size, 'updates_per_worker': args.steps,
            'warmup_updates': args.warmup, 'eval_episodes_per_task': args.eval_episodes,
            'eval_tasks': 'all', 'eval_temperature': study.data['protocol']['eval_temperature'],
            'eval_gaussian': study.data['protocol']['eval_gaussian'], 'video_episodes': 0,
            'dataset_root': str(args.dataset_root.resolve()), 'output': str(output), 'configs': records,
            'xla_preallocate': 'false', 'checkpoint': 'save_and_fresh_restore',
            'memory_limit_fraction': args.memory_limit_fraction,
            'coverage_boundary': '200 updates is not full training; short all-task eval is not 20 episodes/task; sampled NVML peaks are lower bounds'}


def barrier(root, worker_id, name):
    (root / f'{worker_id}.{name}.ready').touch(exist_ok=False)
    deadline = time.monotonic() + 1800
    while not (root / f'{name}.go').exists():
        if (root / 'abort').exists() or time.monotonic() > deadline:
            raise RuntimeError(f'Parent abort/timeout at {name} barrier')
        time.sleep(0.1)


def worker(manifest_path, worker_id):
    # Only the explicitly executing parent supplies these two internal args.
    import resource
    import jax
    import numpy as np
    import impls
    import ogbench
    from impls.agents import agents
    from impls.main import _evaluate_tasks, _parse_args, _validate_restored_checkpoint
    from impls.utils.env_utils import make_env_and_datasets
    from impls.utils.flax_utils import save_agent, restore_agent_from_checkpoint
    from impls.utils.puzzle_datasets import PuzzleBoardGCDataset
    from impls.utils.reproducibility import derive_seed
    from tools.audit_goal_conditioning_study import tree_equal
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    spec = manifest['configs'][worker_id]
    directory = root / f'worker_{worker_id}'
    directory.mkdir(exist_ok=False)
    phase_times = {}
    def phase(name):
        phase_times[name] = time.time()
        write_json(directory / 'phase.json', {'phase': name, 'time': time.time(), 'pid': os.getpid()})
    devices = jax.devices()
    if jax.default_backend() != 'gpu' or len(devices) != 1 or devices[0].platform != 'gpu':
        raise RuntimeError(f'GPU backend with one visible device required: {devices}')
    if os.environ.get('RLC_ASSIGNED_PHYSICAL_GPU') != str(manifest['gpu']):
        raise RuntimeError('Physical GPU assignment mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != manifest['gpu_snapshot']['uuid']:
        raise RuntimeError('Visible GPU UUID mismatch')
    if os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE') != 'false':
        raise RuntimeError('Preallocation must remain false')
    if read_git('rev-parse', 'HEAD') != manifest['source_sha'] or read_git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Source changed after preflight')
    for module in (impls, ogbench):
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Wrong worktree import: {module.__file__}')
    cfg = spec['resolved_agent']
    seed = manifest['seed']
    phase('data_loading')
    env, train, val = make_env_and_datasets(spec['environment'], seed=derive_seed(seed, 3),
                                         dataset_seed=derive_seed(seed, 1), dataset_dir=manifest['dataset_root'])
    try:
        dataset = PuzzleBoardGCDataset(train, cfg, rng=derive_seed(seed, 11))
        validation = PuzzleBoardGCDataset(val, cfg, rng=derive_seed(seed, 12))
        example = dataset.sample(1)
        phase('initialization')
        agent = agents[cfg['agent_name']].create(seed, example['observations'], example['actions'], cfg)
        barrier(root, worker_id, 'compile')
        phase('compile_warmup')
        durations, losses = [], []
        for step in range(1, manifest['updates_per_worker'] + 1):
            if step == manifest['warmup_updates'] + 1:
                barrier(root, worker_id, 'training')
                phase('training')
            start = time.perf_counter()
            batch = dataset.sample(manifest['batch_size_per_worker'])
            if len(batch['observations']) != manifest['batch_size_per_worker']:
                raise AssertionError('Partial batch')
            agent, info = agent.update(batch)
            info = jax.device_get(info)  # completion barrier, not async enqueue timing
            if any(not np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(info)):
                raise RuntimeError(f'Non-finite losses at update {step}')
            durations.append(time.perf_counter() - start)
            losses.append({k: float(v) for k, v in info.items()})
        phase('validation')
        validation_batch = validation.sample(manifest['batch_size_per_worker'], evaluation=True)
        _, info = agent.total_loss(validation_batch, agent.network.params, rng=agent.rng)
        if any(not np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(jax.device_get(info))):
            raise RuntimeError('Non-finite validation loss')
        phase('checkpoint')
        path = save_agent(agent, directory, manifest['updates_per_worker'])
        fresh = agents[cfg['agent_name']].create(seed + 991, example['observations'], example['actions'], cfg)
        restored = restore_agent_from_checkpoint(fresh, path)
        tree_equal((agent.network.params, agent.network.opt_state, agent.rng),
                   (restored.network.params, restored.network.opt_state, restored.rng))
        _validate_restored_checkpoint(agent, restored, example['observations'], example['actor_goals'], example['actions'])
        del fresh, restored
        barrier(root, worker_id, 'evaluation')
        phase('evaluation')
        eval_args = _parse_args(['--agent', cfg['agent_name'], '--eval_tasks', 'all',
                                 '--eval_episodes', str(manifest['eval_episodes_per_task']),
                                 '--eval_temperature', str(manifest['eval_temperature'])])
        eval_args.eval_gaussian = manifest['eval_gaussian']
        metrics = _evaluate_tasks(agent, env, cfg, eval_args, derive_seed(seed, 4))
        if not metrics or any(not np.isfinite(v) for v in metrics.values()):
            raise RuntimeError('Missing/non-finite evaluation metrics')
        phase('done')
        imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                   if (name == 'impls' or name.startswith('impls.') or name == 'ogbench' or name.startswith('ogbench.'))
                   and getattr(module, '__file__', None)}
        if any(not Path(p).is_relative_to(ROOT) for p in imports.values()):
            raise RuntimeError('Cross-worktree child import')
        write_json(directory / 'report.json', {
            'status': 'pass', 'pid': os.getpid(), 'source_sha': manifest['source_sha'],
            'config_id': spec['config_id'], 'config_sha256': spec['config_sha256'],
            'physical_gpu': manifest['gpu'], 'gpu_uuid': manifest['gpu_snapshot']['uuid'],
            'child_visible_devices': [str(d) for d in devices], 'backend': jax.default_backend(),
            'batch_size': cfg['batch_size'], 'updates': len(losses), 'losses': losses,
            'update_seconds': durations, 'steady_update_seconds_mean': float(np.mean(durations[manifest['warmup_updates']:])),
            'phase_started_unix': phase_times, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'checkpoint_fresh_restore': 'pass', 'eval_metrics': metrics, 'imports': imports,
        })
    finally:
        env.close()


def execute(plan, args):
    from tools.audit_goal_conditioning_study import dataset_metadata
    # Children run from ROOT; do not pass a caller-relative manifest path.
    args.output = Path(plan['output'])
    if plan['source_dirty'] or plan['source_branch']:
        raise ValueError('Execution requires a clean detached worktree')
    if not args.expected_source_sha or args.expected_source_sha != plan['source_sha']:
        raise ValueError('--expected-source-sha must match the actual frozen source')
    snapshot = gpu_snapshot(args.gpu)
    if snapshot['processes']:
        raise ValueError('GPU already has compute processes; coordinate with owner, do not kill/share')
    if snapshot['used_mib'] / snapshot['total_mib'] >= args.memory_limit_fraction:
        raise ValueError('GPU baseline already exceeds memory safety limit')
    plan['gpu_snapshot'] = snapshot
    plan['host_before'] = host_snapshot()
    parent = args.output.parent
    while not parent.exists():
        parent = parent.parent
    plan['disk_free_bytes'] = shutil.disk_usage(parent).free
    environment = plan['configs'][0]['environment']
    plan['datasets'] = [dataset_metadata(args.dataset_root / f'{environment}{suffix}.npz') for suffix in ('', '-val')]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / 'manifest.json'
    write_json(manifest, plan)
    processes, logs, samples, failure = [], [], [], None
    observed, overlap, phase_peaks = set(), False, {}
    try:
        for worker_id in range(args.workers):
            child_env = child_environment(plan, worker_id)
            log = (args.output / f'worker_{worker_id}.log').open('x')
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--execute',
                 '--worker-manifest', str(manifest), '--worker-id', str(worker_id)],
                cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT))
        pids = {p.pid for p in processes}
        deadline = time.monotonic() + 3600
        while any(p.poll() is None for p in processes):
            if time.monotonic() > deadline or any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Worker failure or one-hour engineering timeout')
            for phase_name in ('compile', 'training', 'evaluation'):
                if all((args.output / f'{i}.{phase_name}.ready').exists() for i in range(args.workers)):
                    (args.output / f'{phase_name}.go').touch(exist_ok=True)
            sample = gpu_snapshot(args.gpu)
            if sample['uuid'] != snapshot['uuid']:
                raise RuntimeError('GPU UUID changed')
            active = {p['pid'] for p in sample['processes']}
            if active - pids:
                raise RuntimeError('New unknown GPU compute process; gate fails, no sharing authorized')
            observed |= active
            overlap |= len(active & pids) == args.workers
            phases = []
            for i in range(args.workers):
                path = args.output / f'worker_{i}/phase.json'
                phases.append(json.loads(path.read_text())['phase'] if path.exists() else 'startup')
            for name in phases:
                phase_peaks[name] = max(phase_peaks.get(name, 0), sample['used_mib'])
            sample.update(time=time.time(), phases=phases, host=host_snapshot(pids))
            samples.append(sample)
            if sample['used_mib'] / sample['total_mib'] > args.memory_limit_fraction:
                raise RuntimeError('GPU memory safety limit exceeded; do not reduce scientific workload')
            time.sleep(0.25)
    except (Exception, KeyboardInterrupt) as error:
        failure = str(error) or type(error).__name__
        (args.output / 'abort').touch(exist_ok=True)
    finally:
        # Only children created by this invocation are stopped; never external PIDs.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    reports = []
    for i in range(args.workers):
        path = args.output / f'worker_{i}/report.json'
        reports.append(json.loads(path.read_text()) if path.exists() else {'status': 'missing'})
    hits = {}
    for i in range(len(logs)):
        content = (args.output / f'worker_{i}.log').read_text(errors='replace')
        hits[str(i)] = [line for line in content.splitlines()
                       if re.search(r'out.of.memory|resource.exhausted|\bnan\b|non.finite|traceback', line, re.I)]
    overlaps = workload_overlap(reports)
    reports_valid = worker_reports_valid(plan, reports, [p.pid for p in processes])
    passed = (not failure and len(processes) == args.workers and all(p.returncode == 0 for p in processes)
              and reports_valid and all(overlaps.values()) and observed == {p.pid for p in processes}
              and overlap and not any(hits.values())
              and all(name in phase_peaks for name in ('compile_warmup', 'training', 'checkpoint', 'evaluation'))
              and read_git('rev-parse', 'HEAD') == plan['source_sha']
              and not read_git('status', '--porcelain', '--untracked-files=all'))
    write_json(args.output / 'telemetry.json', samples)
    report = {'status': 'pass' if passed else 'fail', 'engineering_only': True,
              'source_sha': plan['source_sha'], 'gpu_uuid': snapshot['uuid'], 'failure': failure,
              'exit_codes': [p.returncode for p in processes], 'all_workers_observed_on_gpu': observed == {p.pid for p in processes},
              'worker_reports_valid': reports_valid, 'workload_phase_overlap': overlaps,
              'simultaneous_gpu_processes_observed': overlap, 'sampled_total_gpu_peak_mib_by_phase': phase_peaks,
              'memory_limit_fraction': args.memory_limit_fraction, 'workers': reports,
              'forbidden_log_hits': hits, 'coverage_boundary': plan['coverage_boundary']}
    write_json(args.output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(args.output / 'report.json')}))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-id', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--configs')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--workers', type=int, choices=(1, 2))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--eval-episodes', type=int, default=1)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-source-sha')
    parser.add_argument('--memory-limit-fraction', type=float, default=0.9)
    args = parser.parse_args(argv)
    if args.worker_manifest is not None:
        if not args.execute or args.worker_id is None:
            parser.error('Internal worker requires explicit execution')
        worker(args.worker_manifest, args.worker_id)
        return 0
    required = ('study', 'configs', 'gpu', 'workers', 'batch_size', 'steps', 'warmup', 'dataset_root', 'output')
    if any(getattr(args, key) is None for key in required):
        parser.error('Explicit ' + ', '.join('--' + key.replace('_', '-') for key in required) + ' required')
    os.environ['JAX_PLATFORMS'] = 'cpu'  # parent configuration inspection never uses GPU
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return execute(plan, args)


if __name__ == '__main__':
    raise SystemExit(main())
