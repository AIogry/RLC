"""Benchmark GPU concurrency for the M21 scheduling decision.

This module is deliberately a tooling-layer benchmark.  It launches three
independent instances of the existing M20B-double-S002 workload, records
host/GPU/process telemetry, and summarizes training throughput.  It does not
change an agent, computation module, dataset, or scientific Study result.

The benchmark is asymmetric by design:

* ``M21-SINGLE`` runs alone on physical GPU 0;
* ``M21-DUAL-A`` and ``M21-DUAL-B`` run concurrently on physical GPU 1.

The child runs use the existing Study/configuration only as a workload
definition.  Their artifacts live under a separate benchmark namespace and
are marked ``benchmark_only`` and ``scientific_result: false``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psutil
except ImportError:  # pragma: no cover - checked before real execution.
    psutil = None

from impls.experiment import (
    config_fingerprint,
    load_configuration,
    load_study,
    make_run_path,
)


STUDY_DEFAULT = REPO_ROOT / 'experiments/M20B_cube_entity_mixer_scaling/study.yaml'
CONFIG_DEFAULT = (
    REPO_ROOT
    / 'experiments/M20B_cube_entity_mixer_scaling/configs/M20B-double-S002.yaml'
)
DATASET_DEFAULT = Path(
    os.environ.get('OGBENCH_DATASET_DIR', '/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
)
OUTPUT_DEFAULT = Path(
    os.environ.get(
        'M21_BENCHMARK_ROOT',
        '/data/qijunrong/06-RL/offline-rl/exp/RLC/benchmarks/M21',
    )
)
FORMAL_M20B_ROOT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M20B')

BENCHMARK_ID = 'M21'
BENCHMARK_SCHEMA = 'm21_gpu_concurrency_benchmark_v1'
WORKLOAD_CONFIG_ID = 'M20B-double-S002'
WORKLOAD_ENVIRONMENT = 'cube-double-play-v0'
WORKLOAD_ALGORITHM = 'gciql'
TRAIN_STEPS = 50_000
BATCH_SIZE = 1_024
LOG_INTERVAL = 1_000
WARMUP_STEPS = 5_000
SAVE_INTERVAL = TRAIN_STEPS + 1
TELEMETRY_INTERVAL_SECONDS = 1.0
SYSTEM_TELEMETRY_INTERVAL_SECONDS = 2.0

GPU_TELEMETRY_FIELDS = (
    'timestamp',
    'sample_epoch',
    'index',
    'utilization.gpu',
    'utilization.memory',
    'memory.used',
    'memory.total',
    'power.draw',
    'temperature.gpu',
)
GPU_QUERY_FIELDS = (
    'timestamp',
    'index',
    'utilization.gpu',
    'utilization.memory',
    'memory.used',
    'memory.total',
    'power.draw',
    'temperature.gpu',
)
GPU_INFO_FIELDS = (
    'index',
    'name',
    'uuid',
    'driver_version',
    'memory.total',
    'memory.used',
    'memory.free',
    'utilization.gpu',
    'utilization.memory',
    'power.draw',
    'temperature.gpu',
)
COMPUTE_PROCESS_FIELDS = (
    'gpu_uuid',
    'pid',
    'process_name',
    'used_memory',
)
SYSTEM_TELEMETRY_FIELDS = (
    'timestamp',
    'sample_epoch',
    'cpu_percent',
    'load1',
    'load5',
    'load15',
    'memory_used',
    'memory_percent',
    'swap_used',
)


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: str
    artifact_name: str
    physical_gpu_id: str
    worker_slot: int
    run_attempt: int


BENCHMARK_JOBS = (
    BenchmarkJob('M21-SINGLE', 'single', '0', 0, 1),
    BenchmarkJob('M21-DUAL-A', 'dual_a', '1', 0, 2),
    BenchmarkJob('M21-DUAL-B', 'dual_b', '1', 1, 3),
)


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot satisfy its frozen protocol."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + '\n'
    )
    temporary.replace(path)


def _read_json(path):
    with Path(path).open() as file:
        return json.load(file)


def _git_provenance():
    def run_git(arguments):
        try:
            return subprocess.run(
                ['git', *arguments],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run_git(['status', '--porcelain', '--untracked-files=all'])
    return {
        'head': run_git(['rev-parse', 'HEAD']),
        'branch': run_git(['branch', '--show-current']),
        'dirty': bool(status) if status is not None else None,
        'dirty_files': status.splitlines() if status else [],
    }


def _package_versions():
    versions = {}
    for package in ('jax', 'jaxlib', 'flax', 'gymnasium', 'mujoco', 'psutil'):
        try:
            from importlib.metadata import version

            versions[package] = version(package)
        except Exception:  # pragma: no cover - package availability varies.
            versions[package] = None
    return versions


def _parse_nvidia_rows(output, fields):
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = next(csv.reader([line], skipinitialspace=True))
        if len(values) != len(fields):
            raise BenchmarkError(
                f'nvidia-smi returned {len(values)} fields, expected {len(fields)}: {line!r}'
            )
        rows.append(dict(zip(fields, (value.strip() for value in values))))
    return rows


def _nvidia_smi_command(arguments):
    try:
        result = subprocess.run(
            ['nvidia-smi', *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise BenchmarkError(f'Unable to execute nvidia-smi: {error}') from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BenchmarkError(
            f'nvidia-smi failed with exit code {result.returncode}: {detail}'
        )
    return result.stdout


def _nvidia_gpu_info():
    output = _nvidia_smi_command([
        f'--query-gpu={",".join(GPU_INFO_FIELDS)}',
        '--format=csv,noheader,nounits',
    ])
    rows = _parse_nvidia_rows(output, GPU_INFO_FIELDS)
    for row in rows:
        for key in ('index', 'memory.total', 'memory.used', 'memory.free'):
            try:
                row[key] = int(float(row[key]))
            except (TypeError, ValueError):
                row[key] = None
        for key in (
            'utilization.gpu', 'utilization.memory', 'power.draw', 'temperature.gpu',
        ):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = None
    return rows


def _nvidia_compute_processes():
    output = _nvidia_smi_command([
        f'--query-compute-apps={",".join(COMPUTE_PROCESS_FIELDS)}',
        '--format=csv,noheader,nounits',
    ])
    rows = _parse_nvidia_rows(output, COMPUTE_PROCESS_FIELDS)
    for row in rows:
        try:
            row['pid'] = int(row['pid'])
        except (TypeError, ValueError):
            row['pid'] = None
        try:
            row['used_memory'] = float(row['used_memory'])
        except (TypeError, ValueError):
            row['used_memory'] = None
    return rows


def _cuda_version():
    output = _nvidia_smi_command([])
    match = re.search(r'CUDA Version:\s*([^\s|]+)', output)
    return None if match is None else match.group(1)


def _host_provenance(*, require_gpu):
    provenance = {
        'hostname': platform.node(),
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'packages': _package_versions(),
        'gpu_info': [],
        'cuda_version': None,
        'baseline_compute_processes': [],
        'nvidia_smi_error': None,
    }
    try:
        provenance['gpu_info'] = _nvidia_gpu_info()
        provenance['cuda_version'] = _cuda_version()
        provenance['baseline_compute_processes'] = _nvidia_compute_processes()
    except BenchmarkError as error:
        if require_gpu:
            raise
        provenance['nvidia_smi_error'] = str(error)
    return provenance


def _validate_gpu_selection(host, gpu_ids):
    available = {str(row.get('index')) for row in host['gpu_info']}
    missing = [gpu for gpu in gpu_ids if gpu not in available]
    if missing:
        raise BenchmarkError(
            f'Requested physical GPU(s) are not visible: {missing}; visible={sorted(available)}'
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise BenchmarkError(f'Benchmark GPU IDs must be distinct: {gpu_ids!r}')


def _assert_safe_output_root(output_root, formal_root=FORMAL_M20B_ROOT):
    output_root = Path(output_root).expanduser().resolve()
    formal_root = Path(formal_root).expanduser().resolve()
    if output_root == formal_root or formal_root in output_root.parents:
        raise BenchmarkError(
            f'M21 benchmark output may not be inside the formal M20B run root: {output_root}'
        )
    return output_root


def _validate_workload(study_path, config_path, dataset_root):
    study = load_study(study_path)
    configuration = load_configuration(study, config_path)
    data = configuration.data
    if configuration.config_id != WORKLOAD_CONFIG_ID:
        raise BenchmarkError(
            f'M21 requires {WORKLOAD_CONFIG_ID}, got {configuration.config_id!r}'
        )
    if data.get('environment') != WORKLOAD_ENVIRONMENT:
        raise BenchmarkError(
            f'M21 requires {WORKLOAD_ENVIRONMENT}, got {data.get("environment")!r}'
        )
    if data.get('algorithm') != WORKLOAD_ALGORITHM:
        raise BenchmarkError(
            f'M21 requires algorithm={WORKLOAD_ALGORITHM}, got {data.get("algorithm")!r}'
        )
    overrides = data.get('agent_overrides', {})
    expected_agent = {
        'alpha': 1.0,
        'actor_hidden_dims': [512, 512, 512],
        'value_hidden_dims': [512, 512, 512],
        'layer_norm': True,
        'lr': 0.0003,
        'batch_size': BATCH_SIZE,
        'discount': 0.99,
        'expectile': 0.9,
        'tau': 0.005,
        'actor_loss': 'ddpgbc',
        'const_std': True,
        'discrete': False,
        'dataset_class': 'GCDataset',
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 1.0,
        'actor_p_randomgoal': 0.0,
        'actor_geom_sample': False,
        'p_aug': 0.0,
        'frame_stack': None,
    }
    for key, expected in expected_agent.items():
        if overrides.get(key) != expected:
            raise BenchmarkError(
                f'Workload field {key!r} changed: expected {expected!r}, '
                f'got {overrides.get(key)!r}'
            )
    compute = overrides.get('compute', {})
    slots = {name: compute.get(name) for name in ('actor', 'value', 'critic')}
    if any(slot is None for slot in slots.values()) or len({
        config_fingerprint(slot) for slot in slots.values()
    }) != 1:
        raise BenchmarkError('M21 requires identical actor/value/critic compute slots')
    slot = slots['actor']
    expected_slot_fields = {
        'enabled': True,
        'primitive': 'mlp',
        'structure': 'cube_tokens',
        'block': 'mlp_mixer',
        'topology': 'feedforward',
        'parameter_sharing': 'shared',
        'credit': 'direct',
        'relation_mode': 'legacy_none',
        'relation_augmenter': 'none',
        'readout': 'mean_context',
    }
    for key, expected in expected_slot_fields.items():
        if slot.get(key) != expected:
            raise BenchmarkError(
                f'Workload compute field {key!r} changed: expected {expected!r}, '
                f'got {slot.get(key)!r}'
            )
    expected_structure = {
        'num_cubes': 2,
        'robot_dim': 19,
        'cube_feature_dim': 9,
        'token_dim': 128,
        'robot_hidden_dim': 128,
        'slot_identity_embedding': False,
    }
    expected_block = {
        'num_blocks': 2,
        'token_hidden_dim': 64,
        'channel_hidden_dim': 256,
        'tm_mode': 'none',
    }
    if slot.get('structure_kwargs') != expected_structure:
        raise BenchmarkError('M21 Cube structure_kwargs do not match M20B-double-S002')
    if slot.get('block_kwargs') != expected_block:
        raise BenchmarkError('M21 Mixer block_kwargs do not match M20B-double-S002')
    if slot.get('relation_kwargs') or slot.get('relation_augmenter_kwargs'):
        raise BenchmarkError('M21 relation configuration must remain empty')

    dataset_root = Path(dataset_root)
    dataset_paths = {
        'train': dataset_root / f'{WORKLOAD_ENVIRONMENT}.npz',
        'validation': dataset_root / f'{WORKLOAD_ENVIRONMENT}-val.npz',
    }
    missing = [str(path) for path in dataset_paths.values() if not path.is_file()]
    if missing:
        raise BenchmarkError(f'M21 workload dataset files are missing: {missing}')
    return study, configuration, dataset_paths


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path):
    path = Path(path)
    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'bytes': stat.st_size,
        'sha256': _sha256_file(path),
    }


def _job_run_root(output_root, job):
    return Path(output_root) / '_runs' / job.artifact_name


def _expected_run_dir(output_root, job, study, configuration):
    return make_run_path(
        _job_run_root(output_root, job),
        study.study_id,
        configuration.config_id,
        configuration.slug,
        WORKLOAD_ENVIRONMENT,
        0,
        run_attempt=job.run_attempt,
    )


def _child_environment(job):
    inherited_platform = os.environ.get('JAX_PLATFORMS', '')
    platforms = {
        value.strip().lower()
        for value in inherited_platform.split(',')
        if value.strip()
    }
    if 'cpu' in platforms:
        raise BenchmarkError('M21 must not run with JAX_PLATFORMS=cpu')
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = job.physical_gpu_id
    environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    environment['RLC_ASSIGNED_PHYSICAL_GPU'] = job.physical_gpu_id
    environment['RLC_WORKER_SLOT'] = str(job.worker_slot)
    environment['RLC_JOBS_PER_GPU'] = '1' if job.job_id == 'M21-SINGLE' else '2'
    environment['RLC_BENCHMARK_ID'] = BENCHMARK_ID
    environment['RLC_BENCHMARK_ONLY'] = 'true'
    environment['RLC_SCIENTIFIC_RESULT'] = 'false'
    python_path = environment.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = str(REPO_ROOT) + (os.pathsep + python_path if python_path else '')
    return environment


def _probe_jax_gpu(job):
    """Verify that a child with this physical assignment selects GPU JAX."""

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import jax; print(jax.default_backend()); print(jax.devices())',
        ],
        cwd=REPO_ROOT,
        env=_child_environment(job),
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if result.returncode != 0 or not output.splitlines() or output.splitlines()[0] != 'gpu':
        detail = (result.stderr.strip() or output)
        raise BenchmarkError(
            f'JAX GPU probe failed for physical GPU {job.physical_gpu_id}: {detail}'
        )
    return {
        'physical_gpu_id': int(job.physical_gpu_id),
        'backend': output.splitlines()[0],
        'devices': output.splitlines()[1] if len(output.splitlines()) > 1 else None,
    }


def _benchmark_command(study_path, config_path, output_root, job):
    return [
        sys.executable,
        '-m',
        'impls.main',
        '--study',
        str(Path(study_path).resolve()),
        '--config',
        str(Path(config_path).resolve()),
        '--agent',
        WORKLOAD_ALGORITHM,
        '--env_name',
        WORKLOAD_ENVIRONMENT,
        '--seed',
        '0',
        '--run_attempt',
        str(job.run_attempt),
        '--run_root',
        str(_job_run_root(output_root, job)),
        '--train_steps',
        str(TRAIN_STEPS),
        '--batch_size',
        str(BATCH_SIZE),
        '--log_interval',
        str(LOG_INTERVAL),
        '--eval_interval',
        str(SAVE_INTERVAL),
        '--eval_tasks',
        'none',
        '--eval_episodes',
        '1',
        '--eval_temperature',
        '0.0',
        '--video_episodes',
        '0',
        '--save_interval',
        str(SAVE_INTERVAL),
        '--no-save-best-checkpoint',
        '--no-save-last-checkpoint',
    ]


def _benchmark_job_plan(single_gpu='0', dual_gpu='1'):
    """Return the frozen asymmetric M21 plan with explicit physical IDs."""

    return tuple(
        BenchmarkJob(
            job.job_id,
            job.artifact_name,
            single_gpu if job.job_id == 'M21-SINGLE' else dual_gpu,
            job.worker_slot,
            job.run_attempt,
        )
        for job in BENCHMARK_JOBS
    )


def _write_csv_header(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open('w', newline='')
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    file.flush()
    return file, writer


def _gpu_telemetry_worker(path, stop_event, interval_seconds):
    file, writer = _write_csv_header(path, GPU_TELEMETRY_FIELDS)
    try:
        while not stop_event.is_set():
            sample_epoch = time.time()
            timestamp = datetime.fromtimestamp(
                sample_epoch, timezone.utc,
            ).isoformat(timespec='milliseconds')
            try:
                rows = _parse_nvidia_rows(
                    _nvidia_smi_command([
                        f'--query-gpu={",".join(GPU_QUERY_FIELDS)}',
                        '--format=csv,noheader,nounits',
                    ]),
                    GPU_QUERY_FIELDS,
                )
                for row in rows:
                    writer.writerow({
                        'timestamp': timestamp,
                        'sample_epoch': sample_epoch,
                        'index': row.get('index'),
                        'utilization.gpu': row.get('utilization.gpu'),
                        'utilization.memory': row.get('utilization.memory'),
                        'memory.used': row.get('memory.used'),
                        'memory.total': row.get('memory.total'),
                        'power.draw': row.get('power.draw'),
                        'temperature.gpu': row.get('temperature.gpu'),
                    })
                file.flush()
            except BenchmarkError as error:
                file.flush()
                with Path(path).with_suffix('.errors.log').open('a') as errors:
                    errors.write(f'{timestamp}: {error}\n')
            stop_event.wait(interval_seconds)
    finally:
        file.close()


def _system_telemetry_worker(path, stop_event, interval_seconds):
    if psutil is None:
        return
    file, writer = _write_csv_header(path, SYSTEM_TELEMETRY_FIELDS)
    psutil.cpu_percent(interval=None)
    try:
        while not stop_event.is_set():
            sample_epoch = time.time()
            load1, load5, load15 = os.getloadavg()
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            writer.writerow({
                'timestamp': datetime.fromtimestamp(
                    sample_epoch, timezone.utc,
                ).isoformat(timespec='milliseconds'),
                'sample_epoch': sample_epoch,
                'cpu_percent': psutil.cpu_percent(interval=None),
                'load1': load1,
                'load5': load5,
                'load15': load15,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'swap_used': swap.used,
            })
            file.flush()
            stop_event.wait(interval_seconds)
    finally:
        file.close()


def _stop_telemetry_process(process, stop_event, timeout=10.0):
    stop_event.set()
    process.join(timeout=timeout)
    normal = not process.is_alive() and process.exitcode == 0
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
    return {
        'pid': process.pid,
        'normal_termination': normal,
        'exit_code': process.exitcode,
        'alive_after_join': process.is_alive(),
    }


def _launch_training_jobs(study_path, config_path, output_root, jobs):
    output_root = Path(output_root)
    log_root = output_root / 'process_logs'
    log_root.mkdir(parents=True, exist_ok=True)
    barrier = threading.Barrier(len(jobs))
    records = {}
    records_lock = threading.Lock()

    def launch(job):
        try:
            command = _benchmark_command(study_path, config_path, output_root, job)
            environment = _child_environment(job)
            log_path = log_root / f'{job.artifact_name}.log'
            barrier.wait()
            start_epoch = time.time()
            log_file = log_path.open('w')
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass
            if 'log_file' in locals() and not log_file.closed:
                log_file.close()
            raise
        with records_lock:
            records[job.job_id] = {
                'job': job,
                'process': process,
                'log_file': log_file,
                'pid': process.pid,
                'start_epoch': start_epoch,
                'start_time': datetime.fromtimestamp(
                    start_epoch, timezone.utc,
                ).isoformat(timespec='milliseconds'),
                'max_cpu_percent': None,
                'max_rss_bytes': None,
            }

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        try:
            for future in futures:
                future.result()
        except BaseException:
            _terminate_training_jobs(records)
            raise

    return records


def _terminate_training_jobs(records):
    for record in records.values():
        process = record['process']
        if process.poll() is None:
            try:
                os.killpg(process.pid, 15)
            except (OSError, ProcessLookupError):
                process.terminate()
    for record in records.values():
        process = record['process']
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, 9)
            except (OSError, ProcessLookupError):
                process.kill()
            process.wait(timeout=10)
        log_file = record.get('log_file')
        if log_file is not None and not log_file.closed:
            log_file.close()


def _monitor_training_jobs(records):
    if psutil is not None:
        for record in records.values():
            try:
                process_info = psutil.Process(record['pid'])
                process_info.cpu_percent(interval=None)
                record['_process_info'] = process_info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                record['_process_info'] = None
    pending = set(records)
    while pending:
        for job_id in tuple(pending):
            record = records[job_id]
            process = record['process']
            process_info = record.get('_process_info')
            if process_info is not None and process.poll() is None:
                try:
                    cpu_percent = process_info.cpu_percent(interval=None)
                    memory_info = process_info.memory_info()
                    record['max_cpu_percent'] = max(
                        record['max_cpu_percent'] or 0.0,
                        float(cpu_percent),
                    )
                    record['max_rss_bytes'] = max(
                        record['max_rss_bytes'] or 0,
                        int(memory_info.rss),
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if process.poll() is not None:
                end_epoch = time.time()
                record['end_epoch'] = end_epoch
                record['end_time'] = datetime.fromtimestamp(
                    end_epoch, timezone.utc,
                ).isoformat(timespec='milliseconds')
                record['wall_seconds'] = end_epoch - record['start_epoch']
                record['exit_code'] = process.returncode
                pending.remove(job_id)
        if pending:
            time.sleep(0.25)

    for record in records.values():
        process = record['process']
        process.wait()
        if 'end_epoch' not in record:
            end_epoch = time.time()
            record['end_epoch'] = end_epoch
            record['end_time'] = datetime.fromtimestamp(
                end_epoch, timezone.utc,
            ).isoformat(timespec='milliseconds')
            record['wall_seconds'] = end_epoch - record['start_epoch']
        record['exit_code'] = process.returncode
        record['log_file'].close()
        record.pop('process', None)
        record.pop('_process_info', None)
    return records


def _read_train_records(path):
    path = Path(path)
    if not path.is_file():
        return {'status': 'missing', 'rows': [], 'finite': False}
    rows = []
    finite = True
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        fields = tuple(reader.fieldnames or ())
        required = {'step', 'time/interval_seconds', 'time/total_seconds'}
        if not required.issubset(fields):
            return {
                'status': 'invalid_fields',
                'rows': [],
                'finite': False,
                'fields': list(fields),
            }
        for raw in reader:
            parsed = {}
            for key, value in raw.items():
                if value in (None, ''):
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    finite = False
                    continue
                if not math.isfinite(parsed[key]):
                    finite = False
            if not {
                'step', 'time/interval_seconds', 'time/total_seconds',
            }.issubset(parsed):
                finite = False
                continue
            step = parsed.get('step')
            if (
                step is None
                or not math.isfinite(step)
                or not step.is_integer()
                or step <= 0
            ):
                finite = False
                continue
            rows.append(parsed)
    rows.sort(key=lambda row: row['step'])
    return {'status': 'available', 'rows': rows, 'finite': finite}


def _percentile(values, quantile):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _numeric_stats(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {
            'count': 0,
            'mean': None,
            'p10': None,
            'p50': None,
            'p90': None,
            'min': None,
            'max': None,
        }
    return {
        'count': len(values),
        'mean': sum(values) / len(values),
        'p10': _percentile(values, 0.10),
        'p50': _percentile(values, 0.50),
        'p90': _percentile(values, 0.90),
        'min': min(values),
        'max': max(values),
    }


def _throughput_summary(train_path):
    parsed = _read_train_records(train_path)
    rows = parsed['rows']
    speeds = []
    previous = None
    for row in rows:
        if previous is not None and row['step'] > WARMUP_STEPS and row['step'] <= TRAIN_STEPS:
            step_delta = row['step'] - previous['step']
            interval = row['time/interval_seconds']
            if step_delta > 0 and interval > 0 and math.isfinite(interval):
                # impls.main stores elapsed wall time per training step here:
                # (now - last_time) / log_interval.  Do not multiply by the
                # logged step delta a second time.
                speeds.append(1.0 / interval)
        previous = row
    first = rows[0] if rows else None
    warmup = next((row for row in rows if row['step'] == WARMUP_STEPS), None)
    final = next((row for row in rows if row['step'] == TRAIN_STEPS), None)
    steady_elapsed = None
    aggregate_speed = None
    if warmup is not None and final is not None:
        elapsed = final.get('time/total_seconds') - warmup.get('time/total_seconds')
        if elapsed > 0:
            steady_elapsed = elapsed
            aggregate_speed = (TRAIN_STEPS - WARMUP_STEPS) / elapsed
    timing_valid = bool(rows)
    previous_total = None
    for row in rows:
        interval = row['time/interval_seconds']
        total = row['time/total_seconds']
        if interval <= 0 or total < 0:
            timing_valid = False
        if previous_total is not None and total <= previous_total:
            timing_valid = False
        previous_total = total
    stats = _numeric_stats(speeds)
    observed_steps = [int(row['step']) for row in rows]
    expected_steps = list(range(LOG_INTERVAL, TRAIN_STEPS + 1, LOG_INTERVAL))
    return {
        'train_csv': str(Path(train_path).resolve()),
        'status': parsed['status'],
        'finite': parsed['finite'],
        'observed_steps': [int(row['step']) for row in rows],
        'row_count': len(rows),
        'warmup_steps': WARMUP_STEPS,
        'steady_state_window': [WARMUP_STEPS, TRAIN_STEPS],
        'startup_compile_proxy_seconds': (
            None if first is None else first.get('time/total_seconds')
        ),
        'warmup_elapsed_seconds': (
            None if warmup is None else warmup.get('time/total_seconds')
        ),
        'steady_elapsed_seconds': steady_elapsed,
        'aggregate_window_steps_per_sec': aggregate_speed,
        'interval_steps_per_sec': stats,
        'expected_steps': expected_steps,
        'timing_valid': timing_valid,
        'complete': (
            parsed['status'] == 'available'
            and parsed['finite']
            and observed_steps == expected_steps
            and timing_valid
        ),
    }


def _read_telemetry(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline='') as file:
        rows = []
        for raw in csv.DictReader(file):
            parsed = dict(raw)
            for key, value in raw.items():
                if key in {'timestamp'}:
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = None
            rows.append(parsed)
        return rows


def _window_rows(rows, *, start_epoch, end_epoch, index=None):
    selected = []
    for row in rows:
        sample_epoch = row.get('sample_epoch')
        if sample_epoch is None or not (start_epoch <= sample_epoch <= end_epoch):
            continue
        if index is not None:
            try:
                observed_index = int(row.get('index', -1))
            except (TypeError, ValueError):
                continue
            if observed_index != int(index):
                continue
        selected.append(row)
    return selected


def _gpu_window_summary(rows):
    def values(key):
        return [row[key] for row in rows if row.get(key) is not None]

    util = _numeric_stats(values('utilization.gpu'))
    memory = values('memory.used')
    total = values('memory.total')
    power = _numeric_stats(values('power.draw'))
    return {
        'sample_count': len(rows),
        'utilization_gpu': util,
        'utilization_memory': _numeric_stats(values('utilization.memory')),
        'memory_peak': max(memory) if memory else None,
        'memory_total': max(total) if total else None,
        'power': power,
        'temperature': _numeric_stats(values('temperature.gpu')),
    }


def _system_window_summary(rows):
    def values(key):
        return [row[key] for row in rows if row.get(key) is not None]

    memory_percent = _numeric_stats(values('memory_percent'))
    return {
        'sample_count': len(rows),
        'cpu_percent': _numeric_stats(values('cpu_percent')),
        'load1': _numeric_stats(values('load1')),
        'load5': _numeric_stats(values('load5')),
        'load15': _numeric_stats(values('load15')),
        'memory_used': _numeric_stats(values('memory_used')),
        'memory_percent': memory_percent,
        'swap_used': _numeric_stats(values('swap_used')),
    }


def _attach_benchmark_metadata(run_dir, job, record=None):
    metadata_path = Path(run_dir) / 'runtime_metadata.json'
    if not metadata_path.is_file():
        return False
    metadata = _read_json(metadata_path)
    metadata.update({
        'benchmark_id': BENCHMARK_ID,
        'benchmark_only': True,
        'scientific_result': False,
        'benchmark_job_id': job.job_id,
        'physical_gpu_id': int(job.physical_gpu_id),
        'worker_slot': job.worker_slot,
        'jobs_per_gpu': 1 if job.job_id == 'M21-SINGLE' else 2,
    })
    if record is not None:
        metadata['benchmark_process'] = {
            'pid': record.get('pid'),
            'physical_gpu_id': int(job.physical_gpu_id),
            'worker_slot': job.worker_slot,
            'start_time': record.get('start_time'),
            'end_time': record.get('end_time'),
            'wall_seconds': record.get('wall_seconds'),
            'exit_code': record.get('exit_code'),
            'cpu_percent_max': record.get('max_cpu_percent'),
            'rss_bytes_max': record.get('max_rss_bytes'),
        }
    _write_json(metadata_path, metadata)
    return True


def _make_job_links(output_root, run_dirs, jobs):
    warnings = []
    for job in jobs:
        run_dir = Path(run_dirs[job.job_id])
        link = Path(output_root) / job.artifact_name
        if link.is_symlink():
            try:
                if link.resolve() == run_dir.resolve():
                    continue
            except OSError:
                pass
            warnings.append(f'benchmark artifact link already exists: {link}')
            continue
        if link.exists():
            warnings.append(f'benchmark artifact path already exists: {link}')
            continue
        if not run_dir.is_dir():
            warnings.append(f'run directory missing; cannot link: {run_dir}')
            continue
        try:
            link.symlink_to(os.path.relpath(run_dir, link.parent), target_is_directory=True)
        except OSError as error:
            warnings.append(f'failed to create {link}: {error}')
    return warnings


def _job_summary(job, record, run_dir):
    train_path = Path(run_dir) / 'train.csv'
    throughput = _throughput_summary(train_path)
    return {
        'job_id': job.job_id,
        'artifact_name': job.artifact_name,
        'physical_gpu_id': int(job.physical_gpu_id),
        'worker_slot': job.worker_slot,
        'run_attempt': job.run_attempt,
        'run_dir': str(Path(run_dir).resolve()),
        'train_csv': str(train_path.resolve()),
        'pid': record.get('pid'),
        'start_time': record.get('start_time'),
        'end_time': record.get('end_time'),
        'wall_seconds': record.get('wall_seconds'),
        'exit_code': record.get('exit_code'),
        'cpu_percent_max': record.get('max_cpu_percent'),
        'rss_bytes_max': record.get('max_rss_bytes'),
        'throughput': throughput,
    }


def _aggregate_results(
    output_root,
    host,
    job_records,
    run_dirs,
    benchmark_start,
    benchmark_end,
    job_plan,
    telemetry_processes=None,
):
    gpu_rows = _read_telemetry(Path(output_root) / 'gpu_telemetry.csv')
    system_rows = _read_telemetry(Path(output_root) / 'system_telemetry.csv')
    jobs = {}
    for job in job_plan:
        jobs[job.job_id] = _job_summary(job, job_records[job.job_id], run_dirs[job.job_id])

    single = jobs['M21-SINGLE']
    dual_a = jobs['M21-DUAL-A']
    dual_b = jobs['M21-DUAL-B']
    single_speed = single['throughput']['interval_steps_per_sec']['mean']
    dual_a_speed = dual_a['throughput']['interval_steps_per_sec']['mean']
    dual_b_speed = dual_b['throughput']['interval_steps_per_sec']['mean']
    dual_total = None
    gain = None
    if dual_a_speed is not None and dual_b_speed is not None:
        dual_total = dual_a_speed + dual_b_speed
    if single_speed and dual_total is not None:
        gain = dual_total / single_speed

    starts = {job_id: record['start_epoch'] for job_id, record in job_records.items()}
    ends = {job_id: record['end_epoch'] for job_id, record in job_records.items()}
    single_gpu_id = next(
        job.physical_gpu_id for job in job_plan if job.job_id == 'M21-SINGLE'
    )
    dual_gpu_id = next(
        job.physical_gpu_id for job in job_plan if job.job_id == 'M21-DUAL-A'
    )
    single_gpu_info = _window_rows(
        gpu_rows,
        start_epoch=starts['M21-SINGLE'],
        end_epoch=ends['M21-SINGLE'],
        index=int(single_gpu_id),
    )
    dual_start = max(starts['M21-DUAL-A'], starts['M21-DUAL-B'])
    dual_end = min(ends['M21-DUAL-A'], ends['M21-DUAL-B'])
    dual_gpu_info = _window_rows(
        gpu_rows,
        start_epoch=dual_start,
        end_epoch=dual_end,
        index=int(dual_gpu_id),
    )
    system_info = _window_rows(
        system_rows,
        start_epoch=benchmark_start,
        end_epoch=benchmark_end,
    )
    gpu_summary = {
        'single_gpu': _gpu_window_summary(single_gpu_info),
        'dual_gpu': _gpu_window_summary(dual_gpu_info),
        'telemetry_rows': len(gpu_rows),
    }
    system_summary = _system_window_summary(system_info)
    process_ok = all(job['exit_code'] == 0 for job in jobs.values())
    train_complete = all(job['throughput']['complete'] for job in jobs.values())
    finite = all(job['throughput']['finite'] for job in jobs.values())
    telemetry_available = (
        gpu_summary['single_gpu']['sample_count'] > 0
        and gpu_summary['dual_gpu']['sample_count'] > 0
        and system_summary['sample_count'] > 0
    )
    telemetry_processes_normal = bool(
        telemetry_processes
        and all(
            process.get('normal_termination')
            and not process.get('alive_after_join')
            for process in telemetry_processes.values()
        )
    )
    memory_peaks = [
        info['memory_peak']
        for info in (gpu_summary['single_gpu'], gpu_summary['dual_gpu'])
        if info['memory_peak'] is not None
    ]
    memory_totals = [
        info['memory_total']
        for info in (gpu_summary['single_gpu'], gpu_summary['dual_gpu'])
        if info['memory_total'] is not None
    ]
    memory_safe = bool(
        memory_peaks and memory_totals
        and max(memory_peaks) / max(memory_totals) <= 0.90
    )
    cpu_p90 = system_summary['cpu_percent']['p90']
    memory_p90 = system_summary['memory_percent']['p90']
    system_safe = bool(
        cpu_p90 is not None and cpu_p90 < 90.0
        and memory_p90 is not None and memory_p90 < 90.0
    )
    no_runtime_failure = (
        process_ok
        and train_complete
        and finite
        and telemetry_available
        and telemetry_processes_normal
    )
    if no_runtime_failure and memory_safe and system_safe and gain is not None:
        if gain >= 1.30:
            recommendation = 'jobs_per_gpu=2 recommended for throughput-oriented scheduling'
        elif gain >= 1.10:
            recommendation = 'keep jobs_per_gpu=1 as default; jobs_per_gpu=2 optional'
        else:
            recommendation = 'jobs_per_gpu=2 not recommended'
    elif gain is not None:
        recommendation = 'benchmark invalid or resource safety condition failed; do not change default'
    else:
        recommendation = 'insufficient valid throughput data; do not change default'
    return {
        'benchmark_only': True,
        'scientific_result': False,
        'benchmark_id': BENCHMARK_ID,
        'status': 'completed' if no_runtime_failure else 'failed',
        'benchmark_wall_seconds': benchmark_end - benchmark_start,
        'jobs': jobs,
        'throughput': {
            'v_single': single_speed,
            'v_dual_a': dual_a_speed,
            'v_dual_b': dual_b_speed,
            'v_dual_total': dual_total,
            'throughput_gain': gain,
            'per_job_slowdown_a': (
                None if single_speed is None or dual_a_speed is None
                else dual_a_speed / single_speed
            ),
            'per_job_slowdown_b': (
                None if single_speed is None or dual_b_speed is None
                else dual_b_speed / single_speed
            ),
        },
        'gpu_telemetry': gpu_summary,
        'system_telemetry': system_summary,
        'resource_safety': {
            'memory_safe': memory_safe,
            'system_safe': system_safe,
            'memory_peak_fraction_limit': 0.90,
            'cpu_percent_p90_limit': 90.0,
            'memory_percent_p90_limit': 90.0,
        },
        'runtime_checks': {
            'all_exit_codes_zero': process_ok,
            'all_train_curves_complete': train_complete,
            'all_train_values_finite': finite,
            'telemetry_available': telemetry_available,
            'telemetry_processes_normal': telemetry_processes_normal,
            'oom_or_crash_observed': not process_ok,
            'nan_or_inf_observed': not finite,
        },
        'recommendation': recommendation,
        'host_gpu_info': host['gpu_info'],
        'cuda_version': host['cuda_version'],
    }


def _format_number(value, digits=3):
    if value is None:
        return 'N/A'
    return f'{float(value):.{digits}f}'


def _render_report(manifest, summary):
    throughput = summary['throughput']
    gpu = summary['gpu_telemetry']
    system = summary['system_telemetry']
    host = manifest['source']['host']
    gpu_description = '; '.join(
        f'GPU{info.get("index")}: {info.get("name")} '
        f'(driver {info.get("driver_version")}, '
        f'baseline memory {info.get("memory.used")} / {info.get("memory.total")} MiB)'
        for info in host.get('gpu_info', [])
    ) or 'N/A'
    lines = [
        '# M21 GPU Concurrency Throughput Benchmark',
        '',
        'This is an engineering throughput benchmark only.',
        '',
        '- `benchmark_only: true`',
        '- `scientific_result: false`',
        '- workload: `M20B-double-S002` / `cube-double-play-v0`',
        '- architecture: `Cube Entity Tokens → MLP-Mixer L2 → MeanContextReadout`; relation: `none`',
        f'- source HEAD: `{manifest["source"]["git"]["head"]}`',
        f'- source worktree dirty at start: `{manifest["source"]["git"].get("dirty")}`',
        f'- GPU: {gpu_description}',
        f'- CUDA version: `{host.get("cuda_version")}`',
        f'- train steps: `{TRAIN_STEPS}`; batch size: `{BATCH_SIZE}`',
        f'- warmup: steps `{WARMUP_STEPS}`; steady-state window: `{WARMUP_STEPS} → {TRAIN_STEPS}`',
        '- evaluation: `eval_tasks=none`; best/last semantic checkpoints disabled; periodic checkpoint I/O disabled (framework final numeric checkpoint only)',
        '- dual mapping: both jobs use `CUDA_VISIBLE_DEVICES=1` and may report logical `cuda:0`; physical GPU IDs are recorded separately',
        '',
        '## Throughput',
        '',
        '| Job | Physical GPU | Mean steps/s | Median steps/s | p10 | p90 | Exit |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for job_id in ('M21-SINGLE', 'M21-DUAL-A', 'M21-DUAL-B'):
        job = summary['jobs'][job_id]
        stats = job['throughput']['interval_steps_per_sec']
        lines.append(
            f'| {job_id} | {job["physical_gpu_id"]} | '
            f'{_format_number(stats["mean"])} | {_format_number(stats["p50"])} | '
            f'{_format_number(stats["p10"])} | {_format_number(stats["p90"])} | '
            f'{job["exit_code"]} |'
        )
    lines.extend([
        '',
        '## Timing',
        '',
        '| Job | Startup/compile proxy (s) | Warmup elapsed (s) | Steady elapsed (s) | Total wall (s) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for job_id in ('M21-SINGLE', 'M21-DUAL-A', 'M21-DUAL-B'):
        job = summary['jobs'][job_id]
        timing = job['throughput']
        lines.append(
            f'| {job_id} | {_format_number(timing["startup_compile_proxy_seconds"])} | '
            f'{_format_number(timing["warmup_elapsed_seconds"])} | '
            f'{_format_number(timing["steady_elapsed_seconds"])} | '
            f'{_format_number(job["wall_seconds"])} |'
        )
    lines.extend([
        '',
        f'- `v_single`: `{_format_number(throughput["v_single"])} steps/s`',
        f'- `v_dual_a`: `{_format_number(throughput["v_dual_a"])} steps/s`',
        f'- `v_dual_b`: `{_format_number(throughput["v_dual_b"])} steps/s`',
        f'- `v_dual_total`: `{_format_number(throughput["v_dual_total"])} steps/s`',
        f'- `throughput_gain`: `{_format_number(throughput["throughput_gain"])}x`',
        f'- `per_job_slowdown_a`: `{_format_number(throughput["per_job_slowdown_a"])}x`',
        f'- `per_job_slowdown_b`: `{_format_number(throughput["per_job_slowdown_b"])}x`',
        '',
        '## GPU and system telemetry',
        '',
        '| Window | GPU-Util mean/p50/p90 | Peak memory (MiB) | Power mean/p90 (W) |',
        '| --- | ---: | ---: | ---: |',
    ])
    for label, info in (('single GPU0', gpu['single_gpu']), ('dual GPU1', gpu['dual_gpu'])):
        util = info['utilization_gpu']
        power = info['power']
        lines.append(
            f'| {label} | {_format_number(util["mean"])} / {_format_number(util["p50"])} / '
            f'{_format_number(util["p90"])} | {_format_number(info["memory_peak"])} | '
            f'{_format_number(power["mean"])} / {_format_number(power["p90"])} |'
        )
    lines.extend([
        '',
        f'- CPU utilization mean/p50/p90: `{_format_number(system["cpu_percent"]["mean"])} / '
        f'{_format_number(system["cpu_percent"]["p50"])} / '
        f'{_format_number(system["cpu_percent"]["p90"])}%`',
        f'- RAM used p90: `{_format_number(system["memory_percent"]["p90"])}%`',
        f'- RAM used peak bytes: `{_format_number(system["memory_used"]["max"], 0)}`',
        f'- process statuses: `{[(job_id, summary["jobs"][job_id]["exit_code"]) for job_id in summary["jobs"]]}`',
        f'- OOM/crash observed: `{summary["runtime_checks"]["oom_or_crash_observed"]}`',
        f'- NaN/Inf observed: `{summary["runtime_checks"]["nan_or_inf_observed"]}`',
        f'- telemetry processes terminated normally: `{summary["runtime_checks"]["telemetry_processes_normal"]}`',
        '',
        '## Decision',
        '',
        f'**{summary["recommendation"]}**',
        '',
    ])
    if summary.get('recovered_from_existing_artifacts'):
        lines.extend([
            '',
            '> This report was finalized from complete child artifacts after a parent aggregation interruption. '
            'Child run metadata and telemetry were preserved; telemetry process termination is marked as recovered.',
        ])
    lines.extend([
        '',
        'M21 was a throughput benchmark only; no M20B scientific result was produced.',
    ])
    return '\n'.join(lines) + '\n'


def _manifest(study, configuration, dataset_paths, host, output_root, jobs):
    return {
        'schema': BENCHMARK_SCHEMA,
        'benchmark_id': BENCHMARK_ID,
        'benchmark_only': True,
        'scientific_result': False,
        'status': 'planned',
        'created_at': _utc_now(),
        'source': {
            'git': _git_provenance(),
            'host': host,
            'study_path': str(study.path.resolve()),
            'config_path': str(configuration.path.resolve()),
            'config_id': configuration.config_id,
            'config_fingerprint': config_fingerprint(configuration.data),
            'dataset': {
                name: _file_provenance(path) for name, path in dataset_paths.items()
            },
        },
        'workload': {
            'algorithm': WORKLOAD_ALGORITHM,
            'environment': WORKLOAD_ENVIRONMENT,
            'train_steps': TRAIN_STEPS,
            'batch_size': BATCH_SIZE,
            'log_interval': LOG_INTERVAL,
            'warmup_steps': WARMUP_STEPS,
            'steady_state_window': [WARMUP_STEPS, TRAIN_STEPS],
            'eval_tasks': 'none',
            'save_best_checkpoint': False,
            'save_last_checkpoint': False,
            'save_interval': SAVE_INTERVAL,
            'xla_preallocate': False,
            'jax_platforms_cpu_forbidden': True,
        },
        'output_root': str(Path(output_root).resolve()),
        'jobs': {
            job.job_id: {
                'artifact_name': job.artifact_name,
                'physical_gpu_id': int(job.physical_gpu_id),
                'worker_slot': job.worker_slot,
                'run_attempt': job.run_attempt,
                'seed': 0,
                'run_root': str(_job_run_root(output_root, job).resolve()),
            }
            for job in jobs
        },
    }


def _epoch_from_iso(value):
    if not value:
        raise BenchmarkError(f'Missing ISO timestamp in benchmark metadata: {value!r}')
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except ValueError as error:
        raise BenchmarkError(f'Invalid ISO timestamp in benchmark metadata: {value!r}') from error


def finalize_existing_benchmark(output_root):
    """Finalize a completed artifact if the parent was interrupted after training.

    This recovery path only reads child metadata and telemetry already written by
    a previous M21 execution.  It never launches training or overwrites a child
    run.  It is intentionally useful for a parent-process failure between
    telemetry shutdown and summary serialization.
    """

    output_root = _assert_safe_output_root(output_root)
    manifest_path = output_root / 'manifest.json'
    if not manifest_path.is_file():
        raise BenchmarkError(f'M21 manifest is missing: {manifest_path}')
    manifest = _read_json(manifest_path)
    if manifest.get('benchmark_id') != BENCHMARK_ID:
        raise BenchmarkError(f'Not an M21 benchmark manifest: {manifest_path}')
    expected_job_ids = {job.job_id for job in BENCHMARK_JOBS}
    manifest_jobs = manifest.get('jobs', {})
    if set(manifest_jobs) != expected_job_ids:
        raise BenchmarkError(
            f'M21 manifest job IDs do not match the frozen plan: {sorted(manifest_jobs)}'
        )

    jobs = []
    records = {}
    run_dirs = {}
    for job_id in ('M21-SINGLE', 'M21-DUAL-A', 'M21-DUAL-B'):
        spec = manifest_jobs[job_id]
        job = BenchmarkJob(
            job_id,
            str(spec['artifact_name']),
            str(spec['physical_gpu_id']),
            int(spec['worker_slot']),
            int(spec['run_attempt']),
        )
        link = output_root / job.artifact_name
        if not link.is_dir():
            raise BenchmarkError(f'M21 job artifact directory is missing: {link}')
        run_dir = link.resolve()
        if output_root not in run_dir.parents:
            raise BenchmarkError(f'M21 job artifact escapes output root: {run_dir}')
        metadata_path = run_dir / 'runtime_metadata.json'
        if not metadata_path.is_file():
            raise BenchmarkError(f'M21 runtime metadata is missing: {metadata_path}')
        metadata = _read_json(metadata_path)
        process = metadata.get('benchmark_process', {})
        if metadata.get('benchmark_only') is not True or metadata.get('scientific_result') is not False:
            raise BenchmarkError(f'M21 markers are invalid in {metadata_path}')
        if metadata.get('status') != 'completed' or process.get('exit_code') != 0:
            raise BenchmarkError(f'M21 child run is not a completed success: {metadata_path}')
        if metadata.get('jax_backend') != 'gpu':
            raise BenchmarkError(f'M21 child did not record GPU JAX backend: {metadata_path}')
        start_time = process.get('start_time') or metadata.get('start_time')
        end_time = process.get('end_time') or metadata.get('end_time')
        start_epoch = _epoch_from_iso(start_time)
        end_epoch = _epoch_from_iso(end_time)
        record = {
            'pid': process.get('pid'),
            'start_epoch': start_epoch,
            'start_time': start_time,
            'end_epoch': end_epoch,
            'end_time': end_time,
            'wall_seconds': process.get('wall_seconds', end_epoch - start_epoch),
            'exit_code': int(process.get('exit_code')),
            'max_cpu_percent': process.get('cpu_percent_max'),
            'max_rss_bytes': process.get('rss_bytes_max'),
        }
        jobs.append(job)
        records[job_id] = record
        run_dirs[job_id] = run_dir

    telemetry_processes = manifest.get('telemetry_processes')
    if telemetry_processes is None:
        # The original parent reached the telemetry shutdown call before the
        # summary serialization failure.  Preserve that recovery assumption
        # explicitly in the manifest instead of silently presenting it as a
        # fresh live observation.
        telemetry_processes = {
            name: {
                'pid': None,
                'normal_termination': True,
                'exit_code': 0,
                'alive_after_join': False,
                'recovered_from_existing_artifacts': True,
            }
            for name in ('gpu', 'system')
        }
    benchmark_start = min(record['start_epoch'] for record in records.values())
    benchmark_end = max(record['end_epoch'] for record in records.values())
    link_warnings = _make_job_links(output_root, run_dirs, jobs)
    for job in jobs:
        _attach_benchmark_metadata(run_dirs[job.job_id], job, records[job.job_id])
    summary = _aggregate_results(
        output_root,
        manifest['source']['host'],
        records,
        run_dirs,
        benchmark_start,
        benchmark_end,
        tuple(jobs),
        telemetry_processes=telemetry_processes,
    )
    summary['telemetry_processes'] = telemetry_processes
    summary['artifact_link_warnings'] = link_warnings
    summary['recovered_from_existing_artifacts'] = True
    manifest['status'] = summary['status']
    manifest['completed_at'] = _utc_now()
    manifest['telemetry_processes'] = telemetry_processes
    manifest['summary_path'] = str((output_root / 'summary.json').resolve())
    manifest['report_path'] = str((output_root / 'M21_report.md').resolve())
    _write_json(output_root / 'manifest.json', manifest)
    _write_json(output_root / 'summary.json', summary)
    (output_root / 'M21_report.md').write_text(_render_report(manifest, summary))
    return summary


def dry_run(study_path=STUDY_DEFAULT, config_path=CONFIG_DEFAULT,
            dataset_root=DATASET_DEFAULT, output_root=OUTPUT_DEFAULT,
            single_gpu='0', dual_gpu='1'):
    if single_gpu == dual_gpu:
        raise BenchmarkError('M21 requires distinct single and dual physical GPU IDs')
    _assert_safe_output_root(output_root)
    _validate_workload(study_path, config_path, dataset_root)
    host = _host_provenance(require_gpu=False)
    job_plan = _benchmark_job_plan(single_gpu, dual_gpu)
    print('M21 GPU concurrency benchmark dry-run (no training started)')
    print(f'GPU{single_gpu}: 1 worker')
    print(f'GPU{dual_gpu}: 2 workers')
    print('planned jobs: 3')
    for job in job_plan:
        gpu = single_gpu if job.job_id == 'M21-SINGLE' else dual_gpu
        print(f'{job.job_id}: physical_gpu={gpu}, worker_slot={job.worker_slot}')
    print(f'workload: {WORKLOAD_CONFIG_ID} / {WORKLOAD_ENVIRONMENT}')
    print(f'train_steps={TRAIN_STEPS} batch_size={BATCH_SIZE} warmup={WARMUP_STEPS}')
    print(f'output_root={Path(output_root).resolve()}')
    if host.get('nvidia_smi_error'):
        print(f'GPU preflight warning (dry-run only): {host["nvidia_smi_error"]}')
    return 0


def execute_benchmark(study_path=STUDY_DEFAULT, config_path=CONFIG_DEFAULT,
                      dataset_root=DATASET_DEFAULT, output_root=OUTPUT_DEFAULT,
                      single_gpu='0', dual_gpu='1'):
    output_root = _assert_safe_output_root(output_root)
    if output_root.exists():
        raise BenchmarkError(
            f'Output root already exists; refusing to overwrite benchmark artifacts: {output_root}'
        )
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise BenchmarkError(
            'M21 requires an unmasked parent CUDA_VISIBLE_DEVICES so physical GPU IDs remain unambiguous'
        )
    if psutil is None:
        raise BenchmarkError('M21 requires psutil for system telemetry')
    study, configuration, dataset_paths = _validate_workload(
        study_path, config_path, dataset_root,
    )
    host = _host_provenance(require_gpu=True)
    _validate_gpu_selection(host, [single_gpu, dual_gpu])
    if host['baseline_compute_processes']:
        raise BenchmarkError(
            'Refusing to benchmark while GPU compute processes are active: '
            f'{host["baseline_compute_processes"]}'
        )
    inherited_platforms = {
        value.strip().lower()
        for value in os.environ.get('JAX_PLATFORMS', '').split(',')
        if value.strip()
    }
    if 'cpu' in inherited_platforms:
        raise BenchmarkError('M21 must not run with JAX_PLATFORMS=cpu')
    # Apply the requested physical IDs to the frozen job plan after validation.
    jobs = _benchmark_job_plan(single_gpu, dual_gpu)
    jax_probes = []
    seen_gpus = set()
    for job in jobs:
        if job.physical_gpu_id not in seen_gpus:
            jax_probes.append(_probe_jax_gpu(job))
            seen_gpus.add(job.physical_gpu_id)
    output_root.mkdir(parents=True)
    (output_root / '_runs').mkdir()
    manifest = _manifest(study, configuration, dataset_paths, host, output_root, jobs)
    manifest['jax_gpu_probes'] = jax_probes
    _write_json(output_root / 'manifest.json', manifest)
    context = mp.get_context('spawn')
    gpu_stop = context.Event()
    system_stop = context.Event()
    gpu_collector = context.Process(
        target=_gpu_telemetry_worker,
        args=(output_root / 'gpu_telemetry.csv', gpu_stop, TELEMETRY_INTERVAL_SECONDS),
        name='m21-gpu-telemetry',
    )
    system_collector = context.Process(
        target=_system_telemetry_worker,
        args=(output_root / 'system_telemetry.csv', system_stop, SYSTEM_TELEMETRY_INTERVAL_SECONDS),
        name='m21-system-telemetry',
    )
    job_records = {}
    benchmark_start = time.time()
    try:
        gpu_collector.start()
        system_collector.start()
        time.sleep(0.25)
        job_records = _launch_training_jobs(study_path, config_path, output_root, jobs)
        benchmark_start = min(record['start_epoch'] for record in job_records.values())
        try:
            job_records = _monitor_training_jobs(job_records)
        except KeyboardInterrupt:
            _terminate_training_jobs(job_records)
            raise
    finally:
        benchmark_end = time.time()
        gpu_stop_info = _stop_telemetry_process(gpu_collector, gpu_stop)
        system_stop_info = _stop_telemetry_process(system_collector, system_stop)
        manifest['telemetry_processes'] = {
            'gpu': gpu_stop_info,
            'system': system_stop_info,
        }
        _write_json(output_root / 'manifest.json', manifest)

    run_dirs = {
        job.job_id: _expected_run_dir(output_root, job, study, configuration)
        for job in jobs
    }
    link_warnings = _make_job_links(output_root, run_dirs, jobs)
    for job in jobs:
        _attach_benchmark_metadata(
            run_dirs[job.job_id], job, job_records[job.job_id],
        )
    summary = _aggregate_results(
        output_root,
        host,
        job_records,
        run_dirs,
        benchmark_start,
        benchmark_end,
        jobs,
        telemetry_processes={
            'gpu': gpu_stop_info,
            'system': system_stop_info,
        },
    )
    summary['telemetry_processes'] = {
        'gpu': gpu_stop_info,
        'system': system_stop_info,
    }
    summary['artifact_link_warnings'] = link_warnings
    manifest['status'] = summary['status']
    manifest['completed_at'] = _utc_now()
    manifest['summary_path'] = str((output_root / 'summary.json').resolve())
    manifest['report_path'] = str((output_root / 'M21_report.md').resolve())
    _write_json(output_root / 'manifest.json', manifest)
    _write_json(output_root / 'summary.json', summary)
    (output_root / 'M21_report.md').write_text(_render_report(manifest, summary))
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary['status'] == 'completed' else 1


def _positive_int(value):
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY_DEFAULT)
    parser.add_argument('--config', type=Path, default=CONFIG_DEFAULT)
    parser.add_argument('--dataset-root', type=Path, default=DATASET_DEFAULT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument('--single-gpu', default='0')
    parser.add_argument('--dual-gpu', default='1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument(
        '--finalize-existing', action='store_true',
        help='Finalize completed M21 artifacts after a parent-process interruption.',
    )
    args = parser.parse_args(argv)
    if sum(bool(mode) for mode in (
        args.dry_run, args.execute, args.finalize_existing,
    )) != 1:
        parser.error(
            'exactly one of --dry-run, --execute, or --finalize-existing is required'
        )
    try:
        if args.dry_run:
            return dry_run(
                args.study, args.config, args.dataset_root, args.output_root,
                args.single_gpu, args.dual_gpu,
            )
        if args.finalize_existing:
            summary = finalize_existing_benchmark(args.output_root)
            print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
            return 0 if summary['status'] == 'completed' else 1
        return execute_benchmark(
            args.study, args.config, args.dataset_root, args.output_root,
            args.single_gpu, args.dual_gpu,
        )
    except (BenchmarkError, OSError, ValueError) as error:
        print(f'M21 benchmark: NO-GO: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
