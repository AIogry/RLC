"""M18-D: checkpoint-only K_train x K_test evaluation for M18.

This is deliberately an evaluation-only diagnostic.  It reconstructs a
completed M18 checkpoint with the same parameter tree, changes only the
SingleState execution iteration count for actor/value/critic, and evaluates
the resulting policy.  It never calls an optimizer update, trains, resumes a
training run, or writes to the source checkpoint directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.reevaluation import ReevaluationError, _resolved_agent_config, validate_source_run
from impls.utils.checkpointing import normalize_checkpoint_selector


STUDY_ID = 'M18'
ENVIRONMENT = 'puzzle-4x4-play-v0'
K_VALUES = (1, 2, 4, 8)
SLOT_NAMES = ('actor', 'value', 'critic')
DEFAULT_OUTPUT_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'


def _parse_csv_set(value, option, *, allowed=None):
    values = tuple(sorted({int(item.strip()) for item in str(value).split(',') if item.strip()}))
    if not values:
        raise ValueError(f'{option} must contain at least one integer')
    if allowed is not None and not set(values).issubset(set(allowed)):
        raise ValueError(f'{option} must be a subset of {tuple(allowed)!r}, got {values!r}')
    return values


def _parse_gpus(value):
    gpus = tuple(item.strip() for item in str(value).split(',') if item.strip())
    if not gpus or len(gpus) != len(set(gpus)) or any(not item.isdigit() for item in gpus):
        raise ValueError('--gpus must contain one or more unique numeric physical GPU IDs')
    return gpus


def _checkpoint_label(selector):
    normalized = normalize_checkpoint_selector(selector)
    return normalized['selector'] if normalized['selector'] != 'step' else f"step{normalized['step']}"


def _require(value, expected, label):
    if value != expected:
        raise ReevaluationError(f'{label}: expected {expected!r}, got {value!r}')


def validate_m18_agent_config(agent_config, expected_k, *, label='resolved agent'):
    """Validate the frozen M18 semantics before applying a test-time K override."""

    if not isinstance(agent_config, dict):
        raise ReevaluationError(f'{label}: agent configuration must be a mapping')
    _require(agent_config.get('agent_name'), 'gciql', f'{label}.agent_name')
    try:
        alpha = float(agent_config.get('alpha'))
    except (TypeError, ValueError) as error:
        raise ReevaluationError(f'{label}.alpha must be numeric') from error
    if abs(alpha - 0.4) > 1e-12:
        raise ReevaluationError(f'{label}.alpha must be 0.4, got {alpha!r}')
    _require(agent_config.get('actor_loss'), 'ddpgbc', f'{label}.actor_loss')
    compute = agent_config.get('compute', {})
    if set(compute) != set(SLOT_NAMES):
        raise ReevaluationError(f'{label}.compute slots must be {SLOT_NAMES!r}')
    for slot_name in SLOT_NAMES:
        slot = compute[slot_name]
        prefix = f'{label}.compute.{slot_name}'
        for key, expected in (
            ('enabled', True),
            ('primitive', 'mlp'),
            ('structure', 'puzzle_tokens'),
            ('block', 'mlp_mixer'),
            ('topology', 'single_state'),
            ('parameter_sharing', 'shared'),
            ('credit', 'direct'),
            ('readout', 'mean_context'),
        ):
            _require(slot.get(key), expected, f'{prefix}.{key}')
        structure = slot.get('structure_kwargs', {})
        for key, expected in (
            ('num_buttons', 16), ('robot_dim', 19), ('button_feature_dim', 4),
            ('token_dim', 128), ('robot_hidden_dim', 128), ('index_embedding', True),
        ):
            _require(structure.get(key), expected, f'{prefix}.structure_kwargs.{key}')
        block = slot.get('block_kwargs', {})
        for key, expected in (
            ('num_blocks', 2), ('token_hidden_dim', 64),
            ('channel_hidden_dim', 256), ('tm_mode', 'none'),
        ):
            _require(block.get(key), expected, f'{prefix}.block_kwargs.{key}')
        topology = slot.get('topology_kwargs', {})
        for key, expected in (
            ('iterations', expected_k), ('input_mapping', 'identity'),
            ('state_dim', 128), ('state_init', 'zero_buffer'),
            ('input_injection', 'z_plus_x'), ('residual', False),
            ('parameter_sharing', 'shared'),
        ):
            _require(topology.get(key), expected, f'{prefix}.topology_kwargs.{key}')
        try:
            init_std = float(topology.get('state_init_std'))
        except (TypeError, ValueError) as error:
            raise ReevaluationError(f'{prefix}.topology_kwargs.state_init_std must be numeric') from error
        if abs(init_std - 1.0) > 1e-12:
            raise ReevaluationError(f'{prefix}.topology_kwargs.state_init_std must be 1.0')
        _require(slot.get('readout_kwargs', {}).get('output_dim'), 512, f'{prefix}.readout_kwargs.output_dim')


def prepare_test_time_config(resolved_agent, k_test):
    """Return a parameter-shape-preserving M18 inference configuration.

    The only modified leaves are the three declarative `iterations` values.
    This intentionally changes execution depth, not adapter/block/readout
    parameters, optimizer state, checkpoint contents, or training semantics.
    """

    if int(k_test) not in K_VALUES:
        raise ReevaluationError(f'K_test must be one of {K_VALUES!r}, got {k_test!r}')
    config = copy.deepcopy(resolved_agent)
    train_k = _uniform_train_k(config, label='source resolved agent')
    validate_m18_agent_config(config, train_k, label='source resolved agent')
    for slot_name in SLOT_NAMES:
        config['compute'][slot_name]['topology_kwargs']['iterations'] = int(k_test)
    validate_m18_agent_config(config, int(k_test), label='test-time resolved agent')
    return config


def _uniform_train_k(agent_config, *, label):
    compute = agent_config.get('compute', {}) if isinstance(agent_config, dict) else {}
    values = []
    for slot_name in SLOT_NAMES:
        try:
            values.append(int(compute[slot_name]['topology_kwargs']['iterations']))
        except (KeyError, TypeError, ValueError) as error:
            raise ReevaluationError(f'{label}: missing integer iterations for {slot_name}') from error
    if len(set(values)) != 1:
        raise ReevaluationError(f'{label}: actor/value/critic train K differs: {values!r}')
    if values[0] not in K_VALUES:
        raise ReevaluationError(f'{label}: K_train must be in {K_VALUES!r}, got {values[0]!r}')
    return values[0]


def _configurations(study_path, config_filter=None):
    study = load_study(study_path)
    if study.study_id != STUDY_ID:
        raise ReevaluationError(f'Expected M18 Study, got {study.study_id!r}')
    if study.data.get('environments') != [ENVIRONMENT] or study.data.get('seeds') != [0]:
        raise ReevaluationError('M18-D requires the frozen Puzzle-4x4, seed-0 matrix')
    configurations = [
        prepare_run_design(study.path, path)[1]
        for path in sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    ]
    observed = {
        configuration.data.get('factors', {}).get('recurrent_compute_budget_K')
        for configuration in configurations
    }
    if observed != set(K_VALUES) or len(configurations) != len(K_VALUES):
        raise ReevaluationError(f'M18-D requires exactly K={K_VALUES!r} source configurations')
    if config_filter is not None:
        known = {configuration.config_id for configuration in configurations}
        unknown = set(config_filter) - known
        if unknown:
            raise ReevaluationError(f'Unknown M18 config IDs: {sorted(unknown)!r}')
        configurations = [item for item in configurations if item.config_id in config_filter]
    return study, configurations


def _output_dir(output_root, configuration, k_test, checkpoint_selector):
    return (
        Path(output_root) / 'M18D'
        / f'{configuration.config_id}__{configuration.slug}'
        / ENVIRONMENT / 'seed_000'
        / f'checkpoint_{_checkpoint_label(checkpoint_selector)}'
        / f'Ktest_{int(k_test)}'
    )


def plan_jobs(study_path, source_run_root, output_root, checkpoint_selector, *, config_filter=None, test_ks=K_VALUES):
    """Build the full planned diagnostic matrix without writing any artifacts."""

    study, configurations = _configurations(study_path, config_filter)
    selector = normalize_checkpoint_selector(checkpoint_selector)
    jobs = []
    for configuration in configurations:
        train_k = configuration.data['factors']['recurrent_compute_budget_K']
        source_run_dir = make_run_path(
            source_run_root, study.study_id, configuration.config_id, configuration.slug,
            ENVIRONMENT, 0, run_attempt=0,
        )
        provenance = None
        error = None
        try:
            provenance = validate_source_run(
                source_run_dir,
                checkpoint_selector=selector,
                expected_study_id=STUDY_ID,
                expected_environment=ENVIRONMENT,
            )
            if provenance['source_config_id'] != configuration.config_id:
                raise ReevaluationError('Source config ID does not match planned M18 configuration')
            resolved_agent = _resolved_agent_config(provenance['resolved_config'])
            actual_train_k = _uniform_train_k(resolved_agent, label=configuration.config_id)
            if actual_train_k != train_k:
                raise ReevaluationError(
                    f'{configuration.config_id}: resolved K_train={actual_train_k}, expected {train_k}'
                )
            validate_m18_agent_config(resolved_agent, train_k, label=configuration.config_id)
        except (FileNotFoundError, OSError, ValueError, ReevaluationError) as caught:
            error = str(caught)
        for k_test in test_ks:
            output_dir = _output_dir(output_root, configuration, k_test, selector)
            jobs.append({
                'study': study,
                'configuration': configuration,
                'K_train': int(train_k),
                'K_test': int(k_test),
                'checkpoint_selector': selector,
                'source_run_dir': Path(source_run_dir),
                'provenance': provenance,
                'output_dir': output_dir,
                'status': 'planned' if provenance is not None and not output_dir.exists() else (
                    'output_exists' if provenance is not None else 'invalid_source'
                ),
                'error': error,
            })
    return jobs


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_task_records(path, rows):
    fields = (
        'K_train', 'K_test', 'study_id', 'config_id', 'environment', 'training_seed',
        'checkpoint_role', 'checkpoint_step', 'task_id', 'task_name', 'episode_index',
        'evaluation_seed', 'task_seed', 'episode_seed', 'actor_seed', 'noise_seed',
        'success', 'episode_return', 'episode_length', 'terminated', 'truncated',
        'paired_episode_id', 'final_info_json',
    )
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _build_restored_agent(provenance, k_test):
    """Instantiate a same-shape test-time-K agent and restore source weights."""

    import numpy as np

    from impls.agents import agents
    from impls.computation.accounting import count_parameters, gciql_architecture_accounting
    from impls.main import _computation_slot_accounting
    from impls.utils.datasets import GCDataset, HGCDataset, MultiHGCDataset
    from impls.utils.env_utils import make_env_and_datasets
    from impls.utils.flax_utils import restore_agent_from_checkpoint
    from impls.utils.reproducibility import derive_seed

    metadata = provenance['source_metadata']
    source_config = _resolved_agent_config(provenance['resolved_config'])
    k_train = _uniform_train_k(source_config, label=provenance['source_config_id'])
    config = prepare_test_time_config(source_config, k_test)
    algorithm = metadata.get('algorithm') or config.get('agent_name')
    if algorithm != 'gciql' or algorithm not in agents:
        raise ReevaluationError(f'M18-D requires a GCIQL source, got {algorithm!r}')
    dataset_dir = metadata.get('dataset_dir')
    if not dataset_dir:
        raise ReevaluationError('Source metadata has no dataset_dir')
    env, raw_train, _ = make_env_and_datasets(
        ENVIRONMENT,
        frame_stack=config.get('frame_stack'),
        seed=derive_seed(provenance['source_training_seed'], 3),
        dataset_seed=derive_seed(provenance['source_training_seed'], 1),
        dataset_dir=dataset_dir,
    )
    dataset_classes = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'MultiHGCDataset': MultiHGCDataset,
    }
    dataset_name = config.get('dataset_class')
    if dataset_name not in dataset_classes:
        env.close()
        raise ReevaluationError(f'Unsupported M18 source dataset class: {dataset_name!r}')
    dataset = dataset_classes[dataset_name](
        raw_train, config, rng=derive_seed(provenance['source_training_seed'], 11)
    )
    example_batch = dataset.sample(1)
    agent = agents[algorithm].create(
        provenance['source_training_seed'],
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    target_slot_accounting = _computation_slot_accounting(agent, config)
    target_architecture = gciql_architecture_accounting(
        agent.network.params, config, target_slot_accounting,
    )
    source_slot_accounting = metadata.get('computation_slot_accounting', {})
    if not isinstance(source_slot_accounting, dict):
        env.close()
        raise ReevaluationError('Source has no computation_slot_accounting provenance')
    for slot_name in SLOT_NAMES:
        source_slot = source_slot_accounting.get(slot_name, {})
        target_slot = target_slot_accounting.get(slot_name, {})
        if source_slot.get('trainable_params') != target_slot.get('trainable_params'):
            env.close()
            raise ReevaluationError(
                f'{slot_name}: trainable parameter count changed from K_train={k_train} to K_test={k_test}'
            )
        if target_slot.get('block_depth_L') != 2 or target_slot.get('iterations_K') != int(k_test):
            env.close()
            raise ReevaluationError(f'{slot_name}: target accounting does not expose the requested M18 L/K')
    source_architecture = metadata.get('architecture_accounting', {})
    if source_architecture.get('total_trainable_params') != target_architecture.get('total_trainable_params'):
        env.close()
        raise ReevaluationError('Total online GCIQL trainable parameter count changed across K_test')
    initial_params = count_parameters(agent.network.params)
    restored = restore_agent_from_checkpoint(agent, provenance['checkpoint_path'])
    if count_parameters(restored.network.params) != initial_params:
        env.close()
        raise ReevaluationError('Checkpoint restore changed the parameter count')
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in __import__('jax').tree_util.tree_leaves(restored.network.params)):
        env.close()
        raise ReevaluationError('Restored M18 checkpoint has non-finite parameters')
    return restored, env, config, example_batch, target_slot_accounting, target_architecture


def run_one(job, *, episodes_per_task, evaluation_seed, eval_temperature, eval_gaussian, assigned_gpu=None):
    """Run one K_train/K_test cell, writing only under its diagnostic output path."""

    if job['provenance'] is None:
        raise ReevaluationError(f'Cannot evaluate invalid source: {job.get("error")}')
    output_dir = Path(job['output_dir'])
    if output_dir.exists():
        raise FileExistsError(f'M18-D output exists; refusing overwrite: {output_dir}')
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / 'm18d_metadata.json'
    provenance = job['provenance']
    metadata = {
        'status': 'running',
        'diagnostic_id': 'M18-D',
        'source_run_dir': provenance['source_run_dir'],
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'source_config_slug': provenance['source_config_slug'],
        'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'source_checkpoint_path': provenance['checkpoint_path'],
        'source_checkpoint_sha256': provenance['checkpoint_sha256'],
        'checkpoint_role': provenance['resolved_checkpoint_role'],
        'checkpoint_step': provenance['checkpoint_step'],
        'K_train': job['K_train'],
        'K_test': job['K_test'],
        'test_time_override': 'compute.actor/value/critic.topology_kwargs.iterations only',
        'assigned_gpu': assigned_gpu,
        'evaluation_protocol': {
            'episodes_per_task': int(episodes_per_task),
            'evaluation_seed': int(evaluation_seed),
            'eval_temperature': float(eval_temperature),
            'eval_gaussian': eval_gaussian,
            'video_episodes': 0,
            'training_or_optimizer_updates': 0,
        },
    }
    _write_json(metadata_path, metadata)
    env = None
    try:
        restored, env, config, example_batch, slot_accounting, architecture = _build_restored_agent(
            provenance, job['K_test']
        )
        import jax
        import numpy as np
        from impls.utils.evaluation import evaluate_episodes

        goal = example_batch['actor_goals']
        action = restored.sample_actions(
            example_batch['observations'][:1], goal[:1], seed=jax.random.PRNGKey(18018)
        )
        if not np.all(np.isfinite(np.asarray(action))):
            raise ReevaluationError('Restored M18-D action probe is non-finite')
        task_infos = getattr(env.unwrapped, 'task_infos', None)
        if task_infos is None or len(task_infos) != 5:
            raise ReevaluationError('M18-D requires the five Puzzle-4x4 evaluation tasks')
        records = []
        for task_id, info in enumerate(task_infos, start=1):
            task_rows = evaluate_episodes(
                restored,
                env,
                task_id=task_id,
                task_name=str(info['task_name']),
                config=config,
                evaluation_seed=int(evaluation_seed),
                episode_indices=range(int(episodes_per_task)),
                eval_temperature=float(eval_temperature),
                eval_gaussian=eval_gaussian,
            )
            for row in task_rows:
                records.append({
                    'K_train': job['K_train'],
                    'K_test': job['K_test'],
                    'study_id': STUDY_ID,
                    'config_id': provenance['source_config_id'],
                    'environment': ENVIRONMENT,
                    'training_seed': provenance['source_training_seed'],
                    'checkpoint_role': provenance['resolved_checkpoint_role'],
                    'checkpoint_step': provenance['checkpoint_step'],
                    **row,
                })
        task_summary = {}
        for task_id in range(1, 6):
            values = [row['success'] for row in records if row['task_id'] == task_id]
            if len(values) != int(episodes_per_task):
                raise ReevaluationError(f'Task {task_id} has {len(values)} records, expected {episodes_per_task}')
            task_summary[str(task_id)] = {
                'task_name': next(row['task_name'] for row in records if row['task_id'] == task_id),
                'success': float(np.mean(values)),
                'episodes': len(values),
            }
        summary = {
            'status': 'completed',
            'diagnostic_id': 'M18-D',
            'source_config_id': provenance['source_config_id'],
            'environment': ENVIRONMENT,
            'training_seed': provenance['source_training_seed'],
            'checkpoint_role': provenance['resolved_checkpoint_role'],
            'checkpoint_step': provenance['checkpoint_step'],
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            'K_train': job['K_train'],
            'K_test': job['K_test'],
            'overall_success': float(np.mean([item['success'] for item in task_summary.values()])),
            'tasks': task_summary,
            'episodes_per_task': int(episodes_per_task),
            'total_episodes': len(records),
            'evaluation_seed': int(evaluation_seed),
            'architecture_accounting': architecture,
            'computation_slot_accounting': slot_accounting,
        }
        _write_task_records(output_dir / 'task_results.csv', records)
        _write_json(output_dir / 'summary.json', summary)
        metadata.update({'status': 'completed', 'total_episodes': len(records)})
        _write_json(metadata_path, metadata)
        return summary
    except BaseException as error:
        metadata.update({'status': 'failed', 'failure_reason': f'{type(error).__name__}: {error}'})
        _write_json(metadata_path, metadata)
        raise
    finally:
        if env is not None:
            env.close()


def aggregate(output_root):
    """Write an auditable M18-D matrix from completed per-cell summaries."""

    root = Path(output_root) / 'M18D'
    rows = []
    if root.exists():
        for path in sorted(root.rglob('summary.json')):
            try:
                with path.open() as file:
                    summary = json.load(file)
                if summary.get('status') == 'completed':
                    rows.append({
                        'K_train': int(summary['K_train']),
                        'K_test': int(summary['K_test']),
                        'overall_success': float(summary['overall_success']),
                        'checkpoint_role': summary.get('checkpoint_role'),
                        'checkpoint_step': summary.get('checkpoint_step'),
                        'summary_path': str(path),
                    })
            except (OSError, TypeError, ValueError, KeyError):
                continue
    rows.sort(key=lambda row: (row['K_train'], row['K_test']))
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'M18D_cross_k_matrix.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('K_train', 'K_test', 'overall_success', 'checkpoint_role', 'checkpoint_step', 'summary_path'))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(row['K_train'], row['K_test']): row['overall_success'] for row in rows}
    lines = [
        '# M18-D cross-K test-time depth matrix（自动生成）',
        '',
        '- 行为训练时 `K_train`，列为仅在推理时覆盖的 `K_test`。',
        '- 对角线是 canonical trained-depth evaluation；上三角是 test-time depth extrapolation probe；下三角是 reduced-compute/early-exit probe。',
        '- 这不是 finetuning、ACT、learned halting 或 test-time scaling proof。',
        '',
        '| K_train \\ K_test | 1 | 2 | 4 | 8 |',
        '|---:|---:|---:|---:|---:|',
    ]
    for k_train in K_VALUES:
        lines.append('| ' + str(k_train) + ' | ' + ' | '.join(
            '' if (k_train, k_test) not in lookup else f'{lookup[(k_train, k_test)]:.6f}'
            for k_test in K_VALUES
        ) + ' |')
    (root / 'M18D_cross_k_matrix.md').write_text('\n'.join(lines) + '\n')
    return rows


def _dispatch(jobs, args):
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
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    '--study', str(args.study),
                    '--source-run-root', str(args.source_run_root),
                    '--output-root', str(args.output_root),
                    '--checkpoint', str(args.checkpoint),
                    '--episodes-per-task', str(args.episodes_per_task),
                    '--evaluation-seed', str(args.evaluation_seed),
                    '--eval-temperature', str(args.eval_temperature),
                    '--worker-source-run-dir', str(job['source_run_dir']),
                    '--worker-config', str(job['configuration'].path),
                    '--worker-k-test', str(job['K_test']),
                    '--assigned-gpu', str(gpu),
                ]
                if args.eval_gaussian is not None:
                    command.extend(['--eval-gaussian', str(args.eval_gaussian)])
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    failures += 1
            finally:
                work.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        return sum(future.result() for future in [executor.submit(worker, gpu) for gpu in args.gpus])


def _worker_job(args):
    _, configuration = prepare_run_design(args.study, args.worker_config)
    selector = normalize_checkpoint_selector(args.checkpoint)
    provenance = validate_source_run(
        args.worker_source_run_dir,
        checkpoint_selector=selector,
        expected_study_id=STUDY_ID,
        expected_environment=ENVIRONMENT,
    )
    source_config = _resolved_agent_config(provenance['resolved_config'])
    k_train = _uniform_train_k(source_config, label=configuration.config_id)
    expected_train_k = configuration.data['factors']['recurrent_compute_budget_K']
    if k_train != expected_train_k:
        raise ReevaluationError('Worker source K_train does not match its declarative M18 configuration')
    job = {
        'configuration': configuration,
        'K_train': k_train,
        'K_test': int(args.worker_k_test),
        'checkpoint_selector': selector,
        'source_run_dir': Path(args.worker_source_run_dir),
        'provenance': provenance,
        'output_dir': _output_dir(args.output_root, configuration, args.worker_k_test, selector),
    }
    run_one(
        job,
        episodes_per_task=args.episodes_per_task,
        evaluation_seed=args.evaluation_seed,
        eval_temperature=args.eval_temperature,
        eval_gaussian=args.eval_gaussian,
        assigned_gpu=args.assigned_gpu,
    )


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M18_puzzle_recurrent_compute_scaling/study.yaml')
    parser.add_argument('--source-run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--checkpoint', default='last', help='best, last, or an explicit numeric checkpoint step.')
    parser.add_argument('--configs', default=None, help='Optional comma-separated M18 config IDs.')
    parser.add_argument('--test-ks', default='1,2,4,8')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--episodes-per-task', type=int, default=20)
    parser.add_argument('--evaluation-seed', type=int, default=18018)
    parser.add_argument('--eval-temperature', type=float, default=0.0)
    parser.add_argument('--eval-gaussian', type=float, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--worker-source-run-dir', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--worker-config', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--worker-k-test', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--assigned-gpu', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.episodes_per_task <= 0:
        parser.error('--episodes-per-task must be positive')
    return args


def main(argv=None):
    args = _args(argv)
    if args.assigned_gpu is not None:
        # This occurs before the worker imports JAX through agent construction.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.assigned_gpu)
    try:
        normalize_checkpoint_selector(args.checkpoint)
        if args.aggregate_only:
            rows = aggregate(args.output_root)
            print(f'M18-D aggregate rows={len(rows)} root={Path(args.output_root) / "M18D"}')
            return 0
        if args.worker_source_run_dir is not None:
            if args.worker_config is None or args.worker_k_test is None:
                raise ReevaluationError('Worker mode requires --worker-config and --worker-k-test')
            _worker_job(args)
            return 0
        if args.dry_run == args.execute:
            raise ReevaluationError('Exactly one of --dry-run or --execute is required')
        args.gpus = _parse_gpus(args.gpus)
        test_ks = _parse_csv_set(args.test_ks, '--test-ks', allowed=K_VALUES)
        config_filter = None
        if args.configs:
            config_filter = {item.strip() for item in args.configs.split(',') if item.strip()}
            if not config_filter:
                raise ReevaluationError('--configs must contain at least one config ID')
        jobs = plan_jobs(
            args.study, args.source_run_root, args.output_root, args.checkpoint,
            config_filter=config_filter, test_ks=test_ks,
        )
        counts = {key: sum(job['status'] == key for job in jobs) for key in ('planned', 'output_exists', 'invalid_source')}
        print(
            f'M18-D plan: total={len(jobs)} planned={counts["planned"]} '
            f'output_exists={counts["output_exists"]} invalid_source={counts["invalid_source"]}'
        )
        for job in jobs:
            prefix = f'Ktrain={job["K_train"]} Ktest={job["K_test"]}'
            if job['status'] == 'planned':
                print(f'[PLANNED] {prefix} source={job["source_run_dir"]} output={job["output_dir"]}')
            else:
                print(f'[{job["status"].upper()}] {prefix} source={job["source_run_dir"]} {job.get("error") or job["output_dir"]}', file=sys.stderr)
        if args.dry_run:
            return 0 if counts['invalid_source'] == 0 and counts['output_exists'] == 0 else 2
        if counts['invalid_source'] or counts['output_exists']:
            raise ReevaluationError('M18-D execute requires every source valid and every output path absent')
        failures = _dispatch(jobs, args)
        rows = aggregate(args.output_root)
        print(f'M18-D execute completed with failures={failures}; aggregate rows={len(rows)}')
        return 1 if failures else 0
    except (FileExistsError, OSError, ValueError, ReevaluationError) as error:
        print(f'M18-D: FAIL: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
