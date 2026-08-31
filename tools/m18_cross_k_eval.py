"""M18-D1 actor inference-depth probe for immutable M18 best checkpoints.

This diagnostic is evaluation-only. A source M18 checkpoint was trained with
one joint K for actor/value/critic. D1 changes only the actor's inference
execution depth; value and critic retain source K and do not generate rollout
actions. It never trains, finetunes, saves checkpoints, or writes to a source
run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.experiment.reevaluation import ReevaluationError, _resolved_agent_config, validate_source_run
from impls.utils.checkpointing import normalize_checkpoint_selector, sha256_file


STUDY_ID = 'M18'
DIAGNOSTIC_ID = 'M18-D1'
ENVIRONMENT = 'puzzle-4x4-play-v0'
K_VALUES = (1, 2, 4, 8)
MAX_ACTOR_TEST_K = 8
SLOT_NAMES = ('actor', 'value', 'critic')
DEFAULT_OUTPUT_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'


def _parse_csv_ints(value, option, *, allowed=None, min_value=None, max_value=None):
    try:
        values = tuple(sorted({int(item.strip()) for item in str(value).split(',') if item.strip()}))
    except ValueError as error:
        raise ValueError(f'{option} must contain comma-separated integers') from error
    if not values:
        raise ValueError(f'{option} must contain at least one integer')
    if allowed is not None and not set(values).issubset(set(allowed)):
        raise ValueError(f'{option} must be a subset of {tuple(allowed)!r}, got {values!r}')
    if min_value is not None and min(values) < int(min_value):
        raise ValueError(f'{option} values must be >= {min_value}')
    if max_value is not None and max(values) > int(max_value):
        raise ValueError(f'{option} values must be <= {max_value}')
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


def _expected_iterations(expected):
    if isinstance(expected, int):
        return {slot_name: int(expected) for slot_name in SLOT_NAMES}
    if not isinstance(expected, dict) or set(expected) != set(SLOT_NAMES):
        raise ReevaluationError(f'Expected per-slot iterations for {SLOT_NAMES!r}, got {expected!r}')
    return {slot_name: int(expected[slot_name]) for slot_name in SLOT_NAMES}


def validate_m18_agent_config(agent_config, expected_iterations, *, label='resolved agent'):
    """Validate frozen M18 semantics with explicitly per-slot K values."""

    expected_by_slot = _expected_iterations(expected_iterations)
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
            ('enabled', True), ('primitive', 'mlp'), ('structure', 'puzzle_tokens'),
            ('block', 'mlp_mixer'), ('topology', 'single_state'),
            ('parameter_sharing', 'shared'), ('credit', 'direct'),
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
            ('iterations', expected_by_slot[slot_name]), ('input_mapping', 'identity'),
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


def prepare_actor_test_time_config(resolved_agent, k_actor_test):
    """Return a same-shape config changing only actor inference K."""

    k_actor_test = int(k_actor_test)
    if not 1 <= k_actor_test <= MAX_ACTOR_TEST_K:
        raise ReevaluationError(
            f'K_actor_test must be in [1, {MAX_ACTOR_TEST_K}], got {k_actor_test!r}'
        )
    config = copy.deepcopy(resolved_agent)
    train_k = _uniform_train_k(config, label='source resolved agent')
    validate_m18_agent_config(config, train_k, label='source resolved agent')
    config['compute']['actor']['topology_kwargs']['iterations'] = k_actor_test
    validate_m18_agent_config(
        config,
        {'actor': k_actor_test, 'value': train_k, 'critic': train_k},
        label='actor-test-time resolved agent',
    )
    return config


def _configurations(study_path, train_ks):
    study = load_study(study_path)
    if study.study_id != STUDY_ID:
        raise ReevaluationError(f'Expected M18 Study, got {study.study_id!r}')
    if study.data.get('environments') != [ENVIRONMENT] or study.data.get('seeds') != [0]:
        raise ReevaluationError('M18-D1 requires the frozen Puzzle-4x4, seed-0 matrix')
    configurations = [
        prepare_run_design(study.path, path)[1]
        for path in sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    ]
    by_k = {
        int(configuration.data.get('factors', {}).get('recurrent_compute_budget_K')): configuration
        for configuration in configurations
    }
    if set(by_k) != set(K_VALUES) or len(configurations) != len(K_VALUES):
        raise ReevaluationError(f'M18-D1 requires exactly source configurations K={K_VALUES!r}')
    return study, [by_k[k] for k in train_ks]


def _stable_checkpoint_sha256(path):
    """Hash a published semantic artifact twice and reject concurrent mutation."""

    path = Path(path)
    if not path.is_file():
        raise ReevaluationError(f'Missing checkpoint: {path}')
    before = path.stat()
    first = sha256_file(path)
    middle = path.stat()
    second = sha256_file(path)
    after = path.stat()
    stamps = {(item.st_size, item.st_mtime_ns) for item in (before, middle, after)}
    if len(stamps) != 1 or first != second:
        raise ReevaluationError(f'Checkpoint changed while being read; refusing restore: {path}')
    return first


def _latest_saved_training_step(source_run_dir):
    values = []
    for path in (Path(source_run_dir) / 'checkpoints').glob('params_*.pkl'):
        match = re.fullmatch(r'params_(\d+)\.pkl', path.name)
        if match and path.is_file():
            values.append(int(match.group(1)))
    return max(values) if values else None


def _validate_best_selection(provenance):
    metadata = provenance.get('checkpoint_metadata') or {}
    if provenance.get('resolved_checkpoint_role') != 'best' or metadata.get('checkpoint_role') != 'best':
        raise ReevaluationError('M18-D1 primary diagnostic requires checkpoint role=best')
    metric = metadata.get('selection_metric')
    try:
        metric_value = float(metadata.get('selection_metric_value'))
    except (TypeError, ValueError) as error:
        raise ReevaluationError('Best checkpoint has no numeric selection_metric_value') from error
    if not isinstance(metric, str) or not metric or not metric_value == metric_value:
        raise ReevaluationError('Best checkpoint selection metadata is invalid')
    return metric, metric_value


def validate_m18_best_source(source_run_dir, *, expected_study_id=STUDY_ID, expected_environment=ENVIRONMENT):
    """Validate a clean completed/running source through its stable best pointer."""

    provenance = validate_source_run(
        source_run_dir,
        checkpoint_selector='best',
        expected_study_id=expected_study_id,
        expected_environment=expected_environment,
        allow_running_source_if_checkpoint_best=True,
    )
    stable_hash = _stable_checkpoint_sha256(provenance['checkpoint_path'])
    if stable_hash != provenance['checkpoint_sha256']:
        raise ReevaluationError('Semantic best checkpoint hash changed during source validation')
    metric, metric_value = _validate_best_selection(provenance)
    provenance = dict(provenance)
    provenance.update({
        'source_checkpoint_selection_metric': metric,
        'source_checkpoint_selection_metric_value': metric_value,
        'source_training_latest_step_at_diagnostic': _latest_saved_training_step(source_run_dir),
        'source_checkpoint_hash_at_validation': stable_hash,
    })
    return provenance


def _output_dir(output_root, configuration, k_actor_test, checkpoint_selector):
    return (
        Path(output_root) / 'M18D' / 'cross_k'
        / f'checkpoint_{_checkpoint_label(checkpoint_selector)}'
        / f'{configuration.config_id}__{configuration.slug}'
        / ENVIRONMENT / 'seed_000'
        / f'KactorTest_{int(k_actor_test)}'
    )


def plan_jobs(
    study_path,
    source_run_root,
    output_root,
    checkpoint_selector='best',
    *,
    train_ks=K_VALUES,
    actor_test_ks=K_VALUES,
):
    """Build the D1 matrix without creating diagnostic files."""

    selector = normalize_checkpoint_selector(checkpoint_selector)
    if selector['selector'] != 'best':
        raise ReevaluationError('M18-D1 currently requires --checkpoint best')
    study, configurations = _configurations(study_path, train_ks)
    jobs = []
    for configuration in configurations:
        train_k = int(configuration.data['factors']['recurrent_compute_budget_K'])
        source_run_dir = make_run_path(
            source_run_root, study.study_id, configuration.config_id, configuration.slug,
            ENVIRONMENT, 0, run_attempt=0,
        )
        provenance = None
        error = None
        try:
            provenance = validate_m18_best_source(source_run_dir)
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
        for k_actor_test in actor_test_ks:
            output_dir = _output_dir(output_root, configuration, k_actor_test, selector)
            jobs.append({
                'study': study,
                'configuration': configuration,
                'K_train': train_k,
                'K_actor_test': int(k_actor_test),
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
        'K_train', 'K_actor_test', 'study_id', 'config_id', 'environment', 'training_seed',
        'checkpoint_role', 'checkpoint_step', 'checkpoint_selection_metric',
        'checkpoint_selection_metric_value', 'task_id', 'task_name', 'episode_index',
        'evaluation_seed', 'task_seed', 'episode_seed', 'actor_seed', 'noise_seed',
        'success', 'episode_return', 'episode_length', 'terminated', 'truncated',
        'paired_episode_id', 'final_info_json',
    )
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def _build_restored_agent(provenance, k_actor_test):
    """Restore a same-shape GCIQL agent preserving value/critic source K."""

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
    config = prepare_actor_test_time_config(source_config, k_actor_test)
    validate_m18_agent_config(
        config,
        {'actor': int(k_actor_test), 'value': k_train, 'critic': k_train},
        label='M18-D1 runtime agent',
    )
    algorithm = metadata.get('algorithm') or config.get('agent_name')
    if algorithm != 'gciql' or algorithm not in agents:
        raise ReevaluationError(f'M18-D1 requires a GCIQL source, got {algorithm!r}')
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
    dataset_classes = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset, 'MultiHGCDataset': MultiHGCDataset}
    dataset_name = config.get('dataset_class')
    if dataset_name not in dataset_classes:
        env.close()
        raise ReevaluationError(f'Unsupported M18 source dataset class: {dataset_name!r}')
    dataset = dataset_classes[dataset_name](
        raw_train, config, rng=derive_seed(provenance['source_training_seed'], 11)
    )
    example_batch = dataset.sample(1)
    agent = agents[algorithm].create(
        provenance['source_training_seed'], example_batch['observations'], example_batch['actions'], config,
    )
    target_slot_accounting = _computation_slot_accounting(agent, config)
    target_architecture = gciql_architecture_accounting(agent.network.params, config, target_slot_accounting)
    source_slot_accounting = metadata.get('computation_slot_accounting', {})
    if not isinstance(source_slot_accounting, dict):
        env.close()
        raise ReevaluationError('Source has no computation_slot_accounting provenance')
    for slot_name in SLOT_NAMES:
        source_slot = source_slot_accounting.get(slot_name, {})
        target_slot = target_slot_accounting.get(slot_name, {})
        if source_slot.get('trainable_params') != target_slot.get('trainable_params'):
            env.close()
            raise ReevaluationError(f'{slot_name}: trainable parameter count changed in D1 config')
        expected_k = int(k_actor_test) if slot_name == 'actor' else k_train
        if target_slot.get('block_depth_L') != 2 or target_slot.get('iterations_K') != expected_k:
            env.close()
            raise ReevaluationError(f'{slot_name}: accounting does not expose requested actor-only K')
    source_architecture = metadata.get('architecture_accounting', {})
    if source_architecture.get('total_trainable_params') != target_architecture.get('total_trainable_params'):
        env.close()
        raise ReevaluationError('Total online GCIQL parameter count changed across actor test depth')
    initial_params = count_parameters(agent.network.params)
    restored = restore_agent_from_checkpoint(agent, provenance['checkpoint_path'])
    if count_parameters(restored.network.params) != initial_params:
        env.close()
        raise ReevaluationError('Checkpoint restore changed the parameter count')
    import jax

    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(restored.network.params)):
        env.close()
        raise ReevaluationError('Restored M18 checkpoint has non-finite parameters')
    return restored, env, config, example_batch, target_slot_accounting, target_architecture


def _task_summary(records, episodes_per_task):
    import numpy as np

    result = {}
    for task_id in range(1, 6):
        rows = [row for row in records if row['task_id'] == task_id]
        if len(rows) != int(episodes_per_task):
            raise ReevaluationError(f'Task {task_id} has {len(rows)} records, expected {episodes_per_task}')
        successes = np.asarray([row['success'] for row in rows], dtype=np.float64)
        returns = np.asarray([row['episode_return'] for row in rows], dtype=np.float64)
        lengths = np.asarray([row['episode_length'] for row in rows], dtype=np.float64)
        result[str(task_id)] = {
            'task_name': next(row['task_name'] for row in rows),
            'success': float(np.mean(successes)),
            'success_count': int(np.rint(np.sum(successes))),
            'episodes': int(len(rows)),
            'mean_episode_return': float(np.mean(returns)),
            'mean_episode_length': float(np.mean(lengths)),
        }
    return result


def run_one(
    job,
    *,
    episodes_per_task,
    evaluation_seed,
    eval_temperature,
    eval_gaussian,
    diagnostic_code_commit,
    assigned_gpu=None,
):
    """Run one immutable D1 cell, writing only under its diagnostic directory."""

    if job['provenance'] is None:
        raise ReevaluationError(f'Cannot evaluate invalid source: {job.get("error")}')
    if not diagnostic_code_commit:
        raise ReevaluationError('M18-D1 execute requires an explicit diagnostic_code_commit')
    output_dir = Path(job['output_dir'])
    if output_dir.exists():
        raise FileExistsError(f'M18-D1 output exists; refusing overwrite: {output_dir}')
    provenance = job['provenance']
    checkpoint_hash_before = _stable_checkpoint_sha256(provenance['checkpoint_path'])
    if checkpoint_hash_before != provenance['checkpoint_sha256']:
        raise ReevaluationError('Source checkpoint changed after planning; refusing execution')
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / 'm18d_metadata.json'
    metadata = {
        'status': 'running',
        'diagnostic_id': DIAGNOSTIC_ID,
        'diagnostic_name': 'Actor Inference-Depth Probe',
        'diagnostic_code_commit': str(diagnostic_code_commit),
        'source_run_dir': provenance['source_run_dir'],
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'source_config_slug': provenance['source_config_slug'],
        'source_run_status_at_diagnostic': provenance['source_run_status_at_validation'],
        'source_training_latest_step_at_diagnostic': provenance['source_training_latest_step_at_diagnostic'],
        'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'source_git_commit': provenance['source_git_commit'],
        'source_checkpoint_path': provenance['checkpoint_path'],
        'source_checkpoint_sha256': provenance['checkpoint_sha256'],
        'source_checkpoint_hash_before': checkpoint_hash_before,
        'source_checkpoint_role': provenance['resolved_checkpoint_role'],
        'source_checkpoint_step': provenance['checkpoint_step'],
        'source_checkpoint_selection_metric': provenance['source_checkpoint_selection_metric'],
        'source_checkpoint_selection_metric_value': provenance['source_checkpoint_selection_metric_value'],
        'K_train_actor_value_critic': job['K_train'],
        'K_train': job['K_train'],
        'K_actor_test': job['K_actor_test'],
        'test_time_override': 'compute.actor.topology_kwargs.iterations only',
        'value_critic_runtime_iterations': job['K_train'],
        'rollout_action_generation_slots': ['actor'],
        'value_or_critic_used_for_rollout_action_generation': False,
        'assigned_gpu': assigned_gpu,
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
        'evaluation_protocol': {
            'episodes_per_task': int(episodes_per_task),
            'evaluation_seed': int(evaluation_seed),
            'paired_seed_scheme': 'common_task_episode_v1',
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
            provenance, job['K_actor_test']
        )
        import numpy as np
        from impls.utils.evaluation import evaluate_episodes

        actor_mean = restored.network.select('actor')(
            example_batch['observations'][:1], example_batch['actor_goals'][:1], temperature=0.0
        ).mode()
        if not np.all(np.isfinite(np.asarray(actor_mean))):
            raise ReevaluationError('Restored M18-D1 deterministic actor probe is non-finite')
        task_infos = getattr(env.unwrapped, 'task_infos', None)
        if task_infos is None or len(task_infos) != 5:
            raise ReevaluationError('M18-D1 requires the five Puzzle-4x4 evaluation tasks')
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
                    'K_actor_test': job['K_actor_test'],
                    'study_id': STUDY_ID,
                    'config_id': provenance['source_config_id'],
                    'environment': ENVIRONMENT,
                    'training_seed': provenance['source_training_seed'],
                    'checkpoint_role': provenance['resolved_checkpoint_role'],
                    'checkpoint_step': provenance['checkpoint_step'],
                    'checkpoint_selection_metric': provenance['source_checkpoint_selection_metric'],
                    'checkpoint_selection_metric_value': provenance['source_checkpoint_selection_metric_value'],
                    **row,
                })
        tasks = _task_summary(records, episodes_per_task)
        successes = np.asarray([row['success'] for row in records], dtype=np.float64)
        returns = np.asarray([row['episode_return'] for row in records], dtype=np.float64)
        lengths = np.asarray([row['episode_length'] for row in records], dtype=np.float64)
        checkpoint_hash_after = _stable_checkpoint_sha256(provenance['checkpoint_path'])
        if checkpoint_hash_after != checkpoint_hash_before:
            raise ReevaluationError('Source checkpoint SHA256 changed during M18-D1 execution')
        summary = {
            'status': 'completed',
            'diagnostic_id': DIAGNOSTIC_ID,
            'diagnostic_name': 'Actor Inference-Depth Probe',
            'diagnostic_code_commit': str(diagnostic_code_commit),
            'source_config_id': provenance['source_config_id'],
            'source_run_status_at_diagnostic': provenance['source_run_status_at_validation'],
            'source_training_latest_step_at_diagnostic': provenance['source_training_latest_step_at_diagnostic'],
            'source_git_commit': provenance['source_git_commit'],
            'environment': ENVIRONMENT,
            'training_seed': provenance['source_training_seed'],
            'checkpoint_role': provenance['resolved_checkpoint_role'],
            'checkpoint_step': provenance['checkpoint_step'],
            'checkpoint_selection_metric': provenance['source_checkpoint_selection_metric'],
            'checkpoint_selection_metric_value': provenance['source_checkpoint_selection_metric_value'],
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            'source_checkpoint_hash_before': checkpoint_hash_before,
            'source_checkpoint_hash_after': checkpoint_hash_after,
            'source_checkpoint_immutable': True,
            'K_train': job['K_train'],
            'K_actor_test': job['K_actor_test'],
            'overall_success': float(np.mean(successes)),
            'success_count': int(np.rint(np.sum(successes))),
            'episodes': int(len(records)),
            'episodes_per_task': int(episodes_per_task),
            'mean_episode_return': float(np.mean(returns)),
            'mean_episode_length': float(np.mean(lengths)),
            'tasks': tasks,
            'evaluation_seed': int(evaluation_seed),
            'evaluation_only': True,
            'finetuning': False,
            'optimizer_updates': 0,
            'architecture_accounting': architecture,
            'computation_slot_accounting': slot_accounting,
        }
        _write_task_records(output_dir / 'task_results.csv', records)
        _write_json(output_dir / 'summary.json', summary)
        metadata.update({
            'status': 'completed',
            'total_episodes': len(records),
            'source_checkpoint_hash_after': checkpoint_hash_after,
            'source_checkpoint_immutable': True,
        })
        _write_json(metadata_path, metadata)
        return summary
    except BaseException as error:
        try:
            checkpoint_hash_after = _stable_checkpoint_sha256(provenance['checkpoint_path'])
            metadata['source_checkpoint_hash_after'] = checkpoint_hash_after
            metadata['source_checkpoint_immutable'] = checkpoint_hash_after == checkpoint_hash_before
        except BaseException as hash_error:
            metadata['source_checkpoint_hash_after_error'] = f'{type(hash_error).__name__}: {hash_error}'
        metadata.update({'status': 'failed', 'failure_reason': f'{type(error).__name__}: {error}'})
        _write_json(metadata_path, metadata)
        raise
    finally:
        if env is not None:
            env.close()


def _completed_summaries(output_root, checkpoint_selector='best'):
    root = Path(output_root) / 'M18D' / 'cross_k' / f'checkpoint_{_checkpoint_label(checkpoint_selector)}'
    rows = []
    if root.is_dir():
        for path in sorted(root.rglob('summary.json')):
            try:
                with path.open() as file:
                    summary = json.load(file)
                if summary.get('status') == 'completed' and summary.get('diagnostic_id') == DIAGNOSTIC_ID:
                    rows.append(summary | {'summary_path': str(path)})
            except (OSError, TypeError, ValueError):
                continue
    return root, sorted(rows, key=lambda row: (int(row['K_train']), int(row['K_actor_test'])))


def aggregate(output_root, checkpoint_selector='best'):
    """Create separate matrix and diagonal-relative summaries once."""

    cross_root, rows = _completed_summaries(output_root, checkpoint_selector)
    if not rows:
        raise ReevaluationError(f'No completed M18-D1 summaries found under {cross_root}')
    summary_root = Path(output_root) / 'M18D' / 'summary' / f'checkpoint_{_checkpoint_label(checkpoint_selector)}'
    if summary_root.exists():
        raise FileExistsError(f'M18-D1 summary exists; refusing overwrite: {summary_root}')
    summary_root.mkdir(parents=True)
    fields = (
        'K_train', 'K_actor_test', 'overall_success', 'success_count', 'episodes',
        'mean_episode_return', 'mean_episode_length', 'checkpoint_step',
        'checkpoint_selection_metric', 'checkpoint_selection_metric_value',
        'checkpoint_sha256', 'source_run_status_at_diagnostic', 'summary_path',
    )
    with (summary_root / 'm18d_cross_k_success.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])
    lookup = {(int(row['K_train']), int(row['K_actor_test'])): float(row['overall_success']) for row in rows}
    delta_rows = []
    for row in rows:
        diagonal = lookup.get((int(row['K_train']), int(row['K_train'])))
        delta_rows.append({
            'K_train': int(row['K_train']),
            'K_actor_test': int(row['K_actor_test']),
            'overall_success': float(row['overall_success']),
            'diagonal_success': diagonal,
            'delta_vs_trained_actor_depth': None if diagonal is None else float(row['overall_success']) - diagonal,
        })
    with (summary_root / 'm18d_cross_k_delta.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=tuple(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)
    lines = [
        '# M18-D1 Actor Inference-Depth Matrix（自动生成）',
        '',
        '- 行：joint train K（actor/value/critic）；列：仅 actor 的 K_actor_test。',
        '- checkpoint=best 是机制诊断选择，不替代 M18 final@1M primary endpoint。',
        '- value/critic 保持 source K，且 rollout action generation 只调用 actor。',
        '- 所有值为描述性结果；不构成因果证明或 adaptive test-time compute 结论。',
        '',
        '| K_train \\ K_actor_test | ' + ' | '.join(str(k) for k in K_VALUES) + ' |',
        '|' + '---:|' * (len(K_VALUES) + 1),
    ]
    for k_train in K_VALUES:
        lines.append('| ' + str(k_train) + ' | ' + ' | '.join(
            '' if (k_train, k_actor_test) not in lookup else f'{lookup[(k_train, k_actor_test)]:.6f}'
            for k_actor_test in K_VALUES
        ) + ' |')
    lines.extend(['', '## Diagonal-relative effect', '', '| K_train | K_actor_test | Success | S(i,j)-S(i,i) |', '|---:|---:|---:|---:|'])
    for row in delta_rows:
        delta = row['delta_vs_trained_actor_depth']
        lines.append(
            f'| {row["K_train"]} | {row["K_actor_test"]} | {row["overall_success"]:.6f} | '
            + ('' if delta is None else f'{delta:.6f}') + ' |'
        )
    (summary_root / 'M18D_cross_k_report.md').write_text('\n'.join(lines) + '\n')
    _write_json(summary_root / 'm18d_cross_k_summary.json', {
        'diagnostic_id': DIAGNOSTIC_ID,
        'checkpoint_role': _checkpoint_label(checkpoint_selector),
        'rows': rows,
        'diagonal_relative_effects': delta_rows,
    })
    return summary_root, rows, delta_rows


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
                    '--diagnostic-code-commit', str(args.diagnostic_code_commit),
                    '--worker-source-run-dir', str(job['source_run_dir']),
                    '--worker-config', str(job['configuration'].path),
                    '--worker-k-actor-test', str(job['K_actor_test']),
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
    provenance = validate_m18_best_source(args.worker_source_run_dir)
    source_config = _resolved_agent_config(provenance['resolved_config'])
    k_train = _uniform_train_k(source_config, label=configuration.config_id)
    expected_train_k = int(configuration.data['factors']['recurrent_compute_budget_K'])
    if k_train != expected_train_k:
        raise ReevaluationError('Worker source K_train does not match declarative M18 configuration')
    job = {
        'configuration': configuration,
        'K_train': k_train,
        'K_actor_test': int(args.worker_k_actor_test),
        'checkpoint_selector': {'selector': 'best'},
        'source_run_dir': Path(args.worker_source_run_dir),
        'provenance': provenance,
        'output_dir': _output_dir(args.output_root, configuration, args.worker_k_actor_test, 'best'),
    }
    run_one(
        job,
        episodes_per_task=args.episodes_per_task,
        evaluation_seed=args.evaluation_seed,
        eval_temperature=args.eval_temperature,
        eval_gaussian=args.eval_gaussian,
        diagnostic_code_commit=args.diagnostic_code_commit,
        assigned_gpu=args.assigned_gpu,
    )


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M18_puzzle_recurrent_compute_scaling/study.yaml')
    parser.add_argument('--source-run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--checkpoint', default='best', help='M18-D1 primary selector; currently best only.')
    parser.add_argument('--train-ks', default='1,2,4,8', help='Joint source K values.')
    parser.add_argument(
        '--actor-test-ks', '--test-ks', dest='actor_test_ks', default='1,2,4,8',
        help='Actor-only test K values; optional dense range 1..8 is supported.',
    )
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--episodes-per-task', type=int, default=20)
    parser.add_argument('--evaluation-seed', type=int, default=18018)
    parser.add_argument('--eval-temperature', type=float, default=0.0)
    parser.add_argument('--eval-gaussian', type=float, default=None)
    parser.add_argument('--diagnostic-code-commit', default=None, help='User-supplied reviewed diagnostic code commit.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--worker-source-run-dir', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--worker-config', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--worker-k-actor-test', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--assigned-gpu', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.episodes_per_task <= 0:
        parser.error('--episodes-per-task must be positive')
    return args


def main(argv=None):
    args = _args(argv)
    if args.assigned_gpu is not None:
        # Runs before the worker lazily imports JAX through agent construction.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.assigned_gpu)
    try:
        selector = normalize_checkpoint_selector(args.checkpoint)
        if selector['selector'] != 'best':
            raise ReevaluationError('M18-D1 currently requires --checkpoint best')
        if args.aggregate_only:
            if args.dry_run or args.execute:
                raise ReevaluationError('--aggregate-only cannot be combined with --dry-run/--execute')
            root, rows, _ = aggregate(args.output_root, selector)
            print(f'M18-D1 aggregate rows={len(rows)} root={root}')
            return 0
        if args.worker_source_run_dir is not None:
            if args.worker_config is None or args.worker_k_actor_test is None:
                raise ReevaluationError('Worker mode requires --worker-config and --worker-k-actor-test')
            if not args.diagnostic_code_commit:
                raise ReevaluationError('Worker mode requires --diagnostic-code-commit')
            _worker_job(args)
            return 0
        if args.dry_run == args.execute:
            raise ReevaluationError('Exactly one of --dry-run or --execute is required')
        args.gpus = _parse_gpus(args.gpus)
        train_ks = _parse_csv_ints(args.train_ks, '--train-ks', allowed=K_VALUES)
        actor_test_ks = _parse_csv_ints(
            args.actor_test_ks, '--actor-test-ks', min_value=1, max_value=MAX_ACTOR_TEST_K,
        )
        if args.execute and not args.diagnostic_code_commit:
            raise ReevaluationError('--execute requires --diagnostic-code-commit from the user-reviewed commit')
        jobs = plan_jobs(
            args.study, args.source_run_root, args.output_root, selector,
            train_ks=train_ks, actor_test_ks=actor_test_ks,
        )
        counts = {key: sum(job['status'] == key for job in jobs) for key in ('planned', 'output_exists', 'invalid_source')}
        print(
            f'M18-D1 actor-only plan: total={len(jobs)} planned={counts["planned"]} '
            f'output_exists={counts["output_exists"]} invalid_source={counts["invalid_source"]}'
        )
        for job in jobs:
            prefix = f'Ktrain={job["K_train"]} KactorTest={job["K_actor_test"]}'
            if job['status'] == 'planned':
                print(
                    f'[PLANNED] {prefix} source_status={job["provenance"]["source_run_status_at_validation"]} '
                    f'source={job["source_run_dir"]} output={job["output_dir"]}'
                )
            else:
                print(
                    f'[{job["status"].upper()}] {prefix} source={job["source_run_dir"]} '
                    f'{job.get("error") or job["output_dir"]}', file=sys.stderr,
                )
        if args.dry_run:
            return 0 if counts['invalid_source'] == 0 and counts['output_exists'] == 0 else 2
        if counts['invalid_source'] or counts['output_exists']:
            raise ReevaluationError('M18-D1 execute requires every source valid and every output path absent')
        failures = _dispatch(jobs, args)
        root, rows, _ = aggregate(args.output_root, selector)
        print(f'M18-D1 execute completed with failures={failures}; aggregate rows={len(rows)} root={root}')
        return 1 if failures else 0
    except (FileExistsError, OSError, ValueError, ReevaluationError) as error:
        print(f'M18-D1: FAIL: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

