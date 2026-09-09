"""Run deterministic post-hoc physical-press diagnosis on one checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path


PRESS_EVENT_FIELDS = (
    'environment', 'task_id', 'task_name', 'episode_index',
    'transition_index', 'step', 'source_button_indices', 'source_mask',
    'num_sources', 'event_type', 'board_before', 'board_after', 'board_delta',
    'predicted_board_delta', 'effect_consistent', 'goal_board', 'Dstar_before',
    'Dstar_after', 'delta_Dstar', 'progress_class',
    'random_progress_probability', 'progress_advantage',
    'steps_since_previous_press', 'remaining_episode_steps',
)
EPISODE_FIELDS = (
    'environment', 'task_id', 'task_name', 'episode_index',
    'episode_seed', 'actor_seed', 'noise_seed', 'success', 'episode_length',
    'terminated', 'truncated', 'horizon_exhausted', 'max_episode_steps',
    'initial_Dstar', 'final_Dstar', 'min_Dstar', 'num_press_events',
    'num_single_press_events', 'num_multi_press_events', 'num_source_presses',
    'num_unique_source_buttons', 'num_repeat_source_presses',
    'first_press_step', 'num_progress_presses', 'num_neutral_presses',
    'num_regress_presses', 'progress_rate', 'mean_progress_advantage',
    'mean_steps_between_press_events', 'median_steps_between_press_events',
    'max_steps_between_press_events', 'final_board_goal_consistent',
)
VECTOR_FIELDS = {
    'source_button_indices', 'source_mask', 'board_before', 'board_after',
    'board_delta', 'predicted_board_delta', 'goal_board',
}
NULL = 'null'


def _parser():
    parser = argparse.ArgumentParser(
        description='Deterministic, observation-only Puzzle press diagnosis.'
    )
    parser.add_argument('--source-run', required=True, type=Path)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--checkpoint-step', type=int)
    selector.add_argument('--checkpoint-role', choices=('best', 'last'))
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--evaluation-seed', required=True, type=int)
    parser.add_argument('--episodes-per-task', required=True, type=int)
    parser.add_argument('--task-id', action='append', type=int, dest='task_ids')
    parser.add_argument('--eval-temperature', type=float, default=0.0)
    parser.add_argument('--eval-gaussian', type=float, default=None)
    parser.add_argument('--jax-platform', choices=('cpu', 'gpu'), default=None)
    parser.add_argument('--cuda-visible-devices', default=None)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def _selector(args):
    if args.checkpoint_step is not None:
        if args.checkpoint_step < 0:
            raise ValueError('--checkpoint-step must be non-negative')
        return {'selector': 'step', 'step': int(args.checkpoint_step)}
    return str(args.checkpoint_role)


def _json_compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def _json_write(path, value):
    with Path(path).open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write('\n')


def _git_metadata(repo_root):
    repo_root = Path(repo_root).resolve()
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout
    return commit, bool(status)


def _scalar_csv(value):
    if value is None:
        return NULL
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return NULL
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return value


def _csv_row(row):
    result = {}
    for key, value in row.items():
        if key in VECTOR_FIELDS:
            result[key] = _json_compact([int(item) for item in value])
        else:
            result[key] = _scalar_csv(value)
    return result


def _write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def _config_get(config, key, default=None):
    if hasattr(config, 'get'):
        return config.get(key, default)
    return getattr(config, key, default)


def _task_ids(task_infos, requested):
    available = tuple(range(1, len(task_infos) + 1))
    if requested is None:
        return available
    result = tuple(dict.fromkeys(int(task_id) for task_id in requested))
    invalid = [task_id for task_id in result if task_id not in available]
    if invalid:
        raise ValueError(f'Unknown task IDs {invalid}; available={available}')
    return result


def _episode_trajectory(trajectory, *, terminated, truncated):
    """Add only rollout termination metadata; no policy-facing values change."""

    enriched = {key: list(values) for key, values in trajectory.items()}
    enriched['terminated'] = [bool(terminated)] * len(enriched['observation'])
    enriched['truncated'] = [bool(truncated)] * len(enriched['observation'])
    return enriched


def _run(args):
    if args.episodes_per_task <= 0:
        raise ValueError('--episodes-per-task must be positive')
    if args.eval_temperature != 0.0:
        raise ValueError('Puzzle diagnosis requires --eval-temperature=0')
    if args.eval_gaussian is not None and args.eval_gaussian < 0:
        raise ValueError('--eval-gaussian must be non-negative or omitted')
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Output root is not empty: {output_root}')

    if args.jax_platform is not None:
        os.environ['JAX_PLATFORMS'] = args.jax_platform
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices)

    from impls.diagnostics.puzzle import analyze_puzzle_episode
    from impls.experiment.reevaluation import (
        _make_restored_agent,
        _restore_probe,
        validate_source_run,
    )
    from impls.utils.checkpointing import tree_fingerprint
    from impls.utils.evaluation import _rollout_episode, common_episode_seeds

    selector = _selector(args)
    provenance = validate_source_run(
        args.source_run,
        checkpoint_selector=selector,
        check_checkpoint_metadata=True,
        allow_running_source_if_checkpoint_best=False,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    restored, env, config, example_batch = _make_restored_agent(provenance)
    try:
        algorithm = provenance['source_metadata'].get('algorithm') or _config_get(config, 'agent_name')
        _restore_probe(
            restored,
            example_batch,
            algorithm,
            provenance['source_training_seed'],
        )
        import jax
        import numpy as np

        base = getattr(env, 'unwrapped', env)
        rows = int(getattr(base, '_num_rows', 0))
        cols = int(getattr(base, '_num_cols', 0))
        num_buttons = rows * cols
        task_infos = getattr(base, 'task_infos', None)
        if rows <= 0 or cols <= 0 or task_infos is None:
            raise ValueError('Source environment is not a canonical task-mode Puzzle environment')
        selected_tasks = _task_ids(task_infos, args.task_ids)
        max_episode_steps = getattr(getattr(env, 'spec', None), 'max_episode_steps', None)
        if max_episode_steps is not None:
            max_episode_steps = int(max_episode_steps)

        diagnostic_commit, diagnostic_dirty = _git_metadata(args.repo_root)
        network_fingerprint_before = tree_fingerprint(restored.network.params)
        episode_rows = []
        press_rows = []
        invariant_totals = {
            'event_effect_consistency_failures': 0,
            'unreachable_residuals': 0,
            'malformed_board_observations': 0,
            'multi_source_events': 0,
        }
        for task_id in selected_tasks:
            task_info = task_infos[int(task_id) - 1]
            task_name = str(task_info['task_name'])
            goal_board = np.asarray(task_info['goal_button_states'], dtype=np.uint8)
            if goal_board.shape != (num_buttons,):
                raise ValueError(f'Invalid canonical goal board for task {task_id}: {goal_board.shape}')
            for episode_index in range(args.episodes_per_task):
                seeds = common_episode_seeds(args.evaluation_seed, task_id, episode_index)
                rollout = _rollout_episode(
                    restored,
                    env,
                    task_id=int(task_id),
                    config=config,
                    episode_seed=seeds['episode_seed'],
                    actor_seed=seeds['actor_seed'],
                    noise_seed=seeds['noise_seed'],
                    eval_temperature=0.0,
                    eval_gaussian=args.eval_gaussian,
                    retain_trajectory=True,
                    render=False,
                    video_frame_skip=1,
                )
                result = analyze_puzzle_episode(
                    _episode_trajectory(
                        rollout['trajectory'],
                        terminated=rollout['terminated'],
                        truncated=rollout['truncated'],
                    ),
                    goal_board=goal_board,
                    rows=rows,
                    cols=cols,
                    max_episode_steps=max_episode_steps,
                )
                for key, value in result['invariants'].items():
                    invariant_totals[key] += int(value)
                for event in result['press_events']:
                    press_rows.append({
                        'environment': provenance['source_environment'],
                        'task_id': int(task_id),
                        'task_name': task_name,
                        'episode_index': int(episode_index),
                        **event,
                    })
                episode_rows.append({
                    'environment': provenance['source_environment'],
                    'task_id': int(task_id),
                    'task_name': task_name,
                    'episode_index': int(episode_index),
                    'episode_seed': seeds['episode_seed'],
                    'actor_seed': seeds['actor_seed'],
                    'noise_seed': seeds['noise_seed'],
                    **result['episode'],
                })

        network_fingerprint_after = tree_fingerprint(restored.network.params)
        if network_fingerprint_before != network_fingerprint_after:
            raise RuntimeError('Diagnostics changed restored network parameters')
        _write_csv(output_root / 'episodes.csv', EPISODE_FIELDS, episode_rows)
        _write_csv(output_root / 'press_events.csv', PRESS_EVENT_FIELDS, press_rows)

        resolved_agent_config = (
            provenance['resolved_config'].get('algorithm_config', {}).get('agent')
            or provenance['resolved_config'].get('agent')
        )
        relevant_keys = (
            'agent_name', 'alpha', 'expectile', 'tau', 'high_alpha', 'low_alpha',
            'subgoal_steps', 'actor_loss', 'batch_size', 'frame_stack', 'discrete',
        )
        actual_parameters = {
            key: resolved_agent_config.get(key)
            for key in relevant_keys
            if isinstance(resolved_agent_config, dict) and key in resolved_agent_config
        }
        manifest = {
            'diagnostic_schema_version': 'puzzle_event_diagnosis_v1',
            'diagnostic_code_commit': diagnostic_commit,
            'diagnostic_git_dirty': diagnostic_dirty,
            'source_study': provenance['source_study_id'],
            'source_config': provenance['source_config_id'],
            'source_config_slug': provenance['source_config_slug'],
            'source_environment': provenance['source_environment'],
            'source_training_seed': provenance['source_training_seed'],
            'source_run_dir': provenance['source_run_dir'],
            'source_commit_sha': provenance['source_git_commit'],
            'source_git_dirty': provenance['source_git_dirty'],
            'checkpoint_path': provenance['checkpoint_path'],
            'checkpoint_selector': provenance['requested_checkpoint_selector'],
            'checkpoint_role': provenance['resolved_checkpoint_role'],
            'checkpoint_step': provenance['checkpoint_step'],
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            'resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
            'resolved_agent_config': resolved_agent_config,
            'actual_algorithm_parameters': actual_parameters,
            'restored_network_fingerprint_before': network_fingerprint_before,
            'restored_network_fingerprint_after': network_fingerprint_after,
            'rows': rows,
            'cols': cols,
            'num_buttons': num_buttons,
            'toggle_matrix_orientation': 'columns_are_press_sources',
            'board_flattening': 'row_major',
            'button_feature_layout': '[state0,state1,pos_x120,velocity]',
            'press_threshold_raw': -0.02,
            'position_scale': 120.0,
            'goal_board_source': 'canonical_task_infos',
            'evaluation_seed': int(args.evaluation_seed),
            'seed_scheme': 'common_task_episode_v1',
            'eval_temperature': 0.0,
            'eval_gaussian': args.eval_gaussian,
            'episodes_per_task': int(args.episodes_per_task),
            'task_ids': list(selected_tasks),
            'episodes': len(episode_rows),
            'jax_backend': jax.default_backend(),
            'jax_devices': [str(device) for device in jax.devices()],
            'invariants': invariant_totals,
            'output_root': str(output_root),
        }
        _json_write(output_root / 'manifest.json', manifest)
        return manifest
    finally:
        env.close()


def main(argv=None):
    args = _parser().parse_args(argv)
    manifest = _run(args)
    print(json.dumps({
        'status': 'completed',
        'output_root': manifest['output_root'],
        'episodes': manifest['episodes'],
        'press_events': max(
            0,
            sum(1 for _ in Path(manifest['output_root']).joinpath('press_events.csv').open()) - 1,
        ),
        'invariants': manifest['invariants'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
