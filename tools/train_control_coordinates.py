#!/usr/bin/env python3
"""Train the standalone M25 Stage-1 linear Boolean coordinate flow."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from impls.agents.control_coordinate import ControlCoordinateAgent
from impls.diagnostics.puzzle.control_coordinates import (
    audit_effect_event_index,
    evaluate_control_coordinates,
    gf2_rank,
)
from impls.experiment import (
    create_run_context,
    finalize_run,
    prepare_run_design,
    update_runtime_metadata,
)
from impls.experiment.management import jsonable
from impls.networks.control_coordinates import resolve_control_coordinate_config
from impls.utils.checkpointing import should_update_best, write_checkpoint_index
from impls.utils.effect_datasets import EffectEventDataset
from impls.utils.flax_utils import save_agent, save_semantic_checkpoint
from tools.control_coordinate_common import (
    json_print,
    json_write,
    load_puzzle_event_index,
    validate_stage1_layout,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / 'experiments/M25_linear_boolean_control_coordinates/study.yaml'


def _parser():
    parser = argparse.ArgumentParser(
        description='Train an engineering-only M25 Stage-1 Boolean flow.'
    )
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--config', default='M25-E001')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset-dir', type=Path, default=None)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--run-attempt', type=int, default=0)
    parser.add_argument('--train-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument(
        '--rank-deficiency-policy', choices=('fail', 'warn'), default=None
    )
    return parser


def resolve_training_design(args):
    study, configuration = prepare_run_design(args.study, args.config)
    data = configuration.data
    if data.get('algorithm') != 'control_coordinate':
        raise ValueError('M25 Stage-1 requires algorithm=control_coordinate')
    if data.get('executable') is not True:
        raise ValueError('M25 Stage-1 configuration must explicitly set executable=true')
    if data.get('protocol_stage') != 'engineering_stage1' or data.get('formal') is not False:
        raise ValueError('This entry point accepts only explicit non-formal engineering configs')
    if study.data.get('protocol', {}).get('formal_training_started') is not False:
        raise ValueError('M25 implementation Study must not declare formal training started')
    if args.seed not in study.data['seeds']:
        raise ValueError(
            f'Run seed {args.seed} is not declared by Study seeds {study.data["seeds"]}'
        )

    flow_config = dict(data.get('control_coordinate') or {})
    event_config = dict(data.get('event_data') or {})
    launcher = dict(data.get('launcher') or {})
    if args.train_steps is not None:
        flow_config['train_steps'] = int(args.train_steps)
    if args.batch_size is not None:
        flow_config['batch_size'] = int(args.batch_size)
    flow_config['seed'] = int(args.seed)
    launcher['train_steps'] = int(flow_config['train_steps'])
    launcher['batch_size'] = int(flow_config['batch_size'])
    rank_policy = args.rank_deficiency_policy or event_config.get(
        'rank_deficiency_policy', 'fail'
    )
    if rank_policy not in ('fail', 'warn'):
        raise ValueError(f'Unsupported rank_deficiency_policy: {rank_policy!r}')
    event_config['rank_deficiency_policy'] = rank_policy
    required_event_fields = {
        'required_standard_fields', 'endpoint_semantics', 'model_input_fields',
        'effect_signature_role', 'extraction_chunk_size', 'future_window_scales',
    }
    missing_event = required_event_fields - set(event_config)
    if missing_event:
        raise ValueError(f'Event configuration is missing fields: {sorted(missing_event)}')
    if event_config['model_input_fields'] != ['start_board', 'end_board']:
        raise ValueError('Stage-1 model_input_fields must be start_board/end_board only')
    if event_config['required_standard_fields'] != [
        'observations', 'actions', 'terminals'
    ]:
        raise ValueError(
            'Stage-1 required_standard_fields must be observations/actions/terminals'
        )
    if event_config['endpoint_semantics'] != [
        'observation_t', 'action_t_index_only', 'observation_t_plus_1'
    ]:
        raise ValueError(
            'Stage-1 endpoints must be exactly observation[t], action[t] by index, '
            'and observation[t + 1]'
        )
    if event_config['effect_signature_role'] != 'diagnostic_only':
        raise ValueError('Raw effect signatures must remain diagnostic_only')
    required_launcher_fields = {
        'log_interval', 'eval_interval', 'save_interval', 'matrix_test_states',
        'goal_pair_count', 'diagnostic_batch_size', 'diagnostic_seed', 'save_best_checkpoint',
        'save_last_checkpoint', 'selection_metric',
    }
    missing_launcher = required_launcher_fields - set(launcher)
    if missing_launcher:
        raise ValueError(f'Launcher configuration is missing fields: {sorted(missing_launcher)}')
    for field in (
        'log_interval', 'eval_interval', 'save_interval', 'matrix_test_states',
        'goal_pair_count', 'diagnostic_batch_size',
    ):
        if int(launcher[field]) <= 0:
            raise ValueError(f'launcher.{field} must be positive')
    if launcher['selection_metric'] != 'metric/hard_axis_success':
        raise ValueError('Checkpoint selection must use metric/hard_axis_success')
    resolved_flow = resolve_control_coordinate_config(flow_config)
    flow_config['permutations'] = resolved_flow['permutations']
    flow_config['permutation_schedule'] = resolved_flow['permutation_schedule']
    return study, configuration, flow_config, event_config, launcher


def _csv_writer(path, fieldnames):
    file = Path(path).open('w', newline='')
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    file.flush()
    return file, writer


def _evaluate(agent, event_index, *, launcher, event_config):
    matrix = np.asarray(agent.effective_matrix()).astype(np.uint8)
    if gf2_rank(matrix) != event_index.num_bits:
        raise RuntimeError('Learned effective matrix lost full GF(2) rank')
    return evaluate_control_coordinates(
        event_index,
        encode_fn=agent.encode,
        inverse_fn=agent.inverse_encode,
        goal_mask_fn=agent.goal_mask,
        effective_matrix=matrix,
        batch_size=int(launcher['diagnostic_batch_size']),
        diagnostic_seed=int(launcher['diagnostic_seed']),
        matrix_test_states=int(launcher['matrix_test_states']),
        goal_pair_count=int(launcher['goal_pair_count']),
        window_scales=tuple(event_config['future_window_scales']),
    )


def run(args):
    (
        study,
        configuration,
        flow_config,
        event_config,
        launcher,
    ) = resolve_training_design(args)
    environment = configuration.data['environment']
    validate_stage1_layout(
        environment, configured_num_bits=flow_config['num_bits']
    )
    event_index, data_provenance = load_puzzle_event_index(
        environment,
        dataset_dir=args.dataset_dir,
        dataset_seed=args.seed,
        configured_num_bits=flow_config['num_bits'],
        chunk_size=int(event_config['extraction_chunk_size']),
    )
    audit = audit_effect_event_index(
        event_index,
        window_scales=tuple(event_config['future_window_scales']),
    )
    json_print({
        'environment': environment,
        'pretraining_event_audit_summary': {
            'number_of_episodes': audit['number_of_episodes'],
            'number_of_task_state_events': audit['number_of_task_state_events'],
            'number_of_unique_raw_xor_effects': audit[
                'number_of_unique_raw_xor_effects'
            ],
            'observed_effect_gf2_rank': audit['observed_effect_gf2_rank'],
            'observed_effects_span_full_task_space': audit[
                'observed_effects_span_full_task_space'
            ],
        },
    })
    observed_rank = audit['observed_effect_gf2_rank']
    if observed_rank < flow_config['num_bits']:
        message = (
            f'Observed event-effect rank {observed_rank} is below '
            f'N={flow_config["num_bits"]}; coordinates are not identifiable'
        )
        if event_config['rank_deficiency_policy'] == 'fail':
            raise ValueError(message)
        print(f'WARNING: {message}')
    if len(event_index) == 0:
        raise ValueError('No nonzero within-episode Puzzle events were observed')

    sampler = EffectEventDataset(
        event_index,
        seed=args.seed,
        event_sampling_mode=flow_config['event_sampling_mode'],
    )
    example_batch = sampler.sample(min(flow_config['batch_size'], len(event_index)))
    agent = ControlCoordinateAgent.create(
        flow_config, example_board=example_batch['start_board']
    )
    resolved_config = {
        'agent': jsonable(agent.config),
        'event_data': event_config,
        'launcher': launcher,
        'scientific_boundary': {
            'stage': 'M25_stage1',
            'model_inputs': ['start_board', 'end_board'],
            'objective': 'axis_locality_only',
            'gciql_integration': False,
            'physical_operation_grounding': False,
        },
    }
    context = create_run_context(
        study=study,
        configuration=configuration,
        run_root=args.run_root,
        algorithm='control_coordinate',
        environment=environment,
        seed=args.seed,
        dataset_dir=data_provenance['dataset_dir'],
        computation=True,
        compute_slots={
            'coordinate_flow': {
                'enabled': True,
                'primitive': 'linear_boolean_coupling',
                'topology': 'reversible_flow',
                'num_layers': flow_config['num_layers'],
            }
        },
        resolved_config=resolved_config,
        repo_root=ROOT,
        runtime_extras={
            'protocol_stage': 'engineering_stage1_nonformal',
            'm25_event_semantics': 'observation[t]_to_observation[t+1]_within_episode',
            'data_provenance': data_provenance,
            'permutation_schedule': flow_config['permutation_schedule'],
            'selection_metric': 'metric/hard_axis_success',
            'metric_semantics': 'hard_binary_axis_locality_not_rl_success',
            'formal_training': False,
        },
        run_attempt=args.run_attempt,
    )
    run_dir = context.run_dir
    train_fields = [
        'step',
        'loss/axis_surrogate',
        'metric/hard_axis_success',
        'metric/hard_axis_distance_mean',
        'metric/hard_axis_distance_median',
        'metric/hard_axis_distance_gt_one',
        'metric/hard_axis_distance_max',
        'grad/norm',
        'grad/min',
        'grad/max',
    ]
    eval_fields = [
        'step',
        'metric/hard_axis_success',
        'metric/hard_axis_distance_mean',
        'metric/hard_axis_distance_median',
        'metric/hard_axis_distance_gt_one',
        'metric/hard_axis_distance_max',
        'metric/observed_effect_rank',
        'metric/one_hot_unique_effects',
        'metric/distinct_axes_used',
        'metric/effective_matrix_rank',
        'metric/same_effect_consistency',
        'metric/cross_episode_consistency',
    ]
    train_file = eval_file = None
    best_record = None
    best_metric = None
    last_record = None
    last_diagnostics = None
    try:
        json_write(run_dir / 'event_audit.json', audit)
        train_file, train_writer = _csv_writer(run_dir / 'train.csv', train_fields)
        eval_file, eval_writer = _csv_writer(run_dir / 'eval.csv', eval_fields)

        def evaluate_and_record(step):
            nonlocal best_record, best_metric, last_diagnostics
            diagnostics = _evaluate(
                agent, event_index, launcher=launcher, event_config=event_config
            )
            last_diagnostics = diagnostics
            hard = diagnostics['hard_event_metrics']
            axes = diagnostics['axis_assignment']
            consistency = diagnostics['same_effect_consistency']
            matrix = diagnostics['effective_matrix']
            row = {
                'step': step,
                'metric/hard_axis_success': hard['hard_axis_success_rate'],
                'metric/hard_axis_distance_mean': hard['mean_hard_latent_event_distance'],
                'metric/hard_axis_distance_median': hard['median_hard_latent_event_distance'],
                'metric/hard_axis_distance_gt_one': hard[
                    'fraction_hard_latent_event_distance_gt_one'
                ],
                'metric/hard_axis_distance_max': hard[
                    'maximum_hard_latent_event_distance'
                ],
                'metric/observed_effect_rank': audit['observed_effect_gf2_rank'],
                'metric/one_hot_unique_effects': axes[
                    'number_mapped_to_exactly_one_latent_axis'
                ],
                'metric/distinct_axes_used': axes[
                    'number_of_distinct_latent_axes_used'
                ],
                'metric/effective_matrix_rank': matrix['effective_matrix_rank'],
                'metric/same_effect_consistency': int(
                    consistency['same_effect_latent_delta_consistency']
                ),
                'metric/cross_episode_consistency': int(
                    consistency['cross_episode_consistency']
                ),
            }
            eval_writer.writerow(row)
            eval_file.flush()
            json_write(run_dir / 'diagnostics' / f'step_{step:08d}.json', diagnostics)
            score = hard['hard_axis_success_rate']
            if should_update_best(score, best_metric):
                best_metric = score
                if launcher['save_best_checkpoint']:
                    best_record = save_semantic_checkpoint(
                        agent,
                        run_dir,
                        'best',
                        step,
                        checkpoint_metadata={
                            'selection_metric': 'metric/hard_axis_success',
                            'selection_metric_value': score,
                            'metric_semantics': 'hard_binary_axis_locality',
                        },
                    )
                    write_checkpoint_index(
                        run_dir,
                        best=best_record,
                        last=last_record,
                        selection_metric='metric/hard_axis_success',
                    )

        evaluate_and_record(0)
        train_steps = int(flow_config['train_steps'])
        for step in range(1, train_steps + 1):
            batch = sampler.sample(int(flow_config['batch_size']))
            agent, info = agent.update(batch)
            if step == 1 or step % int(launcher['log_interval']) == 0 or step == train_steps:
                row = {'step': step}
                row.update({key: float(info[key]) for key in train_fields if key != 'step'})
                train_writer.writerow(row)
                train_file.flush()
            if step % int(launcher['save_interval']) == 0 or step == train_steps:
                save_agent(
                    agent,
                    run_dir / 'checkpoints',
                    step,
                    checkpoint_metadata={
                        'checkpoint_role': 'numeric',
                        'checkpoint_step': step,
                        'metric_semantics': 'M25_stage1_axis_locality',
                    },
                )
            if step % int(launcher['eval_interval']) == 0 or step == train_steps:
                evaluate_and_record(step)

        final_score = last_diagnostics['hard_event_metrics']['hard_axis_success_rate']
        if launcher['save_last_checkpoint']:
            last_record = save_semantic_checkpoint(
                agent,
                run_dir,
                'last',
                train_steps,
                checkpoint_metadata={
                    'selection_metric': 'metric/hard_axis_success',
                    'selection_metric_value': final_score,
                    'metric_semantics': 'hard_binary_axis_locality',
                },
            )
        write_checkpoint_index(
            run_dir,
            best=best_record,
            last=last_record,
            selection_metric='metric/hard_axis_success',
        )
        final_matrix = np.asarray(agent.effective_matrix()).astype(np.uint8)
        json_write(run_dir / 'effective_matrix.json', {
            'orientation': 'A[output_bit][input_bit]',
            'matrix': final_matrix.tolist(),
            'rank': gf2_rank(final_matrix),
        })
        control_summary = {
            'status': 'completed',
            'selection_metric': 'metric/hard_axis_success',
            'best_hard_axis_success': best_metric,
            'final_hard_axis_success': final_score,
            'final_step': train_steps,
            'observed_effect_rank': observed_rank,
            'effective_matrix_rank': gf2_rank(final_matrix),
            'formal_training': False,
        }
        json_write(run_dir / 'control_coordinate_summary.json', control_summary)
        update_runtime_metadata(run_dir, {
            'final_hard_axis_success': final_score,
            'effective_matrix_rank': gf2_rank(final_matrix),
            'structural_invariants': last_diagnostics['structural_invariants'],
        })
        finalize_run(run_dir, 'completed')
        result = {'run_dir': str(run_dir), **control_summary}
        json_print(result)
        return result
    except Exception as error:
        finalize_run(run_dir, 'failed', failure_reason=error)
        raise
    finally:
        if train_file is not None:
            train_file.close()
        if eval_file is not None:
            eval_file.close()


def main():
    run(_parser().parse_args())


if __name__ == '__main__':
    main()
