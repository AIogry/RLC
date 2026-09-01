"""M18-D6 cross-actor × cross-critic preference diagnostic.

This is evaluation-only.  It reuses the immutable D234 fixed batch and its
stored D3 final clipped actions, locks K4/K8 source checkpoints to the exact
D1/D234 artifact SHA256 values, and compares actions only *within* each
critic.  It never trains, updates an optimizer, saves a checkpoint, or falls
back to a current semantic-best pointer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment.reevaluation import ReevaluationError
from impls.utils.checkpointing import tree_fingerprint
from tools import m18_d_reference as reference
from tools import m18_trace_diagnostics as trace


DIAGNOSTIC_ID = 'M18-D6'
STUDY_ID = 'M18'
ENVIRONMENT = 'puzzle-4x4-play-v0'
TRAIN_KS = (4, 8)
EPSILON = 1e-8
TIE_TOLERANCE = 1e-6
DEFAULT_OUTPUT_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics'
DEFAULT_SOURCE_RUN_ROOT = '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])


def _parse_train_ks(value):
    try:
        parsed = tuple(sorted({int(item.strip()) for item in str(value).split(',') if item.strip()}))
    except ValueError as error:
        raise ReevaluationError('--train-ks must contain comma-separated integers') from error
    if parsed != TRAIN_KS:
        raise ReevaluationError(f'M18-D6 requires exactly --train-ks 4,8, got {parsed!r}')
    return parsed


def _scalar(array, label):
    value = np.asarray(array)
    if value.size != 1:
        raise ReevaluationError(f'{label} must be a scalar, got shape {value.shape!r}')
    return value.reshape(()).item()


def _output_dir(output_root, *, reference_batch_size, reference_diagnostic_seed, max_samples):
    full_label = f'fixed_batch_N{int(reference_batch_size)}_seed{int(reference_diagnostic_seed)}'
    if int(max_samples) != int(reference_batch_size):
        full_label += f'__smoke_prefixN{int(max_samples)}'
    return (
        Path(output_root) / 'M18D' / 'cross_actor_critic' / 'checkpoint_locked' / full_label
    )


def _load_d3_final_action(path, *, train_k, expected_sample_id, max_samples, expected_checkpoint_step):
    """Read exactly the D3 saved deterministic clipped action at native K."""

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError) as error:
        raise ReevaluationError(f'Cannot load D3 actor artifact {path}: {error}') from error
    required = {
        'sample_id', 'iteration_k', 'slot', 'K_train', 'checkpoint_role', 'checkpoint_step',
        'ensemble_member', 'clipped_action', 'normal_actor_mode_at_train_k',
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ReevaluationError(f'D3 actor artifact lacks required arrays {missing!r}: {path}')
    if _scalar(arrays['slot'], f'{path}.slot') != 'actor':
        raise ReevaluationError(f'D3 artifact is not actor metrics: {path}')
    if int(_scalar(arrays['K_train'], f'{path}.K_train')) != int(train_k):
        raise ReevaluationError(f'D3 K_train does not match requested K{train_k}: {path}')
    if _scalar(arrays['checkpoint_role'], f'{path}.checkpoint_role') != 'best':
        raise ReevaluationError(f'D3 artifact checkpoint role is not best: {path}')
    checkpoint_step = int(_scalar(arrays['checkpoint_step'], f'{path}.checkpoint_step'))
    if checkpoint_step != int(expected_checkpoint_step):
        raise ReevaluationError(
            f'D3 actor artifact checkpoint step does not match the locked D1/D234 reference: '
            f'expected {int(expected_checkpoint_step)}, got {checkpoint_step}: {path}'
        )
    sample_id = np.asarray(arrays['sample_id'], dtype=np.int64)
    expected_sample_id = np.asarray(expected_sample_id, dtype=np.int64)
    if not np.array_equal(sample_id[:int(max_samples)], expected_sample_id):
        raise ReevaluationError(f'D3 action sample ordering does not match fixed_batch sample_id: {path}')
    iteration_k = np.asarray(arrays['iteration_k'], dtype=np.int64)
    expected_iterations = np.arange(iteration_k.shape[0], dtype=np.int64)
    if not np.array_equal(iteration_k, expected_iterations) or int(train_k) >= len(iteration_k):
        raise ReevaluationError(f'D3 actor iteration axis is not canonical or lacks K={train_k}: {path}')
    clipped = np.asarray(arrays['clipped_action'], dtype=np.float64)
    normal_mode = np.asarray(arrays['normal_actor_mode_at_train_k'], dtype=np.float64)
    if clipped.ndim != 3 or clipped.shape[0] != len(sample_id):
        raise ReevaluationError(f'D3 clipped_action shape is invalid: {clipped.shape!r}')
    if normal_mode.shape != (len(sample_id), clipped.shape[-1]):
        raise ReevaluationError(
            f'D3 normal_actor_mode_at_train_k shape mismatch: {normal_mode.shape!r} vs {clipped.shape!r}'
        )
    action = clipped[:int(max_samples), int(train_k)].copy()
    parity_error = float(np.max(np.abs(action - np.clip(normal_mode[:int(max_samples)], -1.0, 1.0))))
    if parity_error > 1e-6:
        raise ReevaluationError(
            'D3 final clipped action does not match its saved normal actor mode at native K: '
            f'K{train_k}, max_abs_error={parity_error}'
        )
    if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-12):
        raise ReevaluationError(f'D3 final clipped action is non-finite or outside [-1,1]: {path}')
    return action, {
        'path': str(path),
        'iteration_k': int(train_k),
        'checkpoint_step': checkpoint_step,
        'sample_count': int(max_samples),
        'normal_mode_clipped_parity_max_abs_error': parity_error,
        'action_source': 'D3 actor_metrics.npz:clipped_action[:, K_train]',
    }


def load_reference_actions(contract, fixed_batch, *, max_samples):
    """Return a_data, a4, and a8 without forwarding either actor again."""

    sample_id = np.asarray(fixed_batch['sample_id'], dtype=np.int64)
    actions = {'a_data': np.asarray(fixed_batch['dataset_actions'], dtype=np.float64).copy()}
    metadata = {}
    for train_k in TRAIN_KS:
        actor_path = Path(contract['references'][train_k]['actor_metrics_path']).resolve()
        expected_trace_metadata_path = Path(contract['references'][train_k]['trace_metadata_path']).resolve()
        if actor_path.with_name('m18d_metadata.json') != expected_trace_metadata_path:
            raise ReevaluationError(
                f'D3 actor artifact is not the sibling of its locked D234 trace metadata: {actor_path}'
            )
        action, action_metadata = _load_d3_final_action(
            actor_path,
            train_k=train_k,
            expected_sample_id=sample_id,
            max_samples=max_samples,
            expected_checkpoint_step=contract['references'][train_k]['checkpoint_step'],
        )
        actions[f'a{train_k}'] = action
        metadata[f'a{train_k}'] = action_metadata
    shapes = {key: value.shape for key, value in actions.items()}
    if len(set(shapes.values())) != 1:
        raise ReevaluationError(f'D6 actions do not share one [N, action_dim] shape: {shapes!r}')
    if actions['a_data'].shape[0] != int(max_samples):
        raise ReevaluationError('Fixed batch action length does not match requested sample count')
    return actions, metadata


def _critic_members(agent, observations, goals, actions, *, label):
    values = np.asarray(agent.network.select('critic')(observations, goals, actions), dtype=np.float64)
    expected = (2, int(np.asarray(observations).shape[0]))
    if values.shape != expected:
        raise ReevaluationError(f'{label} critic output shape expected {expected!r}, got {values.shape!r}')
    if not np.all(np.isfinite(values)):
        raise ReevaluationError(f'{label} critic returned non-finite values')
    return values


def _strict_sign(delta, *, tolerance=TIE_TOLERANCE):
    values = np.asarray(delta, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ReevaluationError('Preference margin contains non-finite values')
    positive = values > float(tolerance)
    negative = values < -float(tolerance)
    tie = ~(positive | negative)
    return positive, negative, tie


def d6_per_sample(q4_members, q8_members, *, tolerance=TIE_TOLERANCE):
    """Compute D6 within-critic margins/rates from member outputs.

    ``q*_members`` maps ``data``, ``a4``, and ``a8`` to arrays shaped
    ``[2, N]``.  This pure helper deliberately never forms a cross-critic raw
    Q comparison such as ``Q4(a4) > Q8(a8)``.
    """

    all_members = {'Q4': q4_members, 'Q8': q8_members}
    result = {}
    qmin = {}
    for critic_name, mapping in all_members.items():
        if set(mapping) != {'data', 'a4', 'a8'}:
            raise ReevaluationError(f'{critic_name} must have exactly data/a4/a8 action values')
        for action_name, member_values in mapping.items():
            values = np.asarray(member_values, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != 2 or not np.all(np.isfinite(values)):
                raise ReevaluationError(f'{critic_name} {action_name} must have finite [2,N] values')
            if action_name == 'data':
                expected_n = values.shape[1]
            elif values.shape[1] != expected_n:
                raise ReevaluationError(f'{critic_name} member action sample counts differ')
            result[f'{critic_name}_member1_{action_name}'] = values[0]
            result[f'{critic_name}_member2_{action_name}'] = values[1]
            result[f'{critic_name}_disagreement_{action_name}'] = np.abs(values[0] - values[1])
            qmin[(critic_name, action_name)] = np.minimum(values[0], values[1])
            result[f'{critic_name}_{action_name}'] = qmin[(critic_name, action_name)]
    if qmin[('Q4', 'data')].shape != qmin[('Q8', 'data')].shape:
        raise ReevaluationError('Q4 and Q8 do not share a fixed-batch sample ordering')

    result['Delta_Q4_self'] = qmin[('Q4', 'a4')] - qmin[('Q4', 'a8')]
    result['Delta_Q8_self'] = qmin[('Q8', 'a8')] - qmin[('Q8', 'a4')]
    result['Delta_Q4_own_vs_data'] = qmin[('Q4', 'a4')] - qmin[('Q4', 'data')]
    result['Delta_Q4_other_vs_data'] = qmin[('Q4', 'a8')] - qmin[('Q4', 'data')]
    result['Delta_Q8_own_vs_data'] = qmin[('Q8', 'a8')] - qmin[('Q8', 'data')]
    result['Delta_Q8_other_vs_data'] = qmin[('Q8', 'a4')] - qmin[('Q8', 'data')]
    q4_positive, _, q4_tie = _strict_sign(result['Delta_Q4_self'], tolerance=tolerance)
    q8_positive, _, q8_tie = _strict_sign(result['Delta_Q8_self'], tolerance=tolerance)
    result['self_preference_4'] = q4_positive.astype(np.int8)
    result['self_preference_8'] = q8_positive.astype(np.int8)
    result['tie_Q4_self'] = q4_tie.astype(np.int8)
    result['tie_Q8_self'] = q8_tie.astype(np.int8)
    result['joint_self_preference'] = (q4_positive & q8_positive).astype(np.int8)
    result['normalized_Delta_Q4_self'] = result['Delta_Q4_self'] / (
        (np.abs(qmin[('Q4', 'data')]) + np.abs(qmin[('Q4', 'a4')]) + np.abs(qmin[('Q4', 'a8')])) / 3.0
        + EPSILON
    )
    result['normalized_Delta_Q8_self'] = result['Delta_Q8_self'] / (
        (np.abs(qmin[('Q8', 'data')]) + np.abs(qmin[('Q8', 'a4')]) + np.abs(qmin[('Q8', 'a8')])) / 3.0
        + EPSILON
    )
    if not all(np.all(np.isfinite(np.asarray(values))) for values in result.values()):
        raise ReevaluationError('D6 per-sample output contains non-finite values')
    return result


def _describe(values, *, tolerance=None):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            'count': 0, 'mean': None, 'std': None, 'median': None,
            'p10': None, 'p90': None, 'positive_fraction': None,
            'negative_fraction': None, 'tie_fraction': None,
        }
    result = {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
        'positive_fraction': None,
        'negative_fraction': None,
        'tie_fraction': None,
    }
    if tolerance is not None:
        positive, negative, tie = _strict_sign(values, tolerance=tolerance)
        result.update({
            'positive_fraction': float(np.mean(positive)),
            'negative_fraction': float(np.mean(negative)),
            'tie_fraction': float(np.mean(tie)),
        })
    return result


def d6_summary_rows(per_sample, *, tolerance=TIE_TOLERANCE):
    """Aggregate only within-critic descriptive quantities and control margins."""

    margin_names = (
        'Delta_Q4_self', 'Delta_Q8_self',
        'Delta_Q4_own_vs_data', 'Delta_Q4_other_vs_data',
        'Delta_Q8_own_vs_data', 'Delta_Q8_other_vs_data',
        'normalized_Delta_Q4_self', 'normalized_Delta_Q8_self',
    )
    disagreement_names = (
        'Q4_disagreement_data', 'Q4_disagreement_a4', 'Q4_disagreement_a8',
        'Q8_disagreement_data', 'Q8_disagreement_a4', 'Q8_disagreement_a8',
    )
    rows = []
    for name in margin_names:
        rows.append({'metric_family': 'within_critic_margin', 'metric': name, **_describe(per_sample[name], tolerance=tolerance)})
    for name in disagreement_names:
        rows.append({'metric_family': 'critic_member_disagreement', 'metric': name, **_describe(per_sample[name])})
    for name in ('self_preference_4', 'self_preference_8', 'joint_self_preference', 'tie_Q4_self', 'tie_Q8_self'):
        values = np.asarray(per_sample[name], dtype=np.float64)
        rows.append({
            'metric_family': 'rate',
            'metric': name,
            'count': int(len(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'p10': float(np.percentile(values, 10)),
            'p90': float(np.percentile(values, 90)),
            'positive_fraction': None,
            'negative_fraction': None,
            'tie_fraction': None,
        })
    return rows


def plan(
    reference_diagnostics_root,
    source_run_root,
    output_root,
    *,
    train_ks=TRAIN_KS,
    reference_batch_size=reference.DEFAULT_REFERENCE_BATCH_SIZE,
    reference_diagnostic_seed=reference.DEFAULT_REFERENCE_DIAGNOSTIC_SEED,
    max_samples=None,
):
    contract = reference.load_reference_contract(
        reference_diagnostics_root,
        train_ks=train_ks,
        reference_batch_size=reference_batch_size,
        reference_diagnostic_seed=reference_diagnostic_seed,
    )
    batch, count = reference.load_fixed_batch_from_contract(contract, max_samples=max_samples)
    actions, action_metadata = load_reference_actions(contract, batch, max_samples=count)
    provenance = {
        train_k: reference.locked_provenance(contract, source_run_root, train_k)
        for train_k in train_ks
    }
    output_dir = _output_dir(
        output_root,
        reference_batch_size=reference_batch_size,
        reference_diagnostic_seed=reference_diagnostic_seed,
        max_samples=count,
    )
    return {
        'contract': contract,
        'batch': batch,
        'sample_count': count,
        'actions': actions,
        'action_metadata': action_metadata,
        'provenance': provenance,
        'output_dir': output_dir,
        'smoke_only': bool(count != int(reference_batch_size)),
    }


def execute(plan_data, *, diagnostic_code_commit):
    """Restore native-depth critics and write one fresh D6 diagnostic artifact."""

    if not diagnostic_code_commit:
        raise ReevaluationError('M18-D6 execute requires --diagnostic-code-commit from the user-reviewed commit')
    output_dir = Path(plan_data['output_dir'])
    if output_dir.exists():
        raise FileExistsError(f'M18-D6 output exists; refusing overwrite: {output_dir}')
    provenance = plan_data['provenance']
    source_hash_before = {
        train_k: reference.stable_checkpoint_sha256(provenance[train_k]['checkpoint_path'])
        for train_k in TRAIN_KS
    }
    for train_k in TRAIN_KS:
        if source_hash_before[train_k] != provenance[train_k]['checkpoint_sha256']:
            raise ReevaluationError(f'K{train_k} locked source changed after planning; refusing D6 execution')
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / 'metadata.json'
    metadata = {
        'status': 'running',
        'diagnostic_id': DIAGNOSTIC_ID,
        'diagnostic_family': 'cross_actor_critic_preference',
        'diagnostic_code_commit': str(diagnostic_code_commit),
        'source_study_id': STUDY_ID,
        'environment': ENVIRONMENT,
        'checkpoint_selector': reference.LOCKED_REFERENCE_SELECTOR,
        'reference_m18d_root': plan_data['contract']['reference_m18d_root'],
        'reference_fixed_batch_path': plan_data['contract']['fixed_batch_path'],
        'reference_fixed_batch_metadata_path': plan_data['contract']['fixed_batch_metadata_path'],
        'fixed_batch_fingerprint_sha256': plan_data['contract']['fixed_batch_fingerprint_sha256'],
        'fixed_batch_goal_semantics': 'actor_goals',
        'fixed_batch_full_size': int(plan_data['contract']['reference_batch_size']),
        'evaluated_sample_count': int(plan_data['sample_count']),
        'smoke_only': bool(plan_data['smoke_only']),
        'tie_tolerance': float(TIE_TOLERANCE),
        'epsilon': float(EPSILON),
        'action_sources': plan_data['action_metadata'],
        'q_execution_depth_by_critic': {'Q4': 4, 'Q8': 8},
        'preference_scope': 'within_critic_only; cross-critic raw Q scales are not compared',
        'evaluation_only': True,
        'finetuning': False,
        'optimizer_updates': 0,
        'source_checkpoints': {},
    }
    for train_k in TRAIN_KS:
        item = provenance[train_k]
        metadata['source_checkpoints'][f'K{train_k}'] = {
            'K_train': int(train_k),
            'source_config_id': item['source_config_id'],
            'source_run_dir': item['source_run_dir'],
            'source_checkpoint_role': item['resolved_checkpoint_role'],
            'source_checkpoint_step': item['checkpoint_step'],
            'source_checkpoint_path': item['checkpoint_path'],
            'source_checkpoint_sha256': item['checkpoint_sha256'],
            'source_checkpoint_hash_before': source_hash_before[train_k],
            'reference_d1_summary_path': item['reference_d1_summary_path'],
            'reference_trace_metadata_path': item['reference_trace_metadata_path'],
        }
    _write_json(metadata_path, metadata)
    environments = []
    try:
        q_members = {}
        network_fingerprints_before = {}
        network_steps_before = {}
        network_fingerprints_after = {}
        network_steps_after = {}
        for train_k in TRAIN_KS:
            agent, env, _, config = trace._build_restored_agent(provenance[train_k])
            environments.append(env)
            source_k = int(config['compute']['critic']['topology_kwargs']['iterations'])
            if source_k != train_k:
                raise ReevaluationError(f'K{train_k} critic did not restore at native K: got {source_k}')
            network_fingerprints_before[train_k] = tree_fingerprint(agent.network.params)
            network_steps_before[train_k] = int(np.asarray(agent.network.step))
            q_members[f'Q{train_k}'] = {
                action_name: _critic_members(
                    agent,
                    plan_data['batch']['observations'],
                    plan_data['batch']['actor_goals'],
                    plan_data['actions'][{'data': 'a_data', 'a4': 'a4', 'a8': 'a8'}[action_name]],
                    label=f'Q{train_k}({action_name})',
                )
                for action_name in ('data', 'a4', 'a8')
            }
            network_fingerprints_after[train_k] = tree_fingerprint(agent.network.params)
            network_steps_after[train_k] = int(np.asarray(agent.network.step))
            if network_fingerprints_before[train_k] != network_fingerprints_after[train_k]:
                raise ReevaluationError(f'K{train_k} online parameters changed during inference-only D6')
            if network_steps_before[train_k] != network_steps_after[train_k]:
                raise ReevaluationError(f'K{train_k} optimizer/network step changed during inference-only D6')
        per_sample = d6_per_sample(q_members['Q4'], q_members['Q8'], tolerance=TIE_TOLERANCE)
        per_sample = {
            'sample_id': np.asarray(plan_data['batch']['sample_id'], dtype=np.int64),
            **per_sample,
        }
        summary_rows = d6_summary_rows(per_sample, tolerance=TIE_TOLERANCE)
        source_hash_after = {
            train_k: reference.stable_checkpoint_sha256(provenance[train_k]['checkpoint_path'])
            for train_k in TRAIN_KS
        }
        for train_k in TRAIN_KS:
            if source_hash_after[train_k] != source_hash_before[train_k]:
                raise ReevaluationError(f'K{train_k} source checkpoint SHA256 changed during D6')
        np.savez_compressed(output_dir / 'm18d_d6_per_sample.npz', **per_sample)
        fields = (
            'metric_family', 'metric', 'count', 'mean', 'std', 'median', 'p10', 'p90',
            'positive_fraction', 'negative_fraction', 'tie_fraction',
        )
        _write_csv(output_dir / 'm18d_d6_summary.csv', summary_rows, fields)
        summary = {
            'status': 'completed',
            'diagnostic_id': DIAGNOSTIC_ID,
            'diagnostic_family': 'cross_actor_critic_preference',
            'sample_count': int(plan_data['sample_count']),
            'smoke_only': bool(plan_data['smoke_only']),
            'tie_tolerance': float(TIE_TOLERANCE),
            'epsilon': float(EPSILON),
            'fixed_batch_fingerprint_sha256': plan_data['contract']['fixed_batch_fingerprint_sha256'],
            'action_source_contract': plan_data['action_metadata'],
            'q_execution_depth_by_critic': {'Q4': 4, 'Q8': 8},
            'summary_rows': summary_rows,
            'evaluation_only': True,
            'finetuning': False,
            'optimizer_updates': 0,
            'source_checkpoint_hash_before': source_hash_before,
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_immutable': True,
            'online_parameter_fingerprint_before': network_fingerprints_before,
            'online_parameter_fingerprint_after': network_fingerprints_after,
            'online_network_step_before': network_steps_before,
            'online_network_step_after': network_steps_after,
            'raw_q_cross_critic_comparisons_interpreted': False,
        }
        _write_json(output_dir / 'm18d_d6_summary.json', summary)
        metadata.update({
            'status': 'completed',
            'sample_count': int(plan_data['sample_count']),
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_immutable': True,
            'online_parameter_fingerprint_before': network_fingerprints_before,
            'online_parameter_fingerprint_after': network_fingerprints_after,
            'online_network_step_before': network_steps_before,
            'online_network_step_after': network_steps_after,
            'artifacts': [
                str(output_dir / 'm18d_d6_per_sample.npz'),
                str(output_dir / 'm18d_d6_summary.csv'),
                str(output_dir / 'm18d_d6_summary.json'),
            ],
        })
        _write_json(metadata_path, metadata)
        return summary
    except BaseException as error:
        source_hash_after = {}
        for train_k in TRAIN_KS:
            try:
                source_hash_after[train_k] = reference.stable_checkpoint_sha256(provenance[train_k]['checkpoint_path'])
            except BaseException as hash_error:
                source_hash_after[train_k] = f'ERROR {type(hash_error).__name__}: {hash_error}'
        metadata.update({
            'status': 'failed',
            'failure_reason': f'{type(error).__name__}: {error}',
            'source_checkpoint_hash_after': source_hash_after,
            'source_checkpoint_immutable': all(
                source_hash_after.get(k) == source_hash_before.get(k) for k in TRAIN_KS
            ),
        })
        _write_json(metadata_path, metadata)
        raise
    finally:
        for env in environments:
            env.close()


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-diagnostics-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--source-run-root', default=DEFAULT_SOURCE_RUN_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--train-ks', default='4,8')
    parser.add_argument('--checkpoint', default=reference.LOCKED_REFERENCE_SELECTOR)
    parser.add_argument('--reference-batch-size', type=int, default=reference.DEFAULT_REFERENCE_BATCH_SIZE)
    parser.add_argument('--reference-diagnostic-seed', type=int, default=reference.DEFAULT_REFERENCE_DIAGNOSTIC_SEED)
    parser.add_argument('--max-samples', type=int, default=None, help='Prefix of the immutable batch; <N is smoke-only.')
    parser.add_argument('--diagnostic-code-commit', default=None, help='User-supplied reviewed diagnostic code commit.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error('Exactly one of --dry-run or --execute is required')
    if args.reference_batch_size <= 0:
        parser.error('--reference-batch-size must be positive')
    return args


def main(argv=None):
    args = _args(argv)
    try:
        if args.checkpoint != reference.LOCKED_REFERENCE_SELECTOR:
            raise ReevaluationError(
                f'M18-D6 only accepts --checkpoint {reference.LOCKED_REFERENCE_SELECTOR!r}; '
                'current semantic-best selection is prohibited'
            )
        train_ks = _parse_train_ks(args.train_ks)
        plan_data = plan(
            args.reference_diagnostics_root,
            args.source_run_root,
            args.output_root,
            train_ks=train_ks,
            reference_batch_size=args.reference_batch_size,
            reference_diagnostic_seed=args.reference_diagnostic_seed,
            max_samples=args.max_samples,
        )
        output_dir = Path(plan_data['output_dir'])
        print(
            f'M18-D6 locked plan: samples={plan_data["sample_count"]} '
            f'smoke_only={plan_data["smoke_only"]} batch_fingerprint='
            f'{plan_data["contract"]["fixed_batch_fingerprint_sha256"]} output={output_dir}'
        )
        for train_k in TRAIN_KS:
            item = plan_data['provenance'][train_k]
            print(
                f'[LOCKED] K{train_k} role={item["resolved_checkpoint_role"]} '
                f'step={item["checkpoint_step"]} sha256={item["checkpoint_sha256"]} '
                f'path={item["checkpoint_path"]}'
            )
        if output_dir.exists():
            raise FileExistsError(f'M18-D6 output exists; refusing overwrite: {output_dir}')
        if args.dry_run:
            return 0
        if not args.diagnostic_code_commit:
            raise ReevaluationError('--execute requires --diagnostic-code-commit from the user-reviewed commit')
        summary = execute(plan_data, diagnostic_code_commit=args.diagnostic_code_commit)
        print(
            f'M18-D6 execute completed: samples={summary["sample_count"]} '
            f'optimizer_updates=0 output={output_dir}'
        )
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError, ReevaluationError) as error:
        print(f'M18-D6: FAIL: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
