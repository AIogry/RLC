"""Exact CRL policy/critic metrics for fixed-checkpoint diagnostics."""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


def _scalar(value):
    return float(np.asarray(value).mean())


def _info_scalars(info):
    result = {}
    for key, value in info.items():
        normalized = str(key).replace('/', '_')
        result[normalized] = _scalar(value)
    return result


def _policy_tensors(actor, observations, goals, actions=None):
    agent = actor.agent if hasattr(actor, 'agent') else actor
    params = agent.network.params
    observations = jnp.asarray(observations)
    goals = jnp.asarray(goals)
    dist = agent.network.select('actor')(observations, goals, params=params)
    raw = dist.mode()
    clipped = jnp.clip(raw, -1, 1)
    frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
    q1, q2 = agent.network.select('critic')(
        observations, goals, clipped, params=frozen
    )
    result = {
        'action_mean': raw,
        'action_clipped': clipped,
        'action_norm': jnp.linalg.norm(clipped, axis=-1),
        'clip_indicator': jnp.any(jnp.abs(raw) > 1.0 + 1e-7, axis=-1).astype(jnp.float32),
        'raw_action_norm': jnp.linalg.norm(raw, axis=-1),
        'q1': q1,
        'q2': q2,
        'q_min': jnp.minimum(q1, q2),
    }
    if actions is not None:
        actions = jnp.asarray(actions)
        data_q1, data_q2 = agent.network.select('critic')(
            observations, goals, actions, params=frozen
        )
        result.update({
            'q1_data': data_q1,
            'q2_data': data_q2,
            'q_data': jnp.minimum(data_q1, data_q2),
            'dataset_action': actions,
            'log_prob': dist.log_prob(actions),
            'mse': jnp.mean((raw - actions) ** 2, axis=-1),
        })
    return result


def evaluate_training_batch(actor, batch):
    agent = actor.agent if hasattr(actor, 'agent') else actor
    canonical = {
        key: jnp.asarray(batch[key])
        for key in ('observations', 'actions', 'actor_goals', 'value_goals')
    }
    loss, info = agent.policy_extraction_loss(
        canonical, agent.network.params, rng=agent.rng
    )
    tensors = _policy_tensors(
        actor, canonical['observations'], canonical['actor_goals'], canonical['actions']
    )
    result = _info_scalars(info)
    result['actor_loss_return'] = _scalar(loss)
    result.update({
        'q1_mean': _scalar(tensors['q1'].mean()),
        'q2_mean': _scalar(tensors['q2'].mean()),
        'q_min_mean': _scalar(tensors['q_min'].mean()),
        'q_data_mean': _scalar(tensors['q_data'].mean()),
        'q_disagreement_mean': _scalar(jnp.abs(tensors['q1'] - tensors['q2']).mean()),
        'q_data_disagreement_mean': _scalar(jnp.abs(tensors['q1_data'] - tensors['q2_data']).mean()),
        'action_norm_mean': _scalar(tensors['action_norm'].mean()),
        'raw_action_norm_mean': _scalar(tensors['raw_action_norm'].mean()),
        'clipping_fraction': _scalar(tensors['clip_indicator'].mean()),
        'behavior_mse_mean': _scalar(tensors['mse'].mean()),
    })
    return result


def evaluate_goal_batch(actor, batch):
    tensors = _policy_tensors(
        actor, batch['observations'], batch['eval_goals'], batch.get('actions')
    )
    result = {
        'q1_mean': _scalar(tensors['q1'].mean()),
        'q2_mean': _scalar(tensors['q2'].mean()),
        'q_min_mean': _scalar(tensors['q_min'].mean()),
        'q_disagreement_mean': _scalar(jnp.abs(tensors['q1'] - tensors['q2']).mean()),
        'action_norm_mean': _scalar(tensors['action_norm'].mean()),
        'raw_action_norm_mean': _scalar(tensors['raw_action_norm'].mean()),
        'clipping_fraction': _scalar(tensors['clip_indicator'].mean()),
    }
    if 'actions' in batch:
        result['behavior_mse_mean'] = _scalar(tensors['mse'].mean())
        result['q_data_mean'] = _scalar(tensors['q_data'].mean())
    return result


def _chunk(array_dict, start, stop):
    return {
        key: value[start:stop]
        for key, value in array_dict.items()
        if hasattr(value, 'shape') and value.ndim > 0 and value.shape[0] >= stop
    }


def evaluate_bank(actor, bank, *, batch_size=1024):
    arrays = bank.arrays
    kind = bank.manifest['bank_type']
    if kind == 'B_T':
        rows = []
        for batch_index in np.unique(arrays['batch_index']):
            mask = arrays['batch_index'] == batch_index
            row = evaluate_training_batch(actor, {key: value[mask] for key, value in arrays.items()})
            row['batch_index'] = int(batch_index)
            rows.append(row)
        return rows
    if kind not in {'B_DE', 'B_R'}:
        raise ValueError(f'Unsupported bank type: {kind}')
    rows = []
    for start in range(0, len(arrays['observations']), int(batch_size)):
        stop = min(start + int(batch_size), len(arrays['observations']))
        row = evaluate_goal_batch(actor, _chunk(arrays, start, stop))
        row.update({'sample_start': start, 'sample_stop': stop})
        rows.append(row)
    return rows


def evaluate_bank_samples(actor, bank, *, batch_size=1024):
    arrays = bank.arrays
    kind = bank.manifest['bank_type']
    chunks = []
    if kind == 'B_T':
        for batch_index in np.unique(arrays['batch_index']):
            mask = arrays['batch_index'] == batch_index
            chunks.append(_policy_tensors(
                actor, arrays['observations'][mask], arrays['actor_goals'][mask], arrays['actions'][mask]
            ))
    elif kind in {'B_DE', 'B_R'}:
        for start in range(0, len(arrays['observations']), int(batch_size)):
            stop = min(start + int(batch_size), len(arrays['observations']))
            chunks.append(_policy_tensors(
                actor, arrays['observations'][start:stop], arrays['eval_goals'][start:stop],
                arrays['actions'][start:stop] if 'actions' in arrays else None,
            ))
    else:
        raise ValueError(f'Unsupported bank type: {kind}')
    keys = set().union(*(chunk.keys() for chunk in chunks))
    return {
        key: np.concatenate([np.asarray(chunk[key]) for chunk in chunks], axis=0)
        for key in keys if all(key in chunk for chunk in chunks)
    }


def aggregate_rows(rows):
    values = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key not in {'batch_index', 'sample_start', 'sample_stop'}:
                values[key].append(float(value))
    return {key: float(np.mean(value)) for key, value in values.items()} | {'row_count': len(rows)}


def pairwise_contrasts(actor_results, actor_order):
    result = []
    for i, left_name in enumerate(actor_order):
        for right_name in actor_order[i + 1:]:
            left, right = actor_results[left_name], actor_results[right_name]
            left_q, right_q = left['q_min'], right['q_min']
            delta_action = np.linalg.norm(left['action_clipped'] - right['action_clipped'], axis=-1)
            delta_q = right_q - left_q
            scale = max(float(np.mean(np.abs(left_q))), 1e-6)
            result.append({
                'left_actor': left_name,
                'right_actor': right_name,
                'action_l2_mean': float(delta_action.mean()),
                'action_l2_squared_mean': float(np.square(delta_action).mean()),
                'q_delta_right_minus_left': float(delta_q.mean()),
                'q_delta_normalized_by_left_abs_mean': float(delta_q.mean() / scale),
                'q_right_win_rate': float(np.mean(right_q > left_q)),
                'q_tie_rate': float(np.mean(right_q == left_q)),
            })
    return result

