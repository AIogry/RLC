"""Read-only capture of the inputs actually consumed by GC network bodies."""

from __future__ import annotations

import numpy as np

from ..networks.common import GCActor, GCValue
from ..networks.goal_conditioning import goal_conditioning_runtime_metadata
from ..representation.interfaces import StructuredNetworkInput
from ..utils.checkpointing import tree_fingerprint


def capture_goal_conditioning_inputs(agent, observations, raw_goals, actions):
    """Execute real forward paths and capture their preparation return values.

    This uses Flax intermediate capture, not an alternate oracle formula or
    a preparer that the forward implementation might accidentally ignore.
    The V/Q/target calls deliberately all receive the external raw goal.
    """
    variables = {'params': agent.network.params, **agent.network.model_state}
    captured = {}
    for name in ('actor', 'value', 'critic', 'target_critic'):
        args = (observations, raw_goals)
        if name in ('critic', 'target_critic'):
            args += (actions,)
        _, state = agent.network.apply_fn(
            variables, *args, name=name,
            capture_intermediates=lambda module, method: (
                isinstance(module, (GCActor, GCValue)) and method == 'prepare_inputs'
            ),
            mutable=['intermediates'],
        )
        captured[name] = state['intermediates'][f'modules_{name}']['prepare_inputs'][0]
    return captured


def audit_goal_conditioning(agent, observations, raw_goals, actions):
    inputs = capture_goal_conditioning_inputs(agent, observations, raw_goals, actions)
    metadata = goal_conditioning_runtime_metadata(
        agent.config.get('goal_conditioning'), compute_slots=agent.config.get('compute'),
        dataset_class=agent.config.get('dataset_class'),
    )
    report = {'source': 'production_forward_intermediate_capture', 'metadata': metadata, 'modules': {}}
    obs_dim = observations.shape[-1]
    for name, value in inputs.items():
        typed = isinstance(value, StructuredNetworkInput)
        flat = np.asarray(value.flat_inputs if typed else value)
        np.testing.assert_array_equal(flat[..., :obs_dim], observations)
        if name in ('critic', 'target_critic'):
            np.testing.assert_array_equal(flat[..., 2 * obs_dim:], actions)
        if not np.all(np.isfinite(flat)):
            raise ValueError(f'Non-finite actual input in {name}')
        aux = np.asarray(value.token_aux) if typed else None
        if aux is not None and not np.all(np.isfinite(aux)):
            raise ValueError(f'Non-finite actual auxiliary input in {name}')
        report['modules'][name] = {
            'role': 'actor' if name == 'actor' else 'value_side',
            'flat_shape': list(flat.shape), 'token_aux_shape': None if aux is None else list(aux.shape),
            'flat_fingerprint': tree_fingerprint(flat),
            'token_aux_fingerprint': None if aux is None else tree_fingerprint(aux),
            'current_observation_preserved': True,
            'action_tail_preserved': name in ('critic', 'target_critic'),
        }
    return report
