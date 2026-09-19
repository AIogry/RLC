"""Tiny CPU fixtures only, not a scientific Study or training protocol."""

import copy

import numpy as np

from tests.integration.test_gciql_puzzle_goal_conditioning import _batch, _config, _observations


def role(coordinate='residual', transform_id='identity', distance_feature='zero'):
    return dict(coordinate=coordinate, transform_id=transform_id, distance_feature=distance_feature)


def goal_config(actor=None, value_side=None, *, rows=3, cols=3):
    n = rows * cols
    matrix = np.eye(n, dtype=np.uint8)
    if n > 1:
        matrix[0, 1] = 1
    if n > 2:
        matrix[1, 2] = 1
    return {
        'schema_version': 2, 'domain': 'puzzle', 'input_schema': 'token_aux_v1',
        'rows': rows, 'cols': cols, 'num_buttons': n, 'robot_dim': 19, 'button_feature_dim': 4,
        'robot_goal_policy': 'zero', 'button_goal_transient_policy': 'zero',
        'roles': {'actor': copy.deepcopy(actor or role()), 'value_side': copy.deepcopy(value_side or role())},
        'transforms': {
            'P': {'kind': 'permutation', 'permutation': list(range(1, n)) + [0]},
            'B': {'kind': 'gf2_linear', 'matrix': matrix.tolist()},
        },
    }


def agent_config(actor=None, value_side=None):
    config = _config('residual')
    config.goal_conditioning = goal_config(actor, value_side)
    return config


batch = _batch
observations = _observations
