"""Print M15 default-candidate flat versus structured Puzzle accounting.

This is a static architecture audit only.  It does not load a dataset or
launch training.
"""

import json

import jax
import jax.numpy as jnp

from impls.computation.accounting import structured_body_accounting
from impls.computation.structured import PuzzleStructuredBody


WIDTH = 512
TOKEN_DIM = 128
ROBOT_HIDDEN = 128
TOKEN_HIDDEN = 64
CHANNEL_HIDDEN = 256
NUM_BLOCKS = 1
ACTION_DIM = 5


def dense_params(in_dim, out_dim):
    return in_dim * out_dim + out_dim


def dense_macs(in_dim, out_dim):
    return in_dim * out_dim


def flat_body(observation_dim, layer_norm=False):
    values = [
        (2 * observation_dim, WIDTH),
        (WIDTH, WIDTH),
        (WIDTH, WIDTH),
    ]
    return {
        'params': sum(dense_params(*pair) for pair in values) + (3 * 2 * WIDTH if layer_norm else 0),
        'dense_macs': sum(dense_macs(*pair) for pair in values),
        'sequential_depth': len(values),
    }


def structured_body(num_buttons, action_context=False, ensemble=1, layer_norm=False):
    observation_dim = 19 + 4 * num_buttons
    input_dim = 2 * observation_dim + (ACTION_DIM if action_context else 0)
    kwargs = {
        'num_buttons': num_buttons,
        'robot_dim': 19,
        'button_feature_dim': 4,
        'token_dim': TOKEN_DIM,
        'robot_hidden_dim': ROBOT_HIDDEN,
        'token_mlp_hidden_dim': TOKEN_HIDDEN,
        'channel_mlp_hidden_dim': CHANNEL_HIDDEN,
        'num_mixer_blocks': NUM_BLOCKS,
        'index_embedding': True,
        'readout': 'mean',
        'tm_mode': 'none',
    }
    body = PuzzleStructuredBody(
        output_dim=WIDTH,
        input_semantics='goal_pair',
        action_semantics='robot_context' if action_context else 'none',
        layer_norm=layer_norm,
        **kwargs,
    )
    variables = body.init(jax.random.PRNGKey(num_buttons + int(action_context)), jnp.ones((1, input_dim)))
    report = structured_body_accounting(variables['params'], kwargs)
    if ensemble != 1:
        for key in (
            'button_projection_params', 'index_embedding_params',
            'robot_projection_params', 'mixer_params', 'fusion_params',
            'structured_body_params', 'structured_body_dense_macs',
            'structured_dense_macs', 'mixer_dense_macs', 'tm_dense_macs',
            'token_projection_dense_macs', 'robot_projection_dense_macs',
            'fusion_dense_macs',
        ):
            report[key] *= ensemble
        report['unique_dense_layers'] *= ensemble
        report['executed_dense_layers'] *= ensemble
    return report


def main():
    report = {}
    for _, _, num_buttons, _ in ((3, 3, 9, 55), (4, 4, 16, 83), (4, 5, 20, 99), (4, 6, 24, 115)):
        observation_dim = 19 + 4 * num_buttons
        report[f'Puzzle-{num_buttons}'] = {
            'observation_dim': observation_dim,
            'gciql_actor': {
                'canonical_flat': flat_body(observation_dim),
                'structured_mixer': structured_body(num_buttons),
            },
            'gciql_value': {
                'canonical_flat': flat_body(observation_dim, layer_norm=True),
                'structured_mixer': structured_body(num_buttons, layer_norm=True),
            },
            'gciql_critic_ensemble_2': {
                'canonical_flat': {
                    key: value * 2 if key != 'sequential_depth' else value
                    for key, value in flat_body(observation_dim, layer_norm=True).items()
                },
                'structured_mixer': structured_body(
                    num_buttons, action_context=True, ensemble=2, layer_norm=True,
                ),
            },
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
