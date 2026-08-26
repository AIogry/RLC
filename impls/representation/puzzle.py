"""Strict parsing helpers for standard OGBench Puzzle state observations."""

import jax.numpy as jnp


def parse_puzzle_observation(
    x,
    *,
    num_buttons: int,
    robot_dim: int = 19,
    button_feature_dim: int = 4,
):
    """Split a flat Puzzle observation into robot and button components.

    The standard state observation is exactly
    ``[robot_state (19), button_0 (4), ..., button_N-1 (4)]``.  This helper
    intentionally accepts no privileged ``button_states`` field and performs
    no frame stacking, image handling, or shape guessing.
    """

    if isinstance(num_buttons, bool) or not isinstance(num_buttons, int) or num_buttons <= 0:
        raise ValueError(f'num_buttons must be a positive integer, got {num_buttons!r}')
    if isinstance(robot_dim, bool) or not isinstance(robot_dim, int) or robot_dim <= 0:
        raise ValueError(f'robot_dim must be a positive integer, got {robot_dim!r}')
    if isinstance(button_feature_dim, bool) or not isinstance(button_feature_dim, int) or button_feature_dim <= 0:
        raise ValueError(
            'button_feature_dim must be a positive integer, '
            f'got {button_feature_dim!r}'
        )

    x = jnp.asarray(x)
    if x.ndim < 1:
        raise ValueError(f'Puzzle observation must have a final feature axis, got shape {x.shape}')
    expected_dim = robot_dim + num_buttons * button_feature_dim
    if x.shape[-1] != expected_dim:
        raise ValueError(
            'Malformed Puzzle observation: expected final dimension '
            f'{expected_dim} (= {robot_dim} + {num_buttons} * {button_feature_dim}), '
            f'got {x.shape[-1]} for shape {x.shape}'
        )
    robot = x[..., :robot_dim]
    buttons = x[..., robot_dim:]
    buttons = buttons.reshape(*x.shape[:-1], num_buttons, button_feature_dim)
    return robot, buttons
