"""Strict Puzzle parsing and learned structured-representation adapters."""

from numbers import Integral

import flax.linen as nn
import jax.numpy as jnp

from .interfaces import StructuredRepresentation


def _default_init(scale=1.0):
    """The OGBench Dense initializer, kept local to avoid package cycles."""

    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


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


class PuzzleTokenAdapter(nn.Module):
    """Map a standard Puzzle input to tokens plus global context.

    This adapter owns only domain-specific representation parameters:
    button projection, optional absolute index embedding, and the robot/global
    context projection.  It deliberately does *not* execute Mixer blocks,
    recurrent iterations, token readout, or final slot fusion.
    """

    num_buttons: int
    robot_dim: int = 19
    button_feature_dim: int = 4
    token_dim: int = 128
    robot_hidden_dim: int = 128
    index_embedding: bool = True
    input_semantics: str = 'goal_pair'
    action_semantics: str = 'none'
    layer_norm: bool = False

    def setup(self):
        integer_fields = {
            'num_buttons': self.num_buttons,
            'robot_dim': self.robot_dim,
            'button_feature_dim': self.button_feature_dim,
            'token_dim': self.token_dim,
            'robot_hidden_dim': self.robot_hidden_dim,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        if self.input_semantics not in ('goal_pair', 'single_observation', 'latent_vector'):
            raise ValueError(f'Unsupported Puzzle input_semantics: {self.input_semantics!r}')
        if self.input_semantics == 'latent_vector':
            raise ValueError('Puzzle token adapter cannot consume latent_vector input')
        if self.action_semantics not in ('none', 'robot_context', 'latent_dynamics'):
            raise ValueError(f'Unsupported Puzzle action_semantics: {self.action_semantics!r}')
        if self.action_semantics == 'latent_dynamics':
            raise ValueError('Puzzle token adapter cannot consume latent_dynamics input')
        if self.input_semantics == 'single_observation' and self.action_semantics != 'none':
            raise ValueError('single_observation Puzzle representation cannot include action context')

        self.button_projection = nn.Dense(self.token_dim, kernel_init=_default_init())
        if self.index_embedding:
            self.index_embedding_param = self.param(
                'index_embedding',
                nn.initializers.normal(stddev=0.02),
                (self.num_buttons, self.token_dim),
            )
        else:
            self.index_embedding_param = None
        # Keep the legacy semantic names.  The ownership moves from the
        # monolithic body to this adapter, while parameter transplant remains
        # explicit and straightforward.
        self.robot_projection = nn.Dense(self.robot_hidden_dim, kernel_init=_default_init())
        self.robot_layer_norm = nn.LayerNorm() if self.layer_norm else None

    def _split_input(self, x):
        """Apply the frozen slot semantics without inferring trailing fields."""

        obs_dim = self.robot_dim + self.num_buttons * self.button_feature_dim
        if self.input_semantics == 'single_observation':
            if x.shape[-1] != obs_dim:
                raise ValueError(
                    f'Puzzle single_observation adapter expected final dimension {obs_dim}, '
                    f'got {x.shape[-1]}'
                )
            return parse_puzzle_observation(
                x,
                num_buttons=self.num_buttons,
                robot_dim=self.robot_dim,
                button_feature_dim=self.button_feature_dim,
            )

        pair_dim = 2 * obs_dim
        if x.shape[-1] < pair_dim:
            raise ValueError(
                f'Puzzle goal_pair adapter expected at least final dimension {pair_dim}, '
                f'got {x.shape[-1]}'
            )
        state = x[..., :obs_dim]
        goal = x[..., obs_dim:pair_dim]
        robot_state, buttons_state = parse_puzzle_observation(
            state,
            num_buttons=self.num_buttons,
            robot_dim=self.robot_dim,
            button_feature_dim=self.button_feature_dim,
        )
        robot_goal, buttons_goal = parse_puzzle_observation(
            goal,
            num_buttons=self.num_buttons,
            robot_dim=self.robot_dim,
            button_feature_dim=self.button_feature_dim,
        )
        robot = jnp.concatenate([robot_state, robot_goal], axis=-1)
        buttons = jnp.concatenate([buttons_state, buttons_goal], axis=-1)
        if self.action_semantics == 'robot_context':
            action = x[..., pair_dim:]
            if action.shape[-1] <= 0:
                raise ValueError('Puzzle critic robot_context requires a non-empty action vector')
            robot = jnp.concatenate([robot, action], axis=-1)
        elif x.shape[-1] != pair_dim:
            raise ValueError(
                'Puzzle goal_pair adapter received unexpected trailing features; '
                f'expected {pair_dim}, got {x.shape[-1]}'
            )
        return robot, buttons

    def __call__(self, x):
        robot, buttons = self._split_input(jnp.asarray(x))
        tokens = self.button_projection(buttons)
        if self.index_embedding_param is not None:
            tokens = tokens + self.index_embedding_param
        context = self.robot_projection(robot)
        context = nn.gelu(context)
        if self.robot_layer_norm is not None:
            context = self.robot_layer_norm(context)
        return StructuredRepresentation(tokens=tokens, context=context)
