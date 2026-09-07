"""Strict Cube and Scene entity parsers plus learned token adapters.

These adapters mirror the canonical state observations emitted by the local
OGBench manipulation environments.  They intentionally consume only the
standard observation/goal vectors; no privileged dataset-only fields enter a
production network.
"""

from numbers import Integral
from typing import Mapping

import flax.linen as nn
import jax.numpy as jnp

from .interfaces import StructuredRepresentation
from .puzzle import _default_init
from .relations import build_cube_relations, build_scene_relations


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


def parse_cube_observation(
    x,
    *,
    num_cubes: int,
    robot_dim: int = 19,
    cube_feature_dim: int = 9,
):
    """Split canonical Cube state observations into robot and cube slots.

    OGBench ``CubeEnv.compute_observation`` emits
    ``robot(19) + cube_0(9) + ...`` where every cube feature is
    ``scaled_xyz(3), quaternion(4), cos(yaw), sin(yaw)``.  No canonical
    feature is removed by this parser.
    """

    num_cubes = _positive_integer('num_cubes', num_cubes)
    robot_dim = _positive_integer('robot_dim', robot_dim)
    cube_feature_dim = _positive_integer('cube_feature_dim', cube_feature_dim)
    x = jnp.asarray(x)
    if x.ndim < 1:
        raise ValueError(f'Cube observation must have a final feature axis, got {x.shape}')
    expected_dim = robot_dim + num_cubes * cube_feature_dim
    if x.shape[-1] != expected_dim:
        raise ValueError(
            'Malformed Cube observation: expected final dimension '
            f'{expected_dim} (= {robot_dim} + {num_cubes} * {cube_feature_dim}), '
            f'got {x.shape[-1]} for shape {x.shape}'
        )
    robot = x[..., :robot_dim]
    cubes = x[..., robot_dim:].reshape(*x.shape[:-1], num_cubes, cube_feature_dim)
    return robot, cubes


def parse_scene_observation(
    x,
    *,
    robot_dim: int = 19,
    cube_feature_dim: int = 9,
    button_feature_dim: int = 4,
    drawer_feature_dim: int = 2,
    window_feature_dim: int = 2,
):
    """Split the canonical Scene state vector into source-audited entities.

    The exact state ordering is ``robot, cube, button_0, button_1, drawer,
    window``.  Drawer/window each contain the scaled slide position and its
    velocity; button slots contain one-hot state, joint position and velocity.
    """

    for name, value in (
        ('robot_dim', robot_dim),
        ('cube_feature_dim', cube_feature_dim),
        ('button_feature_dim', button_feature_dim),
        ('drawer_feature_dim', drawer_feature_dim),
        ('window_feature_dim', window_feature_dim),
    ):
        _positive_integer(name, value)
    x = jnp.asarray(x)
    if x.ndim < 1:
        raise ValueError(f'Scene observation must have a final feature axis, got {x.shape}')
    expected_dim = (
        robot_dim + cube_feature_dim + 2 * button_feature_dim
        + drawer_feature_dim + window_feature_dim
    )
    if x.shape[-1] != expected_dim:
        raise ValueError(
            'Malformed Scene observation: expected final dimension '
            f'{expected_dim}, got {x.shape[-1]} for shape {x.shape}'
        )
    offset = 0
    robot = x[..., offset:offset + robot_dim]
    offset += robot_dim
    cube = x[..., offset:offset + cube_feature_dim]
    offset += cube_feature_dim
    buttons = x[..., offset:offset + 2 * button_feature_dim]
    buttons = buttons.reshape(*x.shape[:-1], 2, button_feature_dim)
    offset += 2 * button_feature_dim
    drawer = x[..., offset:offset + drawer_feature_dim]
    offset += drawer_feature_dim
    window = x[..., offset:offset + window_feature_dim]
    return robot, cube, buttons, drawer, window


def _split_goal_pair(x, *, observation_dim, action_semantics, label):
    """Split a raw network input while preserving critic-action isolation."""

    x = jnp.asarray(x)
    pair_dim = 2 * observation_dim
    if x.shape[-1] < pair_dim:
        raise ValueError(
            f'{label} goal_pair adapter expected at least final dimension {pair_dim}, '
            f'got {x.shape[-1]}'
        )
    state = x[..., :observation_dim]
    goal = x[..., observation_dim:pair_dim]
    if action_semantics == 'robot_context':
        action = x[..., pair_dim:]
        if action.shape[-1] <= 0:
            raise ValueError(f'{label} critic robot_context requires a non-empty action vector')
    elif action_semantics == 'none':
        if x.shape[-1] != pair_dim:
            raise ValueError(
                f'{label} goal_pair adapter received unexpected trailing features; '
                f'expected {pair_dim}, got {x.shape[-1]}'
            )
        action = None
    else:
        raise ValueError(f'Unsupported {label} action_semantics: {action_semantics!r}')
    return state, goal, action


class CubeTokenAdapter(nn.Module):
    """Map a canonical Cube state/goal pair to three entity tokens and context."""

    num_cubes: int = 3
    robot_dim: int = 19
    cube_feature_dim: int = 9
    token_dim: int = 128
    robot_hidden_dim: int = 128
    # Source audit: role assignments are permuted at task reset, so M20A
    # leaves this false.  It is exposed only to make the decision auditable.
    slot_identity_embedding: bool = False
    input_semantics: str = 'goal_pair'
    action_semantics: str = 'none'
    layer_norm: bool = False
    relation_mode: str = 'legacy_none'
    relation_kwargs: Mapping | None = None

    def setup(self):
        for name, value in (
            ('num_cubes', self.num_cubes),
            ('robot_dim', self.robot_dim),
            ('cube_feature_dim', self.cube_feature_dim),
            ('token_dim', self.token_dim),
            ('robot_hidden_dim', self.robot_hidden_dim),
        ):
            _positive_integer(name, value)
        if self.input_semantics != 'goal_pair':
            raise ValueError('Cube token adapter requires input_semantics=goal_pair')
        if self.action_semantics not in ('none', 'robot_context'):
            raise ValueError(
                'Cube token adapter supports action_semantics=none or robot_context, '
                f'got {self.action_semantics!r}'
            )
        self.cube_entity_encoder = nn.Dense(self.token_dim, kernel_init=_default_init())
        if self.slot_identity_embedding:
            self.cube_slot_embedding = self.param(
                'cube_slot_embedding',
                nn.initializers.normal(stddev=0.02),
                (self.num_cubes, self.token_dim),
            )
        else:
            self.cube_slot_embedding = None
        self.robot_projection = nn.Dense(self.robot_hidden_dim, kernel_init=_default_init())
        self.robot_layer_norm = nn.LayerNorm() if self.layer_norm else None

    @property
    def observation_dim(self):
        return self.robot_dim + self.num_cubes * self.cube_feature_dim

    def _split_input(self, x):
        state, goal, action = _split_goal_pair(
            x,
            observation_dim=self.observation_dim,
            action_semantics=self.action_semantics,
            label='Cube',
        )
        robot_state, cubes_state = parse_cube_observation(
            state,
            num_cubes=self.num_cubes,
            robot_dim=self.robot_dim,
            cube_feature_dim=self.cube_feature_dim,
        )
        robot_goal, cubes_goal = parse_cube_observation(
            goal,
            num_cubes=self.num_cubes,
            robot_dim=self.robot_dim,
            cube_feature_dim=self.cube_feature_dim,
        )
        robot = jnp.concatenate((robot_state, robot_goal), axis=-1)
        if action is not None:
            robot = jnp.concatenate((robot, action), axis=-1)
        cubes = jnp.concatenate((cubes_state, cubes_goal), axis=-1)
        return robot, cubes, cubes_state, cubes_goal

    def __call__(self, x):
        robot, cubes, cubes_state, cubes_goal = self._split_input(x)
        tokens = self.cube_entity_encoder(cubes)
        if self.cube_slot_embedding is not None:
            tokens = tokens + self.cube_slot_embedding
        context = self.robot_projection(robot)
        context = nn.gelu(context)
        if self.robot_layer_norm is not None:
            context = self.robot_layer_norm(context)
        relations = None
        if self.relation_mode != 'legacy_none':
            kwargs = dict(self.relation_kwargs or {})
            relations = build_cube_relations(
                cubes_state,
                cubes_goal,
                mode=self.relation_mode,
                current_support_epsilon_xy=kwargs.get('current_support_epsilon_xy'),
                current_support_epsilon_z=kwargs.get('current_support_epsilon_z'),
                goal_support_epsilon_xy=kwargs.get('goal_support_epsilon_xy'),
                goal_support_epsilon_z=kwargs.get('goal_support_epsilon_z'),
                conflict_radius=kwargs.get('goal_conflict_radius'),
                shuffle_derangement=kwargs.get('shuffle_derangement', (1, 2, 0)),
                cube_height=kwargs.get('cube_height', 0.04),
                xyz_scaler=kwargs.get('xyz_scaler', 10.0),
            )
        return StructuredRepresentation(tokens=tokens, context=context, relations=relations)


class SceneTokenAdapter(nn.Module):
    """Map the heterogeneous canonical Scene pair to five typed entity tokens."""

    robot_dim: int = 19
    cube_feature_dim: int = 9
    button_feature_dim: int = 4
    drawer_feature_dim: int = 2
    window_feature_dim: int = 2
    token_dim: int = 128
    robot_hidden_dim: int = 128
    button_role_embedding: bool = True
    input_semantics: str = 'goal_pair'
    action_semantics: str = 'none'
    layer_norm: bool = False
    relation_mode: str = 'legacy_none'
    relation_kwargs: Mapping | None = None

    def setup(self):
        for name, value in (
            ('robot_dim', self.robot_dim),
            ('cube_feature_dim', self.cube_feature_dim),
            ('button_feature_dim', self.button_feature_dim),
            ('drawer_feature_dim', self.drawer_feature_dim),
            ('window_feature_dim', self.window_feature_dim),
            ('token_dim', self.token_dim),
            ('robot_hidden_dim', self.robot_hidden_dim),
        ):
            _positive_integer(name, value)
        if self.input_semantics != 'goal_pair':
            raise ValueError('Scene token adapter requires input_semantics=goal_pair')
        if self.action_semantics not in ('none', 'robot_context'):
            raise ValueError(
                'Scene token adapter supports action_semantics=none or robot_context, '
                f'got {self.action_semantics!r}'
            )
        # Type-specific projections preserve the heterogeneous canonical
        # feature semantics.  button_encoder is shared across button_0/1.
        self.cube_encoder = nn.Dense(self.token_dim, kernel_init=_default_init())
        self.button_encoder = nn.Dense(self.token_dim, kernel_init=_default_init())
        self.drawer_encoder = nn.Dense(self.token_dim, kernel_init=_default_init())
        self.window_encoder = nn.Dense(self.token_dim, kernel_init=_default_init())
        if self.button_role_embedding:
            self.button_role_embedding_param = self.param(
                'button_role_embedding',
                nn.initializers.normal(stddev=0.02),
                (2, self.token_dim),
            )
        else:
            self.button_role_embedding_param = None
        self.robot_projection = nn.Dense(self.robot_hidden_dim, kernel_init=_default_init())
        self.robot_layer_norm = nn.LayerNorm() if self.layer_norm else None

    @property
    def observation_dim(self):
        return (
            self.robot_dim + self.cube_feature_dim + 2 * self.button_feature_dim
            + self.drawer_feature_dim + self.window_feature_dim
        )

    def _split_input(self, x):
        state, goal, action = _split_goal_pair(
            x,
            observation_dim=self.observation_dim,
            action_semantics=self.action_semantics,
            label='Scene',
        )
        state_parts = parse_scene_observation(
            state,
            robot_dim=self.robot_dim,
            cube_feature_dim=self.cube_feature_dim,
            button_feature_dim=self.button_feature_dim,
            drawer_feature_dim=self.drawer_feature_dim,
            window_feature_dim=self.window_feature_dim,
        )
        goal_parts = parse_scene_observation(
            goal,
            robot_dim=self.robot_dim,
            cube_feature_dim=self.cube_feature_dim,
            button_feature_dim=self.button_feature_dim,
            drawer_feature_dim=self.drawer_feature_dim,
            window_feature_dim=self.window_feature_dim,
        )
        robot = jnp.concatenate((state_parts[0], goal_parts[0]), axis=-1)
        if action is not None:
            robot = jnp.concatenate((robot, action), axis=-1)
        cube = jnp.concatenate((state_parts[1], goal_parts[1]), axis=-1)
        buttons = jnp.concatenate((state_parts[2], goal_parts[2]), axis=-1)
        drawer = jnp.concatenate((state_parts[3], goal_parts[3]), axis=-1)
        window = jnp.concatenate((state_parts[4], goal_parts[4]), axis=-1)
        return robot, cube, buttons, drawer, window, state, goal

    def __call__(self, x):
        robot, cube, buttons, drawer, window, state, goal = self._split_input(x)
        cube_token = self.cube_encoder(cube)[..., None, :]
        button_tokens = self.button_encoder(buttons)
        if self.button_role_embedding_param is not None:
            button_tokens = button_tokens + self.button_role_embedding_param
        drawer_token = self.drawer_encoder(drawer)[..., None, :]
        window_token = self.window_encoder(window)[..., None, :]
        tokens = jnp.concatenate(
            (cube_token, button_tokens, drawer_token, window_token), axis=-2
        )
        context = self.robot_projection(robot)
        context = nn.gelu(context)
        if self.robot_layer_norm is not None:
            context = self.robot_layer_norm(context)
        relations = None
        if self.relation_mode != 'legacy_none':
            relations = build_scene_relations(state, goal, mode=self.relation_mode)
        return StructuredRepresentation(tokens=tokens, context=context, relations=relations)
