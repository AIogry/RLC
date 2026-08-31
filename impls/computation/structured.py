"""Structured computation orchestration and legacy Puzzle reference body."""

from numbers import Integral

import flax.linen as nn
import jax.numpy as jnp

from .blocks.mlp_mixer import MLPMixerBlock
from .interfaces import ComputationOutput
from .primitives.mlp import default_init
from ..representation.interfaces import StructuredRepresentation
from ..representation.puzzle import parse_puzzle_observation


class StructuredComputationBody(nn.Module):
    """Compose representation adapter, computation core, and structured readout.

    The body is domain-agnostic: its adapter decides how raw input becomes
    tokens/context, its core decides how tokens are transformed, and its
    readout decides how a token state becomes the vector required by the
    existing algorithm head.  It is also the only layer that normalizes the
    public single-observation boundary to the internal ``[B, T, D]`` form.
    """

    adapter: nn.Module
    core: nn.Module
    readout: nn.Module

    @staticmethod
    def _normalize(rep):
        if not isinstance(rep, StructuredRepresentation):
            raise TypeError(
                'Structured adapter must return StructuredRepresentation, '
                f'got {type(rep)!r}'
            )
        tokens = jnp.asarray(rep.tokens)
        squeeze_batch = tokens.ndim == 2
        if squeeze_batch:
            tokens = tokens[None, ...]
        if tokens.ndim != 3:
            raise ValueError(
                'Structured computation canonical token shape is [B, T, D]; '
                f'got {tokens.shape}'
            )
        if rep.context is None:
            raise ValueError('Structured representation requires a context for the active readout')
        context = jnp.asarray(rep.context)
        if squeeze_batch:
            if context.ndim != 1:
                raise ValueError(
                    'Unbatched structured context must be [C]; '
                    f'got tokens={tokens.shape}, context={context.shape}'
                )
            context = context[None, ...]
        if context.ndim != 2 or context.shape[0] != tokens.shape[0]:
            raise ValueError(
                'Structured context must be [B, C] after normalization; '
                f'got tokens={tokens.shape}, context={context.shape}'
            )
        mask = rep.mask
        if mask is not None:
            mask = jnp.asarray(mask)
            if squeeze_batch:
                if mask.ndim != 1:
                    raise ValueError(
                        'Unbatched structured mask must be [T]; '
                        f'got {mask.shape}'
                    )
                mask = mask[None, ...]
            if mask.shape != tokens.shape[:-1]:
                raise ValueError(
                    'Structured mask must match [B, T]; '
                    f'got tokens={tokens.shape}, mask={mask.shape}'
                )
        return tokens, context, mask, squeeze_batch

    def __call__(self, x):
        representation = self.adapter(x)
        tokens, context, mask, squeeze_batch = self._normalize(representation)
        computed = self.core(tokens)
        if not isinstance(computed, ComputationOutput):
            computed = ComputationOutput(representation=computed)
        output = self.readout(computed.representation, context=context, mask=mask)
        computed_tokens = computed.representation[0] if squeeze_batch else computed.representation
        if squeeze_batch:
            output = output[0]
        auxiliary = {
            'adapter_auxiliary': representation.auxiliary,
            'computed_tokens': computed_tokens,
        }
        return ComputationOutput(
            representation=output,
            state=computed.state,
            auxiliary=auxiliary,
        )

    def trace_tokens(self, x, max_iterations=None):
        """Expose a structured recurrent trace without changing ``__call__``.

        The returned tensors retain the canonical batch axis even for an
        unbatched public input: ``token_states`` is ``[B, K+1, T, D]`` and
        ``readout_states`` is ``[B, K+1, H]``.  Every readout uses this
        module's restored adapter/context/readout parameters; no diagnostic
        module or alternate readout is constructed.
        """

        representation = self.adapter(x)
        tokens, context, mask, _ = self._normalize(representation)
        topology = getattr(self.core, 'topology', None)
        trace_fn = getattr(topology, 'trace_states', None)
        if trace_fn is None:
            raise ValueError(
                'Structured trace requires a topology exposing trace_states; '
                f'got {type(topology)!r}'
            )
        states = trace_fn(tokens, max_iterations)
        if not states:
            raise ValueError('Structured trace returned no recurrent states')
        readouts = tuple(
            self.readout(state, context=context, mask=mask)
            for state in states
        )
        return {
            'token_states': jnp.stack(states, axis=1),
            'readout_states': jnp.stack(readouts, axis=1),
        }


class PuzzleStructuredBody(nn.Module):
    """Puzzle tokenizer, MLP-Mixer stack, readout, and robot fusion.

    The external contract is a flat vector with arbitrary leading batch axes;
    the internal button representation is always ``[..., T, D]``.  A paired
    body receives ``[observation, goal]`` (and optionally trailing action
    context), while a single-observation body receives one raw observation.
    """

    output_dim: int
    num_buttons: int
    robot_dim: int = 19
    button_feature_dim: int = 4
    token_dim: int = 128
    robot_hidden_dim: int = 128
    token_mlp_hidden_dim: int = 64
    channel_mlp_hidden_dim: int = 256
    num_mixer_blocks: int = 1
    index_embedding: bool = True
    readout: str = 'mean'
    tm_mode: str = 'none'
    input_semantics: str = 'goal_pair'
    action_semantics: str = 'none'
    layer_norm: bool = False
    activate_final: bool = True

    def setup(self):
        integer_fields = {
            'output_dim': self.output_dim,
            'num_buttons': self.num_buttons,
            'robot_dim': self.robot_dim,
            'button_feature_dim': self.button_feature_dim,
            'token_dim': self.token_dim,
            'robot_hidden_dim': self.robot_hidden_dim,
            'token_mlp_hidden_dim': self.token_mlp_hidden_dim,
            'channel_mlp_hidden_dim': self.channel_mlp_hidden_dim,
            'num_mixer_blocks': self.num_mixer_blocks,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        if self.input_semantics not in ('goal_pair', 'single_observation', 'latent_vector'):
            raise ValueError(f'Unsupported Puzzle input_semantics: {self.input_semantics!r}')
        if self.input_semantics == 'latent_vector':
            raise ValueError('Puzzle structured computation cannot consume latent_vector input')
        if self.action_semantics not in ('none', 'robot_context', 'latent_dynamics'):
            raise ValueError(f'Unsupported Puzzle action_semantics: {self.action_semantics!r}')
        if self.action_semantics == 'latent_dynamics':
            raise ValueError('Puzzle structured computation cannot consume latent_dynamics input')
        if self.input_semantics == 'single_observation' and self.action_semantics != 'none':
            raise ValueError('single_observation Puzzle phi cannot include action context')
        if self.readout != 'mean':
            raise ValueError(f'Unsupported Puzzle token readout: {self.readout!r}')
        if self.tm_mode not in ('none', 'lower_triangular'):
            raise ValueError(f'Unsupported Puzzle tm_mode: {self.tm_mode!r}')

        button_input_dim = self.button_feature_dim
        if self.input_semantics == 'goal_pair':
            button_input_dim *= 2
        self.button_projection = nn.Dense(self.token_dim, kernel_init=default_init())
        if self.index_embedding:
            self.index_embedding_param = self.param(
                'index_embedding',
                nn.initializers.normal(stddev=0.02),
                (self.num_buttons, self.token_dim),
            )
        else:
            self.index_embedding_param = None
        self.robot_projection = nn.Dense(self.robot_hidden_dim, kernel_init=default_init())
        self.robot_layer_norm = nn.LayerNorm() if self.layer_norm else None
        self.mixer_blocks = [
            MLPMixerBlock(
                num_tokens=self.num_buttons,
                embed_dim=self.token_dim,
                hidden_dim_tokens=self.token_mlp_hidden_dim,
                hidden_dim_channels=self.channel_mlp_hidden_dim,
                tm_mode=self.tm_mode,
            )
            for _ in range(self.num_mixer_blocks)
        ]
        self.fusion = nn.Dense(self.output_dim, kernel_init=default_init())
        self.fusion_layer_norm = nn.LayerNorm() if self.layer_norm else None

    def _split_input(self, x):
        obs_dim = self.robot_dim + self.num_buttons * self.button_feature_dim
        if self.input_semantics == 'single_observation':
            if x.shape[-1] != obs_dim:
                raise ValueError(
                    f'Puzzle single_observation body expected final dimension {obs_dim}, '
                    f'got {x.shape[-1]}'
                )
            robot, buttons = parse_puzzle_observation(
                x,
                num_buttons=self.num_buttons,
                robot_dim=self.robot_dim,
                button_feature_dim=self.button_feature_dim,
            )
            return robot, buttons

        pair_dim = 2 * obs_dim
        if x.shape[-1] < pair_dim:
            raise ValueError(
                f'Puzzle goal_pair body expected at least final dimension {pair_dim}, '
                f'got {x.shape[-1]}'
            )
        state = x[..., :obs_dim]
        goal = x[..., obs_dim:2 * obs_dim]
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
            action = x[..., 2 * obs_dim:]
            if action.shape[-1] <= 0:
                raise ValueError('Puzzle critic robot_context requires a non-empty action vector')
            robot = jnp.concatenate([robot, action], axis=-1)
        elif x.shape[-1] != pair_dim:
            raise ValueError(
                'Puzzle goal_pair body received unexpected trailing features; '
                f'expected {pair_dim}, got {x.shape[-1]}'
            )
        return robot, buttons

    def __call__(self, x):
        robot, buttons = self._split_input(jnp.asarray(x))
        tokens = self.button_projection(buttons)
        if self.index_embedding_param is not None:
            tokens = tokens + self.index_embedding_param
        squeeze_batch = tokens.ndim == 2
        if squeeze_batch:
            # Training uses [B, T, D], while the public evaluation policy
            # receives one raw observation as [D].  MixerBlock intentionally
            # keeps its canonical rank-3 contract; this adapter only handles
            # the unbatched boundary and removes the temporary axis later.
            tokens = tokens[None, ...]
        if tokens.ndim != 3:
            raise ValueError(
                f'Puzzle token computation expects [B, T, D] internally, got {tokens.shape}'
            )
        for block in self.mixer_blocks:
            tokens = block(tokens)
        if squeeze_batch:
            tokens = tokens[0]
        button_summary = jnp.mean(tokens, axis=-2)
        robot_rep = self.robot_projection(robot)
        robot_rep = nn.gelu(robot_rep)
        if self.robot_layer_norm is not None:
            robot_rep = self.robot_layer_norm(robot_rep)
        fused = jnp.concatenate([button_summary, robot_rep], axis=-1)
        fused = self.fusion(fused)
        if self.activate_final:
            fused = nn.gelu(fused)
            if self.fusion_layer_norm is not None:
                fused = self.fusion_layer_norm(fused)
        return fused
