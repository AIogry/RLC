"""Entity-wise channel MLP blocks for structured token computation.

The modules in this file intentionally duplicate only the channel branch of
``MLPMixerBlock``.  They do not share implementation code with Mixer because
the historical Mixer parameter tree and RNG behaviour are frozen by the M16
and M18 experiments.
"""

from numbers import Integral

import flax.linen as nn
import jax.numpy as jnp

from ..primitives.mlp import default_init


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


class EntityMLPBlock(nn.Module):
    """Apply the Mixer channel branch independently to every token.

    For a canonical token tensor ``H`` with shape ``[B, T, D]`` this module
    computes ``H + Dense2(GELU(Dense1(H)))``.  The two Dense modules act only
    on the final channel axis, so their parameters are shared across all
    token positions and contain no token-axis state.
    """

    embed_dim: int
    hidden_dim_channels: int

    def setup(self):
        _positive_integer('embed_dim', self.embed_dim)
        _positive_integer('hidden_dim_channels', self.hidden_dim_channels)
        # Keep these module names, dimensions, activation, and initializer
        # exactly aligned with MLPMixerBlock's channel branch.
        self.channel_dense1 = nn.Dense(
            self.hidden_dim_channels,
            kernel_init=default_init(),
        )
        self.channel_dense2 = nn.Dense(
            self.embed_dim,
            kernel_init=default_init(),
        )

    def __call__(self, x):
        x = jnp.asarray(x)
        if x.ndim != 3:
            raise ValueError(f'EntityMLPBlock expects [B, T, D], got {x.shape}')
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                'EntityMLPBlock final dimension must equal embed_dim; '
                f'got {x.shape[-1]} and {self.embed_dim}'
            )
        z = self.channel_dense1(x)
        z = nn.gelu(z)
        z = self.channel_dense2(z)
        return x + z


class EntityMLPStack(nn.Module):
    """A stack of untied entity-wise MLP blocks.

    ``num_blocks`` is the intra-block feedforward depth ``L``.  It is not a
    token count and does not affect parameter sharing across token positions.
    """

    num_blocks: int
    embed_dim: int
    hidden_dim_channels: int

    def setup(self):
        num_blocks = _positive_integer('num_blocks', self.num_blocks)
        _positive_integer('embed_dim', self.embed_dim)
        _positive_integer('hidden_dim_channels', self.hidden_dim_channels)
        self.blocks = tuple(
            EntityMLPBlock(
                embed_dim=self.embed_dim,
                hidden_dim_channels=self.hidden_dim_channels,
            )
            for _ in range(num_blocks)
        )

    def __call__(self, x):
        x = jnp.asarray(x)
        if x.ndim != 3:
            raise ValueError(f'EntityMLPStack expects [B, T, D], got {x.shape}')
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                'EntityMLPStack final dimension must equal embed_dim; '
                f'got {x.shape[-1]} and {self.embed_dim}'
            )
        for block in self.blocks:
            x = block(x)
        return x
