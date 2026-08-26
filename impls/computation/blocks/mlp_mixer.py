"""Reference-faithful MLP-Mixer block candidate for the computation layer.

This block is intentionally standalone in M8.  Vanilla CoGHP continues to
use ``impls.networks.coghp.MixerBlock`` until a separate production migration
gate approves switching its import path.
"""

import flax.linen as nn
import jax.numpy as jnp

from ..primitives.mlp import default_init


class MLPMixerBlock(nn.Module):
    """Official CoGHP token/channel mixing block without algorithm wiring."""

    num_tokens: int
    embed_dim: int
    hidden_dim_tokens: int
    hidden_dim_channels: int
    init_scale: float = 1e-2
    decay_alpha: float = 0.9
    tm_mode: str = 'lower_triangular'

    def setup(self):
        if self.tm_mode not in ('none', 'lower_triangular'):
            raise ValueError(
                f'Unsupported tm_mode {self.tm_mode!r}; expected '
                "'none' or 'lower_triangular'"
            )
        self.token_dense1 = nn.Dense(self.hidden_dim_tokens, kernel_init=default_init())
        self.token_dense2 = nn.Dense(self.num_tokens, kernel_init=default_init())
        self.channel_dense1 = nn.Dense(self.hidden_dim_channels, kernel_init=default_init())
        self.channel_dense2 = nn.Dense(self.embed_dim, kernel_init=default_init())
        if self.tm_mode == 'lower_triangular':
            self.tm_weights = self.param(
                'tm_weights',
                nn.initializers.normal(stddev=0.02),
                (self.num_tokens, self.num_tokens),
            )
            self.tm_weights = jnp.tril(self.tm_weights)
        else:
            # Do not create a parameter at all for the non-causal Puzzle
            # path.  Assigning None keeps the parameter tree unchanged.
            self.tm_weights = None

    def __call__(self, x):
        y = jnp.transpose(x, (0, 2, 1))
        y = self.token_dense1(y)
        y = nn.gelu(y)
        y = self.token_dense2(y)
        y = jnp.transpose(y, (0, 2, 1))
        if self.tm_mode == 'lower_triangular':
            y = jnp.einsum('btd,ts->bsd', y, self.tm_weights)
        x = x + y

        z = self.channel_dense1(x)
        z = nn.gelu(z)
        z = self.channel_dense2(z)
        return x + z
