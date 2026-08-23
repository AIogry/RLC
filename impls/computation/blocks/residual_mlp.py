"""Reusable stateless residual MLP blocks and bodies."""

from typing import Sequence

import flax.linen as nn

from ..primitives.mlp import MLP


class ResidualMLPBlock(nn.Module):
    """A local residual mapping ``y = x + F(x)``."""

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    activate_final: bool = True

    def setup(self):
        if not self.hidden_dims or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError(f'ResidualMLPBlock hidden_dims must be positive: {self.hidden_dims!r}')
        self.mapping = MLP(
            hidden_dims=tuple(int(width) for width in self.hidden_dims),
            activate_final=self.activate_final,
            layer_norm=self.layer_norm,
        )

    def __call__(self, x):
        return x + self.mapping(x)


class ResidualMLPStack(nn.Module):
    """A stateless feed-forward body made from independent residual blocks.

    The body first maps raw input into ``state_dim`` and then applies each
    physical block once.  It has no recurrent state, buffers, or decision-local
    lifecycle; FeedForward owns the one-shot topology semantics.
    """

    state_dim: int
    blocks: int = 4
    block_depth: int = 2
    layer_norm: bool = False
    block_activate_final: bool = True

    def setup(self):
        if self.state_dim <= 0:
            raise ValueError(f'state_dim must be positive, got {self.state_dim}')
        if self.blocks <= 0 or self.block_depth <= 0:
            raise ValueError(
                f'ResidualMLPStack blocks and block_depth must be positive, '
                f'got {(self.blocks, self.block_depth)!r}'
            )
        self.input_mapping = MLP(
            hidden_dims=(self.state_dim,),
            activate_final=True,
            layer_norm=self.layer_norm,
        )
        self.residual_blocks = tuple(
            ResidualMLPBlock(
                hidden_dims=(self.state_dim,) * int(self.block_depth),
                layer_norm=self.layer_norm,
                activate_final=self.block_activate_final,
            )
            for _ in range(int(self.blocks))
        )

    def __call__(self, x):
        hidden = self.input_mapping(x)
        for block in self.residual_blocks:
            hidden = block(hidden)
        return hidden
