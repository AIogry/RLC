"""Readouts from structured computation states to algorithm vectors."""

from numbers import Integral

import flax.linen as nn
import jax.numpy as jnp

from .primitives.mlp import default_init


class MeanContextReadout(nn.Module):
    """Mean-pool tokens and fuse them with an adapter-provided context.

    This is the exact readout behavior used by the legacy Puzzle body: token
    mean, context concatenation, one fusion Dense, then optional final GELU
    and LayerNorm.  It contains no Puzzle parsing or computation topology.
    """

    output_dim: int
    layer_norm: bool = False
    activate_final: bool = True

    def setup(self):
        if isinstance(self.output_dim, bool) or not isinstance(self.output_dim, Integral) or self.output_dim <= 0:
            raise ValueError(f'output_dim must be a positive integer, got {self.output_dim!r}')
        self.fusion = nn.Dense(self.output_dim, kernel_init=default_init())
        self.fusion_layer_norm = nn.LayerNorm() if self.layer_norm else None

    def __call__(self, tokens, *, context, mask=None):
        tokens = jnp.asarray(tokens)
        context = jnp.asarray(context)
        if tokens.ndim != 3:
            raise ValueError(f'MeanContextReadout expects tokens [B, T, D], got {tokens.shape}')
        if context.ndim != 2 or context.shape[0] != tokens.shape[0]:
            raise ValueError(
                'MeanContextReadout context must be [B, C] with the same batch size as tokens; '
                f'got tokens={tokens.shape}, context={context.shape}'
            )
        if mask is None:
            summary = jnp.mean(tokens, axis=-2)
        else:
            mask = jnp.asarray(mask)
            if mask.shape != tokens.shape[:-1]:
                raise ValueError(
                    'MeanContextReadout mask must have shape [B, T]; '
                    f'got tokens={tokens.shape}, mask={mask.shape}'
                )
            weights = mask.astype(tokens.dtype)[..., None]
            denominator = jnp.maximum(jnp.sum(weights, axis=-2), 1.0)
            summary = jnp.sum(tokens * weights, axis=-2) / denominator
        fused = jnp.concatenate([summary, context], axis=-1)
        fused = self.fusion(fused)
        if self.activate_final:
            fused = nn.gelu(fused)
            if self.fusion_layer_norm is not None:
                fused = self.fusion_layer_norm(fused)
        return fused
