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


class HybridContextQueryReadout(nn.Module):
    """Mean context readout augmented with a context-conditioned token query.

    This is intentionally a readout *package*, not a parameter/MAC-matched
    attention ablation.  It retains the final fusion activation and optional
    LayerNorm semantics of :class:`MeanContextReadout` while adding learned
    query/key/value projections and scaled dot-product attention.
    """

    output_dim: int
    token_dim: int = 128
    query_dim: int = 128
    layer_norm: bool = False
    activate_final: bool = True

    def setup(self):
        for name, value in (
            ('output_dim', self.output_dim),
            ('token_dim', self.token_dim),
            ('query_dim', self.query_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        self.query_projection = nn.Dense(self.query_dim, kernel_init=default_init())
        self.key_projection = nn.Dense(self.query_dim, kernel_init=default_init())
        self.value_projection = nn.Dense(self.token_dim, kernel_init=default_init())
        self.fusion = nn.Dense(self.output_dim, kernel_init=default_init())
        self.fusion_layer_norm = nn.LayerNorm() if self.layer_norm else None

    def _validate_inputs(self, tokens, context, mask):
        tokens = jnp.asarray(tokens)
        context = jnp.asarray(context)
        if tokens.ndim != 3:
            raise ValueError(
                f'HybridContextQueryReadout expects tokens [B, T, D], got {tokens.shape}'
            )
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                'HybridContextQueryReadout final token dimension must equal token_dim; '
                f'got {tokens.shape[-1]} and {self.token_dim}'
            )
        if context.ndim != 2 or context.shape[0] != tokens.shape[0]:
            raise ValueError(
                'HybridContextQueryReadout context must be [B, C] with matching batch size; '
                f'got tokens={tokens.shape}, context={context.shape}'
            )
        if mask is None:
            mask = jnp.ones(tokens.shape[:2], dtype=bool)
        else:
            mask = jnp.asarray(mask)
            if mask.shape != tokens.shape[:2]:
                raise ValueError(
                    'HybridContextQueryReadout mask must have shape [B, T]; '
                    f'got tokens={tokens.shape}, mask={mask.shape}'
                )
            mask = mask.astype(bool)
        return tokens, context, mask

    def _summaries(self, tokens, context, mask):
        weights = mask.astype(tokens.dtype)[..., None]
        denominator = jnp.maximum(jnp.sum(weights, axis=-2), 1.0)
        mean_summary = jnp.sum(tokens * weights, axis=-2) / denominator

        query = self.query_projection(context)
        keys = self.key_projection(tokens)
        values = self.value_projection(tokens)
        scores = jnp.einsum('bq,btq->bt', query, keys) / jnp.sqrt(float(self.query_dim))
        # ``finfo.min`` avoids NaNs under JAX softmax.  The explicit second
        # masking/renormalization below also defines the all-invalid boundary
        # as zero attention rather than an arbitrary uniform distribution.
        masked_scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        attention = nn.softmax(masked_scores, axis=-1)
        attention = jnp.where(mask, attention, 0.0)
        attention = attention / jnp.maximum(jnp.sum(attention, axis=-1, keepdims=True), 1.0)
        query_summary = jnp.einsum('bt,btd->bd', attention, values)
        return mean_summary, query_summary, attention

    def __call__(self, tokens, *, context, mask=None):
        tokens, context, mask = self._validate_inputs(tokens, context, mask)
        mean_summary, query_summary, _ = self._summaries(tokens, context, mask)
        fused = jnp.concatenate((mean_summary, query_summary, context), axis=-1)
        fused = self.fusion(fused)
        if self.activate_final:
            fused = nn.gelu(fused)
            if self.fusion_layer_norm is not None:
                fused = self.fusion_layer_norm(fused)
        return fused

    def attention_weights(self, tokens, *, context, mask=None):
        """Return diagnostic attention weights without changing normal output."""

        tokens, context, mask = self._validate_inputs(tokens, context, mask)
        _, _, attention = self._summaries(tokens, context, mask)
        return attention
