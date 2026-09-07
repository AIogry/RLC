"""Generic relation augmentation for structured token computation."""

from numbers import Integral

import flax.linen as nn
import jax.numpy as jnp

from .primitives.mlp import default_init


class RelationAugmenter(nn.Module):
    """Add a task-agnostic incoming/outgoing relation update to token states.

    Relation direction is fixed globally as ``R[i, j, k] = i --k--> j``.
    For each type the module aggregates target-token states along outgoing
    edges and source-token states along incoming edges, then maps only those
    aggregate features through a bias-free two-layer MLP.  In particular,
    ``H_i`` is not concatenated into the relation MLP input, which guarantees
    an exact residual identity for an all-zero relation tensor.
    """

    token_dim: int
    relation_hidden_dim: int = 256

    def setup(self):
        for name, value in (
            ('token_dim', self.token_dim),
            ('relation_hidden_dim', self.relation_hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        self.relation_dense1 = nn.Dense(
            self.relation_hidden_dim,
            use_bias=False,
            kernel_init=default_init(),
        )
        self.relation_dense2 = nn.Dense(
            self.token_dim,
            use_bias=False,
            kernel_init=default_init(),
        )

    @staticmethod
    def _normalize_relation_mask(relations, entity_mask, relation_mask):
        """Apply entity and optional relation masks to a binary relation tensor."""

        relations = jnp.asarray(relations)
        if relations.ndim != 4:
            raise ValueError(
                'RelationAugmenter expects relations [B, T, T, K], '
                f'got {relations.shape}'
            )
        batch_size, num_tokens, target_tokens, _ = relations.shape
        if num_tokens != target_tokens:
            raise ValueError(
                'RelationAugmenter relation axes must be square [B, T, T, K], '
                f'got {relations.shape}'
            )
        effective = jnp.ones(relations.shape, dtype=bool)
        if entity_mask is not None:
            entity_mask = jnp.asarray(entity_mask)
            if entity_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    'Entity mask must be [B, T] for RelationAugmenter; '
                    f'got relation={relations.shape}, mask={entity_mask.shape}'
                )
            pair_mask = entity_mask.astype(bool)[:, :, None] & entity_mask.astype(bool)[:, None, :]
            effective = effective & pair_mask[..., None]
        if relation_mask is not None:
            relation_mask = jnp.asarray(relation_mask)
            if relation_mask.shape == (batch_size, num_tokens, num_tokens):
                relation_mask = relation_mask[..., None]
            if relation_mask.shape != relations.shape:
                raise ValueError(
                    'relation_mask must be [B, T, T] or [B, T, T, K]; '
                    f'got relation={relations.shape}, relation_mask={relation_mask.shape}'
                )
            effective = effective & relation_mask.astype(bool)
        return relations * effective.astype(relations.dtype), effective

    @classmethod
    def aggregate_features(cls, tokens, relations, *, mask=None, relation_mask=None):
        """Return parameter-free outgoing/incoming aggregate features.

        This public diagnostic primitive shares the exact production mask and
        direction implementation used by ``__call__``.  It exists so tests
        can verify edge orientation without making claims from random Dense
        weights or constructing a task-specific relation network.
        """

        tokens = jnp.asarray(tokens)
        relations = jnp.asarray(relations)
        if tokens.ndim != 3:
            raise ValueError(f'RelationAugmenter expects tokens [B, T, D], got {tokens.shape}')
        if relations.shape[:3] != tokens.shape[:2] + (tokens.shape[1],):
            raise ValueError(
                'Relation tensor batch/token axes must match tokens [B, T, D]; '
                f'got tokens={tokens.shape}, relations={relations.shape}'
            )
        relations, effective = cls._normalize_relation_mask(relations, mask, relation_mask)
        outgoing_degree = jnp.sum(relations, axis=2)
        outgoing = jnp.einsum('bijk,bjd->bikd', relations, tokens)
        outgoing = outgoing / jnp.maximum(outgoing_degree[..., None], 1.0)
        incoming_degree = jnp.sum(relations, axis=1)
        incoming = jnp.einsum('bjik,bjd->bikd', relations, tokens)
        incoming = incoming / jnp.maximum(incoming_degree[..., None], 1.0)
        features = jnp.concatenate(
            (
                outgoing.reshape(tokens.shape[0], tokens.shape[1], -1),
                incoming.reshape(tokens.shape[0], tokens.shape[1], -1),
            ),
            axis=-1,
        )
        return features, effective

    def __call__(self, tokens, relations, *, mask=None, relation_mask=None):
        tokens = jnp.asarray(tokens)
        relations = jnp.asarray(relations)
        if tokens.ndim != 3:
            raise ValueError(f'RelationAugmenter expects tokens [B, T, D], got {tokens.shape}')
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                'RelationAugmenter token channel width must equal token_dim; '
                f'got {tokens.shape[-1]} and {self.token_dim}'
            )
        features, _ = self.aggregate_features(
            tokens, relations, mask=mask, relation_mask=relation_mask
        )
        delta = self.relation_dense1(features)
        delta = nn.gelu(delta)
        delta = self.relation_dense2(delta)
        return tokens + delta
