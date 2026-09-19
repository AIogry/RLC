"""Static, content-addressed binary coordinate transforms (not a solver).

Validation, inversion and random generation are setup-only. Forward application
uses gather or the existing production GF(2) map. Payloads are self-contained;
paths and seeds alone are never accepted as transform definitions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import jax.numpy as jnp
import numpy as np

from .puzzle_algebra import _validate_concrete_binary, apply_gf2_map, gf2_inverse, gf2_rank


MATRIX_CONVENTION = 'y=B@x mod 2; B[encoded_coordinate,original_operation_coordinate]'
PERMUTATION_CONVENTION = 'y[...,j]=x[...,p[j]]; P[j,p[j]]=1'
_DERIVED_KEYS = {'num_coordinates', 'convention', 'content_sha256', 'rank',
                 'row_weights', 'column_weights'}


def semantic_hash(payload):
    """Stable JSON hash; deliberately independent of generation provenance."""
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}')
    return int(value)


@dataclass(frozen=True)
class GoalCoordinateTransform:
    kind: str
    matrix: tuple[tuple[int, ...], ...]
    permutation: tuple[int, ...] | None
    payload_json: str
    content_sha256: str

    def to_config(self):
        return json.loads(self.payload_json)

    def __call__(self, bits):
        bits = jnp.asarray(bits)
        if bits.ndim < 1 or bits.shape[-1] != len(self.matrix):
            raise ValueError('Coordinate transform input width does not match its matrix')
        _validate_concrete_binary(bits, name='Goal coordinate bits')
        if self.kind == 'identity':
            return bits
        if self.kind == 'permutation':
            return jnp.take(bits, jnp.asarray(self.permutation), axis=-1)
        return apply_gf2_map(self.matrix, bits)

    def inverse(self):
        """Build an inverse at setup, reusing the one public GF(2) solver."""
        return resolve_goal_coordinate_transform(
            {'kind': 'gf2_linear', 'matrix': gf2_inverse(self.matrix).tolist()},
            num_coordinates=len(self.matrix),
        )


def resolve_goal_coordinate_transform(payload, *, num_coordinates):
    n = _integer(num_coordinates, 'num_coordinates')
    if not isinstance(payload, Mapping) and not hasattr(payload, 'items'):
        raise ValueError('Coordinate transform must be an inline mapping')
    kind = payload.get('kind')
    if kind not in ('identity', 'permutation', 'gf2_linear'):
        raise ValueError(f'Unsupported coordinate transform kind: {kind!r}')
    allowed = {'kind', 'provenance'} | _DERIVED_KEYS
    if kind == 'permutation':
        allowed.add('permutation')
    elif kind == 'gf2_linear':
        allowed.add('matrix')
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(f'Unsupported transform fields: {sorted(unexpected)!r}')
    permutation = None
    if kind == 'identity':
        matrix = np.eye(n, dtype=np.uint8)
    elif kind == 'permutation':
        values = payload.get('permutation')
        if not isinstance(values, (list, tuple)) or len(values) != n:
            raise ValueError(f'permutation must have exactly {n} entries')
        permutation = tuple(_integer(v, 'permutation entry', minimum=0) for v in values)
        if sorted(permutation) != list(range(n)):
            raise ValueError('permutation must contain each coordinate exactly once')
        matrix = np.eye(n, dtype=np.uint8)[list(permutation)]
    else:
        matrix = np.asarray(payload.get('matrix'))
        if matrix.shape != (n, n):
            raise ValueError(f'GF(2) coordinate matrix must have shape {(n, n)}, got {matrix.shape}')
        # gf2_rank also performs the shared numeric/binary/finite validation.
        if gf2_rank(matrix) != n:
            raise ValueError('GF(2) coordinate matrix must be invertible (full-rank)')
        matrix = matrix.astype(np.uint8)
    rank = gf2_rank(matrix)
    provenance = payload.get('provenance', {'generator': 'inline', 'version': 1, 'transform_seed': None})
    if not isinstance(provenance, Mapping) and not hasattr(provenance, 'items'):
        raise ValueError('transform provenance must be a mapping')
    if set(provenance) != {'generator', 'version', 'transform_seed'}:
        raise ValueError('transform provenance requires only generator, version, transform_seed')
    if not isinstance(provenance['generator'], str) or not provenance['generator']:
        raise ValueError('transform generator must be a nonempty string')
    version = _integer(provenance['version'], 'generator version')
    seed = provenance['transform_seed']
    if seed is not None:
        seed = _integer(seed, 'transform_seed', minimum=0)
    content_sha256 = semantic_hash({'convention': MATRIX_CONVENTION, 'matrix': matrix.tolist()})
    normalized = {
        'kind': kind,
        'num_coordinates': n,
        'convention': PERMUTATION_CONVENTION if kind == 'permutation' else MATRIX_CONVENTION,
        'content_sha256': content_sha256,
        'rank': rank,
        'row_weights': matrix.sum(axis=1).astype(int).tolist(),
        'column_weights': matrix.sum(axis=0).astype(int).tolist(),
        'provenance': {'generator': provenance['generator'], 'version': version, 'transform_seed': seed},
    }
    if kind == 'permutation':
        normalized['permutation'] = list(permutation)
    elif kind == 'gf2_linear':
        normalized['matrix'] = matrix.tolist()
    for key in _DERIVED_KEYS & set(payload):
        supplied = payload[key]
        if isinstance(supplied, tuple):
            supplied = list(supplied)
        if supplied != normalized[key]:
            raise ValueError(f'Transform derived metadata mismatch for {key!r}')
    return GoalCoordinateTransform(
        kind=kind,
        matrix=tuple(tuple(int(v) for v in row) for row in matrix),
        permutation=permutation,
        payload_json=json.dumps(normalized, sort_keys=True, allow_nan=False),
        content_sha256=content_sha256,
    )


def generate_goal_coordinate_transform(*, kind, num_coordinates, transform_seed, max_attempts=128):
    """Generate an explicit payload with a private RNG and a bounded search.

    No density or derangement condition is implied. Scientific constraints,
    if desired, must be specified in a future protocol, not inferred here.
    """
    n = _integer(num_coordinates, 'num_coordinates')
    seed = _integer(transform_seed, 'transform_seed', minimum=0)
    attempts = _integer(max_attempts, 'max_attempts')
    rng = np.random.default_rng(seed)
    payload = {'kind': kind, 'provenance': {
        'generator': 'goal_coordinate_transform', 'version': 1, 'transform_seed': seed,
    }}
    if kind == 'permutation':
        payload['permutation'] = rng.permutation(n).tolist()
    elif kind == 'gf2_linear':
        for _ in range(attempts):
            matrix = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
            if gf2_rank(matrix) == n:
                payload['matrix'] = matrix.tolist()
                break
        else:
            raise ValueError(f'No invertible GF(2) matrix found in {attempts} attempts')
    elif kind != 'identity':
        raise ValueError(f'Unsupported coordinate transform kind: {kind!r}')
    return resolve_goal_coordinate_transform(payload, num_coordinates=n).to_config()
