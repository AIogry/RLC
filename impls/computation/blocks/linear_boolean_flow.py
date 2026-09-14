"""Exactly reversible linear Boolean flows built from GF(2) couplings.

The primal computation in this module is hard binary.  Real-valued logits
only provide a straight-through derivative for the binary coupling matrices;
they never turn the forward representation into a continuous bottleneck.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def _hard_step_with_sigmoid_jvp(scaled_logits):
    """Return an exact hard step whose JVP is the sigmoid derivative."""

    return (scaled_logits >= 0).astype(scaled_logits.dtype)


@_hard_step_with_sigmoid_jvp.defjvp
def _hard_step_with_sigmoid_jvp_rule(primals, tangents):
    (scaled_logits,), (scaled_tangent,) = primals, tangents
    hard = _hard_step_with_sigmoid_jvp(scaled_logits)
    soft = jax.nn.sigmoid(scaled_logits)
    return hard, scaled_tangent * soft * (1.0 - soft)


def binary_threshold_ste(logits, temperature=1.0):
    """Threshold logits to exact 0/1 with a temperature-scaled STE gradient."""

    if not isinstance(temperature, Real) or isinstance(temperature, bool) or temperature <= 0:
        raise ValueError(f'temperature must be a positive scalar, got {temperature!r}')
    logits = jnp.asarray(logits)
    if not jnp.issubdtype(logits.dtype, jnp.floating):
        logits = logits.astype(jnp.float32)
    return _hard_step_with_sigmoid_jvp(logits / float(temperature))


def binary_xor(left, right):
    """Differentiable XOR with exact Boolean values for binary primals."""

    left = jnp.asarray(left)
    right = jnp.asarray(right)
    return left + right - 2 * left * right


def gf2_parity(source, coupling_matrix):
    """Compute exact hard GF(2) parity without a real-valued matmul forward."""

    source = jnp.asarray(source)
    coupling_matrix = jnp.asarray(coupling_matrix)
    if source.ndim < 1 or coupling_matrix.ndim != 2:
        raise ValueError(
            'GF(2) parity expects source[..., A] and matrix[A, B], got '
            f'{source.shape} and {coupling_matrix.shape}'
        )
    if source.shape[-1] != coupling_matrix.shape[0]:
        raise ValueError(
            'GF(2) parity source/matrix mismatch: '
            f'{source.shape[-1]} != {coupling_matrix.shape[0]}'
        )
    output = jnp.zeros(
        source.shape[:-1] + (coupling_matrix.shape[1],), dtype=source.dtype
    )
    # Iterated Boolean XOR preserves a hard 0/1 primal while JAX differentiates
    # its polynomial extension.  Ordinary real-valued matrix multiplication is
    # deliberately absent from the forward path.
    for source_index in range(source.shape[-1]):
        term = source[..., source_index, None] * coupling_matrix[source_index]
        output = binary_xor(output, term)
    return output


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')


def validate_binary_array(value, *, name='binary array', final_dim=None):
    """Validate concrete binary input while remaining safe under JAX tracing."""

    array = jnp.asarray(value)
    if array.ndim < 1:
        raise ValueError(f'{name} requires a final bit axis, got {array.shape}')
    if final_dim is not None and array.shape[-1] != final_dim:
        raise ValueError(
            f'{name} expected final dimension {final_dim}, got {array.shape[-1]}'
        )
    if isinstance(array, jax.core.Tracer):
        return array
    concrete = np.asarray(array)
    if not np.all(np.isfinite(concrete)):
        raise ValueError(f'{name} contains non-finite values')
    if not np.all((concrete == 0) | (concrete == 1)):
        raise ValueError(f'{name} must contain exactly 0/1 values')
    return array


def make_permutation_schedule(num_bits, num_layers, permutation_seed):
    """Construct deterministic, non-trainable, partition-diverse permutations."""

    _validate_positive_integer(num_bits, 'num_bits')
    _validate_positive_integer(num_layers, 'num_layers')
    if num_bits < 2:
        raise ValueError('A coupling flow requires at least two bits')
    if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, Integral):
        raise ValueError(
            f'permutation_seed must be an integer, got {permutation_seed!r}'
        )

    split_size = num_bits // 2
    maximum_partitions = math.comb(num_bits, split_size)
    if num_layers > maximum_partitions:
        raise ValueError(
            f'num_layers={num_layers} exceeds the {maximum_partitions} distinct '
            f'source partitions available for N={num_bits}'
        )
    rng = np.random.default_rng(int(permutation_seed))
    schedule = []
    partitions = set()

    # Begin with complementary halves when N is even.  This guarantees that
    # every variable appears on both sides after two layers while retaining a
    # seed-specific ordering.  Remaining partitions are drawn independently.
    base = tuple(int(item) for item in rng.permutation(num_bits))
    candidates = [base]
    if num_layers > 1:
        candidates.append(base[split_size:] + base[:split_size])

    attempts = 0
    while len(schedule) < num_layers:
        if candidates:
            candidate = candidates.pop(0)
        else:
            candidate = tuple(int(item) for item in rng.permutation(num_bits))
        partition = tuple(sorted(candidate[:split_size]))
        attempts += 1
        if partition in partitions:
            if attempts > 100_000:
                raise ValueError(
                    'Could not construct the requested number of distinct '
                    f'coupling partitions for N={num_bits}, L={num_layers}'
                )
            continue
        partitions.add(partition)
        schedule.append(candidate)
    return tuple(schedule)


def permutation_schedule_metadata(schedule):
    """Return JSON-safe architectural metadata for a permutation schedule."""

    if not schedule:
        raise ValueError('Permutation schedule must not be empty')
    num_bits = len(schedule[0])
    split_size = num_bits // 2
    source_counts = np.zeros(num_bits, dtype=np.int64)
    target_counts = np.zeros(num_bits, dtype=np.int64)
    layers = []
    for layer_index, permutation in enumerate(schedule):
        permutation = tuple(int(item) for item in permutation)
        if len(permutation) != num_bits or set(permutation) != set(range(num_bits)):
            raise ValueError(f'Invalid permutation at layer {layer_index}: {permutation!r}')
        source = permutation[:split_size]
        target = permutation[split_size:]
        source_counts[list(source)] += 1
        target_counts[list(target)] += 1
        layers.append({
            'layer': layer_index,
            'permutation': list(permutation),
            'source_partition': list(source),
            'target_partition': list(target),
        })
    return {
        'num_bits': num_bits,
        'num_layers': len(schedule),
        'split_size': split_size,
        'layers': layers,
        'source_partition_counts': source_counts.tolist(),
        'target_partition_counts': target_counts.tolist(),
        'all_coordinates_exposed_as_source': bool(np.all(source_counts > 0)),
        'all_coordinates_exposed_as_target': bool(np.all(target_counts > 0)),
    }


def _logit_initializer(mean, std):
    def initializer(key, shape, dtype=jnp.float32):
        value = jnp.full(shape, mean, dtype=dtype)
        if std:
            value = value + std * jax.random.normal(key, shape, dtype=dtype)
        return value

    return initializer


class LinearBooleanCoupling(nn.Module):
    """One conjugated GF(2) coupling with an exact explicit inverse."""

    num_bits: int
    permutation: tuple[int, ...]
    coupling_logit_init_mean: float = -2.0
    coupling_logit_init_std: float = 0.1
    binary_temperature: float = 1.0

    def setup(self):
        _validate_positive_integer(self.num_bits, 'num_bits')
        if self.num_bits < 2:
            raise ValueError('A coupling layer requires at least two bits')
        permutation = tuple(int(item) for item in self.permutation)
        if len(permutation) != self.num_bits or set(permutation) != set(range(self.num_bits)):
            raise ValueError(f'Invalid fixed permutation: {permutation!r}')
        if self.coupling_logit_init_mean >= 0:
            raise ValueError(
                'coupling_logit_init_mean must be negative for near-identity initialization'
            )
        if self.coupling_logit_init_std < 0:
            raise ValueError('coupling_logit_init_std must be non-negative')
        if self.binary_temperature <= 0:
            raise ValueError('binary_temperature must be positive')

        self.split_size = self.num_bits // 2
        self.fixed_permutation = jnp.asarray(permutation, dtype=jnp.int32)
        self.inverse_permutation = jnp.argsort(self.fixed_permutation)
        self.coupling_logits = self.param(
            'coupling_logits',
            _logit_initializer(
                float(self.coupling_logit_init_mean),
                float(self.coupling_logit_init_std),
            ),
            (self.split_size, self.num_bits - self.split_size),
        )

    def hard_coupling_matrix(self):
        return binary_threshold_ste(
            self.coupling_logits, temperature=self.binary_temperature
        )

    def _couple_in_permuted_order(self, permuted):
        source = permuted[..., :self.split_size]
        target = permuted[..., self.split_size:]
        contribution = gf2_parity(source, self.hard_coupling_matrix())
        return jnp.concatenate([source, binary_xor(target, contribution)], axis=-1)

    def forward(self, inputs):
        inputs = validate_binary_array(
            inputs, name='Boolean coupling input', final_dim=self.num_bits
        )
        permuted = inputs[..., self.fixed_permutation]
        coupled = self._couple_in_permuted_order(permuted)
        # Conjugating by the fixed permutation makes zero couplings exactly the
        # identity while still changing the source/target partition per layer.
        return coupled[..., self.inverse_permutation]

    def inverse(self, outputs):
        outputs = validate_binary_array(
            outputs, name='Boolean coupling output', final_dim=self.num_bits
        )
        permuted = outputs[..., self.fixed_permutation]
        # A GF(2) additive coupling is self-inverse because the source half is
        # unchanged.  This is kept as an explicit inverse path for auditability.
        uncoupled = self._couple_in_permuted_order(permuted)
        return uncoupled[..., self.inverse_permutation]

    def __call__(self, inputs):
        return self.forward(inputs)


class LinearBooleanFlow(nn.Module):
    """Composition of exactly invertible linear Boolean coupling layers."""

    num_bits: int
    permutations: tuple[tuple[int, ...], ...]
    coupling_logit_init_mean: float = -2.0
    coupling_logit_init_std: float = 0.1
    binary_temperature: float = 1.0

    def setup(self):
        _validate_positive_integer(self.num_bits, 'num_bits')
        if not self.permutations:
            raise ValueError('LinearBooleanFlow requires at least one coupling layer')
        self.coupling_layers = tuple(
            LinearBooleanCoupling(
                num_bits=self.num_bits,
                permutation=tuple(permutation),
                coupling_logit_init_mean=self.coupling_logit_init_mean,
                coupling_logit_init_std=self.coupling_logit_init_std,
                binary_temperature=self.binary_temperature,
                name=f'coupling_{layer_index:03d}',
            )
            for layer_index, permutation in enumerate(self.permutations)
        )

    def forward(self, inputs):
        value = validate_binary_array(
            inputs, name='Boolean flow input', final_dim=self.num_bits
        )
        for layer in self.coupling_layers:
            value = layer.forward(value)
        return value

    def inverse(self, outputs):
        value = validate_binary_array(
            outputs, name='Boolean flow output', final_dim=self.num_bits
        )
        for layer in reversed(self.coupling_layers):
            value = layer.inverse(value)
        return value

    def hard_coupling_matrices(self):
        return tuple(layer.hard_coupling_matrix() for layer in self.coupling_layers)

    def __call__(self, inputs):
        return self.forward(inputs)


__all__ = [
    'LinearBooleanCoupling',
    'LinearBooleanFlow',
    'binary_threshold_ste',
    'binary_xor',
    'gf2_parity',
    'make_permutation_schedule',
    'permutation_schedule_metadata',
    'validate_binary_array',
]
