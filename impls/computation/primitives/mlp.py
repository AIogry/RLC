"""The original OGBench MLP as a computation primitive.

This is intentionally not a redesigned MLP. Its ordering and defaults match
`offline_rl_baselines/ogbench/impls/utils/networks.py` exactly.
"""

from typing import Any, Sequence

import flax.linen as nn


def default_init(scale=1.0):
    """OGBench's default kernel initializer."""

    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """OGBench-compatible multi-layer perceptron."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x
