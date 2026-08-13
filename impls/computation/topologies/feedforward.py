"""Feed-forward topology."""

import flax.linen as nn

from ..interfaces import ComputationOutput


class FeedForward(nn.Module):
    """Apply a primitive exactly once, with no recurrent state."""

    primitive: nn.Module

    def __call__(self, x):
        return ComputationOutput(representation=self.primitive(x))
