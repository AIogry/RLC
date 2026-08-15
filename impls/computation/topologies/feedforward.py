"""Feed-forward topology."""

import flax.linen as nn

from ..interfaces import ComputationOutput


class FeedForward(nn.Module):
    """Apply a primitive exactly once, with no recurrent state."""

    primitive: nn.Module

    def __call__(self, x, state=None):
        if state is not None:
            raise ValueError('FeedForward topology does not accept non-None state.')
        return ComputationOutput(representation=self.primitive(x))
