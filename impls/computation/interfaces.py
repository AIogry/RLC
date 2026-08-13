"""Public interfaces for computation modules.

The first implementation deliberately keeps this layer small. A topology
returns a `ComputationOutput`, so a recurrent topology can add state later
without making feed-forward computation pretend to have one.
"""

from typing import Any, NamedTuple

import flax.linen as nn


class ComputationOutput(NamedTuple):
    """Result of a computation core."""

    representation: Any
    state: Any = None
    auxiliary: Any = None


class ComputationCore(nn.Module):
    """Apply one topology to one representation primitive."""

    topology: nn.Module

    def __call__(self, x, state=None):
        if state is not None:
            raise ValueError('The feedforward core does not accept recurrent state.')
        output = self.topology(x)
        if isinstance(output, ComputationOutput):
            return output
        return ComputationOutput(representation=output)
