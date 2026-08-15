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
    """Apply one topology to one representation primitive.

    The core owns result normalization, while the selected topology owns the
    meaning of an optional execution state.  This keeps the interface ready
    for a future stateful topology without making the current feed-forward
    path pretend to carry state.
    """

    topology: nn.Module

    def __call__(self, x, state=None):
        output = self.topology(x, state=state)
        if isinstance(output, ComputationOutput):
            return output
        return ComputationOutput(representation=output)
