"""Parameter-free public representation interfaces.

Representations describe how a raw algorithm input is organised before a
computation topology runs.  They intentionally own no Flax state or learned
parameters: ownership belongs to the adapter module that produces them.
"""

from typing import Any, NamedTuple


class StructuredRepresentation(NamedTuple):
    """Canonical structured input for a computation body.

    ``tokens`` is normally ``[B, T, D]`` at the internal computation
    boundary.  Public adapters may return a single-observation ``[T, D]``
    value; :class:`StructuredComputationBody` normalizes that boundary.
    """

    tokens: Any
    context: Any = None
    mask: Any = None
    auxiliary: Any = None
