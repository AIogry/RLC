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
    # Appended fields preserve the four-field positional constructor used by
    # historical M15--M19 adapters and tests.  ``None`` is the explicit
    # relation-free legacy bypass; M20A Zero is instead a real all-zero tensor.
    relations: Any = None
    relation_mask: Any = None
