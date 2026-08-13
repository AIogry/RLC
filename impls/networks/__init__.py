"""Task-specific network modules."""

from .common import GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, MLP

__all__ = ('GCActor', 'GCDiscreteActor', 'GCValue', 'Identity', 'LengthNormalize', 'MLP')
from .common import GCActor, GCDiscreteActor, GCBilinearValue, GCDiscreteBilinearCritic

__all__ = ('GCActor', 'GCDiscreteActor', 'GCBilinearValue', 'GCDiscreteBilinearCritic')
