"""Small, composable computation framework."""

from .factory import ComputationSpec, make_computation_core, resolve_slot_spec
from .interfaces import ComputationCore, ComputationOutput

__all__ = (
    'ComputationCore',
    'ComputationOutput',
    'ComputationSpec',
    'make_computation_core',
    'resolve_slot_spec',
)
