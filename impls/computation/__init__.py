"""Small, composable computation framework."""

from .factory import ComputationSpec, make_computation_core, resolve_slot_spec
from .accounting import structured_body_accounting
from .structured import PuzzleStructuredBody
from .interfaces import ComputationCore, ComputationOutput
from .slots import (
    ComputationSlotDescriptor,
    SLOT_DESCRIPTORS,
    descriptor_for,
    descriptors_for,
    validate_compute_slots,
)

__all__ = (
    'ComputationCore',
    'ComputationOutput',
    'ComputationSpec',
    'make_computation_core',
    'resolve_slot_spec',
    'structured_body_accounting',
    'PuzzleStructuredBody',
    'ComputationSlotDescriptor',
    'SLOT_DESCRIPTORS',
    'descriptor_for',
    'descriptors_for',
    'validate_compute_slots',
)
