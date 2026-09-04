"""Small, composable computation framework."""

from .factory import ComputationSpec, make_computation_core, resolve_slot_spec
from .accounting import modular_structured_body_accounting, structured_body_accounting
from .blocks.entity_mlp import EntityMLPBlock, EntityMLPStack
from .readouts import MeanContextReadout
from .structured import PuzzleStructuredBody, StructuredComputationBody
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
    'EntityMLPBlock',
    'EntityMLPStack',
    'structured_body_accounting',
    'modular_structured_body_accounting',
    'PuzzleStructuredBody',
    'StructuredComputationBody',
    'MeanContextReadout',
    'ComputationSlotDescriptor',
    'SLOT_DESCRIPTORS',
    'descriptor_for',
    'descriptors_for',
    'validate_compute_slots',
)
