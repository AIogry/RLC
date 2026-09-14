"""Composite computation blocks."""

from .entity_mlp import EntityMLPBlock, EntityMLPStack
from .linear_boolean_flow import (
    LinearBooleanCoupling,
    LinearBooleanFlow,
    binary_threshold_ste,
    binary_xor,
    gf2_parity,
    make_permutation_schedule,
    permutation_schedule_metadata,
)
from .mlp_mixer import MLPMixerBlock, MLPMixerStack
from .residual_mlp import ResidualMLPBlock, ResidualMLPStack

__all__ = (
    'EntityMLPBlock',
    'EntityMLPStack',
    'LinearBooleanCoupling',
    'LinearBooleanFlow',
    'MLPMixerBlock',
    'MLPMixerStack',
    'ResidualMLPBlock',
    'ResidualMLPStack',
    'binary_threshold_ste',
    'binary_xor',
    'gf2_parity',
    'make_permutation_schedule',
    'permutation_schedule_metadata',
)
