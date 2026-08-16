"""Gradient/credit propagation policies."""

from .direct import DirectCredit
from .full_bptt import FullBPTTCredit
from .one_step import OneStepCredit

__all__ = ('DirectCredit', 'FullBPTTCredit', 'OneStepCredit')
