"""Input representations used by computation bodies."""

from .interfaces import StructuredRepresentation
from .puzzle import PuzzleTokenAdapter, parse_puzzle_observation

__all__ = ('StructuredRepresentation', 'PuzzleTokenAdapter', 'parse_puzzle_observation')
