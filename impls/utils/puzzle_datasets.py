"""Puzzle-specific goal-success semantics over canonical GCDataset sampling."""

from __future__ import annotations

import dataclasses

import numpy as np

from ..networks.goal_conditioning import goal_conditioning_layout
from ..representation.puzzle_conditioning import extract_button_bits
from .datasets import GCDataset


@dataclasses.dataclass
class PuzzleBoardGCDataset(GCDataset):
    """GCDataset whose success is equality of current and goal board bits.

    Goal/transition sampling, reward and mask construction, augmentation, and
    RNG ownership remain in ``GCDataset``.  This subclass changes only the
    deterministic goal-achievement predicate.
    """

    def __post_init__(self):
        super().__post_init__()
        # Dataset ownership stops at board layout/equality.  Full oracle
        # rank/inverse validation runs in main and agent construction.
        self._goal_layout = goal_conditioning_layout(
            self.config.get('goal_conditioning'),
            compute_slots=self.config.get('compute'),
            dataset_class='PuzzleBoardGCDataset',
        )

    def _compute_value_successes(self, idxs, value_goal_idxs):
        layout = self._goal_layout
        current_bits = extract_button_bits(
            self.get_observations(idxs),
            num_buttons=layout['num_buttons'],
            robot_dim=layout['robot_dim'],
            button_feature_dim=layout['button_feature_dim'],
        )
        goal_bits = extract_button_bits(
            self.get_observations(value_goal_idxs),
            num_buttons=layout['num_buttons'],
            robot_dim=layout['robot_dim'],
            button_feature_dim=layout['button_feature_dim'],
        )
        return np.all(
            np.asarray(current_bits) == np.asarray(goal_bits), axis=-1
        ).astype(float)


__all__ = ['PuzzleBoardGCDataset']
