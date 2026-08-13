"""Small, shared seed helpers for the HIQL runtime."""

import random

import numpy as np


def derive_seed(base_seed, *components):
    """Derive a stable uint32 seed for an independent stream."""
    entropy = [int(base_seed) & 0xFFFFFFFF]
    entropy.extend(int(component) & 0xFFFFFFFF for component in components)
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def seed_everything(seed):
    """Seed compatibility globals; runtime sampling uses explicit streams."""
    random.seed(seed)
    np.random.seed(seed)
