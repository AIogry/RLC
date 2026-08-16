"""Parameter-count helpers for slots and computation cores."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax


def count_parameters(tree) -> int:
    """Return the number of scalar parameters in a JAX/Flax pytree."""

    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = getattr(leaf, 'shape', None)
        total += math.prod(shape) if shape is not None else 1
    return int(total)


def count_non_trainable(tree) -> int:
    """Count scalar elements in a non-parameter variable collection."""

    return count_parameters(tree)


def _lookup_module(tree, name):
    if not isinstance(tree, Mapping):
        raise KeyError(name)
    for key in (name, f'modules_{name}'):
        if key in tree:
            return tree[key]
    raise KeyError(name)


def count_parameters_per_slot(params, slot_names):
    """Count named agent module subtrees, accepting ModuleDict naming."""

    return {name: count_parameters(_lookup_module(params, name)) for name in slot_names}


@dataclass(frozen=True)
class ParameterReport:
    total: int
    per_slot: dict
    per_core: dict


def make_parameter_report(params, *, slot_names=(), core_params=None) -> ParameterReport:
    """Create a report for total, slot, and separately supplied core params."""

    per_core = {}
    if core_params is not None:
        if isinstance(core_params, Mapping):
            per_core = {name: count_parameters(value) for name, value in core_params.items()}
        else:
            per_core = {'core': count_parameters(core_params)}
    return ParameterReport(
        total=count_parameters(params),
        per_slot=count_parameters_per_slot(params, slot_names),
        per_core=per_core,
    )
