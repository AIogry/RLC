"""Parameter-count and dense-MAC helpers for computation slots.

The MAC helpers intentionally count only matrix multiplications represented by
Flax ``Dense`` kernels.  Bias additions, activations, normalization, and
environment/evaluation work are not silently folded into this number.
"""

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


def count_dense_macs(tree) -> int:
    """Count Dense matrix-multiplication MACs from a parameter subtree.

    A Dense kernel has shape ``(input_features, output_features)``.  Counting
    these products from the actual parameter shapes keeps the accounting
    independent of a particular environment's observation/action dimensions.
    """

    if not hasattr(tree, 'items'):
        return 0
    total = 0
    for name, value in tree.items():
        if name == 'kernel':
            shape = getattr(value, 'shape', None)
            if shape is not None and len(shape) == 2:
                total += math.prod(shape)
        elif hasattr(value, 'items'):
            total += count_dense_macs(value)
    return int(total)


def _mapping_get(tree, key, default=None):
    return tree.get(key, default) if hasattr(tree, 'get') else default


def _actor_core_params(actor_params):
    actor_net = _mapping_get(actor_params, 'actor_net', actor_params)
    topology = _mapping_get(actor_net, 'topology')
    return topology if hasattr(topology, 'items') else actor_net


def _actor_readout_params(actor_params):
    readout = {}
    for name in ('mean_net', 'logit_net', 'log_std_net', 'log_stds'):
        value = _mapping_get(actor_params, name)
        if value is not None:
            readout[name] = value
    return readout


def actor_slot_accounting(actor_params, buffer_params=None, *, topology=None, iterations=0):
    """Audit one actor slot using its actual parameter and buffer shapes.

    ``iterations`` is supplied by the resolved configuration rather than
    inferred from the number of parameters, because SingleState reuses one
    physical update module for every execution.  The returned dictionary is
    JSON-friendly and is suitable for runtime metadata and compact audit
    tables.
    """

    actor_params = actor_params if hasattr(actor_params, 'items') else {}
    buffer_params = buffer_params if hasattr(buffer_params, 'items') else {}
    core = _actor_core_params(actor_params)
    input_mapping = _mapping_get(core, 'input_mapping', {})
    update_module = _mapping_get(core, 'update_module', {})
    h_update = _mapping_get(core, 'h_update', {})
    l_update = _mapping_get(core, 'l_update', {})
    if topology == 'single_state':
        iterations = int(iterations)

    input_macs = count_dense_macs(input_mapping)
    update_per_execution = count_dense_macs(update_module)
    h_update_per_execution = count_dense_macs(h_update)
    l_update_per_execution = count_dense_macs(l_update)
    if topology == 'single_state':
        h_update_executions = 0
        l_update_executions = iterations
        update_executions = iterations
        total_update_macs = update_per_execution * iterations
    elif topology == 'two_state':
        h_cycles = int(iterations[0]) if isinstance(iterations, (tuple, list)) else 0
        l_cycles = int(iterations[1]) if isinstance(iterations, (tuple, list)) else 0
        h_update_executions = h_cycles
        l_update_executions = h_cycles * l_cycles
        update_executions = h_update_executions + l_update_executions
        total_update_macs = (
            h_update_per_execution * h_cycles
            + l_update_per_execution * l_cycles
        )
    else:
        h_cycles = l_cycles = 0
        h_update_executions = 0
        l_update_executions = 0
        update_executions = 0
        total_update_macs = 0

    core_macs = input_macs + total_update_macs
    readout_macs = count_dense_macs(_actor_readout_params(actor_params))
    if topology in ('single_state', 'two_state'):
        full_actor_forward_macs = core_macs + readout_macs
    else:
        full_actor_forward_macs = count_dense_macs(actor_params)
    return {
        'topology': topology,
        'input_mapping_dense_macs': input_macs,
        'update_module_dense_macs_per_execution': update_per_execution,
        'h_update_dense_macs_per_execution': h_update_per_execution,
        'l_update_dense_macs_per_execution': l_update_per_execution,
        'h_update_executions': h_update_executions,
        'l_update_executions': l_update_executions,
        'total_update_executions': update_executions,
        # Backward-compatible name: it denotes all H/L executions, not raw
        # schedule-cycle counts.  For TwoState this agrees with
        # len(execution_trace(h_cycles, l_cycles)).
        'update_executions': update_executions,
        'total_update_module_dense_macs': total_update_macs,
        'computation_core_dense_macs': core_macs,
        'readout_dense_macs': readout_macs,
        'full_actor_forward_dense_macs': full_actor_forward_macs,
        'parameter_tree_dense_macs': count_dense_macs(actor_params),
        # Backward-compatible short name; it denotes one forward execution.
        'full_actor_dense_macs': full_actor_forward_macs,
        'trainable_params': count_parameters(actor_params),
        'core_trainable_params': count_parameters(core),
        'buffer_elements': count_non_trainable(buffer_params),
    }


def _path_get(tree, path):
    for key in path:
        if not hasattr(tree, 'get') or key not in tree:
            return {}
        tree = tree[key]
    return tree


def computation_slot_accounting(
    slot_params,
    buffer_params=None,
    *,
    slot_name,
    topology=None,
    primitive=None,
    credit=None,
    topology_kwargs=None,
    core_path=(),
):
    """Return generic accounting for any enabled computation slot.

    ``core_path`` points from the slot module root to the topology-owned
    parameter subtree.  Actor modules use ``('actor_net', 'topology')``;
    CRL bilinear branches use ``('core', 'topology')``.  Keeping this path
    explicit avoids making the accounting helper infer algorithm semantics
    from parameter names while allowing actor and critic slots to share one
    audit schema.
    """

    topology_kwargs = topology_kwargs if hasattr(topology_kwargs, 'get') else {}
    slot_params = slot_params if hasattr(slot_params, 'items') else {}
    buffer_params = buffer_params if hasattr(buffer_params, 'items') else {}
    core = _path_get(slot_params, core_path) if core_path else slot_params
    buffer_core = _path_get(buffer_params, core_path) if core_path else buffer_params
    is_recurrent = topology in ('single_state', 'two_state')

    state_dim = None
    update_depth = None
    iterations = None
    residual = None
    h_cycles = None
    l_cycles = None
    total_update_executions = 0
    state_init = None
    state_init_std = None
    if is_recurrent:
        state_dim = int(topology_kwargs.get('state_dim'))
        update_depth = int(topology_kwargs.get('update_depth', 2))
        state_init = topology_kwargs.get('state_init', 'normal_buffer')
        state_init_std = float(topology_kwargs.get('state_init_std', 1.0))
        if topology == 'single_state':
            iterations = int(topology_kwargs.get('iterations', 1))
            residual = bool(topology_kwargs.get('residual', False))
            total_update_executions = iterations
        else:
            h_cycles = int(topology_kwargs.get('h_cycles', 2))
            l_cycles = int(topology_kwargs.get('l_cycles', 1))
            residual = False
            total_update_executions = h_cycles * (l_cycles + 1)

    input_mapping = _mapping_get(core, 'input_mapping', {})
    update_module = _mapping_get(core, 'update_module', {})
    h_update = _mapping_get(core, 'h_update', {})
    l_update = _mapping_get(core, 'l_update', {})
    return {
        'slot_name': str(slot_name),
        'topology': topology,
        'primitive': primitive,
        'credit': credit,
        'state_dim': state_dim,
        'update_depth': update_depth,
        'iterations': iterations,
        'residual': residual,
        'h_cycles': h_cycles,
        'l_cycles': l_cycles,
        'total_update_executions': total_update_executions,
        'h_update_executions': h_cycles if h_cycles is not None else 0,
        'l_update_executions': h_cycles * l_cycles if h_cycles is not None else 0,
        'state_init': state_init,
        'state_init_std': state_init_std,
        'trainable_params': count_parameters(slot_params),
        'core_trainable_params': count_parameters(core),
        'buffer_elements': count_non_trainable(buffer_params),
        'core_buffer_elements': count_non_trainable(buffer_core),
        'input_mapping_params': count_parameters(input_mapping),
        'update_module_params': count_parameters(update_module),
        'h_update_params': count_parameters(h_update),
        'l_update_params': count_parameters(l_update),
    }


def hiql_policy_accounting(params, buffers=None, slot_specs=None):
    """Audit the independent HIQL high/low actor paths.

    The function consumes resolved parameter trees and therefore does not
    assume AntMaze dimensions.  ``slot_specs`` should map ``high_actor`` and
    ``low_actor`` to resolved slot mappings.
    """

    params = params if hasattr(params, 'items') else {}
    buffers = buffers if hasattr(buffers, 'items') else {}
    slot_specs = slot_specs if hasattr(slot_specs, 'items') else {}
    slots = {}
    for slot_name in ('high_actor', 'low_actor'):
        spec = slot_specs.get(slot_name, {}) or {}
        enabled = bool(spec.get('enabled', False)) if hasattr(spec, 'get') else False
        topology = spec.get('topology') if enabled and hasattr(spec, 'get') else None
        kwargs = spec.get('topology_kwargs', {}) if hasattr(spec, 'get') else {}
        if topology == 'single_state':
            iterations = int(kwargs.get('iterations', 1))
        elif topology == 'two_state':
            iterations = (
                int(kwargs.get('h_cycles', 0)),
                int(kwargs.get('l_cycles', 0)),
            )
        else:
            iterations = 0
        module = _mapping_get(params, f'modules_{slot_name}', _mapping_get(params, slot_name, {}))
        buffer_module = _mapping_get(buffers, f'modules_{slot_name}', _mapping_get(buffers, slot_name, {}))
        slots[slot_name] = actor_slot_accounting(
            module,
            buffer_module,
            topology=topology,
            iterations=iterations,
        )

    high = slots['high_actor']
    low = slots['low_actor']
    return {
        'slots': slots,
        'combined_high_low_computation_core_dense_macs': (
            high['computation_core_dense_macs']
            + low['computation_core_dense_macs']
        ),
        'combined_high_low_full_actor_dense_macs': (
            high['full_actor_dense_macs'] + low['full_actor_dense_macs']
        ),
        'combined_high_low_full_actor_forward_dense_macs': (
            high['full_actor_forward_dense_macs']
            + low['full_actor_forward_dense_macs']
        ),
        'combined_high_low_trainable_params': (
            high['trainable_params'] + low['trainable_params']
        ),
        'combined_high_low_buffer_elements': (
            high['buffer_elements'] + low['buffer_elements']
        ),
        'network_total_trainable_params': count_parameters(params),
        'network_total_buffer_elements': count_non_trainable(buffers),
    }


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
