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
            # Leading axes are ensemble/vmap replicas.  They represent
            # independent Dense transformations and therefore contribute to
            # physical and executed MAC accounting.
            if shape is not None and len(shape) >= 2:
                total += math.prod(shape)
        elif hasattr(value, 'items'):
            total += count_dense_macs(value)
    return int(total)


def count_dense_layers(tree) -> int:
    """Count physical Dense transformations in a parameter subtree."""

    if not hasattr(tree, 'items'):
        return 0
    total = 0
    for name, value in tree.items():
        if name == 'kernel':
            shape = getattr(value, 'shape', None)
            if shape is not None and len(shape) >= 2:
                total += math.prod(shape[:-2]) if len(shape) > 2 else 1
        elif hasattr(value, 'items'):
            total += count_dense_layers(value)
    return int(total)


def _dense_macs(tree, name):
    """Count Dense kernel products in one named structured-body subtree."""

    return count_dense_macs(tree.get(name, {})) if hasattr(tree, 'get') else 0


def _last_kernel_dims(tree):
    """Return ``(in_features, out_features)`` from a Dense subtree."""

    if not hasattr(tree, 'items'):
        return None
    kernel = tree.get('kernel')
    shape = getattr(kernel, 'shape', None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[-2]), int(shape[-1])


def structured_body_accounting(body_params, structure_kwargs=None):
    """Account a Puzzle structured body with token/channel execution factors.

    ``count_dense_macs`` alone counts one kernel product.  A token projection
    executes once per button token, token mixing once per channel, and channel
    mixing once per token.  This helper reports physical parameter elements
    from the actual tree and per-sample forward MACs with those multiplicities.
    """

    kwargs = structure_kwargs if hasattr(structure_kwargs, 'get') else {}
    body_params = body_params if hasattr(body_params, 'items') else {}
    num_tokens = int(kwargs.get('num_buttons', 0))
    token_dim = int(kwargs.get('token_dim', 0))
    token_hidden_dim = int(kwargs.get('token_mlp_hidden_dim', 0))
    channel_hidden_dim = int(kwargs.get('channel_mlp_hidden_dim', 0))
    num_blocks = int(kwargs.get('num_mixer_blocks', 0))
    if min(num_tokens, token_dim, token_hidden_dim, channel_hidden_dim, num_blocks) <= 0:
        raise ValueError(
            'Structured accounting requires positive num_buttons, token_dim, '
            'token_mlp_hidden_dim, channel_mlp_hidden_dim, and num_mixer_blocks'
        )

    mixer_blocks = _module_subtrees(body_params, 'mixer_blocks')
    button_params = count_parameters(body_params.get('button_projection', {}))
    index_params = count_parameters(body_params.get('index_embedding', {}))
    robot_params = count_parameters(body_params.get('robot_projection', {}))
    fusion_params = count_parameters(body_params.get('fusion', {}))
    mixer_params = sum(count_parameters(value) for value in mixer_blocks.values())
    block_macs = 0
    tm_macs = 0
    for block in mixer_blocks.values():
        block_macs += (
            _dense_macs(block, 'token_dense1') * token_dim
            + _dense_macs(block, 'token_dense2') * token_dim
            + _dense_macs(block, 'channel_dense1') * num_tokens
            + _dense_macs(block, 'channel_dense2') * num_tokens
        )
        tm = block.get('tm_weights') if hasattr(block, 'get') else None
        if tm is not None:
            tm_macs += count_parameters(tm) * token_dim

    button_macs = _dense_macs(body_params, 'button_projection') * num_tokens
    robot_macs = _dense_macs(body_params, 'robot_projection')
    fusion_macs = _dense_macs(body_params, 'fusion')
    dense_macs = button_macs + block_macs + tm_macs + robot_macs + fusion_macs
    button_dims = _last_kernel_dims(body_params.get('button_projection', {}))
    robot_dims = _last_kernel_dims(body_params.get('robot_projection', {}))
    fusion_dims = _last_kernel_dims(body_params.get('fusion', {}))
    # Button path: shared projection + 4 Dense layers per Mixer block +
    # fusion.  Robot path is projection + fusion; the longest sequential path
    # is therefore 1 + 4L + 1.
    sequential_depth = 4 * num_blocks + 2
    return {
        'num_tokens': num_tokens,
        'token_dim': token_dim,
        'token_hidden_dim': token_hidden_dim,
        'channel_hidden_dim': channel_hidden_dim,
        'num_mixer_blocks': num_blocks,
        'button_input_dim': button_dims[0] if button_dims else None,
        'robot_input_dim': robot_dims[0] if robot_dims else None,
        'output_dim': fusion_dims[1] if fusion_dims else None,
        'index_embedding': bool(kwargs.get('index_embedding', index_params > 0)),
        'readout': kwargs.get('readout', 'mean'),
        'tm_mode': kwargs.get('tm_mode', 'none'),
        'button_projection_params': int(button_params),
        'index_embedding_params': int(index_params),
        'robot_projection_params': int(robot_params),
        'mixer_params': int(mixer_params),
        'fusion_params': int(fusion_params),
        'structured_body_params': int(count_parameters(body_params)),
        'total_structured_body_params': int(count_parameters(body_params)),
        'structured_body_dense_macs': int(dense_macs),
        'structured_dense_macs': int(dense_macs),
        'total_per_sample_dense_macs': int(dense_macs),
        'structured_sequential_depth': int(sequential_depth),
        'sequential_depth': int(sequential_depth),
        'unique_dense_layers': int(count_dense_layers(body_params)),
        'executed_dense_layers': int(count_dense_layers(body_params)),
        'actor_body_dense_macs': int(dense_macs),
        'token_projection_dense_macs': int(button_macs),
        'mixer_dense_macs': int(block_macs),
        'tm_dense_macs': int(tm_macs),
        'robot_projection_dense_macs': int(robot_macs),
        'fusion_dense_macs': int(fusion_macs),
    }


def _direct_dense_names(tree):
    if not hasattr(tree, 'items'):
        return []
    names = []
    for name in tree:
        text = str(name)
        if text.startswith('Dense_') and text[6:].isdigit():
            names.append(name)
    return sorted(names, key=lambda name: int(str(name).split('_')[-1]))


def gciql_architecture_accounting(params, config, computation_reports=None):
    """Return a uniform per-slot architecture audit for GCIQL.

    The existing computation report intentionally contains only enabled
    computation slots.  M16A also needs canonical Flat rows, so this helper
    audits all three GCIQL modules and separates representation-body values
    from action/scalar readouts.  Structured MACs come from the token-aware
    report; Flat MACs come directly from the canonical Dense tree.
    """

    params = params if hasattr(params, 'items') else {}
    config = config if hasattr(config, 'get') else {}
    computation_reports = computation_reports if hasattr(computation_reports, 'get') else {}
    result = {}
    for slot_name in ('actor', 'value', 'critic'):
        module = params.get(f'modules_{slot_name}', {})
        spec = config.get('compute', {}).get(slot_name, {})
        structured = bool(spec.get('enabled', False)) and spec.get('structure', 'vector') == 'puzzle_tokens'
        if structured:
            slot_report = computation_reports.get(slot_name)
            if not slot_report:
                raise ValueError(f'Missing structured computation report for GCIQL.{slot_name}')
            body_params = int(slot_report['structured_body_params'])
            body_macs = int(slot_report['structured_body_dense_macs'])
            body_depth = int(slot_report['structured_sequential_depth'])
            body_unique = int(slot_report['unique_dense_layers'])
            body_executed = int(slot_report['executed_dense_layers'])
            readout = module.get('mean_net', {}) if slot_name == 'actor' else module.get('value_readout', {})
            structure_fields = {
                key: slot_report.get(key)
                for key in (
                    'num_tokens', 'token_dim', 'token_hidden_dim',
                    'channel_hidden_dim', 'num_mixer_blocks', 'readout',
                    'index_embedding', 'tm_mode',
                )
            }
        else:
            if slot_name == 'actor':
                body = module.get('actor_net', {})
                readout = module.get('mean_net', {})
            else:
                value_net = module.get('value_net', {})
                dense_names = _direct_dense_names(value_net)
                if not dense_names:
                    body = value_net
                    readout = {}
                else:
                    scalar_name = dense_names[-1]
                    body = {name: value_net[name] for name in value_net if name != scalar_name}
                    readout = value_net[scalar_name]
            body_params = count_parameters(body)
            body_macs = count_dense_macs(body)
            body_depth = len(_direct_dense_names(body))
            body_unique = count_dense_layers(body)
            body_executed = body_unique
            structure_fields = {
                'num_tokens': None,
                'token_dim': None,
                'token_hidden_dim': None,
                'channel_hidden_dim': None,
                'num_mixer_blocks': None,
                'readout': None,
                'index_embedding': False,
                'tm_mode': None,
            }
        readout_params = count_parameters(readout)
        readout_macs = count_dense_macs(readout)
        readout_layers = count_dense_layers(readout)
        result[slot_name] = {
            'slot_name': slot_name,
            'structure': 'puzzle_tokens' if structured else 'vector',
            'topology': 'feedforward',
            'block': 'mlp_mixer' if structured else 'plain',
            'trainable_params': count_parameters(module),
            'computation_body_params': body_params,
            'readout_params': readout_params,
            'computation_body_dense_macs': body_macs,
            'readout_dense_macs': readout_macs,
            'total_dense_macs': body_macs + readout_macs,
            'sequential_depth': body_depth + (1 if readout_layers else 0),
            'computation_body_sequential_depth': body_depth,
            'unique_dense_layers': body_unique + readout_layers,
            'executed_dense_layers': body_executed + readout_layers,
            **structure_fields,
        }
    return {
        'algorithm': 'gciql',
        'slots': result,
        'total_trainable_params': sum(item['trainable_params'] for item in result.values()),
        'total_dense_macs': sum(item['total_dense_macs'] for item in result.values()),
    }


def _module_subtrees(tree, prefix):
    """Collect setup-time tuple modules named ``<prefix>_<index>``."""

    if not hasattr(tree, 'items'):
        return {}
    direct = tree.get(prefix)
    if hasattr(direct, 'items'):
        return direct
    return {
        key: value
        for key, value in tree.items()
        if str(key).startswith(f'{prefix}_')
    }


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


def topology_dense_accounting(
    core,
    topology,
    topology_kwargs=None,
    *,
    parameter_sharing='shared',
    block='plain',
):
    """Account physical and executed Dense transformations from tree shape."""

    topology_kwargs = topology_kwargs if hasattr(topology_kwargs, 'get') else {}
    core = core if hasattr(core, 'items') else {}
    if topology == 'feedforward':
        body = _mapping_get(core, 'primitive', core)
        if block == 'residual':
            input_mapping = _mapping_get(body, 'input_mapping', {})
            residual_blocks = _module_subtrees(body, 'residual_blocks')
            unique = count_dense_layers(input_mapping) + sum(
                count_dense_layers(value) for value in residual_blocks.values()
            )
            macs = count_dense_macs(input_mapping) + sum(
                count_dense_macs(value) for value in residual_blocks.values()
            )
            return {
                'sequential_depth': int(unique),
                'unique_dense_layers': int(unique),
                'executed_dense_layers': int(unique),
                'actor_body_dense_macs': int(macs),
            }
        unique = count_dense_layers(body)
        return {
            'sequential_depth': int(unique),
            'unique_dense_layers': int(unique),
            'executed_dense_layers': int(unique),
            'actor_body_dense_macs': int(count_dense_macs(body)),
        }

    input_mapping = _mapping_get(core, 'input_mapping', {})
    input_layers = count_dense_layers(input_mapping)
    input_macs = count_dense_macs(input_mapping)
    if topology == 'single_state':
        iterations = int(topology_kwargs.get('iterations', 1))
        if parameter_sharing == 'shared':
            update = _mapping_get(core, 'update_module', {})
            unique = input_layers + count_dense_layers(update)
            executed = input_layers + iterations * count_dense_layers(update)
            macs = input_macs + iterations * count_dense_macs(update)
        elif parameter_sharing == 'untied':
            updates = _module_subtrees(core, 'update_modules')
            unique = input_layers + sum(count_dense_layers(value) for value in updates.values())
            executed = unique
            macs = input_macs + sum(count_dense_macs(value) for value in updates.values())
        else:
            raise ValueError(f'Unsupported SingleState parameter sharing: {parameter_sharing!r}')
        return {
            'sequential_depth': int(executed),
            'unique_dense_layers': int(unique),
            'executed_dense_layers': int(executed),
            'actor_body_dense_macs': int(macs),
        }
    if topology == 'two_state':
        h_cycles = int(topology_kwargs.get('h_cycles', 2))
        l_cycles = int(topology_kwargs.get('l_cycles', 1))
        h_update = _mapping_get(core, 'h_update', {})
        l_update = _mapping_get(core, 'l_update', {})
        h_layers = count_dense_layers(h_update)
        l_layers = count_dense_layers(l_update)
        executed = input_layers + h_cycles * h_layers + h_cycles * l_cycles * l_layers
        return {
            'sequential_depth': int(executed),
            'unique_dense_layers': int(input_layers + h_layers + l_layers),
            'executed_dense_layers': int(executed),
            'actor_body_dense_macs': int(
                input_macs
                + h_cycles * count_dense_macs(h_update)
                + h_cycles * l_cycles * count_dense_macs(l_update)
            ),
        }
    unique = count_dense_layers(core)
    return {
        'sequential_depth': int(unique),
        'unique_dense_layers': int(unique),
        'executed_dense_layers': int(unique),
        'actor_body_dense_macs': int(count_dense_macs(core)),
    }


def actor_slot_accounting(
    actor_params,
    buffer_params=None,
    *,
    topology=None,
    iterations=0,
    topology_kwargs=None,
    parameter_sharing='shared',
    block='plain',
):
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
    body_core = (
        _mapping_get(core, 'primitive', core)
        if topology == 'feedforward'
        else core
    )
    input_mapping = _mapping_get(body_core, 'input_mapping', {})
    update_module = _mapping_get(body_core, 'update_module', {})
    h_update = _mapping_get(body_core, 'h_update', {})
    l_update = _mapping_get(body_core, 'l_update', {})
    if topology == 'single_state':
        iterations = int(iterations)

    input_macs = count_dense_macs(input_mapping)
    update_per_execution = count_dense_macs(update_module)
    h_update_per_execution = count_dense_macs(h_update)
    l_update_per_execution = count_dense_macs(l_update)
    if topology == 'single_state' and parameter_sharing == 'shared':
        h_update_executions = 0
        l_update_executions = iterations
        update_executions = iterations
        total_update_macs = update_per_execution * iterations
    elif topology == 'single_state' and parameter_sharing == 'untied':
        update_modules = _module_subtrees(body_core, 'update_modules')
        h_update_executions = 0
        l_update_executions = 0
        update_executions = len(update_modules)
        total_update_macs = sum(count_dense_macs(value) for value in update_modules.values())
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

    topology_metrics = topology_dense_accounting(
        body_core,
        topology,
        topology_kwargs or ({'iterations': iterations} if topology == 'single_state' else {}),
        parameter_sharing=parameter_sharing,
        block=block,
    )
    core_macs = (
        topology_metrics['actor_body_dense_macs']
        if topology == 'feedforward'
        else input_macs + total_update_macs
    )
    readout_macs = count_dense_macs(_actor_readout_params(actor_params))
    if topology in ('single_state', 'two_state') or (
        topology == 'feedforward' and block == 'residual'
    ):
        full_actor_forward_macs = core_macs + readout_macs
    else:
        full_actor_forward_macs = count_dense_macs(actor_params)
    return {
        'topology': topology,
        'parameter_sharing': parameter_sharing if topology == 'single_state' else None,
        'block': block if topology == 'feedforward' else None,
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
        **topology_metrics,
        'trainable_params': count_parameters(actor_params),
        'core_trainable_params': count_parameters(body_core),
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
    structure='vector',
    structure_kwargs=None,
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
    body_core = (
        _mapping_get(core, 'primitive', core)
        if topology == 'feedforward'
        else core
    )
    if structure == 'puzzle_tokens':
        if topology != 'feedforward':
            raise ValueError('Puzzle structured accounting requires topology=feedforward')
        structured_metrics = structured_body_accounting(body_core, structure_kwargs)
    elif structure == 'vector':
        structured_metrics = {}
    else:
        raise ValueError(f'Unsupported computation structure for accounting: {structure!r}')
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
    layer_norm = None
    update_activate_final = None
    parameter_sharing = topology_kwargs.get('parameter_sharing', 'shared')
    block = topology_kwargs.get('block', 'plain')
    if is_recurrent:
        state_dim = int(topology_kwargs.get('state_dim'))
        update_depth = int(topology_kwargs.get('update_depth', 2))
        state_init = topology_kwargs.get('state_init', 'normal_buffer')
        state_init_std = float(topology_kwargs.get('state_init_std', 1.0))
        layer_norm = bool(topology_kwargs.get('layer_norm', False))
        update_activate_final = bool(topology_kwargs.get('update_activate_final', True))
        if topology == 'single_state':
            iterations = int(topology_kwargs.get('iterations', 1))
            residual = bool(topology_kwargs.get('residual', False))
            total_update_executions = iterations
        else:
            h_cycles = int(topology_kwargs.get('h_cycles', 2))
            l_cycles = int(topology_kwargs.get('l_cycles', 1))
            residual = False
            total_update_executions = h_cycles * (l_cycles + 1)

    input_mapping = _mapping_get(body_core, 'input_mapping', {})
    update_module = _mapping_get(body_core, 'update_module', {})
    h_update = _mapping_get(body_core, 'h_update', {})
    l_update = _mapping_get(body_core, 'l_update', {})
    topology_metrics = topology_dense_accounting(
        core,
        topology,
        topology_kwargs,
        parameter_sharing=parameter_sharing,
        block=block,
    )
    return {
        'slot_name': str(slot_name),
        'topology': topology,
        'primitive': primitive,
        'structure': structure,
        'structure_kwargs': dict(structure_kwargs or {}),
        'credit': credit,
        'parameter_sharing': parameter_sharing if topology == 'single_state' else None,
        'block': block if topology == 'feedforward' else None,
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
        'layer_norm': layer_norm,
        'update_activate_final': update_activate_final,
        'trainable_params': count_parameters(slot_params),
        'core_trainable_params': count_parameters(core),
        'buffer_elements': count_non_trainable(buffer_params),
        'core_buffer_elements': count_non_trainable(buffer_core),
        'input_mapping_params': count_parameters(input_mapping),
        'update_module_params': count_parameters(update_module),
        'h_update_params': count_parameters(h_update),
        'l_update_params': count_parameters(l_update),
        # Stable generic aliases for downstream milestone comparisons.  The
        # historical actor_body_dense_macs name is retained above.
        'dense_macs': int(topology_metrics.get('actor_body_dense_macs', 0)),
        'computation_dense_macs': int(topology_metrics.get('actor_body_dense_macs', 0)),
        **topology_metrics,
        **structured_metrics,
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
