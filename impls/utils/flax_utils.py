"""Small Flax utilities compatible with the OGBench baseline."""

import functools
import glob
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze
from flax.core.frozen_dict import FrozenDict

from .checkpointing import (
    parameter_module_key,
    resolve_checkpoint,
    sha256_file,
    should_update_best,
    write_checkpoint_index,
    write_checkpoint_metadata,
)


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(f'Expected module args {self.modules.keys()}, got {kwargs.keys()}')
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)

    def diagnostic_trace(self, *args, name=None, **kwargs):
        """Route an explicit diagnostic-only trace request to one submodule.

        Normal network calls continue through ``__call__``.  Keeping this
        routing method on ModuleDict lets a TrainState apply one restored
        module's diagnostic helper without constructing an alternate parameter
        tree or exposing trace state to the training graph.
        """

        if name is None or name not in self.modules:
            raise ValueError(f'Unknown diagnostic module name: {name!r}')
        trace_fn = getattr(self.modules[name], 'diagnostic_trace', None)
        if trace_fn is None:
            raise ValueError(f'Module {name!r} does not expose diagnostic_trace')
        return trace_fn(*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    model_state: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, model_state=None, **kwargs):
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            model_state={} if model_state is None else model_state,
            tx=tx,
            opt_state=None if tx is None else tx.init(params),
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        params = self.params if params is None else params
        method_name = getattr(self.model_def, method) if method is not None else None
        variables = {'params': params}
        if self.model_state:
            variables.update(self.model_state)
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        leaves = jax.tree_util.tree_leaves(grads)
        flat = jnp.concatenate([jnp.reshape(x, -1) for x in leaves])
        info = dict(info)
        info.update({'grad/max': jnp.max(flat), 'grad/min': jnp.min(flat), 'grad/norm': jnp.linalg.norm(flat, ord=1)})
        return self.apply_gradients(grads), info


def synchronize_target_module(network, module_name):
    """Copy an online module's params and non-trainable state to its target.

    Target modules are architectural copies, not independent computation
    slots.  This helper is used at initialization so recurrent buffers are
    exactly equal as well as parameters.  Subsequent Polyak updates in the
    canonical agents intentionally touch parameters only.
    """

    source_key = parameter_module_key(network.params, module_name)
    target_key = parameter_module_key(network.params, f'target_{module_name}')
    params = dict(network.params)
    params[target_key] = jax.tree_util.tree_map(
        lambda value: jnp.array(value), params[source_key]
    )

    model_state = network.model_state
    if model_state and isinstance(model_state, Mapping):
        model_state = dict(model_state)
        collection = model_state.get('buffers')
        if collection is not None and isinstance(collection, Mapping):
            collection = dict(collection)
            source_buffer_key = f'modules_{module_name}'
            target_buffer_key = f'modules_target_{module_name}'
            if source_buffer_key in collection and target_buffer_key in collection:
                collection[target_buffer_key] = jax.tree_util.tree_map(
                    lambda value: jnp.array(value), collection[source_buffer_key]
                )
            model_state['buffers'] = collection
    return network.replace(params=params, model_state=model_state)


def _write_checkpoint(agent, checkpoint_path, checkpoint_metadata=None):
    """Serialize a complete agent PyTree, including optimizer and RNG state."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'agent': flax.serialization.to_state_dict(agent),
        'checkpoint_metadata': checkpoint_metadata,
    }
    with checkpoint_path.open('wb') as file:
        pickle.dump(save_dict, file)
    print(f'Saved to {checkpoint_path}')
    return str(checkpoint_path)


def save_agent(agent, save_dir, epoch, checkpoint_metadata=None):
    """Serialize a complete agent PyTree to the legacy numeric layout."""

    return _write_checkpoint(
        agent,
        Path(save_dir) / f'params_{epoch}.pkl',
        checkpoint_metadata=checkpoint_metadata,
    )


def save_semantic_checkpoint(agent, run_dir, role, step, checkpoint_metadata=None):
    """Save an independent, portable ``best`` or ``last`` checkpoint artifact."""

    if role not in {'best', 'last'}:
        raise ValueError(f'Unsupported semantic checkpoint role: {role!r}')
    run_dir = Path(run_dir)
    role_dir = run_dir / 'checkpoints' / role
    role_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = role_dir / f'params_{int(step)}.pkl'
    for previous in role_dir.glob('params_*.pkl'):
        if previous != checkpoint_path:
            previous.unlink()
    metadata = dict(checkpoint_metadata or {})
    metadata.update({
        'checkpoint_role': role,
        'checkpoint_step': int(step),
    })
    _write_checkpoint(agent, checkpoint_path, checkpoint_metadata=metadata)
    checkpoint_sha = sha256_file(checkpoint_path)
    record = {
        **metadata,
        'path': str(checkpoint_path.relative_to(run_dir)),
        'sha256': checkpoint_sha,
        'metadata_path': str((role_dir / 'checkpoint.json').relative_to(run_dir)),
        'checkpoint_sha256': checkpoint_sha,
    }
    write_checkpoint_metadata(role_dir / 'checkpoint.json', record)
    return record


def restore_agent_from_checkpoint(agent, checkpoint_path):
    """Restore an agent from one already-resolved checkpoint file."""

    with open(checkpoint_path, 'rb') as file:
        loaded = pickle.load(file)
    agent_state = loaded['agent']
    # Checkpoints created before M9 had no TrainState.model_state field.
    # Preserve their baseline restore compatibility with an empty collection.
    network_state = agent_state.get('network') if isinstance(agent_state, Mapping) else None
    if isinstance(network_state, Mapping) and 'model_state' not in network_state:
        agent_state = dict(agent_state)
        agent_state['network'] = dict(network_state)
        agent_state['network']['model_state'] = {}
    restored = flax.serialization.from_state_dict(agent, agent_state)
    print(f'Restored from {checkpoint_path}')
    return restored


def _mapping_like(template, values):
    """Rebuild a mapping with the same container class as ``template``."""

    if isinstance(template, FrozenDict):
        return freeze(values)
    if isinstance(template, dict):
        return dict(values)
    try:
        return type(template)(values)
    except TypeError as error:
        raise TypeError(
            f'Cannot preserve parameter mapping type {type(template)!r}'
        ) from error


def _coerce_subtree_like(source, template, path=()):
    """Convert a source subtree to the target's recursive mapping structure."""

    if isinstance(template, Mapping):
        if not isinstance(source, Mapping):
            raise ValueError(
                f'Incompatible parameter subtree at {path!r}: '
                f'source={type(source)!r}, target={type(template)!r}'
            )
        if set(source) != set(template):
            raise ValueError(
                f'Incompatible parameter keys at {path!r}: '
                f'source={sorted(source)!r}, target={sorted(template)!r}'
            )
        values = {
            key: _coerce_subtree_like(source[key], template[key], path + (key,))
            for key in template
        }
        return _mapping_like(template, values)
    if isinstance(source, Mapping):
        raise ValueError(
            f'Incompatible parameter subtree at {path!r}: '
            f'source={type(source)!r}, target={type(template)!r}'
        )
    return source


def restore_module_from_checkpoint(agent, checkpoint_path, module_name):
    """Restore one module while preserving target params and fresh optimizer state."""

    with open(checkpoint_path, 'rb') as file:
        loaded = pickle.load(file)
    try:
        source_params = loaded['agent']['network']['params']
        source_key = parameter_module_key(source_params, module_name)
        source_module = source_params[source_key]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f'Checkpoint does not contain network.params[{module_name!r}]'
        ) from error
    params_before = agent.network.params
    target_key = parameter_module_key(params_before, module_name)
    target_module = params_before[target_key]
    source_module = _coerce_subtree_like(source_module, target_module)
    source_leaves = jax.tree_util.tree_leaves(source_module)
    target_leaves = jax.tree_util.tree_leaves(target_module)
    if [getattr(leaf, 'shape', None) for leaf in source_leaves] != [
        getattr(leaf, 'shape', None) for leaf in target_leaves
    ]:
        raise ValueError(f'Incompatible parameter shapes for module {module_name!r}')
    values = dict(params_before)
    values[target_key] = source_module
    params_after = _mapping_like(params_before, values)
    if jax.tree_util.tree_structure(params_before) != jax.tree_util.tree_structure(params_after):
        raise ValueError(
            'Parameter PyTree structure changed while restoring '
            f'module {module_name!r}'
        )
    # Deliberately keep the target TrainState optimizer/opt_state untouched;
    # source checkpoints contribute parameters only, never optimizer momentum.
    network = agent.network.replace(params=params_after)
    return agent.replace(network=network)


def restore_agent(agent, restore_path, restore_epoch):
    """Restore an agent from ``restore_path/params_<epoch>.pkl``."""

    candidates = glob.glob(restore_path)
    if len(candidates) != 1:
        raise ValueError(f'Expected one checkpoint directory, found {len(candidates)}: {candidates}')
    checkpoint_dir = candidates[0]
    canonical_checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
    if os.path.isdir(canonical_checkpoint_dir):
        checkpoint_dir = canonical_checkpoint_dir
    checkpoint_path = os.path.join(checkpoint_dir, f'params_{restore_epoch}.pkl')
    return restore_agent_from_checkpoint(agent, checkpoint_path)
