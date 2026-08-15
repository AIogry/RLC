"""Small Flax utilities compatible with the OGBench baseline."""

import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


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


class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=None if tx is None else tx.init(params),
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        params = self.params if params is None else params
        method_name = getattr(self.model_def, method) if method is not None else None
        return self.apply_fn({'params': params}, *args, method=method_name, **kwargs)

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


def save_agent(agent, save_dir, epoch, checkpoint_metadata=None):
    """Serialize a complete agent PyTree, including optimizer and RNG state."""
    os.makedirs(save_dir, exist_ok=True)
    save_dict = {
        'agent': flax.serialization.to_state_dict(agent),
        'checkpoint_metadata': checkpoint_metadata,
    }
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(save_dict, file)
    print(f'Saved to {save_path}')
    return save_path


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
    with open(checkpoint_path, 'rb') as file:
        loaded = pickle.load(file)
    restored = flax.serialization.from_state_dict(agent, loaded['agent'])
    print(f'Restored from {checkpoint_path}')
    return restored
