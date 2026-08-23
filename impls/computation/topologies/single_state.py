"""Decision-local single-state iterative computation topology."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from numbers import Integral

from ..interfaces import ComputationOutput
from ..primitives.mlp import MLP


class SingleState(nn.Module):
    """Run a decision-local state with shared or untied update parameters.

    The topology is intentionally generic: it receives only a tensor and does
    not know whether the tensor came from an observation, goal, or actor.
    ``z_init`` is a persistent, non-trainable buffer.  Each call broadcasts a
    fresh local copy and never mutates the stored buffer.
    """

    state_dim: int
    iterations: int
    residual: bool = False
    input_injection: str = 'z_plus_x'
    state_init: str = 'normal_buffer'
    state_init_std: float = 1.0
    update_depth: int = 2
    layer_norm: bool = False
    update_activate_final: bool = True
    parameter_sharing: str = 'shared'

    def setup(self):
        if self.state_dim <= 0:
            raise ValueError(f'state_dim must be positive, got {self.state_dim}')
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, Integral):
            raise ValueError(f'SingleState iterations must be an integer, got {self.iterations!r}')
        if self.iterations <= 0:
            raise ValueError(f'SingleState iterations must be positive, got {self.iterations}')
        if isinstance(self.update_depth, bool) or not isinstance(self.update_depth, Integral):
            raise ValueError(
                f'SingleState update_depth must be an integer, got {self.update_depth!r}'
            )
        if self.update_depth <= 0:
            raise ValueError(
                f'SingleState update_depth must be positive, got {self.update_depth}'
            )
        if self.input_injection != 'z_plus_x':
            raise ValueError(f'Unsupported SingleState input injection: {self.input_injection!r}')
        if self.state_init not in ('normal_buffer', 'zero_buffer'):
            raise ValueError(f'Unsupported SingleState state init: {self.state_init!r}')
        if self.state_init == 'normal_buffer' and self.state_init_std <= 0:
            raise ValueError(f'state_init_std must be positive, got {self.state_init_std}')
        if self.parameter_sharing not in ('shared', 'untied'):
            raise ValueError(
                f'Unsupported SingleState parameter sharing: {self.parameter_sharing!r}'
            )

        # D_in -> state_dim.  The default/shared path intentionally retains
        # the historical parameter subtree name ``update_module`` so old
        # M12A checkpoints remain restorable.  Untied is the same state graph
        # with a different parameter-tying schedule, not a new topology.
        self.input_mapping = MLP(
            hidden_dims=(self.state_dim,),
            activate_final=True,
            layer_norm=self.layer_norm,
        )
        if self.parameter_sharing == 'shared':
            self.update_module = MLP(
                hidden_dims=(self.state_dim,) * int(self.update_depth),
                activate_final=self.update_activate_final,
                layer_norm=self.layer_norm,
            )
        else:
            self.update_modules = tuple(
                MLP(
                    hidden_dims=(self.state_dim,) * int(self.update_depth),
                    activate_final=self.update_activate_final,
                    layer_norm=self.layer_norm,
                )
                for _ in range(int(self.iterations))
            )
        if self.state_init == 'zero_buffer':
            self.z_init = self.variable(
                'buffers',
                'z_init',
                lambda: jnp.zeros((self.state_dim,)),
            )
        else:
            self.z_init = self.variable(
                'buffers',
                'z_init',
                lambda: jax.random.normal(
                    self.make_rng('buffers'),
                    (self.state_dim,),
                ) * self.state_init_std,
            )

    def __call__(self, x_raw, state=None):
        if state is not None:
            raise ValueError('SingleState topology does not accept an external state; it is decision-local.')
        x_hidden = self.input_mapping(x_raw)
        if x_hidden.shape[-1] != self.state_dim:
            raise ValueError(
                f'SingleState input mapping produced {x_hidden.shape[-1]} features, '
                f'expected state_dim={self.state_dim}'
            )
        z_init = self.z_init.value
        z = jnp.broadcast_to(z_init, x_hidden.shape)
        if self.parameter_sharing == 'shared':
            update_modules = (self.update_module,) * self.iterations
        else:
            update_modules = self.update_modules
        for update_module in update_modules:
            update = update_module(z + x_hidden)
            if update.shape[-1] != self.state_dim:
                raise ValueError(
                    f'SingleState update produced {update.shape[-1]} features, '
                    f'expected state_dim={self.state_dim}'
                )
            z = z + update if self.residual else update
        return ComputationOutput(representation=z, state=z)
