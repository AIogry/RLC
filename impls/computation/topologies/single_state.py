"""Decision-local single-state iterative computation topology."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from numbers import Integral
from typing import Optional

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
    # The config-facing spelling remains ``input_mapping``.  This longer
    # field name leaves the legacy parameter subtree named ``input_mapping``.
    input_mapping_mode: str = 'mlp'
    # A structured caller may supply one generic update block (for example an
    # L-layer Mixer stack).  The shared path stores it under the historical
    # ``update_module`` subtree name, while its internal type is no longer
    # constrained to primitive MLP.
    external_update_block: Optional[nn.Module] = None

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
        if self.input_mapping_mode not in ('mlp', 'identity'):
            raise ValueError(
                'Unsupported SingleState input mapping: '
                f'{self.input_mapping_mode!r}; expected mlp or identity'
            )
        if self.state_init not in ('normal_buffer', 'zero_buffer'):
            raise ValueError(f'Unsupported SingleState state init: {self.state_init!r}')
        if self.state_init == 'normal_buffer' and self.state_init_std <= 0:
            raise ValueError(f'state_init_std must be positive, got {self.state_init_std}')
        if self.parameter_sharing not in ('shared', 'untied'):
            raise ValueError(
                f'Unsupported SingleState parameter sharing: {self.parameter_sharing!r}'
            )
        if self.external_update_block is not None and self.parameter_sharing != 'shared':
            raise ValueError(
                'An externally supplied SingleState update block requires '
                'parameter_sharing=shared; untied structured copies are not a recurrent block.'
            )

        # D_in -> state_dim.  The default/shared path intentionally retains
        # the historical parameter subtree name ``update_module`` so old
        # M12A checkpoints remain restorable.  Untied is the same state graph
        # with a different parameter-tying schedule, not a new topology.
        if self.input_mapping_mode == 'mlp':
            self.input_mapping = MLP(
                hidden_dims=(self.state_dim,),
                activate_final=True,
                layer_norm=self.layer_norm,
            )
        if self.parameter_sharing == 'shared':
            if self.external_update_block is None:
                self.update_module = MLP(
                    hidden_dims=(self.state_dim,) * int(self.update_depth),
                    activate_final=self.update_activate_final,
                    layer_norm=self.layer_norm,
                )
            else:
                # ``external_update_block`` is a configuration field, not a
                # semantic parameter owner.  Clone it under the stable module
                # name so the parameter tree exposes one shared update unit.
                self.update_module = self.external_update_block.clone(name='update_module')
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

    def step(self, z, x_hidden, update_module=None):
        """Apply one transition without depending on the total iteration budget.

        The topology residual is distinct from a block's own internal
        residuals.  Canonical structured recurrence sets this flag to False,
        while an MLP-Mixer block still retains its token/channel residuals.
        """

        if self.input_injection == 'z_plus_x':
            update_input = z + x_hidden
        else:  # Guarded in setup; retained for a future explicit extension.
            raise ValueError(f'Unsupported SingleState input injection: {self.input_injection!r}')
        update_module = self.update_module if update_module is None else update_module
        update = update_module(update_input)
        if update.shape[-1] != self.state_dim:
            raise ValueError(
                f'SingleState update produced {update.shape[-1]} features, '
                f'expected state_dim={self.state_dim}'
            )
        return z + update if self.residual else update

    def __call__(self, x_raw, state=None):
        if state is not None:
            raise ValueError('SingleState topology does not accept an external state; it is decision-local.')
        x_raw = jnp.asarray(x_raw)
        if self.input_mapping_mode == 'identity':
            x_hidden = x_raw
        else:
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
            z = self.step(z, x_hidden, update_module=update_module)
        return ComputationOutput(representation=z, state=z)
