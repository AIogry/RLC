"""Decision-local two-state hierarchical computation topology."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..credit.full_bptt import FullBPTTCredit
from ..credit.one_step import OneStepCredit
from ..interfaces import ComputationOutput
from ..primitives.mlp import MLP


_CREDIT_POLICIES = {
    FullBPTTCredit.name: FullBPTTCredit,
    OneStepCredit.name: OneStepCredit,
}


def execution_trace(h_cycles, l_cycles):
    """Return the exact H/L execution order for one TwoState call."""

    return tuple(
        level
        for _ in range(h_cycles)
        for level in ('L',) * l_cycles + ('H',)
    )


class TwoState(nn.Module):
    """Run independent shared H/L MLP updates over two local states.

    The input is injected directly into the L update only.  Both initial
    states are persistent non-parameter buffers, but each call starts from a
    fresh broadcast copy and never writes a final state back to those buffers.
    """

    state_dim: int
    h_cycles: int
    l_cycles: int
    credit: str = 'full_bptt'
    input_injection: str = 'l_receives_x'
    state_init: str = 'normal_buffer'
    state_init_std: float = 1.0
    layer_norm: bool = False

    def setup(self):
        if self.state_dim <= 0:
            raise ValueError(f'state_dim must be positive, got {self.state_dim}')
        if (self.h_cycles, self.l_cycles) not in ((2, 1), (2, 6)):
            raise ValueError(
                'M9B TwoState schedules must be (h_cycles, l_cycles) in '
                f'((2, 1), (2, 6)), got {(self.h_cycles, self.l_cycles)!r}'
            )
        if self.credit not in _CREDIT_POLICIES:
            raise ValueError(f'Unsupported TwoState credit policy: {self.credit!r}')
        if self.input_injection != 'l_receives_x':
            raise ValueError(f'Unsupported TwoState input injection: {self.input_injection!r}')
        if self.state_init != 'normal_buffer':
            raise ValueError(f'Unsupported TwoState state init: {self.state_init!r}')
        if self.state_init_std <= 0:
            raise ValueError(f'state_init_std must be positive, got {self.state_init_std}')

        self.input_mapping = MLP(
            hidden_dims=(self.state_dim,),
            activate_final=True,
            layer_norm=self.layer_norm,
        )
        self.h_update = MLP(
            hidden_dims=(self.state_dim, self.state_dim),
            activate_final=True,
            layer_norm=self.layer_norm,
        )
        self.l_update = MLP(
            hidden_dims=(self.state_dim, self.state_dim),
            activate_final=True,
            layer_norm=self.layer_norm,
        )

        self.z_h_init = self.variable(
            'buffers',
            'z_h_init',
            lambda: jax.random.normal(
                self.make_rng('buffers'),
                (self.state_dim,),
            ) * self.state_init_std,
        )
        self.z_l_init = self.variable(
            'buffers',
            'z_l_init',
            lambda: jax.random.normal(
                self.make_rng('buffers'),
                (self.state_dim,),
            ) * self.state_init_std,
        )

    def __call__(self, x_raw, state=None):
        if state is not None:
            raise ValueError('TwoState topology does not accept an external state; it is decision-local.')

        x_hidden = self.input_mapping(x_raw)
        if x_hidden.shape[-1] != self.state_dim:
            raise ValueError(
                f'TwoState input mapping produced {x_hidden.shape[-1]} features, '
                f'expected state_dim={self.state_dim}'
            )

        z_h = jnp.broadcast_to(self.z_h_init.value, x_hidden.shape)
        z_l = jnp.broadcast_to(self.z_l_init.value, x_hidden.shape)
        trace = execution_trace(self.h_cycles, self.l_cycles)
        final_pair_start = len(trace) - 2
        credit_policy = _CREDIT_POLICIES[self.credit]

        for index, level in enumerate(trace):
            if self.credit == OneStepCredit.name and index == final_pair_start:
                z_h = credit_policy.prepare_final_state(z_h)
                z_l = credit_policy.prepare_final_state(z_l)
            if level == 'L':
                z_l = self.l_update(z_l + z_h + x_hidden)
            else:
                z_h = self.h_update(z_h + z_l)

        return ComputationOutput(
            representation=z_h,
            state={'z_h': z_h, 'z_l': z_l},
        )
