"""Factory for the intentionally small first-stage computation framework."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Mapping, Optional, Sequence

from .credit.direct import DirectCredit
from .credit.full_bptt import FullBPTTCredit
from .credit.one_step import OneStepCredit
from .interfaces import ComputationCore
from .primitives.mlp import MLP
from .topologies.feedforward import FeedForward
from .topologies.single_state import SingleState
from .topologies.two_state import TwoState


@dataclass(frozen=True)
class ComputationSpec:
    """Static description of one computation slot."""

    primitive: str = 'mlp'
    topology: str = 'feedforward'
    credit: str = 'direct'
    topology_kwargs: Mapping = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Optional[Mapping] = None):
        if value is None:
            return cls()
        return cls(
            primitive=value.get('primitive', 'mlp'),
            topology=value.get('topology', 'feedforward'),
            credit=value.get('credit', 'direct'),
            topology_kwargs=dict(value.get('topology_kwargs', {})),
        )


def resolve_slot_spec(config: Optional[Mapping], slot_name: str):
    """Resolve one optional computation slot from an agent configuration.

    Slot resolution is shared across algorithms.  It only interprets the
    common ``compute.<slot_name>`` configuration and returns ``None`` for a
    disabled or absent slot; it does not construct a network or encode any
    algorithm-specific semantics.
    """

    compute = config.get('compute', {}) if config is not None else {}
    slot = compute.get(slot_name, {}) if compute is not None else {}
    if not slot or not slot.get('enabled', False):
        return None
    return ComputationSpec.from_mapping(slot)


def make_computation_core(
    spec: ComputationSpec,
    *,
    hidden_dims: Sequence[int],
    activate_final: bool = False,
    layer_norm: bool = False,
):
    """Build a computation core from a static slot specification.

    All branching happens while constructing the Flax module, not during a
    JAX-traced forward pass.
    """

    if not isinstance(spec, ComputationSpec):
        spec = ComputationSpec.from_mapping(spec)
    if spec.primitive not in ('mlp', 'original_mlp'):
        raise ValueError(f'Unsupported baseline primitive: {spec.primitive}')
    hidden_dims = tuple(hidden_dims)
    if not hidden_dims:
        raise ValueError('Recurrent computation cores require at least one hidden dimension')
    if any(isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0 for dim in hidden_dims):
        raise ValueError(f'Computation hidden dims must be positive integers, got {hidden_dims!r}')
    hidden_dims = tuple(int(dim) for dim in hidden_dims)
    if spec.topology == 'feedforward':
        if spec.credit != DirectCredit.name:
            raise ValueError(f'FeedForward requires credit={DirectCredit.name!r}, got {spec.credit!r}')
        primitive = MLP(
            hidden_dims=hidden_dims,
            activate_final=activate_final,
            layer_norm=layer_norm,
        )
        return ComputationCore(topology=FeedForward(primitive=primitive))

    if spec.topology == 'single_state':
        if spec.credit != DirectCredit.name:
            raise ValueError(f'SingleState requires credit={DirectCredit.name!r}, got {spec.credit!r}')
        kwargs = dict(spec.topology_kwargs)
        state_dim = int(kwargs.get('state_dim', hidden_dims[-1]))
        if state_dim != hidden_dims[-1]:
            raise ValueError(
                f'SingleState state_dim={state_dim} must match the final branch width '
                f'{hidden_dims[-1]}'
            )
        kwargs.setdefault('iterations', 1)
        kwargs.setdefault('residual', False)
        kwargs.setdefault('input_injection', 'z_plus_x')
        kwargs.setdefault('state_dim', state_dim)
        kwargs.setdefault('state_init', 'normal_buffer')
        kwargs.setdefault('state_init_std', 1.0)
        # Legacy actor configurations omit this field and therefore retain
        # the historical two-Dense update module.  Critic configurations can
        # explicitly request a deeper recurrent update while keeping the
        # caller's branch depth (for example, (512, 512, 512, 512)) intact.
        kwargs.setdefault('update_depth', 2)
        # M9's actor adapter/update use the existing MLP semantics and do not
        # add normalization or a new primitive recipe.
        kwargs['layer_norm'] = False
        return ComputationCore(topology=SingleState(**kwargs))

    if spec.topology == 'two_state':
        if spec.credit not in (FullBPTTCredit.name, OneStepCredit.name):
            raise ValueError(
                'TwoState requires credit in '
                f'{(FullBPTTCredit.name, OneStepCredit.name)!r}, got {spec.credit!r}'
            )
        kwargs = dict(spec.topology_kwargs)
        state_dim = int(kwargs.get('state_dim', hidden_dims[-1]))
        if state_dim != hidden_dims[-1]:
            raise ValueError(
                f'TwoState state_dim={state_dim} must match the final branch width '
                f'{hidden_dims[-1]}'
            )
        kwargs.setdefault('h_cycles', 2)
        kwargs.setdefault('l_cycles', 1)
        kwargs.setdefault('state_dim', state_dim)
        kwargs.setdefault('input_injection', 'l_receives_x')
        kwargs.setdefault('state_init', 'normal_buffer')
        kwargs.setdefault('state_init_std', 1.0)
        kwargs.setdefault('update_depth', 2)
        kwargs['credit'] = spec.credit
        # M9B deliberately keeps the existing GELU MLP semantics without
        # introducing normalization as a hidden scientific factor.
        kwargs['layer_norm'] = False
        return ComputationCore(topology=TwoState(**kwargs))

    raise ValueError(f'Unsupported computation topology: {spec.topology}')
