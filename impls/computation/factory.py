"""Factory for the intentionally small first-stage computation framework."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .credit.direct import DirectCredit
from .interfaces import ComputationCore
from .primitives.mlp import MLP
from .topologies.feedforward import FeedForward


@dataclass(frozen=True)
class ComputationSpec:
    """Static description of one computation slot."""

    primitive: str = 'mlp'
    topology: str = 'feedforward'
    credit: str = 'direct'

    @classmethod
    def from_mapping(cls, value: Optional[Mapping] = None):
        if value is None:
            return cls()
        return cls(
            primitive=value.get('primitive', 'mlp'),
            topology=value.get('topology', 'feedforward'),
            credit=value.get('credit', 'direct'),
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
    if spec.topology != 'feedforward':
        raise ValueError(f'Unsupported baseline topology: {spec.topology}')
    if spec.credit != DirectCredit.name:
        raise ValueError(f'Unsupported baseline credit policy: {spec.credit}')

    primitive = MLP(
        hidden_dims=tuple(hidden_dims),
        activate_final=activate_final,
        layer_norm=layer_norm,
    )
    return ComputationCore(topology=FeedForward(primitive=primitive))
