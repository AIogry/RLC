"""Factory for the intentionally small first-stage computation framework."""

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Mapping, Optional, Sequence

from .credit.direct import DirectCredit
from .credit.full_bptt import FullBPTTCredit
from .credit.one_step import OneStepCredit
from .blocks.residual_mlp import ResidualMLPStack
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
    block: str = 'plain'
    parameter_sharing: str = 'shared'
    block_kwargs: Mapping = field(default_factory=dict)
    structure: str = 'vector'
    structure_kwargs: Mapping = field(default_factory=dict)
    input_semantics: str = 'latent_vector'
    action_semantics: str = 'none'

    @classmethod
    def from_mapping(cls, value: Optional[Mapping] = None):
        if value is None:
            return cls()
        return cls(
            primitive=value.get('primitive', 'mlp'),
            block=value.get('block', 'plain'),
            topology=value.get('topology', 'feedforward'),
            parameter_sharing=value.get(
                'parameter_sharing',
                value.get('topology_kwargs', {}).get('parameter_sharing', 'shared'),
            ),
            credit=value.get('credit', 'direct'),
            topology_kwargs=dict(value.get('topology_kwargs', {})),
            block_kwargs=dict(value.get('block_kwargs', {})),
            structure=value.get('structure', 'vector'),
            structure_kwargs=dict(value.get('structure_kwargs', {})),
            input_semantics=value.get('input_semantics', 'latent_vector'),
            action_semantics=value.get('action_semantics', 'none'),
        )


def resolve_slot_spec(config: Optional[Mapping], slot_name: str):
    """Resolve one optional computation slot from an agent configuration.

    Slot resolution is shared across algorithms.  It interprets the common
    ``compute.<slot_name>`` configuration and injects the slot's declarative
    input/action semantics; it returns ``None`` for a disabled or absent slot.
    """

    compute = config.get('compute', {}) if config is not None else {}
    slot = compute.get(slot_name, {}) if compute is not None else {}
    if not slot or not slot.get('enabled', False):
        return None
    spec = ComputationSpec.from_mapping(slot)
    # These semantics belong to the algorithm slot descriptor.  They are
    # injected centrally rather than exposed as user-configurable composition.
    agent_name = config.get('agent_name') if hasattr(config, 'get') else None
    if agent_name is None:
        # Standalone factory callers (including legacy parity tests) may
        # resolve a slot without an algorithm registry context.  Their
        # explicitly supplied semantics remain valid.
        return spec
    from .slots import descriptor_for
    descriptor = descriptor_for(agent_name, slot_name)
    return replace(
        spec,
        input_semantics=descriptor.input_semantics,
        action_semantics=descriptor.action_semantics,
    )


def make_computation_core(
    spec: ComputationSpec,
    *,
    hidden_dims: Sequence[int],
    activate_final: Optional[bool] = None,
    layer_norm: bool = False,
):
    """Build a computation core from a static slot specification.

    All branching happens while constructing the Flax module, not during a
    JAX-traced forward pass.
    """

    if not isinstance(spec, ComputationSpec):
        spec = ComputationSpec.from_mapping(spec)
    hidden_dims = tuple(hidden_dims)
    if not hidden_dims:
        raise ValueError('Recurrent computation cores require at least one hidden dimension')
    if any(isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0 for dim in hidden_dims):
        raise ValueError(f'Computation hidden dims must be positive integers, got {hidden_dims!r}')
    hidden_dims = tuple(int(dim) for dim in hidden_dims)
    if spec.structure not in ('vector', 'puzzle_tokens'):
        raise ValueError(
            f'Unsupported computation structure: {spec.structure!r}; '
            "expected 'vector' or 'puzzle_tokens'"
        )
    if spec.structure == 'puzzle_tokens':
        if spec.topology != 'feedforward':
            raise ValueError(
                'Puzzle token computation currently supports topology=feedforward only; '
                'token recurrence is deferred to a later milestone'
            )
        if spec.credit != DirectCredit.name:
            raise ValueError(
                f'Puzzle token computation requires credit={DirectCredit.name!r}, '
                f'got {spec.credit!r}'
            )
        if spec.block != 'mlp_mixer':
            raise ValueError("Puzzle token computation requires block='mlp_mixer'")
        from .structured import PuzzleStructuredBody
        kwargs = dict(spec.structure_kwargs)
        primitive = PuzzleStructuredBody(
            output_dim=hidden_dims[-1],
            input_semantics=spec.input_semantics,
            action_semantics=spec.action_semantics,
            layer_norm=layer_norm,
            activate_final=True if activate_final is None else bool(activate_final),
            **kwargs,
        )
        return ComputationCore(topology=FeedForward(primitive=primitive))
    if spec.primitive not in ('mlp', 'original_mlp'):
        raise ValueError(f'Unsupported baseline primitive: {spec.primitive}')
    # Historical direct recurrent-core callers used the actor/update MLP
    # recipe, whose update module ended with an activation.  Keep that
    # fallback while requiring network callers to pass the primitive semantics
    # explicitly (GCActor=True, CRL bilinear critic=False).  FeedForward keeps
    # its original no-final-activation default below.
    recurrent_activate_final = True if activate_final is None else bool(activate_final)
    feedforward_activate_final = False if activate_final is None else bool(activate_final)
    if spec.topology == 'feedforward':
        if spec.credit != DirectCredit.name:
            raise ValueError(f'FeedForward requires credit={DirectCredit.name!r}, got {spec.credit!r}')
        if spec.block == 'plain':
            primitive = MLP(
                hidden_dims=hidden_dims,
                activate_final=feedforward_activate_final,
                layer_norm=layer_norm,
            )
        elif spec.block == 'residual':
            kwargs = dict(spec.block_kwargs)
            state_dim = int(kwargs.get('state_dim', hidden_dims[-1]))
            if state_dim != hidden_dims[-1]:
                raise ValueError(
                    f'ResidualMLPStack state_dim={state_dim} must match the final branch width '
                    f'{hidden_dims[-1]}'
                )
            primitive = ResidualMLPStack(
                state_dim=state_dim,
                blocks=int(kwargs.get('blocks', 4)),
                block_depth=int(kwargs.get('block_depth', 2)),
                layer_norm=bool(kwargs.get('layer_norm', layer_norm)),
                block_activate_final=bool(
                    kwargs.get('block_activate_final', feedforward_activate_final)
                ),
            )
        else:
            raise ValueError(f'Unsupported FeedForward block: {spec.block!r}')
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
        kwargs.setdefault('parameter_sharing', spec.parameter_sharing)
        # The caller owns primitive semantics. Actor callers pass
        # activate_final=True/layer_norm=False; CRL bilinear critic callers
        # pass activate_final=False/layer_norm=True, matching the replaced
        # vanilla branch.
        kwargs['layer_norm'] = bool(layer_norm)
        kwargs['update_activate_final'] = recurrent_activate_final
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
        # Preserve caller primitive semantics while letting the topology own
        # only the H/L execution schedule.
        kwargs['layer_norm'] = bool(layer_norm)
        kwargs['update_activate_final'] = recurrent_activate_final
        return ComputationCore(topology=TwoState(**kwargs))

    raise ValueError(f'Unsupported computation topology: {spec.topology}')
