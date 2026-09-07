"""Factory for the intentionally small first-stage computation framework."""

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Mapping, Optional, Sequence

from .credit.direct import DirectCredit
from .credit.full_bptt import FullBPTTCredit
from .credit.one_step import OneStepCredit
from .blocks.entity_mlp import EntityMLPStack
from .blocks.mlp_mixer import MLPMixerStack
from .blocks.residual_mlp import ResidualMLPStack
from .interfaces import ComputationCore
from .primitives.mlp import MLP
from .readouts import HybridContextQueryReadout, MeanContextReadout
from .relation import RelationAugmenter
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
    readout: str = 'mean_context'
    readout_kwargs: Mapping = field(default_factory=dict)
    # Relations are first-class computation configuration rather than hidden
    # structure kwargs.  The defaults preserve every historical study.
    relation_mode: str = 'legacy_none'
    relation_kwargs: Mapping = field(default_factory=dict)
    relation_augmenter: str = 'none'
    relation_augmenter_kwargs: Mapping = field(default_factory=dict)
    # Set only by mainline Study resolution for a non-executable M20A Phase-2
    # skeleton.  It permits configuration inspection, never forward training
    # with unresolved Cube geometry.
    relation_phase2_blocked: bool = False

    @classmethod
    def from_mapping(cls, value: Optional[Mapping] = None):
        if value is None:
            return cls()
        structure_kwargs = dict(value.get('structure_kwargs', {}))
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
            structure_kwargs=structure_kwargs,
            input_semantics=value.get('input_semantics', 'latent_vector'),
            action_semantics=value.get('action_semantics', 'none'),
            # ``structure_kwargs.readout=mean`` is the frozen M15/M16 spelling.
            # Keep it as a compatible alias while the modular path records a
            # first-class readout choice.
            readout=value.get('readout', structure_kwargs.get('readout', 'mean_context')),
            readout_kwargs=dict(value.get('readout_kwargs', {})),
            relation_mode=value.get('relation_mode', 'legacy_none'),
            relation_kwargs=dict(value.get('relation_kwargs', {})),
            relation_augmenter=value.get('relation_augmenter', 'none'),
            relation_augmenter_kwargs=dict(value.get('relation_augmenter_kwargs', {})),
            relation_phase2_blocked=bool(value.get('relation_phase2_blocked', False)),
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


_ENTITY_STRUCTURE_KEYS = frozenset({
    'num_buttons',
    'robot_dim',
    'button_feature_dim',
    'token_dim',
    'robot_hidden_dim',
    'index_embedding',
})
_ENTITY_BLOCK_KEYS = frozenset({
    'num_blocks',
    # These spellings are accepted only as schema compatibility aliases.
    # M19A itself uses num_blocks/channel_hidden_dim.
    'num_mixer_blocks',
    'channel_hidden_dim',
    'channel_mlp_hidden_dim',
})
_ENTITY_TOKEN_MIXING_KEYS = frozenset({
    'token_hidden_dim',
    'token_mlp_hidden_dim',
    'tm_mode',
    'num_tokens',
    'hidden_dim_tokens',
})


def _entity_mlp_value(kwargs, names, label):
    """Resolve one explicitly supplied EntityMLP setting without fallback."""

    values = [(name, kwargs[name]) for name in names if name in kwargs]
    if not values:
        raise ValueError(f'EntityMLP requires block_kwargs.{label}')
    first = values[0][1]
    if any(value != first for _, value in values[1:]):
        raise ValueError(
            f'EntityMLP conflicting aliases for {label}: '
            f'{[(name, value) for name, value in values]!r}'
        )
    if isinstance(first, bool) or not isinstance(first, Integral) or first <= 0:
        raise ValueError(
            f'EntityMLP block_kwargs.{label} must be a positive integer, got {first!r}'
        )
    return int(first)


def _make_entity_mlp_puzzle_core(spec, *, hidden_dims, activate_final, layer_norm):
    """Construct the tightly scoped M19A EntityMLP structured path.

    This branch intentionally does not share a construction helper with the
    Mixer branch below. Keeping Mixer construction untouched protects its
    historical parameter tree and RNG split semantics.
    """

    if spec.credit != DirectCredit.name:
        raise ValueError(
            f'EntityMLP Puzzle computation requires credit={DirectCredit.name!r}, '
            f'got {spec.credit!r}'
        )
    if spec.topology != 'feedforward':
        raise ValueError(
            'EntityMLP Puzzle computation supports only topology=feedforward; '
            f'got {spec.topology!r}'
        )
    if spec.readout != 'mean_context':
        raise ValueError(
            'EntityMLP Puzzle computation requires readout=mean_context; '
            f'got {spec.readout!r}'
        )
    if spec.topology_kwargs:
        raise ValueError(
            'EntityMLP Puzzle computation does not accept topology_kwargs; '
            'recurrent state and topology modifiers are out of scope'
        )
    if spec.primitive not in ('mlp', 'original_mlp'):
        raise ValueError(f'Unsupported EntityMLP Puzzle primitive: {spec.primitive!r}')

    structure_kwargs = dict(spec.structure_kwargs)
    block_kwargs = dict(spec.block_kwargs)
    token_mixing_keys = (
        set(structure_kwargs) | set(block_kwargs)
    ) & _ENTITY_TOKEN_MIXING_KEYS
    if token_mixing_keys:
        raise ValueError(
            'EntityMLP does not accept token-mixing kwargs: '
            f'{sorted(token_mixing_keys)!r}'
        )
    unexpected_structure = set(structure_kwargs) - _ENTITY_STRUCTURE_KEYS
    if unexpected_structure:
        raise ValueError(
            'Unsupported EntityMLP structure_kwargs: '
            f'{sorted(unexpected_structure)!r}'
        )
    unexpected_block = set(block_kwargs) - _ENTITY_BLOCK_KEYS
    if unexpected_block:
        raise ValueError(
            'Unsupported EntityMLP block_kwargs: '
            f'{sorted(unexpected_block)!r}'
        )
    if 'num_buttons' not in structure_kwargs:
        raise ValueError('EntityMLP Puzzle computation requires structure_kwargs.num_buttons')

    num_blocks = _entity_mlp_value(
        block_kwargs, ('num_blocks', 'num_mixer_blocks'), 'num_blocks'
    )
    channel_hidden_dim = _entity_mlp_value(
        block_kwargs, ('channel_hidden_dim', 'channel_mlp_hidden_dim'),
        'channel_hidden_dim',
    )
    token_dim = int(structure_kwargs.get('token_dim', 128))
    if token_dim <= 0:
        raise ValueError(f'EntityMLP structure_kwargs.token_dim must be positive, got {token_dim!r}')

    from .structured import StructuredComputationBody
    from ..representation.puzzle import PuzzleTokenAdapter

    adapter = PuzzleTokenAdapter(
        num_buttons=int(structure_kwargs['num_buttons']),
        robot_dim=int(structure_kwargs.get('robot_dim', 19)),
        button_feature_dim=int(structure_kwargs.get('button_feature_dim', 4)),
        token_dim=token_dim,
        robot_hidden_dim=int(structure_kwargs.get('robot_hidden_dim', 128)),
        index_embedding=bool(structure_kwargs.get('index_embedding', True)),
        input_semantics=spec.input_semantics,
        action_semantics=spec.action_semantics,
        layer_norm=layer_norm,
    )
    block_unit = EntityMLPStack(
        num_blocks=num_blocks,
        embed_dim=token_dim,
        hidden_dim_channels=channel_hidden_dim,
    )
    structured_core = ComputationCore(topology=FeedForward(primitive=block_unit))
    readout_kwargs = dict(spec.readout_kwargs)
    requested_output_dim = int(readout_kwargs.pop('output_dim', hidden_dims[-1]))
    if requested_output_dim != hidden_dims[-1]:
        raise ValueError(
            'Structured readout output_dim must match the algorithm slot width; '
            f'got {requested_output_dim}, expected {hidden_dims[-1]}'
        )
    if readout_kwargs:
        raise ValueError(f'Unsupported mean_context readout kwargs: {sorted(readout_kwargs)!r}')
    return StructuredComputationBody(
        adapter=adapter,
        core=structured_core,
        readout=MeanContextReadout(
            output_dim=hidden_dims[-1],
            layer_norm=layer_norm,
            activate_final=True if activate_final is None else bool(activate_final),
        ),
    )


_M20_RELATION_STRUCTURES = frozenset({'puzzle_tokens', 'cube_tokens', 'scene_tokens'})
_M20_RELATION_MODES = frozenset({'zero', 'correct', 'shuffled'})


def _require_exact(name, actual, expected):
    if actual != expected:
        raise ValueError(f'M20A requires {name}={expected!r}, got {actual!r}')
    return actual


def _m20_positive_int(mapping, name):
    if name not in mapping:
        raise ValueError(f'M20A requires {name}')
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'M20A {name} must be a positive integer, got {value!r}')
    return int(value)


def _make_m20_relation_core(spec, *, hidden_dims, activate_final, layer_norm):
    """Construct the frozen M20A adapter -> relation -> Mixer -> readout path.

    This is deliberately separate from the historical Puzzle Mixer branch so
    M15--M19 parameter naming, initialization ordering, and numerical forward
    path remain untouched whenever ``relation_mode=legacy_none``.
    """

    if spec.structure not in _M20_RELATION_STRUCTURES:
        raise ValueError(f'Unsupported M20A relation structure: {spec.structure!r}')
    if spec.relation_mode not in _M20_RELATION_MODES:
        raise ValueError(
            'M20A relation treatments require relation_mode in '
            f'{sorted(_M20_RELATION_MODES)!r}, got {spec.relation_mode!r}'
        )
    _require_exact('credit', spec.credit, DirectCredit.name)
    _require_exact('topology', spec.topology, 'feedforward')
    _require_exact('block', spec.block, 'mlp_mixer')
    if spec.primitive not in ('mlp', 'original_mlp'):
        raise ValueError(f'Unsupported M20A primitive: {spec.primitive!r}')
    if spec.topology_kwargs:
        raise ValueError('M20A feedforward relation computation does not accept topology_kwargs')
    _require_exact('relation_augmenter', spec.relation_augmenter, 'relation_mlp')

    structure_kwargs = dict(spec.structure_kwargs)
    block_kwargs = dict(spec.block_kwargs)
    relation_kwargs = dict(spec.relation_kwargs)
    augmenter_kwargs = dict(spec.relation_augmenter_kwargs)
    readout_kwargs = dict(spec.readout_kwargs)

    token_dim = _m20_positive_int(structure_kwargs, 'token_dim')
    _require_exact('structure_kwargs.token_dim', token_dim, 128)
    robot_hidden_dim = _m20_positive_int(structure_kwargs, 'robot_hidden_dim')
    _require_exact('structure_kwargs.robot_hidden_dim', robot_hidden_dim, 128)
    num_blocks = _m20_positive_int(block_kwargs, 'num_blocks')
    token_hidden_dim = _m20_positive_int(block_kwargs, 'token_hidden_dim')
    channel_hidden_dim = _m20_positive_int(block_kwargs, 'channel_hidden_dim')
    _require_exact('block_kwargs.num_blocks', num_blocks, 2)
    _require_exact('block_kwargs.token_hidden_dim', token_hidden_dim, 64)
    _require_exact('block_kwargs.channel_hidden_dim', channel_hidden_dim, 256)
    _require_exact('block_kwargs.tm_mode', block_kwargs.get('tm_mode'), 'none')

    relation_hidden_dim = _m20_positive_int(augmenter_kwargs, 'relation_hidden_dim')
    _require_exact('relation_hidden_dim', relation_hidden_dim, 256)
    _require_exact('relation activation', augmenter_kwargs.get('activation'), 'gelu')
    _require_exact('relation first_use_bias', augmenter_kwargs.get('first_use_bias'), False)
    _require_exact('relation second_use_bias', augmenter_kwargs.get('second_use_bias'), False)
    _require_exact('relation normalization', augmenter_kwargs.get('normalization'), 'none')
    _require_exact('relation dropout', augmenter_kwargs.get('dropout'), 'none')
    _require_exact('relation output_dim', augmenter_kwargs.get('output_dim'), 128)

    requested_output_dim = int(readout_kwargs.pop('output_dim', hidden_dims[-1]))
    if requested_output_dim != hidden_dims[-1]:
        raise ValueError(
            'M20A structured readout output_dim must match the algorithm slot width; '
            f'got {requested_output_dim}, expected {hidden_dims[-1]}'
        )
    if spec.readout == 'mean_context':
        if readout_kwargs:
            raise ValueError(f'Unsupported M20A mean_context readout kwargs: {sorted(readout_kwargs)!r}')
        readout = MeanContextReadout(
            output_dim=hidden_dims[-1],
            layer_norm=layer_norm,
            activate_final=True if activate_final is None else bool(activate_final),
        )
    elif spec.readout == 'hybrid_context_query':
        query_dim = int(readout_kwargs.pop('query_dim', -1))
        _require_exact('HybridContextQuery query_dim', query_dim, 128)
        if readout_kwargs:
            raise ValueError(
                'Unsupported M20A hybrid_context_query readout kwargs: '
                f'{sorted(readout_kwargs)!r}'
            )
        readout = HybridContextQueryReadout(
            output_dim=hidden_dims[-1],
            token_dim=token_dim,
            query_dim=query_dim,
            layer_norm=layer_norm,
            activate_final=True if activate_final is None else bool(activate_final),
        )
    else:
        raise ValueError(
            'M20A readout must be mean_context or hybrid_context_query, '
            f'got {spec.readout!r}'
        )

    if spec.structure == 'puzzle_tokens':
        from ..representation.puzzle import PuzzleTokenAdapter

        num_buttons = _m20_positive_int(structure_kwargs, 'num_buttons')
        rows = _m20_positive_int(relation_kwargs, 'rows')
        cols = _m20_positive_int(relation_kwargs, 'cols')
        if rows * cols != num_buttons:
            raise ValueError(
                'M20A Puzzle relation grid must match num_buttons; '
                f'rows*cols={rows * cols}, num_buttons={num_buttons}'
            )
        _require_exact('Puzzle num_relation_types', relation_kwargs.get('num_relation_types'), 1)
        if 'shuffle_permutation' not in relation_kwargs:
            raise ValueError('M20A Puzzle relation config requires a fixed shuffle_permutation')
        adapter = PuzzleTokenAdapter(
            num_buttons=num_buttons,
            robot_dim=_m20_positive_int(structure_kwargs, 'robot_dim'),
            button_feature_dim=_m20_positive_int(structure_kwargs, 'button_feature_dim'),
            token_dim=token_dim,
            robot_hidden_dim=robot_hidden_dim,
            index_embedding=bool(structure_kwargs.get('index_embedding', True)),
            input_semantics=spec.input_semantics,
            action_semantics=spec.action_semantics,
            layer_norm=layer_norm,
            relation_mode=spec.relation_mode,
            relation_kwargs=relation_kwargs,
        )
        num_tokens = num_buttons
    elif spec.structure == 'cube_tokens':
        from ..representation.manipulation import CubeTokenAdapter

        num_cubes = _m20_positive_int(structure_kwargs, 'num_cubes')
        _require_exact('Cube num_cubes', num_cubes, 3)
        _require_exact('Cube cube_feature_dim', _m20_positive_int(structure_kwargs, 'cube_feature_dim'), 9)
        _require_exact(
            'Cube slot_identity_embedding',
            structure_kwargs.get('slot_identity_embedding'),
            False,
        )
        _require_exact('Cube num_relation_types', relation_kwargs.get('num_relation_types'), 3)
        _require_exact('Cube shuffle_derangement', tuple(relation_kwargs.get('shuffle_derangement', ())), (1, 2, 0))
        if spec.relation_mode in ('correct', 'shuffled'):
            missing = [
                key for key in (
                    'current_support_epsilon_xy', 'current_support_epsilon_z',
                    'goal_support_epsilon_xy', 'goal_support_epsilon_z',
                    'goal_conflict_radius',
                ) if relation_kwargs.get(key) is None
            ]
            if missing and not spec.relation_phase2_blocked:
                raise ValueError(
                    'M20A executable Cube Correct/Shuffled computation requires frozen thresholds; '
                    f'missing={missing!r}'
                )
        adapter = CubeTokenAdapter(
            num_cubes=num_cubes,
            robot_dim=_m20_positive_int(structure_kwargs, 'robot_dim'),
            cube_feature_dim=_m20_positive_int(structure_kwargs, 'cube_feature_dim'),
            token_dim=token_dim,
            robot_hidden_dim=robot_hidden_dim,
            slot_identity_embedding=bool(structure_kwargs.get('slot_identity_embedding', False)),
            input_semantics=spec.input_semantics,
            action_semantics=spec.action_semantics,
            layer_norm=layer_norm,
            relation_mode=spec.relation_mode,
            relation_kwargs=relation_kwargs,
        )
        num_tokens = num_cubes
    else:
        from ..representation.manipulation import SceneTokenAdapter

        _require_exact('Scene num_relation_types', relation_kwargs.get('num_relation_types'), 1)
        _require_exact(
            'Scene button_role_embedding',
            structure_kwargs.get('button_role_embedding'),
            True,
        )
        adapter = SceneTokenAdapter(
            robot_dim=_m20_positive_int(structure_kwargs, 'robot_dim'),
            cube_feature_dim=_m20_positive_int(structure_kwargs, 'cube_feature_dim'),
            button_feature_dim=_m20_positive_int(structure_kwargs, 'button_feature_dim'),
            drawer_feature_dim=_m20_positive_int(structure_kwargs, 'drawer_feature_dim'),
            window_feature_dim=_m20_positive_int(structure_kwargs, 'window_feature_dim'),
            token_dim=token_dim,
            robot_hidden_dim=robot_hidden_dim,
            button_role_embedding=bool(structure_kwargs.get('button_role_embedding', True)),
            input_semantics=spec.input_semantics,
            action_semantics=spec.action_semantics,
            layer_norm=layer_norm,
            relation_mode=spec.relation_mode,
            relation_kwargs=relation_kwargs,
        )
        num_tokens = 5

    from .structured import StructuredComputationBody

    mixer = MLPMixerStack(
        num_blocks=num_blocks,
        num_tokens=num_tokens,
        embed_dim=token_dim,
        hidden_dim_tokens=token_hidden_dim,
        hidden_dim_channels=channel_hidden_dim,
        tm_mode='none',
    )
    core = ComputationCore(topology=FeedForward(primitive=mixer))
    return StructuredComputationBody(
        adapter=adapter,
        relation_augmenter=RelationAugmenter(
            token_dim=token_dim,
            relation_hidden_dim=relation_hidden_dim,
        ),
        core=core,
        readout=readout,
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
    if spec.structure not in ('vector', 'puzzle_tokens', 'cube_tokens', 'scene_tokens'):
        raise ValueError(
            f'Unsupported computation structure: {spec.structure!r}; '
            "expected 'vector', 'puzzle_tokens', 'cube_tokens', or 'scene_tokens'"
        )
    if (
        spec.structure in _M20_RELATION_STRUCTURES
        and spec.relation_mode != 'legacy_none'
    ):
        return _make_m20_relation_core(
            spec,
            hidden_dims=hidden_dims,
            activate_final=activate_final,
            layer_norm=layer_norm,
        )
    if spec.structure in ('cube_tokens', 'scene_tokens'):
        raise ValueError(
            f'{spec.structure} requires an explicit M20A relation treatment; '
            'relation_mode=legacy_none is not a supported production path'
        )
    if spec.structure == 'puzzle_tokens':
        if spec.block == 'entity_mlp':
            return _make_entity_mlp_puzzle_core(
                spec,
                hidden_dims=hidden_dims,
                activate_final=activate_final,
                layer_norm=layer_norm,
            )
        if spec.credit != DirectCredit.name:
            raise ValueError(
                f'Puzzle token computation requires credit={DirectCredit.name!r}, '
                f'got {spec.credit!r}'
            )
        if spec.block != 'mlp_mixer':
            raise ValueError("Puzzle token computation requires block='mlp_mixer'")
        if spec.primitive not in ('mlp', 'original_mlp'):
            raise ValueError(f'Unsupported Puzzle token primitive: {spec.primitive!r}')
        if spec.topology not in ('feedforward', 'single_state'):
            raise ValueError(
                'Puzzle token computation supports topology=feedforward or single_state; '
                f'got {spec.topology!r}'
            )

        # Existing M15/M16 studies store Mixer dimensions under
        # structure_kwargs.  M17 permits their ownership to be recorded under
        # block_kwargs too, without forcing a meaningless historical rename.
        structure_kwargs = dict(spec.structure_kwargs)
        block_kwargs = dict(spec.block_kwargs)
        if 'num_buttons' not in structure_kwargs:
            raise ValueError('Puzzle token computation requires structure_kwargs.num_buttons')
        num_blocks = int(
            block_kwargs.get(
                'num_blocks',
                block_kwargs.get('num_mixer_blocks', structure_kwargs.get('num_mixer_blocks', 1)),
            )
        )
        token_dim = int(structure_kwargs.get('token_dim', 128))
        token_hidden_dim = int(
            block_kwargs.get(
                'token_hidden_dim',
                block_kwargs.get(
                    'token_mlp_hidden_dim',
                    structure_kwargs.get('token_mlp_hidden_dim', 64),
                ),
            )
        )
        channel_hidden_dim = int(
            block_kwargs.get(
                'channel_hidden_dim',
                block_kwargs.get(
                    'channel_mlp_hidden_dim',
                    structure_kwargs.get('channel_mlp_hidden_dim', 256),
                ),
            )
        )
        from .structured import StructuredComputationBody
        from ..representation.puzzle import PuzzleTokenAdapter

        adapter = PuzzleTokenAdapter(
            num_buttons=int(structure_kwargs['num_buttons']),
            robot_dim=int(structure_kwargs.get('robot_dim', 19)),
            button_feature_dim=int(structure_kwargs.get('button_feature_dim', 4)),
            token_dim=token_dim,
            robot_hidden_dim=int(structure_kwargs.get('robot_hidden_dim', 128)),
            index_embedding=bool(structure_kwargs.get('index_embedding', True)),
            input_semantics=spec.input_semantics,
            action_semantics=spec.action_semantics,
            layer_norm=layer_norm,
        )
        block_unit = MLPMixerStack(
            num_blocks=num_blocks,
            num_tokens=int(structure_kwargs['num_buttons']),
            embed_dim=token_dim,
            hidden_dim_tokens=token_hidden_dim,
            hidden_dim_channels=channel_hidden_dim,
            tm_mode=structure_kwargs.get('tm_mode', block_kwargs.get('tm_mode', 'none')),
        )
        if spec.topology == 'feedforward':
            structured_core = ComputationCore(topology=FeedForward(primitive=block_unit))
        else:
            topology_kwargs = dict(spec.topology_kwargs)
            input_mapping = topology_kwargs.pop('input_mapping', 'identity')
            if input_mapping != 'identity':
                raise ValueError(
                    'Structured SingleState requires input_mapping=identity to preserve '
                    'FeedForward(L) == SingleState(L, K=1)'
                )
            state_dim = int(topology_kwargs.pop('state_dim', token_dim))
            if state_dim != token_dim:
                raise ValueError(
                    'Structured SingleState state_dim must equal token_dim; '
                    f'got state_dim={state_dim}, token_dim={token_dim}'
                )
            residual = topology_kwargs.pop('residual', False)
            if residual is not False:
                raise ValueError('Structured SingleState freezes topology residual=False')
            input_injection = topology_kwargs.pop('input_injection', 'z_plus_x')
            if input_injection != 'z_plus_x':
                raise ValueError('Structured SingleState requires input_injection=z_plus_x')
            sharing = topology_kwargs.pop('parameter_sharing', spec.parameter_sharing)
            if sharing != 'shared' or spec.parameter_sharing != 'shared':
                raise ValueError('Structured SingleState requires parameter_sharing=shared')
            allowed = {'iterations', 'state_init', 'state_init_std'}
            unexpected = set(topology_kwargs) - allowed
            if unexpected:
                raise ValueError(
                    'Unsupported structured SingleState topology kwargs: '
                    f'{sorted(unexpected)!r}'
                )
            structured_core = ComputationCore(
                topology=SingleState(
                    state_dim=token_dim,
                    iterations=topology_kwargs.get('iterations', 1),
                    residual=False,
                    input_injection='z_plus_x',
                    state_init=topology_kwargs.get('state_init', 'zero_buffer'),
                    state_init_std=topology_kwargs.get('state_init_std', 1.0),
                    parameter_sharing='shared',
                    input_mapping_mode='identity',
                    external_update_block=block_unit,
                )
            )
        readout_name = spec.readout
        if readout_name not in ('mean', 'mean_context'):
            raise ValueError(
                'Puzzle token computation currently supports readout=mean_context '
                f'(legacy alias mean); got {readout_name!r}'
            )
        readout_kwargs = dict(spec.readout_kwargs)
        requested_output_dim = int(readout_kwargs.pop('output_dim', hidden_dims[-1]))
        if requested_output_dim != hidden_dims[-1]:
            raise ValueError(
                'Structured readout output_dim must match the algorithm slot width; '
                f'got {requested_output_dim}, expected {hidden_dims[-1]}'
            )
        if readout_kwargs:
            raise ValueError(f'Unsupported mean_context readout kwargs: {sorted(readout_kwargs)!r}')
        return StructuredComputationBody(
            adapter=adapter,
            core=structured_core,
            readout=MeanContextReadout(
                output_dim=hidden_dims[-1],
                layer_norm=layer_norm,
                activate_final=True if activate_final is None else bool(activate_final),
            ),
        )
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
