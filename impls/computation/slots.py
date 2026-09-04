"""Declarative computation-slot ontology for canonical RLC agents.

The registry is deliberately independent of any concrete Flax module.  It
describes where an algorithm exposes a replaceable vector computation body,
which configuration dimension supplies its width, and which primitive
semantics belong to that role.  Agents use it for validation while runtime
accounting uses it to resolve parameter and buffer paths.
"""

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class ComputationSlotDescriptor:
    """Static description of one replaceable algorithm computation slot."""

    slot_name: str
    module_path: Tuple[str, ...]
    core_path: Tuple[str, ...]
    role: str
    hidden_dims_source: str
    state_dim_source: str
    layer_norm_semantics: str
    activate_final_semantics: str
    output_dim_source: str = ''
    input_semantics: str = 'latent_vector'
    action_semantics: str = 'none'


def _descriptor(
    slot_name,
    module_path,
    core_path,
    role,
    hidden_dims_source,
    state_dim_source,
    layer_norm_semantics,
    activate_final_semantics,
    output_dim_source=None,
    input_semantics='latent_vector',
    action_semantics='none',
):
    return ComputationSlotDescriptor(
        slot_name=slot_name,
        module_path=tuple(module_path),
        core_path=tuple(core_path),
        role=role,
        hidden_dims_source=hidden_dims_source,
        state_dim_source=state_dim_source,
        layer_norm_semantics=layer_norm_semantics,
        activate_final_semantics=activate_final_semantics,
        output_dim_source=output_dim_source or state_dim_source,
        input_semantics=input_semantics,
        action_semantics=action_semantics,
    )


def _actor(slot_name, module_name):
    return _descriptor(
        slot_name,
        (f'modules_{module_name}',),
        ('actor_net', 'topology'),
        'actor',
        'actor_hidden_dims',
        'actor_hidden_dims[-1]',
        'false',
        'true',
        input_semantics='goal_pair',
    )


def _value(slot_name, module_name, role='value', action_semantics='none'):
    return _descriptor(
        slot_name,
        (f'modules_{module_name}',),
        ('value_net', 'core', 'topology'),
        role,
        'value_hidden_dims',
        'value_hidden_dims[-1]',
        'config.layer_norm',
        'true',
        input_semantics='goal_pair',
        action_semantics=action_semantics,
    )


SLOT_DESCRIPTORS = {
    # M14 canonical algorithms.
    'gcbc': {
        'actor': _actor('actor', 'actor'),
    },
    'gciql': {
        'actor': _actor('actor', 'actor'),
        'value': _value('value', 'value'),
        'critic': _value(
            'critic', 'critic', role='critic', action_semantics='robot_context'
        ),
    },
    'gcivl': {
        'actor': _actor('actor', 'actor'),
        'value': _value('value', 'value'),
    },
    'qrl': {
        'actor': _actor('actor', 'actor'),
        'value': _descriptor(
            'value',
            ('modules_value',),
            ('phi', 'core', 'topology'),
            'value',
            '(*value_hidden_dims, latent_dim)',
            'latent_dim',
            'config.layer_norm',
            'false',
            input_semantics='single_observation',
        ),
        'dynamics': _descriptor(
            'dynamics',
            ('modules_dynamics',),
            ('core', 'topology'),
            'dynamics',
            '(*value_hidden_dims, latent_dim)',
            'latent_dim',
            'config.layer_norm',
            'false',
            input_semantics='latent_vector',
            action_semantics='latent_dynamics',
        ),
    },
    # Existing M9/M11 computation users.  Keeping them in the same registry
    # lets accounting and runtime metadata use one ontology across milestones.
    'hiql': {
        'low_actor': _actor('low_actor', 'low_actor'),
        'high_actor': _actor('high_actor', 'high_actor'),
        'value': _value('value', 'value'),
    },
    'crl': {
        'actor': _actor('actor', 'actor'),
        'critic_state': _descriptor(
            'critic_state', ('modules_critic', 'phi'), ('core', 'topology'),
            'critic', '(*value_hidden_dims, latent_dim)', 'latent_dim',
            'config.layer_norm', 'false',
        ),
        'critic_goal': _descriptor(
            'critic_goal', ('modules_critic', 'psi'), ('core', 'topology'),
            'critic', '(*value_hidden_dims, latent_dim)', 'latent_dim',
            'config.layer_norm', 'false',
        ),
        'value_state': _descriptor(
            'value_state', ('modules_value', 'phi'), ('core', 'topology'),
            'value', '(*value_hidden_dims, latent_dim)', 'latent_dim',
            'config.layer_norm', 'false',
        ),
        'value_goal': _descriptor(
            'value_goal', ('modules_value', 'psi'), ('core', 'topology'),
            'value', '(*value_hidden_dims, latent_dim)', 'latent_dim',
            'config.layer_norm', 'false',
        ),
    },
}


def descriptors_for(agent_name: str):
    """Return the supported slot descriptors for one agent."""

    return SLOT_DESCRIPTORS.get(agent_name, {})


def descriptor_for(agent_name: str, slot_name: str):
    """Return one descriptor or raise a descriptive error."""

    descriptors = descriptors_for(agent_name)
    if slot_name not in descriptors:
        supported = ', '.join(sorted(descriptors)) or '<none>'
        raise ValueError(
            f'Unsupported computation slot {slot_name!r} for {agent_name!r}; '
            f'supported slots: {supported}'
        )
    return descriptors[slot_name]


def validate_compute_slots(agent_name: str, config: Mapping):
    """Validate every configured slot before an agent resolves any module.

    Disabled slots are validated too: a typo must not become a silently
    ignored no-op.  Detailed topology/credit validation remains in the
    computation factory once an enabled slot is materialized.
    """

    descriptors = descriptors_for(agent_name)
    compute = config.get('compute', {}) if config is not None else {}
    if compute is None:
        return
    for slot_name, slot in compute.items():
        if slot_name not in descriptors:
            supported = ', '.join(sorted(descriptors)) or '<none>'
            raise ValueError(
                f'Unsupported computation slot {slot_name!r} for {agent_name!r}; '
                f'supported slots: {supported}'
            )
        if not hasattr(slot, 'get'):
            raise ValueError(
                f'Computation slot {agent_name}.{slot_name} must be a mapping'
            )
        if slot.get('enabled', False) and not slot.get('topology'):
            raise ValueError(
                f'Enabled computation slot {agent_name}.{slot_name} must specify topology'
            )
        if slot.get('enabled', False):
            structure = slot.get('structure', 'vector')
            if structure not in ('vector', 'puzzle_tokens'):
                raise ValueError(
                    f'Unsupported computation structure {structure!r} for '
                    f'{agent_name}.{slot_name}'
                )
            if structure == 'puzzle_tokens':
                block = slot.get('block', 'plain')
                if block == 'entity_mlp':
                    # M19A is deliberately a narrowly scoped control.  Do
                    # not accept recurrent, alternate-readout, or legacy
                    # token-mixing variants and silently reinterpret them.
                    if slot.get('topology') != 'feedforward':
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'requires topology=feedforward'
                        )
                    if slot.get('credit', 'direct') != 'direct':
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'requires credit=direct'
                        )
                    structure_kwargs = slot.get('structure_kwargs', {})
                    if not hasattr(structure_kwargs, 'get'):
                        raise ValueError(
                            f'EntityMLP structure_kwargs for {agent_name}.{slot_name} '
                            'must be a mapping'
                        )
                    if 'num_buttons' not in structure_kwargs:
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'requires structure_kwargs.num_buttons'
                        )
                    readout = slot.get('readout', structure_kwargs.get('readout', 'mean_context'))
                    if readout != 'mean_context':
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'requires readout=mean_context'
                        )
                    topology_kwargs = slot.get('topology_kwargs', {})
                    if not hasattr(topology_kwargs, 'get') or topology_kwargs:
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'does not permit topology_kwargs'
                        )
                    block_kwargs = slot.get('block_kwargs', {})
                    if not hasattr(block_kwargs, 'get'):
                        raise ValueError(
                            f'EntityMLP block_kwargs for {agent_name}.{slot_name} '
                            'must be a mapping'
                        )
                    token_mixing_keys = (
                        set(structure_kwargs) | set(block_kwargs)
                    ) & {
                        'token_hidden_dim', 'token_mlp_hidden_dim', 'tm_mode',
                        'num_tokens', 'hidden_dim_tokens',
                    }
                    if token_mixing_keys:
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            f'cannot include token-mixing kwargs {sorted(token_mixing_keys)!r}'
                        )
                    unexpected_structure = set(structure_kwargs) - {
                        'num_buttons', 'robot_dim', 'button_feature_dim', 'token_dim',
                        'robot_hidden_dim', 'index_embedding',
                    }
                    if unexpected_structure:
                        raise ValueError(
                            f'EntityMLP structure_kwargs for {agent_name}.{slot_name} '
                            f'contain unsupported keys {sorted(unexpected_structure)!r}'
                        )
                    unexpected_block = set(block_kwargs) - {
                        'num_blocks', 'num_mixer_blocks', 'channel_hidden_dim',
                        'channel_mlp_hidden_dim',
                    }
                    if unexpected_block:
                        raise ValueError(
                            f'EntityMLP block_kwargs for {agent_name}.{slot_name} '
                            f'contain unsupported keys {sorted(unexpected_block)!r}'
                        )
                    if not any(name in block_kwargs for name in ('num_blocks', 'num_mixer_blocks')):
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'requires block_kwargs.num_blocks'
                        )
                    if not any(
                        name in block_kwargs
                        for name in ('channel_hidden_dim', 'channel_mlp_hidden_dim')
                    ):
                        raise ValueError(
                            f'EntityMLP Puzzle computation for {agent_name}.{slot_name} '
                            'requires block_kwargs.channel_hidden_dim'
                        )
                    continue
                topology = slot.get('topology')
                if topology not in ('feedforward', 'single_state'):
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        'requires topology=feedforward or single_state'
                    )
                if slot.get('credit', 'direct') != 'direct':
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        'requires credit=direct'
                    )
                if block != 'mlp_mixer':
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        "requires block='mlp_mixer'"
                    )
                structure_kwargs = slot.get('structure_kwargs', {})
                if not hasattr(structure_kwargs, 'get'):
                    raise ValueError(
                        f'Computation structure_kwargs for {agent_name}.{slot_name} '
                        'must be a mapping'
                    )
                if 'num_buttons' not in structure_kwargs:
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        'requires structure_kwargs.num_buttons'
                    )
                readout = slot.get('readout', structure_kwargs.get('readout', 'mean_context'))
                if readout not in ('mean', 'mean_context'):
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        f'has unsupported readout={readout!r}'
                    )
                if topology == 'single_state':
                    topology_kwargs = slot.get('topology_kwargs', {})
                    if not hasattr(topology_kwargs, 'get'):
                        raise ValueError(
                            f'Computation topology_kwargs for {agent_name}.{slot_name} '
                            'must be a mapping'
                        )
                    if topology_kwargs.get('input_mapping', 'identity') != 'identity':
                        raise ValueError(
                            f'Puzzle token SingleState for {agent_name}.{slot_name} '
                            'requires input_mapping=identity'
                        )
                    if topology_kwargs.get('input_injection', 'z_plus_x') != 'z_plus_x':
                        raise ValueError(
                            f'Puzzle token SingleState for {agent_name}.{slot_name} '
                            'requires input_injection=z_plus_x'
                        )
                    if topology_kwargs.get('residual', False) is not False:
                        raise ValueError(
                            f'Puzzle token SingleState for {agent_name}.{slot_name} '
                            'requires residual=false'
                        )
                    sharing = slot.get(
                        'parameter_sharing', topology_kwargs.get('parameter_sharing', 'shared')
                    )
                    if sharing != 'shared':
                        raise ValueError(
                            f'Puzzle token SingleState for {agent_name}.{slot_name} '
                            'requires parameter_sharing=shared'
                        )
