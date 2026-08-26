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
                if slot.get('topology') != 'feedforward':
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        'requires topology=feedforward; token recurrence is deferred'
                    )
                if slot.get('credit', 'direct') != 'direct':
                    raise ValueError(
                        f'Puzzle token computation for {agent_name}.{slot_name} '
                        'requires credit=direct'
                    )
                if slot.get('block', 'plain') != 'mlp_mixer':
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
