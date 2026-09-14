"""Lazy canonical agent registry.

Keeping registry imports lazy prevents standalone representation tools from
loading unrelated goal-conditioning or oracle-diagnostic modules.  Registry
keys and values remain API-compatible with the historical dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module


_AGENT_TARGETS = {
    'hiql': ('.hiql', 'HIQLAgent'),
    'crl': ('.crl', 'CRLAgent'),
    'coghp': ('.coghp', 'CoGHPAgent'),
    'gcbc': ('.gcbc', 'GCBCAgent'),
    'gciql': ('.gciql', 'GCIQLAgent'),
    'gcivl': ('.gcivl', 'GCIVLAgent'),
    'qrl': ('.qrl', 'QRLAgent'),
}
_CONFIG_TARGETS = {
    name: (module, 'get_config') for name, (module, _) in _AGENT_TARGETS.items()
}
_VARIANT_TARGETS = {
    ('crl', 'policy_extractor'): ('.crl_policy_extractor', 'CRLPolicyExtractorAgent'),
}
_CLASS_EXPORTS = {
    attribute: (module, attribute)
    for module, attribute in _AGENT_TARGETS.values()
} | {
    'CRLPolicyExtractorAgent': (
        '.crl_policy_extractor', 'CRLPolicyExtractorAgent'
    ),
    'ControlCoordinateAgent': ('.control_coordinate', 'ControlCoordinateAgent'),
}
_CONFIG_EXPORTS = {
    f'{name}_get_config': target for name, target in _CONFIG_TARGETS.items()
}


def _resolve(target):
    module_name, attribute = target
    return getattr(import_module(module_name, __name__), attribute)


class _LazyRegistry(Mapping):
    def __init__(self, targets):
        self._targets = dict(targets)
        self._resolved = {}

    def __getitem__(self, key):
        if key not in self._resolved:
            self._resolved[key] = _resolve(self._targets[key])
        return self._resolved[key]

    def __iter__(self):
        return iter(self._targets)

    def __len__(self):
        return len(self._targets)


agents = _LazyRegistry(_AGENT_TARGETS)
agent_configs = _LazyRegistry(_CONFIG_TARGETS)
agent_variants = _LazyRegistry(_VARIANT_TARGETS)


def resolve_agent_class(agent_name, runtime_variant=None):
    """Resolve a generic algorithm/runtime variant pair."""

    variant = (agent_name, runtime_variant)
    if variant in agent_variants:
        return agent_variants[variant]
    return agents[agent_name]


def __getattr__(name):
    target = (_CLASS_EXPORTS | _CONFIG_EXPORTS).get(name)
    if target is None:
        raise AttributeError(name)
    value = _resolve(target)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_CLASS_EXPORTS) | set(_CONFIG_EXPORTS))


__all__ = (
    *tuple(_CLASS_EXPORTS),
    *tuple(_CONFIG_EXPORTS),
    'agents',
    'agent_configs',
    'agent_variants',
    'resolve_agent_class',
)
