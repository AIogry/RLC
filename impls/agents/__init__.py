from .hiql import HIQLAgent, get_config as hiql_get_config
from .crl import CRLAgent, get_config as crl_get_config
from .coghp import CoGHPAgent, get_config as coghp_get_config
from .gcbc import GCBCAgent, get_config as gcbc_get_config
from .gciql import GCIQLAgent, get_config as gciql_get_config
from .gcivl import GCIVLAgent, get_config as gcivl_get_config
from .qrl import QRLAgent, get_config as qrl_get_config
from .crl_policy_extractor import CRLPolicyExtractorAgent

agents = {
    'hiql': HIQLAgent,
    'crl': CRLAgent,
    'coghp': CoGHPAgent,
    'gcbc': GCBCAgent,
    'gciql': GCIQLAgent,
    'gcivl': GCIVLAgent,
    'qrl': QRLAgent,
}
agent_configs = {
    'hiql': hiql_get_config,
    'crl': crl_get_config,
    'coghp': coghp_get_config,
    'gcbc': gcbc_get_config,
    'gciql': gciql_get_config,
    'gcivl': gcivl_get_config,
    'qrl': qrl_get_config,
}
agent_variants = {('crl', 'policy_extractor'): CRLPolicyExtractorAgent}


def resolve_agent_class(agent_name, runtime_variant=None):
    """Resolve a generic algorithm/runtime variant pair."""

    return agent_variants.get((agent_name, runtime_variant), agents[agent_name])

__all__ = (
    'HIQLAgent', 'CRLAgent', 'CRLPolicyExtractorAgent', 'CoGHPAgent',
    'GCBCAgent', 'GCIQLAgent', 'GCIVLAgent', 'QRLAgent',
    'agents', 'agent_configs', 'agent_variants', 'resolve_agent_class',
)
