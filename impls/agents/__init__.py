from .hiql import HIQLAgent, get_config as hiql_get_config
from .crl import CRLAgent, get_config as crl_get_config
from .coghp import CoGHPAgent, get_config as coghp_get_config
from .crl_policy_extractor import CRLPolicyExtractorAgent

agents = {'hiql': HIQLAgent, 'crl': CRLAgent, 'coghp': CoGHPAgent}
agent_configs = {'hiql': hiql_get_config, 'crl': crl_get_config, 'coghp': coghp_get_config}
agent_variants = {('crl', 'policy_extractor'): CRLPolicyExtractorAgent}


def resolve_agent_class(agent_name, runtime_variant=None):
    """Resolve a generic algorithm/runtime variant pair."""

    return agent_variants.get((agent_name, runtime_variant), agents[agent_name])

__all__ = (
    'HIQLAgent', 'CRLAgent', 'CRLPolicyExtractorAgent', 'CoGHPAgent',
    'agents', 'agent_configs', 'agent_variants', 'resolve_agent_class',
)
