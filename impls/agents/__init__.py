from .hiql import HIQLAgent, get_config as hiql_get_config
from .crl import CRLAgent, get_config as crl_get_config
from .coghp import CoGHPAgent, get_config as coghp_get_config

agents = {'hiql': HIQLAgent, 'crl': CRLAgent, 'coghp': CoGHPAgent}
agent_configs = {'hiql': hiql_get_config, 'crl': crl_get_config, 'coghp': coghp_get_config}

__all__ = ('HIQLAgent', 'CRLAgent', 'CoGHPAgent', 'agents', 'agent_configs')
