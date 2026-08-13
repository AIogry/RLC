from .hiql import HIQLAgent, get_config as hiql_get_config
from .crl import CRLAgent, get_config as crl_get_config

agents = {'hiql': HIQLAgent, 'crl': CRLAgent}
agent_configs = {'hiql': hiql_get_config, 'crl': crl_get_config}

__all__ = ('HIQLAgent', 'CRLAgent', 'agents', 'agent_configs')
