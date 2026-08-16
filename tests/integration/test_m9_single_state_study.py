import unittest
from pathlib import Path

import jax.numpy as jnp

from impls.agents.crl import CRLAgent
from impls.agents.hiql import HIQLAgent
from impls.main import _make_config, _parse_args
from impls.experiment import load_study, make_run_path, prepare_run_design


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M9A_single_state_iteration' / 'study.yaml'


class M9SingleStateStudyTest(unittest.TestCase):
    def test_matrix_has_26_configurations_and_52_runs(self):
        study = load_study(STUDY_PATH)
        configs = sorted((STUDY_PATH.parent / 'configs').glob('M9A-C*.yaml'))
        self.assertEqual(len(configs), 26)
        self.assertEqual(len(configs) * len(study.data['environments']) * len(study.data['seeds']), 52)
        self.assertEqual(study.data['environments'], [
            'antmaze-medium-navigate-v0',
            'antmaze-large-navigate-v0',
        ])

    def test_overrides_control_actor_slots(self):
        expected = {
            'M9A-C001': (False, False),
            'M9A-C009': (True, False),
            'M9A-C015': (False, True),
            'M9A-C021': (True, True),
        }
        for config_id, slots in expected.items():
            _, configuration = prepare_run_design(STUDY_PATH, config_id)
            args = _parse_args(['--agent', 'hiql'])
            config = _make_config(args, configuration=configuration)
            self.assertEqual(
                (config['compute']['high_actor']['enabled'], config['compute']['low_actor']['enabled']),
                slots,
            )
            if any(slots):
                enabled = config['compute']['high_actor'] if slots[0] else config['compute']['low_actor']
                self.assertEqual(enabled['topology'], 'single_state')
                self.assertEqual(enabled['topology_kwargs']['state_dim'], 512)

    def test_high_and_low_buffers_are_independent(self):
        _, configuration = prepare_run_design(STUDY_PATH, 'M9A-C021')
        config = _make_config(_parse_args(['--agent', 'hiql']), configuration=configuration)
        observations = jnp.zeros((1, 29), dtype=jnp.float32)
        actions = jnp.zeros((1, 8), dtype=jnp.float32)
        agent = HIQLAgent.create(0, observations, actions, config)
        model_state = agent.network.model_state
        high = model_state['buffers']['modules_high_actor']['actor_net']['topology']['z_init']
        low = model_state['buffers']['modules_low_actor']['actor_net']['topology']['z_init']
        self.assertEqual(high.shape, (512,))
        self.assertEqual(low.shape, (512,))
        self.assertFalse(high is low)
        self.assertIn('modules_high_actor', agent.network.params)
        self.assertIn('modules_low_actor', agent.network.params)

    def test_baselines_keep_empty_model_state(self):
        observations = jnp.zeros((1, 29), dtype=jnp.float32)
        actions = jnp.zeros((1, 8), dtype=jnp.float32)
        _, hiql_configuration = prepare_run_design(STUDY_PATH, 'M9A-C001')
        hiql_config = _make_config(_parse_args(['--agent', 'hiql']), configuration=hiql_configuration)
        hiql = HIQLAgent.create(0, observations, actions, hiql_config)
        self.assertEqual(hiql.network.model_state, {})

        _, crl_configuration = prepare_run_design(STUDY_PATH, 'M9A-C002')
        crl_config = _make_config(_parse_args(['--agent', 'crl']), configuration=crl_configuration)
        crl = CRLAgent.create(0, observations, actions, crl_config)
        self.assertEqual(crl.network.model_state, {})

    def test_canonical_run_path_is_stable(self):
        path = make_run_path('runs', 'M9A', 'M9A-C003', 'crl_actor_k1_nores', 'antmaze-medium-navigate-v0', 0)
        self.assertEqual(
            str(path),
            'runs/M9A/M9A-C003__crl_actor_k1_nores/antmaze-medium-navigate-v0/seed_000',
        )


if __name__ == '__main__':
    unittest.main()
