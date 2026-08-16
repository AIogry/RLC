import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

from impls.agents.hiql import HIQLAgent
from impls.main import _make_config, _parse_args
from impls.experiment import load_study, prepare_run_design


ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = ROOT / 'experiments' / 'M9B_two_state' / 'study.yaml'


class M9BStudyTest(unittest.TestCase):
    def test_matrix_and_baseline_reference(self):
        study = load_study(STUDY_PATH)
        configs = sorted((STUDY_PATH.parent / 'configs').glob('M9B-C*.yaml'))
        self.assertEqual(len(configs), 16)
        self.assertEqual(len(configs) * len(study.data['environments']) * len(study.data['seeds']), 32)
        self.assertEqual(study.data['baseline_reference'], {
            'study': 'M9A', 'hiql': 'M9A-C001', 'crl': 'M9A-C002',
        })

    def test_overrides_resolve_two_state_credit_and_schedule(self):
        expected = {
            'M9B-C001': ('crl', ('actor',), 'full_bptt', 2, 1),
            'M9B-C004': ('crl', ('actor',), 'one_step', 2, 6),
            'M9B-C005': ('hiql', ('high_actor',), 'full_bptt', 2, 1),
            'M9B-C012': ('hiql', ('low_actor',), 'one_step', 2, 6),
            'M9B-C013': ('hiql', ('high_actor', 'low_actor'), 'full_bptt', 2, 1),
            'M9B-C016': ('hiql', ('high_actor', 'low_actor'), 'one_step', 2, 6),
        }
        for config_id, (algorithm, enabled_slots, credit, h_cycles, l_cycles) in expected.items():
            _, configuration = prepare_run_design(STUDY_PATH, config_id)
            config = _make_config(_parse_args(['--agent', algorithm]), configuration=configuration)
            for slot_name, slot in config['compute'].items():
                self.assertEqual(slot.get('enabled', False), slot_name in enabled_slots)
                if slot_name in enabled_slots:
                    self.assertEqual(slot['topology'], 'two_state')
                    self.assertEqual(slot['credit'], credit)
                    self.assertEqual(slot['topology_kwargs']['h_cycles'], h_cycles)
                    self.assertEqual(slot['topology_kwargs']['l_cycles'], l_cycles)

    def test_high_low_have_independent_two_state_params_and_buffers(self):
        _, configuration = prepare_run_design(STUDY_PATH, 'M9B-C013')
        config = _make_config(_parse_args(['--agent', 'hiql']), configuration=configuration)
        observations = jnp.zeros((1, 29), dtype=jnp.float32)
        actions = jnp.zeros((1, 8), dtype=jnp.float32)
        agent = HIQLAgent.create(0, observations, actions, config)
        params = agent.network.params
        buffers = agent.network.model_state['buffers']
        high_params = params['modules_high_actor']['actor_net']['topology']
        low_params = params['modules_low_actor']['actor_net']['topology']
        high_buffers = buffers['modules_high_actor']['actor_net']['topology']
        low_buffers = buffers['modules_low_actor']['actor_net']['topology']
        self.assertEqual(set(high_params), {'input_mapping', 'h_update', 'l_update'})
        self.assertEqual(set(low_params), {'input_mapping', 'h_update', 'l_update'})
        h_differences = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(
                lambda high, low: jnp.max(jnp.abs(high - low)),
                high_params['h_update'], low_params['h_update'],
            )
        )
        self.assertGreater(float(max(h_differences)), 0.0)
        self.assertEqual(high_buffers['z_h_init'].shape, (512,))
        self.assertEqual(high_buffers['z_l_init'].shape, (512,))
        self.assertEqual(low_buffers['z_h_init'].shape, (512,))
        self.assertEqual(low_buffers['z_l_init'].shape, (512,))
        self.assertFalse((high_buffers['z_h_init'] == low_buffers['z_h_init']).all())


if __name__ == '__main__':
    unittest.main()
