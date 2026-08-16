"""M8 default-slot and runtime-provenance tests."""

import unittest

from impls.agents.crl import get_config as crl_config
from impls.main import _make_config, _parse_args, _resolved_compute_snapshot


class ComputationProvenanceTest(unittest.TestCase):
    def test_crl_vanilla_defaults_disable_all_computation_slots(self):
        config = crl_config()
        self.assertTrue(config['compute'])
        for slot in ('actor', 'critic_state', 'critic_goal', 'value_state', 'value_goal'):
            self.assertFalse(config['compute'][slot]['enabled'])

    def test_runtime_snapshot_records_resolved_slots(self):
        legacy = _make_config(_parse_args(['--agent', 'crl', '--actor_loss', 'awr']))
        legacy_snapshot = _resolved_compute_snapshot(legacy)
        self.assertEqual(
            set(legacy_snapshot),
            {'actor', 'critic_state', 'critic_goal', 'value_state', 'value_goal'},
        )
        self.assertTrue(all(not slot['enabled'] for slot in legacy_snapshot.values()))

        computation = _make_config(
            _parse_args(['--agent', 'crl', '--actor_loss', 'awr', '--computation'])
        )
        computation_snapshot = _resolved_compute_snapshot(computation)
        self.assertTrue(all(slot['enabled'] for slot in computation_snapshot.values()))
        self.assertEqual(
            {
                tuple(slot[key] for key in ('primitive', 'topology', 'credit'))
                for slot in computation_snapshot.values()
            },
            {('mlp', 'feedforward', 'direct')},
        )

    def test_runtime_snapshot_is_json_serializable_for_configdict(self):
        import json

        config = _make_config(_parse_args(['--agent', 'crl', '--computation']))
        snapshot = _resolved_compute_snapshot(config)
        encoded = json.dumps(snapshot, sort_keys=True)
        self.assertIn('critic_state', encoded)
        self.assertIn('feedforward', encoded)


if __name__ == '__main__':
    unittest.main()
