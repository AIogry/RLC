"""Static contract tests for the declarative M23A diagnostic campaign."""

from __future__ import annotations

import unittest
from pathlib import Path

from impls.diagnostics.puzzle.campaign import load_campaign


class M23ACampaignTest(unittest.TestCase):
    def test_full_matrix_and_frozen_pairing_protocol(self):
        study_path = (
            Path(__file__).resolve().parents[2]
            / 'experiments/M23A_puzzle_direct_rollout_audit/study.yaml'
        )
        study, configurations, protocol = load_campaign(study_path)
        self.assertEqual(study.study_id, 'M23A')
        self.assertEqual(len(configurations), 12)
        self.assertEqual(protocol['task_ids'], (1, 2, 3, 4, 5))
        self.assertEqual(protocol['episodes_per_task'], 50)
        self.assertEqual(protocol['evaluation_seed'], 20260909)
        self.assertEqual(protocol['checkpoint_role'], 'last')
        self.assertEqual(protocol['checkpoint_step'], 1_000_000)
        self.assertTrue(protocol['controlled_goal_replay'])
        by_environment = {}
        for configuration in configurations:
            source = configuration.data['source_policy']
            by_environment.setdefault(configuration.data['environment'], []).append(source)
        self.assertEqual(set(by_environment), set(study.data['environments']))
        for policies in by_environment.values():
            self.assertEqual(
                {policy['label'] for policy in policies},
                {'flat_alpha1p0', 'mixer_l2_alpha1p0', 'mixer_l2_alpha0p4'},
            )
            alpha04 = next(policy for policy in policies if policy['label'] == 'mixer_l2_alpha0p4')
            provenance = alpha04['provenance']
            self.assertEqual(provenance['provenance_status'], 'scoped_exception')
            self.assertEqual(provenance['reason'], 'concurrent M23A-F0 diagnostic development')
            self.assertEqual(
                provenance['evidence_level'],
                'user-attested / partially machine-verified',
            )
            self.assertEqual(provenance['scope']['run_attempt'], 1)
            self.assertEqual(len(provenance['scope']['checkpoint_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
