"""Study, doctor, dry-run, and analyzer gates for M19A."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from impls.experiment import load_study, prepare_run_design
from impls.main import _make_config, _parse_args
from tools import analyze_m19a, m19a_doctor, sweep


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M19A_puzzle_entity_factorization_isolation/study.yaml'
DATASET_ROOT = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
DATASET_AVAILABLE = all(
    (DATASET_ROOT / name).is_file()
    for name in ('puzzle-4x4-play-v0.npz', 'puzzle-4x4-play-v0-val.npz')
)


class M19AEntityFactorizationStudyTest(unittest.TestCase):
    def test_exact_single_config_and_resolved_entity_semantics(self):
        study = load_study(STUDY)
        self.assertEqual(study.study_id, 'M19A')
        self.assertEqual(study.data['algorithms'], ['gciql'])
        self.assertEqual(study.data['environments'], ['puzzle-4x4-play-v0'])
        self.assertEqual(study.data['seeds'], [0])
        self.assertEqual(study.data['protocol']['formal_training_started'], False)
        self.assertEqual(study.data['acceptance']['expected_new_formal_runs'], 1)
        self.assertEqual(set(study.data['historical_anchors']), {'anchor_flat', 'anchor_mixer'})
        configs = sorted((STUDY.parent / 'configs').glob('*.yaml'))
        self.assertEqual([path.name for path in configs], ['M19A-4x4-E001.yaml'])
        _, configuration = prepare_run_design(STUDY, configs[0])
        self.assertEqual(configuration.config_id, 'M19A-4x4-E001')
        self.assertTrue(configuration.data['executable'])
        config = _make_config(_parse_args(['--agent', 'gciql']), configuration=configuration)
        self.assertEqual(config['alpha'], 1.0)
        for slot_name in ('actor', 'value', 'critic'):
            slot = config['compute'][slot_name]
            self.assertTrue(slot['enabled'])
            self.assertEqual(slot['structure'], 'puzzle_tokens')
            self.assertEqual(slot['block'], 'entity_mlp')
            self.assertEqual(slot['block_type'], 'entity_mlp')
            self.assertEqual(slot['topology'], 'feedforward')
            self.assertEqual(slot['credit'], 'direct')
            self.assertEqual(slot['readout'], 'mean_context')
            self.assertEqual(
                dict(slot['block_kwargs']),
                {'num_blocks': 2, 'channel_hidden_dim': 256},
            )
            self.assertFalse(slot['token_interaction'])
            self.assertNotIn('topology_kwargs', slot)

    @unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 data is unavailable')
    def test_doctor_validates_historical_anchors_and_absent_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            report = m19a_doctor.validate(
                STUDY,
                DATASET_ROOT,
                Path(directory) / 'runs',
                ['0'],
                m19a_source_commit='m19a-test-user-supplied-commit',
            )
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['new_formal_runs'], 1)
        self.assertEqual(report['historical_anchors'], 2)
        self.assertTrue(report['cross_commit_anchor_reuse'])
        self.assertEqual(report['resolved_config']['block_type'], 'entity_mlp')
        self.assertEqual(report['resolved_config']['block_depth_L'], 2)
        self.assertEqual(report['accounting']['removed_token_branch_params_actor_or_value'], 4256)
        self.assertEqual(report['accounting']['removed_token_branch_dense_macs_actor_or_value'], 524288)
        self.assertEqual(report['m19a_source_commit_status'], 'user_supplied')

    @unittest.skipUnless(DATASET_AVAILABLE, 'real Puzzle-4x4 data is unavailable')
    def test_generic_dry_run_has_only_one_new_job(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = sweep.main([
                    '--study', str(STUDY),
                    '--gpus', '0',
                    '--run-root', str(Path(directory) / 'runs'),
                    '--dataset-root', str(DATASET_ROOT),
                    '--dry-run',
                ])
        output = stream.getvalue()
        self.assertEqual(result, 0)
        self.assertIn('total=1 planned=1', output)
        self.assertIn('completed=0', output)
        self.assertIn('remaining=1', output)
        self.assertEqual(output.count('[PLANNED]'), 1)
        self.assertIn('M19A-4x4-E001', output)
        self.assertNotIn('M16B-4x4-B000 [PLANNED]', output)
        self.assertNotIn('M16B-4x4-S002 [PLANNED]', output)

    def test_analyzer_keeps_fixed_order_and_never_fills_missing_entity_result(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = analyze_m19a.collect(STUDY, Path(directory) / 'runs')
        self.assertEqual([row['method'] for row in rows], [
            'Flat MLP',
            'Entity Token + Entity MLP',
            'Entity Token + MLP-Mixer',
        ])
        self.assertEqual(rows[0]['run_status'], 'completed')
        self.assertEqual(rows[1]['run_status'], 'missing')
        self.assertEqual(rows[2]['run_status'], 'completed')
        effects = analyze_m19a.descriptive_effects(rows)
        self.assertIsNone(effects[0]['delta_final_success'])
        self.assertIsNone(effects[1]['delta_final_success'])
        report = analyze_m19a.markdown(rows, effects)
        self.assertIn('E001 尚未完成', report)
        self.assertIn('Delta_structured_factorization_package', report)
        self.assertIn('Delta_added_token_mixing_branch', report)


if __name__ == '__main__':
    unittest.main()
