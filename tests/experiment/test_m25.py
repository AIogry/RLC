"""Declarative, import-boundary, and executable-smoke gates for M25."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from impls.diagnostics.puzzle.control_coordinates import gf2_rank
from impls.experiment import load_study, prepare_run_design
from impls.representation.puzzle_effects import (
    detect_board_change_events,
    raw_effect_signature,
)
from impls.utils.effect_datasets import build_effect_event_index
from tools.check_boolean_flow_capacity import random_invertible_gf2_matrix
from tools.control_coordinate_common import validate_stage1_layout
from tools import evaluate_control_coordinates as evaluate_tool
from tools.train_control_coordinates import (
    DEFAULT_STUDY,
    _parser,
    resolve_training_design,
    run,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = DEFAULT_STUDY.parent / 'configs'


def _full_rank_twenty_bit_events():
    effects = np.eye(20, dtype=np.uint8)
    current = np.zeros(20, dtype=np.uint8)
    boards = [current.copy()]
    for effect in effects:
        current = np.bitwise_xor(current, effect)
        boards.append(current.copy())
    boards = np.asarray(boards)
    terminals = np.zeros(len(boards), dtype=np.uint8)
    terminals[-1] = 1
    return build_effect_event_index(
        boards,
        terminals,
        event_mask=detect_board_change_events(boards),
        effect_signature_fn=raw_effect_signature,
    )


class M25StudyTest(unittest.TestCase):
    def test_engineering_matrix_is_explicit_nonformal_and_targets_4x5_4x6(self):
        study = load_study(DEFAULT_STUDY)
        self.assertEqual(study.study_id, 'M25')
        self.assertEqual(study.data['seeds'], [0])
        self.assertFalse(study.data['protocol']['formal_training_started'])
        self.assertFalse(study.data['protocol']['formal_campaign_declared'])
        configs = [
            prepare_run_design(DEFAULT_STUDY, path)[1]
            for path in sorted(CONFIG_DIR.glob('*.yaml'))
        ]
        self.assertEqual(
            {config.data['environment'] for config in configs},
            {'puzzle-4x5-play-v0', 'puzzle-4x6-play-v0'},
        )
        required = {
            'num_bits', 'num_layers', 'permutation_seed',
            'coupling_logit_init_mean', 'coupling_logit_init_std',
            'binary_temperature', 'learning_rate', 'batch_size', 'train_steps',
            'event_sampling_mode',
        }
        for configuration in configs:
            self.assertFalse(configuration.data['formal'])
            self.assertEqual(configuration.data['protocol_stage'], 'engineering_stage1')
            self.assertEqual(
                configuration.data['factors']['depth_status'],
                'engineering_candidate_not_scientifically_fixed',
            )
            self.assertTrue(required <= set(configuration.data['control_coordinate']))
            self.assertEqual(
                configuration.data['control_coordinate']['event_sampling_mode'],
                'uniform',
            )
            self.assertNotIn('seed', configuration.data)

    def test_run_seed_and_exact_permutation_schedule_enter_resolved_config(self):
        args = _parser().parse_args([
            '--run-root', '/tmp/m25-unused', '--config', 'M25-E002', '--seed', '0'
        ])
        _, _, flow, event, launcher = resolve_training_design(args)
        self.assertEqual(flow['seed'], 0)
        self.assertEqual(flow['num_bits'], 24)
        self.assertEqual(len(flow['permutations']), flow['num_layers'])
        self.assertEqual(
            flow['permutation_schedule']['layers'][0]['permutation'],
            list(flow['permutations'][0]),
        )
        self.assertEqual(event['rank_deficiency_policy'], 'fail')
        self.assertEqual(launcher['batch_size'], flow['batch_size'])
        self.assertEqual(launcher['train_steps'], flow['train_steps'])

    def test_strict_layout_gate_excludes_singular_4x4(self):
        self.assertEqual(validate_stage1_layout('puzzle-4x5-play-v0'), (4, 5, 20))
        self.assertEqual(validate_stage1_layout('puzzle-4x6-play-v0'), (4, 6, 24))
        self.assertEqual(validate_stage1_layout('puzzle-3x3-play-v0'), (3, 3, 9))
        with self.assertRaisesRegex(ValueError, 'singular'):
            validate_stage1_layout('puzzle-4x4-play-v0')

    def test_training_import_graph_does_not_load_oracle_puzzle_algebra(self):
        script = (
            'import sys; import tools.train_control_coordinates; '
            'assert "impls.diagnostics.puzzle.algebra" not in sys.modules; '
            'assert "impls.representation.puzzle_algebra" not in sys.modules'
        )
        subprocess.run(
            [sys.executable, '-c', script], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )

    def test_capacity_target_generator_is_random_full_rank_for_both_dimensions(self):
        for num_bits in (20, 24):
            first = random_invertible_gf2_matrix(
                num_bits, np.random.default_rng(100 + num_bits)
            )
            second = random_invertible_gf2_matrix(
                num_bits, np.random.default_rng(100 + num_bits)
            )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(gf2_rank(first), num_bits)

    def test_two_step_training_tool_writes_resolved_run_and_semantic_checkpoints(self):
        event_index = _full_rank_twenty_bit_events()
        provenance = {
            'environment': 'puzzle-4x5-play-v0',
            'rows': 4,
            'cols': 5,
            'num_bits': 20,
            'dataset_dir': '/synthetic/nonprivileged',
            'standard_dataset_fields_required': ['observations', 'actions', 'terminals'],
            'model_input_fields': ['start_board', 'end_board'],
            'actions_used_as_model_input': False,
            'diagnostic_effect_signature_used_as_model_input': False,
        }
        with tempfile.TemporaryDirectory() as directory:
            args = _parser().parse_args([
                '--run-root', directory,
                '--config', 'M25-E001',
                '--seed', '0',
                '--train-steps', '2',
                '--batch-size', '8',
            ])
            with mock.patch(
                'tools.train_control_coordinates.load_puzzle_event_index',
                return_value=(event_index, provenance),
            ), mock.patch('tools.train_control_coordinates.json_print'):
                result = run(args)
            run_dir = Path(result['run_dir'])
            self.assertTrue((run_dir / 'resolved_config.json').is_file())
            self.assertTrue((run_dir / 'event_audit.json').is_file())
            self.assertTrue((run_dir / 'effective_matrix.json').is_file())
            self.assertTrue((run_dir / 'checkpoints/index.json').is_file())
            self.assertTrue((run_dir / 'checkpoints/last/params_2.pkl').is_file())
            with (run_dir / 'runtime_metadata.json').open() as file:
                metadata = json.load(file)
            self.assertEqual(metadata['status'], 'completed')
            self.assertFalse(metadata['formal_training'])
            self.assertEqual(
                metadata['metric_semantics'],
                'hard_binary_axis_locality_not_rl_success',
            )
            with (run_dir / 'resolved_config.json').open() as file:
                resolved = json.load(file)
            agent = resolved['algorithm_config']['agent']
            self.assertEqual(agent['seed'], 0)
            self.assertEqual(agent['event_sampling_mode'], 'uniform')
            self.assertIn('permutation_schedule', agent)
            self.assertEqual(
                resolved['algorithm_config']['scientific_boundary']['gciql_integration'],
                False,
            )
            evaluation_args = evaluate_tool._parser().parse_args([
                '--run-dir', str(run_dir), '--checkpoint-role', 'last'
            ])
            with mock.patch(
                'tools.evaluate_control_coordinates.load_puzzle_event_index',
                return_value=(event_index, provenance),
            ), mock.patch('tools.evaluate_control_coordinates.json_print'):
                evaluation = evaluate_tool.run(evaluation_args)
            self.assertEqual(
                evaluation['checkpoint']['checkpoint_role'], 'last'
            )
            self.assertEqual(
                evaluation['diagnostics']['effective_matrix'][
                    'effective_matrix_rank'
                ],
                20,
            )
            self.assertTrue(
                evaluation['diagnostics']['structural_invariants'][
                    'forward_inverse_exact'
                ]
            )
            self.assertFalse(evaluation['oracle_operation_algebra_used'])


if __name__ == '__main__':
    unittest.main()
