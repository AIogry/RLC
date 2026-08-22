"""HIQL accounting compatibility and entrypoint lifecycle tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from impls.agents import agents
from impls.experiment import load_study, prepare_run_design
from impls.main import (
    _accounting_consistency_audit,
    _actor_parameter_accounting,
    _computation_slot_accounting,
    _make_config,
    _parse_args,
    run,
)


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments' / 'M11B_cross_task_computation' / 'study.yaml'
DATASET = Path('/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
ENVIRONMENT = 'antmaze-giant-navigate-v0'
TARGET_CONFIGS = ('M11B-C008', 'M11B-C009', 'M11B-C010')


class HIQLAccountingCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = load_study(STUDY)

    def _resolved(self, config_id):
        _, configuration = prepare_run_design(STUDY, config_id)
        return configuration, _make_config(
            _parse_args(['--agent', 'hiql']), configuration=configuration,
        )

    def _agent(self, config):
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        actions = jnp.zeros((2, 2), dtype=jnp.float32)
        return agents['hiql'].create(0, observations, actions, config)

    def test_high_low_and_legacy_accounting_are_consistent(self):
        for config_id in TARGET_CONFIGS + ('M11B-C007',):
            with self.subTest(config_id=config_id):
                _, config = self._resolved(config_id)
                agent = self._agent(config)
                legacy = _actor_parameter_accounting(agent, config)
                generic = _computation_slot_accounting(agent, config)
                audit = _accounting_consistency_audit(legacy, generic, config)
                self.assertEqual(audit['status'], 'pass')
                if config_id == 'M11B-C007':
                    self.assertEqual(generic, {})
                else:
                    expected = {
                        'M11B-C008': ('high_actor',),
                        'M11B-C009': ('low_actor',),
                        'M11B-C010': ('high_actor', 'low_actor'),
                    }[config_id]
                    self.assertEqual(tuple(generic), expected)
                    for slot_name in expected:
                        self.assertEqual(generic[slot_name]['topology'], 'single_state')
                        self.assertEqual(generic[slot_name]['state_dim'], 512)
                        self.assertEqual(generic[slot_name]['iterations'], 4)
                        self.assertEqual(generic[slot_name]['buffer_elements'], legacy[slot_name]['buffer_elements'])

    @unittest.skipUnless(
        (DATASET / f'{ENVIRONMENT}.npz').is_file()
        and (DATASET / f'{ENVIRONMENT}-val.npz').is_file(),
        'M11B real dataset is not available',
    )
    def test_real_run_lifecycle_high_low_and_joint(self):
        """Exercise run() through accounting, metadata, one finite update, and finalization."""

        with tempfile.TemporaryDirectory(prefix='m11b_hiql_entrypoint_') as directory:
            run_root = Path(directory)
            for config_id in TARGET_CONFIGS:
                args = _parse_args([
                    '--agent', 'hiql',
                    '--env_name', ENVIRONMENT,
                    '--seed', '0',
                    '--study', str(STUDY),
                    '--config', config_id,
                    '--run_root', str(run_root),
                    '--run_attempt', '0',
                    '--train_steps', '1',
                    '--log_interval', '1',
                    '--eval_interval', '1000000',
                    '--save_interval', '1000000',
                    '--eval_tasks', '1',
                    '--eval_episodes', '1',
                    '--no-save-best-checkpoint',
                    '--no-save-last-checkpoint',
                ])
                with patch(
                    'impls.experiment.management.git_metadata',
                    return_value={'git_commit': '<manual-user-supplied>', 'git_dirty': None},
                ):
                    run_dir = Path(run(args))
                self.assertTrue(run_dir.is_relative_to(run_root))
                self.assertEqual(run_dir.name, 'seed_000')
                with (run_dir / 'runtime_metadata.json').open() as file:
                    metadata = json.load(file)
                self.assertEqual(metadata['status'], 'completed')
                self.assertEqual(metadata['run_attempt'], 0)
                self.assertEqual(metadata['accounting_consistency']['status'], 'pass')
                self.assertIn('computation_slot_accounting', metadata)
                self.assertNotIn('failure_reason', metadata)
                with (run_dir / 'train.csv').open(newline='') as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 1)
                self.assertTrue((run_dir / 'summary.json').is_file())
                self.assertFalse((run_dir / 'failure.json').exists())


if __name__ == '__main__':
    unittest.main()
