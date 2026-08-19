import json
import tempfile
import unittest
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from impls.experiment.management import finalize_run
from impls.utils.checkpointing import (
    resolve_checkpoint,
    should_update_best,
    write_checkpoint_index,
)
from impls.utils.flax_utils import (
    TrainState,
    restore_agent_from_checkpoint,
    save_agent,
    save_semantic_checkpoint,
)


class CheckpointLifecycleTest(unittest.TestCase):
    def _state(self, value):
        model = nn.Dense(1)
        variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
        params = {
            'kernel': jnp.full_like(variables['params']['kernel'], value),
            'bias': jnp.full_like(variables['params']['bias'], value),
        }
        return TrainState.create(model, params, tx=None, model_state={})

    def _metadata(self, metric, step, role):
        return {
            'selection_metric': 'evaluation/overall_success',
            'selection_metric_value': metric,
            'best_step': step,
            'train_steps': 10,
            'environment': 'toy-v0',
            'study_id': 'TEST',
            'config_id': 'TEST-C001',
            'config_slug': 'control',
            'training_seed': 0,
            'git_commit': 'abc',
            'evaluation_protocol_at_selection': {
                'eval_tasks': 1,
                'eval_episodes': 20,
                'eval_temperature': 0.0,
                'eval_gaussian': None,
            },
            'selected_from_training_evaluation': role == 'best',
        }

    def test_best_update_is_strict_and_keeps_earlier_tie(self):
        best_metric = None
        best_step = None
        for metric, step in ((0.4, 100), (0.4, 200), (0.3, 300), (0.5, 400)):
            if should_update_best(metric, best_metric):
                best_metric, best_step = metric, step
        self.assertEqual(best_metric, 0.5)
        self.assertEqual(best_step, 400)
        self.assertFalse(should_update_best(0.5, 0.5))

    def test_semantic_best_last_and_numeric_restore(self):
        root = Path(tempfile.mkdtemp()) / 'run'
        (root / 'checkpoints').mkdir(parents=True)
        best_state = self._state(1.0)
        last_state = self._state(2.0)
        best = save_semantic_checkpoint(best_state, root, 'best', 4, self._metadata(0.7, 4, 'best'))
        last = save_semantic_checkpoint(last_state, root, 'last', 10, self._metadata(0.6, 4, 'last'))
        save_agent(best_state, root / 'checkpoints', 4, {'checkpoint_role': 'numeric'})
        write_checkpoint_index(root, best=best, last=last)

        best_info = resolve_checkpoint(root, 'best')
        last_info = resolve_checkpoint(root, 'last')
        numeric_info = resolve_checkpoint(root, 4)
        self.assertEqual(best_info['checkpoint_role'], 'best')
        self.assertEqual(best_info['checkpoint_step'], 4)
        self.assertEqual(last_info['checkpoint_role'], 'last')
        self.assertEqual(last_info['checkpoint_step'], 10)
        self.assertEqual(numeric_info['checkpoint_role'], 'numeric')
        self.assertEqual(best_info['checkpoint_sha256'], best['sha256'])

        restored_best = restore_agent_from_checkpoint(best_state, best_info['checkpoint_path'])
        restored_last = restore_agent_from_checkpoint(best_state, last_info['checkpoint_path'])
        np.testing.assert_array_equal(restored_best.params['kernel'], best_state.params['kernel'])
        np.testing.assert_array_equal(restored_last.params['kernel'], last_state.params['kernel'])

    def test_best_equals_last_is_explicit_and_both_selectors_resolve(self):
        root = Path(tempfile.mkdtemp()) / 'run'
        (root / 'checkpoints').mkdir(parents=True)
        state = self._state(3.0)
        best = save_semantic_checkpoint(state, root, 'best', 10, self._metadata(0.8, 10, 'best'))
        last = save_semantic_checkpoint(state, root, 'last', 10, self._metadata(0.8, 10, 'last'))
        write_checkpoint_index(root, best=best, last=last)
        index = json.loads((root / 'checkpoints' / 'index.json').read_text())
        self.assertTrue(index['best_equals_last'])
        self.assertEqual(resolve_checkpoint(root, 'best')['checkpoint_step'], 10)
        self.assertEqual(resolve_checkpoint(root, 'last')['checkpoint_step'], 10)

    def test_last_is_only_present_when_written_after_success(self):
        root = Path(tempfile.mkdtemp()) / 'run'
        (root / 'checkpoints').mkdir(parents=True)
        best = save_semantic_checkpoint(self._state(1.0), root, 'best', 5, self._metadata(0.7, 5, 'best'))
        write_checkpoint_index(root, best=best, last=None)
        finalize_run(root, 'failed', 'synthetic interruption')
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(root, 'last')

        last = save_semantic_checkpoint(self._state(2.0), root, 'last', 10, self._metadata(0.6, 5, 'last'))
        write_checkpoint_index(root, best=best, last=last)
        self.assertEqual(resolve_checkpoint(root, 'last')['checkpoint_step'], 10)

    def test_summary_best_step_follows_checkpoint_index(self):
        root = Path(tempfile.mkdtemp()) / 'run'
        (root / 'checkpoints').mkdir(parents=True)
        best = save_semantic_checkpoint(self._state(1.0), root, 'best', 4, self._metadata(0.5, 4, 'best'))
        write_checkpoint_index(root, best=best, last=None)
        (root / 'eval.csv').write_text(
            'step,evaluation/overall_success\n4,0.5\n10,0.9\n'
        )
        summary = finalize_run(root, 'completed')
        self.assertEqual(summary['best_step'], 4)
        self.assertEqual(summary['best_success'], 0.5)
        self.assertEqual(summary['final_success'], 0.9)


if __name__ == '__main__':
    unittest.main()
