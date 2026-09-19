"""CLI contracts for explicit generation and CPU-only production input audit."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class GoalConditioningToolsTest(unittest.TestCase):
    def run_tool(self, script, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / 'tools' / script), *map(str, args)],
            cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT), 'JAX_PLATFORMS': 'cpu', 'PYTHONDONTWRITEBYTECODE': '1'},
            text=True, capture_output=True, timeout=60,
        )

    def test_generator_requires_explicit_identity_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='goal_transform_cli_') as folder:
            output = Path(folder) / 'transform.json'
            required = self.run_tool('generate_goal_coordinate_transform.py', '--kind', 'permutation')
            self.assertNotEqual(required.returncode, 0)
            result = self.run_tool('generate_goal_coordinate_transform.py', '--kind', 'permutation',
                                   '--N', 9, '--transform-seed', 26019, '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = output.read_bytes()
            payload = json.loads(original)
            self.assertEqual(payload['rank'], 9)
            self.assertEqual(sorted(payload['permutation']), list(range(9)))
            self.assertEqual(payload['provenance']['transform_seed'], 26019)
            repeat = self.run_tool('generate_goal_coordinate_transform.py', '--kind', 'permutation',
                                   '--N', 9, '--transform-seed', 42, '--output', output)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_default_audit_captures_cpu_production_inputs_without_data_or_training(self):
        with tempfile.TemporaryDirectory(prefix='goal_audit_cli_') as folder:
            output = Path(folder) / 'audit.json'
            result = self.run_tool('audit_goal_conditioning.py', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['backend'], 'cpu')
            self.assertTrue(report['engineering_only'])
            self.assertFalse(report['real_data_used'])
            self.assertEqual(report['optimizer_updates'], 0)
            self.assertEqual(report['source'], 'production_forward_intermediate_capture')
            for path in report['imports'].values():
                self.assertTrue(Path(path).is_relative_to(ROOT))
            self.assertEqual(report['modules']['critic']['flat_shape'], [2, 115])
            self.assertEqual(report['modules']['critic']['token_aux_shape'], [2, 9, 1])


if __name__ == '__main__':
    unittest.main()
