import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get('RLC_TEST_PYTHON', os.sys.executable)

STUDY_YAML = """
study_id: TEST
name: launcher test
question: launcher test
primary_factors: [variant]
fixed_design: {}
deferred_factors: [future]
algorithms: [hiql]
placements: [baseline]
environments: [toy-a, toy-b]
seeds: [0]
primary_metric: evaluation/overall_success
"""

CONFIG_YAML = """
study_id: TEST
config_id: TEST-C001
slug: control
algorithm: hiql
factors: {variant: 0}
"""


class RunStudyLauncherTest(unittest.TestCase):
    def _fixture(self, fake_sweep=False):
        temp = tempfile.TemporaryDirectory()
        repo = Path(temp.name)
        (repo / 'scripts').mkdir()
        (repo / 'tools').mkdir()
        study_dir = repo / 'experiments' / 'toy'
        (study_dir / 'configs').mkdir(parents=True)
        (study_dir / 'study.yaml').write_text(STUDY_YAML)
        (study_dir / 'configs' / 'TEST-C001.yaml').write_text(CONFIG_YAML)
        dataset = repo / 'dataset'
        dataset.mkdir()
        for environment in ('toy-a', 'toy-b'):
            (dataset / f'{environment}.npz').touch()
            (dataset / f'{environment}-val.npz').touch()
        shutil.copy2(ROOT / 'scripts' / 'run_study.sh', repo / 'scripts' / 'run_study.sh')

        if fake_sweep:
            (repo / 'tools' / 'sweep.py').write_text(
                """import os, sys
from pathlib import Path
Path(os.environ['CAPTURE']).write_text('\\n'.join(sys.argv[1:]))
print('total=2 planned=2 completed=0 failed=0 running=0 retained=0 remaining=2 statuses: planned=2 running=0 completed=0 failed=0 aborted=0 invalid=0')
if '--dry-run' in sys.argv:
    print('[PLANNED] TEST-C001 control toy-a seed=0 GPU=<pending> run_dir=results')
"""
            )
        else:
            shutil.copy2(ROOT / 'tools' / 'sweep.py', repo / 'tools' / 'sweep.py')

        fake_bin = repo / 'bin'
        fake_bin.mkdir()
        nvidia = fake_bin / 'nvidia-smi'
        nvidia.write_text('#!/bin/sh\necho 0\necho 1\n')
        nvidia.chmod(0o755)

        subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
        subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
        subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=repo, check=True)
        return temp, repo, dataset

    def _run(self, repo, dataset, *extra, capture=True):
        capture_path = repo / 'capture.txt'
        env = os.environ.copy()
        env.update({
            'RLC_PYTHON': PYTHON,
            'CAPTURE': str(capture_path),
            'PATH': f'{repo / "bin"}:{env["PATH"]}',
            'PYTHONPATH': str(ROOT),
        })
        command = [
            'bash', 'scripts/run_study.sh',
            '--study', 'experiments/toy/study.yaml',
            '--configs', 'TEST-C001',
            '--gpus', '0,1',
            '--run-root', str(repo / 'results'),
            '--dataset-root', str(dataset),
            '--train-steps', '500000',
            '--batch-size', '1024',
            '--log-interval', '5000',
            '--eval-interval', '100000',
            '--eval-tasks', 'all',
            '--eval-episodes', '20',
            '--save-interval', '500000',
            '--eval-temperature', '0',
            *extra,
        ]
        result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=capture)
        captured = capture_path.read_text() if capture and capture_path.exists() else ''
        return result, captured

    def test_dry_run_forwards_filter_and_log_protocol_without_execute(self):
        temp, repo, dataset = self._fixture(fake_sweep=True)
        with temp:
            result, captured = self._run(repo, dataset, '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Mode: --dry-run', result.stdout)
        self.assertIn('Planned runs: 2', result.stdout)
        self.assertIn('--dry-run', captured)
        self.assertNotIn('--execute', captured)
        self.assertIn('--log_interval=5000', captured)
        self.assertIn('--configs\nTEST-C001', captured)

    def test_execute_mode_forwards_execute_and_log_protocol(self):
        temp, repo, dataset = self._fixture(fake_sweep=True)
        with temp:
            result, captured = self._run(repo, dataset, '--execute')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Mode: --execute', result.stdout)
        self.assertIn('--execute', captured)
        self.assertNotIn('--dry-run', captured)
        self.assertIn('--log_interval=5000', captured)

    def test_mode_is_required_and_mutually_exclusive(self):
        temp, repo, dataset = self._fixture(fake_sweep=True)
        with temp:
            missing, _ = self._run(repo, dataset)
            both, _ = self._run(repo, dataset, '--dry-run', '--execute')
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('exactly one of --dry-run or --execute', missing.stderr)
        self.assertNotEqual(both.returncode, 0)
        self.assertIn('mutually exclusive', both.stderr)


if __name__ == '__main__':
    unittest.main()
