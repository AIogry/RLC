"""CPU tests for manual-only smoke safety and read-only inheritance audits.

No test initializes a Git repository, launches a GPU child or a rollout.
"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import study_gpu_smoke as smoke
from tools.audit_study_inheritance import audit, differences

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml'
REFERENCE = ROOT / 'experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml'


class StudyPreparationToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        for suffix in ('', '-val'):
            (self.data / f'puzzle-4x5-play-v0{suffix}.npz').touch()
        self.args = SimpleNamespace(study=STUDY, configs='M26-C007,M26-C008', gpu=1, workers=2,
                                    batch_size=1024, steps=200, warmup=20, eval_episodes=1,
                                    dataset_root=self.data, output=self.root / 'smoke', execute=False,
                                    memory_limit_fraction=0.9, expected_source_sha='a' * 40)

    def test_plan_retains_full_workload_without_creating_output(self):
        with mock.patch.object(smoke, 'gpu_snapshot', side_effect=AssertionError('GPU query forbidden')):
            plan = smoke.make_plan(self.args)
        self.assertFalse(plan['execute_requested'])
        self.assertEqual(plan['batch_size_per_worker'], 1024)
        self.assertEqual(plan['updates_per_worker'], 200)
        self.assertEqual(plan['eval_tasks'], 'all')
        self.assertEqual(plan['seed'], 0)
        for record in plan['configs']:
            self.assertEqual(record['resolved_agent']['compute']['actor']['structure_kwargs']['token_dim'], 128)
        self.assertFalse(self.args.output.exists())

    def test_default_cli_never_executes_or_queries_gpu(self):
        arguments = ['--study', str(STUDY), '--configs', 'M26-C008', '--gpu', '1', '--workers', '1',
                     '--batch-size', '1024', '--steps', '200', '--warmup', '20',
                     '--dataset-root', str(self.data), '--output', str(self.args.output)]
        with mock.patch.object(smoke, 'execute', side_effect=AssertionError('execute forbidden')), \
             mock.patch.object(smoke, 'gpu_snapshot', side_effect=AssertionError('GPU forbidden')), \
             mock.patch.object(smoke.subprocess, 'Popen', wraps=smoke.subprocess.Popen) as popen, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(smoke.main(arguments), 0)
        # Read-only git subprocesses are allowed; no child workload was launched.
        for call in popen.call_args_list:
            self.assertEqual(call.args[0][0], 'git')
        self.assertFalse(json.loads(output.getvalue())['execute_requested'])
        self.assertFalse(self.args.output.exists())

    def test_plan_rejects_invalid_workload_and_wrong_gpu(self):
        for key, value in [('configs', 'M26-C008'), ('workers', 1), ('batch_size', 32),
                           ('steps', 1000000), ('warmup', 0), ('eval_episodes', 20),
                           ('gpu', 0), ('memory_limit_fraction', 0.95)]:
            args = SimpleNamespace(**vars(self.args))
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.make_plan(args)
        self.assertFalse(self.args.output.exists())

    def test_rejects_formal_source_existing_and_symlink_outputs(self):
        for path in (ROOT / 'smoke', Path('/home/eai/Research/RLC/smoke'),
                     Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M26/smoke'), self.data):
            args = SimpleNamespace(**vars(self.args))  # independent options, same real configs
            args.output = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                smoke.make_plan(args)
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'missing')
        self.args.output = link
        with self.assertRaises(ValueError):
            smoke.make_plan(self.args)

    def test_execute_stops_before_gpu_when_source_not_clean_detached_exact(self):
        plan = smoke.make_plan(self.args)
        with mock.patch.object(smoke, 'gpu_snapshot', side_effect=AssertionError('GPU forbidden')):
            for changes in ({'source_dirty': True}, {'source_branch': 'feature'}, {'source_sha': 'b' * 40}):
                candidate = dict(plan, source_dirty=False, source_branch='', source_sha='a' * 40)
                candidate.update(changes)
                with self.assertRaises(ValueError):
                    smoke.execute(candidate, self.args)
        self.assertFalse(self.args.output.exists())

    def test_occupied_gpu_never_spawns_or_creates_output(self):
        plan = dict(smoke.make_plan(self.args), source_dirty=False, source_branch='', source_sha='a' * 40)
        with mock.patch.object(smoke, 'gpu_snapshot', return_value={'processes': [{'pid': 1234}]}), \
             mock.patch.object(smoke.subprocess, 'Popen', side_effect=AssertionError('No child allowed')):
            with self.assertRaisesRegex(ValueError, 'already has compute'):
                smoke.execute(plan, self.args)
        self.assertFalse(self.args.output.exists())

    def test_gpu_snapshot_maps_physical_index_to_uuid_and_pids(self):
        with mock.patch.object(smoke.subprocess, 'check_output', side_effect=[
            '0, GPU-zero, 24564, 1000\n1, GPU-one, 24564, 20\n',
            'GPU-zero, 12, 900\nGPU-one, 34, 10\n',
        ]):
            snapshot = smoke.gpu_snapshot(1)
        self.assertEqual(snapshot['uuid'], 'GPU-one')
        self.assertEqual(snapshot['processes'], [{'pid': 34, 'used_mib': 10.0}])

    def test_child_environment_cannot_inherit_cpu_or_other_worktree(self):
        plan = {'gpu_snapshot': {'uuid': 'GPU-one'}, 'gpu': 1, 'workers': 2}
        with mock.patch.dict(os.environ, {'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '0',
                                          'PYTHONPATH': '/other/worktree', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'true'}):
            actual = smoke.child_environment(plan, 1)
        for key, expected in {'JAX_PLATFORMS': 'cuda', 'CUDA_VISIBLE_DEVICES': 'GPU-one',
                              'RLC_ASSIGNED_PHYSICAL_GPU': '1', 'RLC_WORKER_SLOT': '1',
                              'RLC_JOBS_PER_GPU': '2', 'PYTHONPATH': str(ROOT),
                              'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}.items():
            self.assertEqual(actual[key], expected)

    def test_worker_report_requires_source_config_device_and_full_workload(self):
        plan = dict(smoke.make_plan(self.args), gpu_snapshot={'uuid': 'GPU-one'})
        reports = [{'status':'pass','source_sha':plan['source_sha'],'pid':i+100,
                    'config_id':c['config_id'],'config_sha256':c['config_sha256'],
                    'physical_gpu':1,'gpu_uuid':'GPU-one','backend':'gpu','batch_size':1024,
                    'updates':200,'checkpoint_fresh_restore':'pass'} for i,c in enumerate(plan['configs'])]
        self.assertTrue(smoke.worker_reports_valid(plan, reports, [100,101]))
        for key, value in [('source_sha','wrong'),('config_sha256','wrong'),('batch_size',512),
                           ('updates',2),('backend','cpu'),('physical_gpu',0),('gpu_uuid','GPU-zero')]:
            changed = [dict(reports[0], **{key:value}), reports[1]]
            with self.subTest(key=key):
                self.assertFalse(smoke.worker_reports_valid(plan, changed, [100,101]))

    def test_overlap_requires_actual_training_and_eval_not_only_process_presence(self):
        phases = {'compile_warmup':1,'training':3,'validation':6,'evaluation':8,'done':10}
        first = {'phase_started_unix':phases}
        second = {'phase_started_unix':{k:v+0.1 for k,v in phases.items()}}
        self.assertTrue(all(smoke.workload_overlap([first,second]).values()))
        sequential = {'phase_started_unix':{k:v+20 for k,v in phases.items()}}
        self.assertFalse(any(smoke.workload_overlap([first,sequential]).values()))
        self.assertFalse(any(smoke.workload_overlap([{}]).values()))

    def test_new_unknown_gpu_process_fails_and_only_owned_children_are_stopped(self):
        plan = dict(smoke.make_plan(self.args), source_dirty=False, source_branch='', source_sha='a'*40)
        class Child:
            returncode = None
            def __init__(self, pid):
                self.pid = pid
                self.terminated = False
            def poll(self):
                return self.returncode
            def terminate(self):
                self.terminated = True
                self.returncode = -15
            def wait(self, **kwargs):
                return self.returncode
        children = [Child(100),Child(101)]
        snapshot = dict(physical_gpu=1,uuid='GPU-one',total_mib=24564,used_mib=20,processes=[])
        occupied = dict(snapshot,processes=[{'pid':999,'used_mib':10}])
        with mock.patch.object(smoke,'gpu_snapshot',side_effect=[snapshot,occupied]), \
             mock.patch.object(smoke,'host_snapshot',return_value={}), \
             mock.patch('tools.audit_goal_conditioning_study.dataset_metadata',return_value={}), \
             mock.patch.object(smoke.subprocess,'Popen',side_effect=children), \
             mock.patch.object(smoke,'read_git',return_value=''), redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.execute(plan,self.args),1)
        self.assertTrue(all(c.terminated for c in children))
        report=json.loads((self.args.output/'report.json').read_text())
        self.assertEqual(report['status'],'fail')
        self.assertIn('unknown GPU',report['failure'])
        self.assertEqual(report['exit_codes'],[-15,-15])

    def test_worker_control_flow_with_mock_devices_data_agent_and_evaluation(self):
        # Contract-only test: no GPU, optimizer, environment step or rollout runs.
        plan = smoke.make_plan(self.args)
        plan.update(updates_per_worker=2, warmup_updates=1, source_sha='a' * 40,
                    gpu_snapshot={'uuid': 'GPU-one'})
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(plan))
        batch_sizes = []
        class DatasetStub:
            def sample(self, size, **kwargs):
                batch_sizes.append(size)
                return {k: np.zeros((size, 5 if k == 'actions' else 99), np.float32)
                        for k in ('observations', 'actions', 'actor_goals', 'value_goals')}
        class AgentStub:
            network = SimpleNamespace(params={'w': np.ones(1)}, opt_state={'mu': np.zeros(1)})
            rng = np.array([0, 0], np.uint32)
            def update(self, batch):
                return self, {'loss': np.float32(1)}
            def total_loss(self, batch, params, rng=None):
                return 1.0, {'loss': np.float32(1)}
        env = mock.Mock()
        fake = AgentStub()
        def read_git(*args):
            return 'a' * 40 if args[0] == 'rev-parse' else ''
        with mock.patch.dict(os.environ, smoke.child_environment(plan, 0)), \
             mock.patch('jax.devices', return_value=[SimpleNamespace(platform='gpu')]), \
             mock.patch('jax.default_backend', return_value='gpu'), \
             mock.patch.object(smoke, 'read_git', side_effect=read_git), \
             mock.patch.object(smoke, 'barrier') as barrier, \
             mock.patch('impls.utils.env_utils.make_env_and_datasets', return_value=(env, {}, {})), \
             mock.patch('impls.utils.puzzle_datasets.PuzzleBoardGCDataset', return_value=DatasetStub()), \
             mock.patch('impls.agents.gciql.GCIQLAgent.create', return_value=fake), \
             mock.patch('impls.utils.flax_utils.save_agent', return_value=self.root / 'checkpoint'), \
             mock.patch('impls.utils.flax_utils.restore_agent_from_checkpoint', return_value=fake), \
             mock.patch('impls.main._validate_restored_checkpoint') as restore_probe, \
             mock.patch('impls.main._evaluate_tasks', return_value={'evaluation/overall_success': 0.0}) as evaluate:
            smoke.worker(manifest, 0)
        self.assertEqual(batch_sizes, [1, 1024, 1024, 1024])
        self.assertEqual([call.args[2] for call in barrier.call_args_list], ['compile', 'training', 'evaluation'])
        restore_probe.assert_called_once()
        self.assertEqual(evaluate.call_args.args[3].eval_tasks, 'all')
        self.assertEqual(evaluate.call_args.args[3].eval_episodes, 1)
        self.assertEqual(evaluate.call_args.args[3].video_episodes, 0)
        self.assertEqual(evaluate.call_args.args[3].eval_temperature, 0.0)
        env.close.assert_called_once()
        report = json.loads((self.root / 'worker_0/report.json').read_text())
        self.assertEqual(report['updates'], 2)
        self.assertEqual(report['batch_size'], 1024)
        self.assertEqual(report['checkpoint_fresh_restore'], 'pass')

    def test_inheritance_declared_only_is_not_runtime_or_formal_ready(self):
        report = audit(STUDY, REFERENCE, ['M24A-C003', 'M24A-C005'], [], {'goal_conditioning'})
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['evidence_level'], 'declared_only')
        self.assertFalse(report['formal_preflight_ready'])
        self.assertEqual(len(report['declared']), 20)
        self.assertEqual(report['runtime'], [])

    def test_inheritance_detects_nonallowed_changes_and_missing_fields(self):
        report = audit(STUDY, REFERENCE, ['M24A-C003'], [], set())
        self.assertEqual(report['status'], 'fail')
        self.assertTrue(report['errors'])
        self.assertEqual(differences({'alpha': 0.4}, {'alpha': 0.3}),
                         {'alpha': {'reference': 0.4, 'candidate': 0.3}})
        self.assertTrue(differences({'eval_gaussian': None}, {})['eval_gaussian']['missing_key'])


if __name__ == '__main__':
    unittest.main()
