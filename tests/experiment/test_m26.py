"""M26 definitions, constrained preparation and actual production forward gates."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jax
import numpy as np

from impls.agents import agents
from impls.diagnostics.goal_conditioning import capture_goal_conditioning_inputs
from impls.experiment import load_study
from impls.experiment.management import jsonable
from impls.networks.goal_conditioning import resolve_goal_conditioning
from impls.representation.puzzle_algebra import build_toggle_matrix, gf2_inverse, gf2_rank
from tools.audit_goal_conditioning_study import resolved_configurations, production_audit
from tools.prepare_goal_coordinate_transform import prepare_transform
from tools import sweep

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml'
M24 = ROOT / 'experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml'
IDS = [f'M26-C{i:03}' for i in range(1, 11)]
MATRIX = [
    ('residual', 'residual', 'identity', 'zero'),
    ('operation', 'operation', 'identity', 'zero'),
    ('residual', 'residual', 'identity', 'exact_press_fraction'),
    ('operation', 'operation', 'identity', 'exact_press_fraction'),
    ('operation', 'operation', 'perm_v1', 'zero'),
    ('operation', 'operation', 'perm_v1', 'exact_press_fraction'),
    ('operation', 'operation', 'dense_v1', 'zero'),
    ('operation', 'operation', 'dense_v1', 'exact_press_fraction'),
    ('operation', 'residual', 'identity', 'zero'),
    ('residual', 'operation', 'identity', 'zero'),
]


class M26ConfigurationTest(unittest.TestCase):
    def test_exact_matrix_roles_and_no_lifecycle_or_seed_in_configs(self):
        study, configs = resolved_configurations(STUDY)
        self.assertEqual([c.config_id for c, _ in configs], IDS)
        self.assertEqual(study.data['environments'], ['puzzle-4x5-play-v0'])
        self.assertEqual(study.data['seeds'], [0])
        fingerprints = []
        for (c, cfg), (actor, value, transform, distance) in zip(configs, MATRIX):
            self.assertTrue(c.data['executable'])
            for forbidden in ('seed', 'seeds', 'status', 'completed', 'dependencies'):
                self.assertNotIn(forbidden, c.data)
            self.assertEqual(c.data['environment'], 'puzzle-4x5-play-v0')
            raw = c.data['agent_overrides']['goal_conditioning']
            self.assertEqual(set(raw['transforms']), set() if transform == 'identity' else {transform})
            self.assertEqual(raw['schema_version'], 2)
            self.assertEqual(raw['input_schema'], 'token_aux_v1')
            expected = {'actor': dict(coordinate=actor, transform_id=transform, distance_feature=distance),
                        'value_side': dict(coordinate=value, transform_id=transform, distance_feature=distance)}
            self.assertEqual(raw['roles'], expected)
            self.assertEqual(study.data['condition_definitions'][c.data['condition_id']],
                             {'config_id': c.config_id, **expected})
            self.assertEqual(jsonable(cfg['goal_conditioning']['roles']), expected)
            plan = resolve_goal_conditioning(cfg['goal_conditioning'])
            self.assertEqual(plan.token_aux_dim, 1)
            fingerprints.append(plan.fingerprint)
        self.assertEqual(len(set(fingerprints)), 10)

    def test_inheritance_allowlist_excludes_only_goal_conditioning(self):
        study, configs = resolved_configurations(STUDY)
        m24, references = resolved_configurations(M24, {'M24A-C003', 'M24A-C005'})
        protocol = copy.deepcopy(m24.data['protocol'])
        del protocol['formal_training_started']
        self.assertEqual(study.data['protocol'], protocol)
        fixed = copy.deepcopy(study.data['fixed_design'])
        for field in ('goal_input_schema', 'paired_token_dim', 'token_aux_dim'):
            del fixed[field]
        self.assertEqual(fixed, m24.data['fixed_design'])
        for _, ref in references:
            reference = jsonable(ref)
            del reference['goal_conditioning']
            for _, cfg in configs:
                candidate = jsonable(cfg)
                del candidate['goal_conditioning']
                self.assertEqual(candidate, reference)
        for _, cfg in configs:
            self.assertEqual(cfg['alpha'], 0.4)
            self.assertEqual(cfg['batch_size'], 1024)

    def test_initial_full_and_reference_selection_and_worker_slots(self):
        study = load_study(STUDY)
        selection = study.data['execution']['selection']
        self.assertEqual(selection['default_initial_configs'], IDS[2:])
        self.assertEqual(selection['full_alternative_configs'], IDS)
        self.assertEqual(selection['matched_reference_supplement'], IDS[:2])
        self.assertEqual(sweep._worker_slots(['1'], 2), [
            {'physical_gpu_id': '1', 'worker_slot': 0}, {'physical_gpu_id': '1', 'worker_slot': 1}])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'runs'
            for selected in (IDS[2:], IDS, IDS[:2]):
                jobs = sweep._jobs(STUDY, root, include_configs=set(selected))
                self.assertEqual([j['configuration'].config_id for j in jobs], selected)
                self.assertEqual(len({j['run_dir'] for j in jobs}), len(selected))
                for job in jobs:
                    self.assertEqual(job['status'], 'planned')
                    self.assertEqual(job['seed'], 0)
                    self.assertEqual(job['run_attempt'], 0)
                    self.assertEqual(job['run_dir'].parts[-1], 'seed_000')
                    self.assertIn('M26', job['run_dir'].parts)
            self.assertFalse(root.exists())

    def test_fixed_transform_payloads_reproduce_and_share(self):
        study, configs = resolved_configurations(STUDY)
        cases = [('perm_v1', 'permutation', 26001, {'derangement': True}, (4, 5), 2),
                 ('dense_v1', 'gf2_linear', 26002, {'min_weight': 5, 'max_weight': 15}, (6, 7), 7)]
        for name, kind, seed, kwargs, indices, accepted in cases:
            result = prepare_transform(kind=kind, num_coordinates=20, transform_seed=seed, max_attempts=10000, **kwargs)
            self.assertEqual(result['selection']['accepted_candidate_1based'], accepted)
            rng = np.random.Generator(np.random.PCG64(seed))
            for attempt in range(1, accepted + 1):
                if kind == 'permutation':
                    candidate = rng.permutation(20)
                    acceptable = bool(np.all(candidate != np.arange(20)))
                else:
                    candidate = rng.integers(0, 2, (20, 20), dtype=np.uint8)
                    acceptable = (gf2_rank(candidate) == 20 and all(
                        np.all((candidate.sum(axis) >= 5) & (candidate.sum(axis) <= 15)) for axis in (0, 1)))
                self.assertEqual(acceptable, attempt == accepted)
            np.testing.assert_array_equal(candidate, result['payload']['permutation' if kind == 'permutation' else 'matrix'])
            for i in indices:
                actual = configs[i][0].data['agent_overrides']['goal_conditioning']['transforms'][name]
                self.assertEqual(actual, result['payload'])
            self.assertEqual(study.data['transform_preparation'][name]['content_sha256'], result['payload']['content_sha256'])
            self.assertEqual(result['payload']['rank'], 20)
            if kind == 'permutation':
                self.assertTrue(np.all(np.array(result['payload']['permutation']) != np.arange(20)))
            else:
                matrix = np.array(result['payload']['matrix'])
                self.assertEqual(gf2_rank(matrix), 20)
                for axis in (0, 1):
                    self.assertTrue(np.all((matrix.sum(axis) >= 5) & (matrix.sum(axis) <= 15)))
                np.testing.assert_array_equal(matrix @ gf2_inverse(matrix) % 2, np.eye(20))

    def test_constrained_preparation_bounded_failures_and_private_rng(self):
        before = np.random.get_state()
        for kind, seed, kwargs in [('permutation', 26001, {'derangement': True}),
                                   ('gf2_linear', 26002, {'min_weight': 5, 'max_weight': 15})]:
            with self.assertRaisesRegex(ValueError, 'No acceptable'):
                prepare_transform(kind=kind, num_coordinates=20, transform_seed=seed, max_attempts=1, **kwargs)
        after = np.random.get_state()
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        for kwargs in ({'num_coordinates': 0}, {'transform_seed': -1}, {'max_attempts': 0}, {'min_weight': 1}):
            arguments = dict(kind='permutation', num_coordinates=20, transform_seed=26001, max_attempts=10000)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                prepare_transform(**arguments)

    def test_cli_prepare_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'payload.json'
            command = [sys.executable, str(ROOT / 'tools/prepare_goal_coordinate_transform.py'),
                       '--kind', 'permutation', '--num-coordinates', '20', '--transform-seed', '26001',
                       '--max-attempts', '10000', '--derangement', '--output', str(output)]
            env = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1')
            first = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = output.read_bytes()
            self.assertEqual(json.loads(before)['selection']['accepted_candidate_1based'], 2)
            self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), before)

    def test_production_width_all_conditions_identical_init_and_forward(self):
        _, configs = resolved_configurations(STUDY)
        report = production_audit(configs)
        self.assertEqual(set(report), set(IDS))
        self.assertEqual(len({v['initial_parameter_optimizer_rng_fingerprint'] for v in report.values()}), 1)
        for value in report.values():
            self.assertTrue(value['production_widths_unchanged'])
            for name, module in value['modules'].items():
                self.assertEqual(module['token_aux_shape'], [2, 20, 1])
                self.assertEqual(module['flat_shape'], [2, 203 if name in ('critic', 'target_critic') else 198])

    def test_actual_roles_preserve_physical_tokens_and_original_x_distance(self):
        _, configs = resolved_configurations(STUDY)
        rng = np.random.default_rng(92)
        def observation():
            bits = rng.integers(0, 2, (2, 20))
            buttons = np.stack([1-bits, bits, np.ones_like(bits), -np.ones_like(bits)], -1)
            return np.concatenate([rng.normal(size=(2, 19)), buttons.reshape(2, 80)], -1).astype(np.float32)
        states, goals = observation(), observation()
        actions = np.ones((2, 5), np.float32)
        residual = (states[:, 19:].reshape(2, 20, 4)[..., 1] != goals[:, 19:].reshape(2, 20, 4)[..., 1]).astype(np.uint8)
        x = residual @ gf2_inverse(build_toggle_matrix(4, 5)).T % 2
        for (c, cfg), (actor, value, transform, distance) in zip(configs, MATRIX):
            with self.subTest(config=c.config_id):
                agent = agents['gciql'].create(0, states, actions, cfg)
                kernel = agent.network.params['modules_actor']['actor_net']['adapter']['button_projection']['kernel']
                self.assertEqual(kernel.shape, (9, 128))
                captured = capture_goal_conditioning_inputs(agent, states, goals, actions)
                for name, actual in captured.items():
                    expected = residual if (actor if name == 'actor' else value) == 'residual' else x
                    if transform != 'identity':
                        payload = cfg['goal_conditioning']['transforms'][transform]
                        expected = expected[:, list(payload['permutation'])] if transform == 'perm_v1' else expected @ np.array(payload['matrix']).T % 2
                    flat = np.asarray(actual.flat_inputs)
                    np.testing.assert_array_equal(flat[:, :99], states)
                    np.testing.assert_array_equal(flat[:, 99:118], 0)
                    goal_buttons = flat[:, 118:198].reshape(2, 20, 4)
                    np.testing.assert_array_equal(goal_buttons[..., 1], expected)
                    np.testing.assert_array_equal(goal_buttons[..., 0], 1-expected)
                    np.testing.assert_array_equal(goal_buttons[..., 2:], 0)
                    aux = np.broadcast_to((x.sum(-1)/20 if distance != 'zero' else np.zeros(2))[:, None, None], (2, 20, 1))
                    np.testing.assert_allclose(actual.token_aux, aux, rtol=0, atol=3e-8)
        jax.clear_caches()


if __name__ == '__main__':
    unittest.main()
