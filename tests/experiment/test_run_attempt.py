import unittest

from impls.experiment import make_run_path


class RunAttemptPathTest(unittest.TestCase):
    def test_zero_attempt_preserves_canonical_path_and_positive_attempt_is_distinct(self):
        canonical = make_run_path(
            '/tmp/runs', 'M11B', 'M11B-C010', 'hiql_high_low_ss_antmaze_giant',
            'antmaze-giant-navigate-v0', 0,
        )
        attempt = make_run_path(
            '/tmp/runs', 'M11B', 'M11B-C010', 'hiql_high_low_ss_antmaze_giant',
            'antmaze-giant-navigate-v0', 0, run_attempt=1,
        )
        self.assertEqual(canonical.name, 'seed_000')
        self.assertEqual(attempt.name, 'seed_000__attempt_001')
        self.assertNotEqual(canonical, attempt)


if __name__ == '__main__':
    unittest.main()
