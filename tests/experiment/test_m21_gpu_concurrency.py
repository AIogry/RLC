import csv
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import m21_gpu_concurrency_benchmark as benchmark


class M21GpuConcurrencyBenchmarkTest(unittest.TestCase):
    def test_frozen_job_plan_and_gpu_specific_command(self):
        jobs = benchmark._benchmark_job_plan('2', '5')
        self.assertEqual(
            [(job.job_id, job.physical_gpu_id, job.worker_slot) for job in jobs],
            [
                ('M21-SINGLE', '2', 0),
                ('M21-DUAL-A', '5', 0),
                ('M21-DUAL-B', '5', 1),
            ],
        )
        command = benchmark._benchmark_command(
            '/tmp/study.yaml', '/tmp/config.yaml', '/tmp/m21', jobs[1],
        )
        self.assertIn('--train_steps', command)
        self.assertEqual(command[command.index('--train_steps') + 1], '50000')
        self.assertEqual(command[command.index('--batch_size') + 1], '1024')
        self.assertEqual(command[command.index('--log_interval') + 1], '1000')
        self.assertEqual(command[command.index('--eval_tasks') + 1], 'none')
        self.assertIn('--no-save-best-checkpoint', command)
        self.assertIn('--no-save-last-checkpoint', command)

    def _write_train_csv(self, path, *, omit_step=None, bad_interval=False):
        fields = ('step', 'time/interval_seconds', 'time/total_seconds', 'training/loss')
        with Path(path).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for index, step in enumerate(
                range(benchmark.LOG_INTERVAL, benchmark.TRAIN_STEPS + 1, benchmark.LOG_INTERVAL),
                start=1,
            ):
                if step == omit_step:
                    continue
                interval = -0.01 if bad_interval and step == 6000 else 0.01
                writer.writerow({
                    'step': step,
                    'time/interval_seconds': interval,
                    'time/total_seconds': index * 10.0,
                    'training/loss': 1.0,
                })

    def test_throughput_uses_warmup_to_final_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            train_path = Path(temporary) / 'train.csv'
            self._write_train_csv(train_path)
            result = benchmark._throughput_summary(train_path)

        self.assertTrue(result['complete'])
        self.assertTrue(result['timing_valid'])
        self.assertEqual(result['row_count'], 50)
        self.assertEqual(result['startup_compile_proxy_seconds'], 10.0)
        self.assertEqual(result['warmup_elapsed_seconds'], 50.0)
        self.assertEqual(result['steady_elapsed_seconds'], 450.0)
        self.assertEqual(result['interval_steps_per_sec']['count'], 45)
        self.assertEqual(result['interval_steps_per_sec']['mean'], 100.0)
        self.assertEqual(result['interval_steps_per_sec']['p50'], 100.0)
        self.assertEqual(result['aggregate_window_steps_per_sec'], 100.0)

    def test_incomplete_or_invalid_training_curve_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            incomplete = Path(temporary) / 'incomplete.csv'
            self._write_train_csv(incomplete, omit_step=10000)
            incomplete_result = benchmark._throughput_summary(incomplete)
            invalid = Path(temporary) / 'invalid.csv'
            self._write_train_csv(invalid, bad_interval=True)
            invalid_result = benchmark._throughput_summary(invalid)

        self.assertFalse(incomplete_result['complete'])
        self.assertNotEqual(incomplete_result['observed_steps'], incomplete_result['expected_steps'])
        self.assertFalse(invalid_result['complete'])
        self.assertFalse(invalid_result['timing_valid'])

    def test_output_root_cannot_be_inside_formal_m20b_root(self):
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._assert_safe_output_root(
                '/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M20B/child'
            )

    def test_cpu_jax_platform_is_rejected_for_child(self):
        with patch.dict(benchmark.os.environ, {'JAX_PLATFORMS': 'gpu,cpu'}, clear=False):
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark._child_environment(benchmark.BENCHMARK_JOBS[0])

    @unittest.skipIf(benchmark.psutil is None, 'psutil is required for system telemetry')
    def test_telemetry_process_stops_normally(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = benchmark.mp.get_context('spawn')
            stop_event = context.Event()
            process = context.Process(
                target=benchmark._system_telemetry_worker,
                args=(Path(temporary) / 'system_telemetry.csv', stop_event, 0.05),
            )
            process.start()
            time.sleep(0.25)
            result = benchmark._stop_telemetry_process(process, stop_event)

            self.assertTrue(result['normal_termination'])
            self.assertEqual(result['exit_code'], 0)
            self.assertFalse(result['alive_after_join'])
            self.assertTrue((Path(temporary) / 'system_telemetry.csv').is_file())

    def test_abnormal_training_process_exit_is_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_file = (Path(temporary) / 'child.log').open('w')
            process = subprocess.Popen(
                [sys.executable, '-c', 'raise SystemExit(7)'],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record = {
                'process': process,
                'log_file': log_file,
                'pid': process.pid,
                'start_epoch': time.time(),
                'start_time': benchmark._utc_now(),
                'max_cpu_percent': None,
                'max_rss_bytes': None,
            }
            records = benchmark._monitor_training_jobs({'bad': record})

        self.assertEqual(records['bad']['exit_code'], 7)
        self.assertIsNotNone(records['bad']['end_time'])
        self.assertGreaterEqual(records['bad']['wall_seconds'], 0.0)


if __name__ == '__main__':
    unittest.main()
