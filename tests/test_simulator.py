from __future__ import annotations

import unittest

from inference_scheduler_lab.experiment import run_experiment
from inference_scheduler_lab.models import Request, ServerConfig
from inference_scheduler_lab.simulator import simulate
from inference_scheduler_lab.workload import make_bursty_workload


class SchedulerSimulatorTests(unittest.TestCase):
    def test_every_policy_completes_every_request(self) -> None:
        report = run_experiment(request_count=48, seed=7)
        self.assertTrue(all(item["requests_completed"] == 48 for item in report["results"]))

    def test_continuous_batching_improves_throughput(self) -> None:
        workload = make_bursty_workload(80, seed=19)
        fcfs = simulate(workload, "fcfs")
        continuous = simulate(workload, "continuous_batch")
        self.assertGreater(
            continuous["throughput_output_tokens_per_second"],
            fcfs["throughput_output_tokens_per_second"] * 1.20,
        )

    def test_kv_capacity_is_never_exceeded(self) -> None:
        result = simulate(make_bursty_workload(100), "slo_aware", ServerConfig(kv_capacity_tokens=3_200))
        self.assertLessEqual(result["peak_kv_utilization"], 1.0)

    def test_percentiles_and_rates_are_bounded(self) -> None:
        result = simulate(make_bursty_workload(30), "static_batch")
        self.assertGreaterEqual(result["p95_latency_ms"], result["p50_latency_ms"])
        self.assertGreaterEqual(result["slo_attainment"], 0.0)
        self.assertLessEqual(result["slo_attainment"], 1.0)
        self.assertGreater(result["jain_fairness"], 0.0)
        self.assertLessEqual(result["jain_fairness"], 1.0)

    def test_oversized_request_is_rejected(self) -> None:
        request = Request("oversized", 0, 500, 100, 100)
        with self.assertRaises(ValueError):
            simulate([request], "fcfs", ServerConfig(kv_capacity_tokens=256))

    def test_results_are_deterministic(self) -> None:
        workload = make_bursty_workload(25, seed=3)
        self.assertEqual(simulate(workload, "continuous_batch"), simulate(workload, "continuous_batch"))


if __name__ == "__main__":
    unittest.main()
