from __future__ import annotations

from .models import ServerConfig
from .simulator import POLICIES, simulate
from .workload import make_bursty_workload


def run_experiment(request_count: int = 160, seed: int = 17) -> dict:
    workload = make_bursty_workload(request_count, seed)
    config = ServerConfig()
    results = [simulate(workload, policy, config) for policy in POLICIES]
    baseline = next(item for item in results if item["policy"] == "fcfs")
    for result in results:
        result["throughput_gain_vs_fcfs"] = (
            result["throughput_output_tokens_per_second"]
            / baseline["throughput_output_tokens_per_second"]
            - 1.0
        )
        result["p95_latency_change_vs_fcfs"] = (
            result["p95_latency_ms"] / baseline["p95_latency_ms"] - 1.0
        )
    return {
        "experiment": "inference-scheduler-policy-comparison-v0.1",
        "workload": {
            "request_count": request_count,
            "seed": seed,
            "arrival_pattern": "bursty",
            "request_mix": "chat, analysis, long-context",
        },
        "server": {
            "token_work_per_ms": config.token_work_per_ms,
            "max_batch_size": config.max_batch_size,
            "kv_capacity_tokens": config.kv_capacity_tokens,
        },
        "results": results,
    }
