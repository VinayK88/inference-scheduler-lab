from __future__ import annotations

import math
from statistics import mean

from .models import Request, RuntimeRequest, ServerConfig

POLICIES = ("fcfs", "static_batch", "continuous_batch", "slo_aware")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _jain_index(values: list[float]) -> float:
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value**2 for value in values)
    return numerator / denominator if denominator else 1.0


def _queue_key(runtime: RuntimeRequest, policy: str, now: int, config: ServerConfig) -> tuple:
    request = runtime.request
    if policy != "slo_aware":
        return (request.arrival_ms, request.request_id)
    predicted_ms = (runtime.remaining_prefill + runtime.remaining_decode) / config.token_work_per_ms
    slack = request.deadline_ms - (now - request.arrival_ms) - predicted_ms
    return (slack, -request.priority, predicted_ms, request.arrival_ms)


def _can_admit(runtime: RuntimeRequest, active: list[RuntimeRequest], config: ServerConfig) -> bool:
    reserved = sum(item.reserved_kv_tokens for item in active)
    return reserved + runtime.reserved_kv_tokens <= config.kv_capacity_tokens


def _admit(
    waiting: list[RuntimeRequest],
    active: list[RuntimeRequest],
    policy: str,
    now: int,
    config: ServerConfig,
) -> None:
    if policy == "fcfs":
        limit = 1
    else:
        limit = config.max_batch_size
    if policy == "static_batch" and active:
        return

    waiting.sort(key=lambda item: _queue_key(item, policy, now, config))
    made_progress = True
    while waiting and len(active) < limit and made_progress:
        made_progress = False
        for index, candidate in enumerate(waiting):
            if _can_admit(candidate, active, config):
                active.append(waiting.pop(index))
                made_progress = True
                break


def simulate(
    requests: list[Request],
    policy: str,
    config: ServerConfig | None = None,
) -> dict:
    """Run a 1 ms discrete-event approximation of an LLM serving loop."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if not requests:
        raise ValueError("at least one request is required")
    config = config or ServerConfig()
    arrivals = sorted(requests, key=lambda item: (item.arrival_ms, item.request_id))
    if any(item.input_tokens + item.output_tokens > config.kv_capacity_tokens for item in arrivals):
        raise ValueError("a request exceeds KV capacity")

    waiting: list[RuntimeRequest] = []
    active: list[RuntimeRequest] = []
    finished: list[RuntimeRequest] = []
    arrival_index = 0
    now = arrivals[0].arrival_ms
    peak_reserved = 0
    max_ticks = 1_000_000

    for _ in range(max_ticks):
        while arrival_index < len(arrivals) and arrivals[arrival_index].arrival_ms <= now:
            request = arrivals[arrival_index]
            waiting.append(RuntimeRequest(request, float(request.input_tokens), float(request.output_tokens)))
            arrival_index += 1

        _admit(waiting, active, policy, now, config)
        peak_reserved = max(peak_reserved, sum(item.reserved_kv_tokens for item in active))

        if active:
            # Batching improves aggregate accelerator utilization but divides the
            # available work fairly across live sequences.
            aggregate_work = config.token_work_per_ms * (1.0 + 0.62 * math.log2(len(active)))
            share = aggregate_work / len(active)
            completed: list[RuntimeRequest] = []
            for runtime in active:
                budget = share
                if runtime.remaining_prefill > 0:
                    consumed = min(budget, runtime.remaining_prefill)
                    runtime.remaining_prefill -= consumed
                    budget -= consumed
                if budget > 0 and runtime.remaining_prefill <= 1e-9 and runtime.remaining_decode > 0:
                    consumed = min(budget, runtime.remaining_decode)
                    runtime.remaining_decode -= consumed
                    if consumed > 0 and runtime.first_token_ms is None:
                        runtime.first_token_ms = now + 1
                if runtime.remaining_decode <= 1e-9:
                    runtime.finished_ms = now + 1
                    if runtime.first_token_ms is None:
                        runtime.first_token_ms = now + 1
                    completed.append(runtime)
            for runtime in completed:
                active.remove(runtime)
                finished.append(runtime)

        if len(finished) == len(arrivals):
            break
        now += 1
    else:
        raise RuntimeError("simulation did not converge")

    latencies = [item.finished_ms - item.request.arrival_ms for item in finished]  # type: ignore[operator]
    ttfts = [item.first_token_ms - item.request.arrival_ms for item in finished]  # type: ignore[operator]
    slowdowns = [
        (item.request.input_tokens + item.request.output_tokens) / latency
        for item, latency in zip(finished, latencies)
    ]
    slo_met = [latency <= item.request.deadline_ms for item, latency in zip(finished, latencies)]
    makespan = max(item.finished_ms for item in finished) - min(item.request.arrival_ms for item in finished)  # type: ignore[arg-type]
    output_tokens = sum(item.request.output_tokens for item in finished)

    return {
        "policy": policy,
        "requests_completed": len(finished),
        "throughput_output_tokens_per_second": output_tokens * 1_000 / makespan,
        "mean_latency_ms": mean(latencies),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p95_time_to_first_token_ms": _percentile(ttfts, 0.95),
        "slo_attainment": mean(slo_met),
        "jain_fairness": _jain_index(slowdowns),
        "peak_kv_utilization": peak_reserved / config.kv_capacity_tokens,
        "makespan_ms": makespan,
    }
