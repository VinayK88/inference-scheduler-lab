from __future__ import annotations

import random

from .models import Request


def make_bursty_workload(count: int = 160, seed: int = 17) -> list[Request]:
    """Create a reproducible mixture of chat, analysis, and long-context work."""
    rng = random.Random(seed)
    arrival = 0
    requests: list[Request] = []
    shapes = [
        (96, 24, 95, 0),
        (256, 64, 210, 0),
        (640, 96, 430, 1),
        (1_024, 160, 680, 2),
    ]
    weights = [0.42, 0.34, 0.18, 0.06]

    for index in range(count):
        # Dense bursts plus occasional gaps expose head-of-line blocking.
        arrival += rng.choices([0, 1, 2, 4, 11], weights=[25, 30, 22, 15, 8])[0]
        input_tokens, output_tokens, deadline, priority = rng.choices(shapes, weights=weights)[0]
        jitter = rng.randint(-12, 18)
        requests.append(
            Request(
                request_id=f"req-{index:04d}",
                arrival_ms=arrival,
                input_tokens=max(32, input_tokens + jitter),
                output_tokens=output_tokens,
                deadline_ms=deadline,
                priority=priority,
            )
        )
    return requests
