from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_ms: int
    input_tokens: int
    output_tokens: int
    deadline_ms: int
    priority: int = 0


@dataclass(frozen=True)
class ServerConfig:
    token_work_per_ms: float = 42.0
    max_batch_size: int = 12
    kv_capacity_tokens: int = 6_400


@dataclass
class RuntimeRequest:
    request: Request
    remaining_prefill: float
    remaining_decode: float
    first_token_ms: int | None = None
    finished_ms: int | None = None

    @property
    def reserved_kv_tokens(self) -> int:
        return self.request.input_tokens + self.request.output_tokens
