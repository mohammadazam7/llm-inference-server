"""The naive server: one request at a time, start to finish.

Every other scheduler in this project is measured against this one on the same arrival
schedule, so it is kept as simple as possible and never optimised.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from llm_inference_server.clock import VirtualClock
from llm_inference_server.runtime import Runtime, Tokens


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    arrival: float = 0.0
    stop_token: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("a request needs a prompt of at least one token")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class Completion:
    request_id: str
    tokens: tuple[int, ...]
    arrival: float
    started: float
    first_token: float
    finished: float

    @property
    def time_to_first_token(self) -> float:
        """From arrival, so time spent waiting in the queue counts."""
        return self.first_token - self.arrival

    @property
    def latency(self) -> float:
        return self.finished - self.arrival


def generate(
    runtime: Runtime, prompt: Tokens, max_new_tokens: int, stop_token: int | None = None
) -> Iterator[int]:
    """Yield new tokens one at a time. The stop token ends generation and is not emitted."""
    sequence = list(prompt)
    for _ in range(max_new_tokens):
        [token] = runtime.step([sequence])
        if token == stop_token:
            return
        sequence.append(token)
        yield token


def serve_one_at_a_time(
    runtime: Runtime, requests: Sequence[Request], clock: VirtualClock
) -> list[Completion]:
    """Serve ``requests`` in arrival order, each to completion before the next starts."""
    completions = []
    for request in sorted(requests, key=lambda r: (r.arrival, r.request_id)):
        clock.advance_to(request.arrival)  # idle until the request exists
        started = clock.now()
        first_token = None
        tokens = []
        for token in generate(runtime, request.prompt, request.max_new_tokens, request.stop_token):
            if first_token is None:
                first_token = clock.now()
            tokens.append(token)
        finished = clock.now()
        completions.append(
            Completion(
                request_id=request.request_id,
                tokens=tuple(tokens),
                arrival=request.arrival,
                started=started,
                # A request that stops before emitting anything has no first token;
                # its finish time is the honest stand-in.
                first_token=finished if first_token is None else first_token,
                finished=finished,
            )
        )
    return completions
