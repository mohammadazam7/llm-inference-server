"""HTTP API for the naive server, with token streaming over server-sent events.

``POST /v1/generate`` takes a prompt as token ids and returns the new tokens, either as
one JSON body or, with ``"stream": true``, as a stream of events:

    event: token
    data: {"index": 0, "token": 113}

    event: done
    data: {"tokens": 16, "finish_reason": "length"}

Requests are served one at a time, in the order they arrive, like
:func:`llm_inference_server.baseline.serve_one_at_a_time`. A client that disconnects
mid-stream frees the model when the next token cannot be delivered, instead of holding
it until ``max_new_tokens``; every later scheduler must keep that behaviour.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from llm_inference_server.runtime import Runtime, Tokens

MAX_PROMPT_TOKENS = 4096


class InvalidRequest(ValueError):
    pass


@dataclass(frozen=True)
class GenerateRequest:
    prompt: tuple[int, ...]
    max_new_tokens: int
    stop_token: int | None
    stream: bool

    @classmethod
    def parse(cls, body: Any, *, vocab_size: int, max_new_tokens_limit: int) -> GenerateRequest:
        if not isinstance(body, dict):
            raise InvalidRequest("the body must be a JSON object")
        prompt = body.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise InvalidRequest("prompt must be a non-empty list of token ids")
        if len(prompt) > MAX_PROMPT_TOKENS:
            raise InvalidRequest(f"prompt is longer than {MAX_PROMPT_TOKENS} tokens")
        if not all(_is_token(t, vocab_size) for t in prompt):
            raise InvalidRequest(f"every prompt token must be an integer in [0, {vocab_size})")
        max_new = body.get("max_new_tokens", 16)
        if not _is_int(max_new) or not 1 <= max_new <= max_new_tokens_limit:
            raise InvalidRequest(
                f"max_new_tokens must be an integer in [1, {max_new_tokens_limit}]"
            )
        stop = body.get("stop_token")
        if stop is not None and not _is_token(stop, vocab_size):
            raise InvalidRequest(f"stop_token must be an integer in [0, {vocab_size})")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise InvalidRequest("stream must be true or false")
        return cls(tuple(prompt), max_new, stop, stream)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_token(value: object, vocab_size: int) -> bool:
    return _is_int(value) and 0 <= value < vocab_size  # type: ignore[operator]


@dataclass
class Stats:
    served: int = 0
    cancelled: int = 0
    tokens: int = 0
    # Seconds each request waited for the model before its first step.
    queue_waits: list[float] = field(default_factory=list)


class Server:
    """The model, a lock that serialises access to it, and counters."""

    def __init__(self, runtime: Runtime, *, max_new_tokens_limit: int = 512) -> None:
        self.runtime = runtime
        self.max_new_tokens_limit = max_new_tokens_limit
        self.lock = asyncio.Lock()
        self.stats = Stats()

    async def _step(self, sequence: Sequence[int]) -> int:
        # A real forward pass blocks; running it in a thread keeps the event loop free
        # to accept connections and notice disconnects while the model works.
        [token] = await asyncio.to_thread(self.runtime.step, [sequence])
        return token

    async def tokens(
        self, request: GenerateRequest
    ) -> AsyncGenerator[tuple[int, str | None], None]:
        """Yield ``(token, None)`` per new token, then ``(-1, finish_reason)``.

        If the consumer stops early (the client went away and the response was torn
        down), the generator is closed at its current ``yield``: the lock is released
        before the next forward pass and the request counts as cancelled.
        """
        queued = time.perf_counter()
        async with self.lock:
            self.stats.queue_waits.append(time.perf_counter() - queued)
            finished = False
            try:
                sequence = list(request.prompt)
                reason = "length"
                for _ in range(request.max_new_tokens):
                    token = await self._step(sequence)
                    if token == request.stop_token:
                        reason = "stop"
                        break
                    sequence.append(token)
                    self.stats.tokens += 1
                    yield token, None
                finished = True
                self.stats.served += 1
            finally:
                if not finished:
                    self.stats.cancelled += 1
        yield -1, reason


class _ClosingStreamingResponse(StreamingResponse):
    """Closes the event generator however the response ends.

    When the client disconnects, Starlette stops iterating the body but leaves the
    generator suspended until garbage collection, which would keep the model locked.
    Closing it here runs its cleanup at once.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.body_iterator.aclose()  # type: ignore[attr-defined]


def _event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, separators=(', ', ': '))}\n\n"


def create_app(runtime: Runtime, *, max_new_tokens_limit: int = 512) -> Starlette:
    server = Server(runtime, max_new_tokens_limit=max_new_tokens_limit)

    async def generate(http: Request) -> Response:
        try:
            body = await http.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "the body is not valid JSON"}, status_code=400)
        try:
            request = GenerateRequest.parse(
                body,
                vocab_size=runtime.vocab_size,
                max_new_tokens_limit=server.max_new_tokens_limit,
            )
        except InvalidRequest as error:
            return JSONResponse({"error": str(error)}, status_code=422)

        if not request.stream:
            tokens: list[int] = []
            reason = None
            async for token, finish in server.tokens(request):
                if finish is None:
                    tokens.append(token)
                else:
                    reason = finish
            return JSONResponse({"tokens": tokens, "finish_reason": reason})

        async def events() -> AsyncIterator[str]:
            index = 0
            async with aclosing(server.tokens(request)) as stream:
                async for token, finish in stream:
                    if finish is None:
                        yield _event("token", {"index": index, "token": token})
                        index += 1
                    else:
                        yield _event("done", {"tokens": index, "finish_reason": finish})

        return _ClosingStreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def stats(_: Request) -> Response:
        s = server.stats
        return JSONResponse({"served": s.served, "cancelled": s.cancelled, "tokens": s.tokens})

    app = Starlette(
        routes=[
            Route("/v1/generate", generate, methods=["POST"]),
            Route("/v1/stats", stats, methods=["GET"]),
        ]
    )
    app.state.server = server
    return app


class PacedRuntime:
    """Wraps a runtime and sleeps for each step, so a simulated model takes real time.

    With ``SimulatedRuntime`` behind it, the HTTP server can be load-tested end to end
    without a GPU: a step of a batch of one costs ``step_cost + per_sequence_cost``.
    """

    def __init__(
        self, inner: Runtime, *, step_cost: float = 0.010, per_sequence_cost: float = 0.002
    ) -> None:
        self.inner = inner
        self.vocab_size = inner.vocab_size
        self._step_cost = step_cost
        self._per_sequence_cost = per_sequence_cost

    def step(self, batch: Sequence[Tokens]) -> list[int]:
        time.sleep(self._step_cost + self._per_sequence_cost * len(batch))
        return self.inner.step(batch)
