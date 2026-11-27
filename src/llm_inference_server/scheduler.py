"""Batching schedulers: static batches as a second baseline, and continuous batching.

Both run several sequences through one forward pass. They differ in when a sequence may
enter and leave the batch.

Static batching fills a batch, then runs it until its longest sequence is done. A short
request is finished early but its slot stays occupied (as padding in a real model), and
nobody waiting can use it. A request that arrives one step after the batch started waits
for the whole batch.

Continuous batching decides membership again at every decoding step: sequences that
finished or were cancelled leave, and waiting requests take the free slots in arrival
order. Nothing waits for a sequence it does not depend on.

Neither changes what a request receives. The runtime computes each sequence from its own
tokens only, so every scheduler must return exactly the tokens the one-at-a-time server
returns; the tests hold them to that.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from llm_inference_server.baseline import Completion, Request
from llm_inference_server.clock import VirtualClock
from llm_inference_server.runtime import Runtime


@dataclass(frozen=True)
class Event:
    """What one step produced for one request: a token, the end of the request, or both."""

    request_id: str
    token: int | None = None
    finished: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class StepResult:
    admitted: tuple[str, ...] = ()
    events: tuple[Event, ...] = ()
    ran: bool = False  # whether a forward pass happened


@dataclass
class _Active:
    request: Request
    sequence: list[int]
    produced: int = 0


@dataclass
class ContinuousBatcher:
    """The step-level scheduler. It knows nothing about time or transport: a caller
    submits requests, calls ``step`` in a loop and forwards the events, which is the
    same whether the caller is a simulation or an HTTP server."""

    runtime: Runtime
    max_batch: int = 8
    _waiting: deque[Request] = field(default_factory=deque, init=False)
    _active: list[_Active] = field(default_factory=list, init=False)
    _cancelled: set[str] = field(default_factory=set, init=False)
    _known: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.max_batch < 1:
            raise ValueError("max_batch must be at least 1")

    @property
    def idle(self) -> bool:
        return not self._waiting and not self._active

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(a.request.request_id for a in self._active)

    @property
    def waiting(self) -> tuple[str, ...]:
        return tuple(r.request_id for r in self._waiting)

    def submit(self, request: Request) -> None:
        if request.request_id in self._known:
            raise ValueError(f"request {request.request_id!r} was already submitted")
        self._known.add(request.request_id)
        self._waiting.append(request)

    def cancel(self, request_id: str) -> bool:
        """Ask for a request to stop. It leaves at the next step boundary, before the
        forward pass, so its slot is free for a waiting request in that same step.
        Returns False when the request is unknown or already finished."""
        if request_id not in self.active and request_id not in self.waiting:
            return False
        self._cancelled.add(request_id)
        return True

    def step(self) -> StepResult:
        events: list[Event] = []

        # 1. Cancelled requests leave, whether they were running or still waiting.
        if self._cancelled:
            for request_id in [*self.active, *self.waiting]:
                if request_id in self._cancelled:
                    events.append(Event(request_id, finished=True, cancelled=True))
            self._active = [a for a in self._active if a.request.request_id not in self._cancelled]
            self._waiting = deque(r for r in self._waiting if r.request_id not in self._cancelled)
            self._cancelled.clear()

        # 2. Waiting requests take the free slots, oldest first.
        admitted = []
        while self._waiting and len(self._active) < self.max_batch:
            request = self._waiting.popleft()
            self._active.append(_Active(request, list(request.prompt)))
            admitted.append(request.request_id)

        if not self._active:
            return StepResult(tuple(admitted), tuple(events), ran=False)

        # 3. One forward pass for everything that is running.
        tokens = self.runtime.step([a.sequence for a in self._active])

        # 4. Each sequence gets its token; finished sequences leave.
        still_running = []
        for entry, token in zip(self._active, tokens, strict=True):
            request = entry.request
            if token == request.stop_token:  # the stop token ends the request, unseen
                events.append(Event(request.request_id, finished=True))
                continue
            entry.sequence.append(token)
            entry.produced += 1
            done = entry.produced == request.max_new_tokens
            events.append(Event(request.request_id, token=token, finished=done))
            if not done:
                still_running.append(entry)
        self._active = still_running
        return StepResult(tuple(admitted), tuple(events), ran=True)


@dataclass
class _Record:
    request: Request
    started: float | None = None
    first_token: float | None = None
    finished: float | None = None
    tokens: list[int] = field(default_factory=list)
    cancelled: bool = False

    def completion(self) -> Completion:
        assert self.finished is not None
        return Completion(
            request_id=self.request.request_id,
            tokens=tuple(self.tokens),
            arrival=self.request.arrival,
            # A request cancelled in the queue never started; its end stands in for both.
            started=self.finished if self.started is None else self.started,
            first_token=self.finished if self.first_token is None else self.first_token,
            finished=self.finished,
            cancelled=self.cancelled,
        )


def _in_arrival_order(requests: Sequence[Request]) -> list[Request]:
    return sorted(requests, key=lambda r: (r.arrival, r.request_id))


def serve_continuous(
    runtime: Runtime,
    requests: Sequence[Request],
    clock: VirtualClock,
    *,
    max_batch: int = 8,
    cancellations: Mapping[str, float] | None = None,
) -> list[Completion]:
    """Replay ``requests`` through a ``ContinuousBatcher`` on the virtual clock.

    ``cancellations`` maps a request id to the time its client gave up. The request
    leaves at the first step boundary at or after that time.
    """
    batcher = ContinuousBatcher(runtime, max_batch=max_batch)
    pending = deque(_in_arrival_order(requests))
    records = {r.request_id: _Record(r) for r in requests}
    if len(records) != len(requests):
        raise ValueError("request ids must be unique")
    to_cancel = dict(cancellations or {})

    while pending or not batcher.idle:
        if batcher.idle:
            clock.advance_to(pending[0].arrival)  # nothing to do until the next arrival
        now = clock.now()
        while pending and pending[0].arrival <= now:
            batcher.submit(pending.popleft())
        for request_id, when in list(to_cancel.items()):
            if when <= now and batcher.cancel(request_id):
                del to_cancel[request_id]

        result = batcher.step()
        after = clock.now()
        for request_id in result.admitted:
            records[request_id].started = now
        for event in result.events:
            record = records[event.request_id]
            if event.cancelled:
                record.cancelled = True
                record.finished = now  # it left before the forward pass
                continue
            if event.token is not None:
                record.tokens.append(event.token)
                if record.first_token is None:
                    record.first_token = after
            if event.finished:
                record.finished = after
    return [records[r.request_id].completion() for r in _in_arrival_order(requests)]


def serve_static_batches(
    runtime: Runtime,
    requests: Sequence[Request],
    clock: VirtualClock,
    *,
    batch_size: int = 8,
    max_wait: float = 0.05,
) -> list[Completion]:
    """Fill a batch, run it until every sequence in it is done, then fill the next.

    A batch starts when it is full or when its oldest request has waited ``max_wait``
    seconds, whichever comes first (and never before the previous batch ends). A
    sequence that finishes early keeps its slot until the batch ends, as padding would
    in a real model, so every forward pass costs the full batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_wait < 0:
        raise ValueError("max_wait cannot be negative")
    pending = deque(_in_arrival_order(requests))
    records = {r.request_id: _Record(r) for r in requests}
    if len(records) != len(requests):
        raise ValueError("request ids must be unique")

    while pending:
        clock.advance_to(pending[0].arrival)
        deadline = max(clock.now(), pending[0].arrival + max_wait)
        batch: list[Request] = []
        while pending and len(batch) < batch_size and pending[0].arrival <= deadline:
            batch.append(pending.popleft())
        # A full batch leaves when its last member arrives; a partial one at the deadline.
        clock.advance_to(batch[-1].arrival if len(batch) == batch_size else deadline)

        started = clock.now()
        sequences = [list(r.prompt) for r in batch]
        running = set(range(len(batch)))
        for request in batch:
            records[request.request_id].started = started
        while running:
            tokens = runtime.step(sequences)
            now = clock.now()
            for i in sorted(running):
                request, record = batch[i], records[batch[i].request_id]
                token = tokens[i]
                if token == request.stop_token:
                    record.finished = now
                    running.discard(i)
                    continue
                sequences[i].append(token)
                record.tokens.append(token)
                if record.first_token is None:
                    record.first_token = now
                if len(record.tokens) == request.max_new_tokens:
                    record.finished = now
                    running.discard(i)
    return [records[r.request_id].completion() for r in _in_arrival_order(requests)]
