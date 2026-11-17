"""The one thing the server needs from a model: next-token scores for a batch.

Schedulers and caches depend on this interface and never on a model library, so they
can be tested with ``SimulatedRuntime``, which is deterministic and charges virtual time
for each forward pass the way a real model charges wall time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from llm_inference_server.clock import VirtualClock

Tokens = Sequence[int]


class Runtime(Protocol):
    vocab_size: int

    def step(self, batch: Sequence[Tokens]) -> list[int]:
        """The next token for each sequence in ``batch``, in order (greedy decoding)."""
        ...


def next_token(sequence: Tokens, vocab_size: int) -> int:
    """The simulated model: a fixed function of the whole sequence so far.

    It depends on every earlier token and its position, so a scheduler that mixes up
    sequences or drops a token produces visibly different output.
    """
    state = 17
    for position, token in enumerate(sequence):
        state = (state * 31 + token * (position + 1)) % 1_000_003
    return state % vocab_size


class SimulatedRuntime:
    """A deterministic stand-in for a model, with a cost model for its forward pass.

    One step costs ``step_cost + per_sequence_cost * len(batch)`` seconds of virtual
    time: a fixed overhead per forward pass plus a part that grows with the batch. With
    a large fixed part, batching is nearly free (a GPU); with only the per-sequence
    part, it buys nothing (a single CPU core). The split is what the experiments vary.
    """

    def __init__(
        self,
        clock: VirtualClock,
        *,
        vocab_size: int = 256,
        step_cost: float = 0.010,
        per_sequence_cost: float = 0.002,
    ) -> None:
        if vocab_size < 2 or step_cost < 0 or per_sequence_cost < 0:
            raise ValueError("vocab_size must be at least 2 and costs cannot be negative")
        self.vocab_size = vocab_size
        self._clock = clock
        self._step_cost = step_cost
        self._per_sequence_cost = per_sequence_cost
        self.batch_sizes: list[int] = []

    def step(self, batch: Sequence[Tokens]) -> list[int]:
        if not batch:
            raise ValueError("a step needs at least one sequence")
        self.batch_sizes.append(len(batch))
        self._clock.advance(self._step_cost + self._per_sequence_cost * len(batch))
        return [next_token(sequence, self.vocab_size) for sequence in batch]
