"""Time as a dependency.

The server reads time through a ``Clock`` so the same scheduling code can run against
the wall clock in production and against a virtual clock in tests and simulations, where
a "second" of model time passes instantly and every run gives the same numbers.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class WallClock:
    def now(self) -> float:
        return time.perf_counter()


class VirtualClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot go backwards")
        self._now += seconds

    def advance_to(self, when: float) -> None:
        """Move to ``when`` if it is in the future; an earlier time is left alone."""
        self._now = max(self._now, when)
