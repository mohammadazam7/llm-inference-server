"""Replay a schedule of requests against a server and measure what clients experience.

A schedule is a seeded list of requests with arrival times, prompt lengths and output
lengths, so two servers can be compared on exactly the same traffic. The report gives
the numbers that matter to a client and to whoever pays for the hardware:

- time to first token (TTFT), from arrival, so waiting in the queue counts
- request latency, from arrival to the last token
- throughput, generated tokens per second of wall time from first arrival to last finish

Percentiles use the nearest-rank method, so a reported p95 is a latency some request
actually had.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_inference_server.baseline import Completion, Request, serve_one_at_a_time
from llm_inference_server.clock import VirtualClock
from llm_inference_server.runtime import SimulatedRuntime

Server = Callable[[SimulatedRuntime, Sequence[Request], VirtualClock], list[Completion]]


@dataclass(frozen=True)
class Traffic:
    """What the requests look like; arrival times come from the schedule functions."""

    prompt_tokens: tuple[int, int] = (16, 128)
    new_tokens: tuple[int, int] = (16, 256)
    vocab_size: int = 256


DEFAULT_TRAFFIC = Traffic()


def _requests(arrivals: Sequence[float], traffic: Traffic, rng: random.Random) -> list[Request]:
    requests = []
    for i, arrival in enumerate(arrivals):
        prompt = tuple(
            rng.randrange(traffic.vocab_size) for _ in range(rng.randint(*traffic.prompt_tokens))
        )
        requests.append(
            Request(f"r{i:05d}", prompt, rng.randint(*traffic.new_tokens), arrival=arrival)
        )
    return requests


def poisson(
    rate: float, count: int, seed: int = 0, traffic: Traffic = DEFAULT_TRAFFIC
) -> list[Request]:
    """``count`` requests arriving at random at ``rate`` per second on average."""
    if rate <= 0 or count <= 0:
        raise ValueError("rate and count must be positive")
    rng = random.Random(seed)
    now = 0.0
    arrivals = []
    for _ in range(count):
        now += rng.expovariate(rate)
        arrivals.append(now)
    return _requests(arrivals, traffic, rng)


def constant(
    rate: float, count: int, seed: int = 0, traffic: Traffic = DEFAULT_TRAFFIC
) -> list[Request]:
    """``count`` requests evenly spaced at ``rate`` per second."""
    if rate <= 0 or count <= 0:
        raise ValueError("rate and count must be positive")
    return _requests([i / rate for i in range(count)], traffic, random.Random(seed))


def burst(count: int, seed: int = 0, traffic: Traffic = DEFAULT_TRAFFIC) -> list[Request]:
    """``count`` requests all arriving at once, the worst case for a queue."""
    if count <= 0:
        raise ValueError("count must be positive")
    return _requests([0.0] * count, traffic, random.Random(seed))


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q percent at or below it."""
    if not values:
        raise ValueError("no values")
    if not 0 < q <= 100:
        raise ValueError("q must be in (0, 100]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


@dataclass(frozen=True)
class Report:
    requests: int
    tokens: int
    duration: float
    ttft_p50: float
    ttft_p95: float
    latency_p50: float
    latency_p95: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.duration if self.duration > 0 else 0.0

    def table(self) -> str:
        rows = [
            ("requests", f"{self.requests}"),
            ("generated tokens", f"{self.tokens}"),
            ("throughput", f"{self.tokens_per_second:.1f} tokens/s"),
            ("TTFT p50 / p95", f"{self.ttft_p50:.3f} s / {self.ttft_p95:.3f} s"),
            ("latency p50 / p95", f"{self.latency_p50:.3f} s / {self.latency_p95:.3f} s"),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}}  {value}" for name, value in rows)


def report(completions: Sequence[Completion]) -> Report:
    if not completions:
        raise ValueError("no completions to report on")
    start = min(c.arrival for c in completions)
    end = max(c.finished for c in completions)
    ttft = [c.time_to_first_token for c in completions]
    latency = [c.latency for c in completions]
    return Report(
        requests=len(completions),
        tokens=sum(len(c.tokens) for c in completions),
        duration=end - start,
        ttft_p50=percentile(ttft, 50),
        ttft_p95=percentile(ttft, 95),
        latency_p50=percentile(latency, 50),
        latency_p95=percentile(latency, 95),
    )


def run(
    server: Server,
    requests: Sequence[Request],
    *,
    step_cost: float = 0.010,
    per_sequence_cost: float = 0.002,
) -> Report:
    """Serve ``requests`` with a fresh simulated model and clock, and report."""
    clock = VirtualClock()
    runtime = SimulatedRuntime(clock, step_cost=step_cost, per_sequence_cost=per_sequence_cost)
    completions = server(runtime, requests, clock)
    if len(completions) != len(requests):
        raise RuntimeError(f"server returned {len(completions)} of {len(requests)} requests")
    return report(completions)


SERVERS: dict[str, Server] = {"one-at-a-time": serve_one_at_a_time}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-loadtest", description="Replay seeded traffic against a simulated server."
    )
    parser.add_argument("--server", choices=sorted(SERVERS), default="one-at-a-time")
    parser.add_argument("--schedule", choices=["poisson", "constant", "burst"], default="poisson")
    parser.add_argument("--rate", type=float, default=2.0, help="requests per second")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-cost", type=float, default=0.010, help="seconds per forward pass")
    parser.add_argument(
        "--per-sequence-cost", type=float, default=0.002, help="extra seconds per batched sequence"
    )
    args = parser.parse_args(argv)
    if args.schedule == "burst":
        requests = burst(args.requests, args.seed)
    elif args.schedule == "constant":
        requests = constant(args.rate, args.requests, args.seed)
    else:
        requests = poisson(args.rate, args.requests, args.seed)
    result = run(
        SERVERS[args.server],
        requests,
        step_cost=args.step_cost,
        per_sequence_cost=args.per_sequence_cost,
    )
    print(f"server {args.server}, {args.schedule} traffic, seed {args.seed} (simulated time)")
    print(result.table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
