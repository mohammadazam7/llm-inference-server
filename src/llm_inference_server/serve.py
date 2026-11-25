"""``llm-serve``: run the naive server over HTTP with the simulated model.

The simulated model sleeps for its step cost, so the server behaves like one backed by a
real model of that speed and can be load-tested without a GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from llm_inference_server.api import PacedRuntime, create_app
from llm_inference_server.clock import VirtualClock
from llm_inference_server.runtime import SimulatedRuntime


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-serve", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--step-cost", type=float, default=0.010, help="seconds per step")
    parser.add_argument(
        "--per-sequence-cost", type=float, default=0.002, help="extra seconds per sequence"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    import uvicorn

    runtime = PacedRuntime(
        SimulatedRuntime(VirtualClock(), vocab_size=args.vocab_size),
        step_cost=args.step_cost,
        per_sequence_cost=args.per_sequence_cost,
    )
    app = create_app(runtime, max_new_tokens_limit=args.max_new_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
