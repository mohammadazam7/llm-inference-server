from itertools import pairwise

import pytest

from llm_inference_server.baseline import Request, generate, serve_one_at_a_time
from llm_inference_server.clock import VirtualClock
from llm_inference_server.runtime import SimulatedRuntime, next_token


def _runtime(clock: VirtualClock, **costs: float) -> SimulatedRuntime:
    return SimulatedRuntime(clock, **{"step_cost": 0.010, "per_sequence_cost": 0.002, **costs})


def test_virtual_clock_only_moves_forward_when_told() -> None:
    clock = VirtualClock()
    clock.advance(1.5)
    clock.advance_to(1.0)
    assert clock.now() == 1.5
    clock.advance_to(4.0)
    assert clock.now() == 4.0
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)


def test_the_simulated_model_depends_on_every_token_and_its_position() -> None:
    base = next_token([5, 6, 7], 256)
    assert next_token([5, 6, 7], 256) == base
    assert next_token([6, 5, 7], 256) != base
    assert next_token([5, 6, 8], 256) != base
    assert all(0 <= next_token([i, i + 1], 50) < 50 for i in range(200))


def test_a_step_charges_fixed_plus_per_sequence_time_and_records_the_batch() -> None:
    clock = VirtualClock()
    runtime = _runtime(clock)
    out = runtime.step([[1, 2], [3], [4, 5, 6]])
    assert out == [next_token(s, 256) for s in ([1, 2], [3], [4, 5, 6])]
    assert clock.now() == pytest.approx(0.010 + 3 * 0.002)
    assert runtime.batch_sizes == [3]
    with pytest.raises(ValueError, match="at least one"):
        runtime.step([])


def test_generate_feeds_each_token_back_in() -> None:
    runtime = _runtime(VirtualClock())
    tokens = list(generate(runtime, [1, 2, 3], max_new_tokens=4))
    sequence = [1, 2, 3]
    for token in tokens:
        assert token == next_token(sequence, 256)
        sequence.append(token)
    assert len(tokens) == 4


def test_generation_ends_at_the_stop_token_without_emitting_it() -> None:
    runtime = _runtime(VirtualClock())
    full = list(generate(runtime, [9], max_new_tokens=6))
    stopped = list(generate(runtime, [9], max_new_tokens=6, stop_token=full[2]))
    assert stopped == full[:2]


def test_requests_are_served_in_arrival_order_and_never_overlap() -> None:
    clock = VirtualClock()
    runtime = _runtime(clock)
    requests = [
        Request("late", (3,), max_new_tokens=5, arrival=0.5),
        Request("early", (1,), max_new_tokens=10, arrival=0.0),
        Request("middle", (2,), max_new_tokens=5, arrival=0.01),
    ]
    done = serve_one_at_a_time(runtime, requests, clock)
    assert [c.request_id for c in done] == ["early", "middle", "late"]
    assert all(size == 1 for size in runtime.batch_sizes)
    for before, after in pairwise(done):
        assert after.started >= before.finished


def test_a_request_behind_a_long_one_waits_for_all_of_it() -> None:
    clock = VirtualClock()
    done = serve_one_at_a_time(
        _runtime(clock),
        [
            Request("long", (1,), max_new_tokens=100, arrival=0.0),
            Request("short", (2,), max_new_tokens=1, arrival=0.0),
        ],
        clock,
    )
    step = 0.012
    long, short = done
    assert long.time_to_first_token == pytest.approx(step)
    assert long.latency == pytest.approx(100 * step)
    # The short request needs one step but waits for the long one's hundred.
    assert short.time_to_first_token == pytest.approx(101 * step)
    assert short.latency == pytest.approx(101 * step)


def test_the_server_idles_until_a_request_arrives() -> None:
    clock = VirtualClock()
    [done] = serve_one_at_a_time(
        _runtime(clock), [Request("r", (1,), max_new_tokens=2, arrival=3.0)], clock
    )
    assert done.started == 3.0
    assert done.time_to_first_token == pytest.approx(0.012)
    assert done.finished == pytest.approx(3.024)


def test_outputs_do_not_depend_on_what_else_was_served() -> None:
    alone_clock = VirtualClock()
    [alone] = serve_one_at_a_time(
        _runtime(alone_clock), [Request("a", (7, 8), max_new_tokens=6)], alone_clock
    )
    clock = VirtualClock()
    done = serve_one_at_a_time(
        _runtime(clock),
        [Request("b", (1,), max_new_tokens=3), Request("a", (7, 8), max_new_tokens=6, arrival=0.1)],
        clock,
    )
    assert next(c for c in done if c.request_id == "a").tokens == alone.tokens


def test_a_request_that_stops_at_once_has_no_tokens_and_a_defined_first_token_time() -> None:
    clock = VirtualClock()
    first = next_token([4], 256)
    [done] = serve_one_at_a_time(
        _runtime(clock), [Request("r", (4,), max_new_tokens=5, stop_token=first)], clock
    )
    assert done.tokens == ()
    assert done.first_token == done.finished


def test_invalid_requests_and_runtimes_are_refused() -> None:
    with pytest.raises(ValueError, match="prompt"):
        Request("r", (), max_new_tokens=1)
    with pytest.raises(ValueError, match="positive"):
        Request("r", (1,), max_new_tokens=0)
    with pytest.raises(ValueError, match="vocab_size"):
        SimulatedRuntime(VirtualClock(), vocab_size=1)
