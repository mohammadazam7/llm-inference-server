import pytest

from llm_inference_server.baseline import Request, serve_one_at_a_time
from llm_inference_server.clock import VirtualClock
from llm_inference_server.loadtest import burst, comparison, main, poisson, run, server
from llm_inference_server.runtime import SimulatedRuntime
from llm_inference_server.scheduler import (
    ContinuousBatcher,
    serve_continuous,
    serve_static_batches,
)


def fresh(step_cost: float = 0.010, per_sequence_cost: float = 0.002):
    clock = VirtualClock()
    return SimulatedRuntime(clock, step_cost=step_cost, per_sequence_cost=per_sequence_cost), clock


def request(name: str, new_tokens: int, arrival: float = 0.0, stop: int | None = None) -> Request:
    prompt = tuple(ord(c) for c in name)
    return Request(name, prompt, new_tokens, arrival=arrival, stop_token=stop)


def mixed_traffic() -> list[Request]:
    requests = poisson(rate=20, count=60, seed=5)
    # Give some requests a stop token they are likely to hit, so early stops are covered.
    return [
        Request(
            r.request_id, r.prompt, r.max_new_tokens, r.arrival, stop_token=7 if i % 3 else None
        )
        for i, r in enumerate(requests)
    ]


@pytest.mark.parametrize("name", ["static-batch", "continuous"])
@pytest.mark.parametrize("max_batch", [1, 3, 8])
def test_batching_never_changes_what_a_request_receives(name: str, max_batch: int) -> None:
    requests = mixed_traffic()
    expected = {c.request_id: c.tokens for c in serve_one_at_a_time(*_args(requests))}
    runtime, clock = fresh()
    got = server(name, max_batch)(runtime, requests, clock)
    assert {c.request_id: c.tokens for c in got} == expected
    assert max(runtime.batch_sizes) <= max_batch
    assert any(
        len(tokens) < r.max_new_tokens
        for r, tokens in zip(requests, expected.values(), strict=True)
    )


def _args(requests: list[Request]):
    runtime, clock = fresh()
    return runtime, requests, clock


def test_a_short_request_leaves_a_continuous_batch_as_soon_as_it_is_done() -> None:
    requests = [request("long", 50), request("short", 2)]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = {c.request_id: c for c in serve_continuous(runtime, requests, clock)}
    assert done["short"].finished == 2.0
    assert done["long"].finished == 50.0
    # Two steps with both sequences, then the long one alone.
    assert runtime.batch_sizes == [2, 2] + [1] * 48


def test_a_late_arrival_joins_the_running_batch_at_the_next_step() -> None:
    requests = [request("early", 10), request("late", 3, arrival=2.5)]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = {c.request_id: c for c in serve_continuous(runtime, requests, clock)}
    # Steps end at 1, 2, 3, ...; the late request is admitted at the boundary at 3.
    assert done["late"].started == 3.0
    assert done["late"].first_token == 4.0
    assert done["late"].finished == 6.0
    assert done["early"].finished == 10.0  # undisturbed


def test_the_same_late_arrival_waits_for_the_whole_static_batch() -> None:
    requests = [request("early", 10), request("late", 3, arrival=2.5)]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = {
        c.request_id: c
        for c in serve_static_batches(runtime, requests, clock, batch_size=2, max_wait=0.0)
    }
    assert done["early"].finished == 10.0
    assert done["late"].started == 10.0
    assert done["late"].finished == 13.0


def test_a_static_batch_starts_when_full_or_when_the_oldest_request_has_waited_enough() -> None:
    full = [request("a", 1, 0.0), request("b", 1, 0.2), request("c", 1, 0.3)]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = serve_static_batches(runtime, full, clock, batch_size=2, max_wait=5.0)
    assert [c.started for c in done] == [0.2, 0.2, 5.3]  # full at 0.2; c alone waits 5 s

    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = serve_static_batches(runtime, [request("a", 1, 0.0)], clock, batch_size=4, max_wait=0.5)
    assert done[0].started == 0.5


def test_a_finished_sequence_keeps_its_slot_in_a_static_batch() -> None:
    requests = [request("long", 5), request("short", 1)]
    runtime, clock = fresh()
    serve_static_batches(runtime, requests, clock, batch_size=2, max_wait=0.0)
    assert runtime.batch_sizes == [2] * 5


def test_waiting_requests_are_admitted_oldest_first() -> None:
    requests = [request(name, 2, arrival=i * 0.001) for i, name in enumerate("abcde")]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = serve_continuous(runtime, requests, clock, max_batch=2)
    starts = [(c.request_id, c.started) for c in done]
    assert starts == [("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 3.0), ("e", 4.0)]


def test_cancelling_a_running_request_frees_its_slot_for_the_next_one() -> None:
    requests = [request("a", 100), request("b", 100), request("c", 3, arrival=0.5)]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    full = serve_one_at_a_time(*_args([requests[0]]))[0].tokens
    done = {
        c.request_id: c
        for c in serve_continuous(runtime, requests, clock, max_batch=2, cancellations={"a": 3.5})
    }
    assert done["a"].cancelled
    assert done["a"].finished == 4.0  # the first step boundary after 3.5
    assert done["a"].tokens == full[:4]  # what it had received until then, nothing else
    assert done["c"].started == 4.0  # the freed slot is used in the same step
    assert not done["b"].cancelled and len(done["b"].tokens) == 100


def test_cancelling_a_waiting_request_means_it_never_runs() -> None:
    requests = [request("a", 5), request("b", 5, arrival=0.1)]
    runtime, clock = fresh(step_cost=1.0, per_sequence_cost=0.0)
    done = {
        c.request_id: c
        for c in serve_continuous(runtime, requests, clock, max_batch=1, cancellations={"b": 0.2})
    }
    assert done["b"].cancelled and done["b"].tokens == ()
    assert done["b"].finished == 1.0
    assert runtime.batch_sizes == [1] * 5


def test_the_batcher_reports_tokens_and_endings_step_by_step() -> None:
    runtime, _ = fresh()
    batcher = ContinuousBatcher(runtime, max_batch=2)
    assert batcher.idle and batcher.step().ran is False
    for name, new_tokens in (("a", 1), ("b", 2), ("c", 1)):
        batcher.submit(request(name, new_tokens))
    first = batcher.step()
    assert first.admitted == ("a", "b") and batcher.waiting == ("c",)
    assert [(e.request_id, e.finished) for e in first.events] == [("a", True), ("b", False)]
    second = batcher.step()
    assert second.admitted == ("c",)
    assert {e.request_id for e in second.events if e.finished} == {"b", "c"}
    assert batcher.idle


def test_the_batcher_refuses_bad_input() -> None:
    runtime, _ = fresh()
    with pytest.raises(ValueError, match="max_batch"):
        ContinuousBatcher(runtime, max_batch=0)
    batcher = ContinuousBatcher(runtime)
    batcher.submit(request("a", 1))
    with pytest.raises(ValueError, match="already submitted"):
        batcher.submit(request("a", 1))
    assert batcher.cancel("nobody") is False
    with pytest.raises(ValueError, match="batch_size"):
        serve_static_batches(runtime, [], VirtualClock(), batch_size=0)
    with pytest.raises(ValueError, match="max_wait"):
        serve_static_batches(runtime, [], VirtualClock(), max_wait=-1)
    with pytest.raises(ValueError, match="unique"):
        serve_continuous(runtime, [request("a", 1), request("a", 2)], VirtualClock())
    with pytest.raises(ValueError, match="unique"):
        serve_static_batches(runtime, [request("a", 1), request("a", 2)], VirtualClock())
    with pytest.raises(ValueError, match="unknown server"):
        server("fastest")


def test_batching_pays_when_a_forward_pass_has_a_fixed_cost() -> None:
    requests = burst(40, seed=2)
    alone = run(server("one-at-a-time"), requests, step_cost=0.010, per_sequence_cost=0.001)
    static = run(server("static-batch"), requests, step_cost=0.010, per_sequence_cost=0.001)
    continuous = run(server("continuous"), requests, step_cost=0.010, per_sequence_cost=0.001)
    assert continuous.tokens == static.tokens == alone.tokens
    assert continuous.tokens_per_second > static.tokens_per_second > alone.tokens_per_second
    assert continuous.tokens_per_second > 3 * alone.tokens_per_second


def test_batching_buys_no_throughput_when_cost_grows_with_the_batch() -> None:
    requests = burst(40, seed=2)
    alone = run(server("one-at-a-time"), requests, step_cost=0.0, per_sequence_cost=0.010)
    continuous = run(server("continuous"), requests, step_cost=0.0, per_sequence_cost=0.010)
    assert continuous.tokens_per_second == pytest.approx(alone.tokens_per_second)
    # It still starts answering sooner, because nobody queues behind a long request.
    assert continuous.ttft_p50 < alone.ttft_p50


def test_cli_compares_all_servers_on_the_same_traffic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--server", "all", "--requests", "30", "--rate", "5", "--seed", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "poisson traffic, 30 requests, seed 1 (simulated time)"
    assert [line.split()[0] for line in lines[2:]] == [
        "one-at-a-time",
        "static-batch",
        "continuous",
    ]
    assert "tokens/s" in comparison({})
