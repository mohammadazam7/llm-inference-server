import pytest

from llm_inference_server.baseline import Completion, serve_one_at_a_time
from llm_inference_server.loadtest import (
    Traffic,
    burst,
    constant,
    main,
    percentile,
    poisson,
    report,
    run,
)


def test_schedules_are_reproducible_and_seed_dependent() -> None:
    assert poisson(5, 50, seed=1) == poisson(5, 50, seed=1)
    assert poisson(5, 50, seed=1) != poisson(5, 50, seed=2)


def test_poisson_arrivals_average_the_requested_rate() -> None:
    requests = poisson(rate=4.0, count=4000, seed=3)
    assert requests[-1].arrival / len(requests) == pytest.approx(0.25, rel=0.05)
    arrivals = [r.arrival for r in requests]
    assert arrivals == sorted(arrivals)


def test_constant_and_burst_shapes() -> None:
    assert [r.arrival for r in constant(2.0, 4)] == [0.0, 0.5, 1.0, 1.5]
    assert {r.arrival for r in burst(10)} == {0.0}


def test_request_sizes_stay_inside_the_traffic_ranges() -> None:
    traffic = Traffic(prompt_tokens=(3, 5), new_tokens=(7, 9), vocab_size=11)
    for request in poisson(10, 300, seed=4, traffic=traffic):
        assert 3 <= len(request.prompt) <= 5
        assert 7 <= request.max_new_tokens <= 9
        assert all(0 <= t < 11 for t in request.prompt)


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 50) == 5.0
    assert percentile(values, 95) == 10.0
    assert percentile(values, 100) == 10.0
    assert percentile([7.0], 50) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile(values, 0)


def test_report_measures_from_arrival_and_counts_tokens() -> None:
    done = [
        Completion("a", (1, 2), arrival=0.0, started=0.0, first_token=0.5, finished=1.0),
        Completion("b", (3,), arrival=0.5, started=1.0, first_token=1.5, finished=3.0),
    ]
    result = report(done)
    assert (result.requests, result.tokens, result.duration) == (2, 3, 3.0)
    assert result.tokens_per_second == pytest.approx(1.0)
    assert (result.ttft_p50, result.ttft_p95) == (0.5, 1.0)
    assert (result.latency_p50, result.latency_p95) == (1.0, 2.5)


def test_queueing_shows_up_as_load_rises() -> None:
    light = run(serve_one_at_a_time, poisson(0.2, 100, seed=5))
    heavy = run(serve_one_at_a_time, poisson(2.0, 100, seed=5))
    # Same requests, ten times closer together: the baseline queues and TTFT explodes.
    assert heavy.ttft_p95 > 10 * light.ttft_p95
    assert heavy.tokens == light.tokens


def test_the_same_traffic_gives_the_same_numbers() -> None:
    first = run(serve_one_at_a_time, poisson(1.0, 50, seed=6))
    assert run(serve_one_at_a_time, poisson(1.0, 50, seed=6)) == first


def test_cli_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--requests", "20", "--rate", "1", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "poisson traffic, 20 requests, seed 7 (simulated time)\nserver one-at-a-time"
    )
    assert "TTFT p50 / p95" in out and "tokens/s" in out
