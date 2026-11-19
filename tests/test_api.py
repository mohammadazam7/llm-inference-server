import asyncio
import json
from typing import Any

import httpx
import pytest
from starlette.requests import ClientDisconnect

from llm_inference_server.api import PacedRuntime, create_app
from llm_inference_server.baseline import generate
from llm_inference_server.clock import VirtualClock
from llm_inference_server.runtime import SimulatedRuntime

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _runtime() -> SimulatedRuntime:
    return SimulatedRuntime(VirtualClock())


def _expected(prompt: list[int], n: int, stop: int | None = None) -> list[int]:
    return list(generate(_runtime(), prompt, n, stop))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_returns_the_same_tokens_as_the_baseline() -> None:
    async with _client(create_app(_runtime())) as client:
        response = await client.post(
            "/v1/generate", json={"prompt": [1, 2, 3], "max_new_tokens": 8}
        )
    assert response.status_code == 200
    assert response.json() == {"tokens": _expected([1, 2, 3], 8), "finish_reason": "length"}


async def test_streams_one_event_per_token_then_done() -> None:
    async with _client(create_app(_runtime())) as client:
        response = await client.post(
            "/v1/generate", json={"prompt": [5, 6], "max_new_tokens": 5, "stream": True}
        )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    tokens = [data["token"] for name, data in events if name == "token"]
    assert tokens == _expected([5, 6], 5)
    assert [data["index"] for name, data in events if name == "token"] == list(range(5))
    assert events[-1] == ("done", {"tokens": 5, "finish_reason": "length"})


async def test_the_stop_token_ends_the_stream_and_is_not_sent() -> None:
    full = _expected([9], 10)
    stop = full[3]
    first = full.index(stop)
    async with _client(create_app(_runtime())) as client:
        response = await client.post(
            "/v1/generate",
            json={"prompt": [9], "max_new_tokens": 10, "stop_token": stop, "stream": True},
        )
    events = _events(response.text)
    assert [d["token"] for n, d in events if n == "token"] == full[:first]
    assert events[-1] == ("done", {"tokens": first, "finish_reason": "stop"})


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"prompt": []}, "prompt"),
        ({"prompt": [1, 999]}, "prompt token"),
        ({"prompt": [1], "max_new_tokens": 0}, "max_new_tokens"),
        ({"prompt": [1], "max_new_tokens": True}, "max_new_tokens"),
        ({"prompt": [1], "max_new_tokens": 10_000}, "max_new_tokens"),
        ({"prompt": [1], "stop_token": -1}, "stop_token"),
        ({"prompt": [1], "stream": "yes"}, "stream"),
        ([1, 2], "JSON object"),
    ],
)
async def test_rejects_invalid_requests(body: object, message: str) -> None:
    async with _client(create_app(_runtime())) as client:
        response = await client.post("/v1/generate", json=body)
    assert response.status_code == 422
    assert message in response.json()["error"]


async def test_rejects_a_body_that_is_not_json() -> None:
    async with _client(create_app(_runtime())) as client:
        response = await client.post("/v1/generate", content=b"{nope")
    assert response.status_code == 400


async def test_a_disconnected_client_frees_the_model() -> None:
    runtime = _runtime()
    app = create_app(runtime)
    body = json.dumps({"prompt": [1], "max_new_tokens": 200, "stream": True}).encode()
    tokens_sent = 0
    three_sent = asyncio.Event()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await three_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal tokens_sent
        if message["type"] == "http.response.body" and b"event: token" in message.get("body", b""):
            tokens_sent += 1
            if tokens_sent == 3:
                three_sent.set()
            # Give the disconnect listener a chance to run, as a real network would.
            await asyncio.sleep(0)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/generate",
        "raw_path": b"/v1/generate",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)

    stats = app.state.server.stats
    assert stats.cancelled == 1
    assert stats.served == 0
    # Stopped within a step or two of the disconnect, nowhere near 200 tokens.
    assert len(runtime.batch_sizes) < 10

    # The lock was released: the next request is served normally.
    async with _client(app) as client:
        response = await client.post("/v1/generate", json={"prompt": [2], "max_new_tokens": 3})
    assert response.json()["tokens"] == _expected([2], 3)
    assert app.state.server.stats.served == 1


async def test_requests_are_served_one_at_a_time() -> None:
    runtime = PacedRuntime(_runtime(), step_cost=0.002, per_sequence_cost=0.0)
    app = create_app(runtime)
    async with _client(app) as client:
        responses = await asyncio.gather(
            *[
                client.post("/v1/generate", json={"prompt": [i + 1], "max_new_tokens": 4})
                for i in range(3)
            ]
        )
    for i, response in enumerate(responses):
        assert response.json()["tokens"] == _expected([i + 1], 4)
    waits = sorted(app.state.server.stats.queue_waits)
    # The first request found the model free; the others waited for at least one request.
    assert waits[0] < 0.004
    assert waits[1] >= 4 * 0.002 * 0.9
    async with _client(app) as client:
        stats = (await client.get("/v1/stats")).json()
    assert stats == {"served": 3, "cancelled": 0, "tokens": 12}


async def test_a_failed_send_also_frees_the_model() -> None:
    """ASGI 2.4 servers report a gone client by failing the send instead."""
    runtime = _runtime()
    app = create_app(runtime)
    body = json.dumps({"prompt": [1], "max_new_tokens": 200, "stream": True}).encode()
    sent = 0

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal sent
        if message["type"] == "http.response.body":
            sent += 1
            if sent > 3:
                raise OSError("connection reset")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/generate",
        "raw_path": b"/v1/generate",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    with pytest.raises(ClientDisconnect):
        await app(scope, receive, send)
    assert app.state.server.stats.cancelled == 1
    assert not app.state.server.lock.locked()
    assert len(runtime.batch_sizes) == 4
