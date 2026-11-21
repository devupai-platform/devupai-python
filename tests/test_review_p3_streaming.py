"""Review P3 findings on streams (``review-p3.md``): M-1 (a dropped stream releases its response), L-1 (a consumer
exception inside a bare ``for`` loop closes the response), L-2 (the SSE line scan is linear) and L-3 (a deeply
nested frame is skipped as ``malformed_frame``).

Expected items, written before any run: M-1 sync 1 + async 1, L-1 2 (exception, break), L-2 1, L-3 2 (both client
kinds): 7 items.
"""

from __future__ import annotations

import gc
import json
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

import anyio
import httpx
import pytest
from helpers.mock_transport import MODEL_A, REQ_ID, MockAPI

from devupai import DebugEvent
from devupai._streaming import SSEParser

PARAMS: dict[str, Any] = {"model": MODEL_A, "messages": [{"role": "user", "content": "Hi"}], "stream": True}


def chunk_frame(content: str) -> str:
    frame = {
        "id": "chatcmpl-fixture",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL_A,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"


class Pieces(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body sent one SSE event per read, so the response stays open until it is read or closed (a ``content=``
    body is read whole by ``httpx`` before the response is returned)."""

    def __init__(self, body: str) -> None:
        self.pieces = [piece.encode("utf-8") + b"\n\n" for piece in body.split("\n\n") if piece]

    def __iter__(self) -> Iterator[bytes]:
        yield from self.pieces

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            yield piece


def sse_handler(body: str) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream", "x-request-id": REQ_ID}, stream=Pieces(body)
        )

    return handle


BODY = chunk_frame("a") + chunk_frame("b") + "data: [DONE]\n\n"


# --- M-1 -----------------------------------------------------------------------------------------------------------


def test_a_sync_stream_dropped_without_iteration_closes_its_response() -> None:
    client = MockAPI(sse_handler(BODY)).sync_client(max_retries=0)
    stream = client.chat.completions.create(**PARAMS)
    response = stream.response
    assert not response.is_closed
    del stream
    gc.collect()
    assert response.is_closed


def test_an_async_stream_dropped_unclosed_emits_a_resource_warning() -> None:
    client = MockAPI(sse_handler(BODY)).async_client(max_retries=0)

    async def main() -> list[warnings.WarningMessage]:
        stream = await client.chat.completions.create(**PARAMS)
        response = stream.response
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del stream
            gc.collect()
        await response.aclose()
        return caught

    caught = anyio.run(main)
    assert [w.category for w in caught if issubclass(w.category, ResourceWarning)] == [ResourceWarning]


# --- L-1 -----------------------------------------------------------------------------------------------------------


def test_a_consumer_exception_in_a_bare_for_loop_closes_the_response() -> None:
    client = MockAPI(sse_handler(BODY)).sync_client(max_retries=0)
    stream = client.chat.completions.create(**PARAMS)

    def consume() -> None:
        for _item in stream:
            raise KeyError("fixture consumer error")

    with pytest.raises(KeyError):
        consume()
    gc.collect()
    assert stream.response.is_closed  # the stream itself is still referenced here


def test_break_out_of_a_bare_for_loop_closes_the_response() -> None:
    client = MockAPI(sse_handler(BODY)).sync_client(max_retries=0)
    stream = client.chat.completions.create(**PARAMS)

    def consume() -> int:
        count = 0
        for _item in stream:
            count += 1
            break
        return count

    assert consume() == 1
    gc.collect()
    assert stream.response.is_closed


# --- L-2 -----------------------------------------------------------------------------------------------------------


def test_the_parser_scans_one_mebibyte_of_lf_events_in_linear_time() -> None:
    event = "data: " + "x" * 10 + "\n\n"
    count = (1 << 20) // len(event)
    lf_text = event * count
    crlf_text = lf_text.replace("\n", "\r\n")
    assert len(lf_text) >= 1_000_000

    start = time.perf_counter()
    lf_events = SSEParser().feed(lf_text)
    elapsed = time.perf_counter() - start

    crlf_events = SSEParser().feed(crlf_text)
    assert elapsed < 0.5, f"{elapsed:.3f} s"
    assert len(lf_events) == count
    assert lf_events == crlf_events


# --- L-3 -----------------------------------------------------------------------------------------------------------


def test_a_deeply_nested_frame_is_skipped_as_malformed(client_kind: str) -> None:
    nested = "[" * 200_000
    body = f"data: {nested}\n\n" + chunk_frame("ok") + "data: [DONE]\n\n"
    events: list[DebugEvent] = []
    client = MockAPI(sse_handler(body)).client(client_kind, max_retries=0, on_debug=events.append)

    if client_kind == "sync":
        sync_stream: Any = client.chat.completions.create(**PARAMS)
        with sync_stream:
            items = list(sync_stream)
    else:

        async def main() -> list[Any]:
            stream: Any = await client.chat.completions.create(**PARAMS)
            async with stream:
                return [item async for item in stream]

        items = anyio.run(main)
    assert [item.choices[0].delta.content for item in items] == ["ok"]
    assert events == [DebugEvent("malformed_frame", {"data": nested})]
