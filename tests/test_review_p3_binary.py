"""P3 review fixes on the binary results (review-p3 M-1 and L-4).

- M-1: a ``BinaryResponse`` that is dropped without being read gives its connection back (its response is closed
  when the object is collected). An ``AsyncBinaryResponse`` cannot close itself from ``__del__``, so dropping one that
  is still open emits a ``ResourceWarning``; dropping a closed one emits nothing.
- L-4: a failure while reading the body (``read``, ``iter_bytes`` / ``aiter_bytes``, ``write_to_file``) is raised as
  ``APIConnectionError(0, "Network error: ...")`` with the response headers and the original error as ``__cause__``;
  a read timeout is raised as ``APITimeoutError``.
"""

from __future__ import annotations

import gc
import inspect
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import anyio
import httpx
import pytest
from helpers.mock_transport import REQ_ID, MockAPI, sequence

from devupai import APIConnectionError, APITimeoutError, AsyncDevupAI, DevupAI

T = TypeVar("T")

PNG = b"\x89PNG\r\n\x1a\nfixture"


class ChunkStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body sent in the given chunks, never read up front."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class FailingStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body that sends one chunk, then raises ``error``."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __iter__(self) -> Iterator[bytes]:
        yield PNG[:4]
        raise self._error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield PNG[:4]
        raise self._error


def reply(stream: httpx.SyncByteStream) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "image/png", "x-request-id": REQ_ID, "x-fixture": "1"}, stream=stream
        )

    return handler


def client_for(kind: str, stream: httpx.SyncByteStream) -> DevupAI | AsyncDevupAI:
    return MockAPI(sequence(reply(stream))).client(kind)


def run(flow: Callable[[], Awaitable[T]]) -> T:
    return anyio.run(flow)


async def fetch(kind: str, client: DevupAI | AsyncDevupAI) -> Any:
    result: Any = client.me.avatar()
    if kind == "async":
        assert inspect.isawaitable(result)
        result = await result
    return result


# --- M-1: a dropped binary result ------------------------------------------------------------------------------------


def test_sync_binary_dropped_unread_closes_its_response() -> None:
    client = client_for("sync", ChunkStream([PNG]))
    assert isinstance(client, DevupAI)
    binary = client.me.avatar()
    response = binary.response
    assert response.is_closed is False
    del binary
    gc.collect()
    assert response.is_closed is True


def test_sync_binary_dropped_after_read_stays_closed_without_error() -> None:
    client = client_for("sync", ChunkStream([PNG]))
    assert isinstance(client, DevupAI)
    binary = client.me.avatar()
    response = binary.response
    assert binary.read() == PNG
    del binary
    gc.collect()
    assert response.is_closed is True


def test_async_binary_dropped_unclosed_emits_a_resource_warning() -> None:
    client = client_for("async", ChunkStream([PNG]))
    assert isinstance(client, AsyncDevupAI)

    async def flow() -> None:
        binary = await client.me.avatar()
        response = binary.response
        with pytest.warns(ResourceWarning, match="AsyncBinaryResponse"):
            del binary
            gc.collect()
        await response.aclose()

    run(flow)


@pytest.mark.parametrize("how", ["close", "read", "async-with"])
def test_async_binary_dropped_after_close_emits_nothing(how: str) -> None:
    client = client_for("async", ChunkStream([PNG]))
    assert isinstance(client, AsyncDevupAI)

    async def flow() -> None:
        binary = await client.me.avatar()
        if how == "close":
            await binary.close()
        elif how == "read":
            assert await binary.read() == PNG
        else:
            async with binary:
                pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del binary
            gc.collect()
        assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []

    run(flow)


# --- L-4: read failures ----------------------------------------------------------------------------------------------

CONNECTION_ERRORS = [
    pytest.param(httpx.ReadError("fixture read error"), id="ReadError"),
    pytest.param(httpx.RemoteProtocolError("fixture protocol error"), id="RemoteProtocolError"),
    pytest.param(httpx.DecodingError("fixture decoding error"), id="DecodingError"),
]


async def consume(kind: str, binary: Any, how: str, target: Path) -> None:
    if how == "read":
        if kind == "async":
            await binary.read()
        else:
            binary.read()
    elif how == "iter_bytes":
        if kind == "async":
            async for _ in binary.aiter_bytes():
                pass
        else:
            for _ in binary.iter_bytes():
                pass
    elif how == "iter_bytes_sized":
        if kind == "async":
            async for _ in binary.aiter_bytes(2):
                pass
        else:
            for _ in binary.iter_bytes(2):
                pass
    else:
        if kind == "async":
            await binary.write_to_file(target)
        else:
            binary.write_to_file(target)


HOW = ["read", "iter_bytes", "iter_bytes_sized", "write_to_file"]


@pytest.mark.parametrize("how", HOW)
@pytest.mark.parametrize("error", CONNECTION_ERRORS)
def test_a_body_read_failure_is_an_api_connection_error(
    client_kind: str, how: str, error: Exception, tmp_path: Path
) -> None:
    client = client_for(client_kind, FailingStream(error))
    seen: list[Any] = []

    async def flow() -> None:
        binary = await fetch(client_kind, client)
        seen.append(binary)
        await consume(client_kind, binary, how, tmp_path / "out.png")

    with pytest.raises(APIConnectionError) as caught:
        run(flow)
    # The connection is given back: the response is closed once the read does not complete.
    assert seen[0].response.is_closed is True
    assert type(caught.value) is APIConnectionError
    assert caught.value.status == 0
    assert caught.value.message == f"Network error: {error}"
    assert caught.value.headers["x-fixture"] == "1"
    assert caught.value.request_id == REQ_ID
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("how", HOW)
def test_a_body_read_timeout_is_an_api_timeout_error(client_kind: str, how: str, tmp_path: Path) -> None:
    error = httpx.ReadTimeout("fixture read timeout")
    client = client_for(client_kind, FailingStream(error))
    seen: list[Any] = []

    async def flow() -> None:
        binary = await fetch(client_kind, client)
        seen.append(binary)
        await consume(client_kind, binary, how, tmp_path / "out.png")

    with pytest.raises(APITimeoutError) as caught:
        run(flow)
    assert seen[0].response.is_closed is True
    assert caught.value.status == 0
    assert caught.value.headers["x-fixture"] == "1"
    assert caught.value.request_id == REQ_ID
    assert caught.value.__cause__ is error
