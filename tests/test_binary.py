"""Binary results (contract §4): ``BinaryResponse`` / ``AsyncBinaryResponse``, ``images.download``, ``me.avatar`` and
``inference.run(response_type="binary")``, on both client kinds.

The mock answers binary requests with a streamed body (a fresh stream per call), so the tests see the body unread
when the method returns, and read it through the result. Every async flow runs inside one event loop: the request,
the reads and the close.

The binary classes are looked up on ``devupai`` inside the tests, so each test reports on its own while they are
missing.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

import anyio
import httpx
import pytest
from helpers.mock_transport import (
    BASE,
    MODEL_A,
    REQ_ID,
    TEST_KEY,
    Entry,
    MockAPI,
    error_response,
    json_response,
    sequence,
)

from devupai import (
    APIResponse,
    AsyncDevupAI,
    DevupAI,
    DevupAPIError,
    InternalServerError,
    NotFoundError,
)

T = TypeVar("T")

ORIGIN = "https://api.example.test"
PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10])
ETAG = '"fixture-etag"'
SIGNED = f"{BASE}/images/proxy?d=fixture-token&s=fixture-signature"
REJECTED_MESSAGE = "images.download only fetches URLs returned by the API."

#: ``final.test.ts``: URLs that must be sent exactly as given, query included.
RAW_URLS = [
    f"{BASE}/images/proxy?d=abc&s=%2F",
    f"{BASE}/images/proxy?d=abc&s=%20",
    f"{BASE}/images/proxy?d=abc&s=%25",
    f"{BASE}/images/proxy?d=abc&s=+",
    f"{BASE}/images/proxy?d=abc&s==",
    f"{BASE}/images/proxy?d=signed%2Bquery&s=sig",
    f"{BASE}/images/proxy?d=nested&s=b&extra=1",
]


# --- helpers --------------------------------------------------------------------------------------------------------


def _devupai() -> Any:
    return importlib.import_module("devupai")


def binary_class(kind: str) -> type[Any]:
    """``devupai.BinaryResponse`` for the sync client, ``devupai.AsyncBinaryResponse`` for the async one."""
    name = "BinaryResponse" if kind == "sync" else "AsyncBinaryResponse"
    cls: type[Any] = getattr(_devupai(), name)
    return cls


class ChunkStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A response body sent in the given chunks, readable by both transports, and never read up front."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def binary_reply(
    content: bytes = PNG,
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
    chunks: list[bytes] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """An entry for ``sequence``: a streamed binary answer, built fresh for each call."""
    merged: dict[str, str] = {"content-type": "image/png", "x-request-id": REQ_ID}
    if headers is not None:
        merged.update(headers)
    parts = [content] if chunks is None else chunks

    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=merged, stream=ChunkStream(list(parts)))

    return reply


def not_modified() -> httpx.Response:
    return httpx.Response(304, headers={"etag": ETAG, "x-request-id": REQ_ID})


class Capture:
    """A mock handler that keeps each ``httpx.Request`` and answers from a sequence."""

    def __init__(self, *entries: Entry) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = sequence(*entries)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def setup(kind: str, *entries: Entry, **opts: Any) -> tuple[MockAPI, Capture, DevupAI | AsyncDevupAI]:
    capture = Capture(*entries)
    api = MockAPI(capture)
    return api, capture, api.client(kind, **opts)


def run(flow: Callable[[], Awaitable[T]]) -> T:
    """Runs ``flow`` in one event loop (also for the sync client, whose calls simply do not await)."""
    return anyio.run(flow)


async def call(kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Calls a client method; the async client's coroutine is awaited."""
    result = fn(*args, **kwargs)
    if kind == "async":
        assert inspect.isawaitable(result), "an async client method returns an awaitable"
        return await result
    assert not inspect.isawaitable(result), "a sync client method returns its result"
    return result


async def read(kind: str, binary: Any) -> bytes:
    data: bytes = (await binary.read()) if kind == "async" else binary.read()
    return data


async def chunks_of(kind: str, binary: Any, chunk_size: int | None = None) -> list[bytes]:
    if kind == "async":
        return [chunk async for chunk in binary.aiter_bytes(chunk_size)]
    return list(binary.iter_bytes(chunk_size))


async def write_to_file(kind: str, binary: Any, path: str | Path) -> None:
    if kind == "async":
        await binary.write_to_file(path)
    else:
        binary.write_to_file(path)


async def close(kind: str, binary: Any) -> None:
    if kind == "async":
        await binary.close()
    else:
        binary.close()


def download(client: DevupAI | AsyncDevupAI, *, with_response: bool = False) -> Callable[..., Any]:
    images = client.images.with_response if with_response else client.images
    method: Callable[..., Any] = images.download
    return method


def avatar(client: DevupAI | AsyncDevupAI, *, with_response: bool = False) -> Callable[..., Any]:
    me = client.me.with_response if with_response else client.me
    method: Callable[..., Any] = me.avatar
    return method


def inference_run(client: DevupAI | AsyncDevupAI, *, with_response: bool = False) -> Callable[..., Any]:
    inference = client.inference.with_response if with_response else client.inference
    method: Callable[..., Any] = inference.run
    return method


def fetch_and_read(kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, bytes]:
    """Calls ``fn`` and reads the binary result, in one event loop."""

    async def flow() -> tuple[Any, bytes]:
        binary = await call(kind, fn, *args, **kwargs)
        return binary, await read(kind, binary)

    return run(flow)


def fetch_and_read_wrapped(kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, bytes]:
    """Calls a ``with_response`` twin and reads its ``data``, in one event loop."""

    async def flow() -> tuple[Any, bytes]:
        wrapped = await call(kind, fn, *args, **kwargs)
        return wrapped, await read(kind, wrapped.data)

    return run(flow)


def visible_text(error: BaseException) -> str:
    """Everything an error shows: its text, its repr, its attributes, and the same for its chained errors."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.extend([str(current), repr(current), repr(current.args), repr(vars(current))])
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


# --- the classes ----------------------------------------------------------------------------------------------------


def test_both_classes_are_exported() -> None:
    module = _devupai()
    assert "BinaryResponse" in module.__all__
    assert "AsyncBinaryResponse" in module.__all__
    assert inspect.isclass(module.BinaryResponse)
    assert inspect.isclass(module.AsyncBinaryResponse)


def test_binary_response_has_the_sync_api() -> None:
    cls = binary_class("sync")
    for name in ("read", "iter_bytes", "write_to_file", "close", "__enter__", "__exit__"):
        assert callable(getattr(cls, name, None)), name
        assert not inspect.iscoroutinefunction(getattr(cls, name)), name


def test_async_binary_response_has_the_async_api() -> None:
    cls = binary_class("async")
    for name in ("read", "write_to_file", "close", "__aenter__", "__aexit__"):
        assert inspect.iscoroutinefunction(getattr(cls, name, None)), name
    assert callable(getattr(cls, "aiter_bytes", None))


def test_result_attributes(client_kind: str) -> None:
    headers = {"content-type": "image/webp", "etag": ETAG}
    _, _, client = setup(client_kind, binary_reply(PNG, headers=headers))

    async def flow() -> tuple[Any, bool, bytes]:
        binary = await call(client_kind, download(client), SIGNED)
        unread = binary.response.is_stream_consumed
        return binary, unread, await read(client_kind, binary)

    binary, consumed_before_read, data = run(flow)
    assert type(binary) is binary_class(client_kind)
    assert isinstance(binary.response, httpx.Response)
    assert consumed_before_read is False
    assert binary.status_code == 200
    assert isinstance(binary.headers, httpx.Headers)
    assert binary.headers["content-type"] == "image/webp"
    assert binary.headers["etag"] == ETAG
    assert binary.request_id == REQ_ID
    assert data == PNG


def test_request_id_is_none_without_the_header(client_kind: str) -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, stream=ChunkStream([PNG]))

    _, _, client = setup(client_kind, reply)
    binary, _ = fetch_and_read(client_kind, download(client), SIGNED)
    assert binary.request_id is None


def test_read_joins_every_chunk(client_kind: str) -> None:
    _, _, client = setup(client_kind, binary_reply(chunks=[b"abc", b"defg", b"h"]))
    _, data = fetch_and_read(client_kind, download(client), SIGNED)
    assert data == b"abcdefgh"


@pytest.mark.parametrize(
    ("chunk_size", "expected"),
    [
        pytest.param(None, None, id="default"),
        pytest.param(3, [b"abc", b"def", b"gh"], id="3"),
        pytest.param(1, [bytes([c]) for c in b"abcdefgh"], id="1"),
        pytest.param(100, [b"abcdefgh"], id="larger-than-body"),
    ],
)
def test_iter_bytes_with_chunk_size(client_kind: str, chunk_size: int | None, expected: list[bytes] | None) -> None:
    _, _, client = setup(client_kind, binary_reply(chunks=[b"abc", b"defg", b"h"]))

    async def flow() -> list[bytes]:
        binary = await call(client_kind, download(client), SIGNED)
        try:
            return await chunks_of(client_kind, binary, chunk_size)
        finally:
            await close(client_kind, binary)

    chunks = run(flow)
    assert b"".join(chunks) == b"abcdefgh"
    if expected is not None:
        assert chunks == expected


def test_write_to_file_with_a_path(client_kind: str, tmp_path: Path) -> None:
    _, _, client = setup(client_kind, binary_reply(chunks=[PNG[:3], PNG[3:]]))
    target = tmp_path / "image.png"

    async def flow() -> None:
        binary = await call(client_kind, download(client), SIGNED)
        await write_to_file(client_kind, binary, target)

    run(flow)
    assert target.read_bytes() == PNG


def test_write_to_file_with_a_str_path(client_kind: str, tmp_path: Path) -> None:
    _, _, client = setup(client_kind, binary_reply(PNG))
    target = tmp_path / "image-str.png"

    async def flow() -> None:
        binary = await call(client_kind, download(client), SIGNED)
        await write_to_file(client_kind, binary, str(target))

    run(flow)
    assert target.read_bytes() == PNG


def test_close_closes_the_response(client_kind: str) -> None:
    _, _, client = setup(client_kind, binary_reply(PNG))

    async def flow() -> tuple[bool, bool]:
        binary = await call(client_kind, download(client), SIGNED)
        before = binary.response.is_closed
        await close(client_kind, binary)
        return before, binary.response.is_closed

    before, after = run(flow)
    assert before is False
    assert after is True


def test_sync_with_block_returns_itself_and_closes() -> None:
    _, _, client = setup("sync", binary_reply(PNG))
    assert isinstance(client, DevupAI)
    result = client.images.download(SIGNED)
    with result as binary:
        assert binary is result
        assert binary.read() == PNG
    assert result.response.is_closed is True


def test_sync_with_block_closes_an_unread_body() -> None:
    _, _, client = setup("sync", binary_reply(PNG))
    assert isinstance(client, DevupAI)
    with client.images.download(SIGNED) as binary:
        pass
    assert binary.response.is_closed is True


def test_async_with_block_returns_itself_and_closes() -> None:
    _, _, client = setup("async", binary_reply(PNG))
    assert isinstance(client, AsyncDevupAI)

    async def flow() -> tuple[Any, Any, bytes]:
        result = await client.images.download(SIGNED)
        async with result as binary:
            data = await binary.read()
        return result, binary, data

    result, binary, data = run(flow)
    assert binary is result
    assert data == PNG
    assert result.response.is_closed is True


def test_async_with_block_closes_an_unread_body() -> None:
    _, _, client = setup("async", binary_reply(PNG))
    assert isinstance(client, AsyncDevupAI)

    async def flow() -> Any:
        async with await client.images.download(SIGNED) as binary:
            pass
        return binary

    binary = run(flow)
    assert binary.response.is_closed is True


# --- images.download ------------------------------------------------------------------------------------------------


def test_download_sends_get_to_the_url_without_the_key(client_kind: str) -> None:
    api, capture, client = setup(client_kind, binary_reply(PNG))
    binary, data = fetch_and_read(client_kind, download(client), SIGNED)
    request = capture.requests[0]
    assert len(api.calls) == 1
    assert request.method == "GET"
    assert str(request.url) == SIGNED
    assert request.url.raw_path == SIGNED.removeprefix(ORIGIN).encode("ascii")
    assert request.url.params["d"] == "fixture-token"
    assert request.url.params["s"] == "fixture-signature"
    assert "authorization" not in api.calls[0].headers
    assert request.content == b""
    assert type(binary) is binary_class(client_kind)
    assert binary.headers["content-type"] == "image/png"
    assert data == PNG


def test_download_never_sends_a_key_even_from_extra_headers(client_kind: str) -> None:
    api, _, client = setup(client_kind, binary_reply(PNG))
    fetch_and_read(client_kind, download(client), SIGNED, extra_headers={"Authorization": f"Bearer {TEST_KEY}"})
    assert "authorization" not in api.calls[0].headers


@pytest.mark.parametrize("url", RAW_URLS)
def test_download_keeps_the_url_byte_for_byte(client_kind: str, url: str) -> None:
    api, capture, client = setup(client_kind, binary_reply(b"image-data"))
    binary, data = fetch_and_read(client_kind, download(client), url)
    request = capture.requests[0]
    assert str(request.url) == url
    assert request.url.raw_path == url.removeprefix(ORIGIN).encode("ascii")
    assert request.method == "GET"
    assert "authorization" not in api.calls[0].headers
    assert type(binary) is binary_class(client_kind)
    assert data == b"image-data"


def test_download_passes_a_range_header(client_kind: str) -> None:
    api, _, client = setup(client_kind, binary_reply(b"ab", status=206))
    binary, data = fetch_and_read(client_kind, download(client), SIGNED, extra_headers={"Range": "bytes=0-1"})
    assert api.calls[0].headers["range"] == "bytes=0-1"
    assert binary.status_code == 206
    assert data == b"ab"


REJECTED_URLS = [
    pytest.param("https://files.example.test/v1/images/proxy?d=a&s=b", id="other-host"),
    pytest.param("https://api.example.test.example.com/v1/images/proxy?d=a&s=b", id="host-suffix"),
    pytest.param("https://example.test/v1/images/proxy?d=a&s=b", id="parent-host"),
    pytest.param("http://api.example.test/v1/images/proxy?d=a&s=b", id="other-scheme-http"),
    pytest.param("ftp://api.example.test/v1/images/proxy?d=a&s=b", id="other-scheme-ftp"),
    pytest.param("https://api.example.test:8443/v1/images/proxy?d=a&s=b", id="other-port"),
    pytest.param("https://api.example.test:80/v1/images/proxy?d=a&s=b", id="port-80-on-https"),
    pytest.param("https://api.example.test:99999/v1/images/proxy?d=a&s=b", id="invalid-port"),
    pytest.param("/v1/images/proxy?d=a&s=b", id="relative"),
    pytest.param("//api.example.test/v1/images/proxy?d=a&s=b", id="scheme-relative"),
    pytest.param("api.example.test/v1/images/proxy?d=a&s=b", id="no-scheme"),
    pytest.param("not a url", id="not-a-url"),
    pytest.param("", id="empty"),
]


def assert_rejected(api: MockAPI, error: BaseException) -> None:
    assert type(error) is DevupAPIError
    assert error.status == 0
    assert error.code == "invalid_url"
    assert error.message == REJECTED_MESSAGE
    assert str(error) == REJECTED_MESSAGE
    assert api.calls == []


@pytest.mark.parametrize("url", REJECTED_URLS)
def test_download_rejects_another_origin_before_any_request(client_kind: str, url: str) -> None:
    api, _, client = setup(client_kind, binary_reply(PNG))
    with pytest.raises(DevupAPIError) as caught:
        run(lambda: call(client_kind, download(client), url))
    assert_rejected(api, caught.value)
    if url:
        assert url not in visible_text(caught.value)


@pytest.mark.parametrize(
    ("url", "secrets"),
    [
        pytest.param(
            "https://fixture-user:fixture-secret@api.example.test/v1/images/proxy?d=a&s=b",
            ["fixture-user", "fixture-secret"],
            id="user-and-password",
        ),
        pytest.param("https://fixture-user@api.example.test/v1/images/proxy?d=a&s=b", ["fixture-user"], id="user-only"),
        pytest.param(
            "https://:fixture-secret@api.example.test/v1/images/proxy?d=a&s=b", ["fixture-secret"], id="password-only"
        ),
    ],
)
def test_download_rejects_userinfo_without_echoing_it(client_kind: str, url: str, secrets: list[str]) -> None:
    api, _, client = setup(client_kind, binary_reply(PNG))
    with pytest.raises(DevupAPIError) as caught:
        run(lambda: call(client_kind, download(client), url))
    assert_rejected(api, caught.value)
    shown = visible_text(caught.value)
    assert url not in shown
    for secret in secrets:
        assert secret not in shown


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(None, id="none"),
        pytest.param(123, id="int"),
        pytest.param(SIGNED.encode("ascii"), id="bytes"),
        pytest.param(httpx.URL(SIGNED), id="httpx-url"),
    ],
)
def test_download_rejects_a_value_that_is_not_a_str(client_kind: str, url: object) -> None:
    api, _, client = setup(client_kind, binary_reply(PNG))
    with pytest.raises(DevupAPIError) as caught:
        run(lambda: call(client_kind, download(client), url))
    assert_rejected(api, caught.value)


@pytest.mark.parametrize(
    ("base_url", "url"),
    [
        pytest.param(BASE, "https://api.example.test:443/v1/images/proxy?d=a&s=b", id="explicit-443-in-url"),
        pytest.param("https://api.example.test:443/v1", SIGNED, id="explicit-443-in-base"),
        pytest.param("http://api.example.test/v1", "http://api.example.test:80/v1/images/proxy?d=a&s=b", id="http-80"),
        pytest.param(BASE, "https://API.EXAMPLE.TEST/v1/images/proxy?d=a&s=b", id="host-case"),
        pytest.param(BASE, "https://api.example.test/elsewhere/file.png", id="any-path-on-the-origin"),
    ],
)
def test_download_normalises_the_origin(client_kind: str, base_url: str, url: str) -> None:
    api, capture, client = setup(client_kind, binary_reply(PNG), base_url=base_url)
    _, data = fetch_and_read(client_kind, download(client), url)
    assert len(api.calls) == 1
    target = httpx.URL(url)
    sent = capture.requests[0].url
    assert sent.host == "api.example.test"
    assert sent.scheme == target.scheme
    assert sent.raw_path == target.raw_path
    assert "authorization" not in api.calls[0].headers
    assert data == PNG


def test_download_compares_with_a_custom_base_url(client_kind: str) -> None:
    base = "http://localhost:3000/api/v1"
    api, capture, client = setup(client_kind, binary_reply(PNG), base_url=base)
    same = "http://localhost:3000/api/v1/images/proxy?d=a&s=b"
    fetch_and_read(client_kind, download(client), same)
    assert str(capture.requests[0].url) == same
    for other in (
        "http://localhost:4000/api/v1/images/proxy?d=a&s=b",
        "http://localhost/api/v1/images/proxy?d=a&s=b",
        "https://localhost:3000/api/v1/images/proxy?d=a&s=b",
        "http://fixture-user:fixture-secret@localhost:3000/api/v1/images/proxy?d=a&s=b",
    ):
        with pytest.raises(DevupAPIError) as caught:
            run(lambda other=other: call(client_kind, download(client), other))
        assert caught.value.code == "invalid_url"
        assert caught.value.message == REJECTED_MESSAGE
    assert len(api.calls) == 1


@pytest.mark.parametrize("status", [503, 502])
def test_download_uses_the_default_retry(client_kind: str, status: int) -> None:
    api, _, client = setup(client_kind, error_response(status), binary_reply(PNG))
    binary, data = fetch_and_read(client_kind, download(client), SIGNED)
    assert len(api.calls) == 2
    assert all(sent.url == SIGNED for sent in api.calls)
    assert all("authorization" not in sent.headers for sent in api.calls)
    assert binary.status_code == 200
    assert data == PNG


def test_download_max_retries_zero_sends_once(client_kind: str) -> None:
    api, _, client = setup(client_kind, error_response(503), binary_reply(PNG))
    with pytest.raises(InternalServerError) as caught:
        fetch_and_read(client_kind, download(client), SIGNED, max_retries=0)
    assert caught.value.status == 503
    assert len(api.calls) == 1


def test_download_error_status_raises(client_kind: str) -> None:
    api, _, client = setup(client_kind, error_response(404, "Not found.", code="file_not_found"))
    with pytest.raises(NotFoundError) as caught:
        fetch_and_read(client_kind, download(client), SIGNED)
    assert caught.value.status == 404
    assert caught.value.code == "file_not_found"
    assert len(api.calls) == 1


def test_download_with_response_twin(client_kind: str) -> None:
    api, capture, client = setup(client_kind, binary_reply(PNG))
    wrapped, data = fetch_and_read_wrapped(client_kind, download(client, with_response=True), SIGNED)
    assert isinstance(wrapped, APIResponse)
    assert type(wrapped.data) is binary_class(client_kind)
    assert wrapped.data.response is wrapped.response
    assert wrapped.request_id == REQ_ID
    assert wrapped.response.status_code == 200
    assert data == PNG
    assert str(capture.requests[0].url) == SIGNED
    assert "authorization" not in api.calls[0].headers


def test_download_with_response_twin_rejects_another_origin(client_kind: str) -> None:
    api, _, client = setup(client_kind, binary_reply(PNG))
    with pytest.raises(DevupAPIError) as caught:
        run(lambda: call(client_kind, download(client, with_response=True), "https://files.example.test/x.png"))
    assert_rejected(api, caught.value)


# --- me.avatar ------------------------------------------------------------------------------------------------------


def avatar_reply() -> Callable[[httpx.Request], httpx.Response]:
    return binary_reply(PNG, headers={"etag": ETAG, "cache-control": "private, max-age=86400"})


def test_avatar_sends_get_with_the_key_and_returns_the_bytes(client_kind: str) -> None:
    api, _, client = setup(client_kind, avatar_reply())
    binary, data = fetch_and_read(client_kind, avatar(client))
    request = api.calls[0]
    assert request.method == "GET"
    assert request.url == f"{BASE}/me/avatar"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert request.headers["accept"] == "*/*"
    assert "if-none-match" not in request.headers
    assert request.content == b""
    assert type(binary) is binary_class(client_kind)
    assert binary.status_code == 200
    assert binary.headers["etag"] == ETAG
    assert binary.headers["content-type"] == "image/png"
    assert binary.request_id == REQ_ID
    assert data == PNG


def test_avatar_if_none_match_and_304_returns_a_binary_response(client_kind: str) -> None:
    api, _, client = setup(client_kind, not_modified())
    binary, data = fetch_and_read(client_kind, avatar(client), if_none_match=ETAG)
    assert api.calls[0].headers["if-none-match"] == ETAG
    assert binary is not None
    assert type(binary) is binary_class(client_kind)
    assert binary.status_code == 304
    assert binary.headers["etag"] == ETAG
    assert binary.request_id == REQ_ID
    assert data == b""


def test_avatar_if_none_match_that_no_longer_matches_returns_the_bytes(client_kind: str) -> None:
    api, _, client = setup(client_kind, avatar_reply())
    binary, data = fetch_and_read(client_kind, avatar(client), if_none_match='"fixture-old-etag"')
    assert api.calls[0].headers["if-none-match"] == '"fixture-old-etag"'
    assert binary.status_code == 200
    assert data == PNG


def test_avatar_with_response_on_304(client_kind: str) -> None:
    _, _, client = setup(client_kind, not_modified())
    wrapped, data = fetch_and_read_wrapped(client_kind, avatar(client, with_response=True), if_none_match=ETAG)
    assert isinstance(wrapped, APIResponse)
    assert type(wrapped.data) is binary_class(client_kind)
    assert wrapped.data.status_code == 304
    assert wrapped.response.status_code == 304
    assert wrapped.response.headers["etag"] == ETAG
    assert wrapped.request_id == REQ_ID
    assert data == b""


def test_avatar_with_response_on_200(client_kind: str) -> None:
    _, _, client = setup(client_kind, avatar_reply())
    wrapped, data = fetch_and_read_wrapped(client_kind, avatar(client, with_response=True))
    assert isinstance(wrapped, APIResponse)
    assert type(wrapped.data) is binary_class(client_kind)
    assert wrapped.data.response is wrapped.response
    assert wrapped.request_id == REQ_ID
    assert data == PNG


def test_avatar_if_none_match_combines_with_extra_headers(client_kind: str) -> None:
    api, _, client = setup(client_kind, avatar_reply())
    fetch_and_read(client_kind, avatar(client), if_none_match=ETAG, extra_headers={"X-Fixture": "1"})
    request = api.calls[0]
    assert request.url == f"{BASE}/me/avatar"
    assert request.headers["x-fixture"] == "1"
    assert request.headers["if-none-match"] == ETAG


def test_avatar_not_found_raises(client_kind: str) -> None:
    _, _, client = setup(
        client_kind, error_response(404, "This account has no profile picture.", code="avatar_not_found")
    )
    with pytest.raises(NotFoundError) as caught:
        fetch_and_read(client_kind, avatar(client))
    assert caught.value.code == "avatar_not_found"
    assert caught.value.status == 404
    assert caught.value.request_id == REQ_ID


# --- inference.run(response_type="binary") --------------------------------------------------------------------------


def audio_reply() -> Callable[[httpx.Request], httpx.Response]:
    return binary_reply(b"RIFF-fixture-audio", headers={"content-type": "audio/wav"})


def test_inference_binary_returns_a_binary_response(client_kind: str) -> None:
    api, _, client = setup(client_kind, audio_reply())
    binary, data = fetch_and_read(
        client_kind,
        inference_run(client),
        MODEL_A,
        {"input": "x"},
        response_type="binary",
        extra_headers={"X-Fixture": "1"},
    )
    request = api.calls[0]
    assert request.method == "POST"
    assert request.url == f"{BASE}/inference/fixture%2Fmodel-a"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert request.headers["accept"] == "*/*"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-fixture"] == "1"
    assert request.json() == {"input": "x"}
    assert type(binary) is binary_class(client_kind)
    assert binary.status_code == 200
    assert binary.headers["content-type"] == "audio/wav"
    assert binary.request_id == REQ_ID
    assert data == b"RIFF-fixture-audio"


@pytest.mark.parametrize("status", [503, 429])
def test_inference_binary_is_sent_again_after_a_billed_retry_status(client_kind: str, status: int) -> None:
    api, _, client = setup(client_kind, error_response(status), audio_reply())
    binary, data = fetch_and_read(client_kind, inference_run(client), MODEL_A, {"input": "x"}, response_type="binary")
    assert len(api.calls) == 2
    assert api.calls[0].content == api.calls[1].content
    assert binary.status_code == 200
    assert data == b"RIFF-fixture-audio"


@pytest.mark.parametrize("status", [502, 504, 408])
def test_inference_binary_is_not_sent_again_after_a_status_that_can_follow_work(client_kind: str, status: int) -> None:
    api, _, client = setup(client_kind, error_response(status), audio_reply())
    with pytest.raises(DevupAPIError) as caught:
        fetch_and_read(client_kind, inference_run(client), MODEL_A, {"input": "x"}, response_type="binary")
    assert caught.value.status == status
    assert len(api.calls) == 1


def test_inference_binary_with_response_twin(client_kind: str) -> None:
    _, _, client = setup(client_kind, audio_reply())
    wrapped, data = fetch_and_read_wrapped(
        client_kind, inference_run(client, with_response=True), MODEL_A, {"input": "x"}, response_type="binary"
    )
    assert isinstance(wrapped, APIResponse)
    assert type(wrapped.data) is binary_class(client_kind)
    assert wrapped.data.response is wrapped.response
    assert wrapped.request_id == REQ_ID
    assert data == b"RIFF-fixture-audio"


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({}, id="default"), pytest.param({"response_type": "json"}, id="json")],
)
def test_inference_json_still_returns_parsed_json(client_kind: str, kwargs: dict[str, Any]) -> None:
    api, _, client = setup(client_kind, json_response({"ok": True, "items": [1, 2]}))

    async def flow() -> Any:
        return await call(client_kind, inference_run(client), MODEL_A, {"input": "x"}, **kwargs)

    result = run(flow)
    assert result == {"ok": True, "items": [1, 2]}
    assert api.calls[0].headers["accept"] == "application/json"
    assert api.calls[0].json() == {"input": "x"}


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({}, id="default"), pytest.param({"response_type": "json"}, id="json")],
)
def test_inference_json_with_response_twin_returns_parsed_json(client_kind: str, kwargs: dict[str, Any]) -> None:
    _, _, client = setup(client_kind, json_response({"ok": True}))

    async def flow() -> Any:
        return await call(client_kind, inference_run(client, with_response=True), MODEL_A, {"input": "x"}, **kwargs)

    wrapped = run(flow)
    assert isinstance(wrapped, APIResponse)
    assert wrapped.data == {"ok": True}
    assert wrapped.request_id == REQ_ID
