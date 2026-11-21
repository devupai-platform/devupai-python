"""Header values, the key in errors and reprs, the deadline while a body is read, and the base URL check.

Some tests here run the real ``httpx.HTTPTransport`` (and its async twin) over an ``httpcore`` mock network
backend, so the HTTP/1.1 header checks of the real stack run while nothing leaves the machine.
"""

from __future__ import annotations

import traceback
from collections.abc import AsyncIterator, Iterator
from typing import Any

import anyio
import httpcore
import httpx
import pytest
from helpers.mock_transport import BASE, REQ_ID, MockAPI, json_response, run_request, sequence

from devupai import APIConnectionError, APITimeoutError, AsyncDevupAI, AuthenticationError, DevupAI
from devupai._client import DEFAULT_RETRY

SECRET = "fixture-hdr-secret-0001"
HEADER_PROTOCOL_MESSAGE = "The request could not be sent: a header value is not valid."
BASE_URL_MESSAGE = "base_url must be an absolute http or https URL."
PLAIN_BASE = "http://api.example.test/v1"

# Captured at import, before the autouse guard of conftest replaces them for each test. The transports below use
# them only with an httpcore mock network backend, so no request leaves the machine.
_REAL_SYNC_HANDLE = httpx.HTTPTransport.handle_request
_REAL_ASYNC_HANDLE = httpx.AsyncHTTPTransport.handle_async_request

_UNAUTHORIZED_BODY = b'{"error":{"message":"Unauthorized.","type":"devup_error","code":"fixture_unauthorized"}}'
_UNAUTHORIZED = [
    b"HTTP/1.1 401 Unauthorized\r\n",
    b"Content-Type: application/json\r\n",
    b"Content-Length: " + str(len(_UNAUTHORIZED_BODY)).encode() + b"\r\n",
    b"x-request-id: " + REQ_ID.encode() + b"\r\n",
    b"\r\n",
    _UNAUTHORIZED_BODY,
]


class MockNetworkTransport(httpx.HTTPTransport):
    """The real HTTP transport over an httpcore mock network backend; records each request it is given."""

    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__()
        self._pool = httpcore.ConnectionPool(network_backend=httpcore.MockBackend(list(buffer)))
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return _REAL_SYNC_HANDLE(self, request)


class AsyncMockNetworkTransport(httpx.AsyncHTTPTransport):
    """The async twin of :class:`MockNetworkTransport`."""

    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__()
        self._pool = httpcore.AsyncConnectionPool(network_backend=httpcore.AsyncMockBackend(list(buffer)))
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await _REAL_ASYNC_HANDLE(self, request)


def real_stack_client(
    client_kind: str, api_key: str, **opts: Any
) -> tuple[DevupAI | AsyncDevupAI, MockNetworkTransport | AsyncMockNetworkTransport]:
    if client_kind == "sync":
        transport = MockNetworkTransport(_UNAUTHORIZED)
        client: DevupAI | AsyncDevupAI = DevupAI(
            api_key=api_key, base_url=PLAIN_BASE, http_client=httpx.Client(transport=transport), **opts
        )
        return client, transport
    async_transport = AsyncMockNetworkTransport(_UNAUTHORIZED)
    async_client = AsyncDevupAI(
        api_key=api_key, base_url=PLAIN_BASE, http_client=httpx.AsyncClient(transport=async_transport), **opts
    )
    return async_client, async_transport


def everything_about(err: BaseException) -> str:
    return str(err) + repr(err) + "".join(traceback.format_exception(err))


def ok_handler(request: httpx.Request) -> httpx.Response:
    return json_response({"ok": True})


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("DEVUP_API_KEY", raising=False)
    monkeypatch.delenv("DEVUPAI_API_KEY", raising=False)
    return monkeypatch


# --- H1: header values are normalised like the Fetch Headers class, then checked -------------------------------


@pytest.mark.parametrize("suffix", [pytest.param(" ", id="space"), pytest.param("\t", id="tab")])
def test_key_with_trailing_whitespace_in_the_argument_is_sent_trimmed(client_kind: str, suffix: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind, api_key=SECRET + suffix), "GET", "/models")
    assert api.calls[0].headers["authorization"] == f"Bearer {SECRET}"


@pytest.mark.parametrize("suffix", [pytest.param(" ", id="space"), pytest.param("\t", id="tab")])
def test_key_with_trailing_whitespace_in_the_environment_is_sent_trimmed(
    clean_env: pytest.MonkeyPatch, client_kind: str, suffix: str
) -> None:
    clean_env.setenv("DEVUP_API_KEY", SECRET + suffix)
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind, api_key=None), "GET", "/models")
    assert api.calls[0].headers["authorization"] == f"Bearer {SECRET}"


def test_header_value_with_surrounding_whitespace_is_sent_trimmed(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, default_headers={"X-Fixture": " \tvalue a\tb \t"})
    run_request(client, "GET", "/models")
    assert api.calls[0].headers["x-fixture"] == "value a\tb"


@pytest.mark.parametrize("bad", [pytest.param("\x01", id="SOH"), pytest.param("\x7f", id="DEL")])
def test_key_with_a_control_character_raises_value_error_without_the_key(client_kind: str, bad: str) -> None:
    api = MockAPI(ok_handler)
    with pytest.raises(ValueError) as info:
        client = api.client(client_kind, api_key=f"{SECRET}{bad}x")
        run_request(client, "GET", "/models")
    assert SECRET not in everything_about(info.value)
    assert api.calls == []


@pytest.mark.parametrize("bad", ["\x01", "\x0b", "\x0c", "\x1f", "\x7f"])
def test_header_value_with_a_control_character_raises_before_any_request(client_kind: str, bad: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", "/models", options={"extra_headers": {"X-Fixture": f"{SECRET}{bad}x"}})
    assert str(info.value) == 'Header "X-Fixture" has a value that is not allowed in an HTTP header.'
    assert SECRET not in everything_about(info.value)
    assert api.calls == []


@pytest.mark.parametrize("suffix", [pytest.param(" ", id="space"), pytest.param("\t", id="tab")])
def test_real_stack_key_with_trailing_whitespace_reaches_the_api_and_is_never_shown(
    client_kind: str, suffix: str, fake_clock: Any
) -> None:
    client, transport = real_stack_client(client_kind, SECRET + suffix)
    with pytest.raises(AuthenticationError) as info:
        run_request(client, "GET", "/models")
    err = info.value
    assert err.status == 401
    assert err.request_id == REQ_ID
    assert SECRET not in everything_about(err)
    assert len(transport.requests) == 1
    assert transport.requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert fake_clock.sleeps == []


def test_real_stack_empty_key_sends_a_bare_bearer_and_gets_the_401(client_kind: str, fake_clock: Any) -> None:
    client, transport = real_stack_client(client_kind, "")
    with pytest.raises(AuthenticationError) as info:
        run_request(client, "GET", "/models")
    assert info.value.status == 401
    assert len(transport.requests) == 1
    assert transport.requests[0].headers["authorization"] == "Bearer"
    assert fake_clock.sleeps == []


def test_local_protocol_error_is_a_fixed_message_without_the_value_and_is_not_retried(
    client_kind: str, fake_clock: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.LocalProtocolError(f"Illegal header value b'Bearer {SECRET} '")

    api = MockAPI(handler)
    with pytest.raises(APIConnectionError) as info:
        run_request(api.client(client_kind, max_retries=2), "GET", "/models", retry=DEFAULT_RETRY)
    err = info.value
    assert type(err) is APIConnectionError
    assert err.status == 0
    assert err.message == HEADER_PROTOCOL_MESSAGE
    assert err.__cause__ is None
    assert err.__suppress_context__ is True
    assert SECRET not in everything_about(err)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


# --- M2: the prepared request never shows the key in its repr ---------------------------------------------------


def test_prepared_request_repr_does_not_contain_the_key(client_kind: str) -> None:
    client = MockAPI(ok_handler).client(client_kind, api_key=SECRET)
    prepared = client._prepare(
        "POST",
        "/x",
        url=None,
        query=None,
        body={"model": "fixture/model-a"},
        auth=True,
        if_none_match=None,
        retry=DEFAULT_RETRY,
        kind="json",
        options={"extra_headers": {"X-Fixture": SECRET}},
    )
    assert SECRET not in repr(prepared)
    assert SECRET not in str(prepared)


# --- M3: the deadline covers the body read ---------------------------------------------------------------------


class ClockStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body sent in chunks; the fake clock moves forward ``step`` seconds before each chunk."""

    def __init__(self, clock: Any, chunks: list[bytes], step: float) -> None:
        self.clock = clock
        self.chunks = chunks
        self.step = step
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.clock.advance(self.step)
            yield chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.clock.advance(self.step)
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("kind", ["json", "text"])
@pytest.mark.parametrize("status", [200, 503])
def test_a_body_that_trickles_past_the_deadline_raises_api_timeout_error(
    client_kind: str, kind: str, status: int, fake_clock: Any
) -> None:
    stream = ClockStream(fake_clock, [b'{"a":', b"1,", b'"b":2}'], step=6)
    api = MockAPI(lambda request: httpx.Response(status, headers={"x-request-id": REQ_ID}, stream=stream))
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=10), "GET", "/models", kind=kind)
    assert info.value.message == "The request timed out after 10 seconds."
    assert len(api.calls) == 1
    assert stream.closed
    assert fake_clock.sleeps == []


class StalledStream(httpx.AsyncByteStream):
    """An async body that sends one chunk and then never sends anything else."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"a":'
        await anyio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_async_a_stalled_body_ends_at_the_deadline() -> None:
    stream = StalledStream()
    api = MockAPI(lambda request: httpx.Response(200, headers={"x-request-id": REQ_ID}, stream=stream))
    client = api.async_client(timeout=0.2)
    with anyio.fail_after(5):  # keeps a client without the deadline from hanging the run
        with pytest.raises(APITimeoutError) as info:
            await client._request("GET", "/models")
    assert info.value.message == "The request timed out after 0.2 seconds."
    assert len(api.calls) == 1
    assert stream.closed


# --- Low: the base URL and an absolute url must be http or https ---------------------------------------------------


@pytest.mark.parametrize("base_url", ["api.example.test/v1", "ftp://files.example.test/v1", "/v1", "https:///v1"])
def test_base_url_that_is_not_absolute_http_raises_value_error_before_any_request(
    client_kind: str, base_url: str, fake_clock: Any
) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, base_url=base_url)
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", "/models")
    assert str(info.value) == BASE_URL_MESSAGE
    assert api.calls == []
    assert fake_clock.sleeps == []


@pytest.mark.parametrize("url", ["files.example.test/a.png", "ftp://files.example.test/a.png", "/a.png"])
def test_url_argument_that_is_not_absolute_http_raises_value_error_before_any_request(
    client_kind: str, url: str, fake_clock: Any
) -> None:
    api = MockAPI(ok_handler)
    with pytest.raises(ValueError):
        run_request(api.client(client_kind), "GET", url=url, auth=False, kind="text")
    assert api.calls == []
    assert fake_clock.sleeps == []


def test_absolute_http_and_https_urls_still_work(client_kind: str) -> None:
    api = MockAPI(sequence(json_response({"ok": True})))
    client = api.client(client_kind, base_url="http://localhost:8080/v1")
    run_request(client, "GET", "/models")
    run_request(client, "GET", url=f"{BASE}/models")
    assert [call.url for call in api.calls] == ["http://localhost:8080/v1/models", f"{BASE}/models"]
