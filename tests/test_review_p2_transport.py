"""Review P2 fixes in the transport (``p2/review-p2.md`` findings R2-2, R2-8 and R2-11).

- R2-2: a body that cannot be decoded (``httpx.DecodingError``, for example a corrupt ``content-encoding: gzip``)
  comes from the API's bytes, not from the caller. On a 2xx it is an ``APIConnectionError`` (P1 §5), never retried.
  On an error status the message falls back to ``HTTP <status>``, so the status error class and its retry rule apply.
- R2-11: any other exception raised while an error body is read is raised unchanged, after one call.
- R2-8: a ``base_url`` that cannot be parsed is refused at construction, so ``repr(client)`` and ``client.base_url``
  never show it (it can carry a password).
"""

from __future__ import annotations

import traceback
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import pytest
from helpers.mock_transport import MODEL_A, REQ_ID, TEST_KEY, MockAPI, json_response, run_request

from devupai import (
    APIConnectionError,
    APITimeoutError,
    AsyncDevupAI,
    DevupAI,
    InternalServerError,
    RateLimitError,
)
from devupai._client import BILLED_POST_RETRY, DEFAULT_RETRY, RetryPolicy

BASE_URL_MESSAGE = "base_url must be an absolute http or https URL."
BASE_URL_CREDENTIALS_MESSAGE = "base_url must not contain a user name or password."

POLICIES = [pytest.param(DEFAULT_RETRY, id="default"), pytest.param(BILLED_POST_RETRY, id="billed-post")]


def call(client: DevupAI | AsyncDevupAI, retry: RetryPolicy, **kwargs: Any) -> Any:
    """A GET for DEFAULT_RETRY, a billed POST for BILLED_POST_RETRY."""
    if retry is BILLED_POST_RETRY:
        return run_request(client, "POST", "/chat/completions", body={"model": MODEL_A}, retry=retry, **kwargs)
    return run_request(client, "GET", "/models", retry=retry, **kwargs)


class ChunkStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body sent as the given chunks, then ``error`` (when set) is raised."""

    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        if self.error is not None:
            raise self.error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def corrupt_gzip(status: int) -> Callable[[httpx.Request], httpx.Response]:
    """A fresh response per call whose ``content-encoding: gzip`` body is not gzip."""

    def make(request: httpx.Request) -> httpx.Response:
        headers = {"x-request-id": REQ_ID, "content-encoding": "gzip", "content-type": "application/json"}
        return httpx.Response(status, headers=headers, stream=ChunkStream([b'{"error": {"message": "x"}}']))

    return make


def decoding_error_stream(status: int) -> Callable[[httpx.Request], httpx.Response]:
    """A fresh response per call whose body stream raises ``httpx.DecodingError`` after one chunk."""

    def make(request: httpx.Request) -> httpx.Response:
        stream = ChunkStream([b'{"a":'], httpx.DecodingError("fixture decoding failure"))
        return httpx.Response(status, headers={"x-request-id": REQ_ID}, stream=stream)

    return make


BAD_BODIES = [
    pytest.param(corrupt_gzip, id="corrupt-gzip"),
    pytest.param(decoding_error_stream, id="decoding-error-stream"),
]


class Answers:
    """Answers call *i* with ``makers[i]``; the last one repeats."""

    def __init__(self, *makers: Callable[[httpx.Request], httpx.Response]) -> None:
        self.makers = makers
        self.count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        maker = self.makers[min(self.count, len(self.makers) - 1)]
        self.count += 1
        return maker(request)


def ok(request: httpx.Request) -> httpx.Response:
    return json_response({"ok": True})


# --- R2-2: a 2xx body that cannot be decoded ------------------------------------------------------------------------


def test_the_bad_bodies_raise_a_decoding_error_that_is_not_a_transport_error() -> None:
    assert issubclass(httpx.DecodingError, httpx.RequestError)
    assert not issubclass(httpx.DecodingError, httpx.TransportError)
    for case in BAD_BODIES:
        factory = case.values[0]
        assert callable(factory)
        response = factory(200)(httpx.Request("GET", "https://api.example.test/v1/models"))
        with pytest.raises(httpx.DecodingError):
            response.read()


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("bad_body", BAD_BODIES)
@pytest.mark.parametrize("kind", ["json", "text"])
def test_2xx_body_that_cannot_be_decoded_is_an_api_connection_error_and_is_not_retried(
    client_kind: str,
    retry: RetryPolicy,
    bad_body: Callable[[int], Callable[[httpx.Request], httpx.Response]],
    kind: str,
    fake_clock: Any,
) -> None:
    api = MockAPI(bad_body(200))
    client = api.client(client_kind, max_retries=5)
    with pytest.raises(APIConnectionError) as info:
        call(client, retry, kind=kind)
    assert type(info.value) is APIConnectionError
    assert not isinstance(info.value, APITimeoutError)
    assert info.value.message.startswith("Network error: ")
    assert info.value.request_id == REQ_ID
    assert isinstance(info.value.__cause__, httpx.DecodingError)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


# --- R2-2: an error body that cannot be decoded keeps the status error and its retry rule --------------------------


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("bad_body", BAD_BODIES)
@pytest.mark.parametrize(
    ("status", "error_class"),
    [pytest.param(503, InternalServerError, id="503"), pytest.param(429, RateLimitError, id="429")],
)
def test_error_body_that_cannot_be_decoded_keeps_the_status_error(
    client_kind: str,
    retry: RetryPolicy,
    bad_body: Callable[[int], Callable[[httpx.Request], httpx.Response]],
    status: int,
    error_class: type[Exception],
    fake_clock: Any,
) -> None:
    api = MockAPI(bad_body(status))
    client = api.client(client_kind, max_retries=0)
    with pytest.raises(error_class) as info:
        call(client, retry)
    err = info.value
    assert type(err) is error_class
    assert isinstance(err, InternalServerError | RateLimitError)
    assert err.status == status
    assert err.message == f"HTTP {status}"
    assert str(err) == f"HTTP {status}"
    assert err.request_id == REQ_ID
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("bad_body", BAD_BODIES)
@pytest.mark.parametrize("status", [503, 429])
def test_error_body_that_cannot_be_decoded_is_still_retried_by_its_policy(
    client_kind: str,
    retry: RetryPolicy,
    bad_body: Callable[[int], Callable[[httpx.Request], httpx.Response]],
    status: int,
    fake_clock: Any,
) -> None:
    # 503 and 429 are retried by both DEFAULT and BILLED_POST.
    api = MockAPI(Answers(bad_body(status), ok))
    client = api.client(client_kind, max_retries=2)
    assert call(client, retry).data == {"ok": True}
    assert len(api.calls) == 2
    assert len(fake_clock.sleeps) == 1


@pytest.mark.parametrize("bad_body", BAD_BODIES)
def test_default_503_with_a_body_that_cannot_be_decoded_is_retried_until_max_retries(
    client_kind: str, bad_body: Callable[[int], Callable[[httpx.Request], httpx.Response]], fake_clock: Any
) -> None:
    api = MockAPI(bad_body(503))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError) as info:
        run_request(client, "GET", "/models", retry=DEFAULT_RETRY)
    assert info.value.message == "HTTP 503"
    assert len(api.calls) == 3
    assert len(fake_clock.sleeps) == 2


@pytest.mark.parametrize("bad_body", BAD_BODIES)
def test_billed_502_with_a_body_that_cannot_be_decoded_is_not_retried(
    client_kind: str, bad_body: Callable[[int], Callable[[httpx.Request], httpx.Response]], fake_clock: Any
) -> None:
    api = MockAPI(Answers(bad_body(502), ok))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError) as info:
        call(client, BILLED_POST_RETRY)
    assert info.value.status == 502
    assert info.value.message == "HTTP 502"
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


# --- R2-11: any other exception while an error body is read is raised unchanged ------------------------------------


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("status", [503, 429, 500])
def test_non_transport_error_while_reading_an_error_body_is_raised_unchanged(
    client_kind: str, retry: RetryPolicy, status: int, fake_clock: Any
) -> None:
    error = RuntimeError("fixture error body failure")
    stream = ChunkStream([b'{"error":'], error)
    api = MockAPI(lambda request: httpx.Response(status, headers={"x-request-id": REQ_ID}, stream=stream))
    client = api.client(client_kind, max_retries=5)
    with pytest.raises(RuntimeError) as info:
        call(client, retry)
    assert info.value is error
    assert len(api.calls) == 1
    assert stream.closed
    assert fake_clock.sleeps == []


# --- R2-8: a base_url that cannot be parsed is refused at construction ----------------------------------------------


def everything_about(err: BaseException) -> str:
    return str(err) + repr(err) + "".join(traceback.format_exception(err))


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("https://user:s3cret-pw@[::1/v1", id="open-ipv6-bracket"),
        pytest.param("https://user:s3cret-pw@[::zz]/v1", id="bad-ipv6-host"),
        pytest.param("https://fixture-login:s3cret-pw@api.example.test:abc/v1", id="port-not-a-number"),
        pytest.param("https://user:s3cret-pw@a\x00b/v1", id="nul-in-host"),
    ],
)
def test_unparseable_base_url_with_credentials_raises_at_construction_without_the_password(
    client_kind: str, base_url: str
) -> None:
    with pytest.raises(httpx.InvalidURL):
        httpx.URL(base_url)
    api = MockAPI(ok)
    with pytest.raises(ValueError) as info:
        api.client(client_kind, base_url=base_url)
    assert str(info.value) == BASE_URL_MESSAGE
    shown = everything_about(info.value)
    for secret in ("pw", "s3cret", "fixture-login", "user:"):
        assert secret not in shown
    assert api.calls == []


UNPARSEABLE_WITH_PASSWORD = "https://user:s3cret-pw@[::1/v1"
# httpx parses "[bad" as a host name, so this one is refused by the credentials check.
PARSEABLE_BRACKET_WITH_PASSWORD = "https://user:pw@[bad"


@pytest.mark.parametrize("client_cls", [DevupAI, AsyncDevupAI], ids=["sync", "async"])
def test_unparseable_base_url_never_reaches_repr_or_base_url(client_cls: type[Any]) -> None:
    # The URL is passed through a variable: the traceback shows source lines, never argument values.
    given = UNPARSEABLE_WITH_PASSWORD
    with pytest.raises(ValueError) as info:
        client_cls(api_key=TEST_KEY, base_url=given)
    assert str(info.value) == BASE_URL_MESSAGE
    assert "s3cret-pw" not in everything_about(info.value)


def test_a_parseable_base_url_with_a_password_and_a_bracket_host_is_refused_as_a_credential(
    client_kind: str,
) -> None:
    given = PARSEABLE_BRACKET_WITH_PASSWORD
    with pytest.raises(ValueError) as info:
        MockAPI(ok).client(client_kind, base_url=given)
    assert str(info.value) == BASE_URL_CREDENTIALS_MESSAGE
    assert ":pw@" not in everything_about(info.value)
    assert "pw" not in str(info.value) + repr(info.value)
