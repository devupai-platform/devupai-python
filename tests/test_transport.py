"""The request loop: response kinds, body errors, conditional GET, redirects, the deadline and closed clients."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import anyio
import httpx
import pytest
from helpers.mock_transport import REQ_ID, MockAPI, error_response, json_response, run_request, sequence

from devupai import (
    APIConnectionError,
    APIResponse,
    APITimeoutError,
    AsyncDevupAI,
    DebugEvent,
    DevupAI,
    DevupAPIError,
    InternalServerError,
    RateLimitError,
)
from devupai._client import BILLED_POST_RETRY, DEFAULT_RETRY, RetryPolicy

CLOSED_MESSAGE = "The client is closed."
NOT_JSON_MESSAGE = "The response body is not valid JSON."
POLICIES = [pytest.param(DEFAULT_RETRY, id="default"), pytest.param(BILLED_POST_RETRY, id="billed-post")]


def timed_out(seconds: str) -> str:
    return f"The request timed out after {seconds} seconds."


class ChunkStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A response body sent in chunks, optionally failing with `httpx.ReadError` after the last chunk."""

    def __init__(self, chunks: list[bytes], *, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        if self.fail:
            raise httpx.ReadError("connection reset")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadError("connection reset")

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def ok_handler(request: httpx.Request) -> httpx.Response:
    return json_response({"ok": True})


def close_client(client: DevupAI | AsyncDevupAI) -> None:
    if isinstance(client, AsyncDevupAI):
        anyio.run(client.close)
    else:
        client.close()


def read_response(client_kind: str, response: httpx.Response) -> bytes:
    if client_kind == "async":
        return anyio.run(response.aread)
    return response.read()


def close_response(client_kind: str, response: httpx.Response) -> None:
    if client_kind == "async":
        anyio.run(response.aclose)
    else:
        response.close()


def failing_body(status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"x-request-id": REQ_ID, "content-type": "application/json"},
        stream=ChunkStream([b'{"a":'], fail=True),
    )


# --- APIResponse -------------------------------------------------------------------------------------------------


def test_api_response_carries_data_response_ids_and_warnings(client_kind: str) -> None:
    api = MockAPI(
        sequence(
            json_response(
                {"balance_dzd": 1},
                headers={
                    "x-request-id": "req_fixture_other",
                    "x-client-request-id": "fixture-client-id",
                    "X-DEVUP-Warnings": "a, b,,c",
                },
            )
        )
    )
    result = run_request(api.client(client_kind), "GET", "/user/balance")
    assert isinstance(result, APIResponse)
    assert result.data == {"balance_dzd": 1}
    assert isinstance(result.response, httpx.Response)
    assert result.response.status_code == 200
    assert result.request_id == "req_fixture_other"
    assert result.client_request_id == "fixture-client-id"
    assert result.warnings == ["a", "b", "c"]
    assert len(api.calls) == 1


def test_api_response_is_a_frozen_dataclass_with_the_contract_fields(client_kind: str) -> None:
    result = run_request(MockAPI(ok_handler).client(client_kind), "GET", "/models")
    assert [f.name for f in dataclasses.fields(result)] == [
        "data",
        "response",
        "request_id",
        "client_request_id",
        "warnings",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.data = None  # type: ignore[misc]


def test_json_response_body_is_read_before_returning(client_kind: str) -> None:
    stream = ChunkStream([b'{"a":', b"1}"])
    api = MockAPI(lambda request: httpx.Response(200, headers={"x-request-id": REQ_ID}, stream=stream))
    result = run_request(api.client(client_kind), "GET", "/models")
    assert result.data == {"a": 1}
    assert result.response.content == b'{"a":1}'
    assert stream.closed


def test_json_kind_strips_a_utf8_bom_before_parsing(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, content=b'\xef\xbb\xbf{"a":1}'))
    assert run_request(api.client(client_kind), "GET", "/models").data == {"a": 1}


def test_text_kind_returns_the_body_with_the_bom_stripped(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, content=b"\xef\xbb\xbfhello \xc3\xa9"))
    result = run_request(api.client(client_kind), "GET", "/audio/transcriptions", kind="text")
    assert result.data == "hello é"


def test_text_kind_replaces_invalid_utf8(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, content=b"a\xffb"))
    result = run_request(api.client(client_kind), "GET", "/x", kind="text")
    assert result.data == "a�b"


def test_text_kind_does_not_parse_json(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, content=b"{ not json"))
    assert run_request(api.client(client_kind), "GET", "/x", kind="text").data == "{ not json"


def test_raw_kind_returns_the_unread_response_and_the_caller_closes_it(client_kind: str) -> None:
    stream = ChunkStream([b"raw-", b"bytes"])
    headers = {"x-request-id": REQ_ID, "x-client-request-id": "cid"}
    api = MockAPI(lambda request: httpx.Response(200, headers=headers, stream=stream))
    result = run_request(api.client(client_kind), "GET", "/images/proxy", kind="raw", auth=False)
    assert isinstance(result.data, httpx.Response)
    assert result.data is result.response
    assert result.request_id == REQ_ID
    assert result.client_request_id == "cid"
    assert not stream.closed
    with pytest.raises(httpx.ResponseNotRead):
        _ = result.data.content
    assert read_response(client_kind, result.data) == b"raw-bytes"
    close_response(client_kind, result.data)
    assert result.data.is_closed


# --- body errors on a 2xx ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("content", [pytest.param(b"{ malformed", id="malformed"), pytest.param(b"", id="empty")])
def test_invalid_json_on_200_raises_devup_api_error_and_is_not_retried(
    client_kind: str, content: bytes, fake_clock: Any
) -> None:
    api = MockAPI(
        lambda request: httpx.Response(
            200, content=content, headers={"x-request-id": REQ_ID, "content-type": "application/json"}
        )
    )
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind), "GET", "/models", retry=DEFAULT_RETRY)
    err = info.value
    assert type(err) is DevupAPIError
    assert err.status == 200
    assert err.message == NOT_JSON_MESSAGE
    assert str(err) == NOT_JSON_MESSAGE
    assert err.request_id == REQ_ID
    assert isinstance(err.__cause__, json.JSONDecodeError)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("kind", ["json", "text"])
def test_body_read_failure_on_2xx_raises_connection_error_and_is_never_retried(
    client_kind: str, retry: RetryPolicy, kind: str, fake_clock: Any
) -> None:
    api = MockAPI(lambda request: failing_body())
    with pytest.raises(APIConnectionError) as info:
        run_request(api.client(client_kind, max_retries=5), "GET", "/models", retry=retry, kind=kind)
    err = info.value
    assert type(err) is APIConnectionError
    assert err.status == 0
    assert err.message == "Network error: connection reset"
    assert err.headers["x-request-id"] == REQ_ID
    assert err.headers["content-type"] == "application/json"
    assert err.request_id == REQ_ID
    assert isinstance(err.__cause__, httpx.ReadError)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


# --- conditional GET, 304 and redirects --------------------------------------------------------------------------


def not_modified(request: httpx.Request) -> httpx.Response:
    return httpx.Response(304, headers={"etag": '"etag"', "x-request-id": REQ_ID})


def test_if_none_match_sends_the_header_and_a_304_returns_none(client_kind: str) -> None:
    api = MockAPI(not_modified)
    result = run_request(api.client(client_kind), "GET", "/models", if_none_match='"etag"')
    assert api.calls[0].headers["if-none-match"] == '"etag"'
    assert isinstance(result, APIResponse)
    assert result.data is None
    assert result.response.status_code == 304
    assert result.request_id == REQ_ID
    assert len(api.calls) == 1


def test_if_none_match_with_raw_kind_returns_the_304_response(client_kind: str) -> None:
    api = MockAPI(not_modified)
    result = run_request(api.client(client_kind), "GET", "/me/avatar", if_none_match='"etag"', kind="raw")
    assert isinstance(result.data, httpx.Response)
    assert result.data.status_code == 304
    assert result.data is result.response
    close_response(client_kind, result.data)


@pytest.mark.parametrize("name", ["If-None-Match", "if-none-match", "IF-NONE-MATCH"])
def test_if_none_match_in_extra_headers_on_a_get_is_a_conditional_request(client_kind: str, name: str) -> None:
    api = MockAPI(not_modified)
    result = run_request(api.client(client_kind), "GET", "/models", options={"extra_headers": {name: '"etag"'}})
    assert api.calls[0].headers["if-none-match"] == '"etag"'
    assert result.data is None
    assert result.response.status_code == 304


def test_if_none_match_in_default_headers_on_a_get_is_a_conditional_request(client_kind: str) -> None:
    api = MockAPI(not_modified)
    result = run_request(api.client(client_kind, default_headers={"If-None-Match": '"etag"'}), "GET", "/models")
    assert result.data is None


def test_if_none_match_in_extra_headers_on_a_post_is_not_a_condition(client_kind: str) -> None:
    api = MockAPI(not_modified)
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind), "POST", "/x", body={}, options={"extra_headers": {"If-None-Match": "*"}})
    assert info.value.status == 304
    assert len(api.calls) == 1


def test_304_without_a_condition_raises_devup_api_error(client_kind: str) -> None:
    api = MockAPI(not_modified)
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind), "GET", "/models")
    assert type(info.value) is DevupAPIError
    assert info.value.status == 304
    assert info.value.request_id == REQ_ID
    assert len(api.calls) == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_not_followed_and_raises_devup_api_error(client_kind: str, status: int) -> None:
    api = MockAPI(
        sequence(
            httpx.Response(status, headers={"location": "https://files.example.test/elsewhere"}),
            json_response({"ok": True}),
        )
    )
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind), "GET", "/models", retry=DEFAULT_RETRY)
    assert type(info.value) is DevupAPIError
    assert info.value.status == status
    assert info.value.headers["location"] == "https://files.example.test/elsewhere"
    assert len(api.calls) == 1
    assert api.calls[0].url.endswith("/models")


# --- retries inside one deadline ---------------------------------------------------------------------------------


def test_retried_status_then_success_returns_the_success(client_kind: str, fake_clock: Any) -> None:
    events: list[DebugEvent] = []
    api = MockAPI(sequence(error_response(503, "busy"), json_response({"ok": True})))
    result = run_request(api.client(client_kind, on_debug=events.append), "GET", "/models")
    assert result.data == {"ok": True}
    assert len(api.calls) == 2
    assert events == [DebugEvent(type="retry", detail={"attempt": 1, "status": 503, "wait_ms": 0})]
    assert api.calls[0].url == api.calls[1].url
    assert api.calls[0].headers == api.calls[1].headers


@pytest.mark.parametrize("max_retries", [2, 3])
def test_server_waits_that_fit_the_deadline_are_taken_and_the_next_one_is_not(
    client_kind: str, max_retries: int, fake_clock: Any
) -> None:
    # timeout 10, Retry-After 4: attempt 1 at t=0 (wait 4 fits in 10), attempt 2 at t=4 (wait 4 fits in 6),
    # attempt 3 at t=8 (2 s left). With max_retries=2 it is the last attempt; with 3 the wait of 4 does not fit.
    api = MockAPI(sequence(error_response(503, "busy", headers={"retry-after": "4"})))
    with pytest.raises(InternalServerError) as info:
        run_request(api.client(client_kind, timeout=10, max_retries=max_retries), "GET", "/models")
    assert info.value.status == 503
    assert info.value.retry_after_ms == 4000
    assert len(api.calls) == 3
    assert fake_clock.sleeps == [4.0, 4.0]


def test_a_wait_that_uses_up_the_deadline_ends_in_a_timeout_at_the_next_attempt(
    client_kind: str, fake_clock: Any
) -> None:
    # timeout 8, Retry-After 4: t=0 wait 4 (fits 8), t=4 wait 4 (fits 4, not greater), t=8 → nothing left.
    api = MockAPI(sequence(error_response(503, "busy", headers={"retry-after": "4"})))
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=8, max_retries=2), "GET", "/models")
    assert info.value.message == timed_out("8")
    assert len(api.calls) == 2
    assert fake_clock.sleeps == [4.0, 4.0]


def test_a_server_wait_longer_than_the_time_left_raises_the_error_at_once(client_kind: str, fake_clock: Any) -> None:
    api = MockAPI(sequence(error_response(429, "busy", headers={"retry-after": "120"})))
    with pytest.raises(RateLimitError) as info:
        run_request(api.client(client_kind, timeout=5), "POST", "/chat/completions", body={})
    assert info.value.retry_after_ms == 120_000
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


def test_a_server_wait_above_sixty_seconds_is_honoured_when_it_fits(client_kind: str, fake_clock: Any) -> None:
    api = MockAPI(sequence(error_response(429, "busy", headers={"retry-after": "120"}), json_response({"ok": True})))
    result = run_request(api.client(client_kind), "GET", "/models")
    assert result.data == {"ok": True}
    assert fake_clock.sleeps == [120.0]


def test_each_attempt_gets_the_time_left_as_its_httpx_timeout(client_kind: str, fake_clock: Any) -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        if len(seen) == 1:
            return error_response(503, "busy", headers={"retry-after": "5"})
        return json_response({"ok": True})

    run_request(MockAPI(handler).client(client_kind, timeout=30), "GET", "/models")
    assert seen[0] == pytest.approx({"connect": 10.0, "read": 30.0, "write": 30.0, "pool": 30.0})
    assert seen[1] == pytest.approx({"connect": 10.0, "read": 25.0, "write": 25.0, "pool": 25.0})


def test_connect_timeout_is_capped_by_the_time_left(client_kind: str) -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return json_response({"ok": True})

    run_request(MockAPI(handler).client(client_kind, timeout=4), "GET", "/models")
    assert seen[0] == pytest.approx({"connect": 4.0, "read": 4.0, "write": 4.0, "pool": 4.0})


# --- timeouts ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("retry", POLICIES)
def test_deadline_passing_during_send_raises_api_timeout_error_and_is_never_retried(
    client_kind: str, retry: RetryPolicy, fake_clock: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock.advance(11)
        raise httpx.ConnectError("refused", request=request)

    api = MockAPI(handler)
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=10), "GET", "/models", retry=retry)
    err = info.value
    assert isinstance(err, APIConnectionError)
    assert isinstance(err, DevupAPIError)
    assert err.status == 0
    assert err.message == timed_out("10")
    assert str(err) == timed_out("10")
    assert isinstance(err.__cause__, httpx.ConnectError)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


@pytest.mark.parametrize("kind", ["json", "text"])
def test_deadline_passing_while_reading_a_2xx_body_raises_api_timeout_error(
    client_kind: str, kind: str, fake_clock: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock.advance(11)
        return failing_body()

    api = MockAPI(handler)
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=10), "GET", "/models", kind=kind)
    assert info.value.message == timed_out("10")
    assert len(api.calls) == 1


def test_deadline_passing_while_reading_an_error_body_raises_api_timeout_error(
    client_kind: str, fake_clock: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock.advance(11)
        return failing_body(503)

    api = MockAPI(handler)
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=10), "GET", "/models")
    assert info.value.message == timed_out("10")
    assert len(api.calls) == 1


def test_error_body_read_failure_before_the_deadline_falls_back_to_the_status(client_kind: str) -> None:
    api = MockAPI(lambda request: failing_body(400))
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind), "GET", "/models")
    assert info.value.status == 400
    assert info.value.message == "HTTP 400"
    assert len(api.calls) == 1


def test_deadline_passing_during_a_retry_wait_raises_api_timeout_error_before_the_next_attempt(
    client_kind: str, fake_clock: Any
) -> None:
    # The debug hook runs just before the wait; moving the clock there leaves no time for attempt 2.
    api = MockAPI(sequence(error_response(503, "busy", headers={"retry-after-ms": "0"}), json_response({"ok": True})))
    client = api.client(client_kind, timeout=10, on_debug=lambda event: fake_clock.advance(20))
    with pytest.raises(APITimeoutError) as info:
        run_request(client, "GET", "/models")
    assert info.value.message == timed_out("10")
    assert len(api.calls) == 1


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("exc_type", [httpx.ReadTimeout, httpx.WriteTimeout])
def test_read_and_write_timeouts_raise_api_timeout_error_and_are_never_retried(
    client_kind: str, retry: RetryPolicy, exc_type: type[httpx.TimeoutException], fake_clock: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("timed out", request=request)

    api = MockAPI(handler)
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=10, max_retries=5), "GET", "/models", retry=retry)
    assert info.value.message == timed_out("10")
    assert isinstance(info.value.__cause__, exc_type)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


@pytest.mark.parametrize(("timeout", "shown"), [(10, "10"), (10.0, "10"), (2.5, "2.5"), (0.25, "0.25"), (600.0, "600")])
def test_timeout_message_uses_the_general_number_format(
    client_kind: str, timeout: float, shown: str, fake_clock: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock.advance(timeout + 1)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(APITimeoutError) as info:
        run_request(MockAPI(handler).client(client_kind, timeout=timeout), "GET", "/models")
    assert info.value.message == timed_out(shown)


def test_per_call_timeout_overrides_a_longer_client_timeout(client_kind: str, fake_clock: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock.advance(6)
        raise httpx.ConnectError("refused", request=request)

    api = MockAPI(handler)
    with pytest.raises(APITimeoutError) as info:
        run_request(api.client(client_kind, timeout=600), "GET", "/models", options={"timeout": 5})
    assert info.value.message == timed_out("5")
    assert len(api.calls) == 1


def test_per_call_timeout_overrides_a_shorter_client_timeout(client_kind: str, fake_clock: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock.advance(6)
        raise httpx.ConnectError("refused", request=request)

    api = MockAPI(handler)
    with pytest.raises(APIConnectionError) as info:
        run_request(api.client(client_kind, timeout=5, max_retries=0), "GET", "/models", options={"timeout": 100})
    assert type(info.value) is APIConnectionError
    assert info.value.message == "Network error: refused"


def test_per_call_max_retries_overrides_the_client_value(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503, "busy")))
    with pytest.raises(InternalServerError):
        run_request(api.client(client_kind, max_retries=0), "GET", "/models", options={"max_retries": 1})
    assert len(api.calls) == 2

    other = MockAPI(sequence(error_response(503, "busy")))
    with pytest.raises(InternalServerError):
        run_request(other.client(client_kind, max_retries=2), "GET", "/models", options={"max_retries": 0})
    assert len(other.calls) == 1


# --- closed client and non-httpx exceptions ----------------------------------------------------------------------


def test_closed_client_raises_runtime_error_before_any_request(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    close_client(client)
    with pytest.raises(RuntimeError) as info:
        run_request(client, "GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE
    assert not isinstance(info.value, APIConnectionError)
    assert api.calls == []


def test_non_httpx_exception_is_raised_unchanged_and_not_retried_by_default(client_kind: str, fake_clock: Any) -> None:
    # P2 decision B-2: DEFAULT retries only httpx.TransportError; any other exception is raised unchanged, at once.
    # Before: APIConnectionError "Network error: boom" chained to the ValueError, max_retries + 1 = 3 calls.
    boom = ValueError("boom")

    def handler(request: httpx.Request) -> httpx.Response:
        raise boom

    api = MockAPI(handler)
    client = api.client(client_kind)
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", "/models", retry=DEFAULT_RETRY)
    assert info.value is boom
    assert client.max_retries == 2
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


def test_non_httpx_exception_is_raised_unchanged_on_a_billed_call(client_kind: str) -> None:
    # P2 decision B-2: the same on BILLED_POST. Before: APIConnectionError "Network error: boom", 1 call.
    boom = ValueError("boom")

    def handler(request: httpx.Request) -> httpx.Response:
        raise boom

    api = MockAPI(handler)
    with pytest.raises(ValueError) as info:
        run_request(api.client(client_kind), "POST", "/chat/completions", body={}, retry=BILLED_POST_RETRY)
    assert info.value is boom
    assert len(api.calls) == 1


def test_exception_without_a_message_is_described_as_unknown_error(client_kind: str) -> None:
    # P2 decision B-2: a ValueError() is now raised unchanged, so the empty message is checked on a transport
    # error. Before: the handler raised ValueError().
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("")

    api = MockAPI(handler)
    with pytest.raises(APIConnectionError) as info:
        run_request(api.client(client_kind, max_retries=0), "GET", "/models")
    assert info.value.message == "Network error: unknown error"


class _CallerStop(BaseException):
    """A BaseException that is not an Exception, like a cancellation."""


def test_base_exception_from_send_propagates_unchanged_and_is_not_retried(client_kind: str, fake_clock: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise _CallerStop()

    api = MockAPI(handler)
    with pytest.raises(_CallerStop):
        run_request(api.client(client_kind, max_retries=5), "GET", "/models")
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []
