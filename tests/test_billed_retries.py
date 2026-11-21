"""Retries and the overall deadline, at the transport level (``_request``).

Billed POSTs (``BILLED_POST_RETRY``) are retried only after 429 and 503, which the API answers before any work runs,
and after a connection that was never opened (``ConnectError``, ``ConnectTimeout``, ``PoolTimeout``; a TLS handshake
failure does not count). Every other request (``DEFAULT_RETRY``) is retried after 408, 429, 502, 503, 504 and any
connection error. A read timeout is never retried. ``max_retries`` caps everything.

Expected item counts of the retry table, written before any run:

- ``test_retry_table_status``: 16 statuses x 2 policies x 2 clients = 64 items.
- ``test_retry_table_connection``: 8 connection classes x 2 policies x 2 clients = 32 items.
- Retry table total: 64 + 32 = 96 items.

The other tests in this file are not part of the 96 and are counted as collected.
"""

from __future__ import annotations

import email.utils
import ssl
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from helpers.mock_transport import (
    MODEL_A,
    MockAPI,
    error_response,
    json_response,
    run_request,
    sequence,
    tls_failure,
)

from devupai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    DebugEvent,
    DevupAPIError,
    InternalServerError,
    NotFoundError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    UnsupportedMediaTypeError,
)
from devupai._client import BILLED_POST_RETRY, DEFAULT_RETRY, RetryPolicy, connection_never_opened

if TYPE_CHECKING:
    from conftest import FakeClock

    from devupai import AsyncDevupAI, DevupAI
    from devupai._response import APIResponse

# The status -> class table of the Node SDK's error mapper (errors.ts:234-252), written out here on purpose
# instead of calling make_api_error, so the test checks the mapping rather than repeating it.
STATUS_CLASS: dict[int, type[DevupAPIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: PaymentRequiredError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: DevupAPIError,
    409: ConflictError,
    413: PayloadTooLargeError,
    415: UnsupportedMediaTypeError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    501: InternalServerError,
    502: InternalServerError,
    503: InternalServerError,
    504: InternalServerError,
}

STATUSES = [400, 401, 402, 403, 404, 408, 409, 413, 415, 422, 429, 500, 501, 502, 503, 504]

POLICIES = [
    pytest.param(DEFAULT_RETRY, id="default"),
    pytest.param(BILLED_POST_RETRY, id="billed"),
]

# Each case builds a fresh exception per call.
CONNECTION_CASES: dict[str, Callable[[], Exception]] = {
    "connect_error": lambda: httpx.ConnectError("refused"),
    "connect_timeout": lambda: httpx.ConnectTimeout("connect timeout"),
    "pool_timeout": lambda: httpx.PoolTimeout("pool timeout"),
    "tls_failure": tls_failure,
    "read_error": lambda: httpx.ReadError("reset"),
    "write_error": lambda: httpx.WriteError("broken pipe"),
    "remote_protocol_error": lambda: httpx.RemoteProtocolError("server disconnected"),
    "read_timeout": lambda: httpx.ReadTimeout("read timeout"),
}

# Connection classes retried under each policy (contract sections 4-5, decisions Q8).
RETRIED_CONNECTION: dict[str, frozenset[str]] = {
    "all": frozenset(set(CONNECTION_CASES) - {"read_timeout"}),
    "not_sent": frozenset({"connect_error", "connect_timeout", "pool_timeout"}),
}


class Raiser:
    """A sequence entry that raises a fresh exception per call and keeps each one, for identity checks."""

    def __init__(self, factory: Callable[[], BaseException]) -> None:
        self.factory = factory
        self.raised: list[BaseException] = []

    def __call__(self, request: httpx.Request) -> BaseException:
        error = self.factory()
        self.raised.append(error)
        return error


def _ok() -> httpx.Response:
    return json_response({"ok": True})


def _post(client: DevupAI | AsyncDevupAI, **kwargs: Any) -> APIResponse[Any]:
    return run_request(client, "POST", "/x", body={"model": MODEL_A}, **kwargs)


def _retry_events(events: list[DebugEvent]) -> list[DebugEvent]:
    return [event for event in events if event.type == "retry"]


# ---------------------------------------------------------------------------------------------------------------
# The retry table: 64 + 32 = 96 items.
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("status", STATUSES)
def test_retry_table_status(status: int, policy: RetryPolicy, client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.5  # first jitter wait: 0.5 x min(60, 0.5 x 2**0) = 0.25 s
    api = MockAPI(sequence(error_response(status), _ok()))
    client = api.client(client_kind, max_retries=2)

    if status in policy.statuses:
        result = _post(client, retry=policy)
        assert result.data == {"ok": True}
        assert len(api.calls) == 2
        assert [call.method for call in api.calls] == ["POST", "POST"]
        assert api.calls[1].url == api.calls[0].url
        assert fake_clock.sleeps == [pytest.approx(0.25)]
    else:
        with pytest.raises(DevupAPIError) as info:
            _post(client, retry=policy)
        err = info.value
        assert type(err) is STATUS_CLASS[status]
        assert err.status == status
        assert len(api.calls) == 1
        assert fake_clock.sleeps == []


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("case", list(CONNECTION_CASES))
def test_retry_table_connection(case: str, policy: RetryPolicy, client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.5
    raiser = Raiser(CONNECTION_CASES[case])
    api = MockAPI(sequence(raiser, _ok()))
    client = api.client(client_kind, max_retries=2)

    if case in RETRIED_CONNECTION[policy.connection_errors]:
        result = _post(client, retry=policy)
        assert result.data == {"ok": True}
        assert len(api.calls) == 2
        assert len(fake_clock.sleeps) == 1
    else:
        expected = APITimeoutError if case == "read_timeout" else APIConnectionError
        with pytest.raises(APIConnectionError) as info:
            _post(client, retry=policy)
        err = info.value
        assert type(err) is expected
        assert err.status == 0
        assert len(raiser.raised) == 1
        assert err.__cause__ is raiser.raised[0]
        assert len(api.calls) == 1
        assert fake_clock.sleeps == []


# ---------------------------------------------------------------------------------------------------------------
# Further tests (not part of the 96).
# ---------------------------------------------------------------------------------------------------------------


def test_policies_match_the_node_sdk() -> None:
    assert DEFAULT_RETRY.statuses == frozenset({408, 429, 502, 503, 504})
    assert DEFAULT_RETRY.connection_errors == "all"
    assert BILLED_POST_RETRY.statuses == frozenset({429, 503})
    assert BILLED_POST_RETRY.connection_errors == "not_sent"


# --- connection_never_opened -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpx.ConnectError("refused"), id="connect_error"),
        pytest.param(httpx.ConnectTimeout("connect timeout"), id="connect_timeout"),
        pytest.param(httpx.PoolTimeout("pool timeout"), id="pool_timeout"),
    ],
)
def test_connection_never_opened_true(error: BaseException) -> None:
    assert connection_never_opened(error) is True


def test_connection_never_opened_tls_via_cause() -> None:
    error = tls_failure()
    assert isinstance(error.__cause__, ssl.SSLError)
    assert connection_never_opened(error) is False


def test_connection_never_opened_tls_via_context() -> None:
    try:
        try:
            raise ssl.SSLError(1, "handshake failure")
        except ssl.SSLError:
            raise httpx.ConnectError("TLS handshake failed")  # noqa: B904 - an implicit __context__ on purpose
    except httpx.ConnectError as caught:
        error = caught
    assert error.__cause__ is None
    assert isinstance(error.__context__, ssl.SSLError)
    assert connection_never_opened(error) is False


def _connect_error_with_ssl_at(depth: int) -> httpx.ConnectError:
    """A ConnectError (depth 0) whose __cause__ chain holds an ssl.SSLError at ``depth``."""
    head = httpx.ConnectError("refused")
    current: BaseException = head
    for level in range(1, depth):
        link = RuntimeError(f"layer {level}")
        current.__cause__ = link
        current = link
    current.__cause__ = ssl.SSLError(1, "handshake failure")
    return head


def test_connection_never_opened_ssl_at_depth_5_is_inspected() -> None:
    assert connection_never_opened(_connect_error_with_ssl_at(5)) is False


def test_connection_never_opened_chain_deeper_than_5_is_not_inspected() -> None:
    assert connection_never_opened(_connect_error_with_ssl_at(6)) is True


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpx.ReadError("reset"), id="read_error"),
        pytest.param(ValueError("not a transport error"), id="value_error"),
    ],
)
def test_connection_never_opened_false(error: BaseException) -> None:
    assert connection_never_opened(error) is False


# --- Retry-After on a billed 429 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "value", "expected"),
    [
        pytest.param("retry-after-ms", "1500", 1.5, id="retry-after-ms"),
        pytest.param("Retry-After", "2", 2.0, id="retry-after-seconds"),
    ],
)
def test_billed_429_waits_what_the_api_asks(
    header: str, value: str, expected: float, client_kind: str, fake_clock: FakeClock
) -> None:
    fake_clock.random_value = 0.5  # must not be used when the API sends a wait
    api = MockAPI(sequence(error_response(429, headers={header: value}), _ok()))
    client = api.client(client_kind, max_retries=2)
    result = _post(client, retry=BILLED_POST_RETRY)
    assert result.data == {"ok": True}
    assert len(api.calls) == 2
    assert fake_clock.sleeps == [pytest.approx(expected)]


def test_billed_429_retry_after_http_date(client_kind: str, fake_clock: FakeClock) -> None:
    date = email.utils.formatdate(fake_clock.wall + 3, usegmt=True)
    api = MockAPI(sequence(error_response(429, headers={"Retry-After": date}), _ok()))
    client = api.client(client_kind, max_retries=2)
    result = _post(client, retry=BILLED_POST_RETRY)
    assert result.data == {"ok": True}
    assert len(api.calls) == 2
    assert fake_clock.sleeps == [pytest.approx(3.0)]


def test_billed_429_wait_longer_than_time_left_raises_at_once(client_kind: str, fake_clock: FakeClock) -> None:
    api = MockAPI(sequence(error_response(429, headers={"Retry-After": "10"}), _ok()))
    client = api.client(client_kind, max_retries=2, timeout=5)
    with pytest.raises(RateLimitError) as info:
        _post(client, retry=BILLED_POST_RETRY)
    assert info.value.retry_after_ms == 10_000
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


# --- Jitter, caps and attempt counts --------------------------------------------------------------------------------


def test_jitter_without_a_server_wait(client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.5
    api = MockAPI(sequence(error_response(503), error_response(503), _ok()))
    client = api.client(client_kind, max_retries=2)
    result = _post(client, retry=BILLED_POST_RETRY)
    assert result.data == {"ok": True}
    assert len(api.calls) == 3
    assert fake_clock.sleeps == [pytest.approx(0.25), pytest.approx(0.5)]


def test_third_503_raises_after_three_calls(client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.5
    api = MockAPI(sequence(error_response(503)))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError) as info:
        _post(client, retry=BILLED_POST_RETRY)
    assert info.value.status == 503
    assert len(api.calls) == 3
    assert fake_clock.sleeps == [pytest.approx(0.25), pytest.approx(0.5)]


def test_jitter_is_capped_at_60_seconds(client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.999999
    api = MockAPI(sequence(error_response(503)))
    client = api.client(client_kind, max_retries=10)
    with pytest.raises(InternalServerError):
        _post(client, retry=BILLED_POST_RETRY)
    assert len(api.calls) == 11
    assert len(fake_clock.sleeps) == 10
    assert all(wait <= 60.0 for wait in fake_clock.sleeps)
    assert fake_clock.sleeps[6] == pytest.approx(32 * 0.999999)
    assert fake_clock.sleeps[7] == pytest.approx(60 * 0.999999)
    assert fake_clock.sleeps[9] == pytest.approx(60 * 0.999999)


def test_max_retries_zero_sends_one_request(client_kind: str, fake_clock: FakeClock) -> None:
    api = MockAPI(sequence(error_response(503)))
    client = api.client(client_kind, max_retries=0)
    with pytest.raises(InternalServerError):
        _post(client, retry=BILLED_POST_RETRY)
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


@pytest.mark.parametrize("max_retries", [0, 1, 3])
def test_attempts_are_max_retries_plus_one(max_retries: int, client_kind: str, fake_clock: FakeClock) -> None:
    api = MockAPI(sequence(error_response(503)))
    client = api.client(client_kind, max_retries=max_retries)
    with pytest.raises(InternalServerError):
        _post(client, retry=BILLED_POST_RETRY)
    assert len(api.calls) == max_retries + 1
    assert len(fake_clock.sleeps) == max_retries


def test_per_call_max_retries_overrides_the_client(client_kind: str, fake_clock: FakeClock) -> None:
    api = MockAPI(sequence(error_response(503), _ok()))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError):
        _post(client, retry=BILLED_POST_RETRY, options={"max_retries": 0})
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


# --- The debug hook -------------------------------------------------------------------------------------------------


def test_on_debug_receives_a_retry_event_for_a_status(client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.5
    events: list[DebugEvent] = []
    api = MockAPI(sequence(error_response(503), _ok()))
    client = api.client(client_kind, max_retries=2, on_debug=events.append)
    _post(client, retry=BILLED_POST_RETRY)
    assert _retry_events(events) == [DebugEvent(type="retry", detail={"attempt": 1, "status": 503, "wait_ms": 250})]


def test_on_debug_receives_status_0_for_a_connection_retry(client_kind: str, fake_clock: FakeClock) -> None:
    fake_clock.random_value = 0.5
    events: list[DebugEvent] = []
    api = MockAPI(sequence(Raiser(lambda: httpx.ConnectError("refused")), _ok()))
    client = api.client(client_kind, max_retries=2, on_debug=events.append)
    _post(client, retry=BILLED_POST_RETRY)
    assert _retry_events(events) == [DebugEvent(type="retry", detail={"attempt": 1, "status": 0, "wait_ms": 250})]


def test_on_debug_that_raises_does_not_change_the_outcome(client_kind: str, fake_clock: FakeClock) -> None:
    def broken_hook(event: DebugEvent) -> None:
        raise RuntimeError("hook failure")

    api = MockAPI(sequence(error_response(503), _ok()))
    client = api.client(client_kind, max_retries=2, on_debug=broken_hook)
    result = _post(client, retry=BILLED_POST_RETRY)
    assert result.data == {"ok": True}
    assert len(api.calls) == 2
