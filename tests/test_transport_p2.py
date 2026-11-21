"""P2 transport changes: a caller client's auth never runs, URLs with credentials are refused, and only
``httpx.TransportError`` subclasses follow the retry rules.

Decisions B, answers 1-2 (contract ``p2/contract.md`` section 1):

1. Every send passes ``auth=None``, so the ``auth`` of a caller ``http_client`` (``BasicAuth``, a custom
   ``httpx.Auth``) is never applied. The only ``Authorization`` header is the SDK's own ``Bearer <api_key>``.
2. A ``base_url`` with a user name or password is refused at construction, an absolute ``url=`` with one when the URL
   is built. Neither message repeats the URL.
3. Only ``httpx.TransportError`` subclasses follow the P1 rules (timeouts, never-opened, ``LocalProtocolError``,
   DEFAULT / BILLED_POST). Any other ``Exception`` raised by ``send`` (an event hook, a caller's own error, an
   ``httpx.HTTPStatusError``) or by the body read is raised unchanged and at once, on both policies.

None of the 96 rows of the retry table (``tests/test_billed_retries.py``: 64 status rows and 32 connection rows) change
their expected value: every row raises an HTTP status or an ``httpx.TransportError`` subclass. This file only adds new
rows; it does not repeat the table.
"""

from __future__ import annotations

import traceback
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from typing import Any

import httpx
import pytest
from helpers.mock_transport import (
    BASE,
    MODEL_A,
    REQ_ID,
    TEST_KEY,
    MockAPI,
    error_response,
    json_response,
    run_request,
    sequence,
)

from devupai import AsyncDevupAI, DebugEvent, DevupAI
from devupai._client import BILLED_POST_RETRY, DEFAULT_RETRY, RetryPolicy

BASE_URL_CREDENTIALS_MESSAGE = "base_url must not contain a user name or password."
URL_CREDENTIALS_MESSAGE = "url must not contain a user name or password."

POLICIES = [pytest.param(DEFAULT_RETRY, id="default"), pytest.param(BILLED_POST_RETRY, id="billed-post")]


def ok_handler(request: httpx.Request) -> httpx.Response:
    return json_response({"ok": True})


def caller_client(api: MockAPI, client_kind: str, **http_kwargs: Any) -> DevupAI | AsyncDevupAI:
    """An SDK client over a caller ``http_client`` built with ``http_kwargs`` and answered by ``api``."""
    transport = httpx.MockTransport(api._handle)
    http: httpx.Client | httpx.AsyncClient
    if client_kind == "sync":
        http = httpx.Client(transport=transport, **http_kwargs)
    else:
        http = httpx.AsyncClient(transport=transport, **http_kwargs)
    return api.client(client_kind, http_client=http)


def everything_about(err: BaseException) -> str:
    return str(err) + repr(err) + "".join(traceback.format_exception(err))


def call(client: DevupAI | AsyncDevupAI, retry: RetryPolicy, **kwargs: Any) -> Any:
    """A GET for DEFAULT_RETRY, a billed POST for BILLED_POST_RETRY."""
    if retry is BILLED_POST_RETRY:
        return run_request(client, "POST", "/chat/completions", body={"model": MODEL_A}, retry=retry, **kwargs)
    return run_request(client, "GET", "/models", retry=retry, **kwargs)


# --- 1. a caller http_client's auth is never applied --------------------------------------------------------------


class RecordingAuth(httpx.Auth):
    """A custom ``httpx.Auth`` that counts how often its flow starts and sets its own header."""

    def __init__(self) -> None:
        self.flows = 0

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        self.flows += 1
        request.headers["Authorization"] = "Fixture caller-auth"
        yield request


def test_caller_basic_auth_is_not_applied_and_the_bearer_key_is_sent(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = caller_client(api, client_kind, auth=("user", "pass"))
    assert run_request(client, "GET", "/models").data == {"ok": True}
    assert len(api.calls) == 1
    assert api.calls[0].headers["authorization"] == f"Bearer {TEST_KEY}"


def test_caller_custom_auth_flow_never_runs_and_the_bearer_key_is_sent(client_kind: str) -> None:
    auth = RecordingAuth()
    api = MockAPI(ok_handler)
    client = caller_client(api, client_kind, auth=auth)
    assert run_request(client, "POST", "/chat/completions", body={"model": MODEL_A}).data == {"ok": True}
    assert auth.flows == 0
    assert len(api.calls) == 1
    assert api.calls[0].headers["authorization"] == f"Bearer {TEST_KEY}"


def test_caller_auth_is_not_applied_on_any_retry_attempt(client_kind: str, fake_clock: Any) -> None:
    auth = RecordingAuth()
    api = MockAPI(sequence(error_response(503, "busy"), json_response({"ok": True})))
    client = caller_client(api, client_kind, auth=auth)
    assert run_request(client, "GET", "/models").data == {"ok": True}
    assert auth.flows == 0
    assert [c.headers["authorization"] for c in api.calls] == [f"Bearer {TEST_KEY}"] * 2


@pytest.mark.parametrize(
    "make_auth",
    [pytest.param(lambda: ("user", "pass"), id="basic"), pytest.param(RecordingAuth, id="custom")],
)
def test_caller_auth_is_not_applied_to_a_request_sent_without_the_key(
    client_kind: str, make_auth: Callable[[], Any]
) -> None:
    auth = make_auth()
    api = MockAPI(sequence(httpx.Response(200, content=b"fixture-bytes")))
    client = caller_client(api, client_kind, auth=auth)
    result = run_request(client, "GET", url="https://files.example.test/a.png", auth=False, kind="text")
    assert result.data == "fixture-bytes"
    assert len(api.calls) == 1
    assert "authorization" not in api.calls[0].headers
    if isinstance(auth, RecordingAuth):
        assert auth.flows == 0


# --- 2. a URL with a user name or password is refused --------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("https://user:pw@api.example.test/v1", id="user-and-password"),
        pytest.param("https://user@api.example.test/v1", id="user-only"),
        pytest.param("https://:pw@api.example.test/v1", id="password-only"),
        pytest.param("https://fixture-login:fixture-pass@api.example.test/v1/", id="trailing-slash"),
        pytest.param("http://fixture-login:fixture-pass@localhost:8080/v1", id="http-with-port"),
    ],
)
def test_base_url_with_credentials_raises_value_error_at_construction(client_kind: str, base_url: str) -> None:
    api = MockAPI(ok_handler)
    with pytest.raises(ValueError) as info:
        api.client(client_kind, base_url=base_url)
    err = info.value
    assert str(err) == BASE_URL_CREDENTIALS_MESSAGE
    shown = everything_about(err)
    # "user" itself is part of the message ("a user name"), so the user-name checks look for the URL forms.
    for leaked in ("user:", "user@", "pw", "fixture-login", "fixture-pass", "api.example.test", "localhost", "://"):
        assert leaked not in str(err)
    for secret in ("fixture-pass", ":pw@"):
        assert secret not in shown
    assert api.calls == []


@pytest.mark.parametrize("client_cls", [DevupAI, AsyncDevupAI], ids=["sync", "async"])
def test_base_url_with_credentials_raises_without_a_caller_http_client(client_cls: type[Any]) -> None:
    with pytest.raises(ValueError) as info:
        client_cls(api_key=TEST_KEY, base_url="https://fixture-login:fixture-pass@api.example.test/v1")
    assert str(info.value) == BASE_URL_CREDENTIALS_MESSAGE


def test_an_at_sign_in_the_base_url_path_is_not_a_credential(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, base_url="https://api.example.test/v1/@fixture")
    assert run_request(client, "GET", "/models").data == {"ok": True}
    assert api.calls[0].url == "https://api.example.test/v1/@fixture/models"
    assert api.calls[0].headers["authorization"] == f"Bearer {TEST_KEY}"


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://user:pw@files.example.test/a.png", id="user-and-password"),
        pytest.param("https://user@files.example.test/a.png", id="user-only"),
        pytest.param("https://:pw@files.example.test/a.png", id="password-only"),
    ],
)
@pytest.mark.parametrize("auth", [pytest.param(False, id="no-key"), pytest.param(True, id="key")])
def test_absolute_url_with_credentials_raises_value_error_and_sends_nothing(
    client_kind: str, url: str, auth: bool, fake_clock: Any
) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", url=url, auth=auth, kind="text")
    err = info.value
    assert str(err) == URL_CREDENTIALS_MESSAGE
    for leaked in ("user:", "user@", "pw", "files.example.test", "://"):
        assert leaked not in str(err)
    assert ":pw@" not in everything_about(err)
    assert api.calls == []
    assert fake_clock.sleeps == []


def test_absolute_url_without_credentials_still_works(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind), "GET", url=f"{BASE}/models")
    assert [c.url for c in api.calls] == [f"{BASE}/models"]


# --- 3. only httpx.TransportError follows the retry rules ----------------------------------------------------------


def _value_error(request: httpx.Request) -> Exception:
    return ValueError("fixture value error")


def _runtime_error(request: httpx.Request) -> Exception:
    return RuntimeError("fixture runtime error")


def _key_error(request: httpx.Request) -> Exception:
    return KeyError("fixture")


def _http_status_error(request: httpx.Request) -> Exception:
    return httpx.HTTPStatusError("fixture status", request=request, response=httpx.Response(500, request=request))


def _too_many_redirects(request: httpx.Request) -> Exception:
    # An httpx.RequestError that is not an httpx.TransportError.
    return httpx.TooManyRedirects("fixture redirects", request=request)


NON_TRANSPORT_ERRORS = [
    pytest.param(_value_error, ValueError, id="ValueError"),
    pytest.param(_runtime_error, RuntimeError, id="RuntimeError"),
    pytest.param(_key_error, KeyError, id="KeyError"),
    pytest.param(_http_status_error, httpx.HTTPStatusError, id="HTTPStatusError"),
    pytest.param(_too_many_redirects, httpx.TooManyRedirects, id="TooManyRedirects"),
]


class Raiser:
    """A handler that raises a fresh exception per call and keeps each one, for identity checks."""

    def __init__(self, factory: Callable[[httpx.Request], Exception]) -> None:
        self.factory = factory
        self.raised: list[Exception] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        error = self.factory(request)
        self.raised.append(error)
        raise error


def test_the_cases_are_not_transport_errors() -> None:
    request = httpx.Request("GET", f"{BASE}/models")
    for case in NON_TRANSPORT_ERRORS:
        factory = case.values[0]
        assert callable(factory)
        assert not isinstance(factory(request), httpx.TransportError), case.id


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize(("factory", "exc_type"), NON_TRANSPORT_ERRORS)
def test_non_transport_error_from_send_is_raised_unchanged_after_one_call(
    client_kind: str,
    retry: RetryPolicy,
    factory: Callable[[httpx.Request], Exception],
    exc_type: type[Exception],
    fake_clock: Any,
) -> None:
    events: list[DebugEvent] = []
    raiser = Raiser(factory)
    api = MockAPI(raiser)
    client = api.client(client_kind, max_retries=5, on_debug=events.append)
    with pytest.raises(exc_type) as info:
        call(client, retry)
    assert type(info.value) is exc_type
    assert len(raiser.raised) == 1
    assert info.value is raiser.raised[0]
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []
    assert events == []


def test_a_non_transport_error_after_a_retried_connect_error_skips_the_remaining_attempts(
    client_kind: str, fake_clock: Any
) -> None:
    second = RuntimeError("fixture second attempt")
    api = MockAPI(sequence(httpx.ConnectError("refused"), second, json_response({"ok": True})))
    with pytest.raises(RuntimeError) as info:
        run_request(api.client(client_kind, max_retries=5), "GET", "/models", retry=DEFAULT_RETRY)
    assert info.value is second
    assert len(api.calls) == 2
    assert len(fake_clock.sleeps) == 1


@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(lambda: httpx.ConnectError("refused"), id="connect_error"),
        pytest.param(lambda: httpx.ReadError("reset"), id="read_error"),
        pytest.param(lambda: httpx.RemoteProtocolError("server disconnected"), id="remote_protocol_error"),
    ],
)
def test_transport_errors_are_still_retried_by_default(
    client_kind: str, make_error: Callable[[], Exception], fake_clock: Any
) -> None:
    fake_clock.random_value = 0.5
    api = MockAPI(sequence(make_error(), make_error(), json_response({"ok": True})))
    result = run_request(api.client(client_kind, max_retries=2), "GET", "/models", retry=DEFAULT_RETRY)
    assert result.data == {"ok": True}
    assert len(api.calls) == 3
    assert fake_clock.sleeps == [pytest.approx(0.25), pytest.approx(0.5)]


# --- event hooks of a caller client --------------------------------------------------------------------------------


def raising_hook(client_kind: str, error: Exception) -> Callable[..., Any]:
    """An event hook raising ``error``: a function for ``httpx.Client``, a coroutine function for the async one."""
    if client_kind == "sync":

        def hook(event: object) -> None:
            raise error

        return hook

    async def async_hook(event: object) -> None:
        raise error

    return async_hook


def recording_hook(client_kind: str, seen: list[object]) -> Callable[..., Any]:
    if client_kind == "sync":

        def hook(event: object) -> None:
            seen.append(event)

        return hook

    async def async_hook(event: object) -> None:
        seen.append(event)

    return async_hook


@pytest.mark.parametrize("retry", POLICIES)
def test_response_event_hook_error_is_raised_at_once_after_one_call(
    client_kind: str, retry: RetryPolicy, fake_clock: Any
) -> None:
    error = RuntimeError("fixture hook failure")
    api = MockAPI(ok_handler)
    http_kwargs = {"event_hooks": {"response": [raising_hook(client_kind, error)]}}
    client = caller_client(api, client_kind, **http_kwargs)
    with pytest.raises(RuntimeError) as info:
        call(client, retry)
    assert info.value is error
    assert len(api.calls) == 1
    assert fake_clock.sleeps == []


@pytest.mark.parametrize("retry", POLICIES)
def test_request_event_hook_error_is_raised_at_once_and_nothing_is_sent(
    client_kind: str, retry: RetryPolicy, fake_clock: Any
) -> None:
    error = RuntimeError("fixture hook failure")
    api = MockAPI(ok_handler)
    http_kwargs = {"event_hooks": {"request": [raising_hook(client_kind, error)]}}
    client = caller_client(api, client_kind, **http_kwargs)
    with pytest.raises(RuntimeError) as info:
        call(client, retry)
    assert info.value is error
    assert api.calls == []
    assert fake_clock.sleeps == []


def test_caller_event_hooks_still_run_on_every_attempt(client_kind: str, fake_clock: Any) -> None:
    requests: list[object] = []
    responses: list[object] = []
    api = MockAPI(sequence(error_response(503, "busy"), json_response({"ok": True})))
    hooks = {"request": [recording_hook(client_kind, requests)], "response": [recording_hook(client_kind, responses)]}
    client = caller_client(api, client_kind, event_hooks=hooks)
    assert run_request(client, "GET", "/models").data == {"ok": True}
    assert len(requests) == 2
    assert [r.status_code for r in responses if isinstance(r, httpx.Response)] == [503, 200]


# --- the body read -------------------------------------------------------------------------------------------------


class FailingStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body that sends one chunk, then raises ``error``."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield b'{"a":'
        raise self.error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"a":'
        raise self.error

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("retry", POLICIES)
@pytest.mark.parametrize("kind", ["json", "text"])
def test_non_transport_error_while_reading_a_2xx_body_is_raised_unchanged(
    client_kind: str, retry: RetryPolicy, kind: str, fake_clock: Any
) -> None:
    error = RuntimeError("fixture body failure")
    stream = FailingStream(error)
    api = MockAPI(lambda request: httpx.Response(200, headers={"x-request-id": REQ_ID}, stream=stream))
    client = api.client(client_kind, max_retries=5)
    with pytest.raises(RuntimeError) as info:
        call(client, retry, kind=kind)
    assert info.value is error
    assert len(api.calls) == 1
    assert stream.closed
    assert fake_clock.sleeps == []
