"""The DEVUP AI API clients: options, request building, retries, the overall deadline and response reading.

``DevupAI`` and ``AsyncDevupAI`` share one base class for validation, request building, error classification and
retry decisions. Only sending, reading, sleeping and closing differ between them.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import ssl
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Self, TypedDict, TypeVar, cast

import anyio
import httpx

from devupai._errors import APIConnectionError, APITimeoutError, DevupAPIError, error_from_body
from devupai._response import APIResponse, make_api_response
from devupai._version import __version__

__all__ = [
    "BILLED_POST_RETRY",
    "CONNECT_TIMEOUT",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY",
    "DEFAULT_TIMEOUT",
    "INITIAL_RETRY_DELAY",
    "MAX_RETRY_DELAY",
    "USER_AGENT",
    "AsyncDevupAI",
    "DebugEvent",
    "DevupAI",
    "RequestOptions",
    "RetryPolicy",
    "connection_never_opened",
]

# Module-level hooks. They are always called through the module, so tests can replace them.
_sleep: Callable[[float], None] = time.sleep


async def _async_sleep(seconds: float) -> None:
    """Waits ``seconds`` without blocking the event loop."""
    await anyio.sleep(seconds)


_random: Callable[[], float] = random.random
_monotonic: Callable[[], float] = time.monotonic
_time: Callable[[], float] = time.time

DEFAULT_BASE_URL = "https://api.devupai.com/v1"
"""Base URL of the API."""
DEFAULT_MAX_RETRIES = 2
"""How many times a request is sent again after a transient error, by default."""
DEFAULT_TIMEOUT = 600.0
"""Timeout of a request in seconds, by default."""
CONNECT_TIMEOUT = 10.0
"""Longest time, in seconds, that opening a connection may take."""
INITIAL_RETRY_DELAY = 0.5
"""Upper bound, in seconds, of the first random wait before a new attempt."""
MAX_RETRY_DELAY = 60.0
"""Largest upper bound, in seconds, of a random wait before a new attempt."""
USER_AGENT = f"devupai-python/{__version__}"
"""Value of the ``User-Agent`` header sent with every request."""

# Beyond this exponent the random wait is already capped by MAX_RETRY_DELAY; capping it avoids a float overflow.
_MAX_BACKOFF_EXPONENT = 32
# How deep connection_never_opened follows __cause__ / __context__.
_MAX_CAUSE_DEPTH = 5

_MAX_RETRIES_MESSAGE = "max_retries must be a non-negative integer."
_TIMEOUT_MESSAGE = "timeout must be a positive number of seconds."
_IDLE_TIMEOUT_MESSAGE = "stream_idle_timeout must be a non-negative number of seconds."
_KEY_TYPE_MESSAGE = "api_key must be a string."
_KEY_CHARS_MESSAGE = "api_key contains characters that are not allowed in an HTTP header."
_MISSING_KEY_MESSAGE = "DEVUP AI API key is missing. Pass api_key=... or set the DEVUP_API_KEY environment variable."
_HEADER_NAME_MESSAGE = "A header name contains characters that are not allowed in an HTTP header."
_CLOSED_MESSAGE = "The client is closed."

# Characters refused in a header value once leading and trailing spaces and tabs are removed: control
# characters (CR, LF and NUL included), DEL, and anything outside latin-1.
_FORBIDDEN_HEADER_CHARS = re.compile(r"[^\t\x20-\x7e\x80-\xff]")
# Removed from both ends of every header value before it is checked, as the Fetch Headers class does.
_HEADER_WHITESPACE = " \t"
_HEADER_PROTOCOL_MESSAGE = "The request could not be sent: a header value is not valid."
_BASE_URL_MESSAGE = "base_url must be an absolute http or https URL."
_URL_MESSAGE = "url must be an absolute http or https URL."
# An RFC 9110 token: the characters allowed in a header name.
_HEADER_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")

ResponseKind = Literal["json", "text", "raw"]


@dataclass(frozen=True)
class RetryPolicy:
    """Which answers a request is sent again after.

    ``statuses`` are the HTTP statuses after which the request is sent again. ``connection_errors`` is ``"all"`` for
    every connection error, or ``"not_sent"`` for only an error showing that the connection was never opened, so
    that nothing of the request reached the API.
    """

    statuses: frozenset[int]
    connection_errors: Literal["all", "not_sent"]


DEFAULT_RETRY = RetryPolicy(frozenset({408, 429, 502, 503, 504}), "all")
"""The retries of every request that is not a billed ``POST``: 408, 429, 502, 503, 504 and any connection error."""

BILLED_POST_RETRY = RetryPolicy(frozenset({429, 503}), "not_sent")
"""The retries of a billed ``POST``: only 429 and 503, which the API answers before any work runs, and a connection
that was never opened. A 408, 502 or 504 answer, or a connection lost after it was opened, can come after the
request was processed and charged, so it is not retried."""


def connection_never_opened(error: BaseException) -> bool:
    """True when ``error`` shows that the connection was never opened, so the request was not sent.

    Only ``httpx.ConnectError``, ``httpx.ConnectTimeout`` and ``httpx.PoolTimeout`` count. A TLS handshake failure
    (an ``ssl.SSLError`` on the error or its ``__cause__`` / ``__context__`` chain, up to depth 5) does not.
    """
    if not isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return False
    current: BaseException | None = error
    depth = 0
    while current is not None and depth <= _MAX_CAUSE_DEPTH:
        if isinstance(current, ssl.SSLError):
            return False
        current = current.__cause__ if current.__cause__ is not None else current.__context__
        depth += 1
    return True


@dataclass(frozen=True)
class DebugEvent:
    """An event passed to the ``on_debug`` client option.

    ``type`` is ``"malformed_frame"`` when a stream frame that is not valid JSON was skipped, and ``"retry"`` when
    the request is sent again. For ``"retry"``, ``detail`` is ``{"attempt", "status", "wait_ms"}`` (``status`` is
    ``0`` when no response was received).
    """

    type: Literal["malformed_frame", "retry"]
    detail: dict[str, object]


class RequestOptions(TypedDict, total=False):
    """Options of one request. Values given here override the client options."""

    timeout: float
    """Timeout of the request in seconds."""
    max_retries: int
    """How many times the request is sent again after a transient error."""
    extra_headers: Mapping[str, str]
    """Headers to send with the request."""
    stream_idle_timeout: float
    """Ends a stream when no data arrives for this many seconds; ``0`` turns it off."""


# --- validation ----------------------------------------------------------------------------------------------------


def _check_max_retries(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(_MAX_RETRIES_MESSAGE)
    if value < 0:
        raise ValueError(_MAX_RETRIES_MESSAGE)
    return value


def _check_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(_TIMEOUT_MESSAGE)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(_TIMEOUT_MESSAGE)
    return number


def _check_idle_timeout(value: object) -> float | None:
    """``0`` turns the idle timeout off and is stored as ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(_IDLE_TIMEOUT_MESSAGE)
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(_IDLE_TIMEOUT_MESSAGE)
    return None if number == 0 else number


def _check_api_key(value: object) -> str:
    """The error message never contains the key."""
    if not isinstance(value, str):
        raise TypeError(_KEY_TYPE_MESSAGE)
    if _FORBIDDEN_HEADER_CHARS.search(value):
        raise ValueError(_KEY_CHARS_MESSAGE)
    return value


def _check_base_url(value: object) -> str:
    """``value`` or the default base URL, without a trailing ``/``."""
    if value is not None and not isinstance(value, str):
        raise TypeError("base_url must be a string.")
    return (value or DEFAULT_BASE_URL).rstrip("/")


_HttpClientT = TypeVar("_HttpClientT", httpx.Client, httpx.AsyncClient)


def _check_http_client(value: object, expected: type[_HttpClientT], message: str) -> _HttpClientT | None:
    """``value`` when it is an instance of ``expected``, ``None`` when it is ``None``; anything else raises."""
    if value is None:
        return None
    if not isinstance(value, expected):
        raise TypeError(message)
    return value


def _resolve_api_key(value: object) -> str:
    """The key given, else ``DEVUP_API_KEY``, else (only when that variable is absent) ``DEVUPAI_API_KEY``."""
    if value is not None:
        return _check_api_key(value)
    if "DEVUP_API_KEY" in os.environ:
        found: str | None = os.environ["DEVUP_API_KEY"]
    else:
        found = os.environ.get("DEVUPAI_API_KEY")
    if found is None or not found.strip():
        raise ValueError(_MISSING_KEY_MESSAGE)
    return _check_api_key(found)


# --- request building ----------------------------------------------------------------------------------------------

# Headers keyed by lower-cased name; each entry is the name as given and its value.
_HeaderMap = dict[str, tuple[str, str]]


def _set_header(headers: _HeaderMap, name: object, value: str) -> None:
    if not isinstance(name, str):
        raise ValueError(_HEADER_NAME_MESSAGE)
    headers.pop(name.lower(), None)
    headers[name.lower()] = (name, value)


def _merge_headers(headers: _HeaderMap, source: Mapping[str, object] | None) -> None:
    """Later values win; names are compared without regard to case. Values that are not strings are skipped."""
    if source is None:
        return
    items: Iterable[tuple[object, object]] = source.items()
    for name, value in items:
        if isinstance(value, str):
            _set_header(headers, name, value)


def _encode_headers(headers: _HeaderMap) -> list[tuple[bytes, bytes]]:
    """Checks every header name and value. The error names the header but never contains its value."""
    encoded: list[tuple[bytes, bytes]] = []
    for name, given_value in headers.values():
        if not _HEADER_TOKEN.fullmatch(name):
            raise ValueError(_HEADER_NAME_MESSAGE)
        value_message = f'Header "{name}" has a value that is not allowed in an HTTP header.'
        value = given_value.strip(_HEADER_WHITESPACE)
        if _FORBIDDEN_HEADER_CHARS.search(value):
            raise ValueError(value_message)
        try:
            raw_value = value.encode("latin-1")
        except UnicodeEncodeError:
            # The original error would repeat the value.
            raise ValueError(value_message) from None
        encoded.append((name.encode("ascii"), raw_value))
    return encoded


def _describe(error: BaseException) -> str:
    return str(error) or "unknown error"


def _decode(content: bytes) -> str:
    return content.decode("utf-8-sig", errors="replace")


class _DeadlinePassed(Exception):
    """Raised while a body is read once the deadline of the call has passed."""


class _DeadlineStream(httpx.SyncByteStream):
    """A response body that stops with ``_DeadlinePassed`` when a chunk arrives after the deadline."""

    def __init__(self, inner: httpx.SyncByteStream, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            if _monotonic() >= self._deadline:
                raise _DeadlinePassed()
            yield chunk

    def close(self) -> None:
        self._inner.close()


class _AsyncDeadlineStream(httpx.AsyncByteStream):
    """The async twin of :class:`_DeadlineStream`."""

    def __init__(self, inner: httpx.AsyncByteStream, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            if _monotonic() >= self._deadline:
                raise _DeadlinePassed()
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


def _read_with_deadline(response: httpx.Response, deadline: float) -> bytes:
    """Reads the whole body; a chunk that arrives after the deadline raises ``_DeadlinePassed``."""
    if isinstance(response.stream, httpx.SyncByteStream):
        response.stream = _DeadlineStream(response.stream, deadline)
    return response.read()


async def _aread_with_deadline(response: httpx.Response, deadline: float) -> bytes:
    """Reads the whole body. A chunk that arrives after the deadline raises ``_DeadlinePassed``, and a read that
    stalls raises ``TimeoutError`` when the deadline passes."""
    if isinstance(response.stream, httpx.AsyncByteStream):
        response.stream = _AsyncDeadlineStream(response.stream, deadline)
    with anyio.fail_after(max(deadline - _monotonic(), 0.0)):
        return await response.aread()


@dataclass(frozen=True)
class _Prepared:
    """A request built and validated once, sent by every attempt."""

    method: str
    url: httpx.URL
    # Left out of the repr: the headers carry the API key, and the body may carry personal data.
    headers: list[tuple[bytes, bytes]] = field(repr=False)
    content: bytes | None = field(repr=False)
    kind: ResponseKind
    retry: RetryPolicy
    timeout: float
    max_retries: int
    stream_idle_timeout: float | None
    conditional: bool
    deadline: float


class _BaseClient:
    """Options, validation, request building, error classification and retry decisions shared by both clients."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        default_headers: Mapping[str, str] | None,
        max_retries: int | None,
        timeout: float | None,
        stream_idle_timeout: float | None,
        on_debug: Callable[[DebugEvent], None] | None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self._base_url = _check_base_url(base_url)
        self._max_retries = DEFAULT_MAX_RETRIES if max_retries is None else _check_max_retries(max_retries)
        self._timeout = DEFAULT_TIMEOUT if timeout is None else _check_timeout(timeout)
        self._stream_idle_timeout = None if stream_idle_timeout is None else _check_idle_timeout(stream_idle_timeout)
        self._default_headers: dict[str, object] = {} if default_headers is None else dict(default_headers)
        self._on_debug = on_debug
        self._closed = False

    @property
    def api_key(self) -> str:
        """The API key sent as ``Authorization: Bearer <api_key>``."""
        return self._api_key

    @property
    def base_url(self) -> str:
        """Base URL of the API, without a trailing ``/``."""
        return self._base_url

    @property
    def max_retries(self) -> int:
        """How many times a request is sent again after a transient error."""
        return self._max_retries

    @property
    def timeout(self) -> float:
        """Timeout of a request in seconds."""
        return self._timeout

    @property
    def stream_idle_timeout(self) -> float | None:
        """Seconds without data after which a stream ends, or ``None`` when it is off."""
        return self._stream_idle_timeout

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self._base_url!r})"

    # --- building ------------------------------------------------------------------------------------------------

    def _prepare(
        self,
        method: Literal["GET", "POST"],
        path: str | None,
        *,
        url: str | None,
        query: Mapping[str, str | None] | None,
        body: object,
        auth: bool,
        if_none_match: str | None,
        retry: RetryPolicy,
        kind: ResponseKind,
        options: RequestOptions | None,
    ) -> _Prepared:
        """Steps 1 to 4 of a request: closed check, effective values, building and validation, deadline."""
        if self._closed:
            raise RuntimeError(_CLOSED_MESSAGE)
        opts: Mapping[str, object] = {} if options is None else options
        given_timeout = opts.get("timeout")
        given_retries = opts.get("max_retries")
        given_idle = opts.get("stream_idle_timeout")
        timeout = self._timeout if given_timeout is None else _check_timeout(given_timeout)
        max_retries = self._max_retries if given_retries is None else _check_max_retries(given_retries)
        idle = self._stream_idle_timeout if given_idle is None else _check_idle_timeout(given_idle)

        target = self._build_url(path, url, query)
        content = (
            None
            if body is None
            else json.dumps(body, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        extra = opts.get("extra_headers")
        headers = self._build_headers(
            kind=kind,
            extra_headers=None if extra is None else _as_header_source(extra),
            if_none_match=if_none_match,
            auth=auth,
            json_body=content is not None,
        )
        encoded = _encode_headers(headers)
        conditional = if_none_match is not None or (method == "GET" and "if-none-match" in headers)
        return _Prepared(
            method=method,
            url=target,
            headers=encoded,
            content=content,
            kind=kind,
            retry=retry,
            timeout=timeout,
            max_retries=max_retries,
            stream_idle_timeout=idle,
            conditional=conditional,
            deadline=_monotonic() + timeout,
        )

    def _build_url(self, path: str | None, url: str | None, query: Mapping[str, str | None] | None) -> httpx.URL:
        if url is not None:
            base, message = url, _URL_MESSAGE
        elif path is not None:
            base, message = f"{self._base_url}{path}", _BASE_URL_MESSAGE
        else:
            raise TypeError("A request needs a path or a url.")
        try:
            target = httpx.URL(base)
        except httpx.InvalidURL:
            raise ValueError(message) from None
        if target.scheme not in ("http", "https") or not target.host:
            raise ValueError(message)
        for name, value in (query or {}).items():
            if value is not None:
                target = target.copy_set_param(name, value)
        return target

    def _build_headers(
        self,
        *,
        kind: ResponseKind,
        extra_headers: Mapping[str, object] | None,
        if_none_match: str | None,
        auth: bool,
        json_body: bool,
    ) -> _HeaderMap:
        headers: _HeaderMap = {}
        _set_header(headers, "Accept", "application/json" if kind == "json" else "*/*")
        _set_header(headers, "User-Agent", USER_AGENT)
        _merge_headers(headers, self._default_headers)
        _merge_headers(headers, extra_headers)
        if if_none_match is not None:
            _set_header(headers, "If-None-Match", if_none_match)
        headers.pop("authorization", None)
        headers.pop("content-type", None)
        if auth:
            _set_header(headers, "Authorization", f"Bearer {self._api_key}")
        if json_body:
            _set_header(headers, "Content-Type", "application/json")
        return headers

    def _attempt_request(self, prepared: _Prepared) -> httpx.Request:
        """The request of one attempt, with the time left as its timeout. Raises when no time is left."""
        remaining = prepared.deadline - _monotonic()
        if remaining <= 0:
            raise self._timeout_error(prepared)
        timeout = httpx.Timeout(remaining, connect=min(remaining, CONNECT_TIMEOUT))
        return httpx.Request(
            prepared.method,
            prepared.url,
            headers=prepared.headers,
            content=prepared.content,
            extensions={"timeout": timeout.as_dict()},
        )

    # --- classification --------------------------------------------------------------------------------------------

    def _timeout_error(self, prepared: _Prepared) -> APITimeoutError:
        return APITimeoutError(0, f"The request timed out after {prepared.timeout:g} seconds.")

    def _past_deadline(self, prepared: _Prepared) -> bool:
        return _monotonic() >= prepared.deadline

    def _send_failure(
        self, prepared: _Prepared, error: Exception, attempt: int
    ) -> tuple[APIConnectionError, float | None, Exception | None]:
        """The error for an exception raised while sending, the wait before a new attempt (``None``: raise), and
        the exception to chain the error to (``None``: chain nothing)."""
        if isinstance(error, httpx.LocalProtocolError):
            # The request was refused before it was sent. The original message can quote a header value (the
            # API key included), so it is neither repeated nor chained. Sending again cannot succeed.
            return APIConnectionError(0, _HEADER_PROTOCOL_MESSAGE), None, None
        if self._past_deadline(prepared) or isinstance(error, (httpx.ReadTimeout, httpx.WriteTimeout)):
            return self._timeout_error(prepared), None, error
        failure = APIConnectionError(0, f"Network error: {_describe(error)}")
        retryable = prepared.retry.connection_errors == "all" or connection_never_opened(error)
        if not retryable or attempt >= prepared.max_retries:
            return failure, None, error
        wait = self._retry_wait(None, attempt, prepared.deadline)
        if wait is not None:
            self._debug_retry(attempt, 0, wait)
        return failure, wait, error

    def _status_wait(self, prepared: _Prepared, error: DevupAPIError, attempt: int) -> float | None:
        """The wait before a new attempt after an error status, or ``None`` when the error is raised."""
        if error.status not in prepared.retry.statuses or attempt >= prepared.max_retries:
            return None
        wait = self._retry_wait(error.retry_after_ms, attempt, prepared.deadline)
        if wait is not None:
            self._debug_retry(attempt, error.status, wait)
        return wait

    def _body_error(self, prepared: _Prepared, response: httpx.Response, error: Exception) -> APIConnectionError:
        """The error for a 2xx body that could not be read. It is never retried."""
        if self._past_deadline(prepared):
            return self._timeout_error(prepared)
        return APIConnectionError(0, f"Network error: {_describe(error)}", headers=response.headers)

    def _is_not_modified(self, prepared: _Prepared, response: httpx.Response) -> bool:
        return prepared.conditional and response.status_code == 304

    def _parse(self, prepared: _Prepared, response: httpx.Response, content: bytes) -> APIResponse[Any]:
        text = _decode(content)
        if prepared.kind == "text":
            return make_api_response(text, response)
        try:
            data: Any = json.loads(text)
        except (ValueError, RecursionError) as error:
            raise DevupAPIError(
                response.status_code, "The response body is not valid JSON.", headers=response.headers
            ) from error
        return make_api_response(data, response)

    def _retry_wait(self, server_wait_ms: float | None, attempt: int, deadline: float) -> float | None:
        """The wait in seconds before the next attempt, or ``None`` when it does not fit in the time left.

        A wait the API asks for is taken as is (no cap but the deadline); otherwise the wait is random, up to
        ``min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * 2 ** attempt)``.
        """
        if server_wait_ms is not None:
            wait = server_wait_ms / 1000
        else:
            exponent = min(attempt, _MAX_BACKOFF_EXPONENT)
            wait = _random() * min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * 2**exponent)
        if wait > deadline - _monotonic():
            return None
        return wait

    def _debug_retry(self, attempt: int, status: int, wait: float) -> None:
        detail: dict[str, object] = {"attempt": attempt + 1, "status": status, "wait_ms": round(wait * 1000)}
        self._debug(DebugEvent(type="retry", detail=detail))

    def _debug(self, event: DebugEvent) -> None:
        """Calls the ``on_debug`` hook. A hook that raises never changes the outcome of the request."""
        hook = self._on_debug
        if hook is None:
            return
        try:
            hook(event)
        except Exception:
            # Deliberately not propagated: on_debug is a diagnostic hook, and its errors must not turn a
            # successful request (or a retry) into a failure.
            return


def _as_header_source(value: object) -> Mapping[str, object] | None:
    """Per-call ``extra_headers`` as a mapping; anything else gives no headers."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


class DevupAI(_BaseClient):
    """The synchronous DEVUP AI API client.

    The key is ``api_key``, else the ``DEVUP_API_KEY`` environment variable, else (only when that variable is
    absent) ``DEVUPAI_API_KEY``. ``timeout`` is in seconds and covers the whole call, retries included. A request is
    sent again up to ``max_retries`` times after a 408, 429, 502, 503 or 504 answer or a connection error, except a
    billed request: it is sent again only after a 429 or 503 answer, or when the connection was never opened.

    Use it as a context manager, or call :meth:`close`, to release its connections. An ``http_client`` passed in
    is used as is and never closed by the client.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        stream_idle_timeout: float | None = None,
        http_client: httpx.Client | None = None,
        on_debug: Callable[[DebugEvent], None] | None = None,
    ) -> None:
        """Creates the client. Raises ``TypeError`` or ``ValueError`` for an option that is not valid."""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            max_retries=max_retries,
            timeout=timeout,
            stream_idle_timeout=stream_idle_timeout,
            on_debug=on_debug,
        )
        given = _check_http_client(http_client, httpx.Client, "http_client must be an httpx.Client.")
        self._owns_http = given is None
        self._http = httpx.Client(follow_redirects=False, trust_env=True, timeout=None) if given is None else given

    def close(self) -> None:
        """Closes the HTTP client this client created. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _request(
        self,
        method: Literal["GET", "POST"],
        path: str | None = None,
        *,
        url: str | None = None,
        query: Mapping[str, str | None] | None = None,
        body: object = None,
        auth: bool = True,
        if_none_match: str | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        kind: ResponseKind = "json",
        options: RequestOptions | None = None,
    ) -> APIResponse[Any]:
        """Sends a request with retries inside one deadline and returns the parsed response."""
        prepared = self._prepare(
            method,
            path,
            url=url,
            query=query,
            body=body,
            auth=auth,
            if_none_match=if_none_match,
            retry=retry,
            kind=kind,
            options=options,
        )
        attempt = 0
        while True:
            request = self._attempt_request(prepared)
            try:
                response = self._http.send(request, stream=True, follow_redirects=False)
            except Exception as error:
                failure, wait, cause = self._send_failure(prepared, error, attempt)
                if wait is None:
                    raise failure from cause
                _sleep(wait)
                attempt += 1
                continue

            if self._is_not_modified(prepared, response):
                if prepared.kind != "raw":
                    response.close()
                return make_api_response(response if prepared.kind == "raw" else None, response)

            if not response.is_success:
                try:
                    text = _decode(_read_with_deadline(response, prepared.deadline))
                except Exception as error:
                    if self._past_deadline(prepared):
                        raise self._timeout_error(prepared) from error
                    # The body could not be read: the message falls back to the HTTP status.
                    text = ""
                finally:
                    response.close()
                api_error = error_from_body(response.status_code, text, response.headers)
                wait = self._status_wait(prepared, api_error, attempt)
                if wait is None:
                    raise api_error
                _sleep(wait)
                attempt += 1
                continue

            if prepared.kind == "raw":
                return make_api_response(response, response)
            try:
                content = _read_with_deadline(response, prepared.deadline)
            except Exception as error:
                raise self._body_error(prepared, response, error) from error
            finally:
                response.close()
            return self._parse(prepared, response, content)


class AsyncDevupAI(_BaseClient):
    """The asynchronous DEVUP AI API client, with the same options as :class:`DevupAI`.

    Use it as an async context manager, or await :meth:`close`, to release its connections. An ``http_client``
    passed in is used as is and never closed by the client.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        stream_idle_timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
        on_debug: Callable[[DebugEvent], None] | None = None,
    ) -> None:
        """Creates the client. Raises ``TypeError`` or ``ValueError`` for an option that is not valid."""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            max_retries=max_retries,
            timeout=timeout,
            stream_idle_timeout=stream_idle_timeout,
            on_debug=on_debug,
        )
        given = _check_http_client(http_client, httpx.AsyncClient, "http_client must be an httpx.AsyncClient.")
        self._owns_http = given is None
        self._http = httpx.AsyncClient(follow_redirects=False, trust_env=True, timeout=None) if given is None else given

    async def close(self) -> None:
        """Closes the HTTP client this client created. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str | None = None,
        *,
        url: str | None = None,
        query: Mapping[str, str | None] | None = None,
        body: object = None,
        auth: bool = True,
        if_none_match: str | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        kind: ResponseKind = "json",
        options: RequestOptions | None = None,
    ) -> APIResponse[Any]:
        """Sends a request with retries inside one deadline and returns the parsed response."""
        prepared = self._prepare(
            method,
            path,
            url=url,
            query=query,
            body=body,
            auth=auth,
            if_none_match=if_none_match,
            retry=retry,
            kind=kind,
            options=options,
        )
        attempt = 0
        while True:
            request = self._attempt_request(prepared)
            try:
                response = await self._http.send(request, stream=True, follow_redirects=False)
            except Exception as error:
                failure, wait, cause = self._send_failure(prepared, error, attempt)
                if wait is None:
                    raise failure from cause
                await _async_sleep(wait)
                attempt += 1
                continue

            if self._is_not_modified(prepared, response):
                if prepared.kind != "raw":
                    await response.aclose()
                return make_api_response(response if prepared.kind == "raw" else None, response)

            if not response.is_success:
                try:
                    text = _decode(await _aread_with_deadline(response, prepared.deadline))
                except TimeoutError as error:
                    raise self._timeout_error(prepared) from error
                except Exception as error:
                    if self._past_deadline(prepared):
                        raise self._timeout_error(prepared) from error
                    # The body could not be read: the message falls back to the HTTP status.
                    text = ""
                finally:
                    await response.aclose()
                api_error = error_from_body(response.status_code, text, response.headers)
                wait = self._status_wait(prepared, api_error, attempt)
                if wait is None:
                    raise api_error
                await _async_sleep(wait)
                attempt += 1
                continue

            if prepared.kind == "raw":
                return make_api_response(response, response)
            try:
                content = await _aread_with_deadline(response, prepared.deadline)
            except TimeoutError as error:
                raise self._timeout_error(prepared) from error
            except Exception as error:
                raise self._body_error(prepared, response, error) from error
            finally:
                await response.aclose()
            return self._parse(prepared, response, content)
