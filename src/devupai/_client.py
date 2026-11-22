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
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Self, TypeAlias, TypedDict, TypeVar, cast

import anyio
import httpx

from devupai._errors import APIConnectionError, APITimeoutError, DevupAPIError, error_from_body
from devupai._response import APIResponse, make_api_response
from devupai._version import __version__
from devupai.resources import (
    AsyncAudio,
    AsyncBalance,
    AsyncChat,
    AsyncEmbeddings,
    AsyncHealth,
    AsyncImages,
    AsyncMe,
    AsyncModels,
    AsyncNativeInference,
    AsyncRerank,
    AsyncResponses,
    AsyncVideo,
    Audio,
    Balance,
    Chat,
    Embeddings,
    Health,
    Images,
    Me,
    Models,
    NativeInference,
    Rerank,
    Responses,
    Video,
)
from devupai.webhooks import Webhooks

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
    "MultipartParts",
    "MultipartValue",
    "RequestOptions",
    "ResponseKind",
    "RetryPolicy",
    "async_request",
    "connection_never_opened",
    "effective_idle_timeout",
    "sync_request",
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
_BASE_URL_CREDENTIALS_MESSAGE = "base_url must not contain a user name or password."
_URL_CREDENTIALS_MESSAGE = "url must not contain a user name or password."
# An RFC 9110 token: the characters allowed in a header name.
_HEADER_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
# Failures of a body read that come from the connection or from the bytes the API sent (a body that cannot be
# decoded, for example a corrupt gzip). Any other exception raised while a body is read is raised unchanged.
_BODY_READ_ERRORS = (httpx.TransportError, httpx.DecodingError)
_BODY_AND_MULTIPART_MESSAGE = "A request takes a JSON body or multipart parts, not both."
_MULTIPART_PART_MESSAGE = (
    "A multipart part must be (name, text) or (name, (filename, content bytes, content type or None))."
)
_MULTIPART_TYPE_MESSAGE = "A multipart part has a content type that is not allowed in an HTTP header."
# A part's name and filename go in a quoted Content-Disposition parameter. As in the WHATWG multipart encoding that
# Node's FormData uses, only CR, LF and '"' are percent-encoded, so a value can never end the parameter or the header
# line; a backslash and every other character are sent as they are.
_FORM_PARAM_REPLACEMENTS: dict[str, str] = {"\r": "%0D", "\n": "%0A", '"': "%22"}
_FORM_PARAM_CHARS = re.compile("|".join(re.escape(char) for char in _FORM_PARAM_REPLACEMENTS))
_OCTET_STREAM = "application/octet-stream"

ResponseKind = Literal["json", "text", "raw", "stream"]
"""How a successful body is read: ``"json"`` and ``"text"`` read it whole inside the deadline; ``"raw"`` and
``"stream"`` return the ``httpx.Response`` with its body unread, and the deadline covers only the time until the
headers arrive."""

MultipartValue: TypeAlias = str | tuple[str, bytes, str | None]
"""One multipart part value: a text, or ``(filename, content, content_type)``; a ``None`` content type is sent as
``application/octet-stream``."""
MultipartParts: TypeAlias = Sequence[tuple[str, MultipartValue]]
"""The parts of a multipart body as ``(name, value)`` pairs, sent in this order."""

# The kinds whose successful response is returned with its body unread.
_UNREAD_KINDS: frozenset[str] = frozenset({"raw", "stream"})
_ACCEPT: dict[str, str] = {"json": "application/json", "stream": "text/event-stream", "text": "*/*", "raw": "*/*"}


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


def _retry_refused(headers: Mapping[str, str]) -> bool:
    """True when the answer carries ``x-should-retry: false``: the request is not sent again.

    The header name is matched without regard to case; the value is trimmed and compared, lower-cased, to ``false``.
    Any other value, or no header, leaves the retry rules unchanged. Only ``false`` is honoured:
    ``x-should-retry: true`` never widens them, because a retry it allowed could send a billed request again (a billed
    ``POST`` is still sent again only after 429, 503 or a connection that was never opened).
    """
    return any(name.lower() == "x-should-retry" and value.strip().lower() == "false" for name, value in headers.items())


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


def _has_credentials(url: httpx.URL) -> bool:
    """True when ``url`` carries a user name or a password (``user:password@host``)."""
    return bool(url.userinfo)


def _check_base_url(value: object) -> str:
    """``value`` or the default base URL, without a trailing ``/``. A user name or password in it is refused."""
    if value is not None and not isinstance(value, str):
        raise TypeError("base_url must be a string.")
    base_url = (value or DEFAULT_BASE_URL).rstrip("/")
    try:
        parsed = httpx.URL(base_url)
    except httpx.InvalidURL:
        # Refused here, not at the first request, so repr() and base_url never show it: it can carry a password.
        raise ValueError(_BASE_URL_MESSAGE) from None
    if _has_credentials(parsed):
        # The message never repeats the URL, so the credentials in it are not shown.
        raise ValueError(_BASE_URL_CREDENTIALS_MESSAGE)
    return base_url


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


def _form_param(name: str, value: str) -> bytes:
    """``name="value"`` for a Content-Disposition header, with the value escaped."""
    escaped = _FORM_PARAM_CHARS.sub(lambda match: _FORM_PARAM_REPLACEMENTS[match.group(0)], value)
    return f'{name}="{escaped}"'.encode()


def _check_part(part: object) -> tuple[str, str | tuple[str, bytes, str]]:
    """One ``(name, value)`` part, checked, with a file part's content type filled in."""
    if not isinstance(part, tuple) or len(part) != 2:  # pyright: ignore[reportUnknownArgumentType]
        raise TypeError(_MULTIPART_PART_MESSAGE)
    name, value = cast("tuple[object, object]", part)
    if not isinstance(name, str):
        raise TypeError(_MULTIPART_PART_MESSAGE)
    if isinstance(value, str):
        return name, value
    if not isinstance(value, tuple) or len(value) != 3:  # pyright: ignore[reportUnknownArgumentType]
        raise TypeError(_MULTIPART_PART_MESSAGE)
    filename, content, content_type = cast("tuple[object, object, object]", value)
    if not isinstance(filename, str) or not isinstance(content, bytes):
        raise TypeError(_MULTIPART_PART_MESSAGE)
    if content_type is None:
        content_type = _OCTET_STREAM
    if not isinstance(content_type, str):
        raise TypeError(_MULTIPART_PART_MESSAGE)
    content_type = content_type.strip(_HEADER_WHITESPACE)
    if not content_type or _FORBIDDEN_HEADER_CHARS.search(content_type) or not content_type.isascii():
        raise ValueError(_MULTIPART_TYPE_MESSAGE)
    return name, (filename, content, content_type)


def _encode_multipart(parts: MultipartParts) -> tuple[bytes, str]:
    """The ``multipart/form-data`` body of ``parts``, in their order, and its ``Content-Type`` value.

    A text part has only a Content-Disposition header; a file part also has its filename and a Content-Type. The
    body is built once, so every attempt of the request sends the same bytes.
    """
    rendered: list[tuple[bytes, bytes]] = []
    for part in parts:
        name, value = _check_part(part)
        if isinstance(value, str):
            head = b"Content-Disposition: form-data; " + _form_param("name", name)
            rendered.append((head + b"\r\n\r\n", value.encode("utf-8")))
            continue
        filename, content, content_type = value
        head = b"Content-Disposition: form-data; " + _form_param("name", name)
        if filename:
            head += b"; " + _form_param("filename", filename)
        head += b"\r\nContent-Type: " + content_type.encode("ascii")
        rendered.append((head + b"\r\n\r\n", content))
    boundary = os.urandom(16).hex().encode("ascii")
    # A boundary must not appear inside a part; with 128 random bits this loop almost never runs twice.
    while any(boundary in head or boundary in data for head, data in rendered):
        boundary = os.urandom(16).hex().encode("ascii")
    chunks: list[bytes] = []
    for head, data in rendered:
        chunks.extend((b"--", boundary, b"\r\n", head, data, b"\r\n"))
    chunks.extend((b"--", boundary, b"--\r\n"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary.decode('ascii')}"


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

    def _effective_idle_timeout(self, options: RequestOptions | None) -> float | None:
        """The idle timeout of one call: the per-call option when given (``0`` turns it off), else the client's."""
        given = None if options is None else options.get("stream_idle_timeout")
        return self._stream_idle_timeout if given is None else _check_idle_timeout(given)

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
        multipart: MultipartParts | None = None,
    ) -> _Prepared:
        """Steps 1 to 4 of a request: closed check, effective values, building and validation, deadline."""
        if self._closed:
            raise RuntimeError(_CLOSED_MESSAGE)
        if body is not None and multipart is not None:
            raise ValueError(_BODY_AND_MULTIPART_MESSAGE)
        opts: Mapping[str, object] = {} if options is None else options
        given_timeout = opts.get("timeout")
        given_retries = opts.get("max_retries")
        timeout = self._timeout if given_timeout is None else _check_timeout(given_timeout)
        max_retries = self._max_retries if given_retries is None else _check_max_retries(given_retries)
        idle = self._effective_idle_timeout(options)

        target = self._build_url(path, url, query)
        content: bytes | None = None
        content_type: str | None = None
        if multipart is not None:
            content, content_type = _encode_multipart(multipart)
        elif body is not None:
            content = json.dumps(body, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            content_type = "application/json"
        extra = opts.get("extra_headers")
        headers = self._build_headers(
            kind=kind,
            extra_headers=None if extra is None else _as_header_source(extra),
            if_none_match=if_none_match,
            auth=auth,
            content_type=content_type,
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
            base, message, credentials_message = url, _URL_MESSAGE, _URL_CREDENTIALS_MESSAGE
        elif path is not None:
            base, message, credentials_message = (
                f"{self._base_url}{path}",
                _BASE_URL_MESSAGE,
                _BASE_URL_CREDENTIALS_MESSAGE,
            )
        else:
            raise TypeError("A request needs a path or a url.")
        try:
            target = httpx.URL(base)
        except httpx.InvalidURL:
            raise ValueError(message) from None
        if _has_credentials(target):
            # httpx would turn them into a Basic Authorization header. The message never repeats the URL.
            raise ValueError(credentials_message)
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
        content_type: str | None,
    ) -> _HeaderMap:
        headers: _HeaderMap = {}
        _set_header(headers, "Accept", _ACCEPT[kind])
        _set_header(headers, "User-Agent", USER_AGENT)
        _merge_headers(headers, self._default_headers)
        _merge_headers(headers, extra_headers)
        if if_none_match is not None:
            _set_header(headers, "If-None-Match", if_none_match)
        headers.pop("authorization", None)
        headers.pop("content-type", None)
        if auth:
            _set_header(headers, "Authorization", f"Bearer {self._api_key}")
        if content_type is not None:
            _set_header(headers, "Content-Type", content_type)
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
        self, prepared: _Prepared, error: httpx.TransportError, attempt: int
    ) -> tuple[APIConnectionError, float | None, Exception | None]:
        """The error for a transport error raised while sending, the wait before a new attempt (``None``: raise),
        and the exception to chain the error to (``None``: chain nothing).

        Only ``httpx.TransportError`` comes here. Any other exception raised by ``send`` (an event hook of a caller
        ``http_client``, an encoding error, a caller's own error) is raised unchanged and is never retried.
        """
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
        # Checked before the status rules: an answer that asks for no new attempt is never sent again, whatever its
        # status, the method or the retry budget (sync and async clients, plain, raw and stream calls all come here).
        if _retry_refused(error.headers):
            return None
        if error.status not in prepared.retry.statuses or attempt >= prepared.max_retries:
            return None
        wait = self._retry_wait(error.retry_after_ms, attempt, prepared.deadline)
        if wait is not None:
            self._debug_retry(attempt, error.status, wait)
        return wait

    def _body_error(self, prepared: _Prepared, response: httpx.Response, error: Exception) -> APIConnectionError:
        """The error for a 2xx body that could not be read (a transport error, or the deadline passed). It is never
        retried."""
        if self._past_deadline(prepared):
            return self._timeout_error(prepared)
        return APIConnectionError(0, f"Network error: {_describe(error)}", headers=response.headers)

    def _is_not_modified(self, prepared: _Prepared, response: httpx.Response) -> bool:
        return prepared.conditional and response.status_code == 304

    def _open_stream(self, prepared: _Prepared, response: httpx.Response) -> None:
        """After the headers of a ``"stream"`` response: from now on each read waits up to the idle timeout, or
        without limit when it is off; the deadline of the call no longer applies.

        The read timeout is the one the HTTP connection looks up when the body is read, so both clients apply it
        there; on the async client the stream also bounds each read with ``anyio.move_on_after``, which covers a
        transport that does not look the timeout up.
        """
        timeouts = response.request.extensions.get("timeout")
        if isinstance(timeouts, dict):
            cast("dict[str, float | None]", timeouts)["read"] = prepared.stream_idle_timeout

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

        self.chat: Chat = Chat(self)
        """Chat completions: ``chat.completions.create``."""
        self.responses: Responses = Responses(self)
        """The Responses API: ``responses.create``."""
        self.models: Models = Models(self)
        """The model catalog: ``models.list`` and ``models.retrieve``."""
        self.images: Images = Images(self)
        """Image generation and editing."""
        self.embeddings: Embeddings = Embeddings(self)
        """Embeddings: ``embeddings.create``."""
        self.rerank: Rerank = Rerank(self)
        """Reranking: ``rerank.create``."""
        self.audio: Audio = Audio(self)
        """Speech and transcriptions."""
        self.video: Video = Video(self)
        """Video generations and edits."""
        self.inference: NativeInference = NativeInference(self)
        """Native inference: ``inference.run``."""
        self.balance: Balance = Balance(self)
        """The account balance: ``balance.retrieve``."""
        self.me: Me = Me(self)
        """The account profile: ``me.retrieve``."""
        self.health: Health = Health(self)
        """The API health check: ``health.check``."""
        self.webhooks: Webhooks = Webhooks()
        """Webhook signature checks; they run locally and send nothing."""

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
        multipart: MultipartParts | None = None,
        auth: bool = True,
        if_none_match: str | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        kind: ResponseKind = "json",
        options: RequestOptions | None = None,
    ) -> APIResponse[Any]:
        """Sends a request with retries inside one deadline and returns the parsed response.

        ``body`` is sent as JSON, ``multipart`` as ``multipart/form-data`` (in the given order); passing both raises
        ``ValueError``. For ``kind="raw"`` and ``kind="stream"`` the data is the ``httpx.Response`` with its body
        unread, and the caller closes it. Nothing is sent again once the headers of a 2xx answer have arrived.
        """
        prepared = self._prepare(
            method,
            path,
            url=url,
            query=query,
            body=body,
            multipart=multipart,
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
                # auth=None: the auth of a caller http_client is never applied; the key goes in the headers.
                response = self._http.send(request, stream=True, auth=None, follow_redirects=False)
            except httpx.TransportError as error:
                failure, wait, cause = self._send_failure(prepared, error, attempt)
                if wait is None:
                    raise failure from cause
                _sleep(wait)
                attempt += 1
                continue

            if self._is_not_modified(prepared, response):
                unread = prepared.kind in _UNREAD_KINDS
                if not unread:
                    response.close()
                return make_api_response(response if unread else None, response)

            if not response.is_success:
                try:
                    text = _decode(_read_with_deadline(response, prepared.deadline))
                except _DeadlinePassed as error:
                    raise self._timeout_error(prepared) from error
                except _BODY_READ_ERRORS as error:
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

            if prepared.kind == "stream":
                self._open_stream(prepared, response)
                return make_api_response(response, response)
            if prepared.kind == "raw":
                return make_api_response(response, response)
            try:
                content = _read_with_deadline(response, prepared.deadline)
            except (_DeadlinePassed, *_BODY_READ_ERRORS) as error:
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

        self.chat: AsyncChat = AsyncChat(self)
        """Chat completions: ``chat.completions.create``."""
        self.responses: AsyncResponses = AsyncResponses(self)
        """The Responses API: ``responses.create``."""
        self.models: AsyncModels = AsyncModels(self)
        """The model catalog: ``models.list`` and ``models.retrieve``."""
        self.images: AsyncImages = AsyncImages(self)
        """Image generation and editing."""
        self.embeddings: AsyncEmbeddings = AsyncEmbeddings(self)
        """Embeddings: ``embeddings.create``."""
        self.rerank: AsyncRerank = AsyncRerank(self)
        """Reranking: ``rerank.create``."""
        self.audio: AsyncAudio = AsyncAudio(self)
        """Speech and transcriptions."""
        self.video: AsyncVideo = AsyncVideo(self)
        """Video generations and edits."""
        self.inference: AsyncNativeInference = AsyncNativeInference(self)
        """Native inference: ``inference.run``."""
        self.balance: AsyncBalance = AsyncBalance(self)
        """The account balance: ``balance.retrieve``."""
        self.me: AsyncMe = AsyncMe(self)
        """The account profile: ``me.retrieve``."""
        self.health: AsyncHealth = AsyncHealth(self)
        """The API health check: ``health.check``."""
        self.webhooks: Webhooks = Webhooks()
        """Webhook signature checks (synchronous on both clients); they run locally and send nothing."""

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
        multipart: MultipartParts | None = None,
        auth: bool = True,
        if_none_match: str | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        kind: ResponseKind = "json",
        options: RequestOptions | None = None,
    ) -> APIResponse[Any]:
        """Sends a request with retries inside one deadline and returns the parsed response, as
        :meth:`DevupAI._request` does.

        For ``kind="stream"`` the async stream applies the idle timeout itself: each read is bounded with
        ``anyio.move_on_after`` (see :func:`effective_idle_timeout`).
        """
        prepared = self._prepare(
            method,
            path,
            url=url,
            query=query,
            body=body,
            multipart=multipart,
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
                # auth=None: the auth of a caller http_client is never applied; the key goes in the headers.
                response = await self._http.send(request, stream=True, auth=None, follow_redirects=False)
            except httpx.TransportError as error:
                failure, wait, cause = self._send_failure(prepared, error, attempt)
                if wait is None:
                    raise failure from cause
                await _async_sleep(wait)
                attempt += 1
                continue

            if self._is_not_modified(prepared, response):
                unread = prepared.kind in _UNREAD_KINDS
                if not unread:
                    await response.aclose()
                return make_api_response(response if unread else None, response)

            if not response.is_success:
                try:
                    text = _decode(await _aread_with_deadline(response, prepared.deadline))
                except (TimeoutError, _DeadlinePassed) as error:
                    raise self._timeout_error(prepared) from error
                except _BODY_READ_ERRORS as error:
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

            if prepared.kind == "stream":
                self._open_stream(prepared, response)
                return make_api_response(response, response)
            if prepared.kind == "raw":
                return make_api_response(response, response)
            try:
                content = await _aread_with_deadline(response, prepared.deadline)
            except TimeoutError as error:
                raise self._timeout_error(prepared) from error
            except (_DeadlinePassed, *_BODY_READ_ERRORS) as error:
                raise self._body_error(prepared, response, error) from error
            finally:
                await response.aclose()
            return self._parse(prepared, response, content)


def sync_request(client: DevupAI, *args: Any, **kwargs: Any) -> APIResponse[Any]:
    """Sends a request through ``client`` with the parameters of ``DevupAI._request``; used by the resources.

    The request method is looked up on the instance at each call, so a method set on the instance is the one used.
    """
    # This module defines the class; these two functions are the one place outside it that calls _request, so the
    # resources never touch a protected member.
    return client._request(*args, **kwargs)  # pyright: ignore[reportPrivateUsage]


async def async_request(client: AsyncDevupAI, *args: Any, **kwargs: Any) -> APIResponse[Any]:
    """The async twin of :func:`sync_request`, with the parameters of ``AsyncDevupAI._request``."""
    return await client._request(*args, **kwargs)  # pyright: ignore[reportPrivateUsage]


def effective_idle_timeout(client: DevupAI | AsyncDevupAI, options: RequestOptions | None) -> float | None:
    """The stream idle timeout of one call, in seconds, or ``None`` when it is off; used by the resources.

    The ``stream_idle_timeout`` of ``options`` wins when it is given (``0`` turns it off); otherwise the client's
    ``stream_idle_timeout`` applies. A value that is not valid raises ``TypeError`` or ``ValueError``. It is the
    value ``_request(..., kind="stream")`` applies to the reads of the response.
    """
    return client._effective_idle_timeout(options)  # pyright: ignore[reportPrivateUsage]
