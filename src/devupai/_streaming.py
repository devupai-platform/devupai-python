"""Server-sent event streams: :class:`Stream` for :class:`~devupai.DevupAI`, :class:`AsyncStream` for
:class:`~devupai.AsyncDevupAI`.

``chat.completions.create(stream=True)`` and ``responses.create(stream=True)`` return one of them. Iterate it once
(``for`` / ``async for``); each item is one parsed event. Leaving a ``with`` / ``async with`` block, or calling
``close()``, closes the response.
"""

from __future__ import annotations

import codecs
import json
import re
import warnings
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Self, TypeVar, cast

import anyio
import httpx

from devupai._errors import APIConnectionError, APITimeoutError, DevupAPIError, make_api_error, parse_warnings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from devupai._client import DebugEvent

__all__ = [
    "DEFAULT_STREAM_ERROR_MESSAGE",
    "ErrorFrame",
    "ServerSentEvent",
    "SSEParser",
    "Stream",
    "AsyncStream",
    "debug_hook",
    "stream_requested",
]

T = TypeVar("T")

DEFAULT_STREAM_ERROR_MESSAGE = "The request could not be completed."
"""The message of a stream error event that carries none."""

ErrorFrame = Callable[[object, str | None], tuple[str, str | None, str | None, str | None] | None]
"""Reads an error event: ``(frame, SSE event name)`` -> ``(message, code, type, param)``, or ``None`` for a regular
event."""

_ITERATED_MESSAGE = "A stream can be iterated only once."
_EXTRA_BODY_STREAM_MESSAGE = "Pass stream=True as an argument, not in extra_body."
_READ_ERRORS = (httpx.TransportError, httpx.DecodingError)
_LINE_END = re.compile(r"\r\n?|\n")
_DROPPED_MESSAGE = (
    "An AsyncStream was dropped without being closed. Use `async with`, or `await stream.close()`, to close its "
    "response."
)


@dataclass(frozen=True)
class ServerSentEvent:
    """One server-sent event. ``event`` is ``None`` when the event has no ``event:`` field."""

    event: str | None
    data: str


class SSEParser:
    """Incremental parser for ``text/event-stream`` content, following the WHATWG event-stream interpretation rules.

    CR, LF and CRLF end a line (a CRLF split across two pieces is one line end), ``:`` starts a comment, one
    optional space after the ``:`` is removed, the ``data`` lines of an event are joined with ``\\n``, ``id`` and
    ``retry`` are ignored, and a blank line dispatches the event.
    """

    def __init__(self) -> None:
        # The pieces of the line that is not ended yet, joined once its line end arrives.
        self._partial: list[str] = []
        self._pending_cr = False
        self._event_type: str | None = None
        self._data = ""
        self._has_data = False

    def feed(self, text: str) -> list[ServerSentEvent]:
        """Adds decoded text and returns the events completed by it."""
        events: list[ServerSentEvent] = []
        if self._pending_cr:
            self._pending_cr = False
            if text.startswith("\n"):
                text = text[1:]
        position = 0
        length = len(text)
        while position < length:
            # One search per line end, so each character is scanned once (linear in the size of the text).
            match = _LINE_END.search(text, position)
            if match is None:
                self._partial.append(text[position:])
                break
            piece = text[position : match.start()]
            if self._partial:
                self._partial.append(piece)
                line = "".join(self._partial)
                self._partial.clear()
            else:
                line = piece
            position = match.end()
            if match.group() == "\r" and position == length:
                # A CR at the end of the text: an LF at the start of the next text belongs to the same line end.
                self._pending_cr = True
            event = self._process_line(line)
            if event is not None:
                events.append(event)
        return events

    def end(self) -> str | None:
        """Ends the input. Returns the data of an event that no blank line ended (it is not dispatched), or
        ``None``."""
        if self._partial:
            line = "".join(self._partial)
            self._partial.clear()
            self._process_line(line)
        pending = self._data if self._has_data else None
        self._data = ""
        self._has_data = False
        self._event_type = None
        return pending

    def _process_line(self, line: str) -> ServerSentEvent | None:
        if line == "":
            if not self._has_data:
                self._event_type = None
                return None
            event = ServerSentEvent(event=self._event_type, data=self._data)
            self._event_type = None
            self._data = ""
            self._has_data = False
            return event
        if line.startswith(":"):
            return None
        field, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event_type = value
        elif field == "data":
            self._data = f"{self._data}\n{value}" if self._has_data else value
            self._has_data = True
        return None


def stream_requested(params: Mapping[str, object], body: Mapping[str, object]) -> bool:
    """Whether a call asks for a stream: its ``stream`` argument is ``True``.

    Raises ``ValueError`` before anything is sent when the merged body's ``stream`` differs from the argument, that
    is when ``stream`` was set through ``extra_body``.
    """
    wanted = params.get("stream") is True
    if (body.get("stream") is True) != wanted:
        raise ValueError(_EXTRA_BODY_STREAM_MESSAGE)
    return wanted


def debug_hook(client: object) -> Callable[[DebugEvent], None] | None:
    """The ``on_debug`` hook of ``client``, or ``None`` when it has none."""
    hook: object = getattr(client, "_on_debug", None)
    if hook is None or not callable(hook):
        return None
    return cast("Callable[[DebugEvent], None]", hook)


_DONE = object()
_SKIP = object()
_END = object()


class _Frames(Generic[T]):
    """The decoding shared by :class:`Stream` and :class:`AsyncStream`: bytes -> SSE events -> items."""

    def __init__(
        self,
        response: httpx.Response,
        parse: Callable[[object], T],
        error_frame: ErrorFrame,
        on_debug: Callable[[DebugEvent], None] | None,
        idle_timeout: float | None,
    ) -> None:
        self.response = response
        self.parse = parse
        self.error_frame = error_frame
        self.on_debug = on_debug
        self.idle_timeout = idle_timeout if idle_timeout is not None and idle_timeout > 0 else None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.parser = SSEParser()
        self.events: deque[ServerSentEvent] = deque()
        self.body_done = False

    def push(self, data: bytes, *, final: bool = False) -> None:
        """Decodes ``data`` and keeps the events it completes; ``final`` marks the end of the body."""
        self.events.extend(self.parser.feed(self.decoder.decode(data, final=final)))
        if final:
            self.body_done = True

    def handle(self, event: ServerSentEvent) -> object:
        """The item of ``event``, or ``_DONE`` / ``_SKIP``. Raises the error of an error event."""
        if event.data.strip() == "[DONE]":
            return _DONE
        try:
            frame: object = json.loads(event.data)
        except (ValueError, RecursionError):
            # RecursionError: a frame nested too deeply to decode is skipped like any other frame that is not JSON.
            self.debug_malformed(event.data)
            return _SKIP
        error = self.error_frame(frame, event.event)
        if error is not None:
            message, code, error_type, param = error
            raise make_api_error(
                self.response.status_code,
                message,
                code=code,
                error_type=error_type,
                param=param,
                headers=self.response.headers,
            )
        return self.parse(frame)

    def finish(self) -> None:
        """Ends the parsing at the end of the body; an event no blank line ended is reported as ``malformed_frame``."""
        pending = self.parser.end()
        if pending is not None:
            self.debug_malformed(pending)

    def debug_malformed(self, data: str) -> None:
        """Calls the ``on_debug`` hook. A hook that raises never changes the outcome of the stream."""
        hook = self.on_debug
        if hook is None:
            return
        from devupai._client import DebugEvent

        try:
            hook(DebugEvent(type="malformed_frame", detail={"data": data}))
        except Exception:
            # Deliberately not propagated: on_debug is a diagnostic hook, and its errors must not end the stream
            # or replace its result.
            return

    def read_error(self, error: Exception) -> DevupAPIError:
        """The error raised when reading the body fails: an idle timeout, or a connection error."""
        if self.idle_timeout is not None and isinstance(error, httpx.ReadTimeout):
            return self.idle_error()
        return APIConnectionError(0, f"Network error: {str(error) or 'unknown error'}", headers=self.response.headers)

    def idle_error(self) -> APITimeoutError:
        return APITimeoutError(
            0,
            f"No data was received on the stream for {self.idle_timeout:g} seconds.",
            headers=self.response.headers,
        )


class Stream(Generic[T]):
    """A server-sent event stream of the sync client. Iterate it once with ``for``; each item is one parsed event.

    ``request_id``, ``client_request_id`` and ``warnings`` can be read before the first item. Use it in a ``with``
    block, or call :meth:`close`, to close the response when you stop early (a loop left by ``break`` or by an
    exception, and a stream dropped unread, close it too). An error event raises a
    :class:`~devupai.DevupAPIError` with the stream's status, a read failure raises
    :class:`~devupai.APIConnectionError`, and an idle stream raises :class:`~devupai.APITimeoutError`.
    """

    response: httpx.Response
    """The HTTP response."""
    request_id: str | None
    """Value of the ``x-request-id`` response header, or ``None`` when it is absent."""
    client_request_id: str | None
    """Value of the ``x-client-request-id`` response header, or ``None`` when it is absent."""
    warnings: list[str]
    """Informational notices from the ``X-DEVUP-Warnings`` response header."""

    def __init__(
        self,
        response: httpx.Response,
        *,
        parse: Callable[[object], T],
        error_frame: ErrorFrame,
        on_debug: Callable[[DebugEvent], None] | None,
        idle_timeout: float | None,
    ) -> None:
        """Wraps the unread ``response``; built by the API methods."""
        self.response = response
        self.request_id = response.headers.get("x-request-id")
        self.client_request_id = response.headers.get("x-client-request-id")
        self.warnings = parse_warnings(response.headers)
        self._frames = _Frames(response, parse, error_frame, on_debug, idle_timeout)
        self._chunks: Iterator[bytes] | None = None
        self._iterated = False
        self._closed = False
        self._finished = False

    def __iter__(self) -> Iterator[T]:
        if self._iterated:
            raise RuntimeError(_ITERATED_MESSAGE)
        self._iterated = True
        return self._iterate()

    def _iterate(self) -> Iterator[T]:
        # A generator, so the response is also closed when the loop ends early: on ``break``, or when the caller's
        # own code raises inside the loop, the generator is finalized and its ``finally`` runs.
        try:
            while True:
                outcome = self._next()
                if outcome is _END:
                    return
                yield cast("T", outcome)
        finally:
            self.close()

    def _next(self) -> object:
        """The next item, or ``_END``."""
        frames = self._frames
        while True:
            if self._closed or self._finished:
                return _END
            if frames.events:
                outcome = frames.handle(frames.events.popleft())
                if outcome is _DONE:
                    self._end()
                    return _END
                if outcome is _SKIP:
                    continue
                return outcome
            if frames.body_done:
                frames.finish()
                self._end()
                return _END
            chunk = self._read()
            if self._closed:
                return _END
            if chunk is None:
                frames.push(b"", final=True)
            else:
                frames.push(chunk)

    def _read(self) -> bytes | None:
        """The next piece of the body, or ``None`` at its end."""
        if self._chunks is None:
            self._chunks = self.response.iter_bytes()
        try:
            return next(self._chunks)
        except StopIteration:
            return None
        except _READ_ERRORS as error:
            if self._closed:
                return None
            raise self._frames.read_error(error) from error

    def _end(self) -> None:
        self._finished = True
        self.close()

    def close(self) -> None:
        """Closes the response. Iteration then ends quietly. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        self.response.close()

    def __del__(self) -> None:
        # A stream dropped without being read or closed gives its connection back.
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            # Deliberately not propagated: an exception in __del__ cannot reach the caller, and a response that
            # cannot be closed here is released with its client.
            return

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncStream(Generic[T]):
    """A server-sent event stream of the async client. Iterate it once with ``async for``; each item is one parsed
    event.

    ``request_id``, ``client_request_id`` and ``warnings`` can be read before the first item. Use it in an
    ``async with`` block, or call :meth:`close`, to close the response when you stop early: a stream dropped
    unclosed emits a ``ResourceWarning``. Errors are raised as in :class:`Stream`.
    """

    response: httpx.Response
    """The HTTP response."""
    request_id: str | None
    """Value of the ``x-request-id`` response header, or ``None`` when it is absent."""
    client_request_id: str | None
    """Value of the ``x-client-request-id`` response header, or ``None`` when it is absent."""
    warnings: list[str]
    """Informational notices from the ``X-DEVUP-Warnings`` response header."""

    def __init__(
        self,
        response: httpx.Response,
        *,
        parse: Callable[[object], T],
        error_frame: ErrorFrame,
        on_debug: Callable[[DebugEvent], None] | None,
        idle_timeout: float | None,
    ) -> None:
        """Wraps the unread ``response``; built by the API methods."""
        self.response = response
        self.request_id = response.headers.get("x-request-id")
        self.client_request_id = response.headers.get("x-client-request-id")
        self.warnings = parse_warnings(response.headers)
        self._frames = _Frames(response, parse, error_frame, on_debug, idle_timeout)
        self._chunks: AsyncIterator[bytes] | None = None
        self._iterated = False
        self._closed = False
        self._finished = False

    def __aiter__(self) -> Self:
        if self._iterated:
            raise RuntimeError(_ITERATED_MESSAGE)
        self._iterated = True
        return self

    async def __anext__(self) -> T:
        try:
            return await self._next()
        except BaseException:
            await self.close()
            raise

    async def _next(self) -> T:
        frames = self._frames
        while True:
            if self._closed or self._finished:
                raise StopAsyncIteration
            if frames.events:
                outcome = frames.handle(frames.events.popleft())
                if outcome is _DONE:
                    await self._end()
                    raise StopAsyncIteration
                if outcome is _SKIP:
                    continue
                return cast("T", outcome)
            if frames.body_done:
                frames.finish()
                await self._end()
                raise StopAsyncIteration
            chunk = await self._read()
            if self._closed:
                raise StopAsyncIteration
            if chunk is None:
                frames.push(b"", final=True)
            else:
                frames.push(chunk)

    async def _read(self) -> bytes | None:
        """The next piece of the body, or ``None`` at its end. Each read waits at most the idle timeout."""
        if self._chunks is None:
            self._chunks = self.response.aiter_bytes()
        chunks = self._chunks
        idle = self._frames.idle_timeout
        try:
            if idle is None:
                return await anext(chunks)
            with anyio.move_on_after(idle):
                return await anext(chunks)
        except StopAsyncIteration:
            return None
        except _READ_ERRORS as error:
            if self._closed:
                return None
            raise self._frames.read_error(error) from error
        # The read did not return within the idle timeout: the scope cancelled it.
        raise self._frames.idle_error()

    async def _end(self) -> None:
        self._finished = True
        await self.close()

    async def close(self) -> None:
        """Closes the response. Iteration then ends quietly. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        # Shielded, so the response is closed also when the surrounding task is being cancelled.
        with anyio.CancelScope(shield=True):
            await self.response.aclose()

    def __del__(self) -> None:
        # An async response cannot be closed from here: say so, as an unclosed file does.
        if getattr(self, "_closed", True):
            return
        warnings.warn(_DROPPED_MESSAGE, ResourceWarning, stacklevel=2, source=self)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
