"""The result of a request: the parsed data together with the HTTP response, and the binary results."""

from __future__ import annotations

import os
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Self, TypeVar

import anyio
import httpx

from devupai._errors import APIConnectionError, APITimeoutError, parse_warnings

T = TypeVar("T")

__all__ = ["APIResponse", "BinaryResponse", "AsyncBinaryResponse", "make_api_response"]

_READ_ERRORS = (httpx.TransportError, httpx.DecodingError)
_READ_TIMEOUT_MESSAGE = "The body of the response was not received in time."
_UNCLOSED_MESSAGE = (
    "An AsyncBinaryResponse was dropped while its response was open. Use `async with`, or `await close()`, so the "
    "connection is given back."
)


def _read_error(error: Exception, response: httpx.Response) -> APIConnectionError:
    """The error raised when reading a binary body does not complete: a timeout, or a connection error."""
    if isinstance(error, httpx.TimeoutException):
        return APITimeoutError(0, _READ_TIMEOUT_MESSAGE, headers=response.headers)
    return APIConnectionError(0, f"Network error: {str(error) or 'unknown error'}", headers=response.headers)


def _checked_chunks(response: httpx.Response, chunks: Iterator[bytes]) -> Iterator[bytes]:
    """``chunks``, with a read that does not complete raised as an SDK error; the response is then closed."""
    try:
        yield from chunks
    except _READ_ERRORS as error:
        response.close()
        raise _read_error(error, response) from error


async def _checked_achunks(response: httpx.Response, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """The async twin of :func:`_checked_chunks`."""
    try:
        async for chunk in chunks:
            yield chunk
    except _READ_ERRORS as error:
        await response.aclose()
        raise _read_error(error, response) from error


@dataclass(frozen=True)
class APIResponse(Generic[T]):
    """The data of a response together with the HTTP response, its request ids and its warnings."""

    data: T
    """The parsed response data."""
    response: httpx.Response
    """The HTTP response. Its body has already been read, except for raw responses."""
    request_id: str | None
    """Value of the ``x-request-id`` response header, or ``None`` when it is absent."""
    client_request_id: str | None
    """Value of the ``x-client-request-id`` response header, or ``None`` when it is absent."""
    warnings: list[str]
    """Informational notices from the ``X-DEVUP-Warnings`` response header."""


def make_api_response(data: T, response: httpx.Response) -> APIResponse[T]:
    """Builds the :class:`APIResponse` of ``response`` carrying ``data``."""
    return APIResponse(
        data=data,
        response=response,
        request_id=response.headers.get("x-request-id"),
        client_request_id=response.headers.get("x-client-request-id"),
        warnings=parse_warnings(response.headers),
    )


class BinaryResponse:
    """A file answer of the API (an image, an audio clip, a video), with its body not read yet.

    Read the bytes with :meth:`read`, :meth:`iter_bytes` or :meth:`write_to_file`. Close it with :meth:`close`, or
    use it in a ``with`` block, when the body is not read to the end; a result that is dropped unread is closed when
    it is collected. A read that does not complete raises :class:`~devupai.APIConnectionError`
    (:class:`~devupai.APITimeoutError` when it times out).

    Example::

        with client.me.avatar() as avatar:
            avatar.write_to_file("avatar.png")
    """

    response: httpx.Response
    """The HTTP response."""
    status_code: int
    """The HTTP status (``304`` for an unchanged file asked for with ``If-None-Match``)."""
    headers: httpx.Headers
    """The response headers (``content-type``, ``etag``, ...)."""
    request_id: str | None
    """Value of the ``x-request-id`` response header, or ``None`` when it is absent."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.request_id = response.headers.get("x-request-id")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(status_code={self.status_code})"

    def __del__(self) -> None:
        # Gives the connection back when the result is dropped unread. ``response`` is missing only when
        # ``__init__`` did not run to the end. Closing a closed response does nothing.
        response: httpx.Response | None = vars(self).get("response")
        if response is not None:
            response.close()

    def read(self) -> bytes:
        """Reads the whole body and returns its bytes. Calling it again returns the same bytes."""
        try:
            return self.response.read()
        except _READ_ERRORS as error:
            self.response.close()
            raise _read_error(error, self.response) from error

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        """Yields the body in pieces: as they arrive, or of ``chunk_size`` bytes each (the last one can be shorter)."""
        return _checked_chunks(self.response, self.response.iter_bytes(chunk_size))

    def write_to_file(self, path: str | os.PathLike[str]) -> None:
        """Writes the body to the file at ``path``, replacing it when it exists, piece by piece."""
        with open(path, "wb") as target:
            for chunk in self.iter_bytes():
                target.write(chunk)

    def close(self) -> None:
        """Closes the response. Calling it again does nothing."""
        self.response.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncBinaryResponse:
    """The async twin of :class:`BinaryResponse`: a file answer of the API, with its body not read yet.

    Read the bytes with :meth:`read`, :meth:`aiter_bytes` or :meth:`write_to_file`. Close it with :meth:`close`, or
    use it in an ``async with`` block, when the body is not read to the end: a result dropped while still open
    keeps its connection and emits a ``ResourceWarning``. A read that does not complete raises
    :class:`~devupai.APIConnectionError` (:class:`~devupai.APITimeoutError` when it times out).

    Example::

        async with await client.me.avatar() as avatar:
            await avatar.write_to_file("avatar.png")
    """

    response: httpx.Response
    """The HTTP response."""
    status_code: int
    """The HTTP status (``304`` for an unchanged file asked for with ``If-None-Match``)."""
    headers: httpx.Headers
    """The response headers (``content-type``, ``etag``, ...)."""
    request_id: str | None
    """Value of the ``x-request-id`` response header, or ``None`` when it is absent."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.request_id = response.headers.get("x-request-id")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(status_code={self.status_code})"

    def __del__(self) -> None:
        # An async close cannot run here, so an open response is reported instead. ``response`` is missing only
        # when ``__init__`` did not run to the end.
        response: httpx.Response | None = vars(self).get("response")
        if response is not None and not response.is_closed:
            warnings.warn(_UNCLOSED_MESSAGE, ResourceWarning, stacklevel=2, source=self)

    async def read(self) -> bytes:
        """Reads the whole body and returns its bytes. Calling it again returns the same bytes."""
        try:
            return await self.response.aread()
        except _READ_ERRORS as error:
            await self.response.aclose()
            raise _read_error(error, self.response) from error

    def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        """Yields the body in pieces: as they arrive, or of ``chunk_size`` bytes each (the last one can be shorter)."""
        return _checked_achunks(self.response, self.response.aiter_bytes(chunk_size))

    async def write_to_file(self, path: str | os.PathLike[str]) -> None:
        """Writes the body to the file at ``path``, replacing it when it exists, piece by piece."""
        async with await anyio.open_file(path, "wb") as target:
            async for chunk in self.aiter_bytes():
                await target.write(chunk)

    async def close(self) -> None:
        """Closes the response. Calling it again does nothing."""
        await self.response.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
