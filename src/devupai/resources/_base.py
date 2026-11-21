"""Shared parts of the API resources: the endpoint metadata, the resource base classes and the request helpers."""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
from urllib.parse import quote, unquote

from devupai._response import APIResponse

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI, RequestOptions, RetryPolicy

__all__ = [
    "NOT_IMPLEMENTED_MESSAGE",
    "Endpoint",
    "endpoint",
    "SyncAPIResource",
    "AsyncAPIResource",
    "is_file_input",
    "contains_file_input",
    "options",
    "json_body",
    "encode_path_segment",
    "encode_model_path",
    "with_data",
    "billed_post_retry",
]

NOT_IMPLEMENTED_MESSAGE = "available in a later build"
"""The message of the ``NotImplementedError`` raised by a form of a method that comes in a later build."""

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

_ENDPOINT_ATTRIBUTE = "__devup_endpoint__"
_INVALID_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


@dataclass(frozen=True)
class Endpoint:
    """The HTTP method and path template of an API method."""

    method: Literal["GET", "POST"]
    """The HTTP method."""
    path: str
    """The path template, relative to ``base_url`` (``"/models/{model}"``), or to the host root when ``root`` is set."""
    billed: bool
    """Whether a successful call is billed to the account."""
    root: bool = False
    """Whether ``path`` is relative to the host root instead of ``base_url``."""


def endpoint(method: Literal["GET", "POST"], path: str, *, billed: bool, root: bool = False) -> Callable[[F], F]:
    """Marks an API method with its :class:`Endpoint`, kept in its ``__devup_endpoint__`` attribute."""
    metadata = Endpoint(method=method, path=path, billed=billed, root=root)

    def mark(fn: F) -> F:
        vars(fn)[_ENDPOINT_ATTRIBUTE] = metadata
        return fn

    return mark


def billed_post_retry() -> RetryPolicy:
    """The retry policy of the billed POST methods."""
    from devupai import _client

    return _client.BILLED_POST_RETRY


class SyncAPIResource:
    """Base class of the resources of :class:`devupai.DevupAI`."""

    def __init__(self, client: DevupAI) -> None:
        self._client = client

    def _send(
        self,
        method: Literal["GET", "POST"],
        path: str | None = None,
        *,
        url: str | None = None,
        query: Mapping[str, str | None] | None = None,
        body: object = None,
        auth: bool = True,
        if_none_match: str | None = None,
        retry: RetryPolicy | None = None,
        kind: Literal["json", "text", "raw"] = "json",
        options: RequestOptions | None = None,
    ) -> APIResponse[Any]:
        from devupai import _client

        return _client.sync_request(
            self._client,
            method,
            path,
            url=url,
            query=query,
            body=body,
            auth=auth,
            if_none_match=if_none_match,
            retry=_client.DEFAULT_RETRY if retry is None else retry,
            kind=kind,
            options=options,
        )


class AsyncAPIResource:
    """Base class of the resources of :class:`devupai.AsyncDevupAI`."""

    def __init__(self, client: AsyncDevupAI) -> None:
        self._client = client

    async def _send(
        self,
        method: Literal["GET", "POST"],
        path: str | None = None,
        *,
        url: str | None = None,
        query: Mapping[str, str | None] | None = None,
        body: object = None,
        auth: bool = True,
        if_none_match: str | None = None,
        retry: RetryPolicy | None = None,
        kind: Literal["json", "text", "raw"] = "json",
        options: RequestOptions | None = None,
    ) -> APIResponse[Any]:
        from devupai import _client

        return await _client.async_request(
            self._client,
            method,
            path,
            url=url,
            query=query,
            body=body,
            auth=auth,
            if_none_match=if_none_match,
            retry=_client.DEFAULT_RETRY if retry is None else retry,
            kind=kind,
            options=options,
        )


def _is_single_file(value: object) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview, os.PathLike)):
        return True
    return callable(getattr(value, "read", None))


def is_file_input(value: object) -> bool:
    """Whether ``value`` is a file: bytes, a path, an object with a ``read`` method, or a ``(name, file, ...)``
    tuple."""
    if isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        return len(items) >= 2 and _is_single_file(items[1])
    return _is_single_file(value)


def contains_file_input(value: object) -> bool:
    """Whether ``value``, or an item of a ``list`` or ``tuple`` value, is a file."""
    if is_file_input(value):
        return True
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return any(is_file_input(item) for item in items)
    return False


def options(
    *, timeout: float | None, max_retries: int | None, extra_headers: Mapping[str, str] | None
) -> RequestOptions:
    """The request options of one call, without the values that are not set."""
    result: RequestOptions = {}
    if timeout is not None:
        result["timeout"] = timeout
    if max_retries is not None:
        result["max_retries"] = max_retries
    if extra_headers is not None:
        result["extra_headers"] = extra_headers
    return result


def json_body(params: Mapping[str, object], extra_body: Mapping[str, object] | None) -> dict[str, object]:
    """The JSON body of a call: ``params``, then ``extra_body`` merged last."""
    body: dict[str, object] = dict(params)
    if extra_body is not None:
        body.update(extra_body)
    return body


def encode_path_segment(value: str) -> str:
    """Encodes ``value`` as one URL path segment, the same way as JavaScript's ``encodeURIComponent``."""
    return quote(value, safe="!*'()")


def encode_model_path(value: str) -> str:
    """Encodes a model path as one URL path segment.

    Leading slashes are removed, and a path that is already encoded is decoded first, so it is encoded exactly once
    (``/`` becomes ``%2F``). A path that is not valid percent-encoded text is encoded as it is.
    """
    clean = value.lstrip("/")
    if _INVALID_PERCENT.search(clean):
        return encode_path_segment(clean)
    try:
        decoded = unquote(clean, errors="strict")
    except UnicodeDecodeError:
        return encode_path_segment(clean)
    return encode_path_segment(decoded)


def with_data(response: APIResponse[object], data: T) -> APIResponse[T]:
    """A copy of ``response`` that carries ``data``."""
    return cast("APIResponse[T]", dataclasses.replace(response, data=data))
