"""Test helpers: a recording mock of the API and response builders.

Every client built here answers through ``httpx.MockTransport``, and every request is recorded (method, URL,
headers, body) before the handler sees it, so tests assert what the SDK sends, not only what it receives.

The ``devupai`` package is imported lazily, so ``tests/conftest.py`` can import this module (for
``LiveNetworkCall``) even when the package is not importable.
"""

from __future__ import annotations

import json
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import httpx

if TYPE_CHECKING:
    from devupai import AsyncDevupAI, DevupAI
    from devupai._response import APIResponse

TEST_KEY = "test-key-0000"
BASE = "https://api.example.test/v1"
#: Synthetic request id in the ``req_`` + 32 hex format.
REQ_ID = "req_" + "0" * 32
#: Synthetic model id (never a real one: the repository is public).
MODEL_A = "fixture/model-a"


class LiveNetworkCall(BaseException):
    """Raised when a test reaches a real transport.

    A ``BaseException`` subclass, so the SDK (which only catches ``Exception``) never turns it into a retry or an
    ``APIConnectionError``.
    """


@dataclass
class RecordedRequest:
    """One request as the mock transport received it."""

    method: str
    url: str
    headers: dict[str, str]  # lower-cased names
    content: bytes

    def json(self) -> Any:
        """The body parsed as JSON."""
        return json.loads(self.content)


Handler = Callable[[httpx.Request], httpx.Response]
Entry = httpx.Response | BaseException | Callable[[httpx.Request], "httpx.Response | BaseException"]


def _copy_response(response: httpx.Response) -> httpx.Response:
    return httpx.Response(response.status_code, headers=response.headers.multi_items(), content=response.content)


def sequence(*entries: Entry) -> Handler:
    """A handler that answers call *i* with entry *i*; the last entry repeats.

    - an ``httpx.Response`` is copied per call (same status, headers and content);
    - a ``BaseException`` is raised;
    - a callable is called with the request; if it returns an exception, that exception is raised.
    """
    if not entries:
        raise ValueError("sequence() needs at least one entry.")
    state = {"index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        entry = entries[min(state["index"], len(entries) - 1)]
        state["index"] += 1
        if isinstance(entry, httpx.Response):
            return _copy_response(entry)
        if isinstance(entry, BaseException):
            raise entry
        result = entry(request)
        if isinstance(result, BaseException):
            raise result
        return result

    return handler


def _has_header(headers: Mapping[str, str] | None, name: str) -> bool:
    return headers is not None and any(key.lower() == name.lower() for key in headers)


def json_response(body: object, status: int = 200, headers: Mapping[str, str] | None = None) -> httpx.Response:
    """A JSON response. ``x-request-id: REQ_ID`` is added unless ``headers`` sets it."""
    merged: dict[str, str] = {"content-type": "application/json"}
    if not _has_header(headers, "x-request-id"):
        merged["x-request-id"] = REQ_ID
    if headers is not None:
        for key, value in headers.items():
            for existing in [k for k in merged if k.lower() == key.lower()]:
                del merged[existing]
            merged[key] = value
    return httpx.Response(status, headers=merged, content=json.dumps(body).encode("utf-8"))


def error_response(
    status: int,
    message: str = "Error",
    *,
    code: str | None = None,
    param: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """An error response: ``{"error": {"message", "type": "devup_error", "code"?, "param"?}}``."""
    error: dict[str, object] = {"message": message, "type": "devup_error"}
    if code is not None:
        error["code"] = code
    if param is not None:
        error["param"] = param
    return json_response({"error": error}, status=status, headers=headers)


def tls_failure() -> httpx.ConnectError:
    """A fresh ``httpx.ConnectError`` raised from an ``ssl.SSLError`` (a real ``__cause__`` chain)."""
    try:
        try:
            raise ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        except ssl.SSLError as cause:
            raise httpx.ConnectError("TLS handshake failed") from cause
    except httpx.ConnectError as error:
        return error


class MockAPI:
    """A recording stand-in for the API, with clients wired to it."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[RecordedRequest] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            RecordedRequest(
                method=request.method,
                url=str(request.url),
                headers={key.lower(): value for key, value in request.headers.items()},
                content=bytes(request.content),
            )
        )
        return self.handler(request)

    def sync_client(self, **opts: Any) -> DevupAI:
        """A ``DevupAI`` answered by this mock (``opts`` override the defaults)."""
        from devupai import DevupAI

        kwargs: dict[str, Any] = {"api_key": TEST_KEY, "base_url": BASE}
        if "http_client" not in opts:
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(self._handle))
        kwargs.update(opts)
        return DevupAI(**kwargs)

    def async_client(self, **opts: Any) -> AsyncDevupAI:
        """An ``AsyncDevupAI`` answered by this mock (``opts`` override the defaults)."""
        from devupai import AsyncDevupAI

        kwargs: dict[str, Any] = {"api_key": TEST_KEY, "base_url": BASE}
        if "http_client" not in opts:
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        kwargs.update(opts)
        return AsyncDevupAI(**kwargs)

    def client(self, kind: str, **opts: Any) -> DevupAI | AsyncDevupAI:
        """``sync_client`` for ``kind="sync"``, ``async_client`` for ``kind="async"``."""
        if kind == "sync":
            return self.sync_client(**opts)
        if kind == "async":
            return self.async_client(**opts)
        raise ValueError(f"Unknown client kind: {kind!r}")


def run_request(client: DevupAI | AsyncDevupAI, *args: Any, **kwargs: Any) -> APIResponse[Any]:
    """Calls ``client._request(*args, **kwargs)``; an async client is run with ``anyio.run``.

    Exceptions propagate unchanged.
    """
    from devupai import AsyncDevupAI

    if isinstance(client, AsyncDevupAI):
        async_client = client

        async def main() -> APIResponse[Any]:
            return await async_client._request(*args, **kwargs)

        return anyio.run(main)
    return client._request(*args, **kwargs)
