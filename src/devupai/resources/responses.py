"""Responses API: ``POST /responses``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias, Unpack, cast, overload

from devupai._response import APIResponse
from devupai._streaming import DEFAULT_STREAM_ERROR_MESSAGE, AsyncStream, Stream, debug_hook, stream_requested
from devupai.resources._base import (
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.types import ResponseObject, ResponseStreamEvent
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI, RequestOptions
    from devupai.types import ResponseCreateParams
    from devupai.types.responses import ResponseCreateParamsNonStreaming, ResponseCreateParamsStreaming

__all__ = ["Responses", "AsyncResponses", "ResponsesWithResponse", "AsyncResponsesWithResponse"]

_PATH = "/responses"

# What the ``with_response`` implementations return: one member per overload.
_SyncResult: TypeAlias = (
    APIResponse[ResponseObject]
    | APIResponse[Stream[ResponseStreamEvent]]
    | APIResponse[ResponseObject | Stream[ResponseStreamEvent]]
)
_AsyncResult: TypeAlias = (
    APIResponse[ResponseObject]
    | APIResponse[AsyncStream[ResponseStreamEvent]]
    | APIResponse[ResponseObject | AsyncStream[ResponseStreamEvent]]
)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _error_frame(frame: object, event: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    """Reads the error event of a Responses stream: an SSE event named ``error``, or an event whose ``type`` is
    ``error``.

    The message, code and param come from the event, then from its ``error`` object.
    """
    record = cast("dict[str, object]", frame) if isinstance(frame, dict) else None
    if event != "error" and (record is None or record.get("type") != "error"):
        return None
    if record is None:
        return (DEFAULT_STREAM_ERROR_MESSAGE, None, None, None)
    nested = record.get("error")
    inner: dict[str, object] = cast("dict[str, object]", nested) if isinstance(nested, dict) else {}

    def first(name: str) -> str | None:
        value = _str_or_none(record.get(name))
        return _str_or_none(inner.get(name)) if value is None else value

    message = first("message")
    return (
        DEFAULT_STREAM_ERROR_MESSAGE if message is None else message,
        first("code"),
        _str_or_none(inner.get("type")),
        first("param"),
    )


def _parse_event(frame: object) -> ResponseStreamEvent:
    return parse_model(ResponseStreamEvent, frame)


def _request_options(
    *,
    timeout: float | None,
    max_retries: int | None,
    extra_headers: Mapping[str, str] | None,
    stream_idle_timeout: float | None,
) -> RequestOptions:
    result = options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers)
    if stream_idle_timeout is not None:
        result["stream_idle_timeout"] = stream_idle_timeout
    return result


def _idle_timeout(client: DevupAI | AsyncDevupAI, request_options: RequestOptions) -> float | None:
    from devupai import _client

    return _client.effective_idle_timeout(client, request_options)


class ResponsesWithResponse(SyncAPIResource):
    """The methods of :class:`Responses`, returning the whole :class:`~devupai.APIResponse`."""

    # APIResponse is invariant in its data type, so this overload's result is not a subtype of the third one's; the
    # overloads are tried in order, and a stream=True call always gets this one.
    @overload
    def create(  # pyright: ignore[reportOverlappingOverload]
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsStreaming],
    ) -> APIResponse[Stream[ResponseStreamEvent]]: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsNonStreaming],
    ) -> APIResponse[ResponseObject]: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> APIResponse[ResponseObject | Stream[ResponseStreamEvent]]: ...
    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> _SyncResult:
        """Creates a model response; see :meth:`Responses.create`."""
        body = json_body(params, extra_body)
        streaming = stream_requested(params, body)
        request_options = _request_options(
            timeout=timeout,
            max_retries=max_retries,
            extra_headers=extra_headers,
            stream_idle_timeout=stream_idle_timeout,
        )
        if streaming:
            response = self._send(
                "POST", _PATH, body=body, retry=billed_post_retry(), kind="stream", options=request_options
            )
            stream = Stream(
                response.data,
                parse=_parse_event,
                error_frame=_error_frame,
                on_debug=debug_hook(self._client),
                idle_timeout=_idle_timeout(self._client, request_options),
            )
            return with_data(response, stream)
        response = self._send("POST", _PATH, body=body, retry=billed_post_retry(), options=request_options)
        return with_data(response, parse_model(ResponseObject, response.data))


class Responses(SyncAPIResource):
    """Responses API: ``POST /responses``."""

    @property
    def with_response(self) -> ResponsesWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return ResponsesWithResponse(self._client)

    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsStreaming],
    ) -> Stream[ResponseStreamEvent]: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsNonStreaming],
    ) -> ResponseObject: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> ResponseObject | Stream[ResponseStreamEvent]: ...
    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> ResponseObject | Stream[ResponseStreamEvent]:
        """Creates a model response.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        With ``stream=True`` the result is a :class:`~devupai.Stream` of
        :class:`~devupai.types.ResponseStreamEvent` items; skip any event type your code does not handle.
        ``timeout`` then covers the time until the response headers arrive, and ``stream_idle_timeout`` (seconds,
        ``0`` turns it off) ends the stream when no data arrives for that long; it overrides the client option. An
        ``error`` event raises a :class:`~devupai.DevupAPIError` with its ``code`` and the stream's request id. Pass
        ``stream`` as an argument: ``stream`` inside ``extra_body`` raises ``ValueError`` before anything is sent.

        Example::

            response = client.responses.create(model="<model-id>", input="Hello")

            with client.responses.create(model="<model-id>", input="Hello", stream=True) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="")
        """
        return self.with_response.create(
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
            stream_idle_timeout=stream_idle_timeout,
            **params,
        ).data


class AsyncResponsesWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncResponses`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsStreaming],
    ) -> APIResponse[AsyncStream[ResponseStreamEvent]]: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsNonStreaming],
    ) -> APIResponse[ResponseObject]: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> APIResponse[ResponseObject | AsyncStream[ResponseStreamEvent]]: ...
    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> _AsyncResult:
        """Creates a model response; see :meth:`AsyncResponses.create`."""
        body = json_body(params, extra_body)
        streaming = stream_requested(params, body)
        request_options = _request_options(
            timeout=timeout,
            max_retries=max_retries,
            extra_headers=extra_headers,
            stream_idle_timeout=stream_idle_timeout,
        )
        if streaming:
            response = await self._send(
                "POST", _PATH, body=body, retry=billed_post_retry(), kind="stream", options=request_options
            )
            stream = AsyncStream(
                response.data,
                parse=_parse_event,
                error_frame=_error_frame,
                on_debug=debug_hook(self._client),
                idle_timeout=_idle_timeout(self._client, request_options),
            )
            return with_data(response, stream)
        response = await self._send("POST", _PATH, body=body, retry=billed_post_retry(), options=request_options)
        return with_data(response, parse_model(ResponseObject, response.data))


class AsyncResponses(AsyncAPIResource):
    """Responses API: ``POST /responses``."""

    @property
    def with_response(self) -> AsyncResponsesWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncResponsesWithResponse(self._client)

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsStreaming],
    ) -> AsyncStream[ResponseStreamEvent]: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParamsNonStreaming],
    ) -> ResponseObject: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> ResponseObject | AsyncStream[ResponseStreamEvent]: ...
    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> ResponseObject | AsyncStream[ResponseStreamEvent]:
        """Creates a model response.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        With ``stream=True`` the result is an :class:`~devupai.AsyncStream` of
        :class:`~devupai.types.ResponseStreamEvent` items, read with ``async for``; the rules of
        :meth:`Responses.create` apply.

        Example::

            response = await client.responses.create(model="<model-id>", input="Hello")
        """
        response = await self.with_response.create(
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
            stream_idle_timeout=stream_idle_timeout,
            **params,
        )
        return response.data
