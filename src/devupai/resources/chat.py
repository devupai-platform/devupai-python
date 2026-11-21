"""Chat completions: ``POST /chat/completions``."""

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
from devupai.types import ChatCompletion, ChatCompletionChunk
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI, RequestOptions
    from devupai.types import ChatCompletionCreateParams
    from devupai.types.chat import ChatCompletionCreateParamsNonStreaming, ChatCompletionCreateParamsStreaming

__all__ = [
    "Chat",
    "AsyncChat",
    "Completions",
    "AsyncCompletions",
    "CompletionsWithResponse",
    "AsyncCompletionsWithResponse",
]

_PATH = "/chat/completions"

# What the ``with_response`` implementations return: one member per overload.
_SyncResult: TypeAlias = (
    APIResponse[ChatCompletion]
    | APIResponse[Stream[ChatCompletionChunk]]
    | APIResponse[ChatCompletion | Stream[ChatCompletionChunk]]
)
_AsyncResult: TypeAlias = (
    APIResponse[ChatCompletion]
    | APIResponse[AsyncStream[ChatCompletionChunk]]
    | APIResponse[ChatCompletion | AsyncStream[ChatCompletionChunk]]
)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _error_frame(frame: object, event: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    """Reads the error event of a chat completion stream: an object with an ``error`` object and no ``choices``.

    Any other frame is a regular chunk.
    """
    if not isinstance(frame, dict):
        return None
    record = cast("dict[str, object]", frame)
    error = record.get("error")
    if "choices" in record or not isinstance(error, dict):
        return None
    fields = cast("dict[str, object]", error)
    message = _str_or_none(fields.get("message"))
    return (
        DEFAULT_STREAM_ERROR_MESSAGE if message is None else message,
        _str_or_none(fields.get("code")),
        _str_or_none(fields.get("type")),
        _str_or_none(fields.get("param")),
    )


def _parse_chunk(frame: object) -> ChatCompletionChunk:
    return parse_model(ChatCompletionChunk, frame)


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


class CompletionsWithResponse(SyncAPIResource):
    """The methods of :class:`Completions`, returning the whole :class:`~devupai.APIResponse`."""

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
        **params: Unpack[ChatCompletionCreateParamsStreaming],
    ) -> APIResponse[Stream[ChatCompletionChunk]]: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsNonStreaming],
    ) -> APIResponse[ChatCompletion]: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> APIResponse[ChatCompletion | Stream[ChatCompletionChunk]]: ...
    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> _SyncResult:
        """Creates a model response for the given chat conversation; see :meth:`Completions.create`."""
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
                parse=_parse_chunk,
                error_frame=_error_frame,
                on_debug=debug_hook(self._client),
                idle_timeout=_idle_timeout(self._client, request_options),
            )
            return with_data(response, stream)
        response = self._send("POST", _PATH, body=body, retry=billed_post_retry(), options=request_options)
        return with_data(response, parse_model(ChatCompletion, response.data))


class Completions(SyncAPIResource):
    """Chat completions: ``POST /chat/completions``."""

    @property
    def with_response(self) -> CompletionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return CompletionsWithResponse(self._client)

    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsStreaming],
    ) -> Stream[ChatCompletionChunk]: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsNonStreaming],
    ) -> ChatCompletion: ...
    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> ChatCompletion | Stream[ChatCompletionChunk]: ...
    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Creates a model response for the given chat conversation.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        With ``stream=True`` the result is a :class:`~devupai.Stream` of :class:`~devupai.types.ChatCompletionChunk`
        items; the last chunk has empty ``choices`` and carries ``usage``. ``timeout`` then covers the time until the
        response headers arrive, and ``stream_idle_timeout`` (seconds, ``0`` turns it off) ends the stream when no
        data arrives for that long; it overrides the client option. An error event inside the stream raises a
        :class:`~devupai.DevupAPIError` with its ``code`` and the stream's request id. Pass ``stream`` as an
        argument: ``stream`` inside ``extra_body`` raises ``ValueError`` before anything is sent.

        Example::

            completion = client.chat.completions.create(
                model="<model-id>",
                messages=[{"role": "user", "content": "Hello"}],
            )

            with client.chat.completions.create(model="<model-id>", messages=messages, stream=True) as stream:
                for chunk in stream:
                    print(chunk.choices[0].delta.content if chunk.choices else "", end="")
        """
        return self.with_response.create(
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
            stream_idle_timeout=stream_idle_timeout,
            **params,
        ).data


class Chat(SyncAPIResource):
    """Chat resource."""

    completions: Completions
    """Chat completions."""

    def __init__(self, client: DevupAI) -> None:
        super().__init__(client)
        self.completions = Completions(client)


class AsyncCompletionsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncCompletions`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsStreaming],
    ) -> APIResponse[AsyncStream[ChatCompletionChunk]]: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsNonStreaming],
    ) -> APIResponse[ChatCompletion]: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> APIResponse[ChatCompletion | AsyncStream[ChatCompletionChunk]]: ...
    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> _AsyncResult:
        """Creates a model response for the given chat conversation; see :meth:`AsyncCompletions.create`."""
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
                parse=_parse_chunk,
                error_frame=_error_frame,
                on_debug=debug_hook(self._client),
                idle_timeout=_idle_timeout(self._client, request_options),
            )
            return with_data(response, stream)
        response = await self._send("POST", _PATH, body=body, retry=billed_post_retry(), options=request_options)
        return with_data(response, parse_model(ChatCompletion, response.data))


class AsyncCompletions(AsyncAPIResource):
    """Chat completions: ``POST /chat/completions``."""

    @property
    def with_response(self) -> AsyncCompletionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncCompletionsWithResponse(self._client)

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsStreaming],
    ) -> AsyncStream[ChatCompletionChunk]: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParamsNonStreaming],
    ) -> ChatCompletion: ...
    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]: ...
    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        stream_idle_timeout: float | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """Creates a model response for the given chat conversation.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        With ``stream=True`` the result is an :class:`~devupai.AsyncStream` of
        :class:`~devupai.types.ChatCompletionChunk` items, read with ``async for``; the rules of
        :meth:`Completions.create` apply.

        Example::

            completion = await client.chat.completions.create(
                model="<model-id>",
                messages=[{"role": "user", "content": "Hello"}],
            )
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


class AsyncChat(AsyncAPIResource):
    """Chat resource."""

    completions: AsyncCompletions
    """Chat completions."""

    def __init__(self, client: AsyncDevupAI) -> None:
        super().__init__(client)
        self.completions = AsyncCompletions(client)
