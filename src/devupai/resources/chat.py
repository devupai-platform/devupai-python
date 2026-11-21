"""Chat completions: ``POST /chat/completions``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Unpack

from devupai._response import APIResponse
from devupai.resources._base import (
    NOT_IMPLEMENTED_MESSAGE,
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.types import ChatCompletion
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI
    from devupai.types import ChatCompletionCreateParams

__all__ = [
    "Chat",
    "AsyncChat",
    "Completions",
    "AsyncCompletions",
    "CompletionsWithResponse",
    "AsyncCompletionsWithResponse",
]

_PATH = "/chat/completions"


def _check_body(body: Mapping[str, object]) -> None:
    """Raises ``NotImplementedError`` when the merged body asks for a stream: streaming comes in a later build."""
    if body.get("stream") is True:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class CompletionsWithResponse(SyncAPIResource):
    """The methods of :class:`Completions`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> APIResponse[ChatCompletion]:
        """Creates a model response for the given chat conversation; see :meth:`Completions.create`."""
        body = json_body(params, extra_body)
        _check_body(body)
        response = self._send(
            "POST",
            _PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ChatCompletion, response.data))


class Completions(SyncAPIResource):
    """Chat completions: ``POST /chat/completions``."""

    @property
    def with_response(self) -> CompletionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return CompletionsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> ChatCompletion:
        """Creates a model response for the given chat conversation.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        ``stream=True`` is available in a later build and raises ``NotImplementedError`` before anything is sent.

        Example::

            completion = client.chat.completions.create(
                model="<model-id>",
                messages=[{"role": "user", "content": "Hello"}],
            )
        """
        return self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
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

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> APIResponse[ChatCompletion]:
        """Creates a model response for the given chat conversation; see :meth:`AsyncCompletions.create`."""
        body = json_body(params, extra_body)
        _check_body(body)
        response = await self._send(
            "POST",
            _PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ChatCompletion, response.data))


class AsyncCompletions(AsyncAPIResource):
    """Chat completions: ``POST /chat/completions``."""

    @property
    def with_response(self) -> AsyncCompletionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncCompletionsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ChatCompletionCreateParams],
    ) -> ChatCompletion:
        """Creates a model response for the given chat conversation.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        ``stream=True`` is available in a later build and raises ``NotImplementedError`` before anything is sent.

        Example::

            completion = await client.chat.completions.create(
                model="<model-id>",
                messages=[{"role": "user", "content": "Hello"}],
            )
        """
        response = await self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data


class AsyncChat(AsyncAPIResource):
    """Chat resource."""

    completions: AsyncCompletions
    """Chat completions."""

    def __init__(self, client: AsyncDevupAI) -> None:
        super().__init__(client)
        self.completions = AsyncCompletions(client)
