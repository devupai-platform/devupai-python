"""Responses API: ``POST /responses``."""

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
from devupai.types import ResponseObject
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai.types import ResponseCreateParams

__all__ = ["Responses", "AsyncResponses", "ResponsesWithResponse", "AsyncResponsesWithResponse"]

_PATH = "/responses"


def _check_body(body: Mapping[str, object]) -> None:
    """Raises ``NotImplementedError`` when the merged body asks for a stream: streaming comes in a later build."""
    if body.get("stream") is True:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class ResponsesWithResponse(SyncAPIResource):
    """The methods of :class:`Responses`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> APIResponse[ResponseObject]:
        """Creates a model response; see :meth:`Responses.create`."""
        body = json_body(params, extra_body)
        _check_body(body)
        response = self._send(
            "POST",
            _PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ResponseObject, response.data))


class Responses(SyncAPIResource):
    """Responses API: ``POST /responses``."""

    @property
    def with_response(self) -> ResponsesWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return ResponsesWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> ResponseObject:
        """Creates a model response.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        ``stream=True`` is available in a later build and raises ``NotImplementedError`` before anything is sent.

        Example::

            response = client.responses.create(model="<model-id>", input="Hello")
        """
        return self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data


class AsyncResponsesWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncResponses`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> APIResponse[ResponseObject]:
        """Creates a model response; see :meth:`AsyncResponses.create`."""
        body = json_body(params, extra_body)
        _check_body(body)
        response = await self._send(
            "POST",
            _PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ResponseObject, response.data))


class AsyncResponses(AsyncAPIResource):
    """Responses API: ``POST /responses``."""

    @property
    def with_response(self) -> AsyncResponsesWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncResponsesWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ResponseCreateParams],
    ) -> ResponseObject:
        """Creates a model response.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        ``stream=True`` is available in a later build and raises ``NotImplementedError`` before anything is sent.

        Example::

            response = await client.responses.create(model="<model-id>", input="Hello")
        """
        response = await self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data
