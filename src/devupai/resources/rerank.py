"""Rerank: ``POST /rerank``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Unpack

from devupai._response import APIResponse
from devupai.resources._base import (
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.types import RerankCreateResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai.types import RerankCreateParams

__all__ = ["Rerank", "AsyncRerank", "RerankWithResponse", "AsyncRerankWithResponse"]

_PATH = "/rerank"


class RerankWithResponse(SyncAPIResource):
    """The methods of :class:`Rerank`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[RerankCreateParams],
    ) -> APIResponse[RerankCreateResponse]:
        """Orders documents by their relevance to a query; see :meth:`Rerank.create`."""
        response = self._send(
            "POST",
            _PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(RerankCreateResponse, response.data))


class Rerank(SyncAPIResource):
    """Rerank: ``POST /rerank``."""

    @property
    def with_response(self) -> RerankWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return RerankWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[RerankCreateParams],
    ) -> RerankCreateResponse:
        """Orders documents by their relevance to a query.

        ``extra_body`` adds fields to the JSON body (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.rerank.create(
                model="<model-id>",
                query="What is DZD?",
                documents=["The Algerian dinar.", "A file format."],
            )
        """
        return self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data


class AsyncRerankWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncRerank`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[RerankCreateParams],
    ) -> APIResponse[RerankCreateResponse]:
        """Orders documents by their relevance to a query; see :meth:`AsyncRerank.create`."""
        response = await self._send(
            "POST",
            _PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(RerankCreateResponse, response.data))


class AsyncRerank(AsyncAPIResource):
    """Rerank: ``POST /rerank``."""

    @property
    def with_response(self) -> AsyncRerankWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncRerankWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[RerankCreateParams],
    ) -> RerankCreateResponse:
        """Orders documents by their relevance to a query.

        ``extra_body`` adds fields to the JSON body (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.rerank.create(
                model="<model-id>",
                query="What is DZD?",
                documents=["The Algerian dinar.", "A file format."],
            )
        """
        response = await self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data
