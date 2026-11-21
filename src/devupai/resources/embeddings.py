"""Embeddings: ``POST /embeddings``."""

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
from devupai.types import EmbeddingCreateResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai.types import EmbeddingCreateParams

__all__ = ["Embeddings", "AsyncEmbeddings", "EmbeddingsWithResponse", "AsyncEmbeddingsWithResponse"]

_PATH = "/embeddings"


class EmbeddingsWithResponse(SyncAPIResource):
    """The methods of :class:`Embeddings`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[EmbeddingCreateParams],
    ) -> APIResponse[EmbeddingCreateResponse]:
        """Creates embedding vectors for the input text; see :meth:`Embeddings.create`."""
        response = self._send(
            "POST",
            _PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(EmbeddingCreateResponse, response.data))


class Embeddings(SyncAPIResource):
    """Embeddings: ``POST /embeddings``."""

    @property
    def with_response(self) -> EmbeddingsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return EmbeddingsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[EmbeddingCreateParams],
    ) -> EmbeddingCreateResponse:
        """Creates embedding vectors for the input text.

        With ``encoding_format="base64"`` each ``embedding`` is a base64 string instead of a list of numbers.
        ``extra_body`` adds fields to the JSON body (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.embeddings.create(model="<model-id>", input="Hello")
        """
        return self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data


class AsyncEmbeddingsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncEmbeddings`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[EmbeddingCreateParams],
    ) -> APIResponse[EmbeddingCreateResponse]:
        """Creates embedding vectors for the input text; see :meth:`AsyncEmbeddings.create`."""
        response = await self._send(
            "POST",
            _PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(EmbeddingCreateResponse, response.data))


class AsyncEmbeddings(AsyncAPIResource):
    """Embeddings: ``POST /embeddings``."""

    @property
    def with_response(self) -> AsyncEmbeddingsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncEmbeddingsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[EmbeddingCreateParams],
    ) -> EmbeddingCreateResponse:
        """Creates embedding vectors for the input text.

        With ``encoding_format="base64"`` each ``embedding`` is a base64 string instead of a list of numbers.
        ``extra_body`` adds fields to the JSON body (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.embeddings.create(model="<model-id>", input="Hello")
        """
        response = await self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data
