"""Models: ``GET /models`` and ``GET /models/{model}``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, overload

from devupai._response import APIResponse
from devupai.resources._base import (
    AsyncAPIResource,
    SyncAPIResource,
    encode_path_segment,
    endpoint,
    options,
    with_data,
)
from devupai.types import ModelListResponse, ModelObject
from devupai.types._base import DevupModel, parse_model

__all__ = ["Models", "AsyncModels", "ModelsWithResponse", "AsyncModelsWithResponse"]

_M = TypeVar("_M", bound=DevupModel)

_LIST_PATH = "/models"
_RETRIEVE_PATH = "/models/{model}"


def _parse_conditional(cls: type[_M], response: APIResponse[Any]) -> APIResponse[_M | None]:
    """The parsed model of ``response``, or ``None`` when it is a ``304 Not Modified``."""
    if response.response.status_code == 304:
        return with_data(response, None)
    return with_data(response, parse_model(cls, response.data))


class ModelsWithResponse(SyncAPIResource):
    """The methods of :class:`Models`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    def list(
        self,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelListResponse]: ...
    @overload
    def list(
        self,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelListResponse | None]: ...
    @endpoint("GET", _LIST_PATH, billed=False)
    def list(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelListResponse] | APIResponse[ModelListResponse | None]:
        """Lists the models available to your API key; see :meth:`Models.list`.

        On ``304 Not Modified`` the ``data`` is ``None``, and ``response`` is the 304 response with its ``ETag``.
        """
        response = self._send(
            "GET",
            _LIST_PATH,
            if_none_match=if_none_match,
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return _parse_conditional(ModelListResponse, response)

    @overload
    def retrieve(
        self,
        model: str,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelObject]: ...
    @overload
    def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelObject | None]: ...
    @endpoint("GET", _RETRIEVE_PATH, billed=False)
    def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelObject] | APIResponse[ModelObject | None]:
        """Retrieves one model by its id; see :meth:`Models.retrieve`.

        On ``304 Not Modified`` the ``data`` is ``None``, and ``response`` is the 304 response with its ``ETag``.
        """
        response = self._send(
            "GET",
            f"/models/{encode_path_segment(model)}",
            if_none_match=if_none_match,
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return _parse_conditional(ModelObject, response)


class Models(SyncAPIResource):
    """Models: ``GET /models`` and ``GET /models/{model}``. Both use the API key.

    Both answers carry an ``ETag``. Pass it back as ``if_none_match`` to receive ``None`` (``304 Not Modified``)
    while the list or the model is unchanged.
    """

    @property
    def with_response(self) -> ModelsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return ModelsWithResponse(self._client)

    @overload
    def list(
        self,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelListResponse: ...
    @overload
    def list(
        self,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelListResponse | None: ...
    @endpoint("GET", _LIST_PATH, billed=False)
    def list(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelListResponse | None:
        """Lists the models available to your API key, with their capabilities and pricing.

        With ``if_none_match`` set to the ``ETag`` of a previous answer, the method returns ``None`` when the list is
        unchanged (``304 Not Modified``); ``with_response.list(...)`` then gives the 304 response and its ``ETag``.

        Example::

            models = client.models.list()

            first = client.models.with_response.list()
            etag = first.response.headers.get("etag")
            # Later: None while the list is unchanged.
            again = client.models.list(if_none_match=etag) if etag else client.models.list()
        """
        return self.with_response.list(
            if_none_match=if_none_match, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        ).data

    @overload
    def retrieve(
        self,
        model: str,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelObject: ...
    @overload
    def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelObject | None: ...
    @endpoint("GET", _RETRIEVE_PATH, billed=False)
    def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelObject | None:
        """Retrieves one model by its id. The id is sent as one URL-encoded path segment, so ids that contain ``/``
        work as they are.

        With ``if_none_match`` set to the ``ETag`` of a previous answer, the method returns ``None`` when the model is
        unchanged (``304 Not Modified``); ``with_response.retrieve(...)`` then gives the 304 response and its
        ``ETag``.

        Example::

            model = client.models.retrieve("<model-id>")
        """
        return self.with_response.retrieve(
            model, if_none_match=if_none_match, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        ).data


class AsyncModelsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncModels`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    async def list(
        self,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelListResponse]: ...
    @overload
    async def list(
        self,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelListResponse | None]: ...
    @endpoint("GET", _LIST_PATH, billed=False)
    async def list(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelListResponse] | APIResponse[ModelListResponse | None]:
        """Lists the models available to your API key; see :meth:`AsyncModels.list`.

        On ``304 Not Modified`` the ``data`` is ``None``, and ``response`` is the 304 response with its ``ETag``.
        """
        response = await self._send(
            "GET",
            _LIST_PATH,
            if_none_match=if_none_match,
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return _parse_conditional(ModelListResponse, response)

    @overload
    async def retrieve(
        self,
        model: str,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelObject]: ...
    @overload
    async def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelObject | None]: ...
    @endpoint("GET", _RETRIEVE_PATH, billed=False)
    async def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[ModelObject] | APIResponse[ModelObject | None]:
        """Retrieves one model by its id; see :meth:`AsyncModels.retrieve`.

        On ``304 Not Modified`` the ``data`` is ``None``, and ``response`` is the 304 response with its ``ETag``.
        """
        response = await self._send(
            "GET",
            f"/models/{encode_path_segment(model)}",
            if_none_match=if_none_match,
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return _parse_conditional(ModelObject, response)


class AsyncModels(AsyncAPIResource):
    """Models: ``GET /models`` and ``GET /models/{model}``. Both use the API key.

    Both answers carry an ``ETag``. Pass it back as ``if_none_match`` to receive ``None`` (``304 Not Modified``)
    while the list or the model is unchanged.
    """

    @property
    def with_response(self) -> AsyncModelsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncModelsWithResponse(self._client)

    @overload
    async def list(
        self,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelListResponse: ...
    @overload
    async def list(
        self,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelListResponse | None: ...
    @endpoint("GET", _LIST_PATH, billed=False)
    async def list(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelListResponse | None:
        """Lists the models available to your API key, with their capabilities and pricing.

        With ``if_none_match`` set to the ``ETag`` of a previous answer, the method returns ``None`` when the list is
        unchanged (``304 Not Modified``); ``with_response.list(...)`` then gives the 304 response and its ``ETag``.

        Example::

            models = await client.models.list()
        """
        response = await self.with_response.list(
            if_none_match=if_none_match, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        )
        return response.data

    @overload
    async def retrieve(
        self,
        model: str,
        *,
        if_none_match: None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelObject: ...
    @overload
    async def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelObject | None: ...
    @endpoint("GET", _RETRIEVE_PATH, billed=False)
    async def retrieve(
        self,
        model: str,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ModelObject | None:
        """Retrieves one model by its id. The id is sent as one URL-encoded path segment, so ids that contain ``/``
        work as they are.

        With ``if_none_match`` set to the ``ETag`` of a previous answer, the method returns ``None`` when the model is
        unchanged (``304 Not Modified``); ``with_response.retrieve(...)`` then gives the 304 response and its
        ``ETag``.

        Example::

            model = await client.models.retrieve("<model-id>")
        """
        response = await self.with_response.retrieve(
            model, if_none_match=if_none_match, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        )
        return response.data
