"""Health check: ``GET /health``, next to the versioned API path."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urljoin

from devupai._response import APIResponse
from devupai.resources._base import AsyncAPIResource, SyncAPIResource, endpoint, options, with_data
from devupai.types import HealthResponse
from devupai.types._base import parse_model

__all__ = ["Health", "AsyncHealth", "HealthWithResponse", "AsyncHealthWithResponse"]

_PATH = "/health"


def _health_url(base_url: str) -> str:
    """The health URL next to the versioned path of ``base_url``: ``https://host/v1`` → ``https://host/health``."""
    return urljoin(base_url + "/", "../health")


class HealthWithResponse(SyncAPIResource):
    """The methods of :class:`Health`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("GET", _PATH, billed=False, root=True)
    def check(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[HealthResponse]:
        """Checks that the API is reachable; see :meth:`Health.check`."""
        response = self._send(
            "GET",
            url=_health_url(self._client.base_url),
            auth=False,
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(HealthResponse, response.data))


class Health(SyncAPIResource):
    """Health check: ``GET /health``, next to the versioned API path."""

    @property
    def with_response(self) -> HealthWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return HealthWithResponse(self._client)

    @endpoint("GET", _PATH, billed=False, root=True)
    def check(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> HealthResponse:
        """Checks that the API is reachable. Sends no API key, and no ``Authorization`` header at all.

        Example::

            health = client.health.check()
        """
        return self.with_response.check(extra_headers=extra_headers, timeout=timeout, max_retries=max_retries).data


class AsyncHealthWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncHealth`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("GET", _PATH, billed=False, root=True)
    async def check(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[HealthResponse]:
        """Checks that the API is reachable; see :meth:`AsyncHealth.check`."""
        response = await self._send(
            "GET",
            url=_health_url(self._client.base_url),
            auth=False,
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(HealthResponse, response.data))


class AsyncHealth(AsyncAPIResource):
    """Health check: ``GET /health``, next to the versioned API path."""

    @property
    def with_response(self) -> AsyncHealthWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncHealthWithResponse(self._client)

    @endpoint("GET", _PATH, billed=False, root=True)
    async def check(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> HealthResponse:
        """Checks that the API is reachable. Sends no API key, and no ``Authorization`` header at all.

        Example::

            health = await client.health.check()
        """
        response = await self.with_response.check(extra_headers=extra_headers, timeout=timeout, max_retries=max_retries)
        return response.data
