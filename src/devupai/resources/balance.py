"""Account balance: ``GET /user/balance``."""

from __future__ import annotations

from collections.abc import Mapping

from devupai._response import APIResponse
from devupai.resources._base import AsyncAPIResource, SyncAPIResource, endpoint, options, with_data
from devupai.types import BalanceResponse
from devupai.types._base import parse_model

__all__ = ["Balance", "AsyncBalance", "BalanceWithResponse", "AsyncBalanceWithResponse"]

_PATH = "/user/balance"


class BalanceWithResponse(SyncAPIResource):
    """The methods of :class:`Balance`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("GET", _PATH, billed=False)
    def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[BalanceResponse]:
        """Retrieves the current account balance in DZD; see :meth:`Balance.retrieve`."""
        response = self._send(
            "GET", _PATH, options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers)
        )
        return with_data(response, parse_model(BalanceResponse, response.data))


class Balance(SyncAPIResource):
    """Account balance: ``GET /user/balance``."""

    @property
    def with_response(self) -> BalanceWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return BalanceWithResponse(self._client)

    @endpoint("GET", _PATH, billed=False)
    def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> BalanceResponse:
        """Retrieves the current account balance in DZD.

        Example::

            balance = client.balance.retrieve()
        """
        return self.with_response.retrieve(extra_headers=extra_headers, timeout=timeout, max_retries=max_retries).data


class AsyncBalanceWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncBalance`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("GET", _PATH, billed=False)
    async def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[BalanceResponse]:
        """Retrieves the current account balance in DZD; see :meth:`AsyncBalance.retrieve`."""
        response = await self._send(
            "GET", _PATH, options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers)
        )
        return with_data(response, parse_model(BalanceResponse, response.data))


class AsyncBalance(AsyncAPIResource):
    """Account balance: ``GET /user/balance``."""

    @property
    def with_response(self) -> AsyncBalanceWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncBalanceWithResponse(self._client)

    @endpoint("GET", _PATH, billed=False)
    async def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> BalanceResponse:
        """Retrieves the current account balance in DZD.

        Example::

            balance = await client.balance.retrieve()
        """
        response = await self.with_response.retrieve(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        )
        return response.data
