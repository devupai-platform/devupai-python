"""Account profile: ``GET /me`` and ``GET /me/avatar``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn

from devupai._response import APIResponse
from devupai.resources._base import (
    NOT_IMPLEMENTED_MESSAGE,
    AsyncAPIResource,
    SyncAPIResource,
    endpoint,
    options,
    with_data,
)
from devupai.types import AccountProfile
from devupai.types._base import parse_model

__all__ = ["Me", "AsyncMe", "MeWithResponse", "AsyncMeWithResponse"]

_PATH = "/me"
_AVATAR_PATH = "/me/avatar"


class MeWithResponse(SyncAPIResource):
    """The methods of :class:`Me`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("GET", _PATH, billed=False)
    def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[AccountProfile]:
        """Retrieves the profile of the account that owns the API key; see :meth:`Me.retrieve`."""
        response = self._send(
            "GET", _PATH, options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers)
        )
        return with_data(response, parse_model(AccountProfile, response.data))

    @endpoint("GET", _AVATAR_PATH, billed=False)
    def avatar(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches the account's profile picture. Available in a later build: raises ``NotImplementedError``."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class Me(SyncAPIResource):
    """Account profile: ``GET /me`` and ``GET /me/avatar``. Both use the API key."""

    @property
    def with_response(self) -> MeWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return MeWithResponse(self._client)

    @endpoint("GET", _PATH, billed=False)
    def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> AccountProfile:
        """Retrieves the profile of the account that owns the API key: a display name, an initial, a masked email
        hint and, when the account has a picture, the URL and ``etag`` of the avatar.

        Example::

            profile = client.me.retrieve()
            print(profile.display_name or profile.initial)
        """
        return self.with_response.retrieve(extra_headers=extra_headers, timeout=timeout, max_retries=max_retries).data

    @endpoint("GET", _AVATAR_PATH, billed=False)
    def avatar(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches the account's profile picture.

        Available in a later build: raises ``NotImplementedError`` before anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncMeWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncMe`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("GET", _PATH, billed=False)
    async def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[AccountProfile]:
        """Retrieves the profile of the account that owns the API key; see :meth:`AsyncMe.retrieve`."""
        response = await self._send(
            "GET", _PATH, options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers)
        )
        return with_data(response, parse_model(AccountProfile, response.data))

    @endpoint("GET", _AVATAR_PATH, billed=False)
    async def avatar(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches the account's profile picture. Available in a later build: raises ``NotImplementedError``."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncMe(AsyncAPIResource):
    """Account profile: ``GET /me`` and ``GET /me/avatar``. Both use the API key."""

    @property
    def with_response(self) -> AsyncMeWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncMeWithResponse(self._client)

    @endpoint("GET", _PATH, billed=False)
    async def retrieve(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> AccountProfile:
        """Retrieves the profile of the account that owns the API key: a display name, an initial, a masked email
        hint and, when the account has a picture, the URL and ``etag`` of the avatar.

        Example::

            profile = await client.me.retrieve()
            print(profile.display_name or profile.initial)
        """
        response = await self.with_response.retrieve(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        )
        return response.data

    @endpoint("GET", _AVATAR_PATH, billed=False)
    async def avatar(
        self,
        *,
        if_none_match: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches the account's profile picture.

        Available in a later build: raises ``NotImplementedError`` before anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
