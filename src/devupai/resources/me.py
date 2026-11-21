"""Account profile: ``GET /me`` and ``GET /me/avatar``."""

from __future__ import annotations

from collections.abc import Mapping

from devupai._response import APIResponse, AsyncBinaryResponse, BinaryResponse
from devupai.resources._base import (
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
    ) -> APIResponse[BinaryResponse]:
        """Fetches the account's profile picture; see :meth:`Me.avatar`."""
        response = self._send(
            "GET",
            _AVATAR_PATH,
            if_none_match=if_none_match,
            kind="raw",
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, BinaryResponse(response.response))


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
    ) -> BinaryResponse:
        """Fetches the account's profile picture, as a :class:`~devupai.BinaryResponse` with its body not read yet.

        Read the ``content-type`` and ``etag`` headers, and the bytes with ``read()``. With ``if_none_match`` set to
        the ``etag`` of a copy you already have (from :meth:`retrieve` or an earlier ``ETag`` header), an unchanged
        picture gives a result with ``status_code == 304`` and an empty body instead of the bytes.

        An account without a picture raises :class:`~devupai.NotFoundError` (code ``avatar_not_found``).

        Example::

            with client.me.avatar() as avatar:
                data = avatar.read()
                etag = avatar.headers.get("etag")
        """
        return self.with_response.avatar(
            if_none_match=if_none_match, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        ).data


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
    ) -> APIResponse[AsyncBinaryResponse]:
        """Fetches the account's profile picture; see :meth:`AsyncMe.avatar`."""
        response = await self._send(
            "GET",
            _AVATAR_PATH,
            if_none_match=if_none_match,
            kind="raw",
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, AsyncBinaryResponse(response.response))


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
    ) -> AsyncBinaryResponse:
        """Fetches the account's profile picture, as an :class:`~devupai.AsyncBinaryResponse` with its body not read
        yet.

        Read the ``content-type`` and ``etag`` headers, and the bytes with ``await read()``. With ``if_none_match``
        set to the ``etag`` of a copy you already have (from :meth:`retrieve` or an earlier ``ETag`` header), an
        unchanged picture gives a result with ``status_code == 304`` and an empty body instead of the bytes.

        An account without a picture raises :class:`~devupai.NotFoundError` (code ``avatar_not_found``).

        Example::

            async with await client.me.avatar() as avatar:
                data = await avatar.read()
                etag = avatar.headers.get("etag")
        """
        response = await self.with_response.avatar(
            if_none_match=if_none_match, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        )
        return response.data
