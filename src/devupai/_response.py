"""The result of a request: the parsed data together with the HTTP response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx

from devupai._errors import parse_warnings

T = TypeVar("T")

__all__ = ["APIResponse", "make_api_response"]


@dataclass(frozen=True)
class APIResponse(Generic[T]):
    """The data of a response together with the HTTP response, its request ids and its warnings."""

    data: T
    """The parsed response data."""
    response: httpx.Response
    """The HTTP response. Its body has already been read, except for raw responses."""
    request_id: str | None
    """Value of the ``x-request-id`` response header, or ``None`` when it is absent."""
    client_request_id: str | None
    """Value of the ``x-client-request-id`` response header, or ``None`` when it is absent."""
    warnings: list[str]
    """Informational notices from the ``X-DEVUP-Warnings`` response header."""


def make_api_response(data: T, response: httpx.Response) -> APIResponse[T]:
    """Builds the :class:`APIResponse` of ``response`` carrying ``data``."""
    return APIResponse(
        data=data,
        response=response,
        request_id=response.headers.get("x-request-id"),
        client_request_id=response.headers.get("x-client-request-id"),
        warnings=parse_warnings(response.headers),
    )
