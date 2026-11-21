"""Shapes shared by several endpoints: billing details, token counts, the error body, and file inputs."""

import os
from typing import IO, Literal, TypeAlias

from ._base import DevupModel

__all__ = [
    "DevUpBilling",
    "ErrorResponse",
    "ErrorResponseError",
    "FileInput",
    "TokenUsage",
    "TokenUsageCompletionTokensDetails",
    "TokenUsagePromptTokensDetails",
]

FileInput: TypeAlias = (
    bytes
    | bytearray
    | memoryview
    | os.PathLike[str]
    | IO[bytes]
    | tuple[str, bytes | IO[bytes]]
    | tuple[str, bytes | IO[bytes], str]
)
"""A file for a multipart field: raw bytes, a path, an open binary file, or ``(filename, content[, content_type])``."""


class DevUpBilling(DevupModel):
    """Billing details of a request (``_devup``)."""

    cost_dzd: float | None = None
    """What the request cost, in DZD."""
    balance_dzd: float | None = None
    """The account balance after the request, in DZD."""
    request_id: str | None = None
    """The request identifier (``req_`` followed by 32 hex characters)."""
    warnings: list[str] | None = None
    """Notices about the request."""
    note: str | None = None
    """A short note about the billing of the request."""


class TokenUsagePromptTokensDetails(DevupModel):
    """A breakdown of the prompt tokens."""

    cached_tokens: int | None = None


class TokenUsageCompletionTokensDetails(DevupModel):
    """A breakdown of the completion tokens."""

    reasoning_tokens: int | None = None


class TokenUsage(DevupModel):
    """Token counts of a request."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: TokenUsagePromptTokensDetails | None = None
    completion_tokens_details: TokenUsageCompletionTokensDetails | None = None


class ErrorResponseError(DevupModel):
    """The error object of an error body."""

    message: str
    type: Literal["devup_error"] | str
    code: str
    """A stable error code."""


class ErrorResponse(DevupModel):
    """The body of an error response."""

    error: ErrorResponseError
