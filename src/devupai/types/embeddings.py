"""Types of ``POST /v1/embeddings``."""

from collections.abc import Sequence
from typing import Literal, Required, TypedDict

from pydantic import Field

from ._base import DevupModel
from .shared import DevUpBilling

__all__ = [
    "EmbeddingCreateParams",
    "EmbeddingCreateResponse",
    "EmbeddingCreateResponseData",
    "EmbeddingCreateResponseUsage",
]


class EmbeddingCreateParams(TypedDict, total=False):
    """The request body of ``POST /v1/embeddings``."""

    model: Required[str]
    input: Required[str | Sequence[str]]
    encoding_format: Literal["float", "base64"]


class EmbeddingCreateResponseData(DevupModel):
    """One embedding."""

    object: Literal["embedding"] | str | None = None
    index: int | None = None
    embedding: list[float] | str | None = None
    """The vector, or a base64 string when ``encoding_format`` is ``"base64"``."""


class EmbeddingCreateResponseUsage(DevupModel):
    """Token counts of the request."""

    prompt_tokens: int | None = None
    total_tokens: int | None = None


class EmbeddingCreateResponse(DevupModel):
    """The response of ``POST /v1/embeddings``."""

    object: Literal["list"] | str
    data: list[EmbeddingCreateResponseData]
    model: str
    usage: EmbeddingCreateResponseUsage | None = None
    devup: DevUpBilling | None = Field(default=None, alias="_devup")
    """Billing details of the request (``_devup`` on the wire)."""
