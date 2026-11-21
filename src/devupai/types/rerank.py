"""Types of ``POST /v1/rerank``."""

from collections.abc import Sequence
from typing import Literal, Required, TypedDict

from pydantic import Field

from ._base import DevupModel
from .shared import DevUpBilling

__all__ = ["RerankCreateParams", "RerankCreateResponse", "RerankCreateResponseResult"]


class RerankCreateParams(TypedDict, total=False):
    """The request body of ``POST /v1/rerank``."""

    model: Required[str]
    query: Required[str]
    documents: Required[Sequence[str]]
    top_n: int
    return_documents: bool


class RerankCreateResponseResult(DevupModel):
    """One document with its relevance score."""

    index: int
    """The position of the document in the request."""
    relevance_score: float
    document: str | None = None
    """The document text, when ``return_documents`` is true."""


class RerankCreateResponse(DevupModel):
    """The response of ``POST /v1/rerank``."""

    object: Literal["list"] | str
    model: str
    results: list[RerankCreateResponseResult]
    devup: DevUpBilling = Field(alias="_devup")
    """Billing details of the request (``_devup`` on the wire)."""
