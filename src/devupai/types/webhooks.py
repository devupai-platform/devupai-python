"""Payloads of the webhook deliveries sent when an asynchronous request completes."""

from typing import Literal, TypeAlias

from ._base import DevupModel

__all__ = [
    "WebhookErrorDetails",
    "WebhookErrorPayload",
    "WebhookEvent",
    "WebhookSuccessPayload",
    "WebhookUsage",
]


class WebhookUsage(DevupModel):
    """Token counts reported in a webhook delivery."""

    prompt_tokens: int
    completion_tokens: int


class WebhookSuccessPayload(DevupModel):
    """The delivery sent when an asynchronous request completes."""

    id: str
    """The delivery identifier (``whd_<uuid>``)."""
    status: Literal["succeeded"] | str
    usage: WebhookUsage
    cost_dzd: float
    """The final cost of the request, in DZD."""
    settlement: Literal["settled"] | str
    """The status of the charge."""
    results: list[dict[str, object]]
    """The results returned by the model."""


class WebhookErrorDetails(DevupModel):
    """The type and message of the error that ended an asynchronous request."""

    type: str
    message: str


class WebhookErrorPayload(DevupModel):
    """The delivery sent when an asynchronous request ends with an error."""

    id: str
    """The delivery identifier (``whd_<uuid>``)."""
    status: Literal["failed"] | str
    usage: WebhookUsage
    """Token counts of the request (zero when it ended with an error)."""
    cost_dzd: float | None
    """The cost of the request, in DZD (``None`` when it ended with an error)."""
    settlement: Literal["failed"] | str
    """The status of the charge."""
    error: WebhookErrorDetails


WebhookEvent: TypeAlias = WebhookSuccessPayload | WebhookErrorPayload
"""A webhook delivery: the results of the request, or the details of its error."""
