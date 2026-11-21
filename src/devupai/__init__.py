"""The official Python client for the DEVUP AI API.

Create a client and call the API::

    from devupai import DevupAI

    client = DevupAI()  # reads the key from DEVUP_API_KEY, then DEVUPAI_API_KEY

``AsyncDevupAI`` offers the same options for ``asyncio`` code. ``create(stream=True)`` returns a ``Stream`` (an
``AsyncStream`` on the async client), and a binary result is a ``BinaryResponse`` (``AsyncBinaryResponse``).

An answer or a connection problem raises a subclass of ``DevupAPIError`` (``APIConnectionError`` when no response
was received); a webhook that cannot be verified raises ``DevupWebhookVerificationError``. An option or header value
that is not valid raises ``TypeError`` or ``ValueError`` before anything is sent, and a closed client raises
``RuntimeError``.

Request parameter and response types live in ``devupai.types``. Webhook signatures are checked with
``client.webhooks``, or with ``verify_webhook_signature`` and ``construct_webhook_event`` without a client.
"""

from devupai import types
from devupai._client import AsyncDevupAI, DebugEvent, DevupAI, RequestOptions
from devupai._errors import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    DevupAPIError,
    DevupWebhookVerificationError,
    InternalServerError,
    NotFoundError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    UnsupportedMediaTypeError,
)
from devupai._response import APIResponse, AsyncBinaryResponse, BinaryResponse
from devupai._streaming import AsyncStream, Stream
from devupai._version import __version__
from devupai.webhooks import Webhooks, construct_webhook_event, verify_webhook_signature

VERSION: str = __version__
"""The version of this package, the same value as ``__version__``."""

__all__ = [
    # Clients and responses.
    "DevupAI",
    "AsyncDevupAI",
    "APIResponse",
    "DebugEvent",
    "RequestOptions",
    # Streams and binary results.
    "Stream",
    "AsyncStream",
    "BinaryResponse",
    "AsyncBinaryResponse",
    # Version.
    "VERSION",
    "__version__",
    # Errors.
    "DevupAPIError",
    "BadRequestError",
    "AuthenticationError",
    "PaymentRequiredError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "PayloadTooLargeError",
    "UnsupportedMediaTypeError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "DevupWebhookVerificationError",
    # Webhooks.
    "Webhooks",
    "verify_webhook_signature",
    "construct_webhook_event",
    # Request parameter and response types.
    "types",
]
