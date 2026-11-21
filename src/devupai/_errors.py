"""Errors raised by the DEVUP AI API client, and the parsers for the response headers they carry."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Literal, TypeAlias, cast

#: Wall clock in epoch seconds. Read at call time, so tests can replace it.
_time: Callable[[], float] = time.time

_RETRY_AFTER_MS = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)
_RETRY_AFTER_SECONDS = re.compile(r"\d+(\.\d+)?", re.ASCII)


def _get_header(headers: Mapping[str, str], name: str) -> str | None:
    """Returns the value of header ``name`` (lower case), looked up without regard to case."""
    value = headers.get(name)
    if value is not None:
        return value
    for key, item in headers.items():
        if key.lower() == name:
            return item
    return None


def _to_optional_str(value: object) -> str | None:
    """A string stays as is, a finite number becomes a string, anything else becomes ``None``."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(int(value)) if value.is_integer() else str(value)
    return None


def parse_retry_after_ms(headers: Mapping[str, str], now: float | None = None) -> float | None:
    """Reads the wait requested by the API before a new attempt, in milliseconds.

    Uses ``retry-after-ms`` first, then ``Retry-After`` (seconds or an HTTP date). Header names are matched without
    regard to case. ``now`` is the current time in epoch seconds and defaults to the wall clock. Returns ``None`` when
    neither header contains a usable value.
    """
    ms = _get_header(headers, "retry-after-ms")
    if ms is not None:
        trimmed_ms = ms.strip()
        if trimmed_ms and _RETRY_AFTER_MS.fullmatch(trimmed_ms):
            value = float(trimmed_ms)
            if math.isfinite(value) and value >= 0:
                return value

    retry_after = _get_header(headers, "retry-after")
    if retry_after is None:
        return None
    trimmed = retry_after.strip()
    if not trimmed:
        return None
    if _RETRY_AFTER_SECONDS.fullmatch(trimmed):
        return float(trimmed) * 1000.0
    try:
        date = parsedate_to_datetime(trimmed)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if date.tzinfo is None:
        date = date.replace(tzinfo=UTC)
    try:
        target = date.timestamp()
    except (ValueError, OverflowError, OSError):
        return None
    current = _time() if now is None else now
    return max(0.0, (target - current) * 1000.0)


def parse_warnings(headers: Mapping[str, str]) -> list[str]:
    """Parses the ``X-DEVUP-Warnings`` header: informational notices about how the request was handled.

    Returns an empty list when the header is absent or empty.
    """
    raw = _get_header(headers, "x-devup-warnings")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


class DevupAPIError(Exception):
    """Error raised by the DEVUP AI API client.

    ``status`` is the HTTP status, or ``0`` when no response was received (connection error or timeout). The error
    that led to this one, when there is one, is chained as ``__cause__``.
    """

    status: int
    """HTTP status, or ``0`` when no response was received."""
    message: str
    """Error message; ``str(error)`` gives the same text."""
    type: str | None
    """Error type from the response body, when present."""
    code: str | None
    """Error code from the response body, when present."""
    param: str | None
    """Request parameter the error refers to, when present."""
    request_id: str | None
    """Value of the ``x-request-id`` response header."""
    client_request_id: str | None
    """Value of the ``x-client-request-id`` response header."""
    retry_after: str | None
    """Raw value of the ``Retry-After`` response header."""
    retry_after_ms: float | None
    """Wait requested by the API before a new attempt, in milliseconds."""
    warnings: list[str]
    """Informational notices from the ``X-DEVUP-Warnings`` response header."""
    headers: dict[str, str]
    """Response headers, with lower-cased names."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: object = None,
        code: object = None,
        param: object = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Creates the error from an HTTP status, a message, the fields of the ``error`` object and the headers."""
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = _to_optional_str(error_type)
        self.code = _to_optional_str(code)
        self.param = param if isinstance(param, str) else None
        self.headers = {} if headers is None else {key.lower(): value for key, value in headers.items()}
        self.request_id = self.headers.get("x-request-id") or None
        self.client_request_id = self.headers.get("x-client-request-id") or None
        self.retry_after = self.headers.get("retry-after") or None
        self.retry_after_ms = None if headers is None else parse_retry_after_ms(self.headers)
        self.warnings = parse_warnings(self.headers)


class BadRequestError(DevupAPIError):
    """HTTP 400: the request is not valid as sent. ``code`` and ``param`` name the reason and the field."""


class AuthenticationError(DevupAPIError):
    """HTTP 401: the API key is missing or not valid."""


class PaymentRequiredError(DevupAPIError):
    """HTTP 402: the account balance (``code`` ``insufficient_balance``), or the budget of a scoped credential
    (``budget_exhausted``), does not cover the request.
    """


class PermissionDeniedError(DevupAPIError):
    """HTTP 403: the API key is not permitted to perform this request."""


class NotFoundError(DevupAPIError):
    """HTTP 404: the requested resource does not exist."""


class ConflictError(DevupAPIError):
    """HTTP 409: the outcome of this request is not confirmed yet (``request_outcome_pending``).

    Do not send it again: contact support with ``request_id``.
    """


class PayloadTooLargeError(DevupAPIError):
    """HTTP 413: the request body is larger than the endpoint takes. Send a smaller body."""


class UnsupportedMediaTypeError(DevupAPIError):
    """HTTP 415: the model does not accept the content type of the request.

    Check the ``Content-Type`` header and the body format against the API reference.
    """


class UnprocessableEntityError(DevupAPIError):
    """HTTP 422: the request is well formed, but a value in it cannot be processed. ``code`` and ``param`` name it."""


class RateLimitError(DevupAPIError):
    """HTTP 429: too many requests. ``retry_after_ms`` gives the wait, when present."""


class InternalServerError(DevupAPIError):
    """HTTP 500 or above.

    After a 500 or 502 (``server_error``, ``DEVUP_COMPUTE_ERROR``) the request may already have been processed and
    billed: check its outcome before sending it again.
    """


class APIConnectionError(DevupAPIError):
    """No response was received from the API (``status`` is ``0``)."""


class APITimeoutError(APIConnectionError):
    """The request did not complete within its timeout (``status`` is ``0``)."""


_STATUS_CLASSES: dict[int, type[DevupAPIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: PaymentRequiredError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    413: PayloadTooLargeError,
    415: UnsupportedMediaTypeError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def make_api_error(
    status: int,
    message: str,
    *,
    error_type: object = None,
    code: object = None,
    param: object = None,
    headers: Mapping[str, str] | None = None,
) -> DevupAPIError:
    """Creates the error for an HTTP status, using the matching subclass of :class:`DevupAPIError`.

    Statuses without a subclass of their own give a plain :class:`DevupAPIError`. It never returns an
    :class:`APIConnectionError`.
    """
    cls = _STATUS_CLASSES.get(status)
    if cls is None:
        cls = InternalServerError if status >= 500 else DevupAPIError
    return cls(status, message, error_type=error_type, code=code, param=param, headers=headers)


def _as_record(value: object) -> dict[str, object] | None:
    """Returns ``value`` as a JSON object, or ``None`` when it is not one."""
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def error_from_body(status: int, text: str, headers: Mapping[str, str] | None) -> DevupAPIError:
    """Creates the error for a response that is not successful, from its status, body text and headers.

    The message is ``error.message`` when the body is a JSON ``error`` object with a non-empty message, the ``error``
    string when the body contains one, and the body text otherwise. A blank message becomes ``HTTP <status>``.
    """
    message = text
    error_type: str | None = None
    code: str | int | float | None = None
    param: str | None = None
    if text:
        try:
            parsed: object = json.loads(text)
        except (ValueError, RecursionError):
            parsed = None
        body = _as_record(parsed)
        if body is not None:
            error = body.get("error")
            fields = _as_record(error)
            if fields is not None:
                error_message = fields.get("message")
                if isinstance(error_message, str) and error_message:
                    message = error_message
                raw_type = fields.get("type")
                error_type = raw_type if isinstance(raw_type, str) else None
                raw_code = fields.get("code")
                if isinstance(raw_code, str | int | float) and not isinstance(raw_code, bool):
                    code = raw_code
                raw_param = fields.get("param")
                param = raw_param if isinstance(raw_param, str) else None
            elif isinstance(error, str) and error:
                message = error
    if not message.strip():
        message = f"HTTP {status}"
    return make_api_error(status, message, error_type=error_type, code=code, param=param, headers=headers)


DevupWebhookVerificationErrorReason: TypeAlias = Literal[
    "malformed_header",
    "timestamp_outside_tolerance",
    "no_matching_signature",
    "invalid_json",
]
"""Reasons a webhook signature check or payload construction can be rejected."""


class DevupWebhookVerificationError(Exception):
    """Error raised when a webhook payload cannot be verified or parsed.

    ``reason`` names the check that did not pass. The error that led to this one, when there is one, is chained as
    ``__cause__``.
    """

    reason: DevupWebhookVerificationErrorReason
    """The check that did not pass."""

    def __init__(self, reason: DevupWebhookVerificationErrorReason, message: str) -> None:
        """Creates the error from its reason and a message."""
        super().__init__(message)
        self.reason = reason
