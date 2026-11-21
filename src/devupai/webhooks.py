"""Webhook helpers: signature verification and typed event construction.

A webhook request carries the header ``X-DevUp-Signature: t=<unix_timestamp>,v1=<hex_signature>[,v1=<hex>]``. The
signature is HMAC-SHA-256, keyed with the UTF-8 bytes of the signing secret, over ``"<timestamp>."`` followed by the
raw request body. Several ``v1`` values may be present while a secret is rotated; one match is enough.

Always pass the raw body exactly as it was received, before any JSON parsing: a re-serialised object changes its
whitespace and key order, and the signature no longer matches.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import time
from collections.abc import Callable
from typing import TypeAlias, cast

from devupai._errors import DevupWebhookVerificationError
from devupai.types import WebhookErrorPayload, WebhookEvent, WebhookSuccessPayload
from devupai.types._base import parse_model

__all__ = [
    "Webhooks",
    "construct_webhook_event",
    "verify_webhook_signature",
]

_time: Callable[[], float] = time.time
"""The wall clock, in seconds. It is always called through the module, so tests can replace it."""

RawBody: TypeAlias = str | bytes | bytearray | memoryview
"""The raw request body: the exact text or bytes that were received."""

DEFAULT_TOLERANCE_SECONDS = 300
"""Default largest difference, in seconds, between the header timestamp and the current time."""

# The code points JavaScript's String.prototype.trim removes, so a header is split the same way by every SDK.
_TRIM_CHARS = "".join(
    chr(code)
    for code in (
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x20,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        0xFEFF,
    )
)
_REPLACEMENT_CHARACTER = chr(0xFFFD)
_DIGITS = re.compile(r"[0-9]+")
_HEX = re.compile(r"[0-9a-fA-F]+")
#: More digits than any Unix time in seconds; a longer ``t`` is outside every tolerance.
_MAX_TIMESTAMP_DIGITS = 20
_TOLERANCE_MESSAGE = "tolerance_seconds must be a non-negative number of seconds."


def _trim(value: str) -> str:
    return value.strip(_TRIM_CHARS)


def _utf8(value: str) -> bytes:
    """UTF-8 bytes of ``value``; a lone surrogate is encoded as U+FFFD, as a WHATWG ``TextEncoder`` does."""
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        replaced = "".join(_REPLACEMENT_CHARACTER if 0xD800 <= ord(char) <= 0xDFFF else char for char in value)
        return replaced.encode("utf-8")


def _valid_tolerance(value: object) -> bool:
    """Whether ``value`` is a finite, non-negative number of seconds (a ``bool`` is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and value >= 0


def _reject_constant(name: str) -> object:
    """``json.loads`` hook: ``NaN``, ``Infinity`` and ``-Infinity`` are not JSON, as in JavaScript's ``JSON.parse``."""
    raise ValueError(f"{name} is not a valid JSON value")


def _number_text(value: float) -> str:
    """A number as JavaScript prints it in a template string, for the error messages (``300``, not ``300.0``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _verify(raw_body: object, signature_header: object, secret: object, tolerance_seconds: float) -> str:
    """Runs every check in order and returns the body as text, or raises ``DevupWebhookVerificationError``."""
    if isinstance(raw_body, str):
        body_bytes = _utf8(raw_body)
        body_text = raw_body
    elif isinstance(raw_body, bytes | bytearray | memoryview):
        body_bytes = bytes(cast("bytes | bytearray | memoryview[int]", raw_body))
        # Like a WHATWG TextDecoder: a leading byte order mark is dropped and invalid bytes become U+FFFD.
        body_text = body_bytes.decode("utf-8-sig", errors="replace")
    else:
        raise DevupWebhookVerificationError(
            "malformed_header", "raw_body must be a str or a bytes-like object (bytes, bytearray or memoryview)"
        )

    if not isinstance(signature_header, str) or _trim(signature_header) == "":
        raise DevupWebhookVerificationError("malformed_header", "Missing or empty signature header")

    timestamp: int | None = None
    # A digit string too long to be a real Unix time: outside any tolerance (Node parses it as Infinity).
    timestamp_too_long = False
    signatures: list[str] = []
    for element in signature_header.split(","):
        trimmed = _trim(element)
        if not trimmed:
            continue
        equal_index = trimmed.find("=")
        if equal_index == -1:
            continue
        key = _trim(trimmed[:equal_index])
        value = _trim(trimmed[equal_index + 1 :])
        if key == "t":
            if _DIGITS.fullmatch(value):
                # The length is checked before int(), which refuses digit strings above Python's conversion limit.
                timestamp_too_long = len(value) > _MAX_TIMESTAMP_DIGITS
                timestamp = None if timestamp_too_long else int(value)
        elif key == "v1" and value:
            signatures.append(value)

    if timestamp is None and not timestamp_too_long:
        raise DevupWebhookVerificationError(
            "malformed_header", "Malformed signature header: missing or non-numeric timestamp 't'"
        )
    if not signatures:
        raise DevupWebhookVerificationError("malformed_header", "Malformed signature header: missing 'v1' signature")

    if not _valid_tolerance(tolerance_seconds):
        raise DevupWebhookVerificationError("timestamp_outside_tolerance", _TOLERANCE_MESSAGE)
    if timestamp is None:
        raise DevupWebhookVerificationError(
            "timestamp_outside_tolerance",
            f"Timestamp outside tolerance (Infinitys > {_number_text(tolerance_seconds)}s)",
        )
    now = math.floor(_time())
    diff = abs(now - timestamp)
    if diff > tolerance_seconds:
        raise DevupWebhookVerificationError(
            "timestamp_outside_tolerance",
            f"Timestamp outside tolerance ({diff}s > {_number_text(tolerance_seconds)}s)",
        )

    if not isinstance(secret, str) or _trim(secret) == "":
        raise DevupWebhookVerificationError("no_matching_signature", "No webhook signing secret provided")

    digest = hmac.new(_utf8(secret), f"{timestamp}.".encode() + body_bytes, hashlib.sha256).digest()
    for signature in signatures:
        if len(signature) % 2 != 0 or not _HEX.fullmatch(signature):
            continue
        if hmac.compare_digest(bytes.fromhex(signature), digest):
            return body_text

    raise DevupWebhookVerificationError("no_matching_signature", "No matching signature found for the provided secret")


def verify_webhook_signature(
    *,
    raw_body: RawBody,
    signature_header: str,
    secret: str,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> bool:
    """Verifies the signature and timestamp of an incoming DEVUP AI webhook request.

    ``raw_body`` must be the exact text or bytes received in the HTTP request, before any JSON parsing.

    Args:
        raw_body: The raw request body. A ``str`` is hashed as UTF-8.
        signature_header: The value of the ``X-DevUp-Signature`` header.
        secret: The webhook signing secret from the DEVUP AI Dashboard.
        tolerance_seconds: The largest allowed difference, in seconds, between the header timestamp and the
            current time.

    Returns:
        ``True`` when one ``v1`` signature matches and the timestamp is within the tolerance, ``False`` otherwise.
        It never raises.
    """
    try:
        _verify(raw_body, signature_header, secret, tolerance_seconds)
    except Exception:
        # Every rejection, and any other error while checking, is reported as False: this function never raises.
        return False
    return True


def construct_webhook_event(
    *,
    raw_body: RawBody,
    signature_header: str,
    secret: str,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> WebhookEvent:
    """Verifies an incoming DEVUP AI webhook request, then parses and returns the typed event.

    ``raw_body`` must be the exact text or bytes received in the HTTP request, before any JSON parsing.

    Args:
        raw_body: The raw request body. A ``str`` is hashed as UTF-8.
        signature_header: The value of the ``X-DevUp-Signature`` header.
        secret: The webhook signing secret from the DEVUP AI Dashboard.
        tolerance_seconds: The largest allowed difference, in seconds, between the header timestamp and the
            current time.

    Returns:
        A ``WebhookErrorPayload`` when the event ``status`` is ``"failed"``, a ``WebhookSuccessPayload`` otherwise.
        Fields that are not part of the model are kept as extras.

    Raises:
        DevupWebhookVerificationError: ``reason`` is ``malformed_header`` (the header or the body type is not
            valid), ``timestamp_outside_tolerance`` (also when ``tolerance_seconds`` is not a finite, non-negative
            number), ``no_matching_signature``, or ``invalid_json`` (the verified body is not a JSON object, or
            contains ``NaN`` or ``Infinity``).
    """
    body_text = _verify(raw_body, signature_header, secret, tolerance_seconds)
    try:
        data: object = json.loads(body_text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise DevupWebhookVerificationError(
            "invalid_json", f"Failed to parse verified webhook payload as JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise DevupWebhookVerificationError("invalid_json", "The verified webhook payload is not a JSON object.")
    fields = cast("dict[str, object]", data)
    if fields.get("status") == "failed":
        return parse_model(WebhookErrorPayload, fields)
    return parse_model(WebhookSuccessPayload, fields)


class Webhooks:
    """Webhook helpers: signature verification and typed event construction.

    Both methods are local and synchronous, on ``DevupAI`` and on ``AsyncDevupAI``. An instance needs no client.
    """

    def verify_signature(
        self,
        *,
        raw_body: RawBody,
        signature_header: str,
        secret: str,
        tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    ) -> bool:
        """Verifies the signature and timestamp of a webhook request; see ``verify_webhook_signature``."""
        return verify_webhook_signature(
            raw_body=raw_body, signature_header=signature_header, secret=secret, tolerance_seconds=tolerance_seconds
        )

    def construct_event(
        self,
        *,
        raw_body: RawBody,
        signature_header: str,
        secret: str,
        tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    ) -> WebhookEvent:
        """Verifies a webhook request and returns the typed event; see ``construct_webhook_event``."""
        return construct_webhook_event(
            raw_body=raw_body, signature_header=signature_header, secret=secret, tolerance_seconds=tolerance_seconds
        )
