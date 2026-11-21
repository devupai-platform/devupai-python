"""Checks the signature of a webhook delivery and reads its event, with the SDK's webhook helpers.

Run it with your webhook signing secret in DEVUP_WEBHOOK_SECRET:

    python examples/webhooks.py

The script signs a small sample delivery with your secret, the way DEVUP AI signs a real one, and then handles it
as a webhook endpoint would. In your server, pass the raw request body (the exact bytes received, before any JSON
parsing) and the request headers to ``handle_delivery``. No API key is needed and no request is sent.
"""

import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping

from devupai import DevupWebhookVerificationError, construct_webhook_event, verify_webhook_signature
from devupai.types import WebhookErrorPayload, WebhookEvent

SIGNATURE_HEADER = "X-DevUp-Signature"


def require_secret() -> str:
    """The signing secret from DEVUP_WEBHOOK_SECRET. Exits with a message when it is not set."""
    secret = os.environ.get("DEVUP_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise SystemExit("Set the DEVUP_WEBHOOK_SECRET environment variable to your webhook signing secret.")
    return secret


def signature_header(headers: Mapping[str, str]) -> str:
    """The value of the signature header (header names are compared without regard to case)."""
    for name, value in headers.items():
        if name.lower() == SIGNATURE_HEADER.lower():
            return value
    return ""


def handle_delivery(raw_body: bytes, headers: Mapping[str, str], secret: str) -> WebhookEvent:
    """Verifies a delivery and returns its typed event.

    Raises ``DevupWebhookVerificationError`` when the signature does not match, the timestamp is outside the
    tolerance (too old, or too far in the future), or the body is not a JSON object. Answer such a request with
    HTTP 400.
    """
    return construct_webhook_event(raw_body=raw_body, signature_header=signature_header(headers), secret=secret)


def sample_delivery(secret: str) -> tuple[bytes, dict[str, str]]:
    """A sample delivery body and its headers, signed with ``secret``.

    The signature is HMAC-SHA-256, keyed with the secret, over ``"<timestamp>."`` followed by the raw body.
    """
    body = json.dumps(
        {
            "id": "whd_00000000-0000-0000-0000-000000000000",
            "status": "succeeded",
            "usage": {"prompt_tokens": 12, "completion_tokens": 30},
            "results": [{"text": "Hello from a webhook."}],
        }
    ).encode("utf-8")
    timestamp = int(time.time())
    digest = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return body, {"Content-Type": "application/json", SIGNATURE_HEADER: f"t={timestamp},v1={digest}"}


def main() -> None:
    secret = require_secret()
    raw_body, headers = sample_delivery(secret)

    # A check that returns True or False and never raises.
    valid = verify_webhook_signature(raw_body=raw_body, signature_header=signature_header(headers), secret=secret)
    print(f"Signature valid: {valid}")

    try:
        event = handle_delivery(raw_body, headers, secret)
    except DevupWebhookVerificationError as err:
        raise SystemExit(f"Delivery rejected ({err.reason}).") from err

    if isinstance(event, WebhookErrorPayload):
        print(f"Event {event.id}: {event.status}: {event.error.message}")
    else:
        print(f"Event {event.id}: {event.status}, {len(event.results)} result(s)")
        for result in event.results:
            print(json.dumps(result))


if __name__ == "__main__":
    main()
