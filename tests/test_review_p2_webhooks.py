"""Webhook fixes from the P2 review: R2-3 (a very long timestamp), R2-9 (a tolerance that is not a valid number of
seconds) and R2-10 (``NaN`` / ``Infinity`` in a verified body).

Every rejection is a ``DevupWebhookVerificationError`` from ``construct_webhook_event`` and ``False`` from
``verify_webhook_signature``; never a raw ``ValueError`` or ``TypeError``, and never a skipped check.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math

import pytest

import devupai.webhooks as webhooks_module
from devupai import DevupWebhookVerificationError, Webhooks, construct_webhook_event, verify_webhook_signature

SECRET = "whsec_review_secret_0123456789abcdef"
WALL_CLOCK = 1_800_000_000.25
NOW = math.floor(WALL_CLOCK)
BODY = json.dumps({"id": "whd_review", "status": "succeeded", "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
TOLERANCE_MESSAGE = "tolerance_seconds must be a non-negative number of seconds."


def signature(payload: str | bytes, timestamp: int | str) -> str:
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hmac.new(SECRET.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


def header(payload: str | bytes, timestamp: int | str = NOW) -> str:
    return f"t={timestamp},v1={signature(payload, timestamp)}"


@pytest.fixture(autouse=True)
def frozen_webhook_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhooks_module, "_time", lambda: WALL_CLOCK)


def construct_error(**kwargs: object) -> DevupWebhookVerificationError:
    with pytest.raises(DevupWebhookVerificationError) as info:
        construct_webhook_event(**kwargs)  # type: ignore[arg-type]
    return info.value


# --- R2-3: a timestamp with thousands of digits ----------------------------------------------------------------------

LONG_TIMESTAMP = "9" * 5000


def test_a_5000_digit_timestamp_is_rejected_by_verify() -> None:
    assert (
        verify_webhook_signature(raw_body=BODY, signature_header=header(BODY, LONG_TIMESTAMP), secret=SECRET) is False
    )


def test_a_5000_digit_timestamp_is_outside_the_tolerance_for_construct() -> None:
    error = construct_error(raw_body=BODY, signature_header=header(BODY, LONG_TIMESTAMP), secret=SECRET)
    assert error.reason == "timestamp_outside_tolerance"
    assert "Timestamp outside tolerance" in str(error)


def test_a_5000_digit_timestamp_with_leading_zeros_is_rejected_too() -> None:
    # Leading zeros keep the value small, but the digit string is still refused without an int conversion error.
    zeros = "0" * 4990 + str(NOW)
    error = construct_error(raw_body=BODY, signature_header=header(BODY, zeros), secret=SECRET)
    assert error.reason == "timestamp_outside_tolerance"


def test_a_long_timestamp_without_v1_is_still_a_malformed_header() -> None:
    # The checks keep their order: a missing v1 is reported before the tolerance.
    error = construct_error(raw_body=BODY, signature_header=f"t={LONG_TIMESTAMP}", secret=SECRET)
    assert error.reason == "malformed_header"


def test_the_client_helper_rejects_a_long_timestamp_the_same_way() -> None:
    webhooks = Webhooks()
    assert not webhooks.verify_signature(raw_body=BODY, signature_header=header(BODY, LONG_TIMESTAMP), secret=SECRET)
    with pytest.raises(DevupWebhookVerificationError) as info:
        webhooks.construct_event(raw_body=BODY, signature_header=header(BODY, LONG_TIMESTAMP), secret=SECRET)
    assert info.value.reason == "timestamp_outside_tolerance"


# --- R2-9: a tolerance that is not a finite, non-negative number --------------------------------------------------

BAD_TOLERANCES = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="minus-inf"),
    pytest.param(-1, id="negative-int"),
    pytest.param(-0.5, id="negative-float"),
    pytest.param("300", id="str"),
    pytest.param(None, id="none"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
]


@pytest.mark.parametrize("tolerance", BAD_TOLERANCES)
def test_verify_returns_false_for_a_bad_tolerance(tolerance: object) -> None:
    # The timestamp is current and the signature is valid: only the tolerance is wrong.
    assert (
        verify_webhook_signature(
            raw_body=BODY,
            signature_header=header(BODY),
            secret=SECRET,
            tolerance_seconds=tolerance,  # type: ignore[arg-type]
        )
        is False
    )


@pytest.mark.parametrize("tolerance", BAD_TOLERANCES)
def test_construct_raises_outside_tolerance_for_a_bad_tolerance(tolerance: object) -> None:
    error = construct_error(raw_body=BODY, signature_header=header(BODY), secret=SECRET, tolerance_seconds=tolerance)
    assert error.reason == "timestamp_outside_tolerance"
    assert str(error) == TOLERANCE_MESSAGE


def test_nan_tolerance_never_skips_the_timestamp_check() -> None:
    old = header(BODY, 1)
    assert not verify_webhook_signature(raw_body=BODY, signature_header=old, secret=SECRET, tolerance_seconds=math.nan)


@pytest.mark.parametrize("tolerance", [pytest.param(0, id="zero"), pytest.param(0.0, id="zero-float")])
def test_a_zero_tolerance_is_valid(tolerance: float) -> None:
    assert verify_webhook_signature(
        raw_body=BODY, signature_header=header(BODY), secret=SECRET, tolerance_seconds=tolerance
    )
    error = construct_error(
        raw_body=BODY, signature_header=header(BODY, NOW - 1), secret=SECRET, tolerance_seconds=tolerance
    )
    assert error.reason == "timestamp_outside_tolerance"
    assert "Timestamp outside tolerance" in str(error)


def test_a_float_tolerance_is_valid() -> None:
    assert verify_webhook_signature(
        raw_body=BODY, signature_header=header(BODY, NOW - 10), secret=SECRET, tolerance_seconds=10.5
    )


# --- R2-10: NaN and Infinity are not JSON ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param('{"id": "whd_review", "status": "succeeded", "x": NaN}', id="nan"),
        pytest.param('{"id": "whd_review", "status": "succeeded", "x": Infinity}', id="infinity"),
        pytest.param('{"id": "whd_review", "status": "succeeded", "x": -Infinity}', id="minus-infinity"),
        pytest.param('{"id": "whd_review", "status": "succeeded", "x": [1, {"y": NaN}]}', id="nested-nan"),
    ],
)
def test_construct_raises_invalid_json_for_nan_or_infinity(body: str) -> None:
    error = construct_error(raw_body=body, signature_header=header(body), secret=SECRET)
    assert error.reason == "invalid_json"
    assert "The verified webhook payload is not valid JSON" in str(error)
    assert error.__cause__ is not None


def test_verify_still_checks_only_the_signature_of_such_a_body() -> None:
    # verify_webhook_signature does not parse the body, as in Node.
    body = '{"x": NaN}'
    assert verify_webhook_signature(raw_body=body, signature_header=header(body), secret=SECRET) is True


def test_the_strings_nan_and_infinity_are_ordinary_json() -> None:
    body = json.dumps({"id": "whd_review", "status": "succeeded", "x": "NaN", "y": "Infinity"})
    event = construct_webhook_event(raw_body=body, signature_header=header(body), secret=SECRET)
    assert event.to_dict()["x"] == "NaN"
