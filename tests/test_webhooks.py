"""Webhook signature checks and typed events: a twin of the Node SDK's 23 webhook cases, plus Python cases.

The clock the webhook module reads is ``devupai.webhooks._time``; every test here patches it (the shared
``fake_clock`` fixture does not). Expected signatures are computed in the test with ``hmac`` and ``hashlib``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math

import anyio
import pytest
from helpers.mock_transport import TEST_KEY

import devupai.webhooks as webhooks_module
from devupai import (
    AsyncDevupAI,
    DevupAI,
    DevupWebhookVerificationError,
    Webhooks,
    construct_webhook_event,
    verify_webhook_signature,
)
from devupai.types import WebhookErrorDetails, WebhookErrorPayload, WebhookSuccessPayload, WebhookUsage

TEST_SECRET = "whsec_test_secret_1234567890abcdef"
OLD_SECRET = "whsec_old_secret_abcdef1234567890"
THIRD_SECRET = "whsec_third_secret_0011223344556677"

WALL_CLOCK = 1_800_000_000.75
"""The patched wall clock. It has a fraction, so a check that does not floor the clock is seen."""
NOW = math.floor(WALL_CLOCK)

DELIVERY_ID = "whd_9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"

SAMPLE_SUCCESS_PAYLOAD = json.dumps(
    {
        "id": DELIVERY_ID,
        "status": "succeeded",
        "usage": {"prompt_tokens": 10, "completion_tokens": 50},
        "cost_dzd": 1.5,
        "settlement": "settled",
        "results": [{"generated_text": "Inference response text."}],
    }
)

SAMPLE_ERROR_PAYLOAD = json.dumps(
    {
        "id": DELIVERY_ID,
        "status": "failed",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "cost_dzd": None,
        "settlement": "failed",
        "error": {"type": "inference_failed", "message": "The inference request timed out."},
    }
)


def compute_signature(payload: str | bytes, secret: str, timestamp: int) -> str:
    """HMAC-SHA-256 over ``"<timestamp>." + body`` with the UTF-8 secret, as lowercase hex."""
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


def header_for(payload: str | bytes, secret: str = TEST_SECRET, timestamp: int = NOW) -> str:
    """A ``t=…,v1=…`` header with one signature."""
    return f"t={timestamp},v1={compute_signature(payload, secret, timestamp)}"


@pytest.fixture(autouse=True)
def frozen_webhook_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the clock of the webhook module to ``WALL_CLOCK``."""
    monkeypatch.setattr(webhooks_module, "_time", lambda: WALL_CLOCK)


def construct_error(**kwargs: object) -> DevupWebhookVerificationError:
    """Runs ``construct_webhook_event`` and returns the verification error it raises."""
    with pytest.raises(DevupWebhookVerificationError) as info:
        construct_webhook_event(**kwargs)  # type: ignore[arg-type]
    return info.value


# --- verifySignature (Node: 14 cases) ------------------------------------------------------------------------------


def test_verify_returns_true_for_a_valid_single_signature_with_string_body() -> None:
    # Node: "returns true for a valid single signature with string body"
    assert (
        verify_webhook_signature(
            raw_body=SAMPLE_SUCCESS_PAYLOAD,
            signature_header=header_for(SAMPLE_SUCCESS_PAYLOAD),
            secret=TEST_SECRET,
        )
        is True
    )


def test_verify_returns_true_for_a_valid_single_signature_with_bytes_body() -> None:
    # Node: "returns true for a valid single signature with Uint8Array body"
    body = SAMPLE_SUCCESS_PAYLOAD.encode("utf-8")
    assert verify_webhook_signature(raw_body=body, signature_header=header_for(body), secret=TEST_SECRET) is True


def test_verify_returns_true_when_the_second_of_two_v1_signatures_matches() -> None:
    # Node: "returns true when the SECOND of two v1 signatures matches (rotation window)"
    old = compute_signature(SAMPLE_SUCCESS_PAYLOAD, OLD_SECRET, NOW)
    new = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)
    header = f"t={NOW},v1={old},v1={new}"
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)


def test_verify_returns_true_when_the_first_of_two_v1_signatures_matches() -> None:
    # Node: "returns true when the FIRST of two v1 signatures matches (rotation window)"
    new = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)
    old = compute_signature(SAMPLE_SUCCESS_PAYLOAD, OLD_SECRET, NOW)
    header = f"t={NOW},v1={new},v1={old}"
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)


def test_verify_returns_false_for_a_wrong_secret() -> None:
    # Node: "returns false for a wrong secret"
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, secret="wrong_secret")
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is False
    )


def test_verify_returns_false_for_a_tampered_body() -> None:
    # Node: "returns false for a tampered body"
    header = header_for(SAMPLE_SUCCESS_PAYLOAD)
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD + " ", signature_header=header, secret=TEST_SECRET)
        is False
    )


def test_verify_returns_false_for_malformed_header_missing_timestamp() -> None:
    # Node: "returns false for malformed header: missing timestamp t"
    header = f"v1={compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)}"
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is False
    )


def test_verify_returns_false_for_malformed_header_non_numeric_timestamp() -> None:
    # Node: "returns false for malformed header: non-numeric timestamp t"
    header = f"t=invalid_timestamp,v1={compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)}"
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is False
    )


def test_verify_returns_false_for_malformed_header_missing_v1() -> None:
    # Node: "returns false for malformed header: missing v1 signature"
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=f"t={NOW}", secret=TEST_SECRET)
        is False
    )


def test_verify_returns_false_for_a_garbage_header() -> None:
    # Node: "returns false for malformed header: completely garbage header string"
    assert (
        verify_webhook_signature(
            raw_body=SAMPLE_SUCCESS_PAYLOAD,
            signature_header="random-unstructured-header-value",
            secret=TEST_SECRET,
        )
        is False
    )


def test_verify_returns_false_when_timestamp_is_older_than_default_tolerance() -> None:
    # Node: "returns false when timestamp is older than default tolerance (300s)"
    expired = NOW - 301
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=expired)
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is False
    )


def test_verify_returns_false_when_timestamp_is_in_future_beyond_default_tolerance() -> None:
    # Node: "returns false when timestamp is in future beyond default tolerance (300s)"
    future = NOW + 301
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=future)
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is False
    )


def test_verify_honours_custom_tolerance_seconds() -> None:
    # Node: "honours custom toleranceSeconds"
    past = NOW - 400
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=past)
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is False
    )
    assert (
        verify_webhook_signature(
            raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET, tolerance_seconds=600
        )
        is True
    )


def test_verify_tolerates_whitespace_and_unknown_header_parameters() -> None:
    # Node: "tolerates whitespace and unknown header parameters"
    signature = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)
    header = f"  t = {NOW} ,  v0 = legacy_val , v1 = {signature}  "
    assert (
        verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
        is True
    )


# --- verifySignature: Python cases ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_body",
    [
        pytest.param(bytes, id="bytes"),
        pytest.param(bytearray, id="bytearray"),
        pytest.param(memoryview, id="memoryview"),
    ],
)
def test_verify_accepts_every_bytes_like_body(make_body: type[bytes] | type[bytearray] | type[memoryview]) -> None:
    raw = SAMPLE_SUCCESS_PAYLOAD.encode("utf-8")
    body = make_body(raw)
    assert verify_webhook_signature(raw_body=body, signature_header=header_for(raw), secret=TEST_SECRET) is True


def test_verify_hashes_a_str_body_as_utf8() -> None:
    body = json.dumps({"id": DELIVERY_ID, "status": "succeeded", "note": "café ✓"}, ensure_ascii=False)
    header = f"t={NOW},v1={compute_signature(body.encode('utf-8'), TEST_SECRET, NOW)}"
    assert verify_webhook_signature(raw_body=body, signature_header=header, secret=TEST_SECRET) is True


def test_verify_accepts_an_uppercase_hex_signature() -> None:
    signature = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW).upper()
    header = f"t={NOW},v1={signature}"
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)


def test_verify_tries_every_v1_value_during_a_rotation() -> None:
    first = compute_signature(SAMPLE_SUCCESS_PAYLOAD, OLD_SECRET, NOW)
    second = compute_signature(SAMPLE_SUCCESS_PAYLOAD, THIRD_SECRET, NOW)
    third = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)
    header = f"t={NOW},v1={first},v1={second},v1={third}"
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
    assert not verify_webhook_signature(
        raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=f"t={NOW},v1={first},v1={second}", secret=TEST_SECRET
    )


def test_verify_skips_a_v1_value_of_odd_length() -> None:
    valid = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)
    # An odd-length value is skipped and the next one is tried.
    header = f"t={NOW},v1=abc,v1={valid}"
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
    # The valid signature with one extra hex digit has odd length: skipped, never matched on a prefix.
    odd = f"t={NOW},v1={valid}0"
    assert not verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=odd, secret=TEST_SECRET)


def test_verify_skips_a_v1_value_that_is_not_hex() -> None:
    valid = compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)
    header = f"t={NOW},v1={'zz' * 32},v1={valid}"
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)


@pytest.mark.parametrize("offset", [pytest.param(-300, id="past"), pytest.param(300, id="future")])
def test_verify_accepts_a_timestamp_exactly_at_the_tolerance(offset: int) -> None:
    # The clock is floored before the comparison, and only a difference above the tolerance is rejected.
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=NOW + offset)
    assert verify_webhook_signature(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)


def test_verify_honours_custom_tolerance_seconds_in_the_future() -> None:
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=NOW + 400)
    assert not verify_webhook_signature(
        raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET
    )
    assert verify_webhook_signature(
        raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET, tolerance_seconds=600
    )


@pytest.mark.parametrize(
    ("raw_body", "signature_header", "secret"),
    [
        pytest.param(123, f"t={NOW},v1=00", TEST_SECRET, id="body-not-bytes-or-str"),
        pytest.param(SAMPLE_SUCCESS_PAYLOAD, None, TEST_SECRET, id="header-not-str"),
        pytest.param(SAMPLE_SUCCESS_PAYLOAD, "   ", TEST_SECRET, id="header-blank"),
        pytest.param(SAMPLE_SUCCESS_PAYLOAD, header_for(SAMPLE_SUCCESS_PAYLOAD), None, id="secret-not-str"),
        pytest.param(SAMPLE_SUCCESS_PAYLOAD, header_for(SAMPLE_SUCCESS_PAYLOAD), "  ", id="secret-blank"),
    ],
)
def test_verify_never_raises_and_returns_false(raw_body: object, signature_header: object, secret: object) -> None:
    result = verify_webhook_signature(
        raw_body=raw_body,  # type: ignore[arg-type]
        signature_header=signature_header,  # type: ignore[arg-type]
        secret=secret,  # type: ignore[arg-type]
    )
    assert result is False


# --- constructEvent (Node: 7 cases) --------------------------------------------------------------------------------


def test_construct_returns_a_success_payload_on_successful_verification() -> None:
    # Node: "returns parsed WebhookSuccessPayload on successful verification"
    event = construct_webhook_event(
        raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header_for(SAMPLE_SUCCESS_PAYLOAD), secret=TEST_SECRET
    )
    assert isinstance(event, WebhookSuccessPayload)
    assert event.id == DELIVERY_ID
    assert event.status == "succeeded"
    assert event.cost_dzd == 1.5
    assert event.settlement == "settled"
    assert event.results[0]["generated_text"] == "Inference response text."


def test_construct_returns_an_error_payload_on_successful_verification_of_a_failure_payload() -> None:
    # Node: "returns parsed WebhookErrorPayload on successful verification of failure payload"
    event = construct_webhook_event(
        raw_body=SAMPLE_ERROR_PAYLOAD, signature_header=header_for(SAMPLE_ERROR_PAYLOAD), secret=TEST_SECRET
    )
    assert isinstance(event, WebhookErrorPayload)
    assert event.id == DELIVERY_ID
    assert event.status == "failed"
    assert event.cost_dzd is None
    assert event.settlement == "failed"
    assert isinstance(event.error, WebhookErrorDetails)
    assert event.error.type == "inference_failed"
    assert event.error.message == "The inference request timed out."


def test_construct_raises_malformed_header_for_a_missing_timestamp() -> None:
    # Node: "throws DevupWebhookVerificationError with 'malformed_header' for missing timestamp"
    header = f"v1={compute_signature(SAMPLE_SUCCESS_PAYLOAD, TEST_SECRET, NOW)}"
    error = construct_error(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
    assert error.reason == "malformed_header"
    assert "timestamp" in str(error)


def test_construct_raises_malformed_header_for_a_missing_v1() -> None:
    # Node: "throws DevupWebhookVerificationError with 'malformed_header' for missing v1"
    error = construct_error(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=f"t={NOW}", secret=TEST_SECRET)
    assert error.reason == "malformed_header"
    assert "v1" in str(error)


def test_construct_raises_timestamp_outside_tolerance_for_an_expired_timestamp() -> None:
    # Node: "throws DevupWebhookVerificationError with 'timestamp_outside_tolerance' for expired timestamp"
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=NOW - 350)
    error = construct_error(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
    assert error.reason == "timestamp_outside_tolerance"
    assert "Timestamp outside tolerance" in str(error)


def test_construct_raises_no_matching_signature_for_a_wrong_signature() -> None:
    # Node: "throws DevupWebhookVerificationError with 'no_matching_signature' for wrong signature"
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, secret="wrong_secret")
    error = construct_error(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
    assert error.reason == "no_matching_signature"
    assert "No matching signature" in str(error)


def test_construct_raises_invalid_json_for_a_valid_signature_over_malformed_json() -> None:
    # Node: "throws DevupWebhookVerificationError with 'invalid_json' for valid signature over malformed JSON"
    body = "{ this is not valid JSON }"
    error = construct_error(raw_body=body, signature_header=header_for(body), secret=TEST_SECRET)
    assert error.reason == "invalid_json"
    assert "Failed to parse verified webhook payload as JSON" in str(error)
    assert error.__cause__ is not None


# --- constructEvent: Python cases ----------------------------------------------------------------------------------


def test_construct_raises_timestamp_outside_tolerance_for_a_future_timestamp() -> None:
    header = header_for(SAMPLE_SUCCESS_PAYLOAD, timestamp=NOW + 350)
    error = construct_error(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET)
    assert error.reason == "timestamp_outside_tolerance"
    assert "Timestamp outside tolerance" in str(error)


def test_construct_raises_no_matching_signature_for_a_blank_secret() -> None:
    header = header_for(SAMPLE_SUCCESS_PAYLOAD)
    error = construct_error(raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret="   ")
    assert error.reason == "no_matching_signature"


def test_construct_raises_malformed_header_for_a_body_of_another_type() -> None:
    error = construct_error(raw_body=123, signature_header=header_for(SAMPLE_SUCCESS_PAYLOAD), secret=TEST_SECRET)
    assert error.reason == "malformed_header"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(SAMPLE_SUCCESS_PAYLOAD, WebhookSuccessPayload, id="succeeded"),
        pytest.param(SAMPLE_ERROR_PAYLOAD, WebhookErrorPayload, id="failed"),
    ],
)
def test_construct_types_the_nested_usage(payload: str, expected: type[object]) -> None:
    event = construct_webhook_event(raw_body=payload, signature_header=header_for(payload), secret=TEST_SECRET)
    assert isinstance(event, expected)
    assert isinstance(event.usage, WebhookUsage)
    usage = json.loads(payload)["usage"]
    assert event.usage.prompt_tokens == usage["prompt_tokens"]
    assert event.usage.completion_tokens == usage["completion_tokens"]


def test_construct_picks_the_model_by_status() -> None:
    # Only status "failed" gives the error payload; any other status gives the success payload.
    other = json.dumps({"id": DELIVERY_ID, "status": "queued", "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    event = construct_webhook_event(raw_body=other, signature_header=header_for(other), secret=TEST_SECRET)
    assert isinstance(event, WebhookSuccessPayload)
    assert event.status == "queued"


def test_construct_keeps_unknown_keys() -> None:
    body = json.dumps(
        {
            "id": DELIVERY_ID,
            "status": "succeeded",
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "new_counter": 3},
            "cost_dzd": 1.5,
            "settlement": "settled",
            "results": [],
            "future_field": {"nested": [1, 2]},
        }
    )
    event = construct_webhook_event(raw_body=body, signature_header=header_for(body), secret=TEST_SECRET)
    assert isinstance(event, WebhookSuccessPayload)
    assert event.model_extra is not None
    assert event.model_extra["future_field"] == {"nested": [1, 2]}
    assert event.usage.model_extra is not None
    assert event.usage.model_extra["new_counter"] == 3
    dumped = event.to_dict()
    assert dumped["future_field"] == {"nested": [1, 2]}
    assert dumped["usage"] == {"prompt_tokens": 10, "completion_tokens": 50, "new_counter": 3}


def test_construct_never_rejects_a_body_that_does_not_match_the_model() -> None:
    # A verified body is never rejected for its shape: the lenient construct keeps what was sent.
    body = json.dumps({"id": 7, "status": "succeeded", "usage": "none", "results": "x"})
    event = construct_webhook_event(raw_body=body, signature_header=header_for(body), secret=TEST_SECRET)
    assert isinstance(event, WebhookSuccessPayload)
    assert event.to_dict()["id"] == 7


@pytest.mark.parametrize(
    "make_body",
    [
        pytest.param(bytes, id="bytes"),
        pytest.param(bytearray, id="bytearray"),
        pytest.param(memoryview, id="memoryview"),
    ],
)
def test_construct_accepts_every_bytes_like_body(make_body: type[bytes] | type[bytearray] | type[memoryview]) -> None:
    raw = SAMPLE_ERROR_PAYLOAD.encode("utf-8")
    event = construct_webhook_event(raw_body=make_body(raw), signature_header=header_for(raw), secret=TEST_SECRET)
    assert isinstance(event, WebhookErrorPayload)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("[]", id="empty-array"),
        pytest.param('[{"id": "whd_1", "status": "succeeded"}]', id="array"),
        pytest.param("null", id="null"),
        pytest.param("42", id="number"),
        pytest.param('"text"', id="string"),
    ],
)
def test_construct_raises_invalid_json_for_a_json_value_that_is_not_an_object(body: str) -> None:
    # A documented difference: the Node SDK returns any JSON value.
    error = construct_error(raw_body=body, signature_header=header_for(body), secret=TEST_SECRET)
    assert error.reason == "invalid_json"
    assert str(error) == "The verified webhook payload is not a JSON object."


# --- client delegation (Node: 2 cases), on both clients ------------------------------------------------------------


def make_client(client_kind: str) -> DevupAI | AsyncDevupAI:
    return DevupAI(api_key=TEST_KEY) if client_kind == "sync" else AsyncDevupAI(api_key=TEST_KEY)


def close_client(client: DevupAI | AsyncDevupAI) -> None:
    if isinstance(client, AsyncDevupAI):
        anyio.run(client.close)
    else:
        client.close()


def test_client_webhooks_verify_signature_delegates_to_the_standalone_function(client_kind: str) -> None:
    # Node: "client.webhooks.verifySignature delegates to standalone verifyWebhookSignature"
    client = make_client(client_kind)
    try:
        assert isinstance(client.webhooks, Webhooks)
        header = header_for(SAMPLE_SUCCESS_PAYLOAD)
        # A sync call on both clients.
        result = client.webhooks.verify_signature(
            raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header, secret=TEST_SECRET
        )
        assert result is True
        assert (
            client.webhooks.verify_signature(
                raw_body=SAMPLE_SUCCESS_PAYLOAD + " ", signature_header=header, secret=TEST_SECRET
            )
            is False
        )
    finally:
        close_client(client)


def test_client_webhooks_construct_event_delegates_to_the_standalone_function(client_kind: str) -> None:
    # Node: "client.webhooks.constructEvent delegates to standalone constructWebhookEvent"
    client = make_client(client_kind)
    try:
        event = client.webhooks.construct_event(
            raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header_for(SAMPLE_SUCCESS_PAYLOAD), secret=TEST_SECRET
        )
        assert isinstance(event, WebhookSuccessPayload)
        assert event.id == DELIVERY_ID
        assert event.status == "succeeded"
        with pytest.raises(DevupWebhookVerificationError) as info:
            client.webhooks.construct_event(
                raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=f"t={NOW}", secret=TEST_SECRET
            )
        assert info.value.reason == "malformed_header"
    finally:
        close_client(client)


def test_a_webhooks_instance_needs_no_client() -> None:
    webhooks = Webhooks()
    assert webhooks.verify_signature(
        raw_body=SAMPLE_SUCCESS_PAYLOAD, signature_header=header_for(SAMPLE_SUCCESS_PAYLOAD), secret=TEST_SECRET
    )
