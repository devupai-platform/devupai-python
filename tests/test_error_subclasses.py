"""Status to class mapping of the errors, through both clients and through ``make_api_error``.

Twin of the Node SDK's ``tests/error-subclasses.test.ts``: HTTP 400, 402, 413, 415 and 422 have their own subclasses
of ``DevupAPIError``, every subclass is still a ``DevupAPIError``, and the other statuses keep their class.
"""

from __future__ import annotations

import pytest
from helpers.mock_transport import REQ_ID, MockAPI, error_response, run_request, sequence

import devupai
from devupai import (
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
from devupai._client import BILLED_POST_RETRY
from devupai._errors import make_api_error

NEW_CLASSES: list[tuple[int, type[DevupAPIError], str, str]] = [
    (400, BadRequestError, "BadRequestError", "invalid_value"),
    (402, PaymentRequiredError, "PaymentRequiredError", "insufficient_balance"),
    (413, PayloadTooLargeError, "PayloadTooLargeError", "payload_too_large"),
    (415, UnsupportedMediaTypeError, "UnsupportedMediaTypeError", "invalid_request_error"),
    (422, UnprocessableEntityError, "UnprocessableEntityError", "missing_required_parameter"),
]

OTHER_CLASSES: list[tuple[int, type[DevupAPIError], str]] = [
    (401, AuthenticationError, "AuthenticationError"),
    (403, PermissionDeniedError, "PermissionDeniedError"),
    (404, NotFoundError, "NotFoundError"),
    (409, ConflictError, "ConflictError"),
    (429, RateLimitError, "RateLimitError"),
    (500, InternalServerError, "InternalServerError"),
    (502, InternalServerError, "InternalServerError"),
    (503, InternalServerError, "InternalServerError"),
    (504, InternalServerError, "InternalServerError"),
    (405, DevupAPIError, "DevupAPIError"),
    (418, DevupAPIError, "DevupAPIError"),
]

#: Every class that extends ``DevupAPIError`` directly, with the Node class name.
DIRECT_SUBCLASSES: list[tuple[type[DevupAPIError], str]] = [
    (BadRequestError, "BadRequestError"),
    (AuthenticationError, "AuthenticationError"),
    (PaymentRequiredError, "PaymentRequiredError"),
    (PermissionDeniedError, "PermissionDeniedError"),
    (NotFoundError, "NotFoundError"),
    (ConflictError, "ConflictError"),
    (PayloadTooLargeError, "PayloadTooLargeError"),
    (UnsupportedMediaTypeError, "UnsupportedMediaTypeError"),
    (UnprocessableEntityError, "UnprocessableEntityError"),
    (RateLimitError, "RateLimitError"),
    (InternalServerError, "InternalServerError"),
    (APIConnectionError, "APIConnectionError"),
]

PUBLIC_ERROR_NAMES = [
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
]

FIXTURE_HEADERS = {"x-request-id": REQ_ID, "x-client-request-id": "fixture-client-id"}


def _raise_for(client_kind: str, status: int, code: str) -> tuple[DevupAPIError, MockAPI]:
    api = MockAPI(
        sequence(error_response(status, "Fixture message.", code=code, param="input", headers=FIXTURE_HEADERS))
    )
    client = api.client(client_kind)
    with pytest.raises(DevupAPIError) as info:
        run_request(client, "GET", "/x", options={"max_retries": 0})
    return info.value, api


@pytest.mark.parametrize(("status", "cls", "name", "code"), NEW_CLASSES, ids=[str(row[0]) for row in NEW_CLASSES])
def test_new_status_gives_its_class_with_every_field(
    client_kind: str, status: int, cls: type[DevupAPIError], name: str, code: str
) -> None:
    err, api = _raise_for(client_kind, status, code)
    assert type(err) is cls
    assert isinstance(err, DevupAPIError)
    assert isinstance(err, Exception)
    assert type(err).__name__ == name
    assert err.status == status
    assert err.code == code
    assert err.type == "devup_error"
    assert err.param == "input"
    assert err.message == "Fixture message."
    assert str(err) == "Fixture message."
    assert err.request_id == REQ_ID
    assert err.client_request_id == "fixture-client-id"
    assert len(api.calls) == 1


@pytest.mark.parametrize(("status", "cls", "name"), OTHER_CLASSES, ids=[str(row[0]) for row in OTHER_CLASSES])
def test_other_status_keeps_its_class(client_kind: str, status: int, cls: type[DevupAPIError], name: str) -> None:
    err, api = _raise_for(client_kind, status, "fixture_code")
    assert type(err) is cls
    assert isinstance(err, DevupAPIError)
    assert type(err).__name__ == name
    assert err.status == status
    assert err.request_id == REQ_ID
    for _, new_cls, _, _ in NEW_CLASSES:
        assert not isinstance(err, new_cls)
    assert not isinstance(err, APIConnectionError)
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    ("status", "cls"), [(row[0], row[1]) for row in NEW_CLASSES], ids=[str(r[0]) for r in NEW_CLASSES]
)
def test_make_api_error_picks_the_new_class(status: int, cls: type[DevupAPIError]) -> None:
    err = make_api_error(status, "m", code="fixture_code", param="model", headers={"x-request-id": REQ_ID})
    assert type(err) is cls
    assert isinstance(err, DevupAPIError)
    assert err.code == "fixture_code"
    assert err.param == "model"
    assert err.request_id == REQ_ID


@pytest.mark.parametrize(
    ("status", "cls"), [(row[0], row[1]) for row in OTHER_CLASSES], ids=[str(r[0]) for r in OTHER_CLASSES]
)
def test_make_api_error_keeps_the_other_classes(status: int, cls: type[DevupAPIError]) -> None:
    assert type(make_api_error(status, "m")) is cls


@pytest.mark.parametrize("status", [405, 410, 418, 0, 100, 204, 302, 304, 499])
def test_an_unmapped_status_stays_a_plain_devup_api_error(status: int) -> None:
    err = make_api_error(status, "m")
    assert type(err) is DevupAPIError
    assert type(err).__name__ == "DevupAPIError"
    assert err.status == status


@pytest.mark.parametrize("status", [500, 501, 502, 503, 504, 599, 600])
def test_any_status_from_500_is_an_internal_server_error(status: int) -> None:
    assert type(make_api_error(status, "m")) is InternalServerError


def test_make_api_error_never_returns_a_connection_error() -> None:
    for status in range(0, 700):
        err = make_api_error(status, "m")
        assert not isinstance(err, APIConnectionError), status
    assert type(make_api_error(0, "m")) is DevupAPIError


def test_the_new_classes_extend_devup_api_error_directly_and_are_not_instances_of_one_another() -> None:
    for _, cls, _, _ in NEW_CLASSES:
        assert cls.__bases__ == (DevupAPIError,)
    bad = make_api_error(400, "m")
    assert not isinstance(bad, PaymentRequiredError)
    assert not isinstance(bad, UnprocessableEntityError)
    assert not isinstance(bad, PayloadTooLargeError)
    assert not isinstance(bad, UnsupportedMediaTypeError)
    unsupported = make_api_error(415, "m")
    assert not isinstance(unsupported, BadRequestError)
    assert not isinstance(unsupported, UnprocessableEntityError)


@pytest.mark.parametrize(("cls", "name"), DIRECT_SUBCLASSES, ids=[row[1] for row in DIRECT_SUBCLASSES])
def test_every_subclass_extends_devup_api_error_and_keeps_the_node_name(cls: type[DevupAPIError], name: str) -> None:
    assert issubclass(cls, DevupAPIError)
    assert cls.__bases__ == (DevupAPIError,)
    assert cls.__name__ == name
    assert cls.__doc__ is not None and cls.__doc__.strip() != ""


def test_devup_api_error_is_an_exception_with_the_node_name() -> None:
    assert issubclass(DevupAPIError, Exception)
    assert DevupAPIError.__name__ == "DevupAPIError"
    assert DevupAPIError.__doc__ is not None and DevupAPIError.__doc__.strip() != ""


def test_timeout_error_is_a_connection_error() -> None:
    assert APITimeoutError.__name__ == "APITimeoutError"
    assert APITimeoutError.__bases__ == (APIConnectionError,)
    assert issubclass(APITimeoutError, APIConnectionError)
    assert issubclass(APITimeoutError, DevupAPIError)
    err = APITimeoutError(0, "The request timed out after 5 seconds.")
    assert isinstance(err, APIConnectionError)
    assert err.status == 0
    assert str(err) == "The request timed out after 5 seconds."


def test_a_415_on_a_billed_call_is_returned_after_one_attempt_also_with_retries_on(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(415, "Fixture message.", code="invalid_request_error")))
    client = api.client(client_kind)
    with pytest.raises(UnsupportedMediaTypeError):
        run_request(client, "POST", "/x", body={"input": "x"}, retry=BILLED_POST_RETRY, options={"max_retries": 2})
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    "reason", ["malformed_header", "timestamp_outside_tolerance", "no_matching_signature", "invalid_json"]
)
def test_webhook_verification_error_has_a_reason_and_is_not_an_api_error(reason: str) -> None:
    assert DevupWebhookVerificationError.__name__ == "DevupWebhookVerificationError"
    assert issubclass(DevupWebhookVerificationError, Exception)
    assert not issubclass(DevupWebhookVerificationError, DevupAPIError)
    err = DevupWebhookVerificationError(reason, "Fixture message.")  # type: ignore[arg-type]
    assert err.reason == reason
    assert str(err) == "Fixture message."
    assert not isinstance(err, DevupAPIError)


@pytest.mark.parametrize("name", PUBLIC_ERROR_NAMES)
def test_every_error_class_is_public(name: str) -> None:
    import devupai._errors as errors_module

    assert name in devupai.__all__
    assert getattr(devupai, name) is getattr(errors_module, name)


def test_the_public_error_list_has_fifteen_names() -> None:
    assert len(PUBLIC_ERROR_NAMES) == len(set(PUBLIC_ERROR_NAMES)) == 15
    assert {name for name in devupai.__all__ if name.endswith("Error")} == set(PUBLIC_ERROR_NAMES)
