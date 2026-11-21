"""The error message rule, the error constructor, and the ``Retry-After`` and warnings header parsers.

Twin of the Node SDK's ``tests/extra.test.ts`` (error message when the body is not an error object) and of the
message rule in Node ``src/core/client.ts`` (``errorFromBody``), plus the pure cases of ``parseRetryAfterMs`` and
``parseWarnings`` in Node ``src/core/errors.ts``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any

import httpx
import pytest
from helpers.mock_transport import REQ_ID, MockAPI, json_response, run_request, sequence

from devupai import BadRequestError, DevupAPIError, InternalServerError
from devupai._errors import error_from_body, parse_retry_after_ms, parse_warnings

NOW = 1_800_000_000.0


def _body(error: object) -> str:
    return json.dumps({"error": error})


# --- error_from_body: the message rule -------------------------------------------------------------------------


def test_error_message_from_the_envelope_is_used() -> None:
    err = error_from_body(401, _body({"message": "JSON msg", "type": "devup_error", "code": "c", "param": "p"}), None)
    assert err.message == "JSON msg"
    assert str(err) == "JSON msg"
    assert err.type == "devup_error"
    assert err.code == "c"
    assert err.param == "p"
    assert err.status == 401


def test_an_empty_error_message_keeps_the_raw_text() -> None:
    text = _body({"message": "", "code": "c"})
    err = error_from_body(400, text, None)
    assert err.message == text
    assert err.code == "c"


def test_a_missing_error_message_keeps_the_raw_text() -> None:
    text = _body({"code": "c"})
    assert error_from_body(400, text, None).message == text


def test_a_non_string_error_message_keeps_the_raw_text() -> None:
    text = _body({"message": 5})
    assert error_from_body(400, text, None).message == text


def test_a_blank_error_message_becomes_http_status() -> None:
    assert error_from_body(418, _body({"message": "   "}), None).message == "HTTP 418"


def test_a_string_error_is_the_message() -> None:
    err = error_from_body(400, '{"error":"text"}', None)
    assert err.message == "text"
    assert err.type is None
    assert err.code is None
    assert err.param is None


def test_an_empty_string_error_keeps_the_raw_text() -> None:
    assert error_from_body(400, '{"error":""}', None).message == '{"error":""}'


@pytest.mark.parametrize("text", ["{ malformed ", "raw text", "text error", "<html>Bad gateway page</html>"])
def test_a_body_that_is_not_json_is_the_message(text: str) -> None:
    err = error_from_body(400, text, None)
    assert err.message == text
    assert err.code is None


@pytest.mark.parametrize("text", ["", " ", "   \n\t ", "\r\n"])
def test_an_empty_or_whitespace_body_becomes_http_status(text: str) -> None:
    err = error_from_body(418, text, None)
    assert err.message == "HTTP 418"
    assert str(err) == "HTTP 418"


@pytest.mark.parametrize("text", ["[1, 2]", '["error"]', '"just a string"', "42", "null", "true"])
def test_a_json_body_that_is_not_an_object_keeps_the_raw_text(text: str) -> None:
    err = error_from_body(400, text, None)
    assert err.message == text
    assert err.type is None
    assert err.code is None
    assert err.param is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("fixture_code", "fixture_code"),
        ("", ""),
        (42, "42"),
        (0, "0"),
        (-3, "-3"),
        (1.0, "1"),
        (1.5, "1.5"),
        (True, None),
        (False, None),
        (None, None),
        ({"a": 1}, None),
        ([1], None),
    ],
    ids=lambda v: repr(v),
)
def test_code_is_a_string_or_a_number_turned_into_a_string(code: object, expected: str | None) -> None:
    err = error_from_body(400, _body({"message": "m", "code": code}), None)
    assert err.code == expected


@pytest.mark.parametrize("param", [5, 1.5, True, None, ["input"], {"name": "input"}], ids=lambda v: repr(v))
def test_a_param_that_is_not_a_string_is_none(param: object) -> None:
    assert error_from_body(400, _body({"message": "m", "param": param}), None).param is None


@pytest.mark.parametrize("error_type", [5, True, None, ["devup_error"]], ids=lambda v: repr(v))
def test_a_type_that_is_not_a_string_is_none(error_type: object) -> None:
    assert error_from_body(400, _body({"message": "m", "type": error_type}), None).type is None


def test_error_from_body_picks_the_class_of_the_status() -> None:
    assert type(error_from_body(400, "x", None)) is BadRequestError
    assert type(error_from_body(500, "", None)) is InternalServerError
    assert type(error_from_body(418, "", None)) is DevupAPIError


def test_error_from_body_reads_the_headers() -> None:
    headers = {"X-Request-ID": REQ_ID, "Retry-After": "2", "X-DEVUP-Warnings": "fixture_notice"}
    err = error_from_body(429, _body({"message": "m"}), headers)
    assert err.request_id == REQ_ID
    assert err.retry_after == "2"
    assert err.retry_after_ms == 2000.0
    assert err.warnings == ["fixture_notice"]


def test_error_from_body_without_headers() -> None:
    err = error_from_body(400, "x", None)
    assert err.headers == {}
    assert err.request_id is None
    assert err.client_request_id is None
    assert err.retry_after is None
    assert err.retry_after_ms is None
    assert err.warnings == []


# --- the same rule through the client --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "cls", "message"),
    [
        (
            httpx.Response(400, headers={"content-type": "application/json"}, content=b"{ malformed "),
            BadRequestError,
            "{ malformed ",
        ),
        (
            httpx.Response(400, headers={"content-type": "application/json"}, content=b"raw text"),
            BadRequestError,
            "raw text",
        ),
        (httpx.Response(500, content=b""), InternalServerError, "HTTP 500"),
        (json_response({"error": {"message": "JSON msg"}}, status=401), DevupAPIError, "JSON msg"),
        (
            httpx.Response(503, headers={"content-type": "text/plain"}, content=b"text error"),
            InternalServerError,
            "text error",
        ),
    ],
    ids=["malformed-json", "raw-text", "empty-500", "json-envelope", "text-plain"],
)
def test_the_message_rule_through_the_client(
    client_kind: str, response: httpx.Response, cls: type[DevupAPIError], message: str
) -> None:
    api = MockAPI(sequence(response))
    client = api.client(client_kind)
    with pytest.raises(DevupAPIError) as info:
        run_request(client, "GET", "/x", options={"max_retries": 0})
    assert isinstance(info.value, cls)
    assert info.value.message == message
    assert str(info.value) == message
    assert len(api.calls) == 1


# --- the constructor -------------------------------------------------------------------------------------------


def test_constructor_lower_cases_the_header_names_and_reads_the_ids() -> None:
    err = DevupAPIError(
        429,
        "m",
        headers={
            "X-Request-ID": REQ_ID,
            "X-Client-Request-Id": "fixture-client-id",
            "Retry-After": "3",
            "X-DEVUP-Warnings": "a, ,b",
        },
    )
    assert err.headers == {
        "x-request-id": REQ_ID,
        "x-client-request-id": "fixture-client-id",
        "retry-after": "3",
        "x-devup-warnings": "a, ,b",
    }
    assert err.request_id == REQ_ID
    assert err.client_request_id == "fixture-client-id"
    assert err.retry_after == "3"
    assert err.retry_after_ms == 3000.0
    assert err.warnings == ["a", "b"]


def test_constructor_accepts_httpx_headers() -> None:
    err = DevupAPIError(500, "m", headers=httpx.Headers({"X-Request-Id": REQ_ID, "Retry-After-Ms": "250"}))
    assert err.headers["x-request-id"] == REQ_ID
    assert err.request_id == REQ_ID
    assert err.retry_after_ms == 250.0


def test_constructor_turns_empty_header_values_into_none() -> None:
    err = DevupAPIError(500, "m", headers={"x-request-id": "", "x-client-request-id": "", "retry-after": ""})
    assert err.request_id is None
    assert err.client_request_id is None
    assert err.retry_after is None
    assert err.retry_after_ms is None


def test_constructor_keeps_retry_after_raw() -> None:
    err = DevupAPIError(503, "m", headers={"retry-after": " 2 "})
    assert err.retry_after == " 2 "
    assert err.retry_after_ms == 2000.0


def test_constructor_without_headers() -> None:
    err = DevupAPIError(0, "Network error: unknown error")
    assert err.status == 0
    assert err.message == "Network error: unknown error"
    assert str(err) == "Network error: unknown error"
    assert err.type is None
    assert err.code is None
    assert err.param is None
    assert err.headers == {}
    assert err.request_id is None
    assert err.client_request_id is None
    assert err.retry_after is None
    assert err.retry_after_ms is None
    assert err.warnings == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("devup_error", "devup_error"),
        (42, "42"),
        (2.0, "2"),
        (2.5, "2.5"),
        (float("inf"), None),
        (float("nan"), None),
        (True, None),
        (None, None),
        (object(), None),
    ],
    ids=lambda v: repr(v),
)
def test_constructor_type_and_code_rule(value: object, expected: str | None) -> None:
    err = DevupAPIError(400, "m", error_type=value, code=value)
    assert err.type == expected
    assert err.code == expected


@pytest.mark.parametrize(("value", "expected"), [("input", "input"), ("", ""), (5, None), (None, None)])
def test_constructor_param_rule(value: object, expected: str | None) -> None:
    assert DevupAPIError(400, "m", param=value).param == expected


def test_constructor_has_no_cause_attribute_and_chains_with_raise_from() -> None:
    cause = ValueError("inner")
    try:
        try:
            raise cause
        except ValueError as exc:
            raise DevupAPIError(0, "outer") from exc
    except DevupAPIError as err:
        assert err.__cause__ is cause
        assert not hasattr(err, "cause")


# --- parse_retry_after_ms --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "1500"}, 1500.0),
        ({"retry-after-ms": " 0 "}, 0.0),
        ({"retry-after-ms": "250.5"}, 250.5),
        ({"retry-after-ms": ".5"}, 0.5),
        ({"retry-after-ms": "1e3"}, 1000.0),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1500.0),
        ({"retry-after-ms": "-1", "retry-after": "3"}, 3000.0),
        ({"retry-after-ms": "abc", "retry-after": "2"}, 2000.0),
        ({"retry-after-ms": "", "retry-after": "1"}, 1000.0),
        ({"retry-after-ms": "   ", "retry-after": "1"}, 1000.0),
        ({"retry-after-ms": "inf", "retry-after": "4"}, 4000.0),
        ({"retry-after-ms": "nan", "retry-after": "4"}, 4000.0),
        ({"retry-after": "2"}, 2000.0),
        ({"retry-after": "1.5"}, 1500.0),
        ({"retry-after": " 0 "}, 0.0),
        ({"Retry-After-Ms": "10"}, 10.0),
        ({"RETRY-AFTER": "1"}, 1000.0),
    ],
    ids=lambda v: repr(v),
)
def test_parse_retry_after_ms_values(headers: dict[str, str], expected: float) -> None:
    result = parse_retry_after_ms(headers, now=NOW)
    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"retry-after-ms": "-1"},
        {"retry-after-ms": "abc"},
        {"retry-after": ""},
        {"retry-after": "   "},
        {"retry-after": "soon"},
        {"retry-after": "not a date"},
        {"x-other": "1"},
    ],
    ids=lambda v: repr(v),
)
def test_parse_retry_after_ms_gives_none(headers: dict[str, str]) -> None:
    assert parse_retry_after_ms(headers, now=NOW) is None


def test_parse_retry_after_ms_accepts_httpx_headers() -> None:
    assert parse_retry_after_ms(httpx.Headers({"Retry-After-Ms": "75"}), now=NOW) == 75.0
    assert parse_retry_after_ms(httpx.Headers({"Retry-After": "2"}), now=NOW) == 2000.0


def test_a_past_http_date_gives_zero() -> None:
    assert parse_retry_after_ms({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, now=NOW) == 0.0


def test_a_future_http_date_gives_the_remaining_wait_and_respects_now() -> None:
    date = "Wed, 21 Oct 2026 07:28:00 GMT"
    epoch = parsedate_to_datetime(date).timestamp()
    assert parse_retry_after_ms({"retry-after": date}, now=epoch - 5) == 5000.0
    assert parse_retry_after_ms({"retry-after": date}, now=epoch - 30) == 30000.0
    assert parse_retry_after_ms({"retry-after": date}, now=epoch + 1) == 0.0


def test_an_http_date_without_now_uses_the_module_clock(fake_clock: Any) -> None:
    # ``fake_clock`` patches ``devupai._errors._time``; its wall clock is a whole number of seconds.
    wall: float = fake_clock.wall
    date = format_datetime(datetime.fromtimestamp(wall + 7, tz=UTC), usegmt=True)
    assert parse_retry_after_ms({"retry-after": date}) == 7000.0


def test_a_naive_http_date_is_read_as_utc() -> None:
    # "-0000" makes ``parsedate_to_datetime`` return a naive datetime.
    date = "Wed, 21 Oct 2026 07:28:00 -0000"
    epoch = datetime(2026, 10, 21, 7, 28, tzinfo=UTC).timestamp()
    assert parse_retry_after_ms({"retry-after": date}, now=epoch - 2) == 2000.0


# --- parse_warnings --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-devup-warnings": "a, b ,,c"}, ["a", "b", "c"]),
        ({"X-DEVUP-Warnings": "fixture_notice"}, ["fixture_notice"]),
        ({"x-devup-warnings": ""}, []),
        ({"x-devup-warnings": " , ,"}, []),
        ({}, []),
    ],
    ids=lambda v: repr(v),
)
def test_parse_warnings(headers: dict[str, str], expected: list[str]) -> None:
    assert parse_warnings(headers) == expected


def test_parse_warnings_accepts_httpx_headers() -> None:
    assert parse_warnings(httpx.Headers({"X-Devup-Warnings": "x,y"})) == ["x", "y"]
