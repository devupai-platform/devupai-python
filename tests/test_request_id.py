"""request_id, client_request_id, warnings and the retry-after fields on APIResponse and on errors."""

from __future__ import annotations

import email.utils
import json
from typing import Any

import httpx
import pytest
from helpers.mock_transport import REQ_ID, MockAPI, error_response, json_response, run_request, sequence

from devupai import DevupAPIError, RateLimitError

CLIENT_ID = "fixture-client.id_01:a"


def raw_error(status: int, headers: dict[str, str]) -> httpx.Response:
    """An error response with exactly `headers` (no header added)."""
    body = {"error": {"message": "Fixture message.", "type": "devup_error", "code": "fixture_code"}}
    return httpx.Response(status, content=json.dumps(body).encode(), headers=headers)


def fail(api: MockAPI, client_kind: str, **kwargs: Any) -> DevupAPIError:
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind, max_retries=0), "GET", "/models", **kwargs)
    return info.value


# --- on APIResponse ----------------------------------------------------------------------------------------------


def test_response_request_id_comes_from_x_request_id(client_kind: str) -> None:
    api = MockAPI(sequence(json_response({"ok": True}, headers={"x-request-id": "req_fixture_2"})))
    result = run_request(api.client(client_kind), "GET", "/models")
    assert result.request_id == "req_fixture_2"
    assert result.response.headers["x-request-id"] == "req_fixture_2"


def test_response_request_id_reads_the_header_case_insensitively(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, json={"ok": True}, headers={"X-Request-Id": "req_mixed"}))
    assert run_request(api.client(client_kind), "GET", "/models").request_id == "req_mixed"


def test_response_without_the_headers_gives_none_none_and_no_warnings(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, json={"balance_dzd": 1}))
    result = run_request(api.client(client_kind), "GET", "/user/balance")
    assert result.data == {"balance_dzd": 1}
    assert result.request_id is None
    assert result.client_request_id is None
    assert result.warnings == []


def test_response_with_an_empty_x_request_id_keeps_the_empty_string(client_kind: str) -> None:
    api = MockAPI(lambda request: httpx.Response(200, json={}, headers={"x-request-id": "", "x-client-request-id": ""}))
    result = run_request(api.client(client_kind), "GET", "/models")
    assert result.request_id == ""
    assert result.client_request_id == ""


def test_response_client_request_id_comes_from_x_client_request_id(client_kind: str) -> None:
    api = MockAPI(sequence(json_response({"ok": True}, headers={"x-client-request-id": CLIENT_ID})))
    result = run_request(
        api.client(client_kind), "GET", "/models", options={"extra_headers": {"x-client-request-id": CLIENT_ID}}
    )
    assert api.calls[0].headers["x-client-request-id"] == CLIENT_ID
    assert result.request_id == REQ_ID
    assert result.client_request_id == CLIENT_ID


@pytest.mark.parametrize("kind", ["text", "raw"])
def test_response_ids_are_set_for_text_and_raw_kinds(client_kind: str, kind: str) -> None:
    headers = {"x-request-id": REQ_ID, "x-client-request-id": CLIENT_ID}
    api = MockAPI(lambda request: httpx.Response(200, content=b"x", headers=headers))
    result = run_request(api.client(client_kind), "GET", "/images/proxy", kind=kind, auth=False)
    assert result.request_id == REQ_ID
    assert result.client_request_id == CLIENT_ID


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param("a, b,,c", ["a", "b", "c"], id="mixed"),
        pytest.param(" fixture_notice_a ,fixture_notice_b,, ", ["fixture_notice_a", "fixture_notice_b"], id="padded"),
        pytest.param("single", ["single"], id="single"),
        pytest.param(" , ,", [], id="only-separators"),
        pytest.param("", [], id="empty"),
    ],
)
def test_response_warnings_are_parsed_from_x_devup_warnings(client_kind: str, header: str, expected: list[str]) -> None:
    api = MockAPI(sequence(json_response({"ok": True}, headers={"X-DEVUP-Warnings": header})))
    assert run_request(api.client(client_kind), "GET", "/models").warnings == expected


def test_response_ids_on_a_304_come_from_the_304(client_kind: str) -> None:
    api = MockAPI(
        lambda request: httpx.Response(
            304, headers={"x-request-id": "req_304", "x-client-request-id": CLIENT_ID, "X-DEVUP-Warnings": "w"}
        )
    )
    result = run_request(api.client(client_kind), "GET", "/models", if_none_match='"e"')
    assert result.request_id == "req_304"
    assert result.client_request_id == CLIENT_ID
    assert result.warnings == ["w"]


# --- on errors ---------------------------------------------------------------------------------------------------


def test_error_carries_request_id_client_request_id_and_warnings(client_kind: str) -> None:
    api = MockAPI(
        sequence(
            error_response(
                400,
                "Fixture message.",
                code="fixture_invalid",
                param="messages",
                headers={"x-client-request-id": CLIENT_ID, "X-DEVUP-Warnings": "a, b,,c"},
            )
        )
    )
    err = fail(api, client_kind)
    assert err.status == 400
    assert err.message == "Fixture message."
    assert err.code == "fixture_invalid"
    assert err.param == "messages"
    assert err.request_id == REQ_ID
    assert err.client_request_id == CLIENT_ID
    assert err.warnings == ["a", "b", "c"]
    assert err.headers["x-client-request-id"] == CLIENT_ID
    assert all(name == name.lower() for name in err.headers)


def test_error_with_an_empty_x_request_id_has_request_id_none(client_kind: str) -> None:
    api = MockAPI(lambda request: raw_error(400, {"x-request-id": "", "x-client-request-id": ""}))
    err = fail(api, client_kind)
    assert err.request_id is None
    assert err.client_request_id is None


def test_error_without_the_headers_has_none_fields_and_no_warnings(client_kind: str) -> None:
    api = MockAPI(lambda request: raw_error(400, {}))
    err = fail(api, client_kind)
    assert err.request_id is None
    assert err.client_request_id is None
    assert err.retry_after is None
    assert err.retry_after_ms is None
    assert err.warnings == []


def test_error_request_id_is_the_one_of_the_last_attempt(client_kind: str) -> None:
    api = MockAPI(
        sequence(
            error_response(503, "busy", headers={"x-request-id": "req_first"}),
            error_response(503, "busy", headers={"x-request-id": "req_last"}),
        )
    )
    with pytest.raises(DevupAPIError) as info:
        run_request(api.client(client_kind, max_retries=1), "GET", "/models")
    assert info.value.request_id == "req_last"
    assert len(api.calls) == 2


def test_error_keeps_the_raw_retry_after_header(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(429, "busy", headers={"retry-after": "60"})))
    err = fail(api, client_kind)
    assert isinstance(err, RateLimitError)
    assert err.retry_after == "60"
    assert err.retry_after_ms == 60_000


@pytest.mark.parametrize(
    ("headers", "retry_after", "expected_ms"),
    [
        pytest.param({"retry-after-ms": "1500"}, None, 1500, id="ms"),
        pytest.param({"retry-after-ms": " 250 ", "retry-after": "9"}, "9", 250, id="ms-wins"),
        pytest.param({"retry-after-ms": "12.5"}, None, 12.5, id="ms-fraction"),
        pytest.param({"retry-after-ms": "1e3"}, None, 1000, id="ms-exponent"),
        pytest.param({"retry-after": "2"}, "2", 2000, id="seconds"),
        pytest.param({"retry-after": "1.5"}, "1.5", 1500, id="seconds-fraction"),
        pytest.param({"retry-after-ms": "-5", "retry-after": "3"}, "3", 3000, id="negative-ms-falls-back"),
        pytest.param({"retry-after-ms": "soon", "retry-after": "3"}, "3", 3000, id="bad-ms-falls-back"),
        pytest.param({"retry-after-ms": "", "retry-after": "4"}, "4", 4000, id="empty-ms-falls-back"),
        pytest.param({"retry-after": "soon"}, "soon", None, id="not-a-date"),
        pytest.param({"Retry-After": "7"}, "7", 7000, id="header-case"),
    ],
)
def test_error_retry_after_ms_from_headers(
    client_kind: str, headers: dict[str, str], retry_after: str | None, expected_ms: float | None
) -> None:
    api = MockAPI(lambda request: raw_error(429, headers))
    err = fail(api, client_kind)
    assert err.retry_after == retry_after
    if expected_ms is None:
        assert err.retry_after_ms is None
    else:
        assert err.retry_after_ms == pytest.approx(expected_ms)


@pytest.mark.parametrize(("offset", "expected_ms"), [(7, 7000), (0, 0), (-30, 0)])
def test_error_retry_after_ms_from_an_http_date_uses_the_wall_clock(
    client_kind: str, fake_clock: Any, offset: int, expected_ms: float
) -> None:
    date = email.utils.formatdate(fake_clock.wall + offset, usegmt=True)
    api = MockAPI(lambda request: raw_error(429, {"retry-after": date}))
    err = fail(api, client_kind)
    assert err.retry_after == date
    assert err.retry_after_ms == pytest.approx(expected_ms)


def test_error_retry_after_ms_from_a_naive_http_date_is_read_as_utc(client_kind: str, fake_clock: Any) -> None:
    # formatdate() without usegmt ends in "-0000", which parsedate_to_datetime turns into a naive datetime.
    naive = email.utils.formatdate(fake_clock.wall + 5)
    assert naive.endswith("-0000")
    api = MockAPI(lambda request: raw_error(429, {"retry-after": naive}))
    err = fail(api, client_kind)
    assert err.retry_after_ms == pytest.approx(5000)


# --- DevupAPIError built directly ----------------------------------------------------------------------------------


def test_error_built_without_headers_has_none_ids_and_empty_collections() -> None:
    err = DevupAPIError(0, "Fixture.")
    assert err.request_id is None
    assert err.client_request_id is None
    assert err.retry_after is None
    assert err.retry_after_ms is None
    assert err.warnings == []
    assert err.headers == {}


def test_error_built_with_headers_reads_them_case_insensitively(fake_clock: Any) -> None:
    err = DevupAPIError(
        404,
        "Msg",
        headers={
            "X-Request-Id": "123",
            "X-Client-Request-Id": CLIENT_ID,
            "Retry-After": "3",
            "X-DEVUP-Warnings": "w1,w2",
        },
    )
    assert err.request_id == "123"
    assert err.client_request_id == CLIENT_ID
    assert err.retry_after == "3"
    assert err.retry_after_ms == 3000
    assert err.warnings == ["w1", "w2"]
    assert err.headers["x-request-id"] == "123"
    assert all(name == name.lower() for name in err.headers)


def test_error_built_with_empty_id_headers_has_none_ids() -> None:
    err = DevupAPIError(400, "m", headers={"x-request-id": "", "x-client-request-id": "", "retry-after": ""})
    assert err.request_id is None
    assert err.client_request_id is None
    assert err.retry_after is None
    assert err.retry_after_ms is None
