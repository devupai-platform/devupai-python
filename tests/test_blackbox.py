"""Constructor options and request building: API key, headers, base URL, validation, URL, query and body."""

from __future__ import annotations

import math
from typing import Any

import anyio
import httpx
import pytest
from helpers.mock_transport import BASE, MODEL_A, TEST_KEY, MockAPI, json_response, run_request

import devupai
from devupai import AsyncDevupAI, DevupAI
from devupai._client import DEFAULT_BASE_URL

KEY_MESSAGE = "api_key contains characters that are not allowed in an HTTP header."
MISSING_KEY_MESSAGE = "DEVUP AI API key is missing. Pass api_key=... or set the DEVUP_API_KEY environment variable."
NAME_MESSAGE = "A header name contains characters that are not allowed in an HTTP header."
MAX_RETRIES_MESSAGE = "max_retries must be a non-negative integer."
TIMEOUT_MESSAGE = "timeout must be a positive number of seconds."
IDLE_MESSAGE = "stream_idle_timeout must be a non-negative number of seconds."

SECRET_KEY = "fixture-secret-key"
SECRET_VALUE = "fixture-secret-value"

FORBIDDEN = [
    pytest.param("\n", id="LF"),
    pytest.param("\r", id="CR"),
    pytest.param("\r\n", id="CRLF"),
    pytest.param("\0", id="NUL"),
]
# The environment cannot hold a NUL character, so the environment rows leave it out.
FORBIDDEN_IN_ENV = [
    pytest.param("\n", id="LF"),
    pytest.param("\r", id="CR"),
    pytest.param("\r\n", id="CRLF"),
]
HEADER_PLACES = ["default_headers", "extra_headers"]


def header_message(name: str) -> str:
    return f'Header "{name}" has a value that is not allowed in an HTTP header.'


class Capture:
    """A handler that keeps every `httpx.Request` it receives and answers with a small JSON body."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return json_response({"ok": True})


def ok_handler(request: httpx.Request) -> httpx.Response:
    return json_response({"ok": True})


def close_client(client: DevupAI | AsyncDevupAI) -> None:
    if isinstance(client, AsyncDevupAI):
        anyio.run(client.close)
    else:
        client.close()


def close_response(client_kind: str, response: httpx.Response) -> None:
    if client_kind == "async":
        anyio.run(response.aclose)
    else:
        response.close()


def with_headers(
    api: MockAPI, client_kind: str, place: str, headers: dict[str, Any], **opts: Any
) -> tuple[DevupAI | AsyncDevupAI, dict[str, Any] | None]:
    """Builds a client with `headers` as client headers or returns them as per-call options."""
    if place == "default_headers":
        return api.client(client_kind, default_headers=headers, **opts), None
    return api.client(client_kind, **opts), {"extra_headers": headers}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("DEVUP_API_KEY", raising=False)
    monkeypatch.delenv("DEVUPAI_API_KEY", raising=False)
    return monkeypatch


# --- API key -----------------------------------------------------------------------------------------------------


def test_api_key_argument_wins_over_both_environment_variables(clean_env: pytest.MonkeyPatch, client_kind: str) -> None:
    clean_env.setenv("DEVUP_API_KEY", "env-key-primary")
    clean_env.setenv("DEVUPAI_API_KEY", "env-key-secondary")
    api = MockAPI(ok_handler)
    client = api.client(client_kind, api_key="argument-key")
    assert client.api_key == "argument-key"
    run_request(client, "GET", "/models")
    assert api.calls[0].headers["authorization"] == "Bearer argument-key"


def test_api_key_none_reads_devup_api_key_first(clean_env: pytest.MonkeyPatch, client_kind: str) -> None:
    clean_env.setenv("DEVUP_API_KEY", "env-key-primary")
    clean_env.setenv("DEVUPAI_API_KEY", "env-key-secondary")
    api = MockAPI(ok_handler)
    client = api.client(client_kind, api_key=None)
    assert client.api_key == "env-key-primary"
    run_request(client, "GET", "/models")
    assert api.calls[0].headers["authorization"] == "Bearer env-key-primary"


def test_api_key_reads_devupai_api_key_when_devup_api_key_is_absent(
    clean_env: pytest.MonkeyPatch, client_kind: str
) -> None:
    clean_env.setenv("DEVUPAI_API_KEY", "env-key-secondary")
    api = MockAPI(ok_handler)
    client = api.client(client_kind, api_key=None)
    assert client.api_key == "env-key-secondary"
    run_request(client, "GET", "/models")
    assert api.calls[0].headers["authorization"] == "Bearer env-key-secondary"


@pytest.mark.parametrize("value", [pytest.param("", id="empty"), pytest.param("   ", id="blank")])
def test_present_but_blank_devup_api_key_raises_and_does_not_fall_through(
    clean_env: pytest.MonkeyPatch, client_kind: str, value: str
) -> None:
    clean_env.setenv("DEVUP_API_KEY", value)
    clean_env.setenv("DEVUPAI_API_KEY", "env-key-secondary")
    with pytest.raises(ValueError) as info:
        MockAPI(ok_handler).client(client_kind, api_key=None)
    assert str(info.value) == MISSING_KEY_MESSAGE


def test_blank_devupai_api_key_raises_the_missing_key_error(clean_env: pytest.MonkeyPatch, client_kind: str) -> None:
    clean_env.setenv("DEVUPAI_API_KEY", "  ")
    with pytest.raises(ValueError) as info:
        MockAPI(ok_handler).client(client_kind, api_key=None)
    assert str(info.value) == MISSING_KEY_MESSAGE


def test_no_api_key_anywhere_raises_value_error_with_the_contract_message(
    clean_env: pytest.MonkeyPatch, client_kind: str
) -> None:
    with pytest.raises(ValueError) as info:
        MockAPI(ok_handler).client(client_kind, api_key=None)
    assert str(info.value) == MISSING_KEY_MESSAGE


def test_no_api_key_anywhere_raises_for_a_client_built_without_options(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError) as info:
        DevupAI()
    assert str(info.value) == MISSING_KEY_MESSAGE
    with pytest.raises(ValueError) as info_async:
        AsyncDevupAI()
    assert str(info_async.value) == MISSING_KEY_MESSAGE


def test_empty_api_key_argument_is_allowed_and_sends_a_bare_bearer(
    clean_env: pytest.MonkeyPatch, client_kind: str
) -> None:
    # The header value is normalised like Node's Headers (leading/trailing space removed), so an empty key
    # sends "Bearer" and the API answers 401, instead of a local refusal of "Bearer " (review P1 M1).
    clean_env.setenv("DEVUP_API_KEY", "env-key-primary")
    api = MockAPI(ok_handler)
    client = api.client(client_kind, api_key="")
    assert client.api_key == ""
    run_request(client, "GET", "/models")
    assert api.calls[0].headers["authorization"] == "Bearer"


@pytest.mark.parametrize("bad", FORBIDDEN)
def test_api_key_argument_with_forbidden_character_raises_value_error_without_the_key(
    clean_env: pytest.MonkeyPatch, client_kind: str, bad: str
) -> None:
    api = MockAPI(ok_handler)
    with pytest.raises(ValueError) as info:
        api.client(client_kind, api_key=f"{SECRET_KEY}{bad}X-Injected: 1")
    assert str(info.value) == KEY_MESSAGE
    assert SECRET_KEY not in str(info.value)
    assert SECRET_KEY not in repr(info.value)
    assert api.calls == []


@pytest.mark.parametrize("variable", ["DEVUP_API_KEY", "DEVUPAI_API_KEY"])
@pytest.mark.parametrize("bad", FORBIDDEN_IN_ENV)
def test_api_key_from_environment_with_forbidden_character_raises_value_error_without_the_key(
    clean_env: pytest.MonkeyPatch, client_kind: str, variable: str, bad: str
) -> None:
    clean_env.setenv(variable, f"{SECRET_KEY}{bad}X-Injected: 1")
    api = MockAPI(ok_handler)
    with pytest.raises(ValueError) as info:
        api.client(client_kind, api_key=None)
    assert str(info.value) == KEY_MESSAGE
    assert SECRET_KEY not in str(info.value)
    assert SECRET_KEY not in repr(info.value)
    assert api.calls == []


@pytest.mark.parametrize("value", [pytest.param(12345, id="int"), pytest.param(b"fixture-secret-key", id="bytes")])
def test_api_key_that_is_not_a_string_raises_type_error(client_kind: str, value: object) -> None:
    with pytest.raises(TypeError) as info:
        MockAPI(ok_handler).client(client_kind, api_key=value)
    assert SECRET_KEY not in str(info.value)


# --- header names and values -------------------------------------------------------------------------------------


@pytest.mark.parametrize("place", HEADER_PLACES)
@pytest.mark.parametrize("bad", FORBIDDEN)
def test_header_value_with_forbidden_character_raises_before_any_request(
    client_kind: str, place: str, bad: str
) -> None:
    api = MockAPI(ok_handler)
    events: list[object] = []
    # Header values are checked when a request is built, not at construction.
    client, options = with_headers(
        api, client_kind, place, {"X-Fixture": f"{SECRET_VALUE}{bad}X-Injected: 1"}, on_debug=events.append
    )
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", "/models", options=options)
    assert not isinstance(info.value, devupai.DevupAPIError)
    assert str(info.value) == header_message("X-Fixture")
    assert SECRET_VALUE not in str(info.value)
    assert SECRET_VALUE not in repr(info.value)
    assert api.calls == []
    assert events == []


@pytest.mark.parametrize("place", HEADER_PLACES)
@pytest.mark.parametrize(
    "value",
    [pytest.param("fixture-secret-value-€", id="euro-sign"), pytest.param("日本", id="cjk")],
)
def test_header_value_that_is_not_latin1_raises_before_any_request(client_kind: str, place: str, value: str) -> None:
    api = MockAPI(ok_handler)
    client, options = with_headers(api, client_kind, place, {"X-Fixture": value})
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", "/models", options=options)
    assert str(info.value) == header_message("X-Fixture")
    assert value not in str(info.value)
    assert api.calls == []


@pytest.mark.parametrize("place", HEADER_PLACES)
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("X-Fixture\nsecret-name", id="LF"),
        pytest.param("X-Fixture secret-name", id="space"),
        pytest.param("X-Fixture:secret-name", id="colon"),
        pytest.param("X-Fixture(secret-name)", id="parens"),
        pytest.param("", id="empty"),
    ],
)
def test_invalid_header_name_raises_value_error_without_echoing_the_name(
    client_kind: str, place: str, name: str
) -> None:
    api = MockAPI(ok_handler)
    client, options = with_headers(api, client_kind, place, {name: "1"})
    with pytest.raises(ValueError) as info:
        run_request(client, "GET", "/models", options=options)
    assert str(info.value) == NAME_MESSAGE
    assert "secret-name" not in str(info.value)
    assert api.calls == []


def test_header_names_with_every_token_character_are_accepted(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    name = "X-!#$%&'*+.^_`|~09az"
    client = api.client(client_kind, default_headers={name: "1"})
    run_request(client, "GET", "/models")
    assert api.calls[0].headers[name.lower()] == "1"


def test_non_string_header_values_are_skipped(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client_headers: dict[str, Any] = {"X-Number": 1, "X-None": None, "X-Kept": "yes"}
    call_headers: dict[str, Any] = {"X-Bytes": b"raw", "X-List": ["a"], "X-Kept-Call": "1"}
    client = api.client(client_kind, default_headers=client_headers)
    run_request(client, "GET", "/models", options={"extra_headers": call_headers})
    headers = api.calls[0].headers
    for skipped in ("x-number", "x-none", "x-bytes", "x-list"):
        assert skipped not in headers
    assert headers["x-kept"] == "yes"
    assert headers["x-kept-call"] == "1"


def test_valid_header_values_with_space_and_tab_are_sent(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, default_headers={"X-Fixture-Global": "a b\tc"})
    run_request(client, "GET", "/models", options={"extra_headers": {"X-Fixture": "fixture-value"}})
    assert api.calls[0].headers["x-fixture-global"] == "a b\tc"
    assert api.calls[0].headers["x-fixture"] == "fixture-value"


def test_latin1_header_value_is_accepted(client_kind: str) -> None:
    capture = Capture()
    client = MockAPI(capture).client(client_kind, default_headers={"X-Fixture": "café"})
    run_request(client, "GET", "/models")
    raw = {name.lower(): value for name, value in capture.requests[0].headers.raw}
    assert raw[b"x-fixture"] == "café".encode("latin-1")


# --- base URL and repr -------------------------------------------------------------------------------------------


def test_base_url_defaults_to_the_public_api() -> None:
    assert DEFAULT_BASE_URL == "https://api.devupai.com/v1"
    sync = DevupAI(api_key=TEST_KEY)
    async_client = AsyncDevupAI(api_key=TEST_KEY)
    try:
        assert sync.base_url == DEFAULT_BASE_URL
        assert async_client.base_url == DEFAULT_BASE_URL
    finally:
        close_client(sync)
        close_client(async_client)


@pytest.mark.parametrize("given", ["http://localhost/api/v1/", "http://localhost/api/v1///"])
def test_trailing_slashes_are_removed_from_base_url(client_kind: str, given: str) -> None:
    client = MockAPI(ok_handler).client(client_kind, base_url=given)
    assert client.base_url == "http://localhost/api/v1"


@pytest.mark.parametrize("given", [pytest.param("", id="empty"), pytest.param(None, id="none")])
def test_empty_or_missing_base_url_gives_the_default(client_kind: str, given: str | None) -> None:
    client = MockAPI(ok_handler).client(client_kind, base_url=given)
    assert client.base_url == DEFAULT_BASE_URL


def test_path_is_appended_to_the_base_url_without_a_double_slash(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, base_url=BASE + "/")
    run_request(client, "GET", "/user/balance")
    assert api.calls[0].url == f"{BASE}/user/balance"
    assert api.calls[0].method == "GET"


def test_repr_shows_base_url_and_never_the_key(client_kind: str) -> None:
    client = MockAPI(ok_handler).client(client_kind, api_key="fixture-secret-key-0001")
    assert BASE in repr(client)
    assert "fixture-secret-key-0001" not in repr(client)
    assert "fixture-secret-key-0001" not in str(client)


def test_public_attributes_reflect_the_options(client_kind: str) -> None:
    client = MockAPI(ok_handler).client(client_kind, max_retries=5, timeout=12.5, stream_idle_timeout=3)
    assert client.api_key == TEST_KEY
    assert client.base_url == BASE
    assert client.max_retries == 5
    assert client.timeout == 12.5
    assert client.stream_idle_timeout == 3


def test_default_option_values(client_kind: str) -> None:
    client = MockAPI(ok_handler).client(client_kind)
    assert client.max_retries == 2
    assert client.timeout == 600.0
    assert client.stream_idle_timeout is None


# --- max_retries / timeout / stream_idle_timeout -----------------------------------------------------------------

BAD_MAX_RETRIES = [
    pytest.param(True, TypeError, id="bool-true"),
    pytest.param(False, TypeError, id="bool-false"),
    pytest.param("2", TypeError, id="str"),
    pytest.param(1.5, TypeError, id="float"),
    pytest.param(2.0, TypeError, id="integral-float"),
    pytest.param(-1, ValueError, id="negative"),
]
BAD_TIMEOUT = [
    pytest.param(True, TypeError, id="bool"),
    pytest.param("5", TypeError, id="str"),
    pytest.param(0, ValueError, id="zero"),
    pytest.param(0.0, ValueError, id="zero-float"),
    pytest.param(-1, ValueError, id="negative"),
    pytest.param(math.inf, ValueError, id="inf"),
    pytest.param(-math.inf, ValueError, id="minus-inf"),
    pytest.param(math.nan, ValueError, id="nan"),
]
BAD_IDLE = [
    pytest.param(True, TypeError, id="bool"),
    pytest.param("1", TypeError, id="str"),
    pytest.param(-1, ValueError, id="negative"),
    pytest.param(-0.5, ValueError, id="negative-float"),
    pytest.param(math.inf, ValueError, id="inf"),
    pytest.param(math.nan, ValueError, id="nan"),
]


@pytest.mark.parametrize(("value", "error"), BAD_MAX_RETRIES)
def test_max_retries_is_validated_at_construction(client_kind: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error) as info:
        MockAPI(ok_handler).client(client_kind, max_retries=value)
    assert str(info.value) == MAX_RETRIES_MESSAGE


@pytest.mark.parametrize(("value", "error"), BAD_MAX_RETRIES)
def test_max_retries_is_validated_per_call_before_any_request(
    client_kind: str, value: object, error: type[Exception]
) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    with pytest.raises(error) as info:
        run_request(client, "GET", "/models", options={"max_retries": value})
    assert str(info.value) == MAX_RETRIES_MESSAGE
    assert api.calls == []


@pytest.mark.parametrize(("value", "error"), BAD_TIMEOUT)
def test_timeout_is_validated_at_construction(client_kind: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error) as info:
        MockAPI(ok_handler).client(client_kind, timeout=value)
    assert str(info.value) == TIMEOUT_MESSAGE


@pytest.mark.parametrize(("value", "error"), BAD_TIMEOUT)
def test_timeout_is_validated_per_call_before_any_request(
    client_kind: str, value: object, error: type[Exception]
) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    with pytest.raises(error) as info:
        run_request(client, "GET", "/models", options={"timeout": value})
    assert str(info.value) == TIMEOUT_MESSAGE
    assert api.calls == []


@pytest.mark.parametrize(("value", "error"), BAD_IDLE)
def test_stream_idle_timeout_is_validated_at_construction(
    client_kind: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error) as info:
        MockAPI(ok_handler).client(client_kind, stream_idle_timeout=value)
    assert str(info.value) == IDLE_MESSAGE


@pytest.mark.parametrize(("value", "error"), BAD_IDLE)
def test_stream_idle_timeout_is_validated_per_call_before_any_request(
    client_kind: str, value: object, error: type[Exception]
) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    with pytest.raises(error) as info:
        run_request(client, "GET", "/models", options={"stream_idle_timeout": value})
    assert str(info.value) == IDLE_MESSAGE
    assert api.calls == []


def test_stream_idle_timeout_zero_means_off_and_is_stored_as_none(client_kind: str) -> None:
    client = MockAPI(ok_handler).client(client_kind, stream_idle_timeout=0)
    assert client.stream_idle_timeout is None


@pytest.mark.parametrize("value", [0, 1, 10])
def test_valid_max_retries_values_are_accepted(client_kind: str, value: int) -> None:
    client = MockAPI(ok_handler).client(client_kind, max_retries=value)
    assert client.max_retries == value


@pytest.mark.parametrize("value", [1, 0.25, 600.0])
def test_valid_timeout_values_are_accepted(client_kind: str, value: float) -> None:
    client = MockAPI(ok_handler).client(client_kind, timeout=value)
    assert client.timeout == value


# --- http_client -------------------------------------------------------------------------------------------------


def test_sync_client_rejects_an_async_http_client() -> None:
    http = httpx.AsyncClient()
    try:
        with pytest.raises(TypeError) as info:
            DevupAI(api_key=TEST_KEY, http_client=http)  # type: ignore[arg-type]
        assert str(info.value) == "http_client must be an httpx.Client."
    finally:
        anyio.run(http.aclose)


def test_async_client_rejects_a_sync_http_client() -> None:
    with httpx.Client() as http:
        with pytest.raises(TypeError) as info:
            AsyncDevupAI(api_key=TEST_KEY, http_client=http)  # type: ignore[arg-type]
        assert str(info.value) == "http_client must be an httpx.AsyncClient."


def test_http_client_of_another_type_raises_type_error(client_kind: str) -> None:
    with pytest.raises(TypeError) as info:
        if client_kind == "sync":
            DevupAI(api_key=TEST_KEY, http_client="not a client")  # type: ignore[arg-type]
        else:
            AsyncDevupAI(api_key=TEST_KEY, http_client="not a client")  # type: ignore[arg-type]
    expected = "httpx.Client" if client_kind == "sync" else "httpx.AsyncClient"
    assert str(info.value) == f"http_client must be an {expected}."


# --- headers sent ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "accept"), [("json", "application/json"), ("text", "*/*"), ("raw", "*/*")])
def test_accept_header_depends_on_the_response_kind(client_kind: str, kind: str, accept: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    result = run_request(client, "GET", "/models", kind=kind)
    if kind == "raw":
        close_response(client_kind, result.data)
    assert api.calls[0].headers["accept"] == accept


def test_user_agent_is_the_sdk_name_and_version(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    run_request(client, "GET", "/models")
    run_request(client, "POST", "/embeddings", body={"input": "x"})
    run_request(client, "GET", url=f"{BASE}/images/proxy?d=a&s=b", auth=False, kind="text")
    assert len(api.calls) == 3
    for call in api.calls:
        assert call.headers["user-agent"] == f"devupai-python/{devupai.__version__}"


def test_user_agent_can_be_overridden_through_default_headers(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, default_headers={"user-agent": "fixture-app/1"})
    run_request(client, "GET", "/models")
    assert api.calls[0].headers["user-agent"] == "fixture-app/1"


def test_user_agent_can_be_overridden_per_call(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, default_headers={"User-Agent": "fixture-app/1"})
    run_request(client, "GET", "/models", options={"extra_headers": {"USER-AGENT": "fixture-call/2"}})
    assert api.calls[0].headers["user-agent"] == "fixture-call/2"


def test_caller_authorization_in_any_case_is_replaced_by_the_api_key(client_kind: str) -> None:
    capture = Capture()
    client = MockAPI(capture).client(client_kind, default_headers={"AUTHORIZATION": "Bearer hacked"})
    run_request(
        client,
        "GET",
        "/models",
        options={"extra_headers": {"Authorization": "Bearer hacked", "aUtHoRiZaTiOn": "Bearer hacked-2"}},
    )
    assert capture.requests[0].headers.get_list("authorization") == [f"Bearer {TEST_KEY}"]


def test_auth_false_sends_no_authorization_even_when_the_caller_sets_one(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, default_headers={"Authorization": "Bearer caller", "X-Global": "true"})
    run_request(client, "GET", "/health", auth=False, options={"extra_headers": {"authorization": "Bearer call"}})
    assert "authorization" not in api.calls[0].headers
    assert api.calls[0].headers["x-global"] == "true"


def test_caller_content_type_in_any_case_is_dropped_and_json_body_gets_application_json(client_kind: str) -> None:
    capture = Capture()
    client = MockAPI(capture).client(client_kind, default_headers={"CONTENT-TYPE": "text/plain"})
    run_request(
        client,
        "POST",
        "/chat/completions",
        body={"model": MODEL_A},
        options={"extra_headers": {"Content-Type": "text/plain", "content-type": "multipart/form-data"}},
    )
    assert capture.requests[0].headers.get_list("content-type") == ["application/json"]


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_no_content_type_without_a_body_even_when_the_caller_sets_one(client_kind: str, method: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind, default_headers={"Content-Type": "application/json"})
    run_request(client, method, "/models", options={"extra_headers": {"content-type": "text/plain"}})
    assert "content-type" not in api.calls[0].headers
    assert api.calls[0].content == b""
    assert api.calls[0].method == method


def test_extra_headers_win_over_default_headers_case_insensitively(client_kind: str) -> None:
    capture = Capture()
    client = MockAPI(capture).client(client_kind, default_headers={"X-Fixture": "client", "X-Only-Client": "c"})
    run_request(client, "GET", "/models", options={"extra_headers": {"x-fixture": "call", "X-Only-Call": "k"}})
    headers = capture.requests[0].headers
    assert headers.get_list("x-fixture") == ["call"]
    assert headers["x-only-client"] == "c"
    assert headers["x-only-call"] == "k"


def test_default_headers_and_extra_headers_override_accept(client_kind: str) -> None:
    capture = Capture()
    client = MockAPI(capture).client(client_kind, default_headers={"accept": "text/plain"})
    run_request(client, "GET", "/models")
    run_request(client, "GET", "/models", options={"extra_headers": {"ACCEPT": "application/x-fixture"}})
    assert capture.requests[0].headers.get_list("accept") == ["text/plain"]
    assert capture.requests[1].headers.get_list("accept") == ["application/x-fixture"]


def test_if_none_match_argument_wins_over_the_header_in_extra_headers(client_kind: str) -> None:
    capture = Capture()
    client = MockAPI(capture).client(client_kind)
    run_request(
        client, "GET", "/models", if_none_match='"etag-arg"', options={"extra_headers": {"if-none-match": '"x"'}}
    )
    assert capture.requests[0].headers.get_list("if-none-match") == ['"etag-arg"']


def test_no_if_none_match_header_without_a_condition(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind), "GET", "/models")
    assert "if-none-match" not in api.calls[0].headers


# --- URL, query and body -----------------------------------------------------------------------------------------


def test_query_parameters_are_set_and_none_values_are_left_out(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    run_request(client, "GET", "/models", query={"a": "1", "skip": None, "c": "x y"})
    url = httpx.URL(api.calls[0].url)
    assert url.path == "/v1/models"
    assert url.params.get("a") == "1"
    assert url.params.get("c") == "x y"
    assert "skip" not in url.params


def test_query_with_only_none_values_leaves_the_url_unchanged(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind), "GET", "/models", query={"skip": None})
    assert api.calls[0].url == f"{BASE}/models"


def test_query_entries_replace_an_existing_parameter_of_the_same_name(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind), "GET", url=f"{BASE}/models?d=a&keep=1", query={"d": "z", "e": "b"})
    params = httpx.URL(api.calls[0].url).params
    assert params.get_list("d") == ["z"]
    assert params.get("keep") == "1"
    assert params.get("e") == "b"


@pytest.mark.parametrize(
    "url",
    [
        f"{BASE}/images/proxy?d=abc&s=%2F",
        f"{BASE}/images/proxy?d=signed%2Bquery&s=sig",
        "https://files.example.test/a/b.png?d=abc&s=%25",
    ],
)
def test_absolute_url_is_used_as_is(client_kind: str, url: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind), "GET", url=url, auth=False, kind="text")
    assert api.calls[0].url == url
    assert api.calls[0].method == "GET"
    assert "authorization" not in api.calls[0].headers


def test_json_body_is_compact_utf8_without_ascii_escapes(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    run_request(api.client(client_kind), "POST", "/embeddings", body={"a": 1, "b": "é"})
    call = api.calls[0]
    assert call.method == "POST"
    assert call.content == '{"a":1,"b":"é"}'.encode()
    assert call.json() == {"a": 1, "b": "é"}
    assert call.headers["content-type"] == "application/json"


def test_json_body_with_nan_raises_value_error_before_any_request(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    with pytest.raises(ValueError):
        run_request(api.client(client_kind), "POST", "/embeddings", body={"a": math.nan})
    assert api.calls == []
