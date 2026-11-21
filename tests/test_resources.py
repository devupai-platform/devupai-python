"""The resources of both clients: what each method sends and what it returns (contract §3, P3 contract §3-§4).

Every method of ``tests/helpers/samples.py`` is called once per client kind against the recording mock: the HTTP
method, the exact URL, the headers, the body (JSON with ``extra_body`` merged last, or the multipart parts of a form),
the per-call ``timeout``, ``max_retries`` and ``extra_headers``, the typed return value (a model, or a
``BinaryResponse`` with the file bytes), and its ``with_response`` twin. Then the cases of one method each:
conditional ``models`` requests, the ``health`` URL, the model path of ``inference.run``, and the attributes of the
clients.

``devupai.types`` and ``devupai.resources`` are imported inside the tests, so each test reports on its own while the
package is incomplete.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from helpers.mock_transport import (
    BASE,
    MODEL_A,
    REQ_ID,
    TEST_KEY,
    Entry,
    MockAPI,
    error_response,
    json_response,
    sequence,
)
from helpers.samples import (
    BINARY_CONTENT,
    BINARY_CONTENT_TYPE,
    FORM_POST_METHODS,
    IMPLEMENTED,
    JSON_POST_METHODS,
    ORIGIN,
    RETURNS,
    SAMPLES,
    call_sample,
    expected_body,
    invoke,
    method_of,
    multipart_parts,
    resolve,
    return_type,
    sample_accept,
    sample_response,
    sample_url,
)

from devupai import APIResponse, AsyncDevupAI, DevupAI, DevupAPIError

ETAG = '"fixture-etag"'
#: The POST methods that send a JSON body and take ``extra_body`` (the multipart-only methods do not).
IMPLEMENTED_POST = JSON_POST_METHODS
CONDITIONAL = ("models.list", "models.retrieve")


def _types() -> Any:
    return importlib.import_module("devupai.types")


def _base() -> Any:
    return importlib.import_module("devupai.resources._base")


class Capture:
    """A mock handler that keeps each ``httpx.Request`` (with its extensions) and answers from a sequence."""

    def __init__(self, *entries: Entry) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = sequence(*entries)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def setup(kind: str, *entries: Entry, **opts: Any) -> tuple[MockAPI, Capture, DevupAI | AsyncDevupAI]:
    capture = Capture(*entries)
    api = MockAPI(capture)
    return api, capture, api.client(kind, **opts)


def not_modified() -> httpx.Response:
    return httpx.Response(304, headers={"etag": ETAG, "x-request-id": REQ_ID})


# --- the parsed fields of each return model --------------------------------------------------------------------------


def _check_chat(result: Any, t: Any) -> None:
    assert result.id == "chatcmpl-fixture"
    assert result.model == MODEL_A
    assert isinstance(result.choices[0].message, t.ChatCompletionMessage)
    assert result.choices[0].message.content == "Hello back"
    assert isinstance(result.devup, t.DevUpBilling)
    assert result.devup.cost_dzd == 1.5
    assert result.to_dict()["_devup"] == {"cost_dzd": 1.5, "balance_dzd": 98.5}


def _check_responses(result: Any, t: Any) -> None:
    assert result.id == "resp_fixture"
    assert result.status == "completed"
    assert result.output == []


def _check_model(result: Any, t: Any) -> None:
    assert result.id == MODEL_A
    assert result.object == "model"
    assert result.context_window is None
    assert isinstance(result.pricing, t.ModelPricing)
    assert result.pricing.currency == "DZD"


def _check_model_list(result: Any, t: Any) -> None:
    assert result.object == "list"
    assert isinstance(result.data[0], t.ModelObject)
    _check_model(result.data[0], t)


def _check_image(result: Any, t: Any) -> None:
    assert result.data[0].url == "https://cdn.example.test/fixture.png"
    assert isinstance(result.devup, t.DevUpBilling)
    assert result.devup.balance_dzd == 98.5


def _check_embedding(result: Any, t: Any) -> None:
    assert result.model == MODEL_A
    assert result.data[0].embedding == [0.0, 1.0]
    assert result.data[0].index == 0


def _check_rerank(result: Any, t: Any) -> None:
    assert result.results[0].relevance_score == 0.5
    assert isinstance(result.devup, t.DevUpBilling)


def _check_speech(result: Any, t: Any) -> None:
    assert result.url == "https://cdn.example.test/fixture.mp3"
    assert isinstance(result.devup, t.DevUpBilling)


def _check_video(result: Any, t: Any) -> None:
    assert result.data[0].url == "https://cdn.example.test/fixture.mp4"
    assert isinstance(result.devup, t.DevUpBilling)


def _check_inference(result: Any, t: Any) -> None:
    assert result == {"ok": True}


def _check_transcription(result: Any, t: Any) -> None:
    assert result.text == "Hello"
    assert isinstance(result.devup, t.DevUpBilling)
    assert result.devup.cost_dzd == 1.5


def _check_binary(result: Any, t: Any) -> None:
    # ``call_sample`` read the body inside the event loop, so the bytes are on the wrapped response.
    assert isinstance(result.response, httpx.Response)
    assert result.response.content == BINARY_CONTENT
    assert result.status_code == 200
    assert result.headers["content-type"] == BINARY_CONTENT_TYPE
    assert result.request_id == REQ_ID


def _check_balance(result: Any, t: Any) -> None:
    assert result.balance_dzd == 10.5


def _check_me(result: Any, t: Any) -> None:
    assert result.object == "account_profile"
    assert result.initial == "F"
    assert result.email_hint is None
    assert result.avatar is None


def _check_health(result: Any, t: Any) -> None:
    assert result.status == "ok"


CHECKS: dict[str, Callable[[Any, Any], None]] = {
    "chat.completions.create": _check_chat,
    "responses.create": _check_responses,
    "models.list": _check_model_list,
    "models.retrieve": _check_model,
    "images.generate": _check_image,
    "images.edit": _check_image,
    "images.download": _check_binary,
    "embeddings.create": _check_embedding,
    "rerank.create": _check_rerank,
    "audio.speech.create": _check_speech,
    "audio.transcriptions.create": _check_transcription,
    "video.generations.create": _check_video,
    "video.edits.create": _check_video,
    "inference.run": _check_inference,
    "balance.retrieve": _check_balance,
    "me.retrieve": _check_me,
    "me.avatar": _check_binary,
    "health.check": _check_health,
}


def test_the_tables_cover_every_implemented_method() -> None:
    assert set(CHECKS) == set(RETURNS) == set(IMPLEMENTED) == set(SAMPLES)
    assert set(FORM_POST_METHODS) == {"audio.transcriptions.create", "video.edits.create"}
    assert {name for name, sample in SAMPLES.items() if sample.binary} == {"images.download", "me.avatar"}


def assert_return(name: str, result: Any, client_kind: str) -> None:
    t = _types()
    type_name = RETURNS[name]
    if type_name is not None:
        expected = return_type(type_name, client_kind)
        assert isinstance(result, expected), f"{name} returned {type(result).__name__}, not {expected.__name__}"
    CHECKS[name](result, t)


# --- every method -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_sends_the_method_url_headers_and_body(client_kind: str, name: str) -> None:
    sample = SAMPLES[name]
    api, _, client = setup(client_kind, sample_response(name))
    call_sample(client, name)
    assert len(api.calls) == 1
    request = api.calls[0]
    assert request.method == sample.http_method
    assert request.url == sample_url(name)
    if sample.auth:
        assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    else:
        assert name in ("health.check", "images.download")
        assert "authorization" not in request.headers
    assert request.headers["accept"] == sample_accept(name)
    assert "if-none-match" not in request.headers
    body = expected_body(name)
    if sample.form is not None:
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        assert multipart_parts(request) == list(sample.form)
    elif body is None:
        assert request.content == b""
        assert "content-type" not in request.headers
    else:
        assert request.headers["content-type"] == "application/json"
        assert request.json() == body


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_returns_the_typed_model(client_kind: str, name: str) -> None:
    _, _, client = setup(client_kind, sample_response(name))
    assert_return(name, call_sample(client, name), client_kind)


@pytest.mark.parametrize("name", IMPLEMENTED_POST)
def test_extra_body_is_merged_last_and_wins(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, sample_response(name))
    body = expected_body(name)
    assert body is not None
    first_key = next(iter(body))
    extra = {first_key: "fixture-override", "fixture_extra": {"nested": [1, 2]}}
    call_sample(client, name, extra_body=extra)
    sent = api.calls[0].json()
    assert sent == {**body, **extra}
    assert list(sent)[-1] == "fixture_extra"


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_extra_headers_are_sent(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, sample_response(name))
    call_sample(client, name, extra_headers={"X-Fixture": "1", "X-Fixture-Two": "two"})
    headers = api.calls[0].headers
    assert headers["x-fixture"] == "1"
    assert headers["x-fixture-two"] == "two"
    assert headers["accept"] == sample_accept(name)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_per_call_timeout_sets_the_request_timeout(client_kind: str, name: str) -> None:
    _, capture, client = setup(client_kind, sample_response(name))
    call_sample(client, name, timeout=5.0)
    timeout = capture.requests[0].extensions["timeout"]
    assert timeout["read"] == 5.0
    assert timeout["connect"] == 5.0


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_without_a_timeout_the_client_timeout_applies(client_kind: str, name: str) -> None:
    _, capture, client = setup(client_kind, sample_response(name), timeout=30.0)
    call_sample(client, name)
    timeout = capture.requests[0].extensions["timeout"]
    assert timeout["read"] == 30.0
    assert timeout["connect"] == 10.0


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_max_retries_zero_sends_once(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, error_response(503), sample_response(name))
    with pytest.raises(DevupAPIError) as caught:
        call_sample(client, name, max_retries=0)
    assert caught.value.status == 503
    assert len(api.calls) == 1


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_max_retries_per_call_overrides_the_client(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, error_response(503), error_response(503), sample_response(name), max_retries=0)
    assert_return(name, call_sample(client, name, max_retries=2), client_kind)
    assert len(api.calls) == 3


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_with_response_returns_the_api_response(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, sample_response(name))
    result = call_sample(client, name, with_response=True)
    assert isinstance(result, APIResponse)
    assert_return(name, result.data, client_kind)
    assert result.request_id == REQ_ID
    assert result.response.status_code == 200
    assert len(api.calls) == 1
    assert api.calls[0].url == sample_url(name)


@pytest.mark.parametrize("name", IMPLEMENTED_POST)
def test_with_response_takes_the_same_options(client_kind: str, name: str) -> None:
    api, capture, client = setup(client_kind, sample_response(name))
    body = expected_body(name)
    assert body is not None
    result = call_sample(
        client, name, with_response=True, extra_body={"fixture_extra": 1}, extra_headers={"X-Fixture": "1"}, timeout=7.0
    )
    assert isinstance(result, APIResponse)
    assert api.calls[0].json() == {**body, "fixture_extra": 1}
    assert api.calls[0].headers["x-fixture"] == "1"
    assert capture.requests[0].extensions["timeout"]["read"] == 7.0


# --- the multipart-only methods (Node's fixed field lists) -----------------------------------------------------------


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
@pytest.mark.parametrize("name", FORM_POST_METHODS)
def test_form_methods_take_the_common_options(client_kind: str, name: str, with_response: bool) -> None:
    api, capture, client = setup(client_kind, sample_response(name))
    result = call_sample(client, name, with_response=with_response, extra_headers={"X-Fixture": "1"}, timeout=7.0)
    assert isinstance(result, APIResponse) is with_response
    assert len(api.calls) == 1
    assert multipart_parts(api.calls[0]) == list(SAMPLES[name].form or ())
    assert api.calls[0].headers["x-fixture"] == "1"
    assert capture.requests[0].extensions["timeout"]["read"] == 7.0


@pytest.mark.parametrize(
    "extra", [{"fixture_extra": "x"}, {"extra_body": {"fixture_extra_body": 1}}], ids=["field", "extra_body"]
)
@pytest.mark.parametrize("name", FORM_POST_METHODS)
def test_form_methods_send_only_their_listed_fields(client_kind: str, name: str, extra: dict[str, object]) -> None:
    # Node sends only its fixed list. In Python these methods take explicit keyword-only parameters (lead decision
    # in P3, matching tests/test_multipart.py): a field outside the list, or ``extra_body``, is a TypeError and
    # nothing is sent; the listed fields alone make up the form.
    api, _, client = setup(client_kind, sample_response(name))
    with pytest.raises(TypeError):
        call_sample(client, name, **extra)
    assert api.calls == []
    call_sample(client, name)
    assert multipart_parts(api.calls[0]) == list(SAMPLES[name].form or ())


# --- images.download: no key, and only the API's own origin ---------------------------------------------------------


def test_images_download_never_sends_authorization_even_from_extra_headers(client_kind: str) -> None:
    api, _, client = setup(client_kind, sample_response("images.download"))
    call_sample(client, "images.download", extra_headers={"Authorization": "Bearer per-call", "X-Fixture": "1"})
    assert "authorization" not in api.calls[0].headers
    assert api.calls[0].headers["x-fixture"] == "1"
    assert api.calls[0].url == sample_url("images.download")


def test_images_download_refuses_another_origin_before_any_request(client_kind: str) -> None:
    api, _, client = setup(client_kind, sample_response("images.download"))
    foreign = "https://cdn.example.test/fixture.png"
    with pytest.raises(DevupAPIError) as caught:
        invoke(method_of(client, "images.download"), foreign)
    assert type(caught.value) is DevupAPIError
    assert caught.value.status == 0
    assert caught.value.code == "invalid_url"
    assert caught.value.message == "images.download only fetches URLs returned by the API."
    assert foreign not in str(caught.value)
    assert api.calls == []


# --- conditional requests on models --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONDITIONAL)
def test_if_none_match_is_sent_and_304_returns_none(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, not_modified())
    assert call_sample(client, name, if_none_match=ETAG) is None
    assert len(api.calls) == 1
    assert api.calls[0].headers["if-none-match"] == ETAG
    assert api.calls[0].url == ORIGIN + SAMPLES[name].path


@pytest.mark.parametrize("name", CONDITIONAL)
def test_if_none_match_that_no_longer_matches_returns_the_model(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, json_response(SAMPLES[name].response, headers={"etag": ETAG}))
    assert_return(name, call_sample(client, name, if_none_match='"fixture-old-etag"'), client_kind)
    assert api.calls[0].headers["if-none-match"] == '"fixture-old-etag"'


@pytest.mark.parametrize("name", CONDITIONAL)
def test_with_response_on_304_carries_the_status_and_request_id(client_kind: str, name: str) -> None:
    _, _, client = setup(client_kind, not_modified())
    result = call_sample(client, name, with_response=True, if_none_match=ETAG)
    assert isinstance(result, APIResponse)
    assert result.data is None
    assert result.response.status_code == 304
    assert result.response.headers["etag"] == ETAG
    assert result.request_id == REQ_ID


@pytest.mark.parametrize("name", CONDITIONAL)
def test_if_none_match_wins_over_the_same_header_in_extra_headers(client_kind: str, name: str) -> None:
    api, _, client = setup(client_kind, not_modified())
    result = call_sample(
        client, name, if_none_match=ETAG, extra_headers={"If-None-Match": '"fixture-other"', "X-Fixture": "1"}
    )
    assert result is None
    assert api.calls[0].headers["if-none-match"] == ETAG
    assert api.calls[0].headers["x-fixture"] == "1"


@pytest.mark.parametrize("name", CONDITIONAL)
def test_a_304_without_if_none_match_raises(client_kind: str, name: str) -> None:
    _, _, client = setup(client_kind, not_modified())
    with pytest.raises(DevupAPIError) as caught:
        call_sample(client, name)
    assert caught.value.status == 304


# --- health -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (BASE, "https://api.example.test/health"),
        ("https://api.example.test/api/v1", "https://api.example.test/api/health"),
        ("http://localhost:3000/api/v1", "http://localhost:3000/api/health"),
    ],
)
def test_health_url_is_next_to_the_versioned_path(client_kind: str, base_url: str, expected: str) -> None:
    api, _, client = setup(client_kind, json_response({"status": "ok"}), base_url=base_url)
    assert_return("health.check", invoke(method_of(client, "health.check")), client_kind)
    assert api.calls[0].url == expected
    assert api.calls[0].method == "GET"


def test_health_never_sends_authorization_even_from_extra_headers(client_kind: str) -> None:
    api, _, client = setup(client_kind, json_response({"status": "ok"}))
    invoke(method_of(client, "health.check"), extra_headers={"Authorization": "Bearer per-call", "X-Fixture": "1"})
    assert "authorization" not in api.calls[0].headers
    assert api.calls[0].headers["x-fixture"] == "1"


# --- path encoding ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_path", "encoded"),
    [
        (MODEL_A, "fixture%2Fmodel-a"),
        ("/fixture/model-a", "fixture%2Fmodel-a"),
        ("///fixture/model-a", "fixture%2Fmodel-a"),
        ("fixture%2Fmodel-a", "fixture%2Fmodel-a"),
        ("/fixture%2Fmodel-a", "fixture%2Fmodel-a"),
        ("fixture/model a", "fixture%2Fmodel%20a"),
        ("fixture/model%20a", "fixture%2Fmodel%20a"),
        ("fixture/model!*'()", "fixture%2Fmodel!*'()"),
        ("fixture/model?a#b&c=d", "fixture%2Fmodel%3Fa%23b%26c%3Dd"),
        ("fixture%zz", "fixture%25zz"),
    ],
)
def test_inference_model_path_is_one_encoded_segment(client_kind: str, model_path: str, encoded: str) -> None:
    api, _, client = setup(client_kind, json_response({"ok": True}))
    result = invoke(method_of(client, "inference.run"), model_path, {"test": True})
    assert result == {"ok": True}
    assert api.calls[0].url == f"{BASE}/inference/{encoded}"
    assert api.calls[0].json() == {"test": True}


@pytest.mark.parametrize(
    ("model", "encoded"),
    [
        (MODEL_A, "fixture%2Fmodel-a"),
        ("fixture/model a?b#c", "fixture%2Fmodel%20a%3Fb%23c"),
        ("fixture/model!*'()", "fixture%2Fmodel!*'()"),
        ("fixture%2Fmodel-a", "fixture%252Fmodel-a"),
    ],
)
def test_models_retrieve_encodes_the_id_as_one_segment(client_kind: str, model: str, encoded: str) -> None:
    api, _, client = setup(client_kind, json_response(SAMPLES["models.retrieve"].response))
    invoke(method_of(client, "models.retrieve"), model)
    assert api.calls[0].url == f"{BASE}/models/{encoded}"


# --- client wiring ------------------------------------------------------------------------------------------------

#: Attribute path on the client → class name on ``DevupAI`` (``Async`` + name on ``AsyncDevupAI``).
WIRING: dict[str, str] = {
    "chat": "Chat",
    "chat.completions": "Completions",
    "responses": "Responses",
    "models": "Models",
    "images": "Images",
    "embeddings": "Embeddings",
    "rerank": "Rerank",
    "audio": "Audio",
    "audio.speech": "Speech",
    "audio.transcriptions": "Transcriptions",
    "video": "Video",
    "video.generations": "VideoGenerations",
    "video.edits": "VideoEdits",
    "inference": "NativeInference",
    "balance": "Balance",
    "me": "Me",
    "health": "Health",
}

#: The resources that carry API methods, and so a ``with_response`` twin.
WITH_METHODS = sorted({name.rpartition(".")[0] for name in SAMPLES})


def test_wiring_covers_every_sample_resource() -> None:
    assert set(WITH_METHODS) <= set(WIRING)


@pytest.mark.parametrize("path", sorted(WIRING))
def test_client_exposes_each_resource(client_kind: str, path: str) -> None:
    base = _base()
    _, _, client = setup(client_kind, json_response({}))
    resource = resolve(client, path)
    prefix = "Async" if client_kind == "async" else ""
    assert type(resource).__name__ == prefix + WIRING[path]
    assert type(resource).__module__.startswith("devupai.resources")
    expected_base = base.AsyncAPIResource if client_kind == "async" else base.SyncAPIResource
    assert isinstance(resource, expected_base)
    other_base = base.SyncAPIResource if client_kind == "async" else base.AsyncAPIResource
    assert not isinstance(resource, other_base)
    assert resource._client is client


@pytest.mark.parametrize("path", WITH_METHODS)
def test_with_response_twin_has_the_documented_name(client_kind: str, path: str) -> None:
    _, _, client = setup(client_kind, json_response({}))
    twin = resolve(client, path).with_response
    prefix = "Async" if client_kind == "async" else ""
    assert type(twin).__name__ == f"{prefix}{WIRING[path]}WithResponse"
    for name in SAMPLES:
        resource_path, _, method = name.rpartition(".")
        if resource_path == path:
            assert callable(getattr(twin, method))


def test_each_client_kind_uses_its_own_classes() -> None:
    api = MockAPI(sequence(json_response({})))
    sync_client = api.sync_client()
    async_client = api.async_client()
    for path in WIRING:
        assert type(resolve(sync_client, path)) is not type(resolve(async_client, path))


def test_webhooks_is_a_webhooks_instance_on_both_clients(client_kind: str) -> None:
    devupai = importlib.import_module("devupai")
    _, _, client = setup(client_kind, json_response({}))
    assert isinstance(client.webhooks, devupai.Webhooks)
    assert "Webhooks" in devupai.__all__
