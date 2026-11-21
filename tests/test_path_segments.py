"""Path segments (decision answer 3, contract §5): a whole segment ``.`` or ``..`` is sent as ``%2E`` / ``%2E%2E``.

A URL parser removes the dot segments ``.`` and ``..`` from a path, so ``/v1/models/..`` would reach ``/v1``. Every
path parameter (a model id, a model path) is therefore sent with those two segments percent-encoded. Every other
value is encoded as in P2: a segment that only contains dots among other characters (``a.b``, ``...``, ``./x``) is
unchanged.

The tests check the two encoders, then the raw path of the request each client sends (``httpx`` ``raw_path``, which
shows whether the path collapsed).
"""

from __future__ import annotations

import importlib
from typing import Any

import httpx
import pytest
from helpers.mock_transport import BASE, MODEL_A, Entry, MockAPI, json_response, sequence
from helpers.samples import SAMPLES, invoke, method_of

from devupai import AsyncDevupAI, DevupAI


def _base() -> Any:
    return importlib.import_module("devupai.resources._base")


class Capture:
    """A mock handler that keeps each ``httpx.Request`` and answers from a sequence."""

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


# --- encode_path_segment --------------------------------------------------------------------------------------------

SEGMENT_CASES = [
    pytest.param(".", "%2E", id="dot"),
    pytest.param("..", "%2E%2E", id="dot-dot"),
    pytest.param("a.b", "a.b", id="dot-inside"),
    pytest.param("...", "...", id="three-dots"),
    pytest.param("....", "....", id="four-dots"),
    pytest.param(".x", ".x", id="leading-dot"),
    pytest.param("x.", "x.", id="trailing-dot"),
    pytest.param("..x", "..x", id="leading-dot-dot"),
    pytest.param("./x", ".%2Fx", id="dot-slash"),
    pytest.param("../x", "..%2Fx", id="dot-dot-slash"),
    pytest.param(" .", "%20.", id="space-dot"),
    pytest.param("%2E", "%252E", id="percent-not-decoded"),
    pytest.param("fixture/model-a", "fixture%2Fmodel-a", id="slash"),
    pytest.param("fixture/model!*'()", "fixture%2Fmodel!*'()", id="kept-characters"),
]


@pytest.mark.parametrize(("value", "encoded"), SEGMENT_CASES)
def test_encode_path_segment(value: str, encoded: str) -> None:
    assert _base().encode_path_segment(value) == encoded


# --- encode_model_path ----------------------------------------------------------------------------------------------

MODEL_PATH_CASES = [
    pytest.param(".", "%2E", id="dot"),
    pytest.param("..", "%2E%2E", id="dot-dot"),
    pytest.param("/.", "%2E", id="slash-dot"),
    pytest.param("/..", "%2E%2E", id="slash-dot-dot"),
    pytest.param("///..", "%2E%2E", id="slashes-dot-dot"),
    pytest.param("%2e", "%2E", id="encoded-dot-lower"),
    pytest.param("%2E", "%2E", id="encoded-dot-upper"),
    pytest.param("%2e%2e", "%2E%2E", id="encoded-dot-dot-lower"),
    pytest.param("%2E%2E", "%2E%2E", id="encoded-dot-dot-upper"),
    pytest.param(".%2e", "%2E%2E", id="half-encoded-dot-dot"),
    pytest.param("/%2e%2e", "%2E%2E", id="slash-encoded-dot-dot"),
    pytest.param("%2E%2E%2E", "...", id="encoded-three-dots"),
    pytest.param("%2E%2E%2Fx", "..%2Fx", id="encoded-dot-dot-slash"),
    pytest.param("a.b", "a.b", id="dot-inside"),
    pytest.param("...", "...", id="three-dots"),
    pytest.param("./x", ".%2Fx", id="dot-slash"),
    pytest.param("../x", "..%2Fx", id="dot-dot-slash"),
    pytest.param(MODEL_A, "fixture%2Fmodel-a", id="model-id"),
    pytest.param("/fixture/model-a", "fixture%2Fmodel-a", id="leading-slash"),
    pytest.param("fixture%2Fmodel-a", "fixture%2Fmodel-a", id="already-encoded"),
    pytest.param("fixture%zz", "fixture%25zz", id="invalid-percent"),
]


@pytest.mark.parametrize(("value", "encoded"), MODEL_PATH_CASES)
def test_encode_model_path(value: str, encoded: str) -> None:
    assert _base().encode_model_path(value) == encoded


# --- through the clients --------------------------------------------------------------------------------------------

RETRIEVE_CASES = [
    pytest.param(".", "%2E", id="dot"),
    pytest.param("..", "%2E%2E", id="dot-dot"),
    pytest.param("a.b", "a.b", id="dot-inside"),
    pytest.param("...", "...", id="three-dots"),
    pytest.param("./x", ".%2Fx", id="dot-slash"),
    pytest.param("../x", "..%2Fx", id="dot-dot-slash"),
    pytest.param(MODEL_A, "fixture%2Fmodel-a", id="model-id"),
]


def assert_raw_path(request: httpx.Request, expected: str) -> None:
    """The path reached the mock as built: no dot segment was removed."""
    assert request.url.raw_path == expected.encode("ascii")
    assert str(request.url) == f"https://api.example.test{expected}"


@pytest.mark.parametrize(("model", "encoded"), RETRIEVE_CASES)
def test_models_retrieve_sends_the_segment_without_collapsing(client_kind: str, model: str, encoded: str) -> None:
    _, capture, client = setup(client_kind, json_response(SAMPLES["models.retrieve"].response))
    invoke(method_of(client, "models.retrieve"), model)
    assert len(capture.requests) == 1
    assert capture.requests[0].method == "GET"
    assert_raw_path(capture.requests[0], f"/v1/models/{encoded}")


@pytest.mark.parametrize(
    ("model", "encoded"), [pytest.param(".", "%2E", id="dot"), pytest.param("..", "%2E%2E", id="dot-dot")]
)
def test_models_retrieve_with_response_twin_encodes_the_same(client_kind: str, model: str, encoded: str) -> None:
    _, capture, client = setup(client_kind, json_response(SAMPLES["models.retrieve"].response))
    invoke(method_of(client, "models.retrieve", with_response=True), model)
    assert_raw_path(capture.requests[0], f"/v1/models/{encoded}")


def test_models_retrieve_keeps_the_base_path(client_kind: str) -> None:
    _, capture, client = setup(
        client_kind, json_response(SAMPLES["models.retrieve"].response), base_url="http://localhost:3000/api/v1"
    )
    invoke(method_of(client, "models.retrieve"), "..")
    assert capture.requests[0].url.raw_path == b"/api/v1/models/%2E%2E"
    assert str(capture.requests[0].url) == "http://localhost:3000/api/v1/models/%2E%2E"


INFERENCE_CASES = [
    pytest.param("..", "%2E%2E", id="dot-dot"),
    pytest.param(".", "%2E", id="dot"),
    pytest.param("/..", "%2E%2E", id="slash-dot-dot"),
    pytest.param("%2e%2e", "%2E%2E", id="encoded-dot-dot"),
    pytest.param("%2E", "%2E", id="encoded-dot"),
    pytest.param("/fixture/model-a", "fixture%2Fmodel-a", id="leading-slash"),
    pytest.param("./x", ".%2Fx", id="dot-slash"),
    pytest.param("a.b", "a.b", id="dot-inside"),
    pytest.param("...", "...", id="three-dots"),
]


@pytest.mark.parametrize(("model_path", "encoded"), INFERENCE_CASES)
def test_inference_run_sends_the_segment_without_collapsing(client_kind: str, model_path: str, encoded: str) -> None:
    api, capture, client = setup(client_kind, json_response({"ok": True}))
    result = invoke(method_of(client, "inference.run"), model_path, {"input": "x"})
    assert result == {"ok": True}
    assert len(capture.requests) == 1
    assert capture.requests[0].method == "POST"
    assert_raw_path(capture.requests[0], f"/v1/inference/{encoded}")
    assert api.calls[0].json() == {"input": "x"}


@pytest.mark.parametrize(
    ("model_path", "encoded"),
    [pytest.param("..", "%2E%2E", id="dot-dot"), pytest.param("%2e", "%2E", id="encoded-dot")],
)
def test_inference_run_with_response_twin_encodes_the_same(client_kind: str, model_path: str, encoded: str) -> None:
    _, capture, client = setup(client_kind, json_response({"ok": True}))
    invoke(method_of(client, "inference.run", with_response=True), model_path, {"input": "x"})
    assert_raw_path(capture.requests[0], f"/v1/inference/{encoded}")


def test_the_base_url_constant_is_the_mock_origin() -> None:
    # assert_raw_path builds the full URL from this origin.
    assert BASE == "https://api.example.test/v1"
