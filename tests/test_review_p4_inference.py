"""``inference.run`` accepts only ``response_type="json"`` or ``"binary"`` (STOP-3 answer 2).

Any other value, a typo such as ``"bytes"`` or ``"JSON"`` or a value that is not a string, raises ``ValueError``
before anything is sent: the mock records 0 requests. The two accepted values keep their behaviour. Every case runs
on both clients, through ``client.inference.run`` and through ``client.inference.with_response.run``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
import httpx
import pytest
from helpers.mock_transport import BASE, MODEL_A, MockAPI, json_response, sequence

from devupai import APIResponse, AsyncBinaryResponse, AsyncDevupAI, BinaryResponse, DevupAI

MESSAGE = 'response_type must be "json" or "binary".'
PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10])

#: Values that are not one of the two accepted strings.
REJECTED: list[object] = ["bytes", "JSON", "", None, 1, "Binary", " json", "binary\n", b"json", True]
REJECTED_IDS = [
    "bytes",
    "JSON",
    "empty",
    "None",
    "int-1",
    "Binary",
    "leading-space",
    "trailing-newline",
    "b-json",
    "True",
]


def _run_method(client: DevupAI | AsyncDevupAI, with_response: bool) -> Callable[..., Any]:
    inference = client.inference.with_response if with_response else client.inference
    method: Callable[..., Any] = inference.run
    return method


def _call(kind: str, client: DevupAI | AsyncDevupAI, with_response: bool, *args: Any, **kwargs: Any) -> Any:
    """Calls ``inference.run`` (or its ``with_response`` twin); an async client runs inside ``anyio.run``."""
    method = _run_method(client, with_response)
    if kind == "sync":
        return method(*args, **kwargs)

    async def main() -> Any:
        return await method(*args, **kwargs)

    return anyio.run(main)


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
@pytest.mark.parametrize("value", REJECTED, ids=REJECTED_IDS)
def test_other_response_type_raises_value_error_before_sending(
    client_kind: str, with_response: bool, value: object
) -> None:
    api = MockAPI(sequence(json_response({"ok": True})))
    client = api.client(client_kind)
    with pytest.raises(ValueError) as caught:
        _call(client_kind, client, with_response, MODEL_A, {"input": "x"}, response_type=value)
    assert str(caught.value) == MESSAGE
    assert not isinstance(caught.value, TypeError)
    assert api.calls == []


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
def test_a_file_payload_with_a_rejected_response_type_sends_nothing(client_kind: str, with_response: bool) -> None:
    # The check comes before the payload is turned into multipart parts.
    api = MockAPI(sequence(json_response({"ok": True})))
    client = api.client(client_kind)
    with pytest.raises(ValueError, match="response_type"):
        _call(client_kind, client, with_response, MODEL_A, {"image": PNG}, response_type="bytes")
    assert api.calls == []


class _PrepareReached(Exception):
    """Raised by the patched preparation step: the call got past the ``response_type`` check."""


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
@pytest.mark.parametrize(("response_type", "expected"), [("bytes", ValueError), ("binary", _PrepareReached)])
def test_the_response_type_check_runs_before_the_request_is_prepared(
    client_kind: str,
    with_response: bool,
    response_type: str,
    expected: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review C-2: a rejected value never reaches the step that reads the payload (files are opened there). The
    # accepted value reaching the patched step shows the patch is the one the call uses.
    import devupai.resources.inference as inference_module

    def prepare(*args: object, **kwargs: object) -> object:
        raise _PrepareReached

    monkeypatch.setattr(inference_module, "_prepare", prepare)
    api = MockAPI(sequence(json_response({"ok": True})))
    client = api.client(client_kind)
    with pytest.raises(expected):
        _call(client_kind, client, with_response, MODEL_A, {"input": "x"}, response_type=response_type)
    assert api.calls == []


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
@pytest.mark.parametrize("kwargs", [{}, {"response_type": "json"}], ids=["default", "json"])
def test_json_still_returns_parsed_json(client_kind: str, with_response: bool, kwargs: dict[str, Any]) -> None:
    api = MockAPI(sequence(json_response({"output": [1, 2]})))
    client = api.client(client_kind)
    result = _call(client_kind, client, with_response, MODEL_A, {"input": "x"}, **kwargs)
    if with_response:
        assert isinstance(result, APIResponse)
        result = result.data
    assert result == {"output": [1, 2]}
    assert len(api.calls) == 1
    assert api.calls[0].method == "POST"
    assert api.calls[0].url == f"{BASE}/inference/fixture%2Fmodel-a"
    assert api.calls[0].json() == {"input": "x"}


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
def test_binary_still_returns_a_binary_response(client_kind: str, with_response: bool) -> None:
    api = MockAPI(sequence(httpx.Response(200, headers={"content-type": "image/png"}, content=PNG)))
    client = api.client(client_kind)
    method = _run_method(client, with_response)
    expected = BinaryResponse if client_kind == "sync" else AsyncBinaryResponse

    def check(result: object) -> object:
        if with_response:
            assert isinstance(result, APIResponse)
            result = result.data
        assert isinstance(result, expected)
        return result

    if client_kind == "sync":
        result = check(method(MODEL_A, {"input": "x"}, response_type="binary"))
        assert isinstance(result, BinaryResponse)
        with result:
            assert result.read() == PNG
    else:

        async def main() -> bytes:
            result = check(await method(MODEL_A, {"input": "x"}, response_type="binary"))
            assert isinstance(result, AsyncBinaryResponse)
            async with result:
                return await result.read()

        assert anyio.run(main) == PNG
    assert len(api.calls) == 1
    assert api.calls[0].url == f"{BASE}/inference/fixture%2Fmodel-a"
