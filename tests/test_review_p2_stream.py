"""Review P2, finding R2-1: the merged JSON body decides whether a call is a stream (Node ``chat.ts:83``).

``stream`` given through ``extra_body`` is part of the body that would be sent, so ``extra_body={"stream": True}``
raises ``NotImplementedError`` before any request, also when ``stream=False`` is passed as a body field.
``extra_body={"stream": False}`` is an ordinary JSON request.
"""

from __future__ import annotations

from typing import Any

import pytest
from helpers.mock_transport import MODEL_A, MockAPI, json_response, sequence
from helpers.samples import SAMPLES, invoke, method_of

MESSAGE = "available in a later build"

BASE_KWARGS: dict[str, dict[str, object]] = {
    "chat.completions.create": {"model": MODEL_A, "messages": [{"role": "user", "content": "Hello"}]},
    "responses.create": {"model": MODEL_A, "input": "Hello"},
}
NAMES = sorted(BASE_KWARGS)
TWIN = pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])


def make(kind: str, name: str) -> tuple[MockAPI, Any]:
    api = MockAPI(sequence(json_response(SAMPLES[name].response)))
    return api, api.client(kind)


@TWIN
@pytest.mark.parametrize("name", NAMES)
def test_extra_body_stream_true_raises_before_any_request(client_kind: str, name: str, with_response: bool) -> None:
    api, client = make(client_kind, name)
    with pytest.raises(NotImplementedError) as caught:
        invoke(method_of(client, name, with_response=with_response), **BASE_KWARGS[name], extra_body={"stream": True})
    assert caught.value.args == (MESSAGE,)
    assert api.calls == []


@TWIN
@pytest.mark.parametrize("name", NAMES)
def test_stream_false_with_extra_body_stream_true_raises(client_kind: str, name: str, with_response: bool) -> None:
    api, client = make(client_kind, name)
    with pytest.raises(NotImplementedError) as caught:
        invoke(
            method_of(client, name, with_response=with_response),
            **BASE_KWARGS[name],
            stream=False,
            extra_body={"stream": True},
        )
    assert caught.value.args == (MESSAGE,)
    assert api.calls == []


@TWIN
@pytest.mark.parametrize("name", NAMES)
def test_extra_body_stream_false_is_sent(client_kind: str, name: str, with_response: bool) -> None:
    api, client = make(client_kind, name)
    invoke(method_of(client, name, with_response=with_response), **BASE_KWARGS[name], extra_body={"stream": False})
    assert len(api.calls) == 1
    assert api.calls[0].json() == {**BASE_KWARGS[name], "stream": False}
