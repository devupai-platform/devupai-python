"""Forms that come in a later build raise ``NotImplementedError`` before anything is sent (decisions B item 4).

Streaming, file inputs (multipart) and binary answers raise ``NotImplementedError("available in a later build")``
on both clients, with no request at all: never a silent non-stream or JSON request in their place. ``stream=False``
is an ordinary JSON request and is sent as given.
"""

from __future__ import annotations

import importlib
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from helpers.mock_transport import MODEL_A, MockAPI, json_response, sequence
from helpers.samples import SAMPLES, invoke, method_of

MESSAGE = "available in a later build"
ETAG = '"fixture-etag"'
URL_INPUT = "https://cdn.example.test/input.png"
PNG = b"\x89PNG\r\n\x1a\n"

#: Each kind of file input of ``FileInput`` (contract §2.1), built fresh for every test.
FILE_KINDS: dict[str, Callable[[], object]] = {
    "bytes": lambda: PNG,
    "bytearray": lambda: bytearray(PNG),
    "memoryview": lambda: memoryview(PNG),
    "path": lambda: Path("fixture-input.png"),
    "bytesio": lambda: io.BytesIO(PNG),
    "tuple": lambda: ("input.png", PNG),
    "tuple-with-type": lambda: ("input.png", io.BytesIO(PNG), "image/png"),
}

MESSAGES = [{"role": "user", "content": "Hello"}]


class Case:
    """One call that must raise: the method name, positional arguments and keyword arguments."""

    def __init__(self, name: str, args: tuple[object, ...], kwargs: Callable[[], dict[str, object]]) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs


def _cases() -> dict[str, Case]:
    cases: dict[str, Case] = {
        "chat-stream": Case(
            "chat.completions.create", (), lambda: {"model": MODEL_A, "messages": MESSAGES, "stream": True}
        ),
        "responses-stream": Case("responses.create", (), lambda: {"model": MODEL_A, "input": "Hello", "stream": True}),
        "images-download": Case("images.download", SAMPLES["images.download"].args, dict),
        "audio-transcriptions": Case(
            "audio.transcriptions.create", (), lambda: {"model": MODEL_A, "file": FILE_KINDS["bytes"]()}
        ),
        "audio-transcriptions-path": Case(
            "audio.transcriptions.create", (), lambda: {"file": FILE_KINDS["path"](), "language": "en"}
        ),
        "video-edits": Case("video.edits.create", (), lambda: {"model": MODEL_A, "video": FILE_KINDS["bytes"]()}),
        "video-edits-with-mask": Case(
            "video.edits.create",
            (),
            lambda: {"model": MODEL_A, "video": FILE_KINDS["bytesio"](), "mask": FILE_KINDS["bytes"](), "prompt": "x"},
        ),
        "inference-binary": Case("inference.run", (MODEL_A, {"inputs": "hello"}), lambda: {"response_type": "binary"}),
        "me-avatar": Case("me.avatar", (), dict),
        "me-avatar-if-none-match": Case("me.avatar", (), lambda: {"if_none_match": ETAG}),
    }

    def add(key: str, name: str, args: tuple[object, ...], kwargs: Callable[[], dict[str, object]]) -> None:
        assert key not in cases, key
        cases[key] = Case(name, args, kwargs)

    def edit_kwargs(
        field: str, make: Callable[[], object], as_list: bool, mixed: bool
    ) -> Callable[[], dict[str, object]]:
        def build() -> dict[str, object]:
            value: object = make()
            if mixed:
                value = [URL_INPUT, value]
            elif as_list:
                value = [value]
            return {"model": MODEL_A, "prompt": "Make it blue", field: value}

        return build

    for kind, make in FILE_KINDS.items():
        add(f"images-edit-image-{kind}", "images.edit", (), edit_kwargs("image", make, False, False))
        add(f"images-edit-image-list-{kind}", "images.edit", (), edit_kwargs("image", make, False, True))
        add(f"images-edit-images-{kind}", "images.edit", (), edit_kwargs("images", make, True, False))
        add(f"images-edit-images-mixed-{kind}", "images.edit", (), edit_kwargs("images", make, False, True))
        add(f"images-edit-mask-{kind}", "images.edit", (), edit_kwargs("mask", make, False, False))
        add(f"images-edit-markup-{kind}", "images.edit", (), edit_kwargs("markup", make, False, False))

    def video_kwargs(field: str, make: Callable[[], object], shape: str) -> Callable[[], dict[str, object]]:
        def build() -> dict[str, object]:
            value: object = make()
            if shape == "list":
                value = [value]
            elif shape == "mixed":
                value = [URL_INPUT, value]
            return {"model": MODEL_A, "prompt": "A slow pan", field: value}

        return build

    for kind, make in FILE_KINDS.items():
        add(f"video-image-{kind}", "video.generations.create", (), video_kwargs("image", make, "single"))
        add(
            f"video-reference-images-{kind}",
            "video.generations.create",
            (),
            video_kwargs("reference_images", make, "list"),
        )
    for field, shape in [
        ("last_frame_image", "single"),
        ("reference_images", "single"),
        ("reference_images", "mixed"),
        ("reference_videos", "list"),
        ("reference_audios", "single"),
    ]:
        add(f"video-{field}-{shape}", "video.generations.create", (), video_kwargs(field, FILE_KINDS["bytes"], shape))

    def inference_payload(make: Callable[[], object], as_list: bool) -> tuple[object, ...]:
        value: object = make()
        return (MODEL_A, {"inputs": "hello", "file": [value] if as_list else value})

    for kind, make in FILE_KINDS.items():
        cases[f"inference-payload-{kind}"] = Case("inference.run", inference_payload(make, False), dict)
    cases["inference-payload-list"] = Case("inference.run", inference_payload(FILE_KINDS["bytes"], True), dict)
    return cases


CASES = _cases()

#: Cases also run through the ``with_response`` twin.
TWIN_CASES = [
    "chat-stream",
    "responses-stream",
    "images-download",
    "audio-transcriptions",
    "video-edits",
    "inference-binary",
    "me-avatar",
    "images-edit-image-bytes",
    "video-image-bytes",
    "inference-payload-bytes",
]


def make_client(kind: str) -> tuple[MockAPI, Any]:
    api = MockAPI(sequence(json_response({"unexpected": True})))
    return api, api.client(kind)


def assert_not_implemented(
    api: MockAPI, fn: Callable[..., Any], args: tuple[object, ...], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(NotImplementedError) as caught:
        invoke(fn, *args, **kwargs)
    assert str(caught.value) == MESSAGE
    assert caught.value.args == (MESSAGE,)
    assert api.calls == [], "a request was sent"


def test_the_message_constant() -> None:
    base = importlib.import_module("devupai.resources._base")
    assert base.NOT_IMPLEMENTED_MESSAGE == MESSAGE


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_raises_before_any_request(client_kind: str, case_id: str) -> None:
    case = CASES[case_id]
    api, client = make_client(client_kind)
    kwargs = case.kwargs()
    assert_not_implemented(api, method_of(client, case.name), case.args, kwargs)
    for value in list(kwargs.values()) + list(case.args):
        if isinstance(value, io.BytesIO):
            assert value.tell() == 0, "a file input was read"


@pytest.mark.parametrize("case_id", TWIN_CASES)
def test_the_with_response_twin_raises_too(client_kind: str, case_id: str) -> None:
    case = CASES[case_id]
    api, client = make_client(client_kind)
    assert_not_implemented(api, method_of(client, case.name, with_response=True), case.args, case.kwargs())


def test_every_always_unavailable_method_has_a_case() -> None:
    always = {name for name, sample in SAMPLES.items() if sample.not_implemented}
    assert always == {"images.download", "audio.transcriptions.create", "video.edits.create", "me.avatar"}
    assert always <= {case.name for case in CASES.values()}


CHAT_RESPONSE = SAMPLES["chat.completions.create"].response
RESPONSES_RESPONSE = SAMPLES["responses.create"].response


@pytest.mark.parametrize(
    ("name", "kwargs", "response", "type_name"),
    [
        (
            "chat.completions.create",
            {"model": MODEL_A, "messages": MESSAGES, "stream": False},
            CHAT_RESPONSE,
            "ChatCompletion",
        ),
        (
            "responses.create",
            {"model": MODEL_A, "input": "Hello", "stream": False},
            RESPONSES_RESPONSE,
            "ResponseObject",
        ),
    ],
)
def test_stream_false_is_sent_as_a_json_request(
    client_kind: str, name: str, kwargs: dict[str, object], response: object, type_name: str
) -> None:
    api = MockAPI(sequence(json_response(response)))
    client = api.client(client_kind)
    result = invoke(method_of(client, name), **kwargs)
    types = importlib.import_module("devupai.types")
    assert isinstance(result, getattr(types, type_name))
    assert len(api.calls) == 1
    body = api.calls[0].json()
    assert body == kwargs
    assert body["stream"] is False
    assert api.calls[0].headers["accept"] == "application/json"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": MODEL_A, "prompt": "Make it blue", "image": URL_INPUT},
        {"model": MODEL_A, "prompt": "Make it blue", "images": [URL_INPUT, {"image_url": URL_INPUT}]},
        {"model": MODEL_A, "prompt": "Make it blue", "image": {"image_url": URL_INPUT}, "mask": URL_INPUT},
    ],
    ids=["url", "url-list", "url-objects"],
)
def test_images_edit_with_urls_only_is_a_json_request(client_kind: str, kwargs: dict[str, object]) -> None:
    api = MockAPI(sequence(json_response(SAMPLES["images.edit"].response)))
    client = api.client(client_kind)
    invoke(method_of(client, "images.edit"), **kwargs)
    assert len(api.calls) == 1
    assert api.calls[0].json() == kwargs
    assert api.calls[0].headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": MODEL_A, "prompt": "A slow pan", "image": URL_INPUT},
        {"model": MODEL_A, "prompt": "A slow pan", "reference_images": [URL_INPUT, {"image_url": URL_INPUT}]},
    ],
    ids=["url", "url-list"],
)
def test_video_generations_with_urls_only_is_a_json_request(client_kind: str, kwargs: dict[str, object]) -> None:
    api = MockAPI(sequence(json_response(SAMPLES["video.generations.create"].response)))
    client = api.client(client_kind)
    invoke(method_of(client, "video.generations.create"), **kwargs)
    assert len(api.calls) == 1
    assert api.calls[0].json() == kwargs


def test_inference_json_response_type_is_a_json_request(client_kind: str) -> None:
    api = MockAPI(sequence(json_response({"ok": True})))
    client = api.client(client_kind)
    result = invoke(method_of(client, "inference.run"), MODEL_A, {"inputs": "hello"}, response_type="json")
    assert result == {"ok": True}
    assert api.calls[0].json() == {"inputs": "hello"}
    assert "responsetype" not in api.calls[0].headers
    assert "response_type" not in api.calls[0].headers
