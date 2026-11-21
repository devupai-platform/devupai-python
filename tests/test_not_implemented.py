"""The behaviour that replaced the P2 placeholders (decisions C item 4, P3 contract §6).

In P2, streaming, file inputs (multipart) and binary answers raised ``NotImplementedError("available in a later
build")`` before anything was sent. P3 removed every placeholder, so this module now checks the real behaviour of each
of those calls, case by case (the P2 case ids are kept, a few are added):

- ``stream=True`` sends ``"stream": true`` with ``Accept: text/event-stream`` and returns a ``Stream`` /
  ``AsyncStream`` of typed items;
- a file input sends ``multipart/form-data`` with the expected parts, in order (default filename ``upload``);
- ``images.download``, ``me.avatar`` and ``inference.run(response_type="binary")`` return a ``BinaryResponse`` /
  ``AsyncBinaryResponse`` with the file bytes;
- ``audio.transcriptions.create`` and ``video.edits.create`` send their fixed parts and return the typed result.

Every case sends exactly one request, also through the ``with_response`` twin. ``NotImplementedError`` and
``NOT_IMPLEMENTED_MESSAGE`` appear nowhere in ``src/`` any more. ``stream=False`` and the URL-only forms are ordinary
JSON requests, as in P2.
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx
import pytest
from helpers.mock_transport import MODEL_A, REQ_ID, TEST_KEY, MockAPI, json_response, sequence
from helpers.samples import (
    BINARY_CONTENT,
    BINARY_CONTENT_TYPE,
    IMAGE_RESPONSE,
    IMPLEMENTED,
    OCTET_STREAM,
    SAMPLES,
    VIDEO_RESPONSE,
    Part,
    Sample,
    binary_response,
    drain,
    file_part,
    invoke,
    invoke_and_read,
    method_of,
    multipart_parts,
    return_type,
    sample_url,
    text_part,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "devupai"
OLD_MESSAGE = "available in a later build"
ETAG = '"fixture-etag"'
URL_INPUT = "https://cdn.example.test/input.png"
PNG = b"\x89PNG\r\n\x1a\n"
PATH_NAME = "fixture-input.png"
_BILLING: dict[str, object] = {"cost_dzd": 1.5, "balance_dzd": 98.5}

#: Each kind of file input of ``FileInput`` (contract §2.1), built fresh for every test. The ``path`` file is
#: written into the test's working directory by the ``input_file`` fixture.
FILE_KINDS: dict[str, Callable[[], object]] = {
    "bytes": lambda: PNG,
    "bytearray": lambda: bytearray(PNG),
    "memoryview": lambda: memoryview(PNG),
    "path": lambda: Path(PATH_NAME),
    "bytesio": lambda: io.BytesIO(PNG),
    "tuple": lambda: ("input.png", PNG),
    "tuple-with-type": lambda: ("input.png", io.BytesIO(PNG), "image/png"),
}

#: The filename and content type each kind is sent with (P3 contract §1 and §3: default filename ``upload``, a
#: ``Path`` gives its name, a file part without a type is ``application/octet-stream``).
SENT_AS: dict[str, tuple[str, str]] = {
    "bytes": ("upload", OCTET_STREAM),
    "bytearray": ("upload", OCTET_STREAM),
    "memoryview": ("upload", OCTET_STREAM),
    "path": (PATH_NAME, OCTET_STREAM),
    "bytesio": ("upload", OCTET_STREAM),
    "tuple": ("input.png", OCTET_STREAM),
    "tuple-with-type": ("input.png", "image/png"),
}

MESSAGES = [{"role": "user", "content": "Hello"}]

CHAT_CHUNK: dict[str, object] = {
    "id": "chatcmpl-fixture",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": MODEL_A,
    "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
}
RESPONSE_EVENT: dict[str, object] = {
    "type": "response.output_text.delta",
    "sequence_number": 1,
    "item_id": "msg_fixture",
    "output_index": 0,
    "content_index": 0,
    "delta": "Hi",
}
TRANSCRIPTION: dict[str, object] = {"text": "Hello", "_devup": _BILLING}
VERBOSE_TRANSCRIPTION: dict[str, object] = {
    "task": "transcribe",
    "language": "en",
    "text": "Hello",
    "duration": 0.5,
    "words": [{"word": "Hello", "start": 0.0, "end": 0.5}],
    "_devup": _BILLING,
}


@pytest.fixture(autouse=True)
def input_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Runs each test in a fresh directory that holds ``fixture-input.png`` (the ``path`` file kind)."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / PATH_NAME
    path.write_bytes(PNG)
    return path


# --- answers ------------------------------------------------------------------------------------------------------


def sse_response(*frames: object) -> httpx.Response:
    """A 200 event stream: one ``data:`` event per frame, then ``[DONE]``."""
    body = b"".join(b"data: " + json.dumps(frame).encode("utf-8") + b"\n\n" for frame in frames) + b"data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream", "x-request-id": REQ_ID}, content=body)


def text_response(text: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/plain", "x-request-id": REQ_ID}, content=text.encode())


def not_modified() -> httpx.Response:
    return binary_response(b"", status=304, headers={"etag": ETAG})


# --- result checks: (result, what the test read from it) -------------------------------------------------------------


def _types() -> Any:
    return importlib.import_module("devupai.types")


def check_chat_stream(result: Any, read: Any) -> None:
    t = _types()
    assert result.request_id == REQ_ID
    assert isinstance(result.response, httpx.Response)
    assert [type(item) for item in read] == [t.ChatCompletionChunk]
    assert read[0].id == "chatcmpl-fixture"
    assert read[0].choices[0].delta.content == "Hi"


def check_responses_stream(result: Any, read: Any) -> None:
    t = _types()
    assert result.request_id == REQ_ID
    assert isinstance(result.response, httpx.Response)
    assert [type(item) for item in read] == [t.ResponseStreamEvent]
    assert read[0].type == "response.output_text.delta"
    assert read[0].sequence_number == 1
    assert read[0].delta == "Hi"


def check_binary(result: Any, read: Any) -> None:
    assert read == BINARY_CONTENT
    assert result.status_code == 200
    assert result.headers["content-type"] == BINARY_CONTENT_TYPE
    assert result.request_id == REQ_ID
    assert isinstance(result.response, httpx.Response)
    assert result.response.status_code == 200


def check_not_modified(result: Any, read: Any) -> None:
    # As in Node: a 304 is a BinaryResponse with status 304, never ``None``.
    assert result is not None
    assert result.status_code == 304
    assert result.headers["etag"] == ETAG
    assert result.request_id == REQ_ID
    assert read == b""


def check_transcription(result: Any, read: Any) -> None:
    assert result.text == "Hello"
    assert result.devup.cost_dzd == 1.5
    assert read is None


def check_verbose_transcription(result: Any, read: Any) -> None:
    t = _types()
    assert result.text == "Hello"
    assert result.language == "en"
    assert isinstance(result.words[0], t.TranscriptionVerboseResponseWord)
    assert result.words[0].end == 0.5
    assert read is None


def check_text_transcription(result: Any, read: Any) -> None:
    assert result == "Hello world\n"
    assert read is None


def check_image(result: Any, read: Any) -> None:
    assert result.data[0].url == "https://cdn.example.test/fixture.png"
    assert result.devup.balance_dzd == 98.5
    assert read is None


def check_video(result: Any, read: Any) -> None:
    assert result.data[0].url == "https://cdn.example.test/fixture.mp4"
    assert result.devup.cost_dzd == 1.5
    assert read is None


def check_inference_json(result: Any, read: Any) -> None:
    assert result == {"ok": True}
    assert read is None


# --- the cases ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One call that P2 refused, and what it does now."""

    name: str  # the dotted method name
    args: Callable[[], tuple[object, ...]]  # fresh per test (a file object is read once)
    kwargs: Callable[[], dict[str, object]]
    response: Callable[[], httpx.Response]
    returns: str  # "Stream", "BinaryResponse" (``Async`` twin on the async client), a ``devupai.types`` name, "str"
    check: Callable[[Any, Any], None]
    accept: str = "application/json"
    body: dict[str, object] | None = None  # the JSON body sent (``None``: see ``parts``, or no body for a GET)
    parts: tuple[Part, ...] | None = None  # the multipart parts sent, in order
    headers: dict[str, str] = field(default_factory=dict)  # request headers that must be sent


def no_args() -> tuple[object, ...]:
    return ()


def _kinds_file(name: str, kind: str) -> Part:
    filename, content_type = SENT_AS[kind]
    return file_part(name, PNG, filename, content_type)


def _cases() -> dict[str, Case]:
    download_url = SAMPLES["images.download"].args[0]
    stream_chat = {"model": MODEL_A, "messages": MESSAGES, "stream": True}
    stream_responses = {"model": MODEL_A, "input": "Hello", "stream": True}
    key_points = [{"x": 0.5, "y": 0.25, "type": "positive"}]
    cases: dict[str, Case] = {
        "chat-stream": Case(
            "chat.completions.create",
            no_args,
            lambda: dict(stream_chat),
            lambda: sse_response(CHAT_CHUNK),
            "Stream",
            check_chat_stream,
            accept="text/event-stream",
            body=stream_chat,
        ),
        "responses-stream": Case(
            "responses.create",
            no_args,
            lambda: dict(stream_responses),
            lambda: sse_response(RESPONSE_EVENT),
            "Stream",
            check_responses_stream,
            accept="text/event-stream",
            body=stream_responses,
        ),
        "images-download": Case(
            "images.download", lambda: (download_url,), dict, binary_response, "BinaryResponse", check_binary, "*/*"
        ),
        "audio-transcriptions": Case(
            "audio.transcriptions.create",
            no_args,
            lambda: {"model": MODEL_A, "file": FILE_KINDS["bytes"]()},
            lambda: json_response(TRANSCRIPTION),
            "TranscriptionResponse",
            check_transcription,
            parts=(file_part("file", PNG), text_part("model", MODEL_A)),
        ),
        "audio-transcriptions-path": Case(
            "audio.transcriptions.create",
            no_args,
            lambda: {"file": FILE_KINDS["path"](), "language": "en"},
            lambda: json_response(TRANSCRIPTION),
            "TranscriptionResponse",
            check_transcription,
            parts=(file_part("file", PNG, PATH_NAME), text_part("language", "en")),
        ),
        "audio-transcriptions-text": Case(
            "audio.transcriptions.create",
            no_args,
            lambda: {"file": FILE_KINDS["bytes"](), "model": MODEL_A, "response_format": "text"},
            lambda: text_response("Hello world\n"),
            "str",
            check_text_transcription,
            accept="*/*",
            parts=(file_part("file", PNG), text_part("model", MODEL_A), text_part("response_format", "text")),
        ),
        "audio-transcriptions-verbose": Case(
            "audio.transcriptions.create",
            no_args,
            lambda: {
                "timestamp_granularities": ["word", "segment"],
                "temperature": 0.5,
                "prompt": "Names: Fixture",
                "response_format": "verbose_json",
                "language": "en",
                "model": MODEL_A,
                "file": FILE_KINDS["tuple-with-type"](),
            },
            lambda: json_response(VERBOSE_TRANSCRIPTION),
            "TranscriptionVerboseResponse",
            check_verbose_transcription,
            parts=(
                file_part("file", PNG, "input.png", "image/png"),
                text_part("model", MODEL_A),
                text_part("language", "en"),
                text_part("response_format", "verbose_json"),
                text_part("prompt", "Names: Fixture"),
                text_part("temperature", "0.5"),
                text_part("timestamp_granularities[]", "word"),
                text_part("timestamp_granularities[]", "segment"),
            ),
        ),
        "video-edits": Case(
            "video.edits.create",
            no_args,
            lambda: {"model": MODEL_A, "video": FILE_KINDS["bytes"]()},
            lambda: json_response(VIDEO_RESPONSE),
            "VideoGenerationResponse",
            check_video,
            parts=(file_part("video", PNG), text_part("model", MODEL_A)),
        ),
        "video-edits-with-mask": Case(
            "video.edits.create",
            no_args,
            lambda: {"model": MODEL_A, "video": FILE_KINDS["bytesio"](), "mask": FILE_KINDS["bytes"](), "prompt": "x"},
            lambda: json_response(VIDEO_RESPONSE),
            "VideoGenerationResponse",
            check_video,
            parts=(
                file_part("video", PNG),
                text_part("model", MODEL_A),
                text_part("prompt", "x"),
                file_part("mask", PNG),
            ),
        ),
        "video-edits-all-fields": Case(
            "video.edits.create",
            no_args,
            # Given in reverse: the parts follow Node's fixed list (video.ts:125-158), whatever the call order.
            lambda: {
                "response_format": "url",
                "output_container_and_codec": "mp4_h264",
                "desired_increase": 2,
                "background_color": "#000000",
                "preserve_audio": False,
                "auto_trim": True,
                "frame_index": 3,
                "key_points": key_points,
                "mask_url": "https://cdn.example.test/mask.png",
                "prompt": "Make the sky blue",
                "model": MODEL_A,
                "video": ("clip.mp4", PNG, "video/mp4"),
            },
            lambda: json_response(VIDEO_RESPONSE),
            "VideoGenerationResponse",
            check_video,
            parts=(
                file_part("video", PNG, "clip.mp4", "video/mp4"),
                text_part("model", MODEL_A),
                text_part("prompt", "Make the sky blue"),
                text_part("mask_url", "https://cdn.example.test/mask.png"),
                text_part("key_points", json.dumps(key_points, separators=(",", ":"))),
                text_part("frame_index", "3"),
                text_part("auto_trim", "true"),
                text_part("preserve_audio", "false"),
                text_part("background_color", "#000000"),
                text_part("desired_increase", "2"),
                text_part("output_container_and_codec", "mp4_h264"),
                text_part("response_format", "url"),
            ),
        ),
        "inference-binary": Case(
            "inference.run",
            lambda: (MODEL_A, {"inputs": "hello"}),
            lambda: {"response_type": "binary"},
            binary_response,
            "BinaryResponse",
            check_binary,
            accept="*/*",
            body={"inputs": "hello"},
        ),
        "me-avatar": Case("me.avatar", no_args, dict, binary_response, "BinaryResponse", check_binary, "*/*"),
        "me-avatar-if-none-match": Case(
            "me.avatar",
            no_args,
            lambda: {"if_none_match": ETAG},
            not_modified,
            "BinaryResponse",
            check_not_modified,
            accept="*/*",
            headers={"if-none-match": ETAG},
        ),
    }

    def add(key: str, case: Case) -> None:
        assert key not in cases, key
        cases[key] = case

    def edit_case(field_name: str, kind: str, shape: str) -> Case:
        make = FILE_KINDS[kind]
        part_name = "image" if field_name in ("image", "images") else field_name

        def build() -> dict[str, object]:
            value: object = make()
            if shape == "mixed":
                value = [URL_INPUT, value]
            elif shape == "list":
                value = [value]
            return {"model": MODEL_A, "prompt": "Make it blue", field_name: value}

        inputs = [_kinds_file(part_name, kind)]
        if shape == "mixed":
            inputs.insert(0, text_part(part_name, URL_INPUT))
        parts = (text_part("model", MODEL_A), text_part("prompt", "Make it blue"), *inputs)
        return Case(
            "images.edit",
            no_args,
            build,
            lambda: json_response(IMAGE_RESPONSE),
            "ImageGenerateResponse",
            check_image,
            parts=parts,
        )

    for kind in FILE_KINDS:
        add(f"images-edit-image-{kind}", edit_case("image", kind, "single"))
        add(f"images-edit-image-list-{kind}", edit_case("image", kind, "mixed"))
        add(f"images-edit-images-{kind}", edit_case("images", kind, "list"))
        add(f"images-edit-images-mixed-{kind}", edit_case("images", kind, "mixed"))
        add(f"images-edit-mask-{kind}", edit_case("mask", kind, "single"))
        add(f"images-edit-markup-{kind}", edit_case("markup", kind, "single"))

    def video_case(field_name: str, kind: str, shape: str) -> Case:
        make = FILE_KINDS[kind]

        def build() -> dict[str, object]:
            value: object = make()
            if shape == "list":
                value = [value]
            elif shape == "mixed":
                value = [URL_INPUT, value]
            return {"model": MODEL_A, "prompt": "A slow pan", field_name: value}

        # A list gives one part per item under the plain field name.
        inputs = [_kinds_file(field_name, kind)]
        if shape == "mixed":
            inputs.insert(0, text_part(field_name, URL_INPUT))
        parts = (text_part("model", MODEL_A), text_part("prompt", "A slow pan"), *inputs)
        return Case(
            "video.generations.create",
            no_args,
            build,
            lambda: json_response(VIDEO_RESPONSE),
            "VideoGenerationResponse",
            check_video,
            parts=parts,
        )

    for kind in FILE_KINDS:
        add(f"video-image-{kind}", video_case("image", kind, "single"))
        add(f"video-reference-images-{kind}", video_case("reference_images", kind, "list"))
    for field_name, shape in [
        ("last_frame_image", "single"),
        ("reference_images", "single"),
        ("reference_images", "mixed"),
        ("reference_videos", "list"),
        ("reference_audios", "single"),
    ]:
        add(f"video-{field_name}-{shape}", video_case(field_name, "bytes", shape))

    def inference_case(kind: str, as_list: bool) -> Case:
        make = FILE_KINDS[kind]

        def args() -> tuple[object, ...]:
            value: object = make()
            return (MODEL_A, {"inputs": "hello", "file": [value] if as_list else value})

        return Case(
            "inference.run",
            args,
            dict,
            lambda: json_response({"ok": True}),
            "dict",
            check_inference_json,
            parts=(text_part("inputs", "hello"), _kinds_file("file", kind)),
        )

    for kind in FILE_KINDS:
        add(f"inference-payload-{kind}", inference_case(kind, False))
    add("inference-payload-list", inference_case("bytes", True))
    return cases


CASES = _cases()

#: Cases also run through the ``with_response`` twin (the P2 list, unchanged).
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

#: The methods that were always a placeholder in P2.
ALWAYS_PLACEHOLDERS = frozenset({"images.download", "audio.transcriptions.create", "video.edits.create", "me.avatar"})


def expected_class(returns: str, client_kind: str) -> type[Any]:
    if returns == "str":
        return str
    if returns == "dict":
        return dict
    return return_type(returns, client_kind)


def file_objects(value: object) -> Iterator[io.IOBase]:
    """The file objects inside a call's arguments (in lists, tuples and payload dicts)."""
    if isinstance(value, io.IOBase):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from file_objects(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from file_objects(item)


def assert_request(api: MockAPI, case: Case) -> None:
    """Exactly one request, with the method, path, key, ``Accept``, headers and body of the case."""
    assert len(api.calls) == 1, f"{len(api.calls)} requests were sent"
    sample: Sample = SAMPLES[case.name]
    request = api.calls[0]
    assert request.method == sample.http_method
    assert urlsplit(request.url).path == sample.path
    if case.name == "images.download":
        assert request.url == sample_url("images.download")
    if sample.auth:
        assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    else:
        assert "authorization" not in request.headers
    assert request.headers["accept"] == case.accept
    for name, value in case.headers.items():
        assert request.headers[name] == value
    if case.parts is not None:
        assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
        assert multipart_parts(request) == list(case.parts)
    elif case.body is not None:
        assert request.headers["content-type"] == "application/json"
        assert request.json() == case.body
    else:
        assert sample.http_method == "GET"
        assert request.content == b""
        assert "content-type" not in request.headers


def run_case(client_kind: str, case_id: str, *, with_response: bool) -> None:
    case = CASES[case_id]
    api = MockAPI(sequence(case.response()))
    client = api.client(client_kind)
    args = case.args()
    kwargs = case.kwargs()
    result, read = invoke_and_read(method_of(client, case.name, with_response=with_response), *args, **kwargs)
    assert_request(api, case)
    data = result
    if with_response:
        api_response = importlib.import_module("devupai").APIResponse
        assert isinstance(result, api_response)
        assert result.request_id == REQ_ID
        data = result.data
    expected = expected_class(case.returns, client_kind)
    assert isinstance(data, expected), f"{case.name} returned {type(data).__name__}, not {expected.__name__}"
    case.check(data, read)
    # The SDK reads a caller's file object and never closes it.
    for file_object in file_objects([args, kwargs]):
        assert not file_object.closed, "the SDK closed a file object it did not open"


# --- tests ----------------------------------------------------------------------------------------------------------


def test_no_placeholder_is_left_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for word in ("NotImplementedError", "NOT_IMPLEMENTED_MESSAGE", OLD_MESSAGE):
            if word in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {word}")
    assert offenders == []
    assert len(list(SRC.rglob("*.py"))) > 10


def test_the_placeholder_message_constant_is_gone() -> None:
    for module_name in ("devupai.resources._base", "devupai.resources", "devupai"):
        module = importlib.import_module(module_name)
        assert not hasattr(module, "NOT_IMPLEMENTED_MESSAGE"), module_name
        assert "NOT_IMPLEMENTED_MESSAGE" not in getattr(module, "__all__", ()), module_name


def test_samples_have_no_placeholder_flag() -> None:
    assert "not_implemented" not in {item.name for item in fields(Sample)}
    assert set(IMPLEMENTED) == set(SAMPLES)


def test_every_former_placeholder_has_a_case() -> None:
    names = {case.name for case in CASES.values()}
    assert ALWAYS_PLACEHOLDERS <= names
    # The forms that were refused by condition in P2 have cases too.
    assert {"chat.completions.create", "responses.create", "images.edit", "video.generations.create"} <= names
    assert {"inference-binary", "inference-payload-bytes", "inference-payload-list"} <= set(CASES)
    assert set(TWIN_CASES) <= set(CASES)
    # The 79 P2 cases, plus the text and verbose transcription forms and the full video edit list.
    assert len(CASES) == 82


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_sends_one_request_and_returns_the_real_result(client_kind: str, case_id: str) -> None:
    run_case(client_kind, case_id, with_response=False)


@pytest.mark.parametrize("case_id", TWIN_CASES)
def test_the_with_response_twin_returns_the_same_result(client_kind: str, case_id: str) -> None:
    run_case(client_kind, case_id, with_response=True)


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
@pytest.mark.parametrize("case_id", ["chat-stream", "responses-stream"])
def test_a_stream_has_its_request_id_before_the_first_item(client_kind: str, case_id: str, with_response: bool) -> None:
    case = CASES[case_id]
    api = MockAPI(sequence(case.response()))
    client = api.client(client_kind)
    fn = method_of(client, case.name, with_response=with_response)

    async def main() -> tuple[tuple[Any, Any, Any], list[Any]]:
        result = fn(**case.kwargs())
        if inspect.isawaitable(result):
            result = await result
        stream = result.data if with_response else result
        before = (stream.request_id, stream.client_request_id, list(stream.warnings))
        return before, await drain(stream)

    before, items = anyio.run(main)
    assert before == (REQ_ID, None, [])
    assert len(items) == 1
    assert len(api.calls) == 1


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
