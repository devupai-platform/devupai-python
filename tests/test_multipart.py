"""Multipart requests and file inputs (contract P3 §1 multipart, §3; decisions C item 2).

The file helpers of ``devupai.resources.shared`` (``read_file_input``, ``is_file_input``, ``form_field``), then the
four multipart methods (``images.edit`` file form, ``audio.transcriptions.create``, ``video.generations.create`` file
form, ``video.edits.create``) and ``inference.run`` with a file in its payload, on both clients.

Each recorded multipart body is parsed with the standard library (``email.parser.BytesParser`` with the HTTP policy)
into an ordered list of parts, so the tests check the part names, their order, the file names, the content types and
the bytes, not only what a form lookup would show.

``devupai.resources.shared`` and ``devupai.types`` are imported inside the tests, so each test reports on its own while
the package is incomplete.
"""

from __future__ import annotations

import builtins
import email.parser
import email.policy
import importlib
import inspect
import io
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    RecordedRequest,
    error_response,
    json_response,
    sequence,
)
from helpers.samples import invoke, method_of

from devupai import APIResponse, AsyncDevupAI, DevupAI

PNG = b"\x89PNG\r\n\x1a\n\x00\xff"
WAV = b"RIFF\x00\x00\x00\x00WAVE"
MP4 = b"\x00\x00\x00\x18ftypmp42"
URL_INPUT = "https://cdn.example.test/input.png"
URL_SECOND = "https://cdn.example.test/second.png"
DATA_URI = "data:image/png;base64,AA=="
OCTET_STREAM = "application/octet-stream"

BILLING: dict[str, object] = {"cost_dzd": 1.5, "balance_dzd": 98.5, "request_id": REQ_ID}
IMAGE_RESULT: dict[str, object] = {"data": [{"url": "https://cdn.example.test/fixture.png"}], "_devup": BILLING}
VIDEO_RESULT: dict[str, object] = {"data": [{"url": "https://cdn.example.test/fixture.mp4"}], "_devup": BILLING}
TRANSCRIPTION_RESULT: dict[str, object] = {"text": "Hello", "_devup": BILLING}
VERBOSE_RESULT: dict[str, object] = {
    "task": "transcribe",
    "language": "en",
    "text": "Hello",
    "duration": 1.5,
    "words": [{"word": "Hello", "start": 0.0, "end": 0.5}],
    "_devup": BILLING,
}
SRT_TEXT = "1\n00:00:00,000 --> 00:00:00,500\nHello\n"

#: Node `video.ts:125-158`: the parts of a video edit, in this order, and no other.
VIDEO_EDIT_FIELDS = (
    "video",
    "model",
    "prompt",
    "mask",
    "mask_url",
    "key_points",
    "frame_index",
    "auto_trim",
    "preserve_audio",
    "background_color",
    "desired_increase",
    "output_container_and_codec",
    "response_format",
)


# --- helpers --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """One part of a recorded multipart body."""

    name: str | None
    filename: str | None
    content_type: str | None  # ``None`` when the part has no Content-Type header
    payload: bytes


def _shared() -> Any:
    return importlib.import_module("devupai.resources.shared")


def _types() -> Any:
    return importlib.import_module("devupai.types")


def setup(kind: str, *entries: Entry) -> tuple[MockAPI, DevupAI | AsyncDevupAI]:
    api = MockAPI(sequence(*(entries or (json_response(IMAGE_RESULT),))))
    return api, api.client(kind)


def both_clients() -> tuple[DevupAI, AsyncDevupAI]:
    api = MockAPI(sequence(json_response({})))
    return api.sync_client(), api.async_client()


def call(client: DevupAI | AsyncDevupAI, name: str, *args: Any, with_response: bool = False, **kwargs: Any) -> Any:
    """Calls the public method ``name`` (``"images.edit"``) on either client and returns its result."""
    return invoke(method_of(client, name, with_response=with_response), *args, **kwargs)


def boundary_of(request: RecordedRequest) -> bytes:
    content_type = request.headers.get("content-type", "")
    assert content_type.startswith("multipart/form-data"), content_type
    marker = "boundary="
    assert marker in content_type, content_type
    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
    assert boundary
    return boundary.encode("latin-1")


def parts_of(request: RecordedRequest) -> list[Part]:
    """The parts of a recorded ``multipart/form-data`` body, in the order they were sent."""
    content_type = request.headers.get("content-type", "")
    boundary_of(request)
    raw = b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + request.content
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
    assert message.is_multipart(), "the body is not a multipart body"
    parts: list[Part] = []
    for item in message.iter_parts():
        name = item.get_param("name", header="content-disposition")
        payload = item.get_payload(decode=True)
        assert isinstance(payload, bytes)
        parts.append(
            Part(
                name=name if isinstance(name, str) else None,
                filename=item.get_filename(),
                content_type=item.get_content_type() if item.get("content-type") is not None else None,
                payload=payload,
            )
        )
    return parts


def names(parts: list[Part]) -> list[str | None]:
    return [part.name for part in parts]


def short(parts: list[Part]) -> list[tuple[str | None, str | None, bytes]]:
    """``(name, filename, payload)`` of each part: a text part has the filename ``None``."""
    return [(part.name, part.filename, part.payload) for part in parts]


def only(parts: list[Part], name: str) -> list[Part]:
    return [part for part in parts if part.name == name]


def normalized_body(request: RecordedRequest) -> bytes:
    """The body with its own boundary replaced, so two attempts can be compared byte for byte."""
    return request.content.replace(boundary_of(request), b"BOUNDARY")


def assert_multipart_post(request: RecordedRequest, path: str) -> None:
    assert request.method == "POST"
    assert request.url == f"{BASE}{path}"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")


def text_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/plain; charset=utf-8", "x-request-id": REQ_ID},
        content=text.encode("utf-8"),
    )


def track_opens(monkeypatch: pytest.MonkeyPatch, target: Path) -> list[Any]:
    """Records every file object opened on ``target`` through ``open`` / ``io.open`` (``Path.open`` included)."""
    handles: list[Any] = []
    real_open = builtins.open
    wanted = os.path.normcase(os.path.abspath(target))

    def tracking_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(file, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and os.path.normcase(os.path.abspath(os.fspath(file))) == wanted:
            handles.append(handle)
        return handle

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(io, "open", tracking_open)
    return handles


def assert_closed_by_sdk(handles: list[Any], target: Path) -> None:
    """The SDK opened ``target`` and closed it again: every handle is closed and the file can be deleted."""
    assert handles, "the path was not opened with open()"
    assert all(handle.closed for handle in handles)
    # On Windows an open file cannot be deleted, so this also fails when a handle is still open there.
    target.unlink()
    assert not target.exists()


class FixturePathLike:
    """A minimal ``os.PathLike`` that is not a ``pathlib.Path``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def __fspath__(self) -> str:
        return str(self._path)


class NumberNamedReader:
    """A binary file object whose ``name`` is not a ``str`` (like a file opened from a descriptor)."""

    name = 7

    def __init__(self, content: bytes) -> None:
        self._buffer = io.BytesIO(content)
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def close(self) -> None:
        self.closed = True


# --- read_file_input / is_file_input / form_field (contract §3) ---------------------------------------------------


def test_default_filename_is_upload() -> None:
    # Decision Q19: a file without a name of its own is sent as `upload`.
    assert _shared().DEFAULT_FILENAME == "upload"


@pytest.mark.parametrize(
    "value",
    [PNG, bytearray(PNG), memoryview(PNG)],
    ids=["bytes", "bytearray", "memoryview"],
)
def test_read_file_input_raw_bytes(value: object) -> None:
    filename, content, content_type = _shared().read_file_input(value)
    assert filename == "upload"
    assert type(content) is bytes
    assert content == PNG
    assert content_type is None


def test_read_file_input_path_is_opened_read_and_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "photo.png"
    target.write_bytes(PNG)
    handles = track_opens(monkeypatch, target)
    filename, content, _content_type = _shared().read_file_input(target)
    assert filename == "photo.png"
    assert content == PNG
    assert_closed_by_sdk(handles, target)


def test_read_file_input_any_pathlike(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "clip.wav"
    target.write_bytes(WAV)
    handles = track_opens(monkeypatch, target)
    filename, content, _content_type = _shared().read_file_input(FixturePathLike(target))
    assert filename == "clip.wav"
    assert content == WAV
    assert_closed_by_sdk(handles, target)


def test_read_file_input_caller_bytesio_is_not_closed() -> None:
    buffer = io.BytesIO(PNG)
    assert _shared().read_file_input(buffer) == ("upload", PNG, None)
    assert not buffer.closed
    buffer.seek(0)
    assert buffer.read() == PNG


def test_read_file_input_caller_open_file_is_not_closed_and_uses_its_basename(tmp_path: Path) -> None:
    target = tmp_path / "clip.wav"
    target.write_bytes(WAV)
    with open(target, "rb") as handle:
        filename, content, _content_type = _shared().read_file_input(handle)
        assert filename == "clip.wav"
        assert content == WAV
        assert not handle.closed
        handle.seek(0)
        assert handle.read() == WAV


def test_read_file_input_named_file_object_uses_the_basename() -> None:
    buffer = io.BytesIO(PNG)
    buffer.name = "nested/dir/photo.png"  # type: ignore[attr-defined]
    filename, content, _content_type = _shared().read_file_input(buffer)
    assert filename == "photo.png"
    assert content == PNG


def test_read_file_input_file_object_with_a_non_str_name_is_upload() -> None:
    reader = NumberNamedReader(WAV)
    assert _shared().read_file_input(reader) == ("upload", WAV, None)
    assert reader.closed is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (("photo.png", PNG), ("photo.png", PNG, None)),
        (("photo.png", PNG, "image/png"), ("photo.png", PNG, "image/png")),
        (("photo.png", io.BytesIO(PNG)), ("photo.png", PNG, None)),
        (("photo.png", io.BytesIO(PNG), "image/png"), ("photo.png", PNG, "image/png")),
    ],
    ids=["name-bytes", "name-bytes-type", "name-file", "name-file-type"],
)
def test_read_file_input_tuples(value: object, expected: tuple[str, bytes, str | None]) -> None:
    assert _shared().read_file_input(value) == expected


def test_read_file_input_tuple_with_a_caller_file_does_not_close_it() -> None:
    buffer = io.BytesIO(WAV)
    assert _shared().read_file_input(("clip.wav", buffer, "audio/wav")) == ("clip.wav", WAV, "audio/wav")
    assert not buffer.closed


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PNG,
        lambda: bytearray(PNG),
        lambda: memoryview(PNG),
        lambda: Path("fixture-input.png"),
        lambda: io.BytesIO(PNG),
        lambda: ("input.png", PNG),
        lambda: ("input.png", io.BytesIO(PNG), "image/png"),
    ],
    ids=["bytes", "bytearray", "memoryview", "path", "bytesio", "tuple", "tuple-with-type"],
)
def test_is_file_input_true(factory: Callable[[], object]) -> None:
    assert _shared().is_file_input(factory()) is True


@pytest.mark.parametrize(
    "value",
    [URL_INPUT, {"image_url": URL_INPUT}, None, 1, 1.5, True, [PNG], ("input.png",), ("input.png", "text")],
    ids=["str", "dict", "none", "int", "float", "bool", "list", "one-tuple", "name-text-tuple"],
)
def test_is_file_input_false(value: object) -> None:
    assert _shared().is_file_input(value) is False


def test_base_reexports_is_file_input_unchanged() -> None:
    base = importlib.import_module("devupai.resources._base")
    assert base.is_file_input is _shared().is_file_input


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "true"),
        (False, "false"),
        (1.0, "1"),
        (0.0, "0"),
        (-3.0, "-3"),
        (0.2, "0.2"),
        (7.5, "7.5"),
        (0, "0"),
        (42, "42"),
        ("fixture", "fixture"),
        ("", ""),
    ],
)
def test_form_field(value: object, expected: str) -> None:
    # Node `shared.ts:89-93`: `String(value)`, so booleans are lower case and 1.0 is "1".
    assert _shared().form_field(value) == expected


# --- images.edit, file form (Node images.ts:21-58, 121-139; tests/images-edit.test.ts) -----------------------------


def test_edit_bytes_image_sends_multipart(client_kind: str) -> None:
    api, client = setup(client_kind)
    result = call(client, "images.edit", model=MODEL_A, prompt="Add a hat", image=PNG)
    assert len(api.calls) == 1
    request = api.calls[0]
    assert_multipart_post(request, "/images/edits")
    parts = parts_of(request)
    assert short(parts) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"Add a hat"),
        ("image", "upload", PNG),
    ]
    assert only(parts, "image")[0].content_type == OCTET_STREAM
    assert isinstance(result, _types().ImageGenerateResponse)
    assert result.data[0].url == "https://cdn.example.test/fixture.png"


def test_edit_caller_content_type_is_ignored(client_kind: str) -> None:
    api, client = setup(client_kind)
    call(
        client,
        "images.edit",
        model=MODEL_A,
        prompt="p",
        image=PNG,
        extra_headers={"Content-Type": "application/json", "X-Fixture": "1"},
    )
    request = api.calls[0]
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert request.headers["x-fixture"] == "1"
    assert names(parts_of(request)) == ["model", "prompt", "image"]


def test_edit_image_list_gives_one_image_part_per_item_in_order(client_kind: str) -> None:
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=[b"first", b"second"])
    assert short(parts_of(api.calls[0])) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"p"),
        ("image", "upload", b"first"),
        ("image", "upload", b"second"),
    ]


def test_edit_images_list_items_are_sent_as_image_parts(client_kind: str) -> None:
    # Node `images.ts:46`: `images` is the list form of `image`; its items go out as `image` parts.
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", images=[b"first", URL_INPUT])
    parts = parts_of(api.calls[0])
    assert "images" not in names(parts)
    assert short(only(parts, "image")) == [("image", "upload", b"first"), ("image", None, URL_INPUT.encode())]


def test_edit_image_then_images_keep_the_field_order(client_kind: str) -> None:
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, image=b"one", images=[b"two", b"three"], prompt="p")
    assert short(parts_of(api.calls[0])) == [
        ("model", None, MODEL_A.encode()),
        ("image", "upload", b"one"),
        ("image", "upload", b"two"),
        ("image", "upload", b"three"),
        ("prompt", None, b"p"),
    ]


def test_edit_text_items_and_image_url_objects_are_text_parts(client_kind: str) -> None:
    # Node `images.ts:30-34` and images-edit.test.ts:126-139: a string item is text, `{image_url}` its string.
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=[PNG, URL_INPUT, {"image_url": DATA_URI}])
    assert short(only(parts_of(api.calls[0]), "image")) == [
        ("image", "upload", PNG),
        ("image", None, URL_INPUT.encode()),
        ("image", None, DATA_URI.encode()),
    ]


def test_edit_mask_and_markup_files_are_their_own_parts(client_kind: str) -> None:
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=PNG, mask=b"mask", markup=bytes([1, 2, 3]))
    parts = parts_of(api.calls[0])
    assert short(parts) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"p"),
        ("image", "upload", PNG),
        ("mask", "upload", b"mask"),
        ("markup", "upload", b"\x01\x02\x03"),
    ]
    assert only(parts, "mask")[0].content_type == OCTET_STREAM
    assert only(parts, "markup")[0].content_type == OCTET_STREAM


def test_edit_text_markup_and_image_url_mask_are_text_parts(client_kind: str) -> None:
    # images-edit.test.ts:107-113: markup given as text is a text part.
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=PNG, mask={"image_url": URL_SECOND}, markup=URL_INPUT)
    parts = parts_of(api.calls[0])
    assert short(only(parts, "mask")) == [("mask", None, URL_SECOND.encode())]
    assert short(only(parts, "markup")) == [("markup", None, URL_INPUT.encode())]


def test_edit_a_file_mask_alone_selects_multipart(client_kind: str) -> None:
    # Node `images.ts:129`: any of image, images, mask or markup holding a file selects the form.
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=URL_INPUT, mask=b"mask")
    parts = parts_of(api.calls[0])
    assert short(only(parts, "image")) == [("image", None, URL_INPUT.encode())]
    assert short(only(parts, "mask")) == [("mask", "upload", b"mask")]


def test_edit_other_fields_scalars_json_text_none_and_extra_body(client_kind: str) -> None:
    # Node `images.ts:42-57`: an object or a list is one JSON-text part, a scalar is `String(value)`, null is left
    # out, and every field of the body is sent (extra_body included, merged last).
    api, client = setup(client_kind)
    call(
        client,
        "images.edit",
        model=MODEL_A,
        prompt="p",
        image=PNG,
        n=2,
        guidance_scale=7.5,
        seed=42,
        negative_prompt=None,
        extra_body={
            "options": {"steps": 4, "tags": ["a", "b"]},
            "loras": ["x", "y"],
            "enhance": True,
            "strength": 1.0,
            "skip": None,
        },
    )
    assert short(parts_of(api.calls[0])) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"p"),
        ("image", "upload", PNG),
        ("n", None, b"2"),
        ("guidance_scale", None, b"7.5"),
        ("seed", None, b"42"),
        ("options", None, b'{"steps":4,"tags":["a","b"]}'),
        ("loras", None, b'["x","y"]'),
        ("enhance", None, b"true"),
        ("strength", None, b"1"),
    ]


def test_edit_path_image_uses_its_name_and_is_closed(
    client_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "photo.png"
    target.write_bytes(PNG)
    api, client = setup(client_kind)
    handles = track_opens(monkeypatch, target)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=target)
    assert short(only(parts_of(api.calls[0]), "image")) == [("image", "photo.png", PNG)]
    assert_closed_by_sdk(handles, target)


def test_edit_tuple_image_keeps_its_name_and_type(client_kind: str) -> None:
    api, client = setup(client_kind)
    call(client, "images.edit", model=MODEL_A, prompt="p", image=("photo.png", PNG, "image/png"))
    assert only(parts_of(api.calls[0]), "image") == [Part("image", "photo.png", "image/png", PNG)]


def test_edit_caller_file_objects_are_not_closed(client_kind: str, tmp_path: Path) -> None:
    target = tmp_path / "mask.png"
    target.write_bytes(b"mask")
    buffer = io.BytesIO(PNG)
    api, client = setup(client_kind)
    with open(target, "rb") as handle:
        call(client, "images.edit", model=MODEL_A, prompt="p", image=buffer, mask=handle)
        assert not handle.closed
        handle.seek(0)
        assert handle.read() == b"mask"
    assert not buffer.closed
    buffer.seek(0)
    assert buffer.read() == PNG
    parts = parts_of(api.calls[0])
    assert short(only(parts, "image")) == [("image", "upload", PNG)]
    assert short(only(parts, "mask")) == [("mask", "mask.png", b"mask")]


def test_edit_billed_retry_sends_the_same_body(client_kind: str) -> None:
    # Contract §1: the content is held as bytes, so a retry sends the same bytes (a caller file is read once).
    api, client = setup(client_kind, error_response(503, "Busy"), json_response(IMAGE_RESULT))
    result = call(client, "images.edit", model=MODEL_A, prompt="p", image=io.BytesIO(PNG), mask=b"mask")
    assert len(api.calls) == 2
    first, second = api.calls
    assert parts_of(first) == parts_of(second)
    assert normalized_body(first) == normalized_body(second)
    assert short(only(parts_of(second), "image")) == [("image", "upload", PNG)]
    assert isinstance(result, _types().ImageGenerateResponse)


def test_edit_without_a_file_stays_json(client_kind: str) -> None:
    # P2 behaviour kept; images-edit.test.ts:153-170.
    api, client = setup(client_kind)
    params: dict[str, object] = {
        "model": MODEL_A,
        "prompt": "Replace the sky",
        "images": [URL_INPUT, {"image_url": DATA_URI}],
        "mask": {"image_url": DATA_URI},
        "size": "1024x1024",
        "n": 1,
    }
    call(client, "images.edit", **params)
    request = api.calls[0]
    assert request.headers["content-type"] == "application/json"
    assert request.json() == params


def test_edit_with_response_twin_in_the_file_form(client_kind: str) -> None:
    api, client = setup(client_kind)
    response = call(client, "images.edit", with_response=True, model=MODEL_A, prompt="p", image=PNG)
    assert isinstance(response, APIResponse)
    assert isinstance(response.data, _types().ImageGenerateResponse)
    assert response.request_id == REQ_ID
    assert names(parts_of(api.calls[0])) == ["model", "prompt", "image"]


# --- audio.transcriptions.create (Node audio.ts:62-107; tests/transcriptions-extras.test.ts, final.test.ts:100-119) --


def test_transcription_parts_are_exactly_the_node_list_in_order(client_kind: str) -> None:
    # Node `audio.ts:83-93`: file, model, language, response_format, prompt, temperature, then one
    # `timestamp_granularities[]` part per value, whatever the keyword order.
    api, client = setup(client_kind, json_response(VERBOSE_RESULT))
    call(
        client,
        "audio.transcriptions.create",
        timestamp_granularities=["word", "segment"],
        temperature=0.2,
        prompt="Fixture, Name",
        response_format="verbose_json",
        language="en",
        model=MODEL_A,
        file=WAV,
    )
    request = api.calls[0]
    assert_multipart_post(request, "/audio/transcriptions")
    parts = parts_of(request)
    assert short(parts) == [
        ("file", "upload", WAV),
        ("model", None, MODEL_A.encode()),
        ("language", None, b"en"),
        ("response_format", None, b"verbose_json"),
        ("prompt", None, b"Fixture, Name"),
        ("temperature", None, b"0.2"),
        ("timestamp_granularities[]", None, b"word"),
        ("timestamp_granularities[]", None, b"segment"),
    ]
    assert parts[0].content_type == OCTET_STREAM


@pytest.mark.parametrize(("temperature", "expected"), [(0, b"0"), (0.0, b"0"), (1.0, b"1"), (0.5, b"0.5")])
def test_transcription_temperature_as_text(client_kind: str, temperature: float, expected: bytes) -> None:
    # transcriptions-extras.test.ts:52-56: temperature 0 is sent as "0".
    api, client = setup(client_kind, json_response(TRANSCRIPTION_RESULT))
    call(client, "audio.transcriptions.create", file=WAV, model=MODEL_A, temperature=temperature)
    assert short(only(parts_of(api.calls[0]), "temperature")) == [("temperature", None, expected)]


def test_transcription_omitted_fields_give_no_parts(client_kind: str) -> None:
    # transcriptions-extras.test.ts:64-81 and extra.test.ts:18-28: no model, no part.
    api, client = setup(client_kind, json_response(TRANSCRIPTION_RESULT))
    call(client, "audio.transcriptions.create", file=WAV)
    call(client, "audio.transcriptions.create", file=WAV, model=MODEL_A, timestamp_granularities=[])
    assert names(parts_of(api.calls[0])) == ["file"]
    assert names(parts_of(api.calls[1])) == ["file", "model"]


@pytest.mark.parametrize("with_response", [False, True], ids=["method", "with_response"])
@pytest.mark.parametrize(
    "unknown",
    [{"seed": 5}, {"chunking_strategy": "auto"}, {"extra_body": {"x": 1}}],
    ids=["seed", "chunking_strategy", "extra_body"],
)
def test_transcription_unknown_keyword_raises_type_error(
    client_kind: str, with_response: bool, unknown: dict[str, object]
) -> None:
    # Lead decision (gap 4): an explicit keyword-only signature, so an unknown keyword is a `TypeError` with no
    # request (Node rejects it at compile time).
    api, client = setup(client_kind, json_response(TRANSCRIPTION_RESULT))
    with pytest.raises(TypeError):
        call(
            client,
            "audio.transcriptions.create",
            with_response=with_response,
            file=WAV,
            model=MODEL_A,
            prompt="p",
            **unknown,
        )
    assert api.calls == []


def test_transcription_methods_take_no_extra_body() -> None:
    # Contract §3: Node sends only the listed fields, so the method takes no `extra_body`.
    client, async_client = both_clients()
    for resource in (
        client.audio.transcriptions,
        client.audio.transcriptions.with_response,
        async_client.audio.transcriptions,
        async_client.audio.transcriptions.with_response,
    ):
        assert "extra_body" not in inspect.signature(resource.create).parameters


@pytest.mark.parametrize("response_format", ["text", "srt", "vtt"])
def test_transcription_text_formats_return_str(client_kind: str, response_format: str) -> None:
    api, client = setup(client_kind, text_response(SRT_TEXT))
    result = call(client, "audio.transcriptions.create", file=WAV, model=MODEL_A, response_format=response_format)
    assert type(result) is str
    assert result == SRT_TEXT
    request = api.calls[0]
    assert request.headers["accept"] == "*/*"
    assert short(only(parts_of(request), "response_format")) == [
        ("response_format", None, response_format.encode()),
    ]


def test_transcription_verbose_json_returns_the_verbose_model(client_kind: str) -> None:
    api, client = setup(client_kind, json_response(VERBOSE_RESULT))
    result = call(client, "audio.transcriptions.create", file=WAV, model=MODEL_A, response_format="verbose_json")
    assert isinstance(result, _types().TranscriptionVerboseResponse)
    assert result.text == "Hello"
    assert result.words is not None
    assert result.words[0].word == "Hello"
    assert api.calls[0].headers["accept"] == "application/json"


@pytest.mark.parametrize("response_format", ["json", None], ids=["json", "absent"])
def test_transcription_json_or_absent_returns_the_plain_model(client_kind: str, response_format: str | None) -> None:
    api, client = setup(client_kind, json_response(TRANSCRIPTION_RESULT))
    kwargs: dict[str, object] = {"file": WAV, "model": MODEL_A}
    if response_format is not None:
        kwargs["response_format"] = response_format
    result = call(client, "audio.transcriptions.create", **kwargs)
    assert type(result) is _types().TranscriptionResponse
    assert result.text == "Hello"
    assert api.calls[0].headers["accept"] == "application/json"


def test_transcription_with_response_twin_of_a_text_format(client_kind: str) -> None:
    api, client = setup(client_kind, text_response(SRT_TEXT))
    response = call(
        client, "audio.transcriptions.create", with_response=True, file=WAV, model=MODEL_A, response_format="srt"
    )
    assert isinstance(response, APIResponse)
    assert response.data == SRT_TEXT
    assert response.request_id == REQ_ID
    assert len(api.calls) == 1


def test_transcription_path_file_uses_its_name_and_is_closed(
    client_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.wav"
    target.write_bytes(WAV)
    api, client = setup(client_kind, json_response(TRANSCRIPTION_RESULT))
    handles = track_opens(monkeypatch, target)
    call(client, "audio.transcriptions.create", file=target, model=MODEL_A)
    assert short(only(parts_of(api.calls[0]), "file")) == [("file", "clip.wav", WAV)]
    assert_closed_by_sdk(handles, target)


def test_transcription_caller_content_type_is_ignored(client_kind: str) -> None:
    api, client = setup(client_kind, json_response(TRANSCRIPTION_RESULT))
    call(client, "audio.transcriptions.create", file=WAV, extra_headers={"Content-Type": "text/plain"})
    assert api.calls[0].headers["content-type"].startswith("multipart/form-data; boundary=")


# --- video.generations.create, file form (Node video.ts:12-38, 75-103; tests/video-multipart.test.ts) --------------


def test_video_generation_file_form_scalars_as_text(client_kind: str) -> None:
    # video-multipart.test.ts:77-102.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    result = call(
        client,
        "video.generations.create",
        model=MODEL_A,
        prompt="A sunrise",
        image=PNG,
        duration=8,
        generate_audio=True,
        response_format="url",
    )
    request = api.calls[0]
    assert_multipart_post(request, "/video/generations")
    parts = parts_of(request)
    assert short(parts) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"A sunrise"),
        ("image", "upload", PNG),
        ("duration", None, b"8"),
        ("generate_audio", None, b"true"),
        ("response_format", None, b"url"),
    ]
    assert only(parts, "image")[0].content_type == OCTET_STREAM
    assert isinstance(result, _types().VideoGenerationResponse)


def test_video_generation_list_fields_give_one_part_per_item_under_the_plain_name(client_kind: str) -> None:
    # Node `video.ts:23-28`; video-multipart.test.ts:112-131: no `[]` suffix.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(
        client,
        "video.generations.create",
        model=MODEL_A,
        prompt="p",
        reference_images=[PNG, URL_INPUT],
        reference_audios=WAV,
        reference_videos=[("clip.mp4", MP4, "video/mp4")],
    )
    parts = parts_of(api.calls[0])
    assert short(parts) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"p"),
        ("reference_images", "upload", PNG),
        ("reference_images", None, URL_INPUT.encode()),
        ("reference_audios", "upload", WAV),
        ("reference_videos", "clip.mp4", MP4),
    ]
    assert only(parts, "reference_videos")[0].content_type == "video/mp4"
    assert not any(name is not None and name.endswith("[]") for name in names(parts))


def test_video_generation_text_reference_next_to_a_file_in_another_field(client_kind: str) -> None:
    # video-multipart.test.ts:133-145.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(
        client,
        "video.generations.create",
        model=MODEL_A,
        prompt="p",
        reference_images=URL_INPUT,
        reference_videos=[MP4],
    )
    parts = parts_of(api.calls[0])
    assert short(only(parts, "reference_images")) == [("reference_images", None, URL_INPUT.encode())]
    assert short(only(parts, "reference_videos")) == [("reference_videos", "upload", MP4)]


def test_video_generation_objects_are_json_text(client_kind: str) -> None:
    # Node `video.ts:33-36`: an object is sent as its JSON text, also as a list item.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(
        client,
        "video.generations.create",
        model=MODEL_A,
        prompt="p",
        image={"url": URL_INPUT},
        last_frame_image=PNG,
        reference_images=[{"url": URL_SECOND}, PNG],
    )
    assert short(parts_of(api.calls[0])) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"p"),
        ("image", None, b'{"url":"' + URL_INPUT.encode() + b'"}'),
        ("last_frame_image", "upload", PNG),
        ("reference_images", None, b'{"url":"' + URL_SECOND.encode() + b'"}'),
        ("reference_images", "upload", PNG),
    ]


def test_video_generation_every_field_is_forwarded_and_none_is_skipped(client_kind: str) -> None:
    # Node `video.ts:94`: every field of the body; video-multipart.test.ts:160-175: unset fields are left out.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(
        client,
        "video.generations.create",
        model=MODEL_A,
        prompt="p",
        image=PNG,
        seed=None,
        fps=24,
        extra_body={"custom_flag": False, "strength": 1.0, "meta": {"a": 1}, "tags": ["x", "y"], "unset": None},
    )
    assert short(parts_of(api.calls[0])) == [
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"p"),
        ("image", "upload", PNG),
        ("fps", None, b"24"),
        ("custom_flag", None, b"false"),
        ("strength", None, b"1"),
        ("meta", None, b'{"a":1}'),
        ("tags", None, b"x"),
        ("tags", None, b"y"),
    ]


def test_video_generation_without_a_file_stays_json(client_kind: str) -> None:
    # video-multipart.test.ts:16-61.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    params: dict[str, object] = {
        "model": MODEL_A,
        "prompt": "A sunrise",
        "duration": 8,
        "generate_audio": True,
        "image": URL_INPUT,
        "reference_images": [{"url": URL_SECOND}],
    }
    call(client, "video.generations.create", **params)
    request = api.calls[0]
    assert request.headers["content-type"] == "application/json"
    assert request.json() == params


def test_video_generation_caller_file_is_not_closed(client_kind: str) -> None:
    buffer = io.BytesIO(PNG)
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(client, "video.generations.create", model=MODEL_A, prompt="p", image=buffer)
    assert not buffer.closed
    assert short(only(parts_of(api.calls[0]), "image")) == [("image", "upload", PNG)]


# --- video.edits.create (Node video.ts:125-158; final.test.ts:158-166) ----------------------------------------------


def test_video_edit_sends_the_fixed_list_in_order(client_kind: str) -> None:
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    points = [{"x": 10, "y": 20, "type": "positive"}]
    result = call(
        client,
        "video.edits.create",
        response_format="url",
        output_container_and_codec="mp4_h264",
        desired_increase=2,
        background_color="#000000",
        preserve_audio=False,
        auto_trim=True,
        frame_index=3,
        key_points=points,
        mask_url=URL_SECOND,
        mask=b"mask",
        prompt="Make the sky blue",
        model=MODEL_A,
        video=MP4,
    )
    request = api.calls[0]
    # Node sends a video edit to the generations path (`video.ts:150-157`).
    assert_multipart_post(request, "/video/generations")
    parts = parts_of(request)
    assert names(parts) == list(VIDEO_EDIT_FIELDS)
    assert short(parts) == [
        ("video", "upload", MP4),
        ("model", None, MODEL_A.encode()),
        ("prompt", None, b"Make the sky blue"),
        ("mask", "upload", b"mask"),
        ("mask_url", None, URL_SECOND.encode()),
        ("key_points", None, b'[{"x":10,"y":20,"type":"positive"}]'),
        ("frame_index", None, b"3"),
        ("auto_trim", None, b"true"),
        ("preserve_audio", None, b"false"),
        ("background_color", None, b"#000000"),
        ("desired_increase", None, b"2"),
        ("output_container_and_codec", None, b"mp4_h264"),
        ("response_format", None, b"url"),
    ]
    assert only(parts, "video")[0].content_type == OCTET_STREAM
    assert isinstance(result, _types().VideoGenerationResponse)


def test_video_edit_minimal_puts_the_video_first(client_kind: str) -> None:
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(client, "video.edits.create", model=MODEL_A, video=MP4)
    assert short(parts_of(api.calls[0])) == [("video", "upload", MP4), ("model", None, MODEL_A.encode())]


def test_video_edit_key_points_string_is_sent_as_is(client_kind: str) -> None:
    # blackbox.test.ts:205: a JSON string is sent unchanged.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    call(client, "video.edits.create", model=MODEL_A, video=MP4, key_points='{"a": 1, "b": "2"}')
    assert short(only(parts_of(api.calls[0]), "key_points")) == [("key_points", None, b'{"a": 1, "b": "2"}')]


@pytest.mark.parametrize("mask", ["absent", None], ids=["absent", "none"])
def test_video_edit_mask_part_only_when_given(client_kind: str, mask: object) -> None:
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    kwargs: dict[str, object] = {"model": MODEL_A, "video": MP4, "prompt": "p"}
    if mask != "absent":
        kwargs["mask"] = mask
    call(client, "video.edits.create", **kwargs)
    assert names(parts_of(api.calls[0])) == ["video", "model", "prompt"]


@pytest.mark.parametrize("with_response", [False, True], ids=["method", "with_response"])
@pytest.mark.parametrize(
    "unknown",
    [{"seed": 5}, {"negative_prompt": "blur"}, {"extra_body": {"x": 1}}],
    ids=["seed", "negative_prompt", "extra_body"],
)
def test_video_edit_unknown_keyword_raises_type_error(
    client_kind: str, with_response: bool, unknown: dict[str, object]
) -> None:
    # Lead decision (gap 4): only Node's fixed list is accepted; any other keyword is a `TypeError`, no request.
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    with pytest.raises(TypeError):
        call(client, "video.edits.create", with_response=with_response, model=MODEL_A, video=MP4, **unknown)
    assert api.calls == []


def test_video_edit_methods_take_no_extra_body() -> None:
    client, async_client = both_clients()
    for resource in (
        client.video.edits,
        client.video.edits.with_response,
        async_client.video.edits,
        async_client.video.edits.with_response,
    ):
        assert "extra_body" not in inspect.signature(resource.create).parameters


def test_video_edit_path_video_and_tuple_mask(
    client_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "input.mp4"
    target.write_bytes(MP4)
    api, client = setup(client_kind, json_response(VIDEO_RESULT))
    handles = track_opens(monkeypatch, target)
    call(client, "video.edits.create", model=MODEL_A, video=target, mask=("mask.png", b"mask", "image/png"))
    parts = parts_of(api.calls[0])
    assert short(only(parts, "video")) == [("video", "input.mp4", MP4)]
    assert only(parts, "mask") == [Part("mask", "mask.png", "image/png", b"mask")]
    assert_closed_by_sdk(handles, target)


# --- inference.run with a file in the payload (Node inference.ts:74-85, the FormData case) --------------------------


def test_inference_payload_with_a_file_is_multipart(client_kind: str) -> None:
    api, client = setup(client_kind, json_response({"ok": True}))
    result = call(
        client,
        "inference.run",
        MODEL_A,
        {
            "file": ("input.bin", b"\x01\x02\x03\x04"),
            "mode": "fast",
            "steps": 4,
            "flag": True,
            "options": {"top_k": 2},
            "tags": ["a", "b"],
            "skip": None,
        },
    )
    request = api.calls[0]
    assert_multipart_post(request, "/inference/fixture%2Fmodel-a")
    assert request.headers["accept"] == "application/json"
    assert short(parts_of(request)) == [
        ("file", "input.bin", b"\x01\x02\x03\x04"),
        ("mode", None, b"fast"),
        ("steps", None, b"4"),
        ("flag", None, b"true"),
        ("options", None, b'{"top_k":2}'),
        ("tags", None, b"a"),
        ("tags", None, b"b"),
    ]
    assert result == {"ok": True}


def test_inference_list_holding_a_file_is_multipart_and_extra_body_is_forwarded(client_kind: str) -> None:
    api, client = setup(client_kind, json_response({"ok": True}))
    call(client, "inference.run", MODEL_A, {"images": [PNG, URL_INPUT]}, extra_body={"seed": 7})
    assert short(parts_of(api.calls[0])) == [
        ("images", "upload", PNG),
        ("images", None, URL_INPUT.encode()),
        ("seed", None, b"7"),
    ]


def test_inference_path_in_the_payload_is_closed(
    client_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.wav"
    target.write_bytes(WAV)
    api, client = setup(client_kind, json_response({"ok": True}))
    handles = track_opens(monkeypatch, target)
    call(client, "inference.run", MODEL_A, {"audio": target, "language": "fr"})
    assert short(parts_of(api.calls[0])) == [("audio", "clip.wav", WAV), ("language", None, b"fr")]
    assert_closed_by_sdk(handles, target)


def test_inference_caller_content_type_is_ignored_in_the_file_form(client_kind: str) -> None:
    api, client = setup(client_kind, json_response({"ok": True}))
    call(client, "inference.run", MODEL_A, {"file": PNG}, extra_headers={"Content-Type": "application/json"})
    assert api.calls[0].headers["content-type"].startswith("multipart/form-data; boundary=")


def test_inference_payload_without_a_file_stays_json(client_kind: str) -> None:
    # inference-run.test.ts:18-31.
    api, client = setup(client_kind, json_response({"ok": True}))
    payload: dict[str, object] = {"inputs": "hello", "options": {"top_k": 2}, "flag": True}
    result = call(client, "inference.run", MODEL_A, payload)
    request = api.calls[0]
    assert request.headers["content-type"] == "application/json"
    assert request.json() == payload
    assert result == {"ok": True}


# --- billed retry of every multipart method ---------------------------------------------------------------------------

#: Each multipart call with a caller file object (read once by the SDK), and a valid success body.
RETRY_CASES: dict[str, tuple[Callable[[], tuple[tuple[object, ...], dict[str, object]]], dict[str, object]]] = {
    "audio.transcriptions.create": (lambda: ((), {"file": io.BytesIO(WAV), "model": MODEL_A}), TRANSCRIPTION_RESULT),
    "video.generations.create": (
        lambda: ((), {"model": MODEL_A, "prompt": "p", "image": io.BytesIO(PNG)}),
        VIDEO_RESULT,
    ),
    "video.edits.create": (lambda: ((), {"model": MODEL_A, "video": io.BytesIO(MP4)}), VIDEO_RESULT),
    "inference.run": (lambda: ((MODEL_A, {"file": io.BytesIO(PNG), "mode": "fast"}), {}), {"ok": True}),
}


@pytest.mark.parametrize("name", list(RETRY_CASES))
def test_multipart_billed_retry_sends_the_same_body(client_kind: str, name: str) -> None:
    factory, success = RETRY_CASES[name]
    args, kwargs = factory()
    api, client = setup(client_kind, error_response(503, "Busy"), json_response(success))
    call(client, name, *args, **kwargs)
    assert len(api.calls) == 2
    first, second = api.calls
    assert parts_of(first) == parts_of(second)
    assert normalized_body(first) == normalized_body(second)
    assert any(part.filename is not None and part.payload for part in parts_of(second))
