"""Review P3 finding L-8: a tuple of several files is a list of inputs, not one ``(filename, content)`` file tuple.

A tuple is a file tuple only when it has 2 or 3 items, its first item is a ``str`` (the filename) and its second item is
bytes-like (``bytes``, ``bytearray``, ``memoryview``) or a binary file object. Any other tuple (``(Path, Path)``,
``(b"...", b"...")``, ``("a.png", Path(...))``) is read like a list: one part per item.

``devupai.resources.shared`` is imported inside the tests, so each test reports on its own.
"""

from __future__ import annotations

import email.parser
import email.policy
import importlib
import io
from pathlib import Path
from typing import Any

import pytest
from helpers.mock_transport import MODEL_A, MockAPI, RecordedRequest, json_response, sequence
from helpers.samples import invoke, method_of

from devupai import AsyncDevupAI, DevupAI

BILLING: dict[str, object] = {"cost_dzd": 1.5, "balance_dzd": 98.5}
IMAGE_RESULT: dict[str, object] = {"data": [{"url": "https://cdn.example.test/fixture.png"}], "_devup": BILLING}
VIDEO_RESULT: dict[str, object] = {"data": [{"url": "https://cdn.example.test/fixture.mp4"}], "_devup": BILLING}
FIRST = b"\x89PNG-first"
SECOND = b"\x89PNG-second"

Short = tuple[str | None, str | None, str | None, bytes]


def _shared() -> Any:
    return importlib.import_module("devupai.resources.shared")


def setup(kind: str, result: dict[str, object]) -> tuple[MockAPI, DevupAI | AsyncDevupAI]:
    api = MockAPI(sequence(json_response(result)))
    return api, api.client(kind)


def parts_of(request: RecordedRequest) -> list[Short]:
    """``(name, filename, content_type, payload)`` of each part of a recorded multipart body, in order."""
    content_type = request.headers.get("content-type", "")
    assert content_type.startswith("multipart/form-data; boundary="), content_type
    raw = b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + request.content
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
    assert message.is_multipart()
    parts: list[Short] = []
    for item in message.iter_parts():
        name = item.get_param("name", header="content-disposition")
        payload = item.get_payload(decode=True)
        assert isinstance(payload, bytes)
        parts.append(
            (
                name if isinstance(name, str) else None,
                item.get_filename(),
                item.get_content_type() if item.get("content-type") is not None else None,
                payload,
            )
        )
    return parts


@pytest.fixture
def two_files(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(FIRST)
    second.write_bytes(SECOND)
    return first, second


# --- is_file_input ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(("a.png", FIRST), id="name-bytes"),
        pytest.param(("a.png", bytearray(FIRST)), id="name-bytearray"),
        pytest.param(("a.png", memoryview(FIRST)), id="name-memoryview"),
        pytest.param(("a.png", FIRST, "image/png"), id="name-bytes-type"),
        pytest.param(("a.png", io.BytesIO(FIRST)), id="name-file"),
        pytest.param(("a.png", io.BytesIO(FIRST), "image/png"), id="name-file-type"),
        pytest.param(("a.png", FIRST, None), id="name-bytes-none-type"),
    ],
)
def test_a_real_file_tuple_is_one_file(value: object) -> None:
    assert _shared().is_file_input(value) is True


@pytest.mark.parametrize(
    "value",
    [
        pytest.param((Path("a.png"), Path("b.png")), id="two-paths"),
        pytest.param((FIRST, SECOND), id="two-bytes"),
        pytest.param((io.BytesIO(FIRST), io.BytesIO(SECOND)), id="two-file-objects"),
        pytest.param(("a.png", Path("b.png")), id="name-then-path"),
        pytest.param((7, FIRST), id="non-str-name"),
        pytest.param(("a.png", FIRST, "image/png", "extra"), id="four-items"),
    ],
)
def test_any_other_tuple_is_not_a_file(value: object) -> None:
    assert _shared().is_file_input(value) is False


def test_a_tuple_of_files_contains_files() -> None:
    assert _shared().contains_file_input((Path("a.png"), Path("b.png"))) is True
    assert _shared().contains_file_input((FIRST, SECOND)) is True


def test_read_file_input_refuses_a_tuple_that_is_not_a_file_tuple() -> None:
    with pytest.raises(TypeError):
        _shared().read_file_input((Path("a.png"), Path("b.png")))
    with pytest.raises(TypeError):
        _shared().read_file_input(("a.png", Path("b.png")))


# --- images.edit ------------------------------------------------------------------------------------------------------


def test_edit_images_tuple_of_two_paths_is_two_image_parts(client_kind: str, two_files: tuple[Path, Path]) -> None:
    api, client = setup(client_kind, IMAGE_RESULT)
    invoke(method_of(client, "images.edit"), model=MODEL_A, prompt="p", images=two_files)
    assert len(api.calls) == 1
    assert parts_of(api.calls[0]) == [
        ("model", None, None, MODEL_A.encode()),
        ("prompt", None, None, b"p"),
        ("image", "a.png", "application/octet-stream", FIRST),
        ("image", "b.png", "application/octet-stream", SECOND),
    ]


def test_edit_image_tuple_of_two_bytes_is_two_image_parts(client_kind: str) -> None:
    api, client = setup(client_kind, IMAGE_RESULT)
    invoke(method_of(client, "images.edit"), model=MODEL_A, prompt="p", image=(FIRST, SECOND))
    assert [part for part in parts_of(api.calls[0]) if part[0] == "image"] == [
        ("image", "upload", "application/octet-stream", FIRST),
        ("image", "upload", "application/octet-stream", SECOND),
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(("a.png", FIRST), ("image", "a.png", "application/octet-stream", FIRST), id="two-items"),
        pytest.param(("a.png", FIRST, "image/png"), ("image", "a.png", "image/png", FIRST), id="with-type"),
    ],
)
def test_edit_a_real_file_tuple_is_still_one_file(client_kind: str, value: object, expected: Short) -> None:
    api, client = setup(client_kind, IMAGE_RESULT)
    invoke(method_of(client, "images.edit"), model=MODEL_A, prompt="p", image=value)
    assert [part for part in parts_of(api.calls[0]) if part[0] == "image"] == [expected]


def test_edit_name_then_path_tuple_is_a_text_part_and_a_file(client_kind: str, two_files: tuple[Path, Path]) -> None:
    # ("a.png", Path) is not a file tuple (its second item is not bytes-like or a file object): it is read as a list,
    # so the str is a text input and the path a file input.
    api, client = setup(client_kind, IMAGE_RESULT)
    invoke(
        method_of(client, "images.edit"),
        model=MODEL_A,
        prompt="p",
        images=("https://cdn.example.test/x.png", two_files[1]),
    )
    assert [part for part in parts_of(api.calls[0]) if part[0] == "image"] == [
        ("image", None, None, b"https://cdn.example.test/x.png"),
        ("image", "b.png", "application/octet-stream", SECOND),
    ]


# --- video.generations.create and inference.run -----------------------------------------------------------------------


def test_video_generation_tuple_of_paths_is_one_part_per_item(client_kind: str, two_files: tuple[Path, Path]) -> None:
    api, client = setup(client_kind, VIDEO_RESULT)
    invoke(method_of(client, "video.generations.create"), model=MODEL_A, prompt="p", reference_images=two_files)
    assert parts_of(api.calls[0]) == [
        ("model", None, None, MODEL_A.encode()),
        ("prompt", None, None, b"p"),
        ("reference_images", "a.png", "application/octet-stream", FIRST),
        ("reference_images", "b.png", "application/octet-stream", SECOND),
    ]


def test_inference_payload_tuple_of_bytes_is_one_part_per_item(client_kind: str) -> None:
    api, client = setup(client_kind, {"ok": True})
    result = invoke(method_of(client, "inference.run"), MODEL_A, {"files": (FIRST, SECOND), "mode": "fast"})
    assert result == {"ok": True}
    assert parts_of(api.calls[0]) == [
        ("files", "upload", "application/octet-stream", FIRST),
        ("files", "upload", "application/octet-stream", SECOND),
        ("mode", None, None, b"fast"),
    ]
