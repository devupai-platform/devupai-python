"""Review P4 finding D-4: an image-edit input item the form cannot use raises ``TypeError`` before anything is sent.

When an input image of ``images.edit`` is a file, the request is sent as ``multipart/form-data``. Each item of
``image``, ``images``, ``mask`` and ``markup`` is then a file, a URL or data URI string, or ``{"image_url": str}``.
Any other item (``{"url": ...}``, ``{"image_url": 3}``, ``5``, a nested list, ...) raises ``TypeError`` naming the
field, on both clients and on ``with_response``, before any file is opened or read and before any request.

Unchanged: ``None`` for a whole field, or as a list item, is left out; a request without a file input is sent as JSON,
as given; every valid item gives the same part as before.
"""

from __future__ import annotations

import builtins
import email.parser
import email.policy
import io
import os
from pathlib import Path
from typing import Any

import pytest
from helpers.mock_transport import MODEL_A, MockAPI, RecordedRequest, json_response, sequence
from helpers.samples import invoke, method_of

from devupai import APIResponse, AsyncDevupAI, DevupAI
from devupai.types import ImageGenerateResponse

BILLING: dict[str, object] = {"cost_dzd": 1.5, "balance_dzd": 98.5}
IMAGE_RESULT: dict[str, object] = {"data": [{"url": "https://cdn.example.test/fixture.png"}], "_devup": BILLING}
PNG = b"\x89PNG-review"
MASK = b"mask-bytes"
URL_INPUT = "https://cdn.example.test/input.png"
DATA_URI = "data:image/png;base64,AA=="
OCTET_STREAM = "application/octet-stream"

Short = tuple[str | None, str | None, str | None, bytes]

FIELDS = ("image", "images", "mask", "markup")
ALLOWED = 'a file, a URL or data URI string, or {"image_url": "..."}.'


def single_message(field: str) -> str:
    return f"{field} must be {ALLOWED}"


def item_message(field: str) -> str:
    return f"Each item of {field} must be {ALLOWED}"


class Opaque:
    """An object that is neither a file nor a string nor a mapping."""


#: Items the form cannot use (one value each).
BAD_ITEMS: dict[str, Any] = {
    "url-key-mapping": {"url": URL_INPUT},
    "image-url-int": {"image_url": 3},
    "image-url-none": {"image_url": None},
    "image-url-nested": {"image_url": {"url": URL_INPUT}},
    "empty-mapping": {},
    "int": 5,
    "float": 1.5,
    "bool": True,
    "object": Opaque(),
}

#: Items the form cannot use that only make sense inside a list.
BAD_LIST_ITEMS: dict[str, Any] = {
    **BAD_ITEMS,
    "nested-list": [PNG],
    "nested-non-file-tuple": (PNG, PNG),
}


def setup(kind: str) -> tuple[MockAPI, DevupAI | AsyncDevupAI]:
    api = MockAPI(sequence(json_response(IMAGE_RESULT)))
    return api, api.client(kind)


def edit(client: DevupAI | AsyncDevupAI, *, with_response: bool = False, **params: Any) -> Any:
    return invoke(method_of(client, "images.edit", with_response=with_response), **params)


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


def track_opens(monkeypatch: pytest.MonkeyPatch, target: Path) -> list[Any]:
    """Records every file object opened on ``target`` through ``open`` / ``io.open``."""
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


def other_file_field(field: str) -> dict[str, object]:
    """A file in another input field, so the request takes the multipart form whatever ``field`` holds."""
    return {"mask": MASK} if field in ("image", "images") else {"image": PNG}


# --- bad items: TypeError, nothing sent ------------------------------------------------------------------------------


def single_cases() -> list[Any]:
    return [pytest.param(field, value, id=f"{field}-{name}") for field in FIELDS for name, value in BAD_ITEMS.items()]


def list_cases() -> list[Any]:
    return [
        pytest.param(field, value, id=f"{field}-{name}") for field in FIELDS for name, value in BAD_LIST_ITEMS.items()
    ]


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with-response"])
@pytest.mark.parametrize(("field", "bad"), single_cases())
def test_a_bad_single_value_raises_type_error_and_sends_nothing(
    client_kind: str, with_response: bool, field: str, bad: Any
) -> None:
    api, client = setup(client_kind)
    params: dict[str, object] = {"model": MODEL_A, "prompt": "p", field: bad, **other_file_field(field)}
    with pytest.raises(TypeError) as caught:
        edit(client, with_response=with_response, **params)
    assert str(caught.value) == single_message(field)
    assert api.calls == []


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with-response"])
@pytest.mark.parametrize(("field", "bad"), list_cases())
def test_a_bad_list_item_raises_type_error_and_sends_nothing(
    client_kind: str, with_response: bool, field: str, bad: Any
) -> None:
    # The list itself holds a file, so the form is used even without another file field.
    api, client = setup(client_kind)
    params: dict[str, object] = {"model": MODEL_A, "prompt": "p", field: [PNG, bad]}
    with pytest.raises(TypeError) as caught:
        edit(client, with_response=with_response, **params)
    assert str(caught.value) == item_message(field)
    assert api.calls == []


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with-response"])
def test_the_review_examples_raise_type_error(client_kind: str, with_response: bool) -> None:
    # The three cases of the review: `mask={"url": ...}` with a file image, `image=[bytes, 5]`, `mask={"image_url": 3}`.
    cases: list[tuple[dict[str, object], str]] = [
        ({"image": PNG, "mask": {"url": URL_INPUT}}, single_message("mask")),
        ({"image": [PNG, 5]}, item_message("image")),
        ({"image": PNG, "mask": {"image_url": 3}}, single_message("mask")),
    ]
    for extra, message in cases:
        api, client = setup(client_kind)
        with pytest.raises(TypeError) as caught:
            edit(client, with_response=with_response, model=MODEL_A, prompt="p", **extra)
        assert str(caught.value) == message
        assert api.calls == []


def test_a_non_file_tuple_with_a_bad_item_raises_type_error(client_kind: str) -> None:
    # (7, bytes) is not a file tuple (its first item is not a str), so it is read as a list: 7 is not usable.
    api, client = setup(client_kind)
    with pytest.raises(TypeError) as caught:
        edit(client, model=MODEL_A, prompt="p", image=(7, PNG))
    assert str(caught.value) == item_message("image")
    assert api.calls == []


def test_a_bad_item_in_extra_body_raises_type_error(client_kind: str) -> None:
    # extra_body is merged into the body, so its input fields are checked the same way.
    api, client = setup(client_kind)
    with pytest.raises(TypeError) as caught:
        edit(client, model=MODEL_A, prompt="p", image=PNG, extra_body={"markup": {"url": URL_INPUT}})
    assert str(caught.value) == single_message("markup")
    assert api.calls == []


# --- files: nothing is opened or read before the TypeError -----------------------------------------------------------


def test_a_caller_file_object_is_not_read_or_closed(client_kind: str) -> None:
    buffer = io.BytesIO(PNG)
    api, client = setup(client_kind)
    with pytest.raises(TypeError):
        edit(client, model=MODEL_A, prompt="p", image=[buffer, 5])
    assert not buffer.closed
    assert buffer.tell() == 0
    assert api.calls == []


def test_a_caller_file_object_before_a_bad_mask_is_not_read_or_closed(client_kind: str, tmp_path: Path) -> None:
    target = tmp_path / "photo.png"
    target.write_bytes(PNG)
    api, client = setup(client_kind)
    with open(target, "rb") as handle:
        with pytest.raises(TypeError):
            edit(client, model=MODEL_A, prompt="p", image=handle, mask={"url": URL_INPUT})
        assert not handle.closed
        assert handle.tell() == 0
    assert api.calls == []


def test_a_path_is_not_opened_when_a_later_item_is_bad(
    client_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "photo.png"
    target.write_bytes(PNG)
    handles = track_opens(monkeypatch, target)
    api, client = setup(client_kind)
    with pytest.raises(TypeError):
        edit(client, model=MODEL_A, prompt="p", image=target, images=[target], markup=5)
    assert handles == []
    target.unlink()  # on Windows this also fails when a handle is still open
    assert api.calls == []


def test_a_path_is_opened_and_closed_by_the_sdk_on_a_valid_edit(
    client_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "photo.png"
    target.write_bytes(PNG)
    handles = track_opens(monkeypatch, target)
    api, client = setup(client_kind)
    edit(client, model=MODEL_A, prompt="p", image=target, mask=MASK)
    assert handles, "the path was not opened with open()"
    assert all(handle.closed for handle in handles)
    target.unlink()
    assert len(api.calls) == 1
    assert [part for part in parts_of(api.calls[0]) if part[0] == "image"] == [
        ("image", "photo.png", OCTET_STREAM, PNG)
    ]


# --- valid items: the same parts as before ----------------------------------------------------------------------------


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with-response"])
def test_every_valid_item_kind_gives_the_same_parts_as_before(
    client_kind: str, with_response: bool, tmp_path: Path
) -> None:
    path = tmp_path / "from-path.png"
    path.write_bytes(b"path-bytes")
    named = io.BytesIO(b"named-bytes")
    named.name = "dir/named.png"  # type: ignore[attr-defined]
    api, client = setup(client_kind)
    result = edit(
        client,
        with_response=with_response,
        model=MODEL_A,
        prompt="p",
        image=[
            PNG,
            bytearray(b"bytearray"),
            memoryview(b"memoryview"),
            path,
            io.BytesIO(b"bytesio"),
            named,
            ("tuple.png", b"tuple-bytes"),
            ("typed.png", io.BytesIO(b"typed-bytes"), "image/png"),
            URL_INPUT,
            DATA_URI,
            {"image_url": URL_INPUT, "detail": "kept out"},
            None,
        ],
        images=(b"tuple-list-a", b"tuple-list-b"),
        mask={"image_url": DATA_URI},
        markup=("markup.png", b"markup-bytes", None),
        n=2,
    )
    if with_response:
        assert isinstance(result, APIResponse)
        assert isinstance(result.data, ImageGenerateResponse)
    else:
        assert isinstance(result, ImageGenerateResponse)
    assert len(api.calls) == 1
    assert parts_of(api.calls[0]) == [
        ("model", None, None, MODEL_A.encode()),
        ("prompt", None, None, b"p"),
        ("image", "upload", OCTET_STREAM, PNG),
        ("image", "upload", OCTET_STREAM, b"bytearray"),
        ("image", "upload", OCTET_STREAM, b"memoryview"),
        ("image", "from-path.png", OCTET_STREAM, b"path-bytes"),
        ("image", "upload", OCTET_STREAM, b"bytesio"),
        ("image", "named.png", OCTET_STREAM, b"named-bytes"),
        ("image", "tuple.png", OCTET_STREAM, b"tuple-bytes"),
        ("image", "typed.png", "image/png", b"typed-bytes"),
        ("image", None, None, URL_INPUT.encode()),
        ("image", None, None, DATA_URI.encode()),
        ("image", None, None, URL_INPUT.encode()),
        ("image", "upload", OCTET_STREAM, b"tuple-list-a"),
        ("image", "upload", OCTET_STREAM, b"tuple-list-b"),
        ("mask", None, None, DATA_URI.encode()),
        ("markup", "markup.png", OCTET_STREAM, b"markup-bytes"),
        ("n", None, None, b"2"),
    ]


def test_none_for_a_whole_field_or_a_list_item_is_left_out(client_kind: str) -> None:
    api, client = setup(client_kind)
    edit(client, model=MODEL_A, prompt="p", image=[None, PNG, None], images=None, mask=None, markup=None)
    assert parts_of(api.calls[0]) == [
        ("model", None, None, MODEL_A.encode()),
        ("prompt", None, None, b"p"),
        ("image", "upload", OCTET_STREAM, PNG),
    ]


# --- the JSON form is unchanged ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"image": 5}, id="int-image"),
        pytest.param({"image": URL_INPUT, "mask": {"url": URL_INPUT}}, id="url-key-mask"),
        pytest.param({"images": [URL_INPUT, {"image_url": 3}]}, id="image-url-int"),
    ],
)
def test_without_a_file_input_the_body_is_sent_as_json_as_given(client_kind: str, extra: dict[str, object]) -> None:
    # No item is a file, so no form is built: the JSON body is sent as given and the API answers it.
    api, client = setup(client_kind)
    params: dict[str, object] = {"model": MODEL_A, "prompt": "p", **extra}
    edit(client, **params)
    assert len(api.calls) == 1
    assert api.calls[0].headers["content-type"] == "application/json"
    assert api.calls[0].json() == params
