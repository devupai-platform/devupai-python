"""File inputs and ``multipart/form-data`` bodies shared by the methods that upload files.

A file input is raw bytes (``bytes``, ``bytearray``, ``memoryview``), a path (any ``os.PathLike``), an open binary file
object, or a ``(filename, content)`` / ``(filename, content, content_type)`` tuple. The SDK opens and closes a path
itself; a file object given by the caller is read and never closed.

A form is a list of ``(name, value)`` parts, sent in order: a ``str`` value is a text part, a
``(filename, content, content_type)`` value is a file part.
"""

from __future__ import annotations

import decimal
import json
import math
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

__all__ = [
    "DEFAULT_FILENAME",
    "MultipartValue",
    "MultipartPart",
    "is_file_input",
    "contains_file_input",
    "read_file_input",
    "form_field",
    "json_text",
    "image_edit_form",
    "every_field_form",
    "fixed_form",
]

DEFAULT_FILENAME = "upload"
"""The filename of a file part whose input has no name of its own (bytes, or a file object without a ``str`` name)."""

MultipartValue: TypeAlias = str | tuple[str, bytes, str | None]
"""The value of one form part: text, or ``(filename, content, content_type)`` for a file."""

MultipartPart: TypeAlias = tuple[str, MultipartValue]
"""One form part: its name and its value."""

#: The request fields of an image edit that carry input images (``images`` is the list form of ``image``).
_IMAGE_INPUT_FIELDS = ("image", "images", "mask", "markup")

_BYTES_TYPES = (bytes, bytearray, memoryview)


# --- file inputs ------------------------------------------------------------------------------------------------------


def _is_bytes(value: object) -> bool:
    """Whether ``value`` is ``bytes``, a ``bytearray`` or a ``memoryview``."""
    return isinstance(value, _BYTES_TYPES)


def _bytes_of(value: object) -> bytes:
    """A copy of ``bytes``, ``bytearray`` or ``memoryview`` content as ``bytes``."""
    return bytes(cast("bytes", value))


def _is_reader(value: object) -> bool:
    """Whether ``value`` is a file object: it has a ``read`` method."""
    return callable(getattr(value, "read", None))


def _is_file_tuple(items: tuple[object, ...]) -> bool:
    """Whether ``items`` is ``(filename, content)`` or ``(filename, content, content_type)``: two or three items, a
    ``str`` filename first, and bytes-like content or a file object second."""
    return len(items) in (2, 3) and isinstance(items[0], str) and (_is_bytes(items[1]) or _is_reader(items[1]))


def is_file_input(value: object) -> bool:
    """Whether ``value`` is a file: bytes, a path, an object with a ``read`` method, or a ``(filename, content)`` /
    ``(filename, content, content_type)`` tuple.

    A tuple is a file tuple only when its first item is a ``str`` and its second item is bytes-like or a file object;
    any other tuple (two paths, two bytes values, ...) is not a file, and a form reads it like a list.
    """
    if isinstance(value, tuple):
        return _is_file_tuple(cast("tuple[object, ...]", value))
    return _is_bytes(value) or isinstance(value, os.PathLike) or _is_reader(value)


def contains_file_input(value: object) -> bool:
    """Whether ``value``, or an item of a ``list`` or ``tuple`` value, is a file."""
    if is_file_input(value):
        return True
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return any(is_file_input(item) for item in items)
    return False


def _read_path(value: os.PathLike[str] | os.PathLike[bytes]) -> tuple[str, bytes]:
    """The name and the bytes of the file at ``value``; the file is opened and closed here."""
    path = os.fsdecode(os.fspath(value))
    with open(path, "rb") as handle:
        data = handle.read()
    return Path(path).name or DEFAULT_FILENAME, data


def _read_object(value: object) -> bytes:
    """The bytes of a caller's file object, read with ``read()``. The object is never closed."""
    read = cast("Callable[[], object]", getattr(value, "read", None))
    data = read()
    if not _is_bytes(data):
        raise TypeError("A file object must be opened in binary mode ('rb').")
    return _bytes_of(data)


def _object_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return Path(name).name or DEFAULT_FILENAME
    return DEFAULT_FILENAME


def _read_content(value: object) -> bytes:
    """The bytes of the content of a ``(filename, content[, content_type])`` tuple: bytes-like, or a file object."""
    if _is_bytes(value):
        return _bytes_of(value)
    return _read_object(value)


def read_file_input(value: object) -> tuple[str, bytes, str | None]:
    """Reads a file input into ``(filename, content, content_type)``.

    - ``bytes``, ``bytearray`` or ``memoryview``: ``("upload", bytes(value), None)``.
    - A path (``os.PathLike``): the file is opened, read and closed by the SDK; the filename is its base name.
    - A binary file object: read with ``read()`` and never closed; the filename is the base name of its ``name``
      when that is a ``str``, ``"upload"`` otherwise.
    - ``(filename, content)`` or ``(filename, content, content_type)``, where ``filename`` is a ``str`` and
      ``content`` is bytes-like or a binary file object (read and never closed).

    The content type is ``None`` unless a tuple gives one.
    """
    if isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        if not _is_file_tuple(items):
            raise TypeError(
                "A file tuple is (filename, content) or (filename, content, content_type), with a str filename and "
                "bytes or a binary file object as content."
            )
        content_type = items[2] if len(items) == 3 else None
        if content_type is not None and not isinstance(content_type, str):
            raise TypeError("The content type of a file tuple must be a str or None.")
        return cast("str", items[0]), _read_content(items[1]), content_type
    if _is_bytes(value):
        return DEFAULT_FILENAME, _bytes_of(value), None
    if isinstance(value, os.PathLike):
        name, data = _read_path(cast("os.PathLike[str]", value))
        return name, data, None
    if callable(getattr(value, "read", None)):
        return _object_name(value), _read_object(value), None
    raise TypeError("A file must be bytes, a path, a binary file object or a (filename, content) tuple.")


# --- text values ------------------------------------------------------------------------------------------------------


def _number_text(value: float) -> str:
    """``value`` written the way JavaScript's ``String(number)`` writes it (``1.0`` is ``"1"``, ``1e-7`` is
    ``"1e-7"``)."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    parsed = decimal.Decimal(repr(abs(value))).normalize().as_tuple()
    digits = "".join(str(digit) for digit in parsed.digits)
    exponent = cast("int", parsed.exponent)
    count = len(digits)
    point = exponent + count  # the position of the decimal point, counted from the first digit
    if count <= point <= 21:
        return sign + digits + "0" * (point - count)
    if 0 < point <= 21:
        return sign + digits[:point] + "." + digits[point:]
    if -6 < point <= 0:
        return sign + "0." + "0" * (-point) + digits
    power = point - 1
    mantissa = digits if count == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'+' if power >= 0 else '-'}{abs(power)}"


def form_field(value: object) -> str:
    """The text of a scalar form field, as JavaScript's ``String(value)`` gives it.

    ``True`` is ``"true"`` and ``False`` is ``"false"``; an integral float is written without a fraction (``1.0`` is
    ``"1"``); a ``str`` is kept as it is.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number_text(value)
    return str(value)


def json_text(value: object) -> str:
    """The compact JSON text of ``value`` (``{"a":1}``), as JavaScript's ``JSON.stringify`` writes it."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# --- forms ------------------------------------------------------------------------------------------------------------


def _items(value: object) -> list[object]:
    """The items of an input field: a list (or a tuple that is not a file) as is, one value as a list of one."""
    if value is None:
        return []
    if not is_file_input(value) and isinstance(value, (list, tuple)):
        return list(cast("list[object] | tuple[object, ...]", value))
    return [value]


def _image_input_part(item: object) -> MultipartValue | None:
    """One input image as a part value: a file, a text value (URL or data URI), or the ``image_url`` of an object."""
    if isinstance(item, str):
        return item
    if is_file_input(item):
        return read_file_input(item)
    if isinstance(item, Mapping):
        image_url = cast("Mapping[str, object]", item).get("image_url")
        if isinstance(image_url, str):
            return image_url
    return None


def image_edit_form(body: Mapping[str, object]) -> list[MultipartPart]:
    """The form of an image edit with a file input, from its fields in order.

    Each item of ``image`` and ``images`` is an ``image`` part, ``mask`` and ``markup`` are parts of their own name: a
    file is a file part, a string a text part, and ``{"image_url": ...}`` a text part with the URL. Every other field
    is sent too: an object or a list as one JSON-text part, a file as a file part, a scalar through
    :func:`form_field`. ``None`` is left out.
    """
    parts: list[MultipartPart] = []
    for name, value in body.items():
        if name in _IMAGE_INPUT_FIELDS:
            part_name = "image" if name == "images" else name
            for item in _items(value):
                part = _image_input_part(item)
                if part is not None:
                    parts.append((part_name, part))
        elif value is None:
            continue
        elif is_file_input(value):
            parts.append((name, read_file_input(value)))
        elif isinstance(value, (Mapping, list, tuple)):
            parts.append((name, json_text(cast("object", value))))
        else:
            parts.append((name, form_field(value)))
    return parts


def _append_part(parts: list[MultipartPart], name: str, value: object) -> None:
    if value is None:
        return
    if is_file_input(value):
        parts.append((name, read_file_input(value)))
    elif isinstance(value, (list, tuple)):
        for item in cast("list[object] | tuple[object, ...]", value):
            _append_part(parts, name, item)
    elif isinstance(value, str):
        parts.append((name, value))
    elif isinstance(value, Mapping):
        parts.append((name, json_text(cast("object", value))))
    else:
        parts.append((name, form_field(value)))


def every_field_form(body: Mapping[str, object]) -> list[MultipartPart]:
    """The form of a request with a file in its fields, every field forwarded in order.

    A file is a file part and a string a text part; a list gives one part per item under the plain field name; an
    object is sent as its JSON text; any other scalar goes through :func:`form_field`. ``None`` is left out. This is
    the form of a video generation and of ``inference.run``.
    """
    parts: list[MultipartPart] = []
    for name, value in body.items():
        _append_part(parts, name, value)
    return parts


def fixed_form(fields: Sequence[tuple[str, object]], *, file_fields: Collection[str] = ()) -> list[MultipartPart]:
    """A form with a fixed list of fields, in the given order.

    A field named in ``file_fields`` is read with :func:`read_file_input` into a file part. Any other field is text: a
    list or a tuple gives one part per item, an object its JSON text, a scalar goes through :func:`form_field`.
    ``None`` is left out, also as a list item.
    """
    parts: list[MultipartPart] = []
    for name, value in fields:
        if value is None:
            continue
        if name in file_fields:
            parts.append((name, read_file_input(value)))
        elif isinstance(value, (list, tuple)):
            for item in cast("list[object] | tuple[object, ...]", value):
                if item is not None:
                    text = json_text(cast("object", item)) if isinstance(item, Mapping) else form_field(item)
                    parts.append((name, text))
        elif isinstance(value, Mapping):
            parts.append((name, json_text(cast("object", value))))
        else:
            parts.append((name, form_field(value)))
    return parts
