"""One sample call per public API method of the client (contract §6).

Each entry gives minimal arguments with synthetic ids, the request the SDK is expected to send (HTTP method, the URL
path after the host exactly as sent, the query of an absolute URL, the multipart parts of a form, whether the key is
sent), and a minimal answer that is valid for the return type: a JSON body, or the bytes of a file for a method that
returns a ``BinaryResponse``. Every method sends a request in this build; none is a placeholder any more (P3
contract §6).

This module never imports ``devupai`` at module level: the tests that use it are collected even when the package is
incomplete.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import anyio
import httpx

from helpers.mock_transport import MODEL_A, REQ_ID, RecordedRequest, json_response

#: Scheme and host of ``helpers.mock_transport.BASE``; a full request URL is ``ORIGIN + Sample.path``.
ORIGIN = "https://api.example.test"

#: The bytes of the file a binary sample answers with, and their content type.
BINARY_CONTENT = b"\x89PNG\r\n\x1a\nfixture-bytes"
BINARY_CONTENT_TYPE = "image/png"

#: The content type of a file part sent without one.
OCTET_STREAM = "application/octet-stream"


@dataclass(frozen=True)
class Part:
    """One part of a ``multipart/form-data`` body: a text part has no filename and no content type."""

    name: str
    filename: str | None
    content_type: str | None
    content: bytes


def text_part(name: str, value: str) -> Part:
    """A text part."""
    return Part(name, None, None, value.encode("utf-8"))


def file_part(name: str, content: bytes, filename: str = "upload", content_type: str = OCTET_STREAM) -> Part:
    """A file part (default filename ``upload``, decision Q19)."""
    return Part(name, filename, content_type, content)


@dataclass(frozen=True)
class Sample:
    """One sample call of an API method and the request it sends."""

    args: tuple[object, ...]
    kwargs: dict[str, object]
    http_method: str
    path: str  # as sent: the URL path after the host, e.g. "/v1/models/fixture%2Fmodel-a", "/health"
    response: object  # a minimal JSON body valid for the return model, or the file bytes when ``binary``
    query: str = ""  # the query sent, without "?" (an absolute URL given whole)
    binary: bool = False  # the method returns a ``BinaryResponse`` (``kind="raw"``)
    form: tuple[Part, ...] | None = None  # the multipart parts sent, in order (``None``: JSON or no body)
    auth: bool = True  # the key is sent


_BILLING: dict[str, object] = {"cost_dzd": 1.5, "balance_dzd": 98.5}

MODEL_OBJECT: dict[str, object] = {
    "id": MODEL_A,
    "object": "model",
    "name": "Fixture Model A",
    "created": 1,
    "owned_by": "fixture",
    "type": "chat",
    "context_window": None,
    "is_replaced": False,
    "pricing": {"unit": "1M tokens", "currency": "DZD", "input": 1.0},
    "supports": {"tools": True},
}

IMAGE_RESPONSE: dict[str, object] = {
    "data": [{"url": "https://cdn.example.test/fixture.png"}],
    "_devup": _BILLING,
}

VIDEO_RESPONSE: dict[str, object] = {"data": [{"url": "https://cdn.example.test/fixture.mp4"}], "_devup": _BILLING}

SAMPLES: dict[str, Sample] = {
    "chat.completions.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "messages": [{"role": "user", "content": "Hello"}]},
        http_method="POST",
        path="/v1/chat/completions",
        response={
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL_A,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hello back"}, "finish_reason": "stop"}
            ],
            "_devup": _BILLING,
        },
    ),
    "responses.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "input": "Hello"},
        http_method="POST",
        path="/v1/responses",
        response={
            "id": "resp_fixture",
            "object": "response",
            "created_at": 1,
            "model": MODEL_A,
            "status": "completed",
            "output": [],
        },
    ),
    "models.list": Sample(
        args=(),
        kwargs={},
        http_method="GET",
        path="/v1/models",
        response={"object": "list", "data": [MODEL_OBJECT]},
    ),
    "models.retrieve": Sample(
        args=(MODEL_A,),
        kwargs={},
        http_method="GET",
        path="/v1/models/fixture%2Fmodel-a",
        response=MODEL_OBJECT,
    ),
    "images.generate": Sample(
        args=(),
        kwargs={"model": MODEL_A, "prompt": "A red square"},
        http_method="POST",
        path="/v1/images/generations",
        response=IMAGE_RESPONSE,
    ),
    "images.edit": Sample(
        args=(),
        kwargs={"model": MODEL_A, "prompt": "Make it blue", "image": "https://cdn.example.test/input.png"},
        http_method="POST",
        path="/v1/images/edits",
        response=IMAGE_RESPONSE,
    ),
    "images.download": Sample(
        # An absolute URL with the same origin as the base URL (Node's rule): sent byte-exact, without the key.
        args=("https://api.example.test/v1/images/proxy?url=https%3A%2F%2Fcdn.example.test%2Ffixture.png",),
        kwargs={},
        http_method="GET",
        path="/v1/images/proxy",
        query="url=https%3A%2F%2Fcdn.example.test%2Ffixture.png",
        response=BINARY_CONTENT,
        binary=True,
        auth=False,
    ),
    "embeddings.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "input": "Hello"},
        http_method="POST",
        path="/v1/embeddings",
        response={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
            "model": MODEL_A,
        },
    ),
    "rerank.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "query": "fruit", "documents": ["apple", "stone"]},
        http_method="POST",
        path="/v1/rerank",
        response={
            "object": "list",
            "model": MODEL_A,
            "results": [{"index": 0, "relevance_score": 0.5}],
            "_devup": _BILLING,
        },
    ),
    "audio.speech.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "input": "Hello"},
        http_method="POST",
        path="/v1/audio/speech",
        response={"url": "https://cdn.example.test/fixture.mp3", "_devup": _BILLING},
    ),
    "audio.transcriptions.create": Sample(
        args=(),
        # Given model first: the parts still follow Node's order, file first.
        kwargs={"model": MODEL_A, "file": b"\x00\x01"},
        http_method="POST",
        path="/v1/audio/transcriptions",
        response={"text": "Hello", "_devup": _BILLING},
        form=(file_part("file", b"\x00\x01"), text_part("model", MODEL_A)),
    ),
    "video.generations.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "prompt": "A slow pan"},
        http_method="POST",
        path="/v1/video/generations",
        response=VIDEO_RESPONSE,
    ),
    "video.edits.create": Sample(
        args=(),
        # Given model first: the parts still follow Node's fixed list, video first.
        kwargs={"model": MODEL_A, "video": b"\x00\x01"},
        http_method="POST",
        path="/v1/video/generations",
        response=VIDEO_RESPONSE,
        form=(file_part("video", b"\x00\x01"), text_part("model", MODEL_A)),
    ),
    "inference.run": Sample(
        args=("/fixture/model-a", {"inputs": "hello"}),
        kwargs={},
        http_method="POST",
        path="/v1/inference/fixture%2Fmodel-a",
        response={"ok": True},
    ),
    "balance.retrieve": Sample(
        args=(),
        kwargs={},
        http_method="GET",
        path="/v1/user/balance",
        response={"balance_dzd": 10.5},
    ),
    "me.retrieve": Sample(
        args=(),
        kwargs={},
        http_method="GET",
        path="/v1/me",
        response={
            "object": "account_profile",
            "display_name": "Fixture Name",
            "initial": "F",
            "email_hint": None,
            "avatar": None,
        },
    ),
    "me.avatar": Sample(
        args=(),
        kwargs={},
        http_method="GET",
        path="/v1/me/avatar",
        response=BINARY_CONTENT,
        binary=True,
    ),
    "health.check": Sample(
        args=(),
        kwargs={},
        http_method="GET",
        path="/health",
        response={"status": "ok"},
        auth=False,
    ),
}

#: The 11 billed POST methods of contract §3.3.
BILLED_POST_METHODS: frozenset[str] = frozenset(
    {
        "chat.completions.create",
        "responses.create",
        "images.generate",
        "images.edit",
        "embeddings.create",
        "rerank.create",
        "audio.speech.create",
        "audio.transcriptions.create",
        "video.generations.create",
        "video.edits.create",
        "inference.run",
    }
)

#: The 7 GET methods of contract §3.3.
GET_METHODS: frozenset[str] = frozenset(
    {
        "models.list",
        "models.retrieve",
        "images.download",
        "balance.retrieve",
        "me.retrieve",
        "me.avatar",
        "health.check",
    }
)

#: The methods that send a request in this build: all of them (P3 removed the placeholders).
IMPLEMENTED: tuple[str, ...] = tuple(SAMPLES)

#: The return type of each method, by its name in ``devupai.types`` or ``devupai`` (``None``: plain parsed JSON).
#: ``BinaryResponse`` is ``AsyncBinaryResponse`` on the async client (see :func:`return_type`).
RETURNS: dict[str, str | None] = {
    "chat.completions.create": "ChatCompletion",
    "responses.create": "ResponseObject",
    "models.list": "ModelListResponse",
    "models.retrieve": "ModelObject",
    "images.generate": "ImageGenerateResponse",
    "images.edit": "ImageGenerateResponse",
    "images.download": "BinaryResponse",
    "embeddings.create": "EmbeddingCreateResponse",
    "rerank.create": "RerankCreateResponse",
    "audio.speech.create": "SpeechResponse",
    "audio.transcriptions.create": "TranscriptionResponse",
    "video.generations.create": "VideoGenerationResponse",
    "video.edits.create": "VideoGenerationResponse",
    "inference.run": None,
    "balance.retrieve": "BalanceResponse",
    "me.retrieve": "AccountProfile",
    "me.avatar": "BinaryResponse",
    "health.check": "HealthResponse",
}

#: Return types that have an ``Async`` twin on the async client.
ASYNC_TWIN_TYPES: frozenset[str] = frozenset({"Stream", "BinaryResponse"})


def return_type(type_name: str, client_kind: str) -> type[Any]:
    """The class named ``type_name`` (``Async`` + name for a stream or binary result on the async client)."""
    if client_kind == "async" and type_name in ASYNC_TWIN_TYPES:
        type_name = "Async" + type_name
    types = importlib.import_module("devupai.types")
    if hasattr(types, type_name):
        found: type[Any] = getattr(types, type_name)
        return found
    devupai = importlib.import_module("devupai")
    found = getattr(devupai, type_name)
    return found


def sample_url(name: str) -> str:
    """The full URL the sample call of ``name`` sends."""
    sample = SAMPLES[name]
    return ORIGIN + sample.path + (f"?{sample.query}" if sample.query else "")


def sample_accept(name: str) -> str:
    """The ``Accept`` header of the sample call: ``*/*`` for a binary result (``kind="raw"``)."""
    return "*/*" if SAMPLES[name].binary else "application/json"


def sample_kind(name: str) -> str:
    """The ``kind`` the sample call passes to the client's request entry."""
    return "raw" if SAMPLES[name].binary else "json"


def binary_response(
    content: bytes = BINARY_CONTENT, status: int = 200, headers: Mapping[str, str] | None = None
) -> httpx.Response:
    """A file answer: ``content-type: image/png`` and ``x-request-id: REQ_ID`` unless ``headers`` sets them."""
    merged = {"content-type": BINARY_CONTENT_TYPE, "x-request-id": REQ_ID}
    for key, value in (headers or {}).items():
        merged[key.lower()] = value
    return httpx.Response(status, headers=merged, content=content)


def sample_response(name: str, headers: Mapping[str, str] | None = None) -> httpx.Response:
    """A 200 answer valid for the sample of ``name``: its JSON body, or its file bytes."""
    sample = SAMPLES[name]
    if sample.binary:
        assert isinstance(sample.response, bytes)
        return binary_response(sample.response, headers=headers)
    return json_response(sample.response, headers=headers)


def expected_body(name: str) -> dict[str, object] | None:
    """The JSON body the sample call of ``name`` sends (``None`` for a GET or a multipart form)."""
    sample = SAMPLES[name]
    if sample.http_method == "GET" or sample.form is not None:
        return None
    if name == "inference.run":
        payload = sample.args[1]
        assert isinstance(payload, dict)
        return dict(payload)
    return dict(sample.kwargs)


#: The POST methods whose sample sends a JSON body (and so take ``extra_body``).
JSON_POST_METHODS: tuple[str, ...] = tuple(name for name in SAMPLES if expected_body(name) is not None)

#: The POST methods whose sample sends a multipart form (Node's fixed field list; no ``extra_body``).
FORM_POST_METHODS: tuple[str, ...] = tuple(name for name, sample in SAMPLES.items() if sample.form is not None)

_DISPOSITION_PARAM = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def multipart_parts(request: RecordedRequest) -> list[Part]:
    """The parts of a recorded ``multipart/form-data`` request, in order.

    Asserts the framing: the ``Content-Type`` carries the boundary, every part is delimited by it, and the body ends
    with the closing delimiter.
    """
    content_type = request.headers.get("content-type", "")
    assert content_type.startswith("multipart/form-data; boundary="), content_type
    boundary = content_type.split("boundary=", 1)[1].strip('"').encode("ascii")
    delimiter = b"--" + boundary
    body = request.content
    assert body.startswith(delimiter + b"\r\n"), body[:80]
    assert body.endswith(delimiter + b"--\r\n"), body[-80:]
    sections = body.split(delimiter)
    assert sections[0] == b"" and sections[-1] == b"--\r\n"
    parts: list[Part] = []
    for section in sections[1:-1]:
        assert section.startswith(b"\r\n") and section.endswith(b"\r\n"), section[:80]
        head, separator, content = section[2:-2].partition(b"\r\n\r\n")
        assert separator, "a part has no blank line after its headers"
        headers: dict[str, str] = {}
        for line in head.decode("utf-8").split("\r\n"):
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        disposition = headers.pop("content-disposition")
        assert disposition.startswith("form-data;"), disposition
        params = dict(_DISPOSITION_PARAM.findall(disposition))
        part_type = headers.pop("content-type", None)
        assert not headers, f"unexpected part headers: {headers}"
        parts.append(Part(params["name"], params.get("filename"), part_type, content))
    return parts


def split_name(name: str) -> tuple[str, str]:
    """``"audio.speech.create"`` → ``("audio.speech", "create")``."""
    resource, _, method = name.rpartition(".")
    return resource, method


def resolve(root: object, dotted: str) -> Any:
    """``getattr`` along a dotted path: ``resolve(client, "chat.completions")``."""
    value: Any = root
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def method_of(client: object, name: str, *, with_response: bool = False) -> Any:
    """The bound method ``name`` of ``client``, or its ``with_response`` twin."""
    resource_path, method = split_name(name)
    resource = resolve(client, resource_path)
    if with_response:
        resource = resource.with_response
    return getattr(resource, method)


def invoke(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Calls ``fn`` inside one event loop; an awaitable result (the async client) is awaited there.

    Exceptions propagate unchanged, whether they are raised by the call itself or by the awaited coroutine.
    """

    async def main() -> Any:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return anyio.run(main)


def _classes(*names: str) -> tuple[type[Any], ...]:
    """The ``devupai`` classes among ``names`` that exist (an empty tuple matches nothing in ``isinstance``)."""
    devupai = importlib.import_module("devupai")
    return tuple(getattr(devupai, name) for name in names if hasattr(devupai, name))


async def drain(value: Any) -> Any:
    """Consumes a stream or a binary result in the running loop (the ``.data`` of an ``APIResponse`` too).

    Returns the list of items of a stream, the bytes of a binary result, and ``None`` for anything else.
    """
    target = value.data if isinstance(value, _classes("APIResponse")) else value
    if isinstance(target, _classes("AsyncStream")):
        return [item async for item in target]
    if isinstance(target, _classes("Stream")):
        return list(target)
    if isinstance(target, _classes("AsyncBinaryResponse")):
        return await target.read()
    if isinstance(target, _classes("BinaryResponse")):
        return target.read()
    return None


def invoke_and_read(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Like :func:`invoke`, then :func:`drain` in the same loop: returns ``(result, what was read)``."""

    async def main() -> tuple[Any, Any]:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result, await drain(result)

    return anyio.run(main)


@dataclass(frozen=True)
class Discovered:
    """One public API method found on a client by introspection (contract §6)."""

    name: str  # dotted, e.g. "chat.completions.create"
    resource: Any
    method: Any  # the bound method
    endpoint: Any  # its ``__devup_endpoint__``


def discover(client: object) -> dict[str, Discovered]:
    """Every public method carrying ``__devup_endpoint__`` on the public resources of ``client``.

    Resources are the public attributes that are ``SyncAPIResource`` / ``AsyncAPIResource`` instances, recursing
    into sub-resources. ``with_response`` twins are not walked here (the tests read them from each resource).
    """
    base = importlib.import_module("devupai.resources._base")
    resource_types = (base.SyncAPIResource, base.AsyncAPIResource)
    found: dict[str, Discovered] = {}
    seen: set[int] = set()

    def walk(obj: object, prefix: str) -> None:
        for attr in sorted(dir(obj)):
            if attr.startswith("_") or attr == "with_response":
                continue
            value = getattr(obj, attr)
            dotted = f"{prefix}{attr}"
            if isinstance(value, resource_types):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                walk(value, f"{dotted}.")
            elif prefix and callable(value) and hasattr(value, "__devup_endpoint__"):
                found[dotted] = Discovered(dotted, obj, value, value.__devup_endpoint__)

    walk(client, "")
    return found


def call_sample(client: object, name: str, *, with_response: bool = False, **overrides: Any) -> Any:
    """Calls the sample of ``name`` on ``client`` with ``overrides`` added to its keyword arguments.

    A binary result is read inside the same event loop, so its bytes are in ``result.response.content`` afterwards
    on both clients.
    """
    sample = SAMPLES[name]
    kwargs: dict[str, Any] = {**sample.kwargs, **overrides}
    result, _ = invoke_and_read(method_of(client, name, with_response=with_response), *sample.args, **kwargs)
    return result
