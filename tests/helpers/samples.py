"""One sample call per public API method of the client (contract §6).

Each entry gives minimal arguments with synthetic ids, the request the SDK is expected to send (HTTP method and the
URL path after the host, exactly as sent), and a minimal JSON body that is valid for the return model. Methods that
raise ``NotImplementedError`` in this build are marked ``not_implemented=True``; their ``path`` is still the path the
method stands for, so the endpoint tests can check it against the spec.

This module never imports ``devupai`` at module level: the tests that use it are collected even when the package is
incomplete.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio

from helpers.mock_transport import MODEL_A

#: Scheme and host of ``helpers.mock_transport.BASE``; a full request URL is ``ORIGIN + Sample.path``.
ORIGIN = "https://api.example.test"


@dataclass(frozen=True)
class Sample:
    """One sample call of an API method and the request it sends."""

    args: tuple[object, ...]
    kwargs: dict[str, object]
    http_method: str
    path: str  # as sent: the URL path after the host, e.g. "/v1/models/fixture%2Fmodel-a", "/health"
    response: object  # a minimal JSON body valid for the return model
    not_implemented: bool = False


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
        args=("https://api.example.test/v1/images/proxy?url=https%3A%2F%2Fcdn.example.test%2Ffixture.png",),
        kwargs={},
        http_method="GET",
        path="/v1/images/proxy",
        response=None,
        not_implemented=True,
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
        kwargs={"model": MODEL_A, "file": b"\x00\x01"},
        http_method="POST",
        path="/v1/audio/transcriptions",
        response={"text": "Hello"},
        not_implemented=True,
    ),
    "video.generations.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "prompt": "A slow pan"},
        http_method="POST",
        path="/v1/video/generations",
        response={"data": [{"url": "https://cdn.example.test/fixture.mp4"}], "_devup": _BILLING},
    ),
    "video.edits.create": Sample(
        args=(),
        kwargs={"model": MODEL_A, "video": b"\x00\x01"},
        http_method="POST",
        path="/v1/video/generations",
        response={"data": [{"url": "https://cdn.example.test/fixture.mp4"}], "_devup": _BILLING},
        not_implemented=True,
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
        response=None,
        not_implemented=True,
    ),
    "health.check": Sample(
        args=(),
        kwargs={},
        http_method="GET",
        path="/health",
        response={"status": "ok"},
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

#: The methods that always raise ``NotImplementedError`` in this build.
NOT_IMPLEMENTED: frozenset[str] = frozenset(name for name, sample in SAMPLES.items() if sample.not_implemented)

#: The methods that send a request in this build.
IMPLEMENTED: tuple[str, ...] = tuple(name for name, sample in SAMPLES.items() if not sample.not_implemented)

#: The return model of each implemented method, by its name in ``devupai.types`` (``None``: plain parsed JSON).
RETURNS: dict[str, str | None] = {
    "chat.completions.create": "ChatCompletion",
    "responses.create": "ResponseObject",
    "models.list": "ModelListResponse",
    "models.retrieve": "ModelObject",
    "images.generate": "ImageGenerateResponse",
    "images.edit": "ImageGenerateResponse",
    "embeddings.create": "EmbeddingCreateResponse",
    "rerank.create": "RerankCreateResponse",
    "audio.speech.create": "SpeechResponse",
    "video.generations.create": "VideoGenerationResponse",
    "inference.run": None,
    "balance.retrieve": "BalanceResponse",
    "me.retrieve": "AccountProfile",
    "health.check": "HealthResponse",
}


def expected_body(name: str) -> dict[str, object] | None:
    """The JSON body the sample call of ``name`` sends (``None`` for a GET)."""
    sample = SAMPLES[name]
    if sample.http_method == "GET":
        return None
    if name == "inference.run":
        payload = sample.args[1]
        assert isinstance(payload, dict)
        return dict(payload)
    return dict(sample.kwargs)


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
    """Calls the sample of ``name`` on ``client`` with ``overrides`` added to its keyword arguments."""
    sample = SAMPLES[name]
    kwargs: dict[str, Any] = {**sample.kwargs, **overrides}
    return invoke(method_of(client, name, with_response=with_response), *sample.args, **kwargs)
