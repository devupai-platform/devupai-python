"""Runs every script in ``examples/`` against a recording mock of the API.

Each script runs as ``__main__`` with ``runpy.run_path``. ``devupai.DevupAI`` and ``devupai.AsyncDevupAI`` are
replaced by subclasses whose default ``http_client`` answers through ``httpx.MockTransport`` (and whose default
``base_url`` is a test origin), so the scripts run unchanged and nothing reaches the network. Every request is
recorded before it is answered.

The test also checks that every script has a case here, that a script which sends a model id stops before any
request when ``DEVUP_MODEL`` is not set, and that the text of ``examples/`` follows the public word list and holds no
model id.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import hmac
import json
import re
import runpy
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from helpers.mock_transport import MODEL_A, REQ_ID, TEST_KEY, RecordedRequest
from public_words import line_hits

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
BASE = "https://api.example.test/v1"
ORIGIN = "https://api.example.test"

#: A synthetic webhook signing secret (never a real one).
WEBHOOK_SECRET = "fixture-webhook-secret-0000"
#: Synthetic file contents.
IMAGE_BYTES = b"\x89PNG\r\n\x1a\nfixture-image-bytes"
UPLOAD_BYTES = b"RIFF0000WAVEfixture-upload"

SPEECH_URL = f"{ORIGIN}/v1/files/speech-fixture"
IMAGE_URL = f"{ORIGIN}/v1/files/image-fixture"

CHAT_TEXT = "Hello from the mock."
STREAM_PIECES = ["Hello", " from", " the", " stream."]
RESPONSES_TEXT = "Hello from the Responses mock."
TRANSCRIPT_TEXT = "Fixture transcript text."
BALANCE = 1234.5
MODEL_B = "fixture/model-b"

_BILLING: dict[str, object] = {"cost_dzd": 1.5, "balance_dzd": 98.5}


def _model(model_id: str, model_type: str) -> dict[str, object]:
    return {
        "id": model_id,
        "object": "model",
        "name": f"Fixture {model_id}",
        "created": 1,
        "owned_by": "fixture",
        "type": model_type,
        "context_window": None,
        "is_replaced": False,
        "pricing": {"unit": "1M tokens", "currency": "DZD", "input": 1.0},
        "supports": {
            "tools": False,
            "vision": False,
            "reasoning": False,
            "structured_output": False,
            "audio_input": False,
            "video_input": False,
            "reasoning_can_be_disabled": False,
            "pdf_input": False,
        },
    }


# --- the mock API ----------------------------------------------------------------------------------------------


def _json(body: object, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "application/json", "x-request-id": REQ_ID},
        content=json.dumps(body).encode("utf-8"),
    )


def _sse(frames: list[object]) -> httpx.Response:
    body = b"".join(b"data: " + json.dumps(frame).encode("utf-8") + b"\n\n" for frame in frames) + b"data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream", "x-request-id": REQ_ID}, content=body)


def _chunk(content: str | None, finish_reason: str | None = None) -> dict[str, object]:
    delta: dict[str, object] = {} if content is None else {"content": content}
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL_A,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class ExampleAPI:
    """A recording stand-in for the API, answering by method and path."""

    def __init__(self) -> None:
        self.calls: list[RecordedRequest] = []
        self.http_clients: list[httpx.Client | httpx.AsyncClient] = []
        self.image_data: list[dict[str, object]] = [{"url": IMAGE_URL}]

    def handle(self, request: httpx.Request) -> httpx.Response:
        content = bytes(request.content)
        self.calls.append(
            RecordedRequest(
                method=request.method,
                url=str(request.url),
                headers={key.lower(): value for key, value in request.headers.items()},
                content=content,
            )
        )
        key = (request.method, request.url.path)
        if key == ("POST", "/v1/chat/completions"):
            if json.loads(content).get("stream") is True:
                frames: list[object] = [{**_chunk(None), "choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
                frames.extend(_chunk(piece) for piece in STREAM_PIECES)
                frames.append(_chunk(None, "stop"))
                return _sse(frames)
            return _json(
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": MODEL_A,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": CHAT_TEXT}, "finish_reason": "stop"}
                    ],
                    "_devup": _BILLING,
                }
            )
        if key == ("POST", "/v1/responses"):
            return _json(
                {
                    "id": "resp_fixture",
                    "object": "response",
                    "created_at": 1,
                    "model": MODEL_A,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": RESPONSES_TEXT}],
                        }
                    ],
                }
            )
        if key == ("GET", "/v1/models"):
            return _json(
                {"object": "list", "data": [_model(MODEL_A, "text-generation"), _model(MODEL_B, "text-to-image")]}
            )
        if key == ("POST", "/v1/images/generations"):
            return _json({"data": self.image_data, "_devup": _BILLING})
        if key == ("POST", "/v1/embeddings"):
            return _json(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 0.75]},
                        {"object": "embedding", "index": 1, "embedding": [0.125, 0.5, -0.25]},
                    ],
                    "model": MODEL_A,
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            )
        if key == ("POST", "/v1/audio/speech"):
            return _json({"url": SPEECH_URL, "_devup": _BILLING})
        if key == ("POST", "/v1/audio/transcriptions"):
            return _json({"text": TRANSCRIPT_TEXT, "_devup": _BILLING})
        if key == ("GET", "/v1/user/balance"):
            return _json({"balance_dzd": BALANCE})
        return _json({"error": {"message": "Not found", "type": "devup_error", "code": "not_found"}}, status=404)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ExampleAPI]:
    """Routes ``devupai.DevupAI`` / ``devupai.AsyncDevupAI`` to the mock, sets the environment and the working
    folder, and closes the HTTP clients it made."""
    import devupai

    mock = ExampleAPI()
    sync_base = devupai.DevupAI
    async_base = devupai.AsyncDevupAI

    class MockedDevupAI(sync_base):  # type: ignore[valid-type,misc]
        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("base_url", BASE)
            if "http_client" not in kwargs:
                http_client = httpx.Client(transport=httpx.MockTransport(mock.handle))
                mock.http_clients.append(http_client)
                kwargs["http_client"] = http_client
            super().__init__(**kwargs)

    class MockedAsyncDevupAI(async_base):  # type: ignore[valid-type,misc]
        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("base_url", BASE)
            if "http_client" not in kwargs:
                http_client = httpx.AsyncClient(transport=httpx.MockTransport(mock.handle))
                mock.http_clients.append(http_client)
                kwargs["http_client"] = http_client
            super().__init__(**kwargs)

    monkeypatch.setattr(devupai, "DevupAI", MockedDevupAI)
    monkeypatch.setattr(devupai, "AsyncDevupAI", MockedAsyncDevupAI)
    monkeypatch.setenv("DEVUP_API_KEY", TEST_KEY)
    monkeypatch.delenv("DEVUPAI_API_KEY", raising=False)
    monkeypatch.setenv("DEVUP_MODEL", MODEL_A)
    monkeypatch.setenv("DEVUP_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["example"])
    yield mock
    for http_client in mock.http_clients:
        if isinstance(http_client, httpx.AsyncClient):
            asyncio.run(http_client.aclose())
        else:
            http_client.close()


def run_example(name: str, monkeypatch: pytest.MonkeyPatch, *args: str) -> dict[str, Any]:
    """Runs ``examples/<name>.py`` as ``__main__`` with ``args`` as its command-line arguments."""
    path = EXAMPLES / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    return runpy.run_path(str(path), run_name="__main__")


def _paths(api: ExampleAPI) -> list[tuple[str, str]]:
    return [(call.method, httpx.URL(call.url).path) for call in api.calls]


# --- one case per script ---------------------------------------------------------------------------------------


class RecordedOutput:
    """``capsys`` that also keeps everything a script printed, so the runner can check it after the case."""

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._capsys = capsys
        self.text = ""

    def readouterr(self) -> pytest.CaptureResult[str]:
        captured = self._capsys.readouterr()
        self.text += captured.out + captured.err
        return captured


@dataclass(frozen=True)
class Ctx:
    api: ExampleAPI
    monkeypatch: pytest.MonkeyPatch
    capsys: RecordedOutput
    tmp_path: Path


def case_chat(ctx: Ctx) -> None:
    run_example("chat", ctx.monkeypatch)
    assert _paths(ctx.api) == [("POST", "/v1/chat/completions")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert body["messages"] and body.get("stream") in (None, False)
    assert CHAT_TEXT in ctx.capsys.readouterr().out


def case_chat_stream(ctx: Ctx) -> None:
    run_example("chat_stream", ctx.monkeypatch)
    assert _paths(ctx.api) == [("POST", "/v1/chat/completions")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert body["stream"] is True
    assert "".join(STREAM_PIECES) in ctx.capsys.readouterr().out


def case_responses(ctx: Ctx) -> None:
    run_example("responses", ctx.monkeypatch)
    assert _paths(ctx.api) == [("POST", "/v1/responses")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert body["input"]
    assert RESPONSES_TEXT in ctx.capsys.readouterr().out


def case_models_list(ctx: Ctx) -> None:
    ctx.monkeypatch.delenv("DEVUP_MODEL", raising=False)
    run_example("models_list", ctx.monkeypatch)
    assert _paths(ctx.api) == [("GET", "/v1/models")]
    out = ctx.capsys.readouterr().out
    assert MODEL_A in out and MODEL_B in out
    assert "text-generation" in out and "text-to-image" in out
    assert "DEVUP_MODEL" in out

    # With a type argument, only the models of that type are listed.
    run_example("models_list", ctx.monkeypatch, "text-to-image")
    assert _paths(ctx.api) == [("GET", "/v1/models")] * 2
    out = ctx.capsys.readouterr().out
    assert MODEL_B in out and MODEL_A not in out

    # A type that no model has: the script names the types in the list and ends normally (exit status 0, so no
    # SystemExit reaches this test).
    run_example("models_list", ctx.monkeypatch, "fixture-unknown-type")
    assert _paths(ctx.api) == [("GET", "/v1/models")] * 3
    captured = ctx.capsys.readouterr()
    assert (
        captured.out == "No model of type 'fixture-unknown-type'. Types in the list: text-generation, text-to-image\n"
    )
    assert captured.err == ""


def case_images(ctx: Ctx) -> None:
    run_example("images", ctx.monkeypatch)
    assert _paths(ctx.api) == [("POST", "/v1/images/generations")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert body["prompt"]
    assert IMAGE_URL in ctx.capsys.readouterr().out

    # An image returned as base64 is saved in the current folder.
    ctx.api.image_data = [{"b64_json": base64.b64encode(IMAGE_BYTES).decode("ascii")}]
    run_example("images", ctx.monkeypatch)
    saved = [path for path in ctx.tmp_path.iterdir() if path.name.startswith("image-")]
    assert len(saved) == 1
    assert saved[0].read_bytes() == IMAGE_BYTES
    assert saved[0].name in ctx.capsys.readouterr().out


def case_embeddings(ctx: Ctx) -> None:
    run_example("embeddings", ctx.monkeypatch)
    assert _paths(ctx.api) == [("POST", "/v1/embeddings")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert isinstance(body["input"], list) and len(body["input"]) == 2
    out = ctx.capsys.readouterr().out
    assert "2 embeddings" in out
    assert "3 dimensions" in out
    assert "0.25" in out


def case_speech(ctx: Ctx) -> None:
    run_example("speech", ctx.monkeypatch)
    # One request: the script prints the audio link the API returns and writes no file.
    assert _paths(ctx.api) == [("POST", "/v1/audio/speech")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert body["input"]
    assert list(ctx.tmp_path.iterdir()) == []
    out = ctx.capsys.readouterr().out
    assert SPEECH_URL in out
    assert "DZD" not in out and "cost" not in out.lower()


def case_transcription(ctx: Ctx) -> None:
    audio = ctx.tmp_path / "clip.wav"
    audio.write_bytes(UPLOAD_BYTES)
    run_example("transcription", ctx.monkeypatch, str(audio))
    assert _paths(ctx.api) == [("POST", "/v1/audio/transcriptions")]
    call = ctx.api.calls[0]
    assert call.headers["content-type"].startswith("multipart/form-data")
    assert b'name="model"\r\n\r\n' + MODEL_A.encode() in call.content
    assert b'filename="clip.wav"' in call.content
    assert UPLOAD_BYTES in call.content
    assert TRANSCRIPT_TEXT in ctx.capsys.readouterr().out


def case_balance(ctx: Ctx) -> None:
    ctx.monkeypatch.delenv("DEVUP_MODEL", raising=False)
    run_example("balance", ctx.monkeypatch)
    assert _paths(ctx.api) == [("GET", "/v1/user/balance")]
    assert str(BALANCE) in ctx.capsys.readouterr().out


def case_webhooks(ctx: Ctx) -> None:
    ctx.monkeypatch.delenv("DEVUP_MODEL", raising=False)
    run_example("webhooks", ctx.monkeypatch)
    assert ctx.api.calls == []
    captured = ctx.capsys.readouterr()
    assert "whd_" in captured.out
    assert WEBHOOK_SECRET not in captured.out + captured.err


def case_async_chat(ctx: Ctx) -> None:
    run_example("async_chat", ctx.monkeypatch)
    assert _paths(ctx.api) == [("POST", "/v1/chat/completions")]
    body = ctx.api.calls[0].json()
    assert body["model"] == MODEL_A
    assert body["stream"] is True
    assert "".join(STREAM_PIECES) in ctx.capsys.readouterr().out


CASES: dict[str, Callable[[Ctx], None]] = {
    "chat": case_chat,
    "chat_stream": case_chat_stream,
    "responses": case_responses,
    "models_list": case_models_list,
    "images": case_images,
    "embeddings": case_embeddings,
    "speech": case_speech,
    "transcription": case_transcription,
    "balance": case_balance,
    "webhooks": case_webhooks,
    "async_chat": case_async_chat,
}

#: The scripts that send a model id: they read it from ``DEVUP_MODEL``.
MODEL_SCRIPTS = ["chat", "chat_stream", "responses", "images", "embeddings", "speech", "transcription", "async_chat"]
#: The scripts that send no model id.
NO_MODEL_SCRIPTS = ["models_list", "balance", "webhooks"]


def _script_names() -> list[str]:
    return sorted(path.stem for path in EXAMPLES.glob("*.py"))


def test_every_example_has_a_case() -> None:
    assert _script_names() == sorted(CASES)
    assert sorted(MODEL_SCRIPTS + NO_MODEL_SCRIPTS) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_example_runs_against_the_mock(
    name: str,
    api: ExampleAPI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = RecordedOutput(capsys)
    CASES[name](Ctx(api=api, monkeypatch=monkeypatch, capsys=output, tmp_path=tmp_path))
    output.readouterr()
    assert output.text, f"{name}.py printed nothing"
    # No script prints the API key (or the Authorization header that carries it).
    assert TEST_KEY not in output.text
    assert all(call.headers.get("authorization") == f"Bearer {TEST_KEY}" for call in api.calls)


# --- DEVUP_MODEL is required where a model id is sent ------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
@pytest.mark.parametrize("name", MODEL_SCRIPTS)
def test_a_model_script_stops_without_devup_model(
    name: str, value: str | None, api: ExampleAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if value is None:
        monkeypatch.delenv("DEVUP_MODEL", raising=False)
    else:
        monkeypatch.setenv("DEVUP_MODEL", value)
    args: list[str] = []
    if name == "transcription":
        audio = tmp_path / "clip.wav"
        audio.write_bytes(UPLOAD_BYTES)
        args.append(str(audio))
    with pytest.raises(SystemExit) as excinfo:
        run_example(name, monkeypatch, *args)
    message = excinfo.value.code
    assert isinstance(message, str), "the script must exit with a message (a non-zero status)"
    assert "DEVUP_MODEL" in message
    assert "examples/models_list.py" in message
    assert TEST_KEY not in message
    assert api.calls == []


def test_transcription_stops_without_a_path(api: ExampleAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_example("transcription", monkeypatch)
    message = excinfo.value.code
    assert isinstance(message, str)
    assert "transcription.py" in message
    assert api.calls == []


def test_webhooks_stops_without_the_secret(api: ExampleAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEVUP_WEBHOOK_SECRET", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        run_example("webhooks", monkeypatch)
    message = excinfo.value.code
    assert isinstance(message, str)
    assert "DEVUP_WEBHOOK_SECRET" in message


def _sign(body: bytes, secret: str, timestamp: int) -> str:
    digest = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def test_the_webhook_handler_accepts_a_signature_computed_here(api: ExampleAPI) -> None:
    from devupai import DevupWebhookVerificationError, types

    module = runpy.run_path(str(EXAMPLES / "webhooks.py"), run_name="examples_webhooks")
    handle_delivery = module["handle_delivery"]
    body = json.dumps(
        {
            "id": "whd_00000000-0000-0000-0000-000000000001",
            "status": "succeeded",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "results": [{"text": "fixture"}],
        }
    ).encode("utf-8")
    header = _sign(body, WEBHOOK_SECRET, int(time.time()))

    event = handle_delivery(body, {"X-DevUp-Signature": header}, WEBHOOK_SECRET)
    assert isinstance(event, types.WebhookSuccessPayload)
    assert event.id == "whd_00000000-0000-0000-0000-000000000001"

    with pytest.raises(DevupWebhookVerificationError):
        handle_delivery(body, {"X-DevUp-Signature": _sign(body, "another-secret", int(time.time()))}, WEBHOOK_SECRET)
    with pytest.raises(DevupWebhookVerificationError):
        handle_delivery(body + b" ", {"X-DevUp-Signature": header}, WEBHOOK_SECRET)
    with pytest.raises(DevupWebhookVerificationError):
        handle_delivery(body, {}, WEBHOOK_SECRET)
    assert api.calls == []


# --- text of examples/ ---------------------------------------------------------------------------------------------


def _example_files() -> list[Path]:
    return sorted(path for path in EXAMPLES.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


def test_examples_folder_is_not_empty() -> None:
    names = {path.name for path in _example_files()}
    assert {f"{name}.py" for name in CASES} <= names


def test_no_listed_word_in_examples() -> None:
    hits: list[str] = []
    files = _example_files()
    assert files
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(re.split(r"\r?\n", path.read_text(encoding="utf-8")), start=1):
            found = line_hits(name, line)
            if found:
                hits.append(f"{name}:{number} [{', '.join(dict.fromkeys(found))}]")
    assert "\n".join(hits) == ""


#: An ``owner/name`` token, the form of a model id.
_SLASHED = re.compile(r"(?<![\w./:-])[A-Za-z0-9][\w.-]*/[\w.:/-]*[\w]")
_MIME_PREFIXES = ("application/", "audio/", "image/", "text/", "video/", "multipart/")


def _looks_like_a_model_id(token: str) -> bool:
    """True for an ``owner/name`` token that is not a repository path or a media type."""
    if token.startswith(_MIME_PREFIXES):
        return False
    return not (ROOT / token).exists()


def test_the_slashed_token_check_finds_model_ids() -> None:
    assert _looks_like_a_model_id("fixture/model-a")
    assert _looks_like_a_model_id("some-org/Some-Model-7B")
    assert _SLASHED.findall("examples python") == []  # no slash: never matched by the pattern
    assert not _looks_like_a_model_id("tests/test_examples.py")
    assert not _looks_like_a_model_id("audio/mpeg")
    assert _SLASHED.findall("see https://example.test/a/b and org/model-1") == ["org/model-1"]


def test_no_model_id_in_examples() -> None:
    bad: list[str] = []
    for path in _example_files():
        name = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for token in _SLASHED.findall(text):
            if _looks_like_a_model_id(token):
                bad.append(f"{name}: {token}")
        if path.suffix != ".py":
            continue
        # A model value in code is never a literal: it comes from DEVUP_MODEL.
        for node in ast.walk(ast.parse(text)):
            values: list[ast.expr] = []
            if isinstance(node, ast.keyword) and node.arg == "model":
                values.append(node.value)
            elif isinstance(node, ast.Dict):
                values.extend(
                    value
                    for key, value in zip(node.keys, node.values, strict=True)
                    if isinstance(key, ast.Constant) and key.value == "model"
                )
            bad.extend(
                f"{name}:{value.lineno} literal model value"
                for value in values
                if isinstance(value, ast.Constant | ast.JoinedStr)
            )
    assert bad == []


def test_model_scripts_read_devup_model() -> None:
    for name in MODEL_SCRIPTS:
        assert "DEVUP_MODEL" in (EXAMPLES / f"{name}.py").read_text(encoding="utf-8"), name


def test_every_script_is_runnable_and_documented() -> None:
    for name in CASES:
        source = (EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert ast.get_docstring(tree), f"{name}.py has no module docstring"
        assert f"python examples/{name}.py" in (ast.get_docstring(tree) or ""), f"{name}.py docstring: how to run"
        assert 'if __name__ == "__main__":' in source, name
        assert "api_key" not in source, f"{name}.py must let the client read DEVUP_API_KEY"
