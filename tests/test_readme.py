"""README checks: the section order, the install line, the code examples and the error table.

Every ``python`` block must parse, every ``client.<...>(...)`` call in it must resolve to a callable on both
``DevupAI`` and ``AsyncDevupAI``, every API method must have an example, every stream and binary result is opened in
``with`` / ``async with``, and model ids are placeholders only. Nothing is sent: the clients use a mock transport.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from helpers.mock_transport import BASE, MODEL_A, TEST_KEY

import devupai
from devupai import AsyncDevupAI, DevupAI

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

#: The level-1 and level-2 headings of the README, in order (the section order of the Node.js SDK README, without its
#: Node-only sections).
EXPECTED_SECTIONS = [
    "Overview",
    "Contents",
    "Why DEVUP AI",
    "Requirements",
    "Installation",
    "Quick Start",
    "Capability Matrix",
    "Core SDK",
    "Chat Completions",
    "Streaming",
    "Reasoning Output",
    "Responses API",
    "Images",
    "Model Options",
    "Downloading Generated Files",
    "Embeddings",
    "Reranking",
    "Audio",
    "Video",
    "Native Inference",
    "Models, Balance, and Health",
    "Webhooks",
    "Configuration",
    "Request Options",
    "Request IDs, Headers, and Notices",
    "Retries",
    "Timeouts and Cancellation",
    "Error Handling",
    "Request Size Limits",
    "Type Hints",
    "Runtime and Security",
    "Models",
    "Get an API Key",
    "Resources",
    "License",
]

#: Methods that run locally and carry no endpoint, but still need an example.
LOCAL_METHODS = {("webhooks", "construct_event"), ("webhooks", "verify_signature")}

#: Results that are file answers (``BinaryResponse`` / ``AsyncBinaryResponse``) whatever their arguments.
BINARY_METHODS = {("images", "download"), ("me", "avatar")}

_FENCE = re.compile(r"^```([A-Za-z0-9_-]*)\s*$")
_HEADING = re.compile(r"^(#{1,2}) (.+?)\s*$")
_PLACEHOLDER = re.compile(r"<[^<>\s][^<>]*>")
# Looks like "organisation/model": two path segments without a scheme, a space or a file extension at the end.
_MODEL_ID_LIKE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:-]*")
_MEDIA_TYPES = {"application", "audio", "font", "image", "multipart", "text", "video"}


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _lines_outside_fences() -> Iterator[str]:
    inside = False
    for line in _text().splitlines():
        if _FENCE.match(line.strip()):
            inside = not inside
            continue
        if not inside:
            yield line


def _blocks(language: str) -> list[tuple[int, str]]:
    """``(first line number, code)`` of every fenced block of ``language``."""
    found: list[tuple[int, str]] = []
    current: list[str] | None = None
    current_language = ""
    start = 0
    for number, line in enumerate(_text().splitlines(), start=1):
        fence = _FENCE.match(line.strip())
        if fence is None:
            if current is not None:
                current.append(line)
            continue
        if current is None:
            current, current_language, start = [], fence.group(1), number + 1
        else:
            if current_language == language:
                found.append((start, "\n".join(current)))
            current = None
    assert current is None, "a code block is not closed"
    return found


def _python_trees() -> list[tuple[int, ast.Module]]:
    return [(start, ast.parse(code, filename=f"README.md:{start}")) for start, code in _blocks("python")]


def _chain(node: ast.expr) -> tuple[str, ...] | None:
    """The attribute names of ``client.a.b.c`` (``("a", "b", "c")``), or ``None`` when ``node`` is another form."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "client" and parts:
        return tuple(reversed(parts))
    return None


def _client_calls() -> list[tuple[int, ast.Call, tuple[str, ...]]]:
    calls: list[tuple[int, ast.Call, tuple[str, ...]]] = []
    for start, tree in _python_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                chain = _chain(node.func)
                if chain is not None:
                    calls.append((start + node.lineno - 1, node, chain))
    return calls


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _is_constant(node: ast.expr | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={})


@pytest.fixture
def clients() -> Iterator[tuple[DevupAI, AsyncDevupAI]]:
    sync_client = DevupAI(
        api_key=TEST_KEY, base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )
    async_client = AsyncDevupAI(
        api_key=TEST_KEY, base_url=BASE, http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    )
    yield sync_client, async_client
    sync_client.close()
    anyio.run(async_client.close)


def _api_methods(client: DevupAI) -> set[tuple[str, ...]]:
    """The attribute path of every API method of ``client`` (methods marked with ``__devup_endpoint__``)."""
    found: set[tuple[str, ...]] = set()

    def visit(resource: object, path: tuple[str, ...]) -> None:
        for name in dir(type(resource)):
            if name.startswith("_") or name == "with_response":
                continue
            if hasattr(getattr(type(resource), name), "__devup_endpoint__"):
                found.add((*path, name))
        for name, value in vars(resource).items():
            if not name.startswith("_") and type(value).__module__.startswith("devupai.resources"):
                visit(value, (*path, name))

    for name, value in vars(client).items():
        if not name.startswith("_") and type(value).__module__.startswith("devupai.resources"):
            visit(value, (name,))
    return found


# --- structure ---------------------------------------------------------------------------------------------------


def test_the_sections_follow_the_node_readme_order() -> None:
    headings = [match.group(2) for line in _lines_outside_fences() if (match := _HEADING.match(line))]
    assert headings == EXPECTED_SECTIONS


def test_the_install_line_is_plain_pip_install() -> None:
    commands = [line.strip() for _, code in _blocks("bash") for line in code.splitlines()]
    assert "pip install devupai" in commands
    assert "--pre" not in _text()


def test_the_python_floor_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.11"
    assert "Python 3.11 or later" in _text()


# --- code examples -----------------------------------------------------------------------------------------------


def test_every_python_block_parses() -> None:
    trees = _python_trees()
    assert len(trees) >= 20


def test_every_client_call_resolves_on_both_clients(clients: tuple[DevupAI, AsyncDevupAI]) -> None:
    calls = _client_calls()
    assert calls, "the README has no client calls"
    missing: list[str] = []
    for line, _, chain in calls:
        for client in clients:
            target: object = client
            try:
                for name in chain:
                    target = getattr(target, name)
            except AttributeError:
                missing.append(f"README.md:{line} {type(client).__name__}: client.{'.'.join(chain)}")
                continue
            if not callable(target):
                missing.append(f"README.md:{line} {type(client).__name__}: client.{'.'.join(chain)} is not callable")
    assert missing == []


def test_every_api_method_has_an_example(clients: tuple[DevupAI, AsyncDevupAI]) -> None:
    written = {chain for _, _, chain in _client_calls()}
    methods = _api_methods(clients[0]) | LOCAL_METHODS
    assert len(methods) >= 20
    assert sorted(".".join(path) for path in methods - written) == []
    assert any("with_response" in chain for chain in written), "no with_response example"


def _context_managed(node: ast.expr, parents: dict[ast.AST, ast.AST], kind: type[ast.With | ast.AsyncWith]) -> bool:
    item = parents.get(node)
    return isinstance(item, ast.withitem) and item.context_expr is node and isinstance(parents.get(item), kind)


def test_every_stream_and_binary_result_is_opened_in_a_with_block() -> None:
    wrong: list[str] = []
    counts = {"async stream": 0, "async binary": 0, "sync stream": 0, "sync binary": 0}
    for start, tree in _python_trees():
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _chain(node.func)
            if chain is None or "with_response" in chain:
                continue
            stream = _is_constant(_keyword(node, "stream"), True)
            binary = chain[-2:] in BINARY_METHODS or (
                chain[-2:] == ("inference", "run") and _is_constant(_keyword(node, "response_type"), "binary")
            )
            if not (stream or binary):
                continue
            what = "stream" if stream else "binary"
            awaited = parents.get(node)
            line = start + node.lineno - 1
            if isinstance(awaited, ast.Await):
                counts[f"async {what}"] += 1
                if not _context_managed(awaited, parents, ast.AsyncWith):
                    wrong.append(f"README.md:{line} async {what} without `async with await ...`")
            else:
                counts[f"sync {what}"] += 1
                if not _context_managed(node, parents, ast.With):
                    wrong.append(f"README.md:{line} {what} without `with ...`")
    assert wrong == []
    assert all(count >= 1 for count in counts.values()), counts


def _is_env_model(node: ast.expr) -> bool:
    """``os.environ["DEVUP_MODEL"]``."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and _is_constant(node.slice, "DEVUP_MODEL")
    )


def test_model_ids_are_placeholders_only() -> None:
    wrong: list[str] = []
    placeholders = 0
    for start, tree in _python_trees():
        values: list[ast.expr] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "model":
                values.append(node.value)
            elif isinstance(node, ast.Dict):
                values.extend(v for k, v in zip(node.keys, node.values, strict=True) if _is_constant(k, "model"))
            elif isinstance(node, ast.Call) and node.args:
                chain = _chain(node.func)
                if chain is not None and chain[-2:] in {("models", "retrieve"), ("inference", "run")}:
                    values.append(node.args[0])
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in re.split(r"\s+", node.value):
                    if _MODEL_ID_LIKE.fullmatch(token) and token.split("/", 1)[0].lower() not in _MEDIA_TYPES:
                        wrong.append(f"README.md:{start + node.lineno - 1} looks like a model id: {token!r}")
        for value in values:
            line = start + value.lineno - 1
            if isinstance(value, ast.Constant):
                if isinstance(value.value, str) and _PLACEHOLDER.fullmatch(value.value):
                    placeholders += 1
                else:
                    wrong.append(f"README.md:{line} model value {value.value!r} is not a <placeholder>")
            elif not (_is_env_model(value) or isinstance(value, (ast.Name, ast.Attribute))):
                wrong.append(f"README.md:{line} model value is not a placeholder or os.environ['DEVUP_MODEL']")
    assert wrong == []
    assert placeholders >= 10
    assert "client.models.list()" in _text()


# --- statements pinned to the code and the spec ------------------------------------------------------------------


def _section(title: str) -> str:
    """The text of the section ``title`` (up to the next level 1 or 2 heading), with whitespace collapsed."""
    lines: list[str] = []
    inside = False
    for line in _lines_outside_fences():
        heading = _HEADING.match(line)
        if heading:
            inside = heading.group(2) == title
            continue
        if inside:
            lines.append(line)
    return " ".join(" ".join(lines).split())


def _chat_handler(request: httpx.Request) -> httpx.Response:
    if b'"stream":true' in request.content:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n")
    completion = {"id": "c", "object": "chat.completion", "created": 1, "model": MODEL_A, "choices": []}
    return httpx.Response(200, json=completion)


def test_the_extra_body_stream_rule_is_stated_as_the_code_applies_it() -> None:
    # The code refuses only an extra_body stream that disagrees with the argument (_streaming.stream_requested).
    calls: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(bytes(request.content))
        return _chat_handler(request)

    client = DevupAI(
        api_key=TEST_KEY, base_url=BASE, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    params: dict[str, Any] = {"model": "<chat-model-id>", "messages": []}
    refused = [{"extra_body": {"stream": True}}, {"stream": True, "extra_body": {"stream": False}}]
    sent = [{"extra_body": {"stream": False}}, {"stream": True, "extra_body": {"stream": True}}]
    for case in refused:
        with pytest.raises(ValueError):
            client.chat.completions.create(**params, **case)
    assert calls == []
    for case in sent:
        result = client.chat.completions.create(**params, **case)
        close = getattr(result, "close", None)
        if callable(close):
            close()
    assert len(calls) == len(sent)
    client.close()

    text = " ".join(_text().split())
    assert "`stream` inside `extra_body` raises" not in text
    assert "`stream` set in `extra_body`" not in text
    streaming = _section("Streaming")
    assert "when `extra_body` sets `stream` to `True` without the `stream=True` argument" in streaming
    assert "or `stream=True` comes with an `extra_body` `stream` that is not `True`" in streaming


#: The SDK words for the spec operations that carry the 10 MiB request size limit (HTTP 413). ``createMessage`` has no
#: method in this SDK.
_LIMITED_OPERATIONS = {
    "createChatCompletion": "chat completions",
    "createResponse": "Responses",
    "createEmbedding": "embeddings",
    "createReranking": "rerank",
    "createImageGeneration": "image generation",
    "createImageEdit": "image generation and edits",
    "createSpeech": "speech",
    "createTranscription": "transcriptions",
    "createVideoGeneration": "video",
    "createNativeInference": "native inference",
}
_NOT_IN_THE_SDK = {"createMessage"}


def _operations_with_a_size_limit() -> set[str]:
    found: set[str] = set()
    operation = ""
    for line in (ROOT / "openapi" / "devupai-v1.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("components:"):
            break
        match = re.search(r"operationId:\s*(\w+)", line)
        if match:
            operation = match.group(1)
        elif operation and ("10,485,760" in line or line.strip() == "'413':"):
            found.add(operation)
    return found


def test_the_size_limit_list_names_every_limited_endpoint() -> None:
    operations = _operations_with_a_size_limit()
    assert len(operations) >= 10
    assert sorted(operations - set(_LIMITED_OPERATIONS) - _NOT_IN_THE_SDK) == []
    section = _section("Request Size Limits")
    assert "10,485,760 bytes" in section
    missing = [
        word for operation, word in _LIMITED_OPERATIONS.items() if operation in operations and word not in section
    ]
    assert missing == []


# --- errors ------------------------------------------------------------------------------------------------------


def _error_table_classes() -> list[str]:
    section: list[str] = []
    inside = False
    for line in _lines_outside_fences():
        heading = _HEADING.match(line)
        if heading:
            inside = heading.group(2) == "Error Handling"
            continue
        if inside:
            section.append(line)
    return [match.group(1) for line in section if (match := re.match(r"^\|\s*`(\w+)`\s*\|", line))]


def test_the_error_table_lists_every_exported_error_class() -> None:
    exported = {
        name
        for name in devupai.__all__
        if isinstance(getattr(devupai, name), type) and issubclass(getattr(devupai, name), Exception)
    }
    assert len(exported) >= 15
    listed = _error_table_classes()
    error_rows = [name for name in listed if name in exported or name.endswith("Error")]
    assert sorted(exported - set(error_rows)) == []
    assert sorted(set(error_rows) - exported) == []
    assert len(error_rows) == len(set(error_rows)), "a class is listed twice"
    assert "request_id" in _text()
