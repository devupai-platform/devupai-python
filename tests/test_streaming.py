"""Streaming: ``Stream`` / ``AsyncStream``, the SSE parser and ``create(stream=True)`` (block P3, contract §2).

A test that twins one of the 28 cases of Node ``tests/streaming.test.ts`` names it in a comment
(``# Node: <describe> > <case>``). The other tests cover the Python surface (``with`` / ``async with``, ``close()``,
one iteration only), the Responses error frames, the transport rules of contract §1 and the README promise (the last
chat chunk has empty ``choices`` and carries ``usage``).

Response bodies are ``httpx`` streams defined here (:class:`Body`). A body yields its byte pieces and can then raise
an ``httpx`` error, or a caller's exception. A stalled read is simulated by a body that raises ``httpx.ReadTimeout``
(the sync client sets the idle timeout as the read timeout of the response, contract §1). Only the two async
idle-timeout tests wait on the real clock, for a fraction of a second.

Expected item counts, written before any run:

- Node twins: 28 cases in 27 tests (the Node ``it.each`` is one parametrised test). 4 are pure ``SSEParser`` tests
  (4 items); the other 23 run on both client kinds: 27 client parameter sets x 2 = 54 items (the request-id test has
  2 header sets, the unterminated-event test 2 endings, the per-call idle test 3 values). Node twins: 58 items.
- Extra tests: 17 ``SSEParser`` items, 46 client items (23 parameter sets x 2 kinds), 5 async-only items and
  1 export item: 69 items.
- Total: 127 items.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio
import httpx
import pytest
from helpers.mock_transport import BASE, MODEL_A, REQ_ID, TEST_KEY, MockAPI, error_response, sequence
from helpers.samples import method_of

import devupai
from devupai import (
    APIConnectionError,
    APIResponse,
    APITimeoutError,
    BadRequestError,
    DebugEvent,
    DevupAPIError,
    InternalServerError,
)
from devupai.types import ChatCompletionChunk, ResponseStreamEvent, TokenUsage

if TYPE_CHECKING:
    from conftest import FakeClock

CHAT = "chat.completions.create"
RESPONSES = "responses.create"
CHUNK_ID = "chatcmpl-00000000-0000-4000-8000-000000000000"
DONE = "data: [DONE]\n\n"
DEFAULT_MESSAGE = "The request could not be completed."
ITERATED_MESSAGE = "A stream can be iterated only once."
USAGE: dict[str, object] = {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}
CHAT_PARAMS: dict[str, Any] = {"model": MODEL_A, "messages": [{"role": "user", "content": "Hi"}], "stream": True}
RESPONSE_PARAMS: dict[str, Any] = {"model": MODEL_A, "input": "Hello", "stream": True}


# --- frames --------------------------------------------------------------------------------------------------------


def compact(value: object) -> str:
    """JSON text as a server writes it: compact, and with non-ASCII characters kept as UTF-8."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def chunk(**fields: object) -> dict[str, object]:
    """The chunk shape of a chat completion stream."""
    return {"id": CHUNK_ID, "object": "chat.completion.chunk", "created": 1700000000, "model": MODEL_A, **fields}


def content_chunk(content: str, finish_reason: str | None = None) -> dict[str, object]:
    return chunk(choices=[{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}])


def data_event(value: object) -> str:
    return f"data: {compact(value)}\n\n"


def real_chat_stream() -> list[str]:
    """The real shape: role chunk, content, finish, a final ``choices: []`` chunk with ``usage``, ``[DONE]``."""
    return [
        data_event(
            chunk(
                service_tier=None,
                choices=[
                    {"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}
                ],
            )
        ),
        data_event(
            chunk(
                service_tier="default",
                choices=[{"index": 0, "delta": {"content": "Hello"}, "logprobs": None, "finish_reason": None}],
            )
        ),
        data_event(
            chunk(
                service_tier="default",
                choices=[{"index": 0, "delta": {"content": " world"}, "logprobs": None, "finish_reason": "stop"}],
            )
        ),
        data_event(chunk(service_tier="default", choices=[], usage=USAGE)),
        DONE,
    ]


def response_event(sequence_number: int, event_type: str, **fields: object) -> str:
    """One Responses stream event, with its SSE ``event:`` line."""
    frame = {"type": event_type, "sequence_number": sequence_number, **fields}
    return f"event: {event_type}\ndata: {compact(frame)}\n\n"


def responses_text_stream() -> list[str]:
    return [
        response_event(0, "response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="Hel"),
        response_event(1, "response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta="lo"),
        response_event(2, "response.output_text.done", item_id="msg_1", output_index=0, content_index=0, text="Hello"),
    ]


def contents(chunks: Sequence[Any]) -> list[str | None]:
    """The ``delta.content`` of the first choice of each chunk (``None`` when a chunk has no choice)."""
    return [c.choices[0].delta.content if c.choices else None for c in chunks]


# --- bodies --------------------------------------------------------------------------------------------------------


class CallerAbort(BaseException):
    """A caller's own interruption raised while the body is read (the twin of a user abort in Node)."""


class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A response body that yields ``pieces`` one read at a time.

    After the last piece it raises ``error`` when given, and the async form stalls for ever when ``hang`` is set.
    ``delay`` (async only) waits that many real seconds before each piece; ``on_piece`` runs before each piece.
    """

    def __init__(
        self,
        pieces: Sequence[str | bytes],
        *,
        error: BaseException | None = None,
        hang: bool = False,
        delay: float = 0.0,
        on_piece: Callable[[], None] | None = None,
    ) -> None:
        self.pieces = [piece.encode("utf-8") if isinstance(piece, str) else piece for piece in pieces]
        self.error = error
        self.hang = hang
        self.delay = delay
        self.on_piece = on_piece
        self.sent = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for piece in self.pieces:
            if self.on_piece is not None:
                self.on_piece()
            self.sent += 1
            yield piece
        if self.error is not None:
            raise self.error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            if self.delay:
                await anyio.sleep(self.delay)
            if self.on_piece is not None:
                self.on_piece()
            self.sent += 1
            yield piece
        if self.error is not None:
            raise self.error
        if self.hang:
            await anyio.sleep_forever()

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class SSEServer:
    """A handler that answers every call with a new ``text/event-stream`` response over a new :class:`Body`."""

    def __init__(
        self,
        pieces: Sequence[str | bytes],
        *,
        headers: dict[str, str] | None = None,
        error: BaseException | None = None,
        hang: bool = False,
        delay: float = 0.0,
        on_piece: Callable[[], None] | None = None,
    ) -> None:
        self.pieces = list(pieces)
        self.headers = {"content-type": "text/event-stream", "x-request-id": REQ_ID, **(headers or {})}
        self.error = error
        self.hang = hang
        self.delay = delay
        self.on_piece = on_piece
        self.bodies: list[Body] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = Body(self.pieces, error=self.error, hang=self.hang, delay=self.delay, on_piece=self.on_piece)
        self.bodies.append(body)
        return httpx.Response(200, headers=self.headers, stream=body)

    @property
    def body(self) -> Body:
        """The body of the last call."""
        return self.bodies[-1]


def setup(kind: str, handler: Callable[[httpx.Request], httpx.Response], **opts: Any) -> tuple[MockAPI, Any]:
    """A mock API and a client of ``kind`` answered by it; ``max_retries`` is ``0`` unless given."""
    opts.setdefault("max_retries", 0)
    api = MockAPI(handler)
    return api, api.client(kind, **opts)


# --- running a stream ----------------------------------------------------------------------------------------------


@dataclass
class Outcome:
    """What a streaming call gave: the call result, the stream, the items read, and the exception raised."""

    result: Any = None
    stream: Any = None
    items: list[Any] = field(default_factory=list)
    error: BaseException | None = None


#: Called for each item; ``"break"`` leaves the loop, ``"close"`` closes the stream and goes on iterating.
ItemAction = Callable[[Any, Any], str | None]


def consume(
    kind: str,
    client: Any,
    name: str = CHAT,
    *,
    with_response: bool = False,
    on_open: Callable[[Any], None] | None = None,
    on_item: ItemAction | None = None,
    **kwargs: Any,
) -> Outcome:
    """Calls ``name`` and reads the stream inside ``with`` / ``async with``; an exception is recorded, not raised."""
    fn = method_of(client, name, with_response=with_response)
    out = Outcome()
    if kind == "sync":
        try:
            out.result = fn(**kwargs)
            out.stream = out.result.data if with_response else out.result
            if on_open is not None:
                on_open(out.stream)
            with out.stream:
                for item in out.stream:
                    out.items.append(item)
                    action = on_item(out.stream, item) if on_item is not None else None
                    if action == "break":
                        break
                    if action == "close":
                        out.stream.close()
        except BaseException as error:  # the tests assert on it, whatever it is
            out.error = error
        return out

    async def main() -> None:
        try:
            out.result = await fn(**kwargs)
            out.stream = out.result.data if with_response else out.result
            if on_open is not None:
                on_open(out.stream)
            async with out.stream:
                async for item in out.stream:
                    out.items.append(item)
                    action = on_item(out.stream, item) if on_item is not None else None
                    if action == "break":
                        break
                    if action == "close":
                        await out.stream.close()
        except BaseException as error:  # the tests assert on it, whatever it is
            out.error = error

    anyio.run(main)
    return out


def ok(out: Outcome) -> list[Any]:
    """The items of a stream that ended without an exception."""
    if out.error is not None:
        raise out.error
    return out.items


def stream_classes() -> tuple[type[Any], type[Any]]:
    from devupai import AsyncStream, Stream

    return Stream, AsyncStream


def stream_class(kind: str) -> type[Any]:
    sync_cls, async_cls = stream_classes()
    return sync_cls if kind == "sync" else async_cls


def parser_types() -> tuple[type[Any], type[Any]]:
    from devupai._streaming import ServerSentEvent, SSEParser

    return SSEParser, ServerSentEvent


def sse(event: str | None, data: str) -> Any:
    _, event_cls = parser_types()
    return event_cls(event=event, data=data)


def recorder() -> tuple[list[DebugEvent], Callable[[DebugEvent], None]]:
    events: list[DebugEvent] = []
    return events, events.append


def raising_hook(calls: list[DebugEvent]) -> Callable[[DebugEvent], None]:
    def hook(event: DebugEvent) -> None:
        calls.append(event)
        raise RuntimeError("fixture hook error")

    return hook


# --- chat stream ---------------------------------------------------------------------------------------------------


# Node: chat stream > sends stream and stream_options in the body, and yields the real shape with usage on the final
# chunk
def test_chat_stream_sends_the_body_and_yields_the_real_shape_with_usage_on_the_final_chunk(client_kind: str) -> None:
    server = SSEServer(real_chat_stream())
    api, client = setup(client_kind, server)
    out = consume(client_kind, client, **CHAT_PARAMS, stream_options={"include_usage": True})
    chunks = ok(out)
    assert isinstance(out.stream, stream_class(client_kind))

    assert len(api.calls) == 1
    request = api.calls[0]
    assert request.method == "POST"
    assert request.url == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "text/event-stream"
    assert request.json() == {**CHAT_PARAMS, "stream_options": {"include_usage": True}}

    assert len(chunks) == 4
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    assert contents(chunks) == ["", "Hello", " world", None]
    assert all(c.usage is None for c in chunks[:3])
    last = chunks[3]
    assert last.choices == []
    assert isinstance(last.usage, TokenUsage)
    assert last.usage.total_tokens == 12
    assert last.usage.prompt_tokens == 10
    assert last.usage.completion_tokens == 2
    assert last.usage.prompt_tokens_details is not None
    assert last.usage.prompt_tokens_details.cached_tokens == 0
    assert last.usage.completion_tokens_details is not None
    assert last.usage.completion_tokens_details.reasoning_tokens == 0
    assert last.service_tier == "default"
    assert all(c.devup is None for c in chunks)


# Node: chat stream > exposes response, requestId and warnings on the stream
@pytest.mark.parametrize(
    ("extra_headers", "client_request_id"),
    [({}, None), ({"x-client-request-id": "fixture-client-1"}, "fixture-client-1")],
    ids=["no_client_request_id", "client_request_id"],
)
def test_stream_exposes_response_request_ids_and_warnings_before_the_first_item(
    client_kind: str, extra_headers: dict[str, str], client_request_id: str | None
) -> None:
    server = SSEServer(
        real_chat_stream(), headers={"x-devup-warnings": "fixture_notice_a, fixture_notice_b", **extra_headers}
    )
    _, client = setup(client_kind, server)
    seen: dict[str, Any] = {}

    def on_open(stream: Any) -> None:
        seen["read_before"] = server.body.sent
        seen["response"] = stream.response
        seen["request_id"] = stream.request_id
        seen["client_request_id"] = stream.client_request_id
        seen["warnings"] = stream.warnings

    out = consume(client_kind, client, on_open=on_open, **CHAT_PARAMS)
    assert len(ok(out)) == 4
    assert seen["read_before"] == 0
    assert isinstance(seen["response"], httpx.Response)
    assert seen["response"].status_code == 200
    assert seen["request_id"] == REQ_ID
    assert seen["client_request_id"] == client_request_id
    assert seen["warnings"] == ["fixture_notice_a", "fixture_notice_b"]


# Node: chat stream > withResponse() on a stream gives the request id before reading the body
def test_with_response_on_a_stream_gives_the_request_id_before_reading_the_body(client_kind: str) -> None:
    server = SSEServer(real_chat_stream())
    _, client = setup(client_kind, server)
    seen: dict[str, Any] = {}
    out = consume(
        client_kind,
        client,
        with_response=True,
        on_open=lambda _stream: seen.update(read_before=server.body.sent),
        **CHAT_PARAMS,
    )
    assert len(ok(out)) == 4
    assert isinstance(out.result, APIResponse)
    assert isinstance(out.result.data, stream_class(client_kind))
    assert out.result.response.headers["content-type"] == "text/event-stream"
    assert out.result.request_id == REQ_ID
    assert out.result.warnings == []
    assert seen["read_before"] == 0


# Node: chat stream > an in-stream error frame throws DevupAPIError with the frame code and the stream request id
def test_chat_error_frame_raises_the_base_error_with_the_frame_code_and_the_stream_request_id(client_kind: str) -> None:
    server = SSEServer(
        [
            data_event(content_chunk("Hel")),
            data_event({"error": {"message": DEFAULT_MESSAGE, "type": "devup_error", "code": "fixture_conflict"}}),
            DONE,
        ]
    )
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert len(out.items) == 1
    error = out.error
    assert type(error) is DevupAPIError  # the base class with the stream's status (Q28)
    assert error.code == "fixture_conflict"
    assert error.type == "devup_error"
    assert error.message == DEFAULT_MESSAGE
    assert str(error) == DEFAULT_MESSAGE
    assert error.request_id == REQ_ID
    assert error.status == 200


# Node: chat stream > a malformed frame goes to onDebug and is skipped, never silently
def test_malformed_frame_goes_to_on_debug_and_is_skipped(client_kind: str) -> None:
    events, hook = recorder()
    server = SSEServer(
        [data_event(content_chunk("a")), "data: {not json}\n\n", data_event(content_chunk("b", "stop")), DONE]
    )
    _, client = setup(client_kind, server, on_debug=hook)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert contents(chunks) == ["a", "b"]
    malformed = [event for event in events if event.type == "malformed_frame"]
    assert malformed == [DebugEvent("malformed_frame", {"data": "{not json}"})]


# Node: chat stream > a malformed frame without onDebug does not break the stream
def test_malformed_frame_without_on_debug_does_not_break_the_stream(client_kind: str) -> None:
    server = SSEServer(["data: {oops\n\n", data_event(chunk(choices=[])), DONE])
    _, client = setup(client_kind, server)
    assert len(ok(consume(client_kind, client, **CHAT_PARAMS))) == 1


# --- SSE parsing through a client ----------------------------------------------------------------------------------


# Node: SSE parsing > joins multi-line data with a newline, ignores comments and other fields, tolerates CRLF
def test_multi_line_data_comments_other_fields_and_crlf(client_kind: str) -> None:
    server = SSEServer(
        [
            ": keep-alive\r\n\r\n",
            'id: 1\r\nretry: 1000\r\ndata: {"id":"chatcmpl-1","object":"chat.completion.chunk",\r\n',
            'data: "created":1,"model":"fixture/model-a","choices":[]}\r\n\r\n',
            "data: [DONE]\r\n\r\n",
        ]
    )
    _, client = setup(client_kind, server)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 1
    only = chunks[0]
    assert (only.id, only.object, only.created, only.model, only.choices) == (
        "chatcmpl-1",
        "chat.completion.chunk",
        1,
        MODEL_A,
        [],
    )


# Node: SSE parsing > handles events split across network chunks byte by byte, including UTF-8 characters
def test_events_split_byte_by_byte_including_utf8_characters(client_kind: str) -> None:
    text = "مرحبا é 👋"
    raw = (data_event(content_chunk(text)) + DONE).encode("utf-8")
    server = SSEServer([bytes([b]) for b in raw])
    _, client = setup(client_kind, server)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == text


# Node: SSE parsing > a body without [DONE] whose last event %s yields the complete events only and reports the last
# one to onDebug (it.each: "ends with a single newline", "ends with no newline")
@pytest.mark.parametrize("ending", ["\n", ""], ids=["single_newline", "no_newline"])
def test_unterminated_last_event_is_dropped_and_reported(client_kind: str, ending: str) -> None:
    events, hook = recorder()
    last_data = compact(chunk(choices=[], usage=USAGE))
    server = SSEServer([data_event(chunk(choices=[])), f"data: {last_data}{ending}"])
    _, client = setup(client_kind, server, on_debug=hook)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 1
    assert chunks[0].choices == []
    assert chunks[0].usage is None
    assert events == [DebugEvent("malformed_frame", {"data": last_data})]


# Node: SSE parsing > a body without [DONE] whose last event is complete ends the stream normally with no debug event
def test_end_of_body_after_a_complete_event_ends_the_stream_normally(client_kind: str) -> None:
    events, hook = recorder()
    server = SSEServer([data_event(chunk(choices=[])), data_event(chunk(choices=[], usage=USAGE))])
    _, client = setup(client_kind, server, on_debug=hook)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 2
    assert isinstance(chunks[1].usage, TokenUsage)
    assert chunks[1].usage.total_tokens == 12
    assert events == []


# Node: SSE parsing > one large event delivered in many small chunks is parsed whole
def test_one_large_event_in_many_small_pieces_is_parsed_whole(client_kind: str) -> None:
    text = "é👋 fixture " * 4000  # about 60 KB of UTF-8 in one event
    raw = (data_event(content_chunk(text)) + DONE).encode("utf-8")
    pieces = [raw[i : i + 13] for i in range(0, len(raw), 13)]
    assert len(pieces) > 1000
    events, hook = recorder()
    server = SSEServer(pieces)
    _, client = setup(client_kind, server, on_debug=hook)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == text
    assert events == []


# Node: SSE parsing > ends at [DONE] and ignores anything after it
def test_done_ends_the_stream_and_nothing_after_it_is_read(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[])), DONE, data_event(chunk(choices=[]))])
    _, client = setup(client_kind, server)
    assert len(ok(consume(client_kind, client, **CHAT_PARAMS))) == 1
    assert server.body.sent == 2
    assert server.body.closed


# Node: SSE parsing > breaking out of the loop cancels the body
def test_leaving_the_with_block_early_closes_the_body(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[])), data_event(chunk(choices=[])), DONE])
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, on_item=lambda _stream, _item: "break", **CHAT_PARAMS)
    assert len(ok(out)) == 1
    assert server.body.closed
    assert out.stream.response.is_closed


# Node: SSE parsing > abort() cancels the body
def test_close_before_reading_closes_the_body_and_is_idempotent(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[])), DONE])
    _, client = setup(client_kind, server)
    if client_kind == "sync":
        stream = client.chat.completions.create(**CHAT_PARAMS)
        stream.close()
        stream.close()
        items = list(stream)
    else:

        async def main() -> tuple[Any, list[Any]]:
            stream = await client.chat.completions.create(**CHAT_PARAMS)
            await stream.close()
            await stream.close()
            return stream, [item async for item in stream]

        stream, items = anyio.run(main)
    assert server.body.closed
    assert stream.response.is_closed
    assert items == []
    assert server.body.sent == 0


# --- SSEParser (pure) ----------------------------------------------------------------------------------------------


# Node: SSEParser (incremental scan) > a large multi-line event pushed one character at a time is returned once, by the
# push that ends it
def test_parser_large_multi_line_event_fed_one_character_at_a_time() -> None:
    parser_cls, _ = parser_types()
    parser = parser_cls()
    line_a = "a" * 20_000
    line_b = "b" * 20_000
    text = f"event: fixture\ndata: {line_a}\ndata: {line_b}\n\ndata: next\n\n"
    results: list[tuple[int, list[Any]]] = []
    for index, char in enumerate(text):
        events = parser.feed(char)
        if events:
            results.append((index, events))
    first_end = text.index("\n\n", text.index(line_b)) + 1
    assert results == [
        (first_end, [sse("fixture", f"{line_a}\n{line_b}")]),
        (len(text) - 1, [sse(None, "next")]),
    ]
    assert parser.end() is None


# Node: SSEParser (incremental scan) > a CRLF split across two pushes is one line break
def test_parser_crlf_split_across_two_feeds_is_one_line_break() -> None:
    parser_cls, _ = parser_types()
    parser = parser_cls()
    assert parser.feed("data: x\r") == []
    assert parser.feed("\ndata: y\r") == []
    assert parser.feed("\n\r") == [sse(None, "x\ny")]
    assert parser.feed("\n") == []
    assert parser.end() is None


# Node: SSEParser (incremental scan) > text after a partial line is joined to it, in chunks of any size
def test_parser_partial_lines_are_joined_in_pieces_of_any_size() -> None:
    parser_cls, _ = parser_types()
    text = f"data: {'z' * 5_000}\n\n"
    for size in [1, 2, 7, 64, 4_096]:
        parser = parser_cls()
        events: list[Any] = []
        for start in range(0, len(text), size):
            events.extend(parser.feed(text[start : start + size]))
        assert events == [sse(None, "z" * 5_000)], f"size {size}"
        assert parser.end() is None


# Node: SSEParser (incremental scan) > end() returns the data of an unterminated event
def test_parser_end_returns_the_data_of_an_unterminated_event() -> None:
    parser_cls, _ = parser_types()
    parser = parser_cls()
    assert parser.feed("data: partial") == []
    assert parser.end() == "partial"


# Extra parser cases: the WHATWG rules one at a time.
@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["data: a\n\n"], [(None, "a")]),
        (["data: a\r\n\r\n"], [(None, "a")]),
        (["data: a\rdata: b\r\r"], [(None, "a\nb")]),
        (["data: a\r", "\r"], [(None, "a")]),
        (["data: a\r", "\n", "\r", "\n"], [(None, "a")]),
        (["data:  two\n\ndata:none\n\n"], [(None, " two"), (None, "none")]),
        (["data\n\n"], [(None, "")]),
        ([": comment\n\n"], []),
        ([":comment\ndata: y\n\n"], [(None, "y")]),
        (["id: 7\nretry: 10\nfoo: bar\ndata: z\n\n"], [(None, "z")]),
        (["event: one\ndata: 1\n\ndata: 2\n\n"], [("one", "1"), (None, "2")]),
        (["event: lost\n\ndata: 3\n\n"], [(None, "3")]),
        (["event:error\ndata: 4\n\n"], [("error", "4")]),
        (["data: 1\ndata:\ndata: 3\n\n"], [(None, "1\n\n3")]),
        (["\n\n\n"], []),
    ],
    ids=[
        "lf",
        "crlf",
        "cr_only",
        "cr_then_cr_across_feeds",
        "cr_lf_cr_lf_one_byte_each",
        "one_optional_space_only",
        "field_without_colon",
        "comment_alone_dispatches_nothing",
        "comment_then_data",
        "id_retry_and_unknown_fields_ignored",
        "event_type_reset_after_dispatch",
        "event_type_reset_by_blank_line_without_data",
        "event_without_space",
        "empty_data_line_kept",
        "blank_lines_only",
    ],
)
def test_parser_rules(pieces: list[str], expected: list[tuple[str | None, str]]) -> None:
    parser_cls, _ = parser_types()
    parser = parser_cls()
    events: list[Any] = []
    for piece in pieces:
        events.extend(parser.feed(piece))
    assert events == [sse(event, data) for event, data in expected]
    assert parser.end() is None


def test_parser_end_processes_a_buffered_line_and_resets() -> None:
    parser_cls, _ = parser_types()
    parser = parser_cls()
    assert parser.feed("event: x\ndata: a\ndata: b") == []
    assert parser.end() == "a\nb"
    assert parser.feed("data: c\n\n") == [sse(None, "c")]
    assert parser.end() is None


def test_server_sent_event_is_a_frozen_value() -> None:
    event = sse("fixture", "data")
    assert (event.event, event.data) == ("fixture", "data")
    assert event == sse("fixture", "data")
    with pytest.raises(AttributeError):
        event.data = "other"


# --- errors while reading the body ---------------------------------------------------------------------------------


# Node: errors while reading the stream body > a network error during iteration throws APIConnectionError (a
# DevupAPIError) with the cause and the response headers
def test_read_failure_raises_api_connection_error_with_the_cause_and_the_headers(client_kind: str) -> None:
    cause = httpx.ReadError("connection reset")
    server = SSEServer([data_event(chunk(choices=[]))], error=cause)
    api, client = setup(client_kind, server)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert len(out.items) == 1
    error = out.error
    assert isinstance(error, APIConnectionError)
    assert isinstance(error, DevupAPIError)
    assert not isinstance(error, APITimeoutError)
    assert error.status == 0
    assert error.message == "Network error: connection reset"
    assert error.__cause__ is cause
    assert error.request_id == REQ_ID
    assert len(api.calls) == 1


# Node: errors while reading the stream body > a user abort through the request signal during iteration is not turned
# into APIConnectionError
def test_a_caller_interruption_during_iteration_is_passed_on_unchanged(client_kind: str) -> None:
    reason = CallerAbort("fixture caller abort")
    server = SSEServer([data_event(chunk(choices=[]))], error=reason)
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert len(out.items) == 1
    assert out.error is reason
    assert not isinstance(out.error, APIConnectionError)


# Node: errors while reading the stream body > stream.abort() during iteration ends the loop without an error
def test_close_during_iteration_ends_the_loop_without_an_error(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[])), data_event(chunk(choices=[])), data_event(chunk(choices=[]))])
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, on_item=lambda _stream, _item: "close", **CHAT_PARAMS)
    assert out.error is None
    assert 1 <= len(out.items) < 3
    assert server.body.closed
    assert out.stream.response.is_closed


# --- on_debug hook errors --------------------------------------------------------------------------------------------


# Node: onDebug hook errors in a stream > a throwing onDebug hook on a malformed frame does not change the stream
def test_a_raising_on_debug_hook_on_a_malformed_frame_does_not_change_the_stream(client_kind: str) -> None:
    calls: list[DebugEvent] = []
    server = SSEServer(
        [data_event(content_chunk("a")), "data: {not json}\n\n", data_event(content_chunk("b", "stop")), DONE]
    )
    _, client = setup(client_kind, server, on_debug=raising_hook(calls))
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert contents(chunks) == ["a", "b"]
    assert calls == [DebugEvent("malformed_frame", {"data": "{not json}"})]


# Node: onDebug hook errors in a stream > a throwing onDebug hook on an unterminated final event still ends the stream
# normally
def test_a_raising_on_debug_hook_on_an_unterminated_last_event_ends_the_stream_normally(client_kind: str) -> None:
    calls: list[DebugEvent] = []
    server = SSEServer([data_event(chunk(choices=[])), "data: {partial"])
    _, client = setup(client_kind, server, on_debug=raising_hook(calls))
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 1
    assert calls == [DebugEvent("malformed_frame", {"data": "{partial"})]


# --- timeouts ------------------------------------------------------------------------------------------------------


# Node: stream timeouts > the request timeout covers only the time until headers, not the whole stream
def test_the_request_timeout_covers_only_the_time_until_the_headers(client_kind: str, fake_clock: FakeClock) -> None:
    server = SSEServer(real_chat_stream(), on_piece=lambda: fake_clock.advance(40))
    _, client = setup(client_kind, server, timeout=60)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 4  # the body took 200 s on the fake clock, more than the 60 s timeout


# Node: stream timeouts > streamIdleTimeout aborts with APITimeoutError when no bytes arrive for that long, and cancels
# the body
def test_stream_idle_timeout_raises_api_timeout_error_and_closes_the_body(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[]))], error=httpx.ReadTimeout("timed out"))
    api, client = setup(client_kind, server, stream_idle_timeout=2)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert len(out.items) == 1
    error = out.error
    assert isinstance(error, APITimeoutError)
    assert error.status == 0
    assert error.message == "No data was received on the stream for 2 seconds."
    assert str(error) == error.message
    assert error.request_id == REQ_ID
    assert server.body.closed
    assert len(api.calls) == 1


# Node: stream timeouts > per-call streamIdleTimeout overrides the client option
@pytest.mark.parametrize(
    ("client_idle", "call_idle", "shown"),
    [(20, 2, "2"), (None, 0.5, "0.5"), (2, 7, "7")],
    ids=["call_shorter", "call_only", "call_longer"],
)
def test_per_call_stream_idle_timeout_overrides_the_client_option(
    client_kind: str, client_idle: float | None, call_idle: float, shown: str
) -> None:
    server = SSEServer([data_event(chunk(choices=[]))], error=httpx.ReadTimeout("timed out"))
    _, client = setup(client_kind, server, stream_idle_timeout=client_idle)
    out = consume(client_kind, client, stream_idle_timeout=call_idle, **CHAT_PARAMS)
    assert isinstance(out.error, APITimeoutError)
    assert out.error.message == f"No data was received on the stream for {shown} seconds."


# Node: stream timeouts > an error in the body after the first byte is not retried
def test_a_read_failure_after_the_headers_is_not_retried_on_a_billed_call(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[]))], error=httpx.ReadError("connection reset"))
    api, client = setup(client_kind, server, max_retries=2)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert isinstance(out.error, APIConnectionError)
    assert len(api.calls) == 1


# --- extra: async idle timeout on the real clock -------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_stream_idle_timeout_ends_a_stalled_read() -> None:
    server = SSEServer([data_event(chunk(choices=[]))], hang=True)
    client = MockAPI(server).async_client(max_retries=0, stream_idle_timeout=0.05)
    seen: list[Any] = []
    with anyio.fail_after(5):  # a guard: the stream itself must end after 0.05 s
        stream = await client.chat.completions.create(**CHAT_PARAMS)
        with pytest.raises(APITimeoutError) as caught:
            async with stream:
                async for item in stream:
                    seen.append(item)
    assert caught.value.status == 0
    assert caught.value.message == "No data was received on the stream for 0.05 seconds."
    assert len(seen) == 1
    assert server.body.closed


@pytest.mark.anyio
@pytest.mark.parametrize("call_idle", [5, 0], ids=["longer", "off"])
async def test_async_per_call_stream_idle_timeout_overrides_the_client_option(call_idle: float) -> None:
    server = SSEServer([data_event(chunk(choices=[])), DONE], delay=0.05)
    client = MockAPI(server).async_client(max_retries=0, stream_idle_timeout=0.01)
    with anyio.fail_after(5):
        stream = await client.chat.completions.create(**CHAT_PARAMS, stream_idle_timeout=call_idle)
        async with stream:
            items = [item async for item in stream]
    assert len(items) == 1


@pytest.mark.anyio
async def test_async_per_call_none_keeps_the_client_stream_idle_timeout() -> None:
    server = SSEServer([data_event(chunk(choices=[])), DONE], delay=0.2)
    client = MockAPI(server).async_client(max_retries=0, stream_idle_timeout=0.05)
    with anyio.fail_after(5):
        stream = await client.chat.completions.create(**CHAT_PARAMS, stream_idle_timeout=None)
        with pytest.raises(APITimeoutError) as caught:
            async with stream:
                async for _item in stream:
                    pass
    assert caught.value.message == "No data was received on the stream for 0.05 seconds."


@pytest.mark.anyio
async def test_async_stream_without_idle_timeout_waits_for_slow_data() -> None:
    server = SSEServer([data_event(chunk(choices=[])), DONE], delay=0.05)
    client = MockAPI(server).async_client(max_retries=0)
    with anyio.fail_after(5):
        stream = await client.chat.completions.create(**CHAT_PARAMS)
        async with stream:
            items = [item async for item in stream]
    assert len(items) == 1


# --- extra: chat error frames --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "message", "code", "error_type", "param"),
    [
        ({"error": {"code": "fixture_conflict"}}, DEFAULT_MESSAGE, "fixture_conflict", None, None),
        ({"error": {}}, DEFAULT_MESSAGE, None, None, None),
        (
            {"error": {"message": "Fixture message.", "type": "devup_error", "code": "fixture_code", "param": "tools"}},
            "Fixture message.",
            "fixture_code",
            "devup_error",
            "tools",
        ),
    ],
    ids=["default_message", "empty_error_object", "every_field"],
)
def test_chat_error_frame_fields(
    client_kind: str,
    frame: dict[str, object],
    message: str,
    code: str | None,
    error_type: str | None,
    param: str | None,
) -> None:
    server = SSEServer([data_event(frame), DONE])
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert out.items == []
    error = out.error
    assert type(error) is DevupAPIError
    assert (error.status, error.message, error.code, error.type, error.param) == (200, message, code, error_type, param)
    assert error.request_id == REQ_ID


def test_a_chat_frame_with_choices_and_an_error_object_is_a_regular_chunk(client_kind: str) -> None:
    server = SSEServer([data_event(chunk(choices=[], error={"message": "Fixture message."})), DONE])
    _, client = setup(client_kind, server)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert len(chunks) == 1
    assert isinstance(chunks[0], ChatCompletionChunk)


def test_chat_stream_without_stream_options_still_ends_with_the_usage_chunk(client_kind: str) -> None:
    # README: the API adds the final usage chunk to every chat completion stream.
    server = SSEServer(real_chat_stream())
    api, client = setup(client_kind, server)
    chunks = ok(consume(client_kind, client, **CHAT_PARAMS))
    assert api.calls[0].json() == CHAT_PARAMS
    last = chunks[-1]
    assert last.choices == []
    assert isinstance(last.usage, TokenUsage)
    assert last.usage.total_tokens == 12
    assert {c.id for c in chunks} == {CHUNK_ID}


# --- extra: responses streams --------------------------------------------------------------------------------------


def test_responses_stream_sends_the_body_and_yields_stream_events(client_kind: str) -> None:
    server = SSEServer(responses_text_stream())
    api, client = setup(client_kind, server)
    out = consume(client_kind, client, RESPONSES, **RESPONSE_PARAMS)
    events = ok(out)
    assert isinstance(out.stream, stream_class(client_kind))
    request = api.calls[0]
    assert request.method == "POST"
    assert request.url == f"{BASE}/responses"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["content-type"] == "application/json"
    assert request.json() == RESPONSE_PARAMS
    assert all(isinstance(event, ResponseStreamEvent) for event in events)
    assert [event.type for event in events] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
    ]
    assert [event.sequence_number for event in events] == [0, 1, 2]
    assert [event.delta for event in events[:2]] == ["Hel", "lo"]
    assert events[2].text == "Hello"


def test_responses_with_response_returns_an_api_response_holding_a_stream(client_kind: str) -> None:
    server = SSEServer(responses_text_stream(), headers={"x-devup-warnings": "fixture_notice_a"})
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, RESPONSES, with_response=True, **RESPONSE_PARAMS)
    assert len(ok(out)) == 3
    assert isinstance(out.result, APIResponse)
    assert isinstance(out.result.data, stream_class(client_kind))
    assert out.result.request_id == REQ_ID
    assert out.result.warnings == ["fixture_notice_a"]
    assert out.result.data.warnings == ["fixture_notice_a"]


@pytest.mark.parametrize(
    ("piece", "message", "code", "error_type", "param"),
    [
        (
            'event: error\ndata: {"code":"fixture_code","message":"Fixture message.","param":"input"}\n\n',
            "Fixture message.",
            "fixture_code",
            None,
            "input",
        ),
        (
            data_event(
                {
                    "type": "error",
                    "sequence_number": 1,
                    "error": {"message": "Nested message.", "code": "nested_code", "type": "devup_error", "param": "t"},
                }
            ),
            "Nested message.",
            "nested_code",
            "devup_error",
            "t",
        ),
        (
            data_event(
                {
                    "type": "error",
                    "sequence_number": 1,
                    "message": "Frame message.",
                    "code": "frame_code",
                    "param": "frame_param",
                    "error": {"message": "Nested.", "code": "nested", "type": "devup_error", "param": "nested_param"},
                }
            ),
            "Frame message.",
            "frame_code",
            "devup_error",
            "frame_param",
        ),
        ('event: error\ndata: "fixture text"\n\n', DEFAULT_MESSAGE, None, None, None),
        (data_event({"type": "error", "sequence_number": 1}), DEFAULT_MESSAGE, None, None, None),
    ],
    ids=["sse_event_error", "type_error_nested", "frame_fields_first", "non_object_frame", "default_message"],
)
def test_responses_error_frame_raises_the_base_error(
    client_kind: str, piece: str, message: str, code: str | None, error_type: str | None, param: str | None
) -> None:
    server = SSEServer([responses_text_stream()[0], piece, responses_text_stream()[1]])
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, RESPONSES, **RESPONSE_PARAMS)
    assert len(out.items) == 1
    error = out.error
    assert type(error) is DevupAPIError
    assert (error.status, error.message, error.code, error.type, error.param) == (200, message, code, error_type, param)
    assert error.request_id == REQ_ID


def test_responses_read_failure_is_not_retried(client_kind: str) -> None:
    server = SSEServer(responses_text_stream()[:1], error=httpx.ReadError("connection reset"))
    api, client = setup(client_kind, server, max_retries=2)
    out = consume(client_kind, client, RESPONSES, **RESPONSE_PARAMS)
    assert len(out.items) == 1
    assert isinstance(out.error, APIConnectionError)
    assert not isinstance(out.error, APITimeoutError)
    assert out.error.message == "Network error: connection reset"
    assert len(api.calls) == 1


# --- extra: the Python stream surface ------------------------------------------------------------------------------


def test_a_stream_can_be_iterated_only_once(client_kind: str) -> None:
    server = SSEServer(real_chat_stream())
    _, client = setup(client_kind, server)
    if client_kind == "sync":
        stream = client.chat.completions.create(**CHAT_PARAMS)
        first = list(stream)
        with pytest.raises(RuntimeError) as caught:
            for _item in stream:
                pass
    else:

        async def main() -> tuple[list[Any], pytest.ExceptionInfo[RuntimeError]]:
            stream = await client.chat.completions.create(**CHAT_PARAMS)
            items = [item async for item in stream]
            with pytest.raises(RuntimeError) as info:
                async for _item in stream:
                    pass
            return items, info

        first, caught = anyio.run(main)
    assert len(first) == 4
    assert caught.value.args == (ITERATED_MESSAGE,)
    assert len(server.bodies) == 1


def test_leaving_the_with_block_without_iterating_closes_the_response(client_kind: str) -> None:
    server = SSEServer(real_chat_stream())
    _, client = setup(client_kind, server)
    if client_kind == "sync":
        stream = client.chat.completions.create(**CHAT_PARAMS)
        with stream as entered:
            assert entered is stream
    else:

        async def main() -> Any:
            stream = await client.chat.completions.create(**CHAT_PARAMS)
            async with stream as entered:
                assert entered is stream
            return stream

        stream = anyio.run(main)
    assert stream.response.is_closed
    assert server.body.closed
    assert server.body.sent == 0


def test_an_exception_inside_the_with_block_closes_the_stream_and_propagates(client_kind: str) -> None:
    server = SSEServer(real_chat_stream())
    _, client = setup(client_kind, server)

    def boom(_stream: Any, _item: Any) -> str | None:
        raise ValueError("fixture failure")

    out = consume(client_kind, client, on_item=boom, **CHAT_PARAMS)
    assert isinstance(out.error, ValueError)
    assert out.error.args == ("fixture failure",)
    assert len(out.items) == 1
    assert server.body.closed


def test_close_after_the_end_is_a_no_op(client_kind: str) -> None:
    server = SSEServer(real_chat_stream())
    _, client = setup(client_kind, server)
    if client_kind == "sync":
        stream = client.chat.completions.create(**CHAT_PARAMS)
        items = list(stream)
        stream.close()
        stream.close()
    else:

        async def main() -> tuple[Any, list[Any]]:
            stream = await client.chat.completions.create(**CHAT_PARAMS)
            collected = [item async for item in stream]
            await stream.close()
            await stream.close()
            return stream, collected

        stream, items = anyio.run(main)
    assert len(items) == 4
    assert stream.response.is_closed
    assert server.body.closed


def test_stream_true_returns_the_public_stream_classes(client_kind: str) -> None:
    sync_cls, async_cls = stream_classes()
    server = SSEServer(real_chat_stream())
    _, client = setup(client_kind, server)
    out = consume(client_kind, client, **CHAT_PARAMS)
    ok(out)
    if client_kind == "sync":
        assert type(out.stream) is sync_cls
    else:
        assert type(out.stream) is async_cls


def test_stream_classes_are_exported() -> None:
    import devupai._streaming as streaming

    sync_cls, async_cls = stream_classes()
    assert sync_cls is streaming.Stream
    assert async_cls is streaming.AsyncStream
    assert "Stream" in devupai.__all__
    assert "AsyncStream" in devupai.__all__


# --- extra: errors before the headers (the P1 rules) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "error_cls", "calls"),
    [
        (error_response(400, "Fixture bad request.", code="fixture_invalid"), BadRequestError, 1),
        (error_response(500, "Fixture server error."), InternalServerError, 1),
        (error_response(503, "Fixture unavailable."), None, 2),
        (error_response(429, "Fixture rate limit."), None, 2),
    ],
    ids=["400", "500_not_retried", "503_then_stream", "429_then_stream"],
)
def test_an_error_before_the_headers_follows_the_billed_rule(
    client_kind: str, first: httpx.Response, error_cls: type[DevupAPIError] | None, calls: int
) -> None:
    server = SSEServer(real_chat_stream())
    api, client = setup(client_kind, sequence(first, server), max_retries=2)
    out = consume(client_kind, client, **CHAT_PARAMS)
    assert len(api.calls) == calls
    if error_cls is None:
        assert len(ok(out)) == 4
        assert all(call.headers["accept"] == "text/event-stream" for call in api.calls)
        assert all(call.json() == CHAT_PARAMS for call in api.calls)
    else:
        assert type(out.error) is error_cls
        assert out.error.status == first.status_code
        assert out.stream is None
        assert server.bodies == []


# --- extra: the transport (contract §1) ----------------------------------------------------------------------------


def test_a_read_failure_on_a_non_billed_stream_is_not_retried(client_kind: str) -> None:
    from devupai._client import DEFAULT_RETRY

    sync_cls, async_cls = stream_classes()
    server = SSEServer([data_event({"n": 1})], error=httpx.ReadError("connection reset"))
    api, client = setup(client_kind, server, max_retries=2)

    def parse(frame: object) -> object:
        return frame

    def error_frame(frame: object, event: str | None) -> None:
        return None

    if client_kind == "sync":
        result = client._request("GET", "/fixture/events", kind="stream", retry=DEFAULT_RETRY)
        assert isinstance(result.data, httpx.Response)
        stream = sync_cls(result.data, parse=parse, error_frame=error_frame, on_debug=None, idle_timeout=None)
        items: list[Any] = []
        with pytest.raises(APIConnectionError) as caught:
            with stream:
                for item in stream:
                    items.append(item)
    else:

        async def main() -> tuple[Any, list[Any], pytest.ExceptionInfo[APIConnectionError]]:
            response = await client._request("GET", "/fixture/events", kind="stream", retry=DEFAULT_RETRY)
            assert isinstance(response.data, httpx.Response)
            stream = async_cls(response.data, parse=parse, error_frame=error_frame, on_debug=None, idle_timeout=None)
            collected: list[Any] = []
            with pytest.raises(APIConnectionError) as info:
                async with stream:
                    async for item in stream:
                        collected.append(item)
            return response, collected, info

        result, items, caught = anyio.run(main)
    assert result.request_id == REQ_ID
    assert items == [{"n": 1}]
    assert caught.value.message == "Network error: connection reset"
    assert not isinstance(caught.value, APITimeoutError)
    assert len(api.calls) == 1
    assert api.calls[0].url == f"{BASE}/fixture/events"
    assert api.calls[0].headers["accept"] == "text/event-stream"
    assert server.body.closed
