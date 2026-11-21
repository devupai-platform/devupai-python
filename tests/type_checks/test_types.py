"""Return types of every public method, on both clients and on every ``with_response`` twin.

``mypy --strict`` and ``pyright`` strict read this module. ``_check_sync``, ``_check_async`` and their P3 twins
(streams, transcriptions by ``response_format``, video edits and binary results) are never called: ``assert_type`` is
checked statically only. ``test_typing_module_imports`` is the one test that runs, so pytest imports the module.
"""

from __future__ import annotations

import inspect
from typing import Literal, assert_type

from devupai import (
    APIResponse,
    AsyncBinaryResponse,
    AsyncDevupAI,
    AsyncStream,
    BinaryResponse,
    DevupAI,
    Stream,
    construct_webhook_event,
    verify_webhook_signature,
)
from devupai.types import (
    AccountProfile,
    BalanceResponse,
    ChatCompletion,
    ChatCompletionChunk,
    EmbeddingCreateResponse,
    HealthResponse,
    ImageGenerateResponse,
    ModelListResponse,
    ModelObject,
    RerankCreateResponse,
    ResponseObject,
    ResponseStreamEvent,
    SpeechResponse,
    TranscriptionResponse,
    TranscriptionVerboseResponse,
    VideoGenerationResponse,
    WebhookEvent,
)

MODEL = "fixture/model-a"
ETAG = 'W/"fixture-etag"'
BODY = '{"id": "whd_1", "status": "succeeded"}'
HEADER = "t=1800000000,v1=00"
SECRET = "whsec_fixture"
AUDIO = b"\x00\x01"
DOWNLOAD_URL = "https://api.example.test/v1/images/proxy?url=x"

TranscriptionFormat = Literal["json", "verbose_json", "text", "srt", "vtt"]


def _etag_or_none() -> str | None:
    """An ``If-None-Match`` value whose static type is ``str | None``."""
    return None


def _flag() -> bool:
    """A ``stream`` value whose static type is ``bool`` (neither literal)."""
    return False


def _any_format() -> TranscriptionFormat:
    """A ``response_format`` value whose static type is the whole literal union."""
    return "json"


def _check_sync_p3(client: DevupAI) -> None:
    """P3: streams, transcriptions, video edits and binary results (sync client)."""
    flag = _flag()
    fmt = _any_format()

    # chat.completions: the stream overloads
    assert_type(
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True),
        Stream[ChatCompletionChunk],
    )
    assert_type(
        client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True, stream_idle_timeout=5.0
        ),
        Stream[ChatCompletionChunk],
    )
    assert_type(
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=False),
        ChatCompletion,
    )
    assert_type(
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=flag),
        ChatCompletion | Stream[ChatCompletionChunk],
    )
    assert_type(
        client.chat.completions.with_response.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True
        ),
        APIResponse[Stream[ChatCompletionChunk]],
    )
    assert_type(
        client.chat.completions.with_response.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=False
        ),
        APIResponse[ChatCompletion],
    )
    assert_type(
        client.chat.completions.with_response.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=flag
        ),
        APIResponse[ChatCompletion | Stream[ChatCompletionChunk]],
    )
    chat_stream = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True)
    assert_type(chat_stream.request_id, str | None)
    assert_type(chat_stream.warnings, list[str])
    with chat_stream as opened:
        for chunk in opened:
            assert_type(chunk, ChatCompletionChunk)
    assert_type(chat_stream.close(), None)

    # responses: the stream overloads
    assert_type(client.responses.create(model=MODEL, input="Hi", stream=True), Stream[ResponseStreamEvent])
    assert_type(
        client.responses.create(model=MODEL, input="Hi", stream=True, stream_idle_timeout=None),
        Stream[ResponseStreamEvent],
    )
    assert_type(client.responses.create(model=MODEL, input="Hi", stream=False), ResponseObject)
    assert_type(
        client.responses.create(model=MODEL, input="Hi", stream=flag), ResponseObject | Stream[ResponseStreamEvent]
    )
    assert_type(
        client.responses.with_response.create(model=MODEL, input="Hi", stream=True),
        APIResponse[Stream[ResponseStreamEvent]],
    )
    assert_type(
        client.responses.with_response.create(model=MODEL, input="Hi", stream=False), APIResponse[ResponseObject]
    )
    assert_type(
        client.responses.with_response.create(model=MODEL, input="Hi", stream=flag),
        APIResponse[ResponseObject | Stream[ResponseStreamEvent]],
    )
    for event in client.responses.create(model=MODEL, input="Hi", stream=True):
        assert_type(event, ResponseStreamEvent)

    # audio.transcriptions: the response_format overloads
    assert_type(client.audio.transcriptions.create(file=AUDIO, model=MODEL), TranscriptionResponse)
    assert_type(client.audio.transcriptions.create(file=AUDIO, response_format="json"), TranscriptionResponse)
    assert_type(
        client.audio.transcriptions.create(file=AUDIO, response_format="verbose_json"), TranscriptionVerboseResponse
    )
    assert_type(client.audio.transcriptions.create(file=AUDIO, response_format="text"), str)
    assert_type(client.audio.transcriptions.create(file=AUDIO, response_format="srt"), str)
    assert_type(client.audio.transcriptions.create(file=AUDIO, response_format="vtt"), str)
    assert_type(
        client.audio.transcriptions.create(file=AUDIO, response_format=fmt),
        TranscriptionResponse | TranscriptionVerboseResponse | str,
    )
    assert_type(
        client.audio.transcriptions.create(
            file=AUDIO, timestamp_granularities=["word"], extra_headers={"X-Fixture": "1"}, timeout=5.0, max_retries=0
        ),
        TranscriptionResponse,
    )
    assert_type(client.audio.transcriptions.with_response.create(file=AUDIO), APIResponse[TranscriptionResponse])
    assert_type(
        client.audio.transcriptions.with_response.create(file=AUDIO, response_format="verbose_json"),
        APIResponse[TranscriptionVerboseResponse],
    )
    assert_type(client.audio.transcriptions.with_response.create(file=AUDIO, response_format="text"), APIResponse[str])

    # video.edits
    assert_type(client.video.edits.create(model=MODEL, video=AUDIO), VideoGenerationResponse)
    assert_type(
        client.video.edits.create(
            model=MODEL, video=AUDIO, mask=AUDIO, key_points=[{"x": 0.5, "y": 0.5, "type": "positive"}]
        ),
        VideoGenerationResponse,
    )
    assert_type(client.video.edits.with_response.create(model=MODEL, video=AUDIO), APIResponse[VideoGenerationResponse])

    # binary results
    assert_type(client.images.download(DOWNLOAD_URL), BinaryResponse)
    assert_type(client.images.download(DOWNLOAD_URL, extra_headers={"Range": "bytes=0-1"}), BinaryResponse)
    assert_type(client.images.with_response.download(DOWNLOAD_URL), APIResponse[BinaryResponse])
    assert_type(client.me.avatar(), BinaryResponse)
    assert_type(client.me.avatar(if_none_match=ETAG), BinaryResponse)
    assert_type(client.me.with_response.avatar(), APIResponse[BinaryResponse])
    assert_type(client.inference.run(MODEL, {"inputs": "Hi"}, response_type="binary"), BinaryResponse)
    assert_type(client.inference.run(MODEL, {"inputs": "Hi"}, response_type="json"), object)
    assert_type(
        client.inference.with_response.run(MODEL, {"inputs": "Hi"}, response_type="binary"), APIResponse[BinaryResponse]
    )
    binary = client.images.download(DOWNLOAD_URL)
    assert_type(binary.read(), bytes)
    assert_type(binary.status_code, int)
    assert_type(binary.request_id, str | None)
    for piece in binary.iter_bytes():
        assert_type(piece, bytes)
    assert_type(binary.write_to_file("fixture.png"), None)
    with client.me.avatar() as avatar:
        assert_type(avatar, BinaryResponse)


async def _check_async_p3(client: AsyncDevupAI) -> None:
    """P3: streams, transcriptions, video edits and binary results (async client)."""
    flag = _flag()
    fmt = _any_format()

    # chat.completions: the stream overloads
    assert_type(
        await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True),
        AsyncStream[ChatCompletionChunk],
    )
    assert_type(
        await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=False),
        ChatCompletion,
    )
    assert_type(
        await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=flag),
        ChatCompletion | AsyncStream[ChatCompletionChunk],
    )
    assert_type(
        await client.chat.completions.with_response.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True
        ),
        APIResponse[AsyncStream[ChatCompletionChunk]],
    )
    assert_type(
        await client.chat.completions.with_response.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=False
        ),
        APIResponse[ChatCompletion],
    )
    assert_type(
        await client.chat.completions.with_response.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=flag
        ),
        APIResponse[ChatCompletion | AsyncStream[ChatCompletionChunk]],
    )
    chat_stream = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True, stream_idle_timeout=5.0
    )
    assert_type(chat_stream.request_id, str | None)
    async with chat_stream as opened:
        async for chunk in opened:
            assert_type(chunk, ChatCompletionChunk)
    assert_type(await chat_stream.close(), None)

    # responses: the stream overloads
    assert_type(await client.responses.create(model=MODEL, input="Hi", stream=True), AsyncStream[ResponseStreamEvent])
    assert_type(await client.responses.create(model=MODEL, input="Hi", stream=False), ResponseObject)
    assert_type(
        await client.responses.create(model=MODEL, input="Hi", stream=flag),
        ResponseObject | AsyncStream[ResponseStreamEvent],
    )
    assert_type(
        await client.responses.with_response.create(model=MODEL, input="Hi", stream=True),
        APIResponse[AsyncStream[ResponseStreamEvent]],
    )
    assert_type(
        await client.responses.with_response.create(model=MODEL, input="Hi", stream=False), APIResponse[ResponseObject]
    )
    assert_type(
        await client.responses.with_response.create(model=MODEL, input="Hi", stream=flag),
        APIResponse[ResponseObject | AsyncStream[ResponseStreamEvent]],
    )
    async for event in await client.responses.create(model=MODEL, input="Hi", stream=True):
        assert_type(event, ResponseStreamEvent)

    # audio.transcriptions: the response_format overloads
    assert_type(await client.audio.transcriptions.create(file=AUDIO, model=MODEL), TranscriptionResponse)
    assert_type(await client.audio.transcriptions.create(file=AUDIO, response_format="json"), TranscriptionResponse)
    assert_type(
        await client.audio.transcriptions.create(file=AUDIO, response_format="verbose_json"),
        TranscriptionVerboseResponse,
    )
    assert_type(await client.audio.transcriptions.create(file=AUDIO, response_format="text"), str)
    assert_type(await client.audio.transcriptions.create(file=AUDIO, response_format="srt"), str)
    assert_type(await client.audio.transcriptions.create(file=AUDIO, response_format="vtt"), str)
    assert_type(
        await client.audio.transcriptions.create(file=AUDIO, response_format=fmt),
        TranscriptionResponse | TranscriptionVerboseResponse | str,
    )
    assert_type(await client.audio.transcriptions.with_response.create(file=AUDIO), APIResponse[TranscriptionResponse])
    assert_type(
        await client.audio.transcriptions.with_response.create(file=AUDIO, response_format="verbose_json"),
        APIResponse[TranscriptionVerboseResponse],
    )
    assert_type(
        await client.audio.transcriptions.with_response.create(file=AUDIO, response_format="text"), APIResponse[str]
    )

    # video.edits
    assert_type(await client.video.edits.create(model=MODEL, video=AUDIO), VideoGenerationResponse)
    assert_type(
        await client.video.edits.with_response.create(model=MODEL, video=AUDIO), APIResponse[VideoGenerationResponse]
    )

    # binary results
    assert_type(await client.images.download(DOWNLOAD_URL), AsyncBinaryResponse)
    assert_type(await client.images.with_response.download(DOWNLOAD_URL), APIResponse[AsyncBinaryResponse])
    assert_type(await client.me.avatar(), AsyncBinaryResponse)
    assert_type(await client.me.avatar(if_none_match=ETAG), AsyncBinaryResponse)
    assert_type(await client.me.with_response.avatar(), APIResponse[AsyncBinaryResponse])
    assert_type(await client.inference.run(MODEL, {"inputs": "Hi"}, response_type="binary"), AsyncBinaryResponse)
    assert_type(await client.inference.run(MODEL, {"inputs": "Hi"}, response_type="json"), object)
    assert_type(
        await client.inference.with_response.run(MODEL, {"inputs": "Hi"}, response_type="binary"),
        APIResponse[AsyncBinaryResponse],
    )
    binary = await client.images.download(DOWNLOAD_URL)
    assert_type(await binary.read(), bytes)
    assert_type(binary.status_code, int)
    async for piece in binary.aiter_bytes():
        assert_type(piece, bytes)
    assert_type(await binary.write_to_file("fixture.png"), None)
    async with await client.me.avatar() as avatar:
        assert_type(avatar, AsyncBinaryResponse)


def _check_sync(client: DevupAI) -> None:
    maybe = _etag_or_none()

    # chat.completions
    assert_type(
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}]), ChatCompletion
    )
    assert_type(
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            extra_body={"fixture_option": 1},
            extra_headers={"X-Fixture": "1"},
            timeout=5.0,
            max_retries=0,
        ),
        ChatCompletion,
    )
    assert_type(
        client.chat.completions.with_response.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}]),
        APIResponse[ChatCompletion],
    )

    # responses
    assert_type(client.responses.create(model=MODEL, input="Hi"), ResponseObject)
    assert_type(client.responses.with_response.create(model=MODEL, input="Hi"), APIResponse[ResponseObject])

    # models: the 304 overloads
    assert_type(client.models.list(), ModelListResponse)
    assert_type(client.models.list(if_none_match=None), ModelListResponse)
    assert_type(client.models.list(if_none_match=ETAG), ModelListResponse | None)
    assert_type(client.models.list(if_none_match=maybe), ModelListResponse | None)
    assert_type(client.models.list(extra_headers={"X-Fixture": "1"}, timeout=5.0, max_retries=0), ModelListResponse)
    assert_type(client.models.retrieve(MODEL), ModelObject)
    assert_type(client.models.retrieve(MODEL, if_none_match=None), ModelObject)
    assert_type(client.models.retrieve(MODEL, if_none_match=ETAG), ModelObject | None)
    assert_type(client.models.retrieve(MODEL, if_none_match=maybe), ModelObject | None)
    assert_type(client.models.with_response.list(), APIResponse[ModelListResponse])
    assert_type(client.models.with_response.list(if_none_match=ETAG), APIResponse[ModelListResponse | None])
    assert_type(client.models.with_response.retrieve(MODEL), APIResponse[ModelObject])
    assert_type(client.models.with_response.retrieve(MODEL, if_none_match=ETAG), APIResponse[ModelObject | None])

    # images
    assert_type(client.images.generate(model=MODEL, prompt="A lighthouse"), ImageGenerateResponse)
    assert_type(
        client.images.with_response.generate(model=MODEL, prompt="A lighthouse"), APIResponse[ImageGenerateResponse]
    )
    assert_type(
        client.images.edit(model=MODEL, prompt="Make it blue", image="https://example.test/a.png"),
        ImageGenerateResponse,
    )
    assert_type(
        client.images.with_response.edit(model=MODEL, prompt="Make it blue", image="https://example.test/a.png"),
        APIResponse[ImageGenerateResponse],
    )

    # embeddings
    assert_type(client.embeddings.create(model=MODEL, input="Hi"), EmbeddingCreateResponse)
    assert_type(client.embeddings.with_response.create(model=MODEL, input="Hi"), APIResponse[EmbeddingCreateResponse])

    # rerank
    assert_type(client.rerank.create(model=MODEL, query="q", documents=["a", "b"]), RerankCreateResponse)
    assert_type(
        client.rerank.with_response.create(model=MODEL, query="q", documents=["a", "b"]),
        APIResponse[RerankCreateResponse],
    )

    # audio.speech
    assert_type(client.audio.speech.create(model=MODEL, input="Hi"), SpeechResponse)
    assert_type(client.audio.speech.with_response.create(model=MODEL, input="Hi"), APIResponse[SpeechResponse])

    # video.generations
    assert_type(client.video.generations.create(model=MODEL, prompt="A wave"), VideoGenerationResponse)
    assert_type(
        client.video.generations.with_response.create(model=MODEL, prompt="A wave"),
        APIResponse[VideoGenerationResponse],
    )

    # inference
    assert_type(client.inference.run(MODEL, {"inputs": "Hi"}), object)
    assert_type(client.inference.with_response.run(MODEL, {"inputs": "Hi"}), APIResponse[object])

    # balance, me, health
    assert_type(client.balance.retrieve(), BalanceResponse)
    assert_type(client.balance.with_response.retrieve(), APIResponse[BalanceResponse])
    assert_type(client.me.retrieve(), AccountProfile)
    assert_type(client.me.with_response.retrieve(), APIResponse[AccountProfile])
    assert_type(client.health.check(), HealthResponse)
    assert_type(client.health.with_response.check(), APIResponse[HealthResponse])

    # webhooks: local and sync
    assert_type(client.webhooks.verify_signature(raw_body=BODY, signature_header=HEADER, secret=SECRET), bool)
    assert_type(
        client.webhooks.construct_event(raw_body=BODY.encode(), signature_header=HEADER, secret=SECRET),
        WebhookEvent,
    )
    assert_type(
        verify_webhook_signature(raw_body=BODY, signature_header=HEADER, secret=SECRET, tolerance_seconds=600), bool
    )
    assert_type(construct_webhook_event(raw_body=BODY, signature_header=HEADER, secret=SECRET), WebhookEvent)


async def _check_async(client: AsyncDevupAI) -> None:
    maybe = _etag_or_none()

    # chat.completions
    assert_type(
        await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}]),
        ChatCompletion,
    )
    assert_type(
        await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            extra_body={"fixture_option": 1},
            extra_headers={"X-Fixture": "1"},
            timeout=5.0,
            max_retries=0,
        ),
        ChatCompletion,
    )
    assert_type(
        await client.chat.completions.with_response.create(model=MODEL, messages=[{"role": "user", "content": "Hi"}]),
        APIResponse[ChatCompletion],
    )

    # responses
    assert_type(await client.responses.create(model=MODEL, input="Hi"), ResponseObject)
    assert_type(await client.responses.with_response.create(model=MODEL, input="Hi"), APIResponse[ResponseObject])

    # models: the 304 overloads
    assert_type(await client.models.list(), ModelListResponse)
    assert_type(await client.models.list(if_none_match=None), ModelListResponse)
    assert_type(await client.models.list(if_none_match=ETAG), ModelListResponse | None)
    assert_type(await client.models.list(if_none_match=maybe), ModelListResponse | None)
    assert_type(
        await client.models.list(extra_headers={"X-Fixture": "1"}, timeout=5.0, max_retries=0), ModelListResponse
    )
    assert_type(await client.models.retrieve(MODEL), ModelObject)
    assert_type(await client.models.retrieve(MODEL, if_none_match=None), ModelObject)
    assert_type(await client.models.retrieve(MODEL, if_none_match=ETAG), ModelObject | None)
    assert_type(await client.models.retrieve(MODEL, if_none_match=maybe), ModelObject | None)
    assert_type(await client.models.with_response.list(), APIResponse[ModelListResponse])
    assert_type(await client.models.with_response.list(if_none_match=ETAG), APIResponse[ModelListResponse | None])
    assert_type(await client.models.with_response.retrieve(MODEL), APIResponse[ModelObject])
    assert_type(
        await client.models.with_response.retrieve(MODEL, if_none_match=ETAG), APIResponse[ModelObject | None]
    )

    # images
    assert_type(await client.images.generate(model=MODEL, prompt="A lighthouse"), ImageGenerateResponse)
    assert_type(
        await client.images.with_response.generate(model=MODEL, prompt="A lighthouse"),
        APIResponse[ImageGenerateResponse],
    )
    assert_type(
        await client.images.edit(model=MODEL, prompt="Make it blue", image="https://example.test/a.png"),
        ImageGenerateResponse,
    )
    assert_type(
        await client.images.with_response.edit(model=MODEL, prompt="Make it blue", image="https://example.test/a.png"),
        APIResponse[ImageGenerateResponse],
    )

    # embeddings
    assert_type(await client.embeddings.create(model=MODEL, input="Hi"), EmbeddingCreateResponse)
    assert_type(
        await client.embeddings.with_response.create(model=MODEL, input="Hi"), APIResponse[EmbeddingCreateResponse]
    )

    # rerank
    assert_type(await client.rerank.create(model=MODEL, query="q", documents=["a", "b"]), RerankCreateResponse)
    assert_type(
        await client.rerank.with_response.create(model=MODEL, query="q", documents=["a", "b"]),
        APIResponse[RerankCreateResponse],
    )

    # audio.speech
    assert_type(await client.audio.speech.create(model=MODEL, input="Hi"), SpeechResponse)
    assert_type(await client.audio.speech.with_response.create(model=MODEL, input="Hi"), APIResponse[SpeechResponse])

    # video.generations
    assert_type(await client.video.generations.create(model=MODEL, prompt="A wave"), VideoGenerationResponse)
    assert_type(
        await client.video.generations.with_response.create(model=MODEL, prompt="A wave"),
        APIResponse[VideoGenerationResponse],
    )

    # inference
    assert_type(await client.inference.run(MODEL, {"inputs": "Hi"}), object)
    assert_type(await client.inference.with_response.run(MODEL, {"inputs": "Hi"}), APIResponse[object])

    # balance, me, health
    assert_type(await client.balance.retrieve(), BalanceResponse)
    assert_type(await client.balance.with_response.retrieve(), APIResponse[BalanceResponse])
    assert_type(await client.me.retrieve(), AccountProfile)
    assert_type(await client.me.with_response.retrieve(), APIResponse[AccountProfile])
    assert_type(await client.health.check(), HealthResponse)
    assert_type(await client.health.with_response.check(), APIResponse[HealthResponse])

    # webhooks: local, and sync on the async client too
    assert_type(client.webhooks.verify_signature(raw_body=BODY, signature_header=HEADER, secret=SECRET), bool)
    assert_type(
        client.webhooks.construct_event(raw_body=BODY.encode(), signature_header=HEADER, secret=SECRET),
        WebhookEvent,
    )


def test_typing_module_imports() -> None:
    assert callable(_check_sync)
    assert inspect.iscoroutinefunction(_check_async)
    assert callable(_check_sync_p3)
    assert inspect.iscoroutinefunction(_check_async_p3)
    assert _etag_or_none() is None
    assert _flag() is False
    assert _any_format() == "json"
    for name in (Stream, AsyncStream, BinaryResponse, AsyncBinaryResponse):
        assert isinstance(name, type)
