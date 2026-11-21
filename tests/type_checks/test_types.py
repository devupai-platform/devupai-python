"""Return types of every public method, on both clients and on every ``with_response`` twin.

``mypy --strict`` and ``pyright`` strict read this module. ``_check_sync`` and ``_check_async`` are never called:
``assert_type`` is checked statically only. ``test_typing_module_imports`` is the one test that runs, so pytest
imports the module.
"""

from __future__ import annotations

import inspect
from typing import assert_type

from devupai import APIResponse, AsyncDevupAI, DevupAI, construct_webhook_event, verify_webhook_signature
from devupai.types import (
    AccountProfile,
    BalanceResponse,
    ChatCompletion,
    EmbeddingCreateResponse,
    HealthResponse,
    ImageGenerateResponse,
    ModelListResponse,
    ModelObject,
    RerankCreateResponse,
    ResponseObject,
    SpeechResponse,
    VideoGenerationResponse,
    WebhookEvent,
)

MODEL = "fixture/model-a"
ETAG = 'W/"fixture-etag"'
BODY = '{"id": "whd_1", "status": "succeeded"}'
HEADER = "t=1800000000,v1=00"
SECRET = "whsec_fixture"


def _etag_or_none() -> str | None:
    """An ``If-None-Match`` value whose static type is ``str | None``."""
    return None


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
    assert _etag_or_none() is None
