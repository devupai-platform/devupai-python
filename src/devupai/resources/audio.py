"""Audio: text to speech (``POST /audio/speech``) and speech to text (``POST /audio/transcriptions``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn, Unpack

from devupai._response import APIResponse
from devupai.resources._base import (
    NOT_IMPLEMENTED_MESSAGE,
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.types import SpeechResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI
    from devupai.types import SpeechCreateParams, TranscriptionCreateParams

__all__ = [
    "Audio",
    "AsyncAudio",
    "Speech",
    "AsyncSpeech",
    "SpeechWithResponse",
    "AsyncSpeechWithResponse",
    "Transcriptions",
    "AsyncTranscriptions",
    "TranscriptionsWithResponse",
    "AsyncTranscriptionsWithResponse",
]

_SPEECH_PATH = "/audio/speech"
_TRANSCRIPTIONS_PATH = "/audio/transcriptions"


class SpeechWithResponse(SyncAPIResource):
    """The methods of :class:`Speech`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _SPEECH_PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[SpeechCreateParams],
    ) -> APIResponse[SpeechResponse]:
        """Generates spoken audio from the input text; see :meth:`Speech.create`."""
        response = self._send(
            "POST",
            _SPEECH_PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(SpeechResponse, response.data))


class Speech(SyncAPIResource):
    """Text to speech: ``POST /audio/speech``."""

    @property
    def with_response(self) -> SpeechWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return SpeechWithResponse(self._client)

    @endpoint("POST", _SPEECH_PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[SpeechCreateParams],
    ) -> SpeechResponse:
        """Generates spoken audio from the input text.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            speech = client.audio.speech.create(model="<model-id>", input="Hello")
        """
        return self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data


class TranscriptionsWithResponse(SyncAPIResource):
    """The methods of :class:`Transcriptions`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[TranscriptionCreateParams],
    ) -> NoReturn:
        """Transcribes an audio file; see :meth:`Transcriptions.create`. Available in a later build."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class Transcriptions(SyncAPIResource):
    """Speech to text: ``POST /audio/transcriptions`` (multipart)."""

    @property
    def with_response(self) -> TranscriptionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return TranscriptionsWithResponse(self._client)

    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[TranscriptionCreateParams],
    ) -> NoReturn:
        """Transcribes an audio file.

        The audio file is sent as multipart form data, which is available in a later build: this method raises
        ``NotImplementedError`` before anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class Audio(SyncAPIResource):
    """Audio resources."""

    speech: Speech
    """Text to speech."""
    transcriptions: Transcriptions
    """Speech to text."""

    def __init__(self, client: DevupAI) -> None:
        super().__init__(client)
        self.speech = Speech(client)
        self.transcriptions = Transcriptions(client)


class AsyncSpeechWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncSpeech`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _SPEECH_PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[SpeechCreateParams],
    ) -> APIResponse[SpeechResponse]:
        """Generates spoken audio from the input text; see :meth:`AsyncSpeech.create`."""
        response = await self._send(
            "POST",
            _SPEECH_PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(SpeechResponse, response.data))


class AsyncSpeech(AsyncAPIResource):
    """Text to speech: ``POST /audio/speech``."""

    @property
    def with_response(self) -> AsyncSpeechWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncSpeechWithResponse(self._client)

    @endpoint("POST", _SPEECH_PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[SpeechCreateParams],
    ) -> SpeechResponse:
        """Generates spoken audio from the input text.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            speech = await client.audio.speech.create(model="<model-id>", input="Hello")
        """
        response = await self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data


class AsyncTranscriptionsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncTranscriptions`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[TranscriptionCreateParams],
    ) -> NoReturn:
        """Transcribes an audio file; see :meth:`AsyncTranscriptions.create`. Available in a later build."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncTranscriptions(AsyncAPIResource):
    """Speech to text: ``POST /audio/transcriptions`` (multipart)."""

    @property
    def with_response(self) -> AsyncTranscriptionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncTranscriptionsWithResponse(self._client)

    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[TranscriptionCreateParams],
    ) -> NoReturn:
        """Transcribes an audio file.

        The audio file is sent as multipart form data, which is available in a later build: this method raises
        ``NotImplementedError`` before anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncAudio(AsyncAPIResource):
    """Audio resources."""

    speech: AsyncSpeech
    """Text to speech."""
    transcriptions: AsyncTranscriptions
    """Speech to text."""

    def __init__(self, client: AsyncDevupAI) -> None:
        super().__init__(client)
        self.speech = AsyncSpeech(client)
        self.transcriptions = AsyncTranscriptions(client)
