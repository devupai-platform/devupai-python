"""Audio: text to speech (``POST /audio/speech``) and speech to text (``POST /audio/transcriptions``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Unpack, overload

from devupai._response import APIResponse
from devupai.resources._base import (
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.resources.shared import MultipartPart, fixed_form
from devupai.types import SpeechResponse, TranscriptionResponse, TranscriptionVerboseResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI
    from devupai.types import FileInput, SpeechCreateParams

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

TextTranscriptionFormat = Literal["text", "srt", "vtt"]
"""The transcription formats returned as plain text (a ``str``)."""
TranscriptionFormat = Literal["json", "verbose_json", "text", "srt", "vtt"]
"""Every transcription ``response_format``."""
TimestampGranularity = Literal["word", "segment"]
"""A ``timestamp_granularities`` value."""

_TEXT_FORMATS = frozenset({"text", "srt", "vtt"})


def _transcription_parts(
    *,
    file: FileInput,
    model: str | None,
    language: str | None,
    response_format: str | None,
    prompt: str | None,
    temperature: float | None,
    timestamp_granularities: Sequence[str] | None,
) -> list[MultipartPart]:
    """The form parts of a transcription: ``file``, then the set fields in a fixed order, then one
    ``timestamp_granularities[]`` part per value. A field set to ``None`` is left out."""
    return fixed_form(
        (
            ("file", file),
            ("model", model),
            ("language", language),
            ("response_format", response_format),
            ("prompt", prompt),
            ("temperature", temperature),
            ("timestamp_granularities[]", timestamp_granularities),
        ),
        file_fields=("file",),
    )


def _transcription_kind(response_format: str | None) -> Literal["json", "text"]:
    return "text" if response_format in _TEXT_FORMATS else "json"


def _transcription_result(
    response: APIResponse[object], response_format: str | None
) -> APIResponse[TranscriptionResponse] | APIResponse[TranscriptionVerboseResponse] | APIResponse[str]:
    if response_format in _TEXT_FORMATS:
        return with_data(response, str(response.data))
    if response_format == "verbose_json":
        return with_data(response, parse_model(TranscriptionVerboseResponse, response.data))
    return with_data(response, parse_model(TranscriptionResponse, response.data))


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

    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TextTranscriptionFormat,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[str]: ...
    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["verbose_json"],
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionVerboseResponse]: ...
    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["json"] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionResponse]: ...
    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionResponse] | APIResponse[TranscriptionVerboseResponse] | APIResponse[str]: ...
    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionResponse] | APIResponse[TranscriptionVerboseResponse] | APIResponse[str]:
        """Transcribes an audio file; see :meth:`Transcriptions.create`."""
        parts = _transcription_parts(
            file=file,
            model=model,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
        response = self._send(
            "POST",
            _TRANSCRIPTIONS_PATH,
            multipart=parts,
            retry=billed_post_retry(),
            kind=_transcription_kind(response_format),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return _transcription_result(response, response_format)


class Transcriptions(SyncAPIResource):
    """Speech to text: ``POST /audio/transcriptions`` (multipart)."""

    @property
    def with_response(self) -> TranscriptionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return TranscriptionsWithResponse(self._client)

    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TextTranscriptionFormat,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> str: ...
    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["verbose_json"],
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionVerboseResponse: ...
    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["json"] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionResponse: ...
    @overload
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionResponse | TranscriptionVerboseResponse | str: ...
    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionResponse | TranscriptionVerboseResponse | str:
        """Transcribes an audio file.

        The file and the fields are sent as ``multipart/form-data``: ``file`` (bytes, a path, an open binary file
        or a ``(filename, content[, content_type])`` tuple), then ``model``, ``language``, ``response_format``,
        ``prompt``, ``temperature`` and one ``timestamp_granularities[]`` part per value. A field left as ``None``
        is not sent. A path is opened and closed by the SDK; an open file is read and left open.

        The result follows ``response_format``: ``text``, ``srt`` and ``vtt`` return a ``str``; ``verbose_json``
        returns a :class:`~devupai.types.TranscriptionVerboseResponse`; ``json`` (the default) returns a
        :class:`~devupai.types.TranscriptionResponse`.

        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            from pathlib import Path

            result = client.audio.transcriptions.create(file=Path("clip.wav"), model="<model-id>")
            print(result.text)
        """
        return self.with_response.create(
            file=file,
            model=model,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        ).data


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

    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TextTranscriptionFormat,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[str]: ...
    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["verbose_json"],
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionVerboseResponse]: ...
    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["json"] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionResponse]: ...
    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionResponse] | APIResponse[TranscriptionVerboseResponse] | APIResponse[str]: ...
    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[TranscriptionResponse] | APIResponse[TranscriptionVerboseResponse] | APIResponse[str]:
        """Transcribes an audio file; see :meth:`AsyncTranscriptions.create`."""
        parts = _transcription_parts(
            file=file,
            model=model,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
        response = await self._send(
            "POST",
            _TRANSCRIPTIONS_PATH,
            multipart=parts,
            retry=billed_post_retry(),
            kind=_transcription_kind(response_format),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return _transcription_result(response, response_format)


class AsyncTranscriptions(AsyncAPIResource):
    """Speech to text: ``POST /audio/transcriptions`` (multipart)."""

    @property
    def with_response(self) -> AsyncTranscriptionsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncTranscriptionsWithResponse(self._client)

    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TextTranscriptionFormat,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> str: ...
    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["verbose_json"],
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionVerboseResponse: ...
    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: Literal["json"] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionResponse: ...
    @overload
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionResponse | TranscriptionVerboseResponse | str: ...
    @endpoint("POST", _TRANSCRIPTIONS_PATH, billed=True)
    async def create(
        self,
        *,
        file: FileInput,
        model: str | None = None,
        language: str | None = None,
        response_format: TranscriptionFormat | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TranscriptionResponse | TranscriptionVerboseResponse | str:
        """Transcribes an audio file.

        The file and the fields are sent as ``multipart/form-data``: ``file`` (bytes, a path, an open binary file
        or a ``(filename, content[, content_type])`` tuple), then ``model``, ``language``, ``response_format``,
        ``prompt``, ``temperature`` and one ``timestamp_granularities[]`` part per value. A field left as ``None``
        is not sent. A path is opened and closed by the SDK; an open file is read and left open.

        The result follows ``response_format``: ``text``, ``srt`` and ``vtt`` return a ``str``; ``verbose_json``
        returns a :class:`~devupai.types.TranscriptionVerboseResponse`; ``json`` (the default) returns a
        :class:`~devupai.types.TranscriptionResponse`.

        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            from pathlib import Path

            result = await client.audio.transcriptions.create(file=Path("clip.wav"), model="<model-id>")
            print(result.text)
        """
        response = await self.with_response.create(
            file=file,
            model=model,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        )
        return response.data


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
