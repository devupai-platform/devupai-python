"""Types of the audio endpoints: ``POST /v1/audio/speech`` and ``POST /v1/audio/transcriptions``."""

# No ``from __future__ import annotations`` here: it would empty ``__required_keys__`` of the TypedDicts.

from collections.abc import Sequence
from typing import Literal, Required, TypedDict

from pydantic import Field

from devupai.types._base import DevupModel
from devupai.types.shared import DevUpBilling, FileInput

__all__ = [
    "SpeechCreateParams",
    "SpeechResponse",
    "TranscriptionCreateParams",
    "TranscriptionResponse",
    "TranscriptionVerboseResponse",
    "TranscriptionVerboseResponseWord",
]


# --- requests ---------------------------------------------------------------------------------------------------


class SpeechCreateParams(TypedDict, total=False):
    """The JSON body of ``POST /v1/audio/speech``."""

    model: Required[str]
    input: Required[str]
    voice: str
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
    referenceAudioUrl: str


class TranscriptionCreateParams(TypedDict, total=False):
    """The fields of a transcription, sent as ``multipart/form-data``."""

    file: Required[FileInput]
    model: str
    language: str
    prompt: str
    temperature: float
    timestamp_granularities: Sequence[Literal["word", "segment"]]
    response_format: Literal["json", "verbose_json", "text", "srt", "vtt"]


# --- responses --------------------------------------------------------------------------------------------------


class SpeechResponse(DevupModel):
    """The response of ``POST /v1/audio/speech``."""

    url: str
    """The link to the audio file."""
    devup: DevUpBilling = Field(alias="_devup")
    """The cost of the request and the balance after it (``_devup`` on the wire)."""


class TranscriptionResponse(DevupModel):
    """The JSON response of a transcription (``response_format`` ``json``)."""

    text: str
    """The transcribed text."""
    devup: DevUpBilling = Field(alias="_devup")
    """The cost of the request and the balance after it (``_devup`` on the wire)."""


class TranscriptionVerboseResponseWord(DevupModel):
    """One word of the audio, with its start and end in seconds."""

    word: str | None = None
    start: float | None = None
    end: float | None = None


class TranscriptionVerboseResponse(DevupModel):
    """The JSON response of a transcription with ``response_format`` ``verbose_json``."""

    task: str | None = None
    language: str | None = None
    text: str
    """The transcribed text."""
    duration: float | None = None
    """The length of the audio, in seconds."""
    segments: list[dict[str, object]] | None = None
    words: list[TranscriptionVerboseResponseWord] | None = None
    devup: DevUpBilling = Field(alias="_devup")
    """The cost of the request and the balance after it (``_devup`` on the wire)."""
