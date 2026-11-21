"""Types of the video endpoints: ``POST /v1/video/generations`` (JSON and ``multipart/form-data``) and the video
edit."""

# No ``from __future__ import annotations`` here: it would empty ``__required_keys__`` of the TypedDicts.

from collections.abc import Sequence
from typing import Literal, Required, TypeAlias, TypedDict

from pydantic import Field

from devupai.types._base import DevupModel
from devupai.types.shared import DevUpBilling, FileInput

__all__ = [
    "VideoEditCreateParams",
    "VideoFrameImageObjectParam",
    "VideoFrameImageParam",
    "VideoGenerationCreateFormParams",
    "VideoGenerationCreateParams",
    "VideoGenerationResponse",
    "VideoGenerationResponseData",
    "VideoKeyPoint",
    "VideoReferenceAudioAudioUrlParam",
    "VideoReferenceAudioParam",
    "VideoReferenceAudioUrlParam",
    "VideoReferenceImageImageUrlParam",
    "VideoReferenceImageParam",
    "VideoReferenceImageUrlParam",
    "VideoReferenceVideoParam",
    "VideoReferenceVideoUrlParam",
    "VideoReferenceVideoVideoUrlParam",
    "VideoReferenceVoiceAudioParam",
]


# --- inputs -----------------------------------------------------------------------------------------------------

VideoReferenceVoiceAudioParam: TypeAlias = str
"""The voice of the person in a reference image or video: an audio file as an HTTPS URL or a data URI."""


class VideoFrameImageObjectParam(TypedDict, total=False):
    """A frame image given as an object: ``{"image_url": ...}``, an HTTPS URL or a data URI."""

    image_url: Required[str]


VideoFrameImageParam: TypeAlias = str | VideoFrameImageObjectParam
"""The first or last frame of a video: an HTTPS URL or a data URI, as a string or as ``{"image_url": ...}``."""


class VideoReferenceImageUrlParam(TypedDict, total=False):
    """A reference image given as ``{"url": ...}``."""

    url: Required[str]
    voice_audio: VideoReferenceVoiceAudioParam


class VideoReferenceImageImageUrlParam(TypedDict, total=False):
    """A reference image given as ``{"image_url": ...}``."""

    image_url: Required[str]
    voice_audio: VideoReferenceVoiceAudioParam


VideoReferenceImageParam: TypeAlias = str | VideoReferenceImageUrlParam | VideoReferenceImageImageUrlParam
"""A reference image: an HTTPS URL or a data URI, as a string, as ``{"url": ...}`` or as ``{"image_url": ...}``."""


class VideoReferenceVideoUrlParam(TypedDict, total=False):
    """A reference video given as ``{"url": ...}``."""

    url: Required[str]
    voice_audio: VideoReferenceVoiceAudioParam


class VideoReferenceVideoVideoUrlParam(TypedDict, total=False):
    """A reference video given as ``{"video_url": ...}``."""

    video_url: Required[str]
    voice_audio: VideoReferenceVoiceAudioParam


VideoReferenceVideoParam: TypeAlias = str | VideoReferenceVideoUrlParam | VideoReferenceVideoVideoUrlParam
"""A reference video: an HTTPS URL or a data URI, as a string, as ``{"url": ...}`` or as ``{"video_url": ...}``."""


class VideoReferenceAudioUrlParam(TypedDict, total=False):
    """A reference audio file given as ``{"url": ...}``."""

    url: Required[str]


class VideoReferenceAudioAudioUrlParam(TypedDict, total=False):
    """A reference audio file given as ``{"audio_url": ...}``."""

    audio_url: Required[str]


VideoReferenceAudioParam: TypeAlias = str | VideoReferenceAudioUrlParam | VideoReferenceAudioAudioUrlParam
"""A reference audio file: an HTTPS URL or a data URI, as a string, as ``{"url": ...}`` or as
``{"audio_url": ...}``."""


class VideoKeyPoint(TypedDict, total=False):
    """One key point of a video edit in ``key_points`` mode."""

    x: Required[float]
    y: Required[float]
    type: Required[Literal["positive", "negative"]]


# --- requests ---------------------------------------------------------------------------------------------------


class VideoGenerationCreateParams(TypedDict, total=False):
    """The JSON body of ``POST /v1/video/generations``. Each model lists the options it takes on its model page.

    A ``reference_*`` field takes a list of items, or a single item read as a list of one.
    """

    model: Required[str]
    prompt: Required[str]
    negative_prompt: str
    # Deprecated: use ``duration``.
    num_frames: int
    width: int
    height: int
    seed: int
    duration: int
    aspect_ratio: str
    size: str
    generate_audio: bool
    fps: int
    camera_motion: str
    image: VideoFrameImageParam
    last_frame_image: VideoFrameImageParam
    reference_images: Sequence[VideoReferenceImageParam] | VideoReferenceImageParam
    reference_videos: Sequence[VideoReferenceVideoParam] | VideoReferenceVideoParam
    reference_audios: Sequence[VideoReferenceAudioParam] | VideoReferenceAudioParam
    voice: str
    voice_language: str
    voice_instructions: str
    resolution: str
    response_format: Literal["url", "b64_json"]


class VideoGenerationCreateFormParams(TypedDict, total=False):
    """The fields of a video generation sent as ``multipart/form-data``, to send the frames and reference files as
    files. Each media field is a file, or an HTTPS URL or a data URI as text; a ``reference_*`` field takes one
    item or a list of items, in order."""

    model: Required[str]
    prompt: Required[str]
    image: FileInput | str
    last_frame_image: FileInput | str
    reference_images: FileInput | str | Sequence[FileInput | str]
    reference_videos: FileInput | str | Sequence[FileInput | str]
    reference_audios: FileInput | str | Sequence[FileInput | str]
    voice: str
    voice_language: str
    voice_instructions: str
    duration: int
    aspect_ratio: str
    size: str
    width: int
    height: int
    # Deprecated: use ``duration``.
    num_frames: int
    generate_audio: bool
    fps: int
    camera_motion: str
    seed: int
    negative_prompt: str
    resolution: str
    response_format: Literal["url", "b64_json"]


class VideoEditCreateParams(TypedDict, total=False):
    """The fields of a video edit, sent as ``multipart/form-data``.

    ``key_points`` is a JSON string or a list of :class:`VideoKeyPoint` (sent as a JSON string).
    """

    model: Required[str]
    video: Required[FileInput]
    prompt: str
    mask: FileInput
    mask_url: str
    key_points: str | Sequence[VideoKeyPoint]
    frame_index: int
    auto_trim: bool
    preserve_audio: bool
    background_color: str
    desired_increase: Literal[2, 4]
    output_container_and_codec: str
    response_format: Literal["url", "b64_json"]


# --- responses --------------------------------------------------------------------------------------------------


class VideoGenerationResponseData(DevupModel):
    """The video: ``url`` when ``response_format`` is ``url`` (the default) or the video is larger than the inline
    size limit, ``b64_json`` otherwise."""

    url: str | None = None
    """The link to the video."""
    b64_json: str | None = None
    """The video bytes, base64-encoded."""


class VideoGenerationResponse(DevupModel):
    """The response of a video generation or edit."""

    data: list[VideoGenerationResponseData]
    devup: DevUpBilling = Field(alias="_devup")
    """The cost of the request and the balance after it (``_devup`` on the wire)."""
