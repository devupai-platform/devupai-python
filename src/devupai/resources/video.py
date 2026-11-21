"""Video: generation (``POST /video/generations``) and editing (multipart, on the same path)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn, Unpack

from devupai._response import APIResponse
from devupai.resources._base import (
    NOT_IMPLEMENTED_MESSAGE,
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    contains_file_input,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.types import VideoGenerationResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI
    from devupai.types import VideoEditCreateParams, VideoGenerationCreateParams

__all__ = [
    "Video",
    "AsyncVideo",
    "VideoGenerations",
    "AsyncVideoGenerations",
    "VideoGenerationsWithResponse",
    "AsyncVideoGenerationsWithResponse",
    "VideoEdits",
    "AsyncVideoEdits",
    "VideoEditsWithResponse",
    "AsyncVideoEditsWithResponse",
]

_PATH = "/video/generations"


def _check_generation_body(body: Mapping[str, object]) -> None:
    """Raises ``NotImplementedError`` when a field is a file: the multipart form comes in a later build."""
    if any(contains_file_input(value) for value in body.values()):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class VideoGenerationsWithResponse(SyncAPIResource):
    """The methods of :class:`VideoGenerations`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> APIResponse[VideoGenerationResponse]:
        """Generates a video from a prompt; see :meth:`VideoGenerations.create`."""
        body = json_body(params, extra_body)
        _check_generation_body(body)
        response = self._send(
            "POST",
            _PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(VideoGenerationResponse, response.data))


class VideoGenerations(SyncAPIResource):
    """Video generation: ``POST /video/generations`` (JSON)."""

    @property
    def with_response(self) -> VideoGenerationsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return VideoGenerationsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> VideoGenerationResponse:
        """Generates a video from a prompt.

        The media inputs (``image``, ``last_frame_image``, ``reference_images``, ``reference_videos``,
        ``reference_audios``) are HTTPS URLs or data URIs, as strings or as objects that carry one. The params are sent
        as JSON. A file given in any field (bytes, a path or an open file) needs the multipart form, which is
        available in a later build: it raises ``NotImplementedError`` before anything is sent.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            video = client.video.generations.create(model="<model-id>", prompt="A sunrise")
        """
        return self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data


class VideoEditsWithResponse(SyncAPIResource):
    """The methods of :class:`VideoEdits`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[VideoEditCreateParams],
    ) -> NoReturn:
        """Edits a video; see :meth:`VideoEdits.create`. Available in a later build."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class VideoEdits(SyncAPIResource):
    """Video editing: ``POST /video/generations`` (multipart)."""

    @property
    def with_response(self) -> VideoEditsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return VideoEditsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[VideoEditCreateParams],
    ) -> NoReturn:
        """Edits a video, on models that offer video editing.

        The video (and an optional mask) are sent as multipart form data, which is available in a later build: this
        method raises ``NotImplementedError`` before anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class Video(SyncAPIResource):
    """Video resources."""

    generations: VideoGenerations
    """Video generation from a prompt."""
    edits: VideoEdits
    """Video editing."""

    def __init__(self, client: DevupAI) -> None:
        super().__init__(client)
        self.generations = VideoGenerations(client)
        self.edits = VideoEdits(client)


class AsyncVideoGenerationsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncVideoGenerations`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> APIResponse[VideoGenerationResponse]:
        """Generates a video from a prompt; see :meth:`AsyncVideoGenerations.create`."""
        body = json_body(params, extra_body)
        _check_generation_body(body)
        response = await self._send(
            "POST",
            _PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(VideoGenerationResponse, response.data))


class AsyncVideoGenerations(AsyncAPIResource):
    """Video generation: ``POST /video/generations`` (JSON)."""

    @property
    def with_response(self) -> AsyncVideoGenerationsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncVideoGenerationsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> VideoGenerationResponse:
        """Generates a video from a prompt.

        The media inputs (``image``, ``last_frame_image``, ``reference_images``, ``reference_videos``,
        ``reference_audios``) are HTTPS URLs or data URIs, as strings or as objects that carry one. The params are sent
        as JSON. A file given in any field (bytes, a path or an open file) needs the multipart form, which is
        available in a later build: it raises ``NotImplementedError`` before anything is sent.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            video = await client.video.generations.create(model="<model-id>", prompt="A sunrise")
        """
        response = await self.with_response.create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data


class AsyncVideoEditsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncVideoEdits`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[VideoEditCreateParams],
    ) -> NoReturn:
        """Edits a video; see :meth:`AsyncVideoEdits.create`. Available in a later build."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncVideoEdits(AsyncAPIResource):
    """Video editing: ``POST /video/generations`` (multipart)."""

    @property
    def with_response(self) -> AsyncVideoEditsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncVideoEditsWithResponse(self._client)

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **params: Unpack[VideoEditCreateParams],
    ) -> NoReturn:
        """Edits a video, on models that offer video editing.

        The video (and an optional mask) are sent as multipart form data, which is available in a later build: this
        method raises ``NotImplementedError`` before anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncVideo(AsyncAPIResource):
    """Video resources."""

    generations: AsyncVideoGenerations
    """Video generation from a prompt."""
    edits: AsyncVideoEdits
    """Video editing."""

    def __init__(self, client: AsyncDevupAI) -> None:
        super().__init__(client)
        self.generations = AsyncVideoGenerations(client)
        self.edits = AsyncVideoEdits(client)
