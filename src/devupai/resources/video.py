"""Video: generation (``POST /video/generations``) and editing (multipart, on the same path)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Unpack, cast, overload

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
from devupai.resources.shared import MultipartValue, contains_file_input, every_field_form, fixed_form, json_text
from devupai.types import VideoGenerationResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai._client import AsyncDevupAI, DevupAI
    from devupai.types import (
        FileInput,
        VideoGenerationCreateFormParams,
        VideoGenerationCreateParams,
        VideoKeyPoint,
    )

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

#: The file fields of a video edit.
_EDIT_FILE_FIELDS = ("video", "mask")


def _generation_request(body: Mapping[str, object]) -> tuple[object, list[tuple[str, MultipartValue]] | None]:
    """The JSON body or the form of a video generation: the form, with every field, when a field is a file (or a
    list with a file in it); JSON otherwise."""
    if any(contains_file_input(value) for value in body.values()):
        return None, every_field_form(body)
    return body, None


def _edit_form(
    *,
    video: object,
    model: object,
    prompt: object,
    mask: object,
    mask_url: object,
    key_points: object,
    frame_index: object,
    auto_trim: object,
    preserve_audio: object,
    background_color: object,
    desired_increase: object,
    output_container_and_codec: object,
    response_format: object,
) -> list[tuple[str, MultipartValue]]:
    """The form of a video edit: exactly these fields, in this order. ``key_points`` given as a list is sent as its
    JSON text, and as a string it is sent as it is."""
    points = key_points if key_points is None or isinstance(key_points, str) else json_text(key_points)
    return fixed_form(
        [
            ("video", video),
            ("model", model),
            ("prompt", prompt),
            ("mask", mask),
            ("mask_url", mask_url),
            ("key_points", points),
            ("frame_index", frame_index),
            ("auto_trim", auto_trim),
            ("preserve_audio", preserve_audio),
            ("background_color", background_color),
            ("desired_increase", desired_increase),
            ("output_container_and_codec", output_container_and_codec),
            ("response_format", response_format),
        ],
        file_fields=_EDIT_FILE_FIELDS,
    )


class VideoGenerationsWithResponse(SyncAPIResource):
    """The methods of :class:`VideoGenerations`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> APIResponse[VideoGenerationResponse]: ...

    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateFormParams],
    ) -> APIResponse[VideoGenerationResponse]: ...

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> APIResponse[VideoGenerationResponse]:
        """Generates a video from a prompt; see :meth:`VideoGenerations.create`."""
        body, form = _generation_request(json_body(params, extra_body))
        response = self._send(
            "POST",
            _PATH,
            body=body,
            multipart=form,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(VideoGenerationResponse, response.data))


class VideoGenerations(SyncAPIResource):
    """Video generation: ``POST /video/generations`` (JSON, or multipart when a media input is a file)."""

    @property
    def with_response(self) -> VideoGenerationsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return VideoGenerationsWithResponse(self._client)

    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> VideoGenerationResponse: ...

    @overload
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateFormParams],
    ) -> VideoGenerationResponse: ...

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> VideoGenerationResponse:
        """Generates a video from a prompt.

        The request takes one of two forms, chosen from the fields:

        - When no field is a file, the params are sent as JSON. The media inputs (``image``, ``last_frame_image``,
          ``reference_images``, ``reference_videos``, ``reference_audios``) are then HTTPS URLs or data URIs, as
          strings or as objects that carry one.
        - When a field is a file (bytes, a path, an open binary file, or a ``(filename, content)`` /
          ``(filename, content, content_type)`` tuple), or a list with a file in it, the params are sent as
          ``multipart/form-data``. Every field is sent: a file as a file part, a string as a text part, a list as one
          part per item under the plain field name, an object as its JSON text, and any other value as text
          (``duration=8`` is ``"8"``, ``generate_audio=True`` is ``"true"``). A path is opened and closed by the SDK;
          an open file is read and left open.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            video = client.video.generations.create(model="<model-id>", prompt="A sunrise")

            video = client.video.generations.create(
                model="<model-id>",
                prompt="The person in image 1 walks towards the camera",
                image=Path("first-frame.png"),
                reference_images=[Path("person.png"), "https://example.com/background.png"],
                duration=8,
            )
        """
        # The implementation behind the overloads takes any fields: the call goes through its plain signature.
        create = cast("Callable[..., APIResponse[VideoGenerationResponse]]", self.with_response.create)
        return create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data


class VideoEditsWithResponse(SyncAPIResource):
    """The methods of :class:`VideoEdits`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    def create(
        self,
        *,
        model: str,
        video: FileInput,
        prompt: str | None = None,
        mask: FileInput | None = None,
        mask_url: str | None = None,
        key_points: str | Sequence[VideoKeyPoint] | None = None,
        frame_index: int | None = None,
        auto_trim: bool | None = None,
        preserve_audio: bool | None = None,
        background_color: str | None = None,
        desired_increase: Literal[2, 4] | None = None,
        output_container_and_codec: str | None = None,
        response_format: Literal["url", "b64_json"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[VideoGenerationResponse]:
        """Edits a video; see :meth:`VideoEdits.create`."""
        form = _edit_form(
            video=video,
            model=model,
            prompt=prompt,
            mask=mask,
            mask_url=mask_url,
            key_points=key_points,
            frame_index=frame_index,
            auto_trim=auto_trim,
            preserve_audio=preserve_audio,
            background_color=background_color,
            desired_increase=desired_increase,
            output_container_and_codec=output_container_and_codec,
            response_format=response_format,
        )
        response = self._send(
            "POST",
            _PATH,
            multipart=form,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(VideoGenerationResponse, response.data))


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
        model: str,
        video: FileInput,
        prompt: str | None = None,
        mask: FileInput | None = None,
        mask_url: str | None = None,
        key_points: str | Sequence[VideoKeyPoint] | None = None,
        frame_index: int | None = None,
        auto_trim: bool | None = None,
        preserve_audio: bool | None = None,
        background_color: str | None = None,
        desired_increase: Literal[2, 4] | None = None,
        output_container_and_codec: str | None = None,
        response_format: Literal["url", "b64_json"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> VideoGenerationResponse:
        """Edits a video, on models that offer video editing.

        The video (and an optional mask) are uploaded as ``multipart/form-data``, each as bytes, a path, an open
        binary file, or a ``(filename, content)`` / ``(filename, content, content_type)`` tuple. The form carries
        exactly these fields, in this order: ``video``, ``model``, ``prompt``, ``mask``, ``mask_url``, ``key_points``,
        ``frame_index``, ``auto_trim``, ``preserve_audio``, ``background_color``, ``desired_increase``,
        ``output_container_and_codec``, ``response_format``; a field left as ``None`` is not sent. ``key_points`` is a
        JSON string, or a list of points sent as its JSON text. Any other keyword raises ``TypeError``.

        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.video.edits.create(
                model="<model-id>",
                video=Path("clip.mp4"),
                prompt="Make the sky blue",
            )
        """
        return self.with_response.create(
            model=model,
            video=video,
            prompt=prompt,
            mask=mask,
            mask_url=mask_url,
            key_points=key_points,
            frame_index=frame_index,
            auto_trim=auto_trim,
            preserve_audio=preserve_audio,
            background_color=background_color,
            desired_increase=desired_increase,
            output_container_and_codec=output_container_and_codec,
            response_format=response_format,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        ).data


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

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> APIResponse[VideoGenerationResponse]: ...

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateFormParams],
    ) -> APIResponse[VideoGenerationResponse]: ...

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> APIResponse[VideoGenerationResponse]:
        """Generates a video from a prompt; see :meth:`AsyncVideoGenerations.create`."""
        body, form = _generation_request(json_body(params, extra_body))
        response = await self._send(
            "POST",
            _PATH,
            body=body,
            multipart=form,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(VideoGenerationResponse, response.data))


class AsyncVideoGenerations(AsyncAPIResource):
    """Video generation: ``POST /video/generations`` (JSON, or multipart when a media input is a file)."""

    @property
    def with_response(self) -> AsyncVideoGenerationsWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncVideoGenerationsWithResponse(self._client)

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateParams],
    ) -> VideoGenerationResponse: ...

    @overload
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[VideoGenerationCreateFormParams],
    ) -> VideoGenerationResponse: ...

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> VideoGenerationResponse:
        """Generates a video from a prompt.

        The request takes one of two forms, chosen from the fields:

        - When no field is a file, the params are sent as JSON. The media inputs (``image``, ``last_frame_image``,
          ``reference_images``, ``reference_videos``, ``reference_audios``) are then HTTPS URLs or data URIs, as
          strings or as objects that carry one.
        - When a field is a file (bytes, a path, an open binary file, or a ``(filename, content)`` /
          ``(filename, content, content_type)`` tuple), or a list with a file in it, the params are sent as
          ``multipart/form-data``. Every field is sent: a file as a file part, a string as a text part, a list as one
          part per item under the plain field name, an object as its JSON text, and any other value as text
          (``duration=8`` is ``"8"``, ``generate_audio=True`` is ``"true"``). A path is opened and closed by the SDK;
          an open file is read and left open.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            video = await client.video.generations.create(model="<model-id>", prompt="A sunrise")
        """
        # The implementation behind the overloads takes any fields: the call goes through its plain signature.
        create = cast("Callable[..., Awaitable[APIResponse[VideoGenerationResponse]]]", self.with_response.create)
        response = await create(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data


class AsyncVideoEditsWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncVideoEdits`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _PATH, billed=True)
    async def create(
        self,
        *,
        model: str,
        video: FileInput,
        prompt: str | None = None,
        mask: FileInput | None = None,
        mask_url: str | None = None,
        key_points: str | Sequence[VideoKeyPoint] | None = None,
        frame_index: int | None = None,
        auto_trim: bool | None = None,
        preserve_audio: bool | None = None,
        background_color: str | None = None,
        desired_increase: Literal[2, 4] | None = None,
        output_container_and_codec: str | None = None,
        response_format: Literal["url", "b64_json"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> APIResponse[VideoGenerationResponse]:
        """Edits a video; see :meth:`AsyncVideoEdits.create`."""
        form = _edit_form(
            video=video,
            model=model,
            prompt=prompt,
            mask=mask,
            mask_url=mask_url,
            key_points=key_points,
            frame_index=frame_index,
            auto_trim=auto_trim,
            preserve_audio=preserve_audio,
            background_color=background_color,
            desired_increase=desired_increase,
            output_container_and_codec=output_container_and_codec,
            response_format=response_format,
        )
        response = await self._send(
            "POST",
            _PATH,
            multipart=form,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(VideoGenerationResponse, response.data))


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
        model: str,
        video: FileInput,
        prompt: str | None = None,
        mask: FileInput | None = None,
        mask_url: str | None = None,
        key_points: str | Sequence[VideoKeyPoint] | None = None,
        frame_index: int | None = None,
        auto_trim: bool | None = None,
        preserve_audio: bool | None = None,
        background_color: str | None = None,
        desired_increase: Literal[2, 4] | None = None,
        output_container_and_codec: str | None = None,
        response_format: Literal["url", "b64_json"] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> VideoGenerationResponse:
        """Edits a video, on models that offer video editing.

        The video (and an optional mask) are uploaded as ``multipart/form-data``, each as bytes, a path, an open
        binary file, or a ``(filename, content)`` / ``(filename, content, content_type)`` tuple. The form carries
        exactly these fields, in this order: ``video``, ``model``, ``prompt``, ``mask``, ``mask_url``, ``key_points``,
        ``frame_index``, ``auto_trim``, ``preserve_audio``, ``background_color``, ``desired_increase``,
        ``output_container_and_codec``, ``response_format``; a field left as ``None`` is not sent. ``key_points`` is a
        JSON string, or a list of points sent as its JSON text. Any other keyword raises ``TypeError``.

        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.video.edits.create(
                model="<model-id>",
                video=Path("clip.mp4"),
                prompt="Make the sky blue",
            )
        """
        response = await self.with_response.create(
            model=model,
            video=video,
            prompt=prompt,
            mask=mask,
            mask_url=mask_url,
            key_points=key_points,
            frame_index=frame_index,
            auto_trim=auto_trim,
            preserve_audio=preserve_audio,
            background_color=background_color,
            desired_increase=desired_increase,
            output_container_and_codec=output_container_and_codec,
            response_format=response_format,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        )
        return response.data


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
