"""Images: generation (``POST /images/generations``), editing (``POST /images/edits``) and generated files."""

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
from devupai.types import ImageGenerateResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai.types import ImageEditJsonParams, ImageGenerateParams

__all__ = [
    "Images",
    "AsyncImages",
    "ImagesWithResponse",
    "AsyncImagesWithResponse",
]

_GENERATE_PATH = "/images/generations"
_EDIT_PATH = "/images/edits"
# A generated file is fetched from the absolute URL the API returned, so the template is the whole URL.
_DOWNLOAD_PATH = "{url}"

#: The request fields of an image edit that carry input images (``images`` is the list form of ``image``).
_IMAGE_INPUT_FIELDS = ("image", "images", "mask", "markup")


def _check_edit_body(body: Mapping[str, object]) -> None:
    """Raises ``NotImplementedError`` when an input image is a file: the multipart form comes in a later build."""
    if any(contains_file_input(body.get(name)) for name in _IMAGE_INPUT_FIELDS):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class ImagesWithResponse(SyncAPIResource):
    """The methods of :class:`Images`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _GENERATE_PATH, billed=True)
    def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageGenerateParams],
    ) -> APIResponse[ImageGenerateResponse]:
        """Generates images from a prompt; see :meth:`Images.generate`."""
        response = self._send(
            "POST",
            _GENERATE_PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ImageGenerateResponse, response.data))

    @endpoint("POST", _EDIT_PATH, billed=True)
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> APIResponse[ImageGenerateResponse]:
        """Edits one or more images as the prompt describes; see :meth:`Images.edit`."""
        body = json_body(params, extra_body)
        _check_edit_body(body)
        response = self._send(
            "POST",
            _EDIT_PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ImageGenerateResponse, response.data))

    @endpoint("GET", _DOWNLOAD_PATH, billed=False)
    def download(
        self,
        url: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches a generated file; see :meth:`Images.download`. Available in a later build."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class Images(SyncAPIResource):
    """Images: generation, editing, and fetching generated files."""

    @property
    def with_response(self) -> ImagesWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return ImagesWithResponse(self._client)

    @endpoint("POST", _GENERATE_PATH, billed=True)
    def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageGenerateParams],
    ) -> ImageGenerateResponse:
        """Generates images from a prompt (``POST /images/generations``, JSON).

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.images.generate(model="<model-id>", prompt="A red bicycle")
        """
        return self.with_response.generate(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data

    @endpoint("POST", _EDIT_PATH, billed=True)
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> ImageGenerateResponse:
        """Edits one or more images as the prompt describes (``POST /images/edits``, JSON).

        The input images (``image``, ``images``, ``mask``, ``markup``) are HTTPS URLs or data URIs, as strings or as
        ``{"image_url": ...}`` objects. The params are sent as JSON, exactly as given. A file given as an input image
        (bytes, a path or an open file) needs the multipart form, which is available in a later build: it raises
        ``NotImplementedError`` before anything is sent.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.images.edit(
                model="<model-id>",
                prompt="Replace the sky with a starry night",
                images=["https://example.com/photo.png"],
                mask={"image_url": "https://example.com/mask.png"},
            )
        """
        return self.with_response.edit(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        ).data

    @endpoint("GET", _DOWNLOAD_PATH, billed=False)
    def download(
        self,
        url: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches a generated image or video file from the URL returned by the API.

        The answer is a file, which is available in a later build: this method raises ``NotImplementedError`` before
        anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncImagesWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncImages`, returning the whole :class:`~devupai.APIResponse`."""

    @endpoint("POST", _GENERATE_PATH, billed=True)
    async def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageGenerateParams],
    ) -> APIResponse[ImageGenerateResponse]:
        """Generates images from a prompt; see :meth:`AsyncImages.generate`."""
        response = await self._send(
            "POST",
            _GENERATE_PATH,
            body=json_body(params, extra_body),
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ImageGenerateResponse, response.data))

    @endpoint("POST", _EDIT_PATH, billed=True)
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> APIResponse[ImageGenerateResponse]:
        """Edits one or more images as the prompt describes; see :meth:`AsyncImages.edit`."""
        body = json_body(params, extra_body)
        _check_edit_body(body)
        response = await self._send(
            "POST",
            _EDIT_PATH,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, parse_model(ImageGenerateResponse, response.data))

    @endpoint("GET", _DOWNLOAD_PATH, billed=False)
    async def download(
        self,
        url: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches a generated file; see :meth:`AsyncImages.download`. Available in a later build."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class AsyncImages(AsyncAPIResource):
    """Images: generation, editing, and fetching generated files."""

    @property
    def with_response(self) -> AsyncImagesWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncImagesWithResponse(self._client)

    @endpoint("POST", _GENERATE_PATH, billed=True)
    async def generate(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageGenerateParams],
    ) -> ImageGenerateResponse:
        """Generates images from a prompt (``POST /images/generations``, JSON).

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.images.generate(model="<model-id>", prompt="A red bicycle")
        """
        response = await self.with_response.generate(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data

    @endpoint("POST", _EDIT_PATH, billed=True)
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> ImageGenerateResponse:
        """Edits one or more images as the prompt describes (``POST /images/edits``, JSON).

        The input images (``image``, ``images``, ``mask``, ``markup``) are HTTPS URLs or data URIs, as strings or as
        ``{"image_url": ...}`` objects. The params are sent as JSON, exactly as given. A file given as an input image
        (bytes, a path or an open file) needs the multipart form, which is available in a later build: it raises
        ``NotImplementedError`` before anything is sent.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the JSON body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.images.edit(
                model="<model-id>",
                prompt="Replace the sky with a starry night",
                images=["https://example.com/photo.png"],
            )
        """
        response = await self.with_response.edit(
            extra_headers=extra_headers, timeout=timeout, max_retries=max_retries, extra_body=extra_body, **params
        )
        return response.data

    @endpoint("GET", _DOWNLOAD_PATH, billed=False)
    async def download(
        self,
        url: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> NoReturn:
        """Fetches a generated image or video file from the URL returned by the API.

        The answer is a file, which is available in a later build: this method raises ``NotImplementedError`` before
        anything is sent.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
