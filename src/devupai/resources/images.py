"""Images: generation (``POST /images/generations``), editing (``POST /images/edits``) and generated files."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Unpack, cast, overload

import httpx

from devupai._errors import DevupAPIError
from devupai._response import APIResponse, AsyncBinaryResponse, BinaryResponse
from devupai.resources._base import (
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.resources.shared import MultipartValue, contains_file_input, image_edit_form
from devupai.types import ImageGenerateResponse
from devupai.types._base import parse_model

if TYPE_CHECKING:
    from devupai.types import ImageEditJsonParams, ImageEditParams, ImageGenerateParams

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

#: The message of the error raised by ``images.download`` for a URL it does not fetch. It never contains the URL.
_INVALID_FILE_URL_MESSAGE = "images.download only fetches URLs returned by the API."

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _edit_request(body: Mapping[str, object]) -> tuple[object, list[tuple[str, MultipartValue]] | None]:
    """The JSON body or the form of an image edit: the form when an input image is a file, JSON otherwise."""
    if any(contains_file_input(body.get(name)) for name in _IMAGE_INPUT_FIELDS):
        return None, image_edit_form(body)
    return body, None


def _origin(value: str) -> tuple[str, str, int] | None:
    """The ``(scheme, host, port)`` of an absolute http or https URL without a user name or password, with the default
    port filled in; ``None`` for any other value."""
    try:
        parsed = httpx.URL(value)
    except (httpx.InvalidURL, ValueError, TypeError):
        return None
    if parsed.scheme not in _DEFAULT_PORTS or not parsed.host or parsed.userinfo:
        return None
    port = _DEFAULT_PORTS[parsed.scheme] if parsed.port is None else parsed.port
    return parsed.scheme, parsed.host, port


def _download_url(base_url: str, url: object) -> str:
    """``url`` when it is an absolute URL on the same origin as ``base_url`` (scheme, host and port), without a user
    name or password. Otherwise a ``DevupAPIError`` (code ``invalid_url``) is raised before any request."""
    if isinstance(url, str):
        target = _origin(url)
        if target is not None and target == _origin(base_url):
            return url
    # Raised outside any ``except`` block, so no error that shows the URL is chained to it.
    raise DevupAPIError(0, _INVALID_FILE_URL_MESSAGE, code="invalid_url")


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

    @overload
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> APIResponse[ImageGenerateResponse]: ...

    @overload
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditParams],
    ) -> APIResponse[ImageGenerateResponse]: ...

    @endpoint("POST", _EDIT_PATH, billed=True)
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> APIResponse[ImageGenerateResponse]:
        """Edits one or more images as the prompt describes; see :meth:`Images.edit`."""
        body, form = _edit_request(json_body(params, extra_body))
        response = self._send(
            "POST",
            _EDIT_PATH,
            body=body,
            multipart=form,
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
    ) -> APIResponse[BinaryResponse]:
        """Fetches a generated file; see :meth:`Images.download`."""
        target = _download_url(self._client.base_url, url)
        response = self._send(
            "GET",
            url=target,
            auth=False,
            kind="raw",
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, BinaryResponse(response.response))


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

    @overload
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> ImageGenerateResponse: ...

    @overload
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditParams],
    ) -> ImageGenerateResponse: ...

    @endpoint("POST", _EDIT_PATH, billed=True)
    def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> ImageGenerateResponse:
        """Edits one or more images as the prompt describes (``POST /images/edits``).

        The request takes one of two forms, chosen from the input images (``image``, ``images``, ``mask``,
        ``markup``):

        - When every input is an HTTPS URL or a data URI (a string or ``{"image_url": ...}``), the params are sent as
          JSON, exactly as given.
        - When at least one input is a file (bytes, a path, an open binary file, or a ``(filename, content)`` /
          ``(filename, content, content_type)`` tuple), the params are sent as ``multipart/form-data``: one ``image``
          part per input image, in order; one ``mask`` and one ``markup`` part when given (a file, or a URL or data
          URI as text); every other field as a text part (an object or a list as its JSON text). A path is opened and
          closed by the SDK; an open file is read and left open.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.images.edit(
                model="<model-id>",
                prompt="Replace the sky with a starry night",
                images=["https://example.com/photo.png"],
                mask={"image_url": "https://example.com/mask.png"},
            )

            result = client.images.edit(model="<model-id>", prompt="Add a hat", image=Path("photo.png"))
        """
        # The implementation behind the overloads takes any fields: the call goes through its plain signature.
        edit = cast("Callable[..., APIResponse[ImageGenerateResponse]]", self.with_response.edit)
        return edit(
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
    ) -> BinaryResponse:
        """Fetches a generated image or video file from the URL returned by the API.

        Pass the URL exactly as you received it (for example ``result.data[0].url``). Only URLs on the same origin as
        the client's ``base_url`` (scheme, host and port), without a user name or password, are fetched; any other
        value raises a :class:`~devupai.DevupAPIError` with the code ``invalid_url`` before any request is sent. No
        API key is sent. Send a ``Range`` header in ``extra_headers`` to fetch part of the file.

        The result is a :class:`~devupai.BinaryResponse` with the body unread: read it with ``read()``,
        ``iter_bytes()`` or ``write_to_file()``, and close it (or use it in a ``with`` block).

        Example::

            with client.images.download(result.data[0].url) as file:
                file.write_to_file("image.png")
        """
        return self.with_response.download(
            url, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        ).data


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

    @overload
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> APIResponse[ImageGenerateResponse]: ...

    @overload
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditParams],
    ) -> APIResponse[ImageGenerateResponse]: ...

    @endpoint("POST", _EDIT_PATH, billed=True)
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> APIResponse[ImageGenerateResponse]:
        """Edits one or more images as the prompt describes; see :meth:`AsyncImages.edit`."""
        body, form = _edit_request(json_body(params, extra_body))
        response = await self._send(
            "POST",
            _EDIT_PATH,
            body=body,
            multipart=form,
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
    ) -> APIResponse[AsyncBinaryResponse]:
        """Fetches a generated file; see :meth:`AsyncImages.download`."""
        target = _download_url(self._client.base_url, url)
        response = await self._send(
            "GET",
            url=target,
            auth=False,
            kind="raw",
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        return with_data(response, AsyncBinaryResponse(response.response))


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

    @overload
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditJsonParams],
    ) -> ImageGenerateResponse: ...

    @overload
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: Unpack[ImageEditParams],
    ) -> ImageGenerateResponse: ...

    @endpoint("POST", _EDIT_PATH, billed=True)
    async def edit(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        **params: object,
    ) -> ImageGenerateResponse:
        """Edits one or more images as the prompt describes (``POST /images/edits``).

        The request takes one of two forms, chosen from the input images (``image``, ``images``, ``mask``,
        ``markup``):

        - When every input is an HTTPS URL or a data URI (a string or ``{"image_url": ...}``), the params are sent as
          JSON, exactly as given.
        - When at least one input is a file (bytes, a path, an open binary file, or a ``(filename, content)`` /
          ``(filename, content, content_type)`` tuple), the params are sent as ``multipart/form-data``: one ``image``
          part per input image, in order; one ``mask`` and one ``markup`` part when given (a file, or a URL or data
          URI as text); every other field as a text part (an object or a list as its JSON text). A path is opened and
          closed by the SDK; an open file is read and left open.

        The body fields are passed as keyword arguments. ``extra_body`` adds fields to the body (merged last),
        ``extra_headers`` adds request headers, and ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.images.edit(
                model="<model-id>",
                prompt="Replace the sky with a starry night",
                images=["https://example.com/photo.png"],
            )
        """
        # The implementation behind the overloads takes any fields: the call goes through its plain signature.
        edit = cast("Callable[..., Awaitable[APIResponse[ImageGenerateResponse]]]", self.with_response.edit)
        response = await edit(
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
    ) -> AsyncBinaryResponse:
        """Fetches a generated image or video file from the URL returned by the API.

        Pass the URL exactly as you received it (for example ``result.data[0].url``). Only URLs on the same origin as
        the client's ``base_url`` (scheme, host and port), without a user name or password, are fetched; any other
        value raises a :class:`~devupai.DevupAPIError` with the code ``invalid_url`` before any request is sent. No
        API key is sent. Send a ``Range`` header in ``extra_headers`` to fetch part of the file.

        The result is an :class:`~devupai.AsyncBinaryResponse` with the body unread: read it with ``await read()``,
        ``aiter_bytes()`` or ``await write_to_file()``, and close it (or use it in an ``async with`` block).

        Example::

            async with await client.images.download(result.data[0].url) as file:
                await file.write_to_file("image.png")
        """
        response = await self.with_response.download(
            url, extra_headers=extra_headers, timeout=timeout, max_retries=max_retries
        )
        return response.data
