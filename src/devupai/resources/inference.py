"""Native inference: ``POST /inference/{model_path}``, a model-specific payload sent as it is."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, overload

from devupai._response import APIResponse, AsyncBinaryResponse, BinaryResponse
from devupai.resources._base import (
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    encode_model_path,
    endpoint,
    json_body,
    options,
    with_data,
)
from devupai.resources.shared import MultipartPart, contains_file_input, every_field_form

__all__ = [
    "NativeInference",
    "AsyncNativeInference",
    "NativeInferenceWithResponse",
    "AsyncNativeInferenceWithResponse",
]

_PATH_TEMPLATE = "/inference/{model_path}"
_PAYLOAD_MESSAGE = "The payload of inference.run must be a mapping of field names to values."
_RESPONSE_TYPE_MESSAGE = 'response_type must be "json" or "binary".'

ResponseType = Literal["json", "binary"]
"""How the answer of :meth:`NativeInference.run` is read: parsed as JSON, or as a file."""

_RESPONSE_TYPES: frozenset[str] = frozenset({"json", "binary"})


def _is_binary(response_type: ResponseType) -> bool:
    """Whether the answer is read as a file. Any value other than the strings ``"json"`` and ``"binary"`` raises
    ``ValueError``, so a typo is reported before anything is sent."""
    if not isinstance(response_type, str) or response_type not in _RESPONSE_TYPES:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(_RESPONSE_TYPE_MESSAGE)
    return response_type == "binary"


def _prepare(
    model_path: str,
    payload: Mapping[str, object],
    extra_body: Mapping[str, object] | None,
) -> tuple[str, dict[str, object] | None, list[MultipartPart] | None]:
    """The path of a call, and its JSON body or, when a field is a file (or a list contains one), the multipart
    parts of every field."""
    if not isinstance(payload, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(_PAYLOAD_MESSAGE)
    path = f"/inference/{encode_model_path(model_path)}"
    body = json_body(payload, extra_body)
    if any(contains_file_input(value) for value in body.values()):
        return path, None, every_field_form(body)
    return path, body, None


class NativeInferenceWithResponse(SyncAPIResource):
    """The methods of :class:`NativeInference`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: Literal["binary"],
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[BinaryResponse]: ...
    @overload
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: Literal["json"] = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[object]: ...
    @overload
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[object] | APIResponse[BinaryResponse]: ...
    @endpoint("POST", _PATH_TEMPLATE, billed=True)
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[object] | APIResponse[BinaryResponse]:
        """Sends a model-specific payload to a model; see :meth:`NativeInference.run`."""
        binary = _is_binary(response_type)
        path, body, parts = _prepare(model_path, payload, extra_body)
        response = self._send(
            "POST",
            path,
            body=body,
            multipart=parts,
            retry=billed_post_retry(),
            kind="raw" if binary else "json",
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        if binary:
            return with_data(response, BinaryResponse(response.response))
        return response


class NativeInference(SyncAPIResource):
    """Native inference: ``POST /inference/{model_path}``."""

    @property
    def with_response(self) -> NativeInferenceWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return NativeInferenceWithResponse(self._client)

    @overload
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: Literal["binary"],
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> BinaryResponse: ...
    @overload
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> object: ...
    @endpoint("POST", _PATH_TEMPLATE, billed=True)
    def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> object:
        """Sends a model-specific payload to a model and returns its result.

        ``model_path`` is the model id: leading slashes are removed, and an id that is already percent-encoded is
        decoded first, so it is sent as one path segment encoded exactly once.

        The payload is sent as JSON, as it is. When a field is a file (bytes, a path, an open binary file or a
        ``(filename, content[, content_type])`` tuple), or a list contains one, the payload is sent as
        ``multipart/form-data`` instead: a file is a file part, a ``str`` a text part, a list gives one part per item
        under the field name, a mapping is sent as its JSON text and a number or a boolean as text. A path is opened
        and closed by the SDK; an open file is read and left open.

        The answer is parsed as JSON. With ``response_type="binary"``, for a model that answers with a file (an
        image, an audio clip or a video), it is returned as a :class:`~devupai.BinaryResponse` with its body not
        read yet. ``response_type`` is the string ``"json"`` or ``"binary"``; any other value raises ``ValueError``
        before anything is sent.

        ``extra_body`` adds fields to the payload (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.inference.run("<model-id>", {"input": "Hello"})

            with client.inference.run("<model-id>", {"prompt": "A sunrise"}, response_type="binary") as image:
                image.write_to_file("sunrise.png")
        """
        return self.with_response.run(
            model_path,
            payload,
            response_type=response_type,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
        ).data


class AsyncNativeInferenceWithResponse(AsyncAPIResource):
    """The methods of :class:`AsyncNativeInference`, returning the whole :class:`~devupai.APIResponse`."""

    @overload
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: Literal["binary"],
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[AsyncBinaryResponse]: ...
    @overload
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: Literal["json"] = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[object]: ...
    @overload
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[object] | APIResponse[AsyncBinaryResponse]: ...
    @endpoint("POST", _PATH_TEMPLATE, billed=True)
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> APIResponse[object] | APIResponse[AsyncBinaryResponse]:
        """Sends a model-specific payload to a model; see :meth:`AsyncNativeInference.run`."""
        binary = _is_binary(response_type)
        path, body, parts = _prepare(model_path, payload, extra_body)
        response = await self._send(
            "POST",
            path,
            body=body,
            multipart=parts,
            retry=billed_post_retry(),
            kind="raw" if binary else "json",
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )
        if binary:
            return with_data(response, AsyncBinaryResponse(response.response))
        return response


class AsyncNativeInference(AsyncAPIResource):
    """Native inference: ``POST /inference/{model_path}``."""

    @property
    def with_response(self) -> AsyncNativeInferenceWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncNativeInferenceWithResponse(self._client)

    @overload
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: Literal["binary"],
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> AsyncBinaryResponse: ...
    @overload
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> object: ...
    @endpoint("POST", _PATH_TEMPLATE, billed=True)
    async def run(
        self,
        model_path: str,
        payload: Mapping[str, object],
        *,
        response_type: ResponseType = "json",
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> object:
        """Sends a model-specific payload to a model and returns its result.

        ``model_path`` is the model id: leading slashes are removed, and an id that is already percent-encoded is
        decoded first, so it is sent as one path segment encoded exactly once.

        The payload is sent as JSON, as it is. When a field is a file (bytes, a path, an open binary file or a
        ``(filename, content[, content_type])`` tuple), or a list contains one, the payload is sent as
        ``multipart/form-data`` instead: a file is a file part, a ``str`` a text part, a list gives one part per item
        under the field name, a mapping is sent as its JSON text and a number or a boolean as text. A path is opened
        and closed by the SDK; an open file is read and left open.

        The answer is parsed as JSON. With ``response_type="binary"``, for a model that answers with a file (an
        image, an audio clip or a video), it is returned as an :class:`~devupai.AsyncBinaryResponse` with its body
        not read yet. ``response_type`` is the string ``"json"`` or ``"binary"``; any other value raises
        ``ValueError`` before anything is sent.

        ``extra_body`` adds fields to the payload (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.inference.run("<model-id>", {"input": "Hello"})

            async with await client.inference.run(
                "<model-id>", {"prompt": "A sunrise"}, response_type="binary"
            ) as image:
                await image.write_to_file("sunrise.png")
        """
        response = await self.with_response.run(
            model_path,
            payload,
            response_type=response_type,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body,
        )
        return response.data
