"""Native inference: ``POST /inference/{model_path}``, a model-specific payload sent as it is."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from devupai._response import APIResponse
from devupai.resources._base import (
    NOT_IMPLEMENTED_MESSAGE,
    AsyncAPIResource,
    SyncAPIResource,
    billed_post_retry,
    contains_file_input,
    encode_model_path,
    endpoint,
    is_file_input,
    json_body,
    options,
)

__all__ = [
    "NativeInference",
    "AsyncNativeInference",
    "NativeInferenceWithResponse",
    "AsyncNativeInferenceWithResponse",
]

_PATH_TEMPLATE = "/inference/{model_path}"

ResponseType = Literal["json", "binary"]
"""How the answer of :meth:`NativeInference.run` is read: parsed as JSON, or as a file."""


def _prepare(
    model_path: str,
    payload: Mapping[str, object],
    response_type: ResponseType,
    extra_body: Mapping[str, object] | None,
) -> tuple[str, dict[str, object]]:
    """The path and JSON body of a call, after the checks for the forms that come in a later build."""
    if response_type == "binary" or is_file_input(payload):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
    body = json_body(payload, extra_body)
    if any(contains_file_input(value) for value in body.values()):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
    return f"/inference/{encode_model_path(model_path)}", body


class NativeInferenceWithResponse(SyncAPIResource):
    """The methods of :class:`NativeInference`, returning the whole :class:`~devupai.APIResponse`."""

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
    ) -> APIResponse[object]:
        """Sends a model-specific payload to a model; see :meth:`NativeInference.run`."""
        path, body = _prepare(model_path, payload, response_type, extra_body)
        return self._send(
            "POST",
            path,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )


class NativeInference(SyncAPIResource):
    """Native inference: ``POST /inference/{model_path}``."""

    @property
    def with_response(self) -> NativeInferenceWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return NativeInferenceWithResponse(self._client)

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
        """Sends a model-specific payload to a model and returns its result, parsed from JSON.

        The payload is sent as JSON, as it is. ``model_path`` is the model id: leading slashes are removed, and an id
        that is already percent-encoded is decoded first, so it is sent as one path segment encoded exactly once.

        ``response_type="binary"`` (an answer that is a file) and a file in the payload (bytes, a path or an open
        file, which needs the multipart form) are available in a later build: they raise ``NotImplementedError``
        before anything is sent.

        ``extra_body`` adds fields to the JSON body (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = client.inference.run("<model-id>", {"input": "Hello"})
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
    ) -> APIResponse[object]:
        """Sends a model-specific payload to a model; see :meth:`AsyncNativeInference.run`."""
        path, body = _prepare(model_path, payload, response_type, extra_body)
        return await self._send(
            "POST",
            path,
            body=body,
            retry=billed_post_retry(),
            options=options(timeout=timeout, max_retries=max_retries, extra_headers=extra_headers),
        )


class AsyncNativeInference(AsyncAPIResource):
    """Native inference: ``POST /inference/{model_path}``."""

    @property
    def with_response(self) -> AsyncNativeInferenceWithResponse:
        """The same methods, returning the whole :class:`~devupai.APIResponse` (headers, request id, warnings)."""
        return AsyncNativeInferenceWithResponse(self._client)

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
        """Sends a model-specific payload to a model and returns its result, parsed from JSON.

        The payload is sent as JSON, as it is. ``model_path`` is the model id: leading slashes are removed, and an id
        that is already percent-encoded is decoded first, so it is sent as one path segment encoded exactly once.

        ``response_type="binary"`` (an answer that is a file) and a file in the payload (bytes, a path or an open
        file, which needs the multipart form) are available in a later build: they raise ``NotImplementedError``
        before anything is sent.

        ``extra_body`` adds fields to the JSON body (merged last), ``extra_headers`` adds request headers, and
        ``timeout`` and ``max_retries`` apply to this call only.

        Example::

            result = await client.inference.run("<model-id>", {"input": "Hello"})
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
