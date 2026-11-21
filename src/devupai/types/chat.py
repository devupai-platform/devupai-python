"""Types of ``POST /v1/chat/completions``: the request body, the response, and the stream chunks."""

from collections.abc import Mapping, Sequence
from typing import Literal, Required, TypeAlias, TypedDict

from pydantic import Field

from ._base import DevupModel
from .shared import DevUpBilling, ErrorResponseError, TokenUsage

__all__ = [
    "ChatCompletion",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionChunkDeltaToolCall",
    "ChatCompletionChunkDeltaToolCallFunction",
    "ChatCompletionCreateParams",
    "ChatCompletionCreateParamsReasoning",
    "ChatCompletionCreateParamsStreamOptions",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    "ChatCompletionStreamError",
    "ChatContentPart",
    "ChatContentPartFile",
    "ChatContentPartFileFile",
    "ChatContentPartFileParam",
    "ChatContentPartFileParamFile",
    "ChatContentPartImageUrl",
    "ChatContentPartImageUrlImageUrl",
    "ChatContentPartImageUrlParam",
    "ChatContentPartImageUrlParamImageUrl",
    "ChatContentPartParam",
    "ChatContentPartText",
    "ChatContentPartTextParam",
]


# --------------------------------------------------------------------------------------------------- request side


class ChatContentPartTextParam(TypedDict, total=False):
    """A text part of a message."""

    type: Required[Literal["text"]]
    text: Required[str]


class ChatContentPartImageUrlParamImageUrl(TypedDict, total=False):
    """The image of an image part."""

    url: Required[str]
    """An ``https://`` URL or a ``data:`` URL of the image."""


class ChatContentPartImageUrlParam(TypedDict, total=False):
    """An image part of a message."""

    type: Required[Literal["image_url"]]
    image_url: Required[ChatContentPartImageUrlParamImageUrl]


class ChatContentPartFileParamFile(TypedDict, total=False):
    """The document of a file part."""

    filename: str
    file_data: Required[str]
    """The document as a ``data:application/pdf;base64,`` URL."""


class ChatContentPartFileParam(TypedDict, total=False):
    """A PDF document part of a message."""

    type: Required[Literal["file"]]
    file: Required[ChatContentPartFileParamFile]


ChatContentPartParam: TypeAlias = (
    ChatContentPartTextParam | ChatContentPartImageUrlParam | ChatContentPartFileParam | dict[str, object]
)
"""One part of a message: text, an image, a document, or another part type given as a plain object."""


class ChatCompletionMessageParam(TypedDict, total=False):
    """A message of the conversation."""

    role: Required[Literal["system", "user", "assistant", "tool"]]
    content: Required[str | Sequence[ChatContentPartParam] | None]
    reasoning_content: str | None
    name: str
    tool_calls: Sequence[Mapping[str, object]]
    tool_call_id: str


class ChatCompletionCreateParamsReasoning(TypedDict, total=False):
    """Reasoning settings of the request."""

    enabled: bool


class ChatCompletionCreateParamsStreamOptions(TypedDict, total=False):
    """Options of a streamed request."""

    include_usage: bool


class ChatCompletionCreateParams(TypedDict, total=False):
    """The request body of ``POST /v1/chat/completions``."""

    model: Required[str]
    messages: Required[Sequence[ChatCompletionMessageParam]]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    stop: str | Sequence[str]
    frequency_penalty: float
    presence_penalty: float
    tools: Sequence[Mapping[str, object]]
    tool_choice: str | Mapping[str, object]
    reasoning_effort: str
    reasoning: ChatCompletionCreateParamsReasoning
    response_format: Mapping[str, object]
    service_tier: Literal["priority", "flex"]
    max_completion_tokens: int
    n: int | None
    stream_options: ChatCompletionCreateParamsStreamOptions
    parallel_tool_calls: bool
    seed: int
    user: str
    logprobs: bool


# -------------------------------------------------------------------------------------------------- response side


class ChatContentPartText(DevupModel):
    """A text part of a message."""

    type: Literal["text"] | str
    text: str


class ChatContentPartImageUrlImageUrl(DevupModel):
    """The image of an image part."""

    url: str


class ChatContentPartImageUrl(DevupModel):
    """An image part of a message."""

    type: Literal["image_url"] | str
    image_url: ChatContentPartImageUrlImageUrl


class ChatContentPartFileFile(DevupModel):
    """The document of a file part."""

    filename: str | None = None
    file_data: str


class ChatContentPartFile(DevupModel):
    """A PDF document part of a message."""

    type: Literal["file"] | str
    file: ChatContentPartFileFile


ChatContentPart: TypeAlias = ChatContentPartText | ChatContentPartImageUrl | ChatContentPartFile | dict[str, object]
"""One part of a message: text, an image, a document, or another part type as a plain object."""


class ChatCompletionMessage(DevupModel):
    """A message returned by the model."""

    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[ChatContentPart] | None
    reasoning_content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None


class ChatCompletionChoice(DevupModel):
    """One choice of a chat completion."""

    index: int | None = None
    message: ChatCompletionMessage | None = None
    logprobs: dict[str, object] | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | str | None = None


class ChatCompletion(DevupModel):
    """The response of ``POST /v1/chat/completions``."""

    id: str
    object: Literal["chat.completion"] | str
    created: int
    model: str
    service_tier: str | None = None
    choices: list[ChatCompletionChoice]
    system_fingerprint: str | None = None
    usage: TokenUsage | None = None
    devup: DevUpBilling | None = Field(default=None, alias="_devup")
    """Billing details of the request (``_devup`` on the wire)."""


class ChatCompletionChunkDeltaToolCallFunction(DevupModel):
    """The function part of a tool call delta."""

    name: str | None = None
    arguments: str | None = None


class ChatCompletionChunkDeltaToolCall(DevupModel):
    """A tool call delta."""

    index: int
    id: str | None = None
    type: Literal["function"] | str | None = None
    function: ChatCompletionChunkDeltaToolCallFunction | None = None


class ChatCompletionChunkDelta(DevupModel):
    """The new content of a choice in a chunk."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionChunkDeltaToolCall] | None = None


class ChatCompletionChunkChoice(DevupModel):
    """One choice of a stream chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    logprobs: dict[str, object] | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | str | None = None


class ChatCompletionChunk(DevupModel):
    """One chunk of a streamed chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] | str
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: TokenUsage | None = None
    service_tier: str | None = None
    system_fingerprint: str | None = None
    devup: DevUpBilling | None = Field(default=None, alias="_devup")
    """Billing details of the request (``_devup`` on the wire)."""


class ChatCompletionStreamError(DevupModel):
    """The last event of a chat stream that could not be completed."""

    error: ErrorResponseError
