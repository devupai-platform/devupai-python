"""Types of ``POST /v1/responses``: the request body, the response, and the stream events."""

from collections.abc import Mapping, Sequence
from typing import Literal, Required, TypeAlias, TypedDict

from pydantic import Field

from ._base import DevupModel
from .shared import DevUpBilling

__all__ = [
    "ResponseCreateParams",
    "ResponseCreateParamsReasoning",
    "ResponseCreateParamsTool",
    "ResponseCreateParamsToolChoice",
    "ResponseObject",
    "ResponseObjectError",
    "ResponseObjectIncompleteDetails",
    "ResponseObjectUsage",
    "ResponseObjectUsageInputTokensDetails",
    "ResponseObjectUsageOutputTokensDetails",
    "ResponseStreamErrorEvent",
    "ResponseStreamErrorEventError",
    "ResponseStreamEvent",
    "ResponseStreamEventType",
]


# --------------------------------------------------------------------------------------------------- request side


class ResponseCreateParamsReasoning(TypedDict, total=False):
    """Reasoning settings of the request (other keys can be given as a plain object)."""

    effort: Literal["none", "low", "medium", "high", "xhigh"]


class ResponseCreateParamsTool(TypedDict, total=False):
    """A tool the model may call (its other keys can be given as a plain object)."""

    type: Required[Literal["function", "custom", "namespace"]]


class ResponseCreateParamsToolChoice(TypedDict, total=False):
    """Asks the model to call one named function."""

    type: Required[Literal["function"]]
    name: Required[str]


class ResponseCreateParams(TypedDict, total=False):
    """The request body of ``POST /v1/responses``."""

    model: Required[str]
    input: str | Sequence[Mapping[str, object]]
    instructions: str
    max_output_tokens: int
    reasoning: ResponseCreateParamsReasoning | Mapping[str, object]
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    user: str
    text: str | Mapping[str, object]
    metadata: Mapping[str, object]
    tools: Sequence[ResponseCreateParamsTool | Mapping[str, object]]
    tool_choice: Literal["auto", "none", "required"] | ResponseCreateParamsToolChoice
    parallel_tool_calls: bool
    service_tier: str
    truncation: str | Mapping[str, object]
    store: bool
    stream: bool
    include: Sequence[str]
    safety_identifier: str
    top_logprobs: int
    max_tool_calls: int
    stream_options: Mapping[str, object]
    previous_response_id: str | None
    client_metadata: Mapping[str, object]
    background: Literal[False] | None


# -------------------------------------------------------------------------------------------------- response side

#: A plain JSON object. Named at module level because a field called ``object`` hides the builtin in a class body.
_JsonObject: TypeAlias = dict[str, object]


class ResponseObjectError(DevupModel):
    """The error of a response that could not be completed."""

    message: str | None = None
    type: str | None = None
    code: str | None = None


class ResponseObjectIncompleteDetails(DevupModel):
    """Why a response is incomplete."""

    reason: str | None = None


class ResponseObjectUsageInputTokensDetails(DevupModel):
    """A breakdown of the input tokens."""

    cached_tokens: int | None = None


class ResponseObjectUsageOutputTokensDetails(DevupModel):
    """A breakdown of the output tokens."""

    reasoning_tokens: int | None = None


class ResponseObjectUsage(DevupModel):
    """Token counts of a response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens_details: ResponseObjectUsageInputTokensDetails | None = None
    output_tokens_details: ResponseObjectUsageOutputTokensDetails | None = None


class ResponseObject(DevupModel):
    """The response of ``POST /v1/responses``, also carried by some stream events."""

    id: str | None = None
    object: str | None = None
    created_at: int | None = None
    model: str | None = None
    status: str | None = None
    output: list[_JsonObject] | None = None
    error: ResponseObjectError | None = None
    incomplete_details: ResponseObjectIncompleteDetails | None = None
    usage: ResponseObjectUsage | None = None
    devup: DevUpBilling | None = Field(default=None, alias="_devup")
    """Billing details of the request (``_devup`` on the wire)."""


ResponseStreamEventType: TypeAlias = Literal[
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.completed",
    "response.incomplete",
    "response.failed",
    "error",
]
"""The event types a ``POST /v1/responses`` stream sends. Other ``response.*`` types may appear: skip the ones your
code does not handle."""


class ResponseStreamEvent(DevupModel):
    """One event of a ``POST /v1/responses`` stream. Switch on ``type``."""

    type: str
    sequence_number: int
    response: ResponseObject | None = None
    output_index: int | None = None
    item_id: str | None = None
    item: dict[str, object] | None = None
    content_index: int | None = None
    summary_index: int | None = None
    part: dict[str, object] | None = None
    delta: str | None = None
    text: str | None = None
    arguments: str | None = None


class ResponseStreamErrorEventError(DevupModel):
    """The error object of a stream error event."""

    message: str
    type: Literal["devup_error"] | str
    code: str


class ResponseStreamErrorEvent(DevupModel):
    """The last event of a Responses stream whose response could not be completed (``type: "error"``)."""

    type: Literal["error"] | str
    sequence_number: int
    code: str
    message: str
    param: None = None
    error: ResponseStreamErrorEventError
