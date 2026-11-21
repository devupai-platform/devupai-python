"""Types of the model catalog: ``GET /v1/models`` and ``GET /v1/models/{model}``."""

from typing import Literal

from devupai.types._base import DevupModel
from devupai.types.images import ImageParams

__all__ = [
    "ModelListResponse",
    "ModelObject",
    "ModelObjectReasoning",
    "ModelObjectSupports",
    "ModelObjectToolCalls",
    "ModelObjectToolChoice",
    "ModelPricing",
]


class ModelPricing(DevupModel):
    """How a model is billed: the unit, the currency, and the amount per unit for each kind of use."""

    unit: str
    """The billing unit, for example tokens, an image or a second."""
    currency: Literal["DZD"] | str
    input: float
    output: float | None = None
    formula: str | None = None
    cached: float | None = None


class ModelObjectSupports(DevupModel):
    """The features a model supports. Present on every model."""

    tools: bool
    vision: bool
    reasoning: bool
    structured_output: bool
    audio_input: bool
    video_input: bool
    reasoning_can_be_disabled: bool
    pdf_input: bool


class ModelObjectToolCalls(DevupModel):
    """Tool calling details of a model. Each key is present only when known."""

    chat_completions: bool | None = None
    """``False``: tool calls of this model are served on ``/v1/responses``."""
    parallel: bool | None = None
    """``True``: the model can return several tool calls in one reply."""


class ModelObjectToolChoice(DevupModel):
    """For each ``tool_choice`` value whose handling is known: ``True`` when the model enforces it."""

    none: bool | None = None
    required: bool | None = None
    function: bool | None = None


class ModelObjectReasoning(DevupModel):
    """The reasoning mode of a model and, when known, the request parameter that controls it with its levels."""

    mode: Literal["always", "on_request", "none"] | str
    parameter: str | None = None
    """The request parameter that turns reasoning on or sets its level."""
    values: list[str | int | float | bool] | None = None
    """The reasoning levels the model takes with ``parameter``, from least to most reasoning."""


class ModelObject(DevupModel):
    """One model of the catalog. On models that list image options, ``image_params`` gives them."""

    id: str
    """The model identifier."""
    object: Literal["model"] | str
    name: str
    """The display name."""
    created: int
    """When the model was added to the catalog, in Unix seconds."""
    owned_by: str
    family: str | None = None
    """The slug of the model family, when the model belongs to one."""
    type: str
    """The model type, for example ``text-generation`` or ``text-to-image``. New values can be added."""
    context_window: int | None
    """The token budget shared by the prompt and the reply, or ``None`` when not set."""
    context_length: str | None = None
    """A short display label of ``context_window``."""
    max_output_tokens: int | None = None
    """The largest output length the model accepts."""
    is_replaced: bool
    """Whether a newer version replaces the model."""
    pricing: ModelPricing
    capabilities: list[str] | None = None
    supports: ModelObjectSupports
    tool_calls: ModelObjectToolCalls | None = None
    tool_choice: ModelObjectToolChoice | None = None
    reasoning: ModelObjectReasoning | None = None
    image_params: ImageParams | None = None


class ModelListResponse(DevupModel):
    """The response of ``GET /v1/models``."""

    object: Literal["list"] | str
    data: list[ModelObject]
