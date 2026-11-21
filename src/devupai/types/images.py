"""Types of the image endpoints: ``POST /v1/images/generations`` and ``POST /v1/images/edits``, and the image
options a model lists in ``image_params``."""

# No ``from __future__ import annotations`` here: it would empty ``__required_keys__`` of the TypedDicts.

from collections.abc import Sequence
from typing import Literal, Required, TypedDict

from pydantic import Field

from devupai.types._base import DevupModel
from devupai.types.shared import DevUpBilling, FileInput

__all__ = [
    "ImageEditJsonParams",
    "ImageEditParams",
    "ImageGenerateParams",
    "ImageGenerateResponse",
    "ImageGenerateResponseData",
    "ImageGenerateResponseUsage",
    "ImageGenerateResponseUsageInputTokensDetails",
    "ImageGenerateResponseUsageOutputTokensDetails",
    "ImageInputUrlParam",
    "ImageParams",
    "ImageParamsInputs",
    "ImageParamsInputsImage",
    "ImageParamsOption",
    "ImageParamsOptionFree",
    "ImageParamsOptionRange",
]


# --- requests ---------------------------------------------------------------------------------------------------


class ImageInputUrlParam(TypedDict, total=False):
    """One input image given as an object: ``{"image_url": ...}``, an HTTPS URL or a data URI."""

    image_url: Required[str]


class ImageGenerateParams(TypedDict, total=False):
    """The JSON body of ``POST /v1/images/generations``.

    A model can also take options of its own, listed in ``image_params.options`` of its model object
    (:class:`ImageParams`): each one is sent under its own name.
    """

    model: Required[str]
    prompt: str
    negative_prompt: str
    size: str
    n: int
    width: int
    height: int
    quality: str
    complexity: str
    background: str
    output_format: str
    output_compression: int
    aspect_ratio: str
    num_inference_steps: int
    guidance_scale: float
    seed: int
    response_format: Literal["url", "b64_json"]


class ImageEditJsonParams(ImageGenerateParams, total=False):
    """The JSON body of ``POST /v1/images/edits``: the fields of :class:`ImageGenerateParams` plus the input
    images. Each image is an HTTPS URL or a data URI, as a string or as :class:`ImageInputUrlParam`."""

    images: Sequence[str | ImageInputUrlParam]
    image: str | ImageInputUrlParam | Sequence[str | ImageInputUrlParam]
    mask: str | ImageInputUrlParam
    markup: str | ImageInputUrlParam


class ImageEditParams(TypedDict, total=False):
    """The fields of ``POST /v1/images/edits`` sent as ``multipart/form-data``.

    ``image`` is one input or several in order (the first is the one a mask or markup applies to); each input is a
    file, or an HTTPS URL or a data URI as text. A model's own options are sent as text fields under their own names.
    """

    model: Required[str]
    prompt: str
    image: FileInput | str | Sequence[FileInput | str]
    mask: FileInput | str
    markup: FileInput | str
    size: str
    n: int
    width: int
    height: int
    quality: str
    complexity: str
    background: str
    output_format: str
    output_compression: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    negative_prompt: str
    aspect_ratio: str
    response_format: Literal["url", "b64_json"]


# --- responses --------------------------------------------------------------------------------------------------


class ImageGenerateResponseData(DevupModel):
    """One image: ``url`` when ``response_format`` is ``url`` (the default), ``b64_json`` when it is ``b64_json``."""

    url: str | None = None
    """The link to the image."""
    b64_json: str | None = None
    """The image bytes, base64-encoded."""


class ImageGenerateResponseUsageInputTokensDetails(DevupModel):
    """The input tokens, by kind."""

    text_tokens: int | None = None
    image_tokens: int | None = None


class ImageGenerateResponseUsageOutputTokensDetails(DevupModel):
    """The output tokens, by kind."""

    image_tokens: int | None = None
    text_tokens: int | None = None


class ImageGenerateResponseUsage(DevupModel):
    """The token counts of the request, on models billed by the tokens a request uses."""

    input_tokens: int | None = None
    input_tokens_details: ImageGenerateResponseUsageInputTokensDetails | None = None
    output_tokens: int | None = None
    output_tokens_details: ImageGenerateResponseUsageOutputTokensDetails | None = None
    total_tokens: int | None = None


class ImageGenerateResponse(DevupModel):
    """The response of an image generation or edit."""

    data: list[ImageGenerateResponseData]
    """The images."""
    created: int | None = None
    """When the response was made, in Unix seconds."""
    size: str | None = None
    quality: str | None = None
    output_format: str | None = None
    background: str | None = None
    usage: ImageGenerateResponseUsage | None = None
    devup: DevUpBilling = Field(alias="_devup")
    """The cost of the request and the balance after it (``_devup`` on the wire)."""


class ImageParamsOptionRange(DevupModel):
    """The numbers an option accepts: from ``min`` to ``max``, in steps of ``step`` when listed."""

    min: float
    max: float
    step: float | None = None


class ImageParamsOptionFree(DevupModel):
    """``size`` only: the rules other sizes the model takes must follow, besides ``values``."""

    multiple_of: int | None = None
    min_edge: int | None = None
    max_edge: int | None = None
    min_ratio: float | None = None
    max_ratio: float | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    when: dict[str, list[str | int | float]] | None = None
    """The rules apply only while these options are set to one of the listed values."""


class ImageParamsOption(DevupModel):
    """One option of a model in ``image_params.options``. Without ``kind`` it is a field of
    :class:`ImageGenerateParams`; with ``kind`` it is an option of the model's own, sent under the name in
    ``field``."""

    field: str
    """The request field name."""
    kind: Literal["boolean", "choice", "int", "decimal", "text", "structured"] | str | None = None
    values: list[str | int | float] | None = None
    """The accepted values, case kept."""
    range: ImageParamsOptionRange | None = None
    default: str | int | float | bool | None = None
    """The value used when the request leaves the option out."""
    required: Literal[True] | bool | None = None
    """``True``: the request must send the option."""
    free: ImageParamsOptionFree | None = None
    max_length: int | None = None
    endpoints: list[Literal["generations", "edits"] | str] | None = None
    """The endpoints the option applies on. Absent: every endpoint of the model."""
    send_when: dict[str, list[str | int | float]] | None = None
    """The option applies only while these options are set to one of the listed values."""
    schema_: dict[str, object] | None = Field(default=None, alias="schema")
    """``structured`` options only: the JSON Schema the value follows (``schema`` on the wire)."""
    length_matches: Literal["input_images"] | str | None = None


class ImageParamsInputsImage(DevupModel):
    """The input images of an edit: at least ``min`` and at most ``max``."""

    min: int
    max: int
    roles: list[Literal["source", "reference"] | str] | None = None
    """The role of each image by position: ``source`` (changed by the edit) or ``reference`` (guides it)."""
    max_pixels: int | None = None
    max_edge: int | None = None


class ImageParamsInputs(DevupModel):
    """What an edit takes besides the prompt: the input images, and whether a mask or markup is taken."""

    image: ImageParamsInputsImage | None = None
    mask: Literal[True] | bool | None = None
    markup: Literal[True] | bool | None = None
    mask_required: Literal[True] | bool | None = None
    mask_max_images: int | None = None
    mask_format: Literal["alpha", "black_and_white"] | str | None = None


class ImageParams(DevupModel):
    """The image options of a model (``image_params`` of a model object): the image endpoints it is served on, the
    options it takes, the input images an edit takes, and how it takes a prompt."""

    endpoints: list[Literal["generations", "edits"] | str]
    options: list[ImageParamsOption]
    inputs: ImageParamsInputs | None = None
    prompt: Literal["none", "optional"] | str | None = None
    """``none``: the model takes no prompt. ``optional``: a prompt is optional. Absent: a prompt is required."""
    prompt_max_length: int | None = None
