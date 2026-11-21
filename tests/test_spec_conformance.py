"""Spec conformance: every in-scope shape of the vendored spec against the Python types in ``devupai.types``.

The 54 shapes are the 50 component schemas (all components minus the 6 Messages-only ones) and the 4 webhook payload
shapes. For each shape and each Python name it maps to (contract table §2.3), per property (``$ref`` followed,
``allOf`` merged), the tests compare:

- the property names (model: the field alias when set, else the field name; TypedDict: the keys);
- the required set (TypedDict ``__required_keys__``; model ``model_fields[n].is_required()``);
- the enum values (request TypedDict: the ``Literal`` arguments exactly, no open ``str``; response model: the
  ``Literal`` arguments inside ``Literal[...] | str``, as a set, the open member ignored);
- nullability (``type: [T, "null"]``, a ``null`` member, a ``null`` enum value ⇔ ``None`` in the annotation, for
  TypedDicts and required model fields; an optional model field always allows ``None``);
- unions (``oneOf`` / ``anyOf`` / a list of types): each spec member maps to a Python member of the same kind;
- ``$ref``: the annotation contains the Python type mapped to the referenced schema, on the same side.

Inline objects (also the items of arrays, and the inline members of unions) are compared the same way against the
nested TypedDict / model the Python annotation resolves to. A schema used on both sides (contract §2.2) is checked
against its ``…Param`` TypedDict and against its model.
"""

from __future__ import annotations

import collections.abc
import importlib
import io
import os
import types
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, NotRequired, Required, Union, get_args, get_origin

import pydantic
import pytest
import typing_extensions
import yaml

SPEC_PATH = Path(__file__).resolve().parent.parent / "openapi" / "devupai-v1.yaml"
SPEC: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_bytes())
COMPONENTS: dict[str, Any] = SPEC["components"]["schemas"]
REF_PREFIX = "#/components/schemas/"

MESSAGES_ONLY = frozenset(
    {
        "MessagesContentBlock",
        "MessagesErrorResponse",
        "MessagesMessage",
        "MessagesRequest",
        "MessagesResponse",
        "MessagesStreamEvent",
    }
)

# Webhook shape ids (the shapes are inline in ``webhooks.inferenceCallback``).
WEBHOOK_SUCCESS = "webhooks.inferenceCallback#success"
WEBHOOK_FAILURE = "webhooks.inferenceCallback#failure"
WEBHOOK_USAGE = "webhooks.inferenceCallback#success.usage"
WEBHOOK_ERROR = "webhooks.inferenceCallback#failure.error"
WEBHOOK_IDS = (WEBHOOK_SUCCESS, WEBHOOK_FAILURE, WEBHOOK_USAGE, WEBHOOK_ERROR)

# Python kinds.
TD = "typeddict"
MODEL = "model"
TD_UNION = "typeddict-union"
MODEL_UNION = "model-union"
LITERAL = "literal"
STR = "str"

PARAM = "param"
RESPONSE = "response"
SIDE = {TD: PARAM, TD_UNION: PARAM, STR: PARAM, MODEL: RESPONSE, MODEL_UNION: RESPONSE, LITERAL: PARAM}

#: Contract §2.3, fixed: (spec schema name or webhook shape id, ((Python name, kind), ...)).
MAPPING: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    ("DevUpBilling", (("DevUpBilling", MODEL),)),
    ("TokenUsage", (("TokenUsage", MODEL),)),
    ("ErrorResponse", (("ErrorResponse", MODEL),)),
    ("ChatCompletionRequest", (("ChatCompletionCreateParams", TD),)),
    ("ChatMessage", (("ChatCompletionMessageParam", TD), ("ChatCompletionMessage", MODEL))),
    ("ChatContentPart", (("ChatContentPartParam", TD_UNION), ("ChatContentPart", MODEL_UNION))),
    ("ChatContentPartText", (("ChatContentPartTextParam", TD), ("ChatContentPartText", MODEL))),
    ("ChatContentPartImageUrl", (("ChatContentPartImageUrlParam", TD), ("ChatContentPartImageUrl", MODEL))),
    ("ChatContentPartFile", (("ChatContentPartFileParam", TD), ("ChatContentPartFile", MODEL))),
    ("ChatCompletionResponse", (("ChatCompletion", MODEL),)),
    ("ChatCompletionChunk", (("ChatCompletionChunk", MODEL),)),
    ("ChatCompletionStreamError", (("ChatCompletionStreamError", MODEL),)),
    ("ResponsesRequest", (("ResponseCreateParams", TD),)),
    ("ResponsesResponse", (("ResponseObject", MODEL),)),
    ("ResponsesStreamEventType", (("ResponseStreamEventType", LITERAL),)),
    ("ResponsesStreamEvent", (("ResponseStreamEvent", MODEL),)),
    ("ResponsesStreamErrorEvent", (("ResponseStreamErrorEvent", MODEL),)),
    ("EmbeddingRequest", (("EmbeddingCreateParams", TD),)),
    ("EmbeddingResponse", (("EmbeddingCreateResponse", MODEL),)),
    ("RerankRequest", (("RerankCreateParams", TD),)),
    ("RerankResponse", (("RerankCreateResponse", MODEL),)),
    ("ImageGenerationRequest", (("ImageGenerateParams", TD),)),
    ("ImageEditJsonRequest", (("ImageEditJsonParams", TD),)),
    ("ImageEditRequest", (("ImageEditParams", TD),)),
    ("ImageInputUrl", (("ImageInputUrlParam", TD),)),
    ("ImageGenerationResponse", (("ImageGenerateResponse", MODEL),)),
    ("ImageParams", (("ImageParams", MODEL),)),
    ("ImageParamsOption", (("ImageParamsOption", MODEL),)),
    ("ImageParamsInputs", (("ImageParamsInputs", MODEL),)),
    ("SpeechRequest", (("SpeechCreateParams", TD),)),
    ("SpeechResponse", (("SpeechResponse", MODEL),)),
    ("TranscriptionRequest", (("TranscriptionCreateParams", TD),)),
    ("TranscriptionResponse", (("TranscriptionResponse", MODEL),)),
    ("TranscriptionVerboseResponse", (("TranscriptionVerboseResponse", MODEL),)),
    ("VideoGenerationRequest", (("VideoGenerationCreateParams", TD),)),
    ("VideoGenerationFormRequest", (("VideoGenerationCreateFormParams", TD),)),
    ("VideoEditRequest", (("VideoEditCreateParams", TD),)),
    ("VideoFrameImage", (("VideoFrameImageParam", TD_UNION),)),
    ("VideoReferenceImage", (("VideoReferenceImageParam", TD_UNION),)),
    ("VideoReferenceVideo", (("VideoReferenceVideoParam", TD_UNION),)),
    ("VideoReferenceAudio", (("VideoReferenceAudioParam", TD_UNION),)),
    ("VideoReferenceVoiceAudio", (("VideoReferenceVoiceAudioParam", STR),)),
    ("VideoGenerationResponse", (("VideoGenerationResponse", MODEL),)),
    ("ModelListResponse", (("ModelListResponse", MODEL),)),
    ("ModelObject", (("ModelObject", MODEL),)),
    ("ModelPricing", (("ModelPricing", MODEL),)),
    ("BalanceResponse", (("BalanceResponse", MODEL),)),
    ("AccountProfile", (("AccountProfile", MODEL),)),
    ("AccountAvatar", (("AccountAvatar", MODEL),)),
    ("HealthResponse", (("HealthResponse", MODEL),)),
    (WEBHOOK_SUCCESS, (("WebhookSuccessPayload", MODEL),)),
    (WEBHOOK_FAILURE, (("WebhookErrorPayload", MODEL),)),
    (WEBHOOK_USAGE, (("WebhookUsage", MODEL),)),
    (WEBHOOK_ERROR, (("WebhookErrorDetails", MODEL),)),
]

#: One row per (spec id, Python name, kind).
TARGETS: list[tuple[str, str, str]] = [(spec_id, name, kind) for spec_id, names in MAPPING for name, kind in names]
TARGET_IDS = [f"{spec_id}->{name}" for spec_id, name, _ in TARGETS]
STRUCTURED_KINDS = (TD, MODEL, TD_UNION, MODEL_UNION)
STRUCTURED = [t for t in TARGETS if t[2] in STRUCTURED_KINDS]
STRUCTURED_IDS = [f"{spec_id}->{name}" for spec_id, name, _ in STRUCTURED]

#: (component name, side) -> Python name, for ``$ref`` checks.
MAPPED_NAME: dict[tuple[str, str], str] = {
    (spec_id, SIDE[kind]): name for spec_id, names in MAPPING if spec_id in COMPONENTS for name, kind in names
}


# --------------------------------------------------------------------------------------------------- spec side


def _ref_target(ref: str) -> str:
    assert ref.startswith(REF_PREFIX), f"unsupported $ref {ref!r}"
    return ref[len(REF_PREFIX) :]


def ref_name(schema: Any) -> str | None:
    """The component a schema points to: a direct ``$ref``, or an ``allOf`` of exactly one ``$ref``."""
    if not isinstance(schema, dict):
        return None
    if "$ref" in schema:
        return _ref_target(schema["$ref"])
    parts = schema.get("allOf")
    structural = {"type", "properties", "oneOf", "anyOf", "items", "enum"}
    if isinstance(parts, list) and len(parts) == 1 and "$ref" in parts[0] and not structural & schema.keys():
        return _ref_target(parts[0]["$ref"])
    return None


def resolve(schema: Any) -> dict[str, Any]:
    """``$ref`` followed and ``allOf`` merged (properties and required united, other keys first-wins)."""
    assert isinstance(schema, dict), f"schema is not an object: {schema!r}"
    while "$ref" in schema:
        schema = COMPONENTS[_ref_target(schema["$ref"])]
    if "allOf" not in schema:
        return schema
    merged: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for part in [*schema["allOf"], {k: v for k, v in schema.items() if k != "allOf"}]:
        resolved = resolve(part)
        for key, value in resolved.items():
            if key == "properties":
                properties.update(value)
            elif key == "required":
                required.extend(r for r in value if r not in required)
            else:
                merged.setdefault(key, value)
    if properties:
        merged["properties"] = properties
    if required:
        merged["required"] = required
    return merged


def _structural(members: list[Any]) -> bool:
    """False for a ``oneOf`` of constraint-only fragments (e.g. ``[{required: [url]}, {required: [b64_json]}]``)."""
    keys = {"type", "$ref", "properties", "oneOf", "anyOf", "allOf", "enum", "items"}
    return any(isinstance(m, dict) and keys & m.keys() for m in members)


def spec_members(schema: Any) -> list[tuple[str | None, dict[str, Any]]]:
    """The union members at this level: ``(component name or None, resolved member)``.

    A ``$ref`` member is kept whole (its own union, if any, belongs to the referenced component); ``oneOf`` /
    ``anyOf`` are flattened; a list of types gives one member per type.
    """
    name = ref_name(schema)
    if name is not None:
        return [(name, resolve(schema))]
    resolved = resolve(schema)
    for key in ("oneOf", "anyOf"):
        members = resolved.get(key)
        if isinstance(members, list) and _structural(members):
            out: list[tuple[str | None, dict[str, Any]]] = []
            for member in members:
                out.extend(spec_members(member))
            return out
    declared = resolved.get("type")
    if isinstance(declared, list):
        return [(None, {**resolved, "type": one}) for one in declared]
    return [(None, resolved)]


def spec_kind(schema: dict[str, Any]) -> str:
    declared = schema.get("type")
    if declared == "null":
        return "null"
    if declared == "string":
        return "binary" if schema.get("format") == "binary" else "string"
    if declared in ("integer", "number", "boolean", "array", "object"):
        return str(declared)
    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "any"


def spec_properties(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties", {})
    assert isinstance(props, dict)
    return props


def spec_required(schema: dict[str, Any]) -> set[str]:
    return set(schema.get("required", []))


def spec_nullable(schema: Any) -> bool:
    for _, member in spec_members(schema):
        if spec_kind(member) == "null" or None in member.get("enum", []):
            return True
    return False


def _typed(value: object) -> tuple[str, object]:
    """A comparable key that keeps ``True`` apart from ``1`` and ``False`` apart from ``0``."""
    return (type(value).__name__, value)


def spec_enum_level(schema: Any) -> tuple[set[tuple[str, object]], bool]:
    """The enum values at this level (array items included, objects and ``$ref`` members excluded), and whether a
    free (non-enum) string member exists at this level."""
    values: set[tuple[str, object]] = set()
    free_string = False
    for name, member in spec_members(schema):
        if name is not None:
            continue
        kind = spec_kind(member)
        if kind == "array" and isinstance(member.get("items"), dict):
            sub_values, sub_free = spec_enum_level(member["items"])
            values |= sub_values
            free_string = free_string or sub_free
        elif kind in ("object", "null"):
            continue
        elif "enum" in member:
            values |= {_typed(v) for v in member["enum"] if v is not None}
        elif kind == "string":
            free_string = True
    return values, free_string


def webhook_payloads() -> dict[str, dict[str, Any]]:
    """The webhook ``oneOf`` members keyed by their ``status`` enum (``succeeded`` / ``failed``)."""
    schema = SPEC["webhooks"]["inferenceCallback"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    found: dict[str, dict[str, Any]] = {}
    for member in schema["oneOf"]:
        status = member["properties"]["status"]["enum"]
        found[status[0] if len(status) == 1 else repr(status)] = member
    return found


def spec_schema(spec_id: str) -> dict[str, Any]:
    if spec_id in COMPONENTS:
        schema = COMPONENTS[spec_id]
        assert isinstance(schema, dict)
        return schema
    payloads = webhook_payloads()
    return {
        WEBHOOK_SUCCESS: payloads["succeeded"],
        WEBHOOK_FAILURE: payloads["failed"],
        WEBHOOK_USAGE: payloads["succeeded"]["properties"]["usage"],
        WEBHOOK_ERROR: payloads["failed"]["properties"]["error"],
    }[spec_id]


# ------------------------------------------------------------------------------------------------- Python side


def types_module() -> types.ModuleType:
    return importlib.import_module("devupai.types")


def py_object(name: str) -> Any:
    module = types_module()
    assert hasattr(module, name), f"devupai.types has no {name!r}"
    return getattr(module, name)


def devup_model() -> type[pydantic.BaseModel]:
    base: type[pydantic.BaseModel] = importlib.import_module("devupai.types._base").DevupModel
    return base


_WRAPPERS: tuple[object, ...] = tuple(
    w
    for w in (Required, NotRequired, getattr(typing, "ReadOnly", None), getattr(typing_extensions, "ReadOnly", None))
    if w is not None
)
_ALIAS_TYPES: tuple[type, ...] = tuple(
    t for t in (getattr(typing, "TypeAliasType", None), getattr(typing_extensions, "TypeAliasType", None)) if t
)


def unwrap(annotation: Any) -> Any:
    """Strips ``Annotated``, ``Required`` / ``NotRequired`` / ``ReadOnly``, type alias objects and forward refs."""
    while True:
        if isinstance(annotation, typing.ForwardRef):
            annotation = py_object(annotation.__forward_arg__)
            continue
        if isinstance(annotation, str):
            annotation = py_object(annotation)
            continue
        if _ALIAS_TYPES and isinstance(annotation, _ALIAS_TYPES):
            annotation = annotation.__value__
            continue
        origin = get_origin(annotation)
        if origin is typing.Annotated or origin in _WRAPPERS:
            annotation = get_args(annotation)[0]
            continue
        return annotation


def py_members(annotation: Any) -> list[Any]:
    """The flattened union members of an annotation (``None`` as ``NoneType``)."""
    annotation = unwrap(annotation)
    if annotation is None:
        return [type(None)]
    if get_origin(annotation) in (Union, types.UnionType):
        out: list[Any] = []
        for arg in get_args(annotation):
            out.extend(m for m in py_members(arg) if m not in out)
        return out
    return [annotation]


def is_td(obj: Any) -> bool:
    return isinstance(obj, type) and typing_extensions.is_typeddict(obj)


def is_model(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, pydantic.BaseModel)


def is_class(obj: Any) -> bool:
    return is_td(obj) or is_model(obj)


_ARRAY_ORIGINS = (
    list,
    tuple,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Iterable,
    collections.abc.Collection,
)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)
_BINARY = (bytes, bytearray, memoryview, os.PathLike, IO, typing.BinaryIO, io.IOBase)


def py_kinds(member: Any) -> set[str]:
    member = unwrap(member)
    if member is type(None):
        return {"null"}
    if member is Any or member is object:
        return {"any"}
    origin = get_origin(member)
    if origin is Literal:
        kinds: set[str] = set()
        for value in get_args(member):
            if value is None:
                kinds.add("null")
            elif isinstance(value, bool):
                kinds.add("boolean")
            elif isinstance(value, int):
                kinds.add("integer")
            elif isinstance(value, str):
                kinds.add("string")
            else:
                kinds.add("other")
        return kinds
    if is_class(member):
        return {"object"}
    base = origin if origin is not None else member
    if base in _MAPPING_ORIGINS:
        return {"object"}
    if base in _ARRAY_ORIGINS:
        return {"array"}
    if base is bool:
        return {"boolean"}
    if base is int:
        return {"integer"}
    if base is float:
        return {"number"}
    if base is str:
        return {"string"}
    if (isinstance(base, type) and issubclass(base, _BINARY)) or base in _BINARY:
        return {"binary"}
    return {"other"}


def array_item_members(members: list[Any]) -> list[Any]:
    out: list[Any] = []
    for member in members:
        if "array" in py_kinds(member):
            args = get_args(unwrap(member))
            if args:
                out.extend(m for m in py_members(args[0]) if m not in out)
    return out


def kinds_of(members: list[Any]) -> set[str]:
    out: set[str] = set()
    for member in members:
        out |= py_kinds(member)
    return out


def kind_matches(spec: str, python: set[str]) -> bool:
    return spec == "any" or spec in python or "any" in python


def py_enum_level(members: list[Any]) -> tuple[set[tuple[str, object]], set[str]]:
    """``Literal`` arguments at this level (array items included, nested classes excluded) and the kinds of the
    other (open) members at this level."""
    literals: set[tuple[str, object]] = set()
    open_kinds: set[str] = set()
    for member in members:
        unwrapped = unwrap(member)
        if get_origin(unwrapped) is Literal:
            literals |= {_typed(v) for v in get_args(unwrapped) if v is not None}
            continue
        kinds = py_kinds(unwrapped)
        if "array" in kinds:
            sub_literals, sub_open = py_enum_level(array_item_members([unwrapped]))
            literals |= sub_literals
            open_kinds |= sub_open
        elif kinds & {"object", "null"}:
            continue
        else:
            open_kinds |= kinds
    return literals, open_kinds


def type_hints(cls: Any) -> dict[str, Any]:
    hints: dict[str, Any] = typing.get_type_hints(cls, include_extras=True)
    return hints


def py_fields(cls: Any) -> dict[str, Any]:
    """Wire name -> annotation. Model: the alias when set, else the field name. TypedDict: the keys."""
    hints = type_hints(cls)
    if is_model(cls):
        return {(info.alias or name): hints.get(name, info.annotation) for name, info in cls.model_fields.items()}
    return dict(hints)


def py_required(cls: Any) -> set[str]:
    if is_model(cls):
        return {(info.alias or name) for name, info in cls.model_fields.items() if info.is_required()}
    return set(cls.__required_keys__)


def _required_markers(cls: Any) -> set[str]:
    """The keys an explicit ``Required[...]`` marks (to explain a ``__required_keys__`` mismatch)."""
    return {key for key, ann in type_hints(cls).items() if get_origin(ann) is Required}


def py_name(obj: Any) -> str:
    return getattr(obj, "__name__", repr(obj))


# ----------------------------------------------------------------------------------------------- the pair walk


@dataclass(frozen=True)
class Pair:
    """A spec object (top-level or inline) and the Python class it is compared with."""

    path: str
    spec: dict[str, Any]
    cls: Any
    side: str


@dataclass
class Walk:
    pairs: list[Pair] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _choose_class(member: dict[str, Any], members: list[Any]) -> tuple[Any | None, bool]:
    """The Python class an inline spec object maps to, and whether an open mapping member stands for it instead."""
    classes = [m for m in members if is_class(m)]
    keys = set(spec_properties(member))
    exact = [c for c in classes if set(py_fields(c)) == keys]
    if len(exact) == 1:
        return exact[0], False
    has_mapping = any("object" in py_kinds(m) and not is_class(m) for m in members)
    if member.get("additionalProperties") is True and has_mapping:
        return None, True
    if len(classes) == 1:
        return classes[0], False
    return None, False


def walk_object(walk: Walk, spec: dict[str, Any], cls: Any, side: str, path: str, depth: int = 0) -> None:
    walk.pairs.append(Pair(path, spec, cls, side))
    assert depth < 12, f"{path}: nesting too deep"
    fields = py_fields(cls)
    for name, prop in spec_properties(spec).items():
        if name in fields and ref_name(prop) is None:
            walk_members(walk, prop, py_members(fields[name]), side, f"{path}.{name}", depth + 1)


def walk_members(walk: Walk, schema: Any, members: list[Any], side: str, path: str, depth: int) -> None:
    for name, member in spec_members(schema):
        if name is not None:
            continue
        kind = spec_kind(member)
        if kind == "object" and spec_properties(member):
            cls, open_mapping = _choose_class(member, members)
            if cls is not None:
                walk_object(walk, member, cls, side, path, depth)
            elif not open_mapping:
                found = ", ".join(py_name(m) for m in members) or "nothing"
                walk.problems.append(
                    f"{path}: inline object {sorted(spec_properties(member))} maps to no nested TypedDict/model "
                    f"(annotation members: {found})"
                )
        elif kind == "array" and isinstance(member.get("items"), dict) and ref_name(member["items"]) is None:
            walk_members(walk, member["items"], array_item_members(members), side, f"{path}[]", depth)


def walk_target(spec_id: str, name: str, kind: str) -> Walk:
    obj = py_object(name)
    schema = spec_schema(spec_id)
    walk = Walk()
    if kind in (TD, MODEL):
        walk_object(walk, resolve(schema), obj, SIDE[kind], spec_id)
    elif kind in (TD_UNION, MODEL_UNION):
        walk_members(walk, schema, py_members(obj), SIDE[kind], spec_id, 0)
    return walk


def properties_of(pair: Pair) -> Iterator[tuple[str, Any, Any]]:
    """``(name, spec property, annotation)`` for every spec property the Python class also has."""
    fields = py_fields(pair.cls)
    for name, prop in spec_properties(pair.spec).items():
        if name in fields:
            yield name, prop, fields[name]


def where(pair: Pair, prop: str | None = None) -> str:
    suffix = f".{prop}" if prop is not None else ""
    return f"{pair.path}{suffix} ({py_name(pair.cls)})"


def values_text(values: set[tuple[str, object]]) -> str:
    return "[" + ", ".join(sorted(repr(v) for _, v in values)) + "]"


def report(problems: list[str]) -> str:
    return f"{len(problems)} mismatch(es):\n" + "\n".join(problems)


# ------------------------------------------------------------------------------------------------------- tests


def test_mapping_covers_the_54_in_scope_shapes() -> None:
    assert len(MAPPING) == 54
    spec_ids = [spec_id for spec_id, _ in MAPPING]
    assert len(set(spec_ids)) == 54, "a spec shape is listed twice"
    components = [s for s in spec_ids if s not in WEBHOOK_IDS]
    assert len(components) == 50
    assert set(components) == set(COMPONENTS) - MESSAGES_ONLY
    assert MESSAGES_ONLY <= set(COMPONENTS)
    assert [s for s in spec_ids if s in WEBHOOK_IDS] == list(WEBHOOK_IDS)
    python_names = [name for _, name, _ in TARGETS]
    assert len(set(python_names)) == len(python_names), "a Python name is listed twice"


def test_the_webhook_shapes_are_found_in_the_spec() -> None:
    payloads = webhook_payloads()
    assert set(payloads) == {"succeeded", "failed"}
    for spec_id in WEBHOOK_IDS:
        schema = spec_schema(spec_id)
        assert schema.get("type") == "object" and spec_properties(schema), spec_id


@pytest.mark.parametrize(("spec_id", "name", "kind"), TARGETS, ids=TARGET_IDS)
def test_python_type_has_the_mapped_kind(spec_id: str, name: str, kind: str) -> None:
    obj = py_object(name)
    if kind == TD:
        assert is_td(obj), f"{spec_id}: {name} must be a TypedDict"
    elif kind == MODEL:
        assert isinstance(obj, type) and issubclass(obj, devup_model()), f"{spec_id}: {name} must subclass DevupModel"
    elif kind in (TD_UNION, MODEL_UNION):
        members = py_members(obj)
        assert len(members) >= 2, f"{spec_id}: {name} must be a Union alias, got {obj!r}"
        wanted = is_td if kind == TD_UNION else (lambda m: isinstance(m, type) and issubclass(m, devup_model()))
        assert not any(is_class(m) and not wanted(m) for m in members), f"{spec_id}: {name} mixes params and models"
    elif kind == LITERAL:
        assert get_origin(unwrap(obj)) is Literal, f"{spec_id}: {name} must be a Literal alias"
    else:
        assert unwrap(obj) is str, f"{spec_id}: {name} must be an alias of str"


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_inline_objects_resolve_to_nested_types(spec_id: str, name: str, kind: str) -> None:
    walk = walk_target(spec_id, name, kind)
    assert not walk.problems, report(walk.problems)


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_property_names_match(spec_id: str, name: str, kind: str) -> None:
    problems: list[str] = []
    for pair in walk_target(spec_id, name, kind).pairs:
        spec_names = set(spec_properties(pair.spec))
        python_names = set(py_fields(pair.cls))
        if spec_names != python_names:
            problems.append(
                f"{where(pair)}: missing {sorted(spec_names - python_names)}, extra {sorted(python_names - spec_names)}"
            )
    assert not problems, report(problems)


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_required_sets_match(spec_id: str, name: str, kind: str) -> None:
    problems: list[str] = []
    for pair in walk_target(spec_id, name, kind).pairs:
        wanted = spec_required(pair.spec) & set(spec_properties(pair.spec))
        found = py_required(pair.cls)
        if wanted != found:
            hint = ""
            if is_td(pair.cls) and _required_markers(pair.cls) != found:
                hint = " (Required[...] markers are not in __required_keys__: postponed annotations in that module?)"
            problems.append(
                f"{where(pair)}: required should be {sorted(wanted)}, is {sorted(found)}; "
                f"missing {sorted(wanted - found)}, extra {sorted(found - wanted)}{hint}"
            )
    assert not problems, report(problems)


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_enum_values_match(spec_id: str, name: str, kind: str) -> None:
    problems: list[str] = []
    for pair in walk_target(spec_id, name, kind).pairs:
        for prop_name, prop, annotation in properties_of(pair):
            values, free_string = spec_enum_level(prop)
            literals, open_kinds = py_enum_level(py_members(annotation))
            label = where(pair, prop_name)
            if not values:
                if literals:
                    problems.append(
                        f"{label}: the spec has no enum, the annotation has Literal {values_text(literals)}"
                    )
                continue
            if literals != values:
                problems.append(
                    f"{label}: enum {values_text(values)} != Literal {values_text(literals)}; "
                    f"missing {values_text(values - literals)}, extra {values_text(literals - values)}"
                )
            enum_kinds = {"string" if t == "str" else "boolean" if t == "bool" else "integer" for t, _ in values}
            if pair.side == PARAM:
                if "string" in enum_kinds and "string" in open_kinds and not free_string:
                    problems.append(f"{label}: a request enum must be the Literal exactly, without an open str")
            elif not open_kinds & (enum_kinds | {"string", "any"}):
                problems.append(f"{label}: a response enum must stay open (Literal[...] | str), got only Literal")
            elif "string" in enum_kinds and not open_kinds & {"string", "any"}:
                problems.append(f"{label}: a response string enum must be Literal[...] | str")
    assert not problems, report(problems)


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_nullability_matches(spec_id: str, name: str, kind: str) -> None:
    problems: list[str] = []
    for pair in walk_target(spec_id, name, kind).pairs:
        required = py_required(pair.cls)
        for prop_name, prop, annotation in properties_of(pair):
            members = py_members(annotation)
            kinds = kinds_of(members)
            if "any" in kinds:
                continue
            allows_none = "null" in kinds
            nullable = spec_nullable(prop)
            label = where(pair, prop_name)
            if is_model(pair.cls) and prop_name not in required:
                if not allows_none:
                    problems.append(f"{label}: an optional model field must allow None")
            elif allows_none != nullable:
                state = "nullable" if nullable else "not nullable"
                problems.append(f"{label}: the spec is {state}, None in the annotation is {allows_none}")
    assert not problems, report(problems)


def _union_problems(schema: Any, members: list[Any], path: str) -> list[str]:
    problems: list[str] = []
    spec = [(n, m) for n, m in spec_members(schema) if spec_kind(m) != "null"]
    kinds = kinds_of(members)
    if len(spec) > 1:
        for index, (name, member) in enumerate(spec):
            if name is not None:
                continue  # checked by the $ref test
            kind = spec_kind(member)
            if not kind_matches(kind, kinds):
                problems.append(f"{path}: union member {index} ({kind}) has no Python member of that kind")
    for name, member in spec:
        if name is None and spec_kind(member) == "array" and isinstance(member.get("items"), dict):
            problems.extend(_union_problems(member["items"], array_item_members(members), f"{path}[]"))
    return problems


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_union_members_map_by_kind(spec_id: str, name: str, kind: str) -> None:
    problems: list[str] = []
    if kind in (TD_UNION, MODEL_UNION):
        problems.extend(_union_problems(spec_schema(spec_id), py_members(py_object(name)), f"{spec_id} ({name})"))
    for pair in walk_target(spec_id, name, kind).pairs:
        for prop_name, prop, annotation in properties_of(pair):
            problems.extend(_union_problems(prop, py_members(annotation), where(pair, prop_name)))
    assert not problems, report(problems)


def _ref_problems(schema: Any, members: list[Any], side: str, path: str) -> list[str]:
    problems: list[str] = []
    for name, member in spec_members(schema):
        if name is not None:
            mapped = MAPPED_NAME.get((name, side))
            if mapped is None:
                problems.append(f"{path}: $ref {name} has no mapped Python type on the {side} side")
                continue
            missing = [m for m in py_members(py_object(mapped)) if m not in members]
            if missing:
                found = ", ".join(py_name(m) for m in members)
                problems.append(f"{path}: $ref {name} should be {mapped}; annotation members: {found}")
        elif spec_kind(member) == "array" and isinstance(member.get("items"), dict):
            problems.extend(_ref_problems(member["items"], array_item_members(members), side, f"{path}[]"))
    return problems


@pytest.mark.parametrize(("spec_id", "name", "kind"), STRUCTURED, ids=STRUCTURED_IDS)
def test_refs_point_to_the_mapped_types(spec_id: str, name: str, kind: str) -> None:
    problems: list[str] = []
    if kind in (TD_UNION, MODEL_UNION):
        problems.extend(_ref_problems(spec_schema(spec_id), py_members(py_object(name)), SIDE[kind], spec_id))
    for pair in walk_target(spec_id, name, kind).pairs:
        for prop_name, prop, annotation in properties_of(pair):
            problems.extend(_ref_problems(prop, py_members(annotation), pair.side, where(pair, prop_name)))
    assert not problems, report(problems)


def test_response_stream_event_type_is_the_exact_enum() -> None:
    enum = COMPONENTS["ResponsesStreamEventType"]["enum"]
    args = get_args(unwrap(py_object("ResponseStreamEventType")))
    assert len(args) == len(set(args)), "duplicate Literal values"
    assert set(args) == set(enum)


def test_video_reference_voice_audio_param_is_str() -> None:
    assert COMPONENTS["VideoReferenceVoiceAudio"]["type"] == "string"
    assert unwrap(py_object("VideoReferenceVoiceAudioParam")) is str


def _devup_models() -> list[tuple[str, str]]:
    return [
        (spec_id, name)
        for spec_id, name, kind in TARGETS
        if kind == MODEL and "_devup" in spec_properties(resolve(spec_schema(spec_id)))
    ]


DEVUP_MODELS = _devup_models()


def test_devup_models_are_found() -> None:
    names = {name for _, name in DEVUP_MODELS}
    assert {"ChatCompletion", "ChatCompletionChunk", "ResponseObject", "SpeechResponse"} <= names


@pytest.mark.parametrize(("spec_id", "name"), DEVUP_MODELS, ids=[f"{s}->{n}" for s, n in DEVUP_MODELS])
def test_devup_is_exposed_as_field_devup_with_alias(spec_id: str, name: str) -> None:
    cls = py_object(name)
    fields = cls.model_fields
    assert "devup" in fields, f"{spec_id}: {name} must expose _devup as the field devup"
    assert fields["devup"].alias == "_devup", f"{spec_id}: {name}.devup must keep the alias _devup"
    assert "_devup" not in fields
    # Required-ness and None follow the spec (checked by the required and nullability tests).
    assert py_object("DevUpBilling") in py_members(type_hints(cls)["devup"])


def test_image_params_option_schema_is_exposed_as_schema_() -> None:
    assert "schema" in spec_properties(COMPONENTS["ImageParamsOption"])
    fields = py_object("ImageParamsOption").model_fields
    assert "schema_" in fields
    assert fields["schema_"].alias == "schema"
    assert "schema" not in fields
