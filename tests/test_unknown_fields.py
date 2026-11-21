"""Unknown data survives on every response model (contract §2.1): the SDK never rejects a body.

For each of the 34 response models of contract §2.3 (every "(model)" row and the 4 webhook models), starting from a
minimal valid body built from the spec's required properties:

- an unknown top-level key is kept in ``model_extra`` and in ``to_dict()``;
- an unknown key inside a nested object is kept (the first nested object of the schema, followed to its deepest);
- an unknown value on every string enum field (top-level and inside inline objects) is kept as the string;
- a wrong-typed value does not raise (lenient construct) and the other fields are kept;
- a body that is not a JSON object gives an empty model, without an exception;
- ``_devup`` is exposed as ``.devup`` and written back as ``_devup`` by ``to_dict()``.

``ChatContentPart`` (a Union alias, not a model) is covered through ``ChatCompletionMessage``. A last test sends a
JSON 200 with unknown keys through the client.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import anyio
import pytest
import yaml
from helpers.mock_transport import MODEL_A, REQ_ID, MockAPI, json_response

SPEC_PATH = Path(__file__).resolve().parent.parent / "openapi" / "devupai-v1.yaml"
SPEC: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_bytes())
COMPONENTS: dict[str, Any] = SPEC["components"]["schemas"]
REF_PREFIX = "#/components/schemas/"

WEBHOOK_SUCCESS = "webhooks.inferenceCallback#success"
WEBHOOK_FAILURE = "webhooks.inferenceCallback#failure"
WEBHOOK_USAGE = "webhooks.inferenceCallback#success.usage"
WEBHOOK_ERROR = "webhooks.inferenceCallback#failure.error"

#: (Python model name, spec schema name or webhook shape id): every "(model)" row of contract §2.3 + 4 webhook models.
RESPONSE_MODELS: list[tuple[str, str]] = [
    ("DevUpBilling", "DevUpBilling"),
    ("TokenUsage", "TokenUsage"),
    ("ErrorResponse", "ErrorResponse"),
    ("ChatCompletionMessage", "ChatMessage"),
    ("ChatContentPartText", "ChatContentPartText"),
    ("ChatContentPartImageUrl", "ChatContentPartImageUrl"),
    ("ChatContentPartFile", "ChatContentPartFile"),
    ("ChatCompletion", "ChatCompletionResponse"),
    ("ChatCompletionChunk", "ChatCompletionChunk"),
    ("ChatCompletionStreamError", "ChatCompletionStreamError"),
    ("ResponseObject", "ResponsesResponse"),
    ("ResponseStreamEvent", "ResponsesStreamEvent"),
    ("ResponseStreamErrorEvent", "ResponsesStreamErrorEvent"),
    ("EmbeddingCreateResponse", "EmbeddingResponse"),
    ("RerankCreateResponse", "RerankResponse"),
    ("ImageGenerateResponse", "ImageGenerationResponse"),
    ("ImageParams", "ImageParams"),
    ("ImageParamsOption", "ImageParamsOption"),
    ("ImageParamsInputs", "ImageParamsInputs"),
    ("SpeechResponse", "SpeechResponse"),
    ("TranscriptionResponse", "TranscriptionResponse"),
    ("TranscriptionVerboseResponse", "TranscriptionVerboseResponse"),
    ("VideoGenerationResponse", "VideoGenerationResponse"),
    ("ModelListResponse", "ModelListResponse"),
    ("ModelObject", "ModelObject"),
    ("ModelPricing", "ModelPricing"),
    ("BalanceResponse", "BalanceResponse"),
    ("AccountProfile", "AccountProfile"),
    ("AccountAvatar", "AccountAvatar"),
    ("HealthResponse", "HealthResponse"),
    ("WebhookSuccessPayload", WEBHOOK_SUCCESS),
    ("WebhookErrorPayload", WEBHOOK_FAILURE),
    ("WebhookUsage", WEBHOOK_USAGE),
    ("WebhookErrorDetails", WEBHOOK_ERROR),
]
MODEL_IDS = [name for name, _ in RESPONSE_MODELS]

UNKNOWN_ENUM_VALUE = "new_value"
WirePath = tuple[str | int, ...]


# --------------------------------------------------------------------------------------------------- spec side


def resolve(schema: Any) -> dict[str, Any]:
    """``$ref`` followed and ``allOf`` merged (properties and required united, other keys first-wins)."""
    assert isinstance(schema, dict), f"schema is not an object: {schema!r}"
    while "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith(REF_PREFIX), ref
        schema = COMPONENTS[ref[len(REF_PREFIX) :]]
    if "allOf" not in schema:
        return schema
    merged: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for part in [*schema["allOf"], {k: v for k, v in schema.items() if k != "allOf"}]:
        for key, value in resolve(part).items():
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


def _structural(members: Any) -> bool:
    keys = {"type", "$ref", "properties", "oneOf", "anyOf", "allOf", "enum", "items"}
    return isinstance(members, list) and any(isinstance(m, dict) and keys & m.keys() for m in members)


def members(schema: Any) -> list[dict[str, Any]]:
    """The non-null alternatives at this level, resolved (``oneOf`` / ``anyOf`` flattened, a type list split)."""
    resolved = resolve(schema)
    for key in ("oneOf", "anyOf"):
        if _structural(resolved.get(key)):
            out: list[dict[str, Any]] = []
            for member in resolved[key]:
                out.extend(members(member))
            return out
    declared = resolved.get("type")
    if isinstance(declared, list):
        return [{**resolved, "type": one} for one in declared if one != "null"]
    return [] if declared == "null" else [resolved]


def kind(schema: dict[str, Any]) -> str:
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared
    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "any"


def props(schema: dict[str, Any]) -> dict[str, Any]:
    found = schema.get("properties", {})
    assert isinstance(found, dict)
    return found


def required_of(schema: dict[str, Any]) -> list[str]:
    required = list(schema.get("required", []))
    for key in ("oneOf", "anyOf"):
        fragments = schema.get(key)
        # A constraint-only ``oneOf`` (e.g. ``[{required: [url]}, {required: [b64_json]}]``): satisfy the first one.
        if isinstance(fragments, list) and fragments and not _structural(fragments):
            required.extend(r for r in fragments[0].get("required", []) if r not in required)
    return required


def spec_schema(spec_id: str) -> dict[str, Any]:
    if spec_id in COMPONENTS:
        return resolve(COMPONENTS[spec_id])
    schema = SPEC["webhooks"]["inferenceCallback"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    by_status = {m["properties"]["status"]["enum"][0]: m for m in schema["oneOf"]}
    return {
        WEBHOOK_SUCCESS: by_status["succeeded"],
        WEBHOOK_FAILURE: by_status["failed"],
        WEBHOOK_USAGE: by_status["succeeded"]["properties"]["usage"],
        WEBHOOK_ERROR: by_status["failed"]["properties"]["error"],
    }[spec_id]


def build(schema: Any, name: str = "", depth: int = 0) -> Any:
    """A minimal valid value: required properties only, one item per array, the first alternative of a union."""
    assert depth < 16, f"{name}: nesting too deep"
    alternatives = members(schema)
    if not alternatives:
        return None
    chosen = alternatives[0]
    enum = [v for v in chosen.get("enum", []) if v is not None]
    if enum:
        return enum[0]
    chosen_kind = kind(chosen)
    if chosen_kind == "object":
        properties = props(chosen)
        return {key: build(properties[key], key, depth + 1) for key in required_of(chosen) if key in properties}
    if chosen_kind == "array":
        items = chosen.get("items")
        return [build(items, name, depth + 1)] if isinstance(items, dict) else []
    if chosen_kind == "string":
        if name == "model":
            return MODEL_A
        if name == "request_id":
            return REQ_ID
        if chosen.get("format") == "uri":
            return "https://example.test/file"
        return "x"
    if chosen_kind == "integer":
        return max(1, int(chosen.get("minimum", 1)))
    if chosen_kind == "number":
        return 1.5
    if chosen_kind == "boolean":
        return True
    return {}


def _single_object(schema: Any) -> tuple[WirePath, dict[str, Any]] | None:
    """The steps to the one inline-or-referenced object (with properties) this schema holds, if it holds exactly one."""
    alternatives = members(schema)
    if len(alternatives) != 1:
        return None
    only = alternatives[0]
    if kind(only) == "object" and props(only):
        return (), only
    if kind(only) == "array" and isinstance(only.get("items"), dict):
        inner = _single_object(only["items"])
        if inner is not None:
            return (0, *inner[0]), inner[1]
    return None


def nested_object_path(schema: dict[str, Any], depth: int = 0) -> WirePath | None:
    """The path to the first nested object of a schema, followed to its deepest nested object."""
    if depth > 8:
        return None
    for key, prop in props(schema).items():
        found = _single_object(prop)
        if found is not None:
            steps, child = found
            deeper = nested_object_path(child, depth + 1)
            return (key, *steps, *(deeper or ()))
    return None


def string_enum_paths(schema: dict[str, Any], prefix: WirePath = (), depth: int = 0) -> list[WirePath]:
    """Every string enum field of a schema, top level and inside inline objects / array items (not via ``$ref``)."""
    found: list[WirePath] = []
    if depth > 8:
        return found
    for key, prop in props(schema).items():
        alternatives = members(prop)
        if len(alternatives) != 1:
            continue
        only = alternatives[0]
        if kind(only) == "string" and only.get("enum"):
            found.append((*prefix, key))
        elif kind(only) == "array" and isinstance(only.get("items"), dict):
            items = only["items"]
            item_alternatives = members(items)
            if (
                len(item_alternatives) == 1
                and kind(item_alternatives[0]) == "string"
                and item_alternatives[0].get("enum")
            ):
                found.append((*prefix, key, 0))
            elif "$ref" not in items and len(item_alternatives) == 1 and kind(item_alternatives[0]) == "object":
                found.extend(string_enum_paths(item_alternatives[0], (*prefix, key, 0), depth + 1))
        elif kind(only) == "object" and "$ref" not in prop and "allOf" not in prop:
            found.extend(string_enum_paths(only, (*prefix, key), depth + 1))
    return found


def body_with_path(schema: dict[str, Any], path: WirePath) -> tuple[dict[str, Any], Any]:
    """A minimal body in which every step of ``path`` exists; returns ``(body, the container of the last step)``."""
    body = build(schema)
    assert isinstance(body, dict)
    container: Any = body
    current = schema
    for index, step in enumerate(path):
        alternatives = members(current)
        only = alternatives[0]
        if isinstance(step, int):
            sub = only["items"]
            if not container:
                container.append(build(sub))
        else:
            sub = props(only)[step]
            if step not in container:
                container[step] = build(sub, step)
        if index < len(path) - 1:
            container = container[step]
        current = sub
    return body, container


# ------------------------------------------------------------------------------------------------- Python side


def model_class(name: str) -> Any:
    module = importlib.import_module("devupai.types")
    assert hasattr(module, name), f"devupai.types has no {name!r}"
    return getattr(module, name)


def parse(name: str, data: object) -> Any:
    base = importlib.import_module("devupai.types._base")
    cls = model_class(name)
    result = base.parse_model(cls, data)
    assert isinstance(result, cls), f"parse_model returned {type(result).__name__}, not {name}"
    assert isinstance(result, base.DevupModel)
    return result


def field_name(model: Any, wire: str) -> str:
    for name, info in type(model).model_fields.items():
        if (info.alias or name) == wire:
            return str(name)
    raise AssertionError(f"{type(model).__name__} has no field for {wire!r}")


def attr_at(model: Any, path: WirePath) -> Any:
    """Follows a wire path through parsed models (by field), dicts (by key) and lists (by index)."""
    current = model
    for step in path:
        if isinstance(step, int):
            assert isinstance(current, list), f"{path!r}: expected a list at {step!r}, got {current!r}"
            current = current[step]
        elif isinstance(current, dict):
            current = current[step]
        else:
            current = getattr(current, field_name(current, step))
    return current


def dict_at(data: Any, path: WirePath) -> Any:
    current = data
    for step in path:
        current = current[step]
    return current


def ids(paths: list[tuple[str, str, WirePath]]) -> list[str]:
    return [f"{name}:{'.'.join(str(s) for s in path)}" for name, _, path in paths]


NESTED_CASES: list[tuple[str, str, WirePath]] = [
    (name, spec_id, path)
    for name, spec_id in RESPONSE_MODELS
    if (path := nested_object_path(spec_schema(spec_id))) is not None
]
ENUM_CASES: list[tuple[str, str, WirePath]] = [
    (name, spec_id, path) for name, spec_id in RESPONSE_MODELS for path in string_enum_paths(spec_schema(spec_id))
]
DEVUP_CASES: list[tuple[str, str]] = [
    (name, spec_id) for name, spec_id in RESPONSE_MODELS if "_devup" in props(spec_schema(spec_id))
]


# ------------------------------------------------------------------------------------------------------- tests


def test_the_34_response_models_are_listed() -> None:
    assert len(RESPONSE_MODELS) == 34
    assert len(set(MODEL_IDS)) == 34
    assert len({spec_id for _, spec_id in RESPONSE_MODELS}) == 34


def test_the_case_lists_reach_the_known_fields() -> None:
    nested = {(name, path) for name, _, path in NESTED_CASES}
    assert ("ChatCompletion", ("choices", 0, "message")) in nested
    assert ("ModelObject", ("pricing",)) in nested
    enums = {(name, path) for name, _, path in ENUM_CASES}
    assert ("ChatCompletion", ("choices", 0, "finish_reason")) in enums
    assert ("ModelObject", ("reasoning", "mode")) in enums
    assert ("ImageParams", ("endpoints", 0)) in enums
    assert {name for name, _ in DEVUP_CASES} >= {"ChatCompletion", "RerankCreateResponse", "SpeechResponse"}


@pytest.mark.parametrize(("name", "spec_id"), RESPONSE_MODELS, ids=MODEL_IDS)
def test_the_minimal_body_validates(name: str, spec_id: str) -> None:
    body = build(spec_schema(spec_id))
    model_class(name).model_validate(body)


@pytest.mark.parametrize(("name", "spec_id"), RESPONSE_MODELS, ids=MODEL_IDS)
def test_unknown_top_level_key_survives(name: str, spec_id: str) -> None:
    body = {**build(spec_schema(spec_id)), "unknown_top": 1}
    result = parse(name, body)
    assert result.model_extra is not None and result.model_extra.get("unknown_top") == 1
    assert result.to_dict()["unknown_top"] == 1


@pytest.mark.parametrize(("name", "spec_id", "path"), NESTED_CASES, ids=ids(NESTED_CASES))
def test_unknown_nested_key_survives(name: str, spec_id: str, path: WirePath) -> None:
    schema = spec_schema(spec_id)
    body, container = body_with_path(schema, path)
    target = container[path[-1]]
    assert isinstance(target, dict), f"{name}: {path!r} is not an object in the built body"
    target["unknown_nested"] = {"deep": [1, 2]}
    result = parse(name, copy.deepcopy(body))
    nested = attr_at(result, path)
    assert getattr(nested, "model_extra", None), f"{name}.{path!r}: expected a nested model, got {nested!r}"
    assert nested.model_extra["unknown_nested"] == {"deep": [1, 2]}
    assert dict_at(result.to_dict(), path)["unknown_nested"] == {"deep": [1, 2]}


@pytest.mark.parametrize(("name", "spec_id", "path"), ENUM_CASES, ids=ids(ENUM_CASES))
def test_unknown_enum_value_survives_as_the_string(name: str, spec_id: str, path: WirePath) -> None:
    schema = spec_schema(spec_id)
    body, container = body_with_path(schema, path)
    container[path[-1]] = UNKNOWN_ENUM_VALUE
    result = parse(name, copy.deepcopy(body))
    assert attr_at(result, path) == UNKNOWN_ENUM_VALUE
    assert dict_at(result.to_dict(), path) == UNKNOWN_ENUM_VALUE


def _wrong_type_target(schema: dict[str, Any]) -> tuple[str, object]:
    """The property to corrupt (an object one first, else a non-string one, else the first) and its wrong value."""
    properties = props(schema)
    kinds = {key: {kind(m) for m in members(prop)} for key, prop in properties.items()}
    for wanted in ({"object"}, {"integer", "number", "boolean", "array"}):
        for key, found in kinds.items():
            if found & wanted:
                return key, "wrong-type"
    first = next(iter(properties))
    return first, {"wrong": "type"}


@pytest.mark.parametrize(("name", "spec_id"), RESPONSE_MODELS, ids=MODEL_IDS)
def test_wrong_typed_value_does_not_raise_and_keeps_the_other_fields(name: str, spec_id: str) -> None:
    schema = spec_schema(spec_id)
    key, wrong = _wrong_type_target(schema)
    body = {**build(schema), "unknown_top": 1}
    body[key] = wrong
    result = parse(name, copy.deepcopy(body))
    assert result.model_extra is not None and result.model_extra.get("unknown_top") == 1
    for other, value in body.items():
        if other in (key, "unknown_top"):
            continue
        kept = attr_at(result, (other,))
        if isinstance(value, str | int | float | bool):
            assert kept == value, f"{name}.{other}: {kept!r} != {value!r}"
        else:
            assert kept is not None, f"{name}.{other} was dropped"


@pytest.mark.parametrize(("name", "spec_id"), RESPONSE_MODELS, ids=MODEL_IDS)
def test_non_object_body_gives_an_empty_model(name: str, spec_id: str) -> None:
    for value in (None, "text", [1, 2], 42, True, 1.5):
        result = parse(name, value)
        assert result.to_dict() == {}, f"{name} from {value!r}"
        assert not result.model_extra


@pytest.mark.parametrize(("name", "spec_id"), DEVUP_CASES, ids=[name for name, _ in DEVUP_CASES])
def test_devup_round_trips(name: str, spec_id: str) -> None:
    billing = {"cost_dzd": 1.5, "balance_dzd": 10.25, "request_id": REQ_ID, "warnings": ["w"], "billing_extra": "kept"}
    body = {**build(spec_schema(spec_id)), "_devup": billing}
    result = parse(name, copy.deepcopy(body))
    devup = result.devup
    assert isinstance(devup, model_class("DevUpBilling"))
    assert devup.cost_dzd == 1.5
    assert devup.balance_dzd == 10.25
    assert devup.request_id == REQ_ID
    assert devup.model_extra is not None and devup.model_extra["billing_extra"] == "kept"
    data = result.to_dict()
    assert "devup" not in data
    assert data["_devup"] == billing
    assert parse(name, data).to_dict() == data


def _chat_body() -> dict[str, Any]:
    schema = spec_schema("ChatCompletionResponse")
    body, choice = body_with_path(schema, ("choices", 0, "message"))
    choice["message"]["unknown_nested"] = 7
    choice["finish_reason"] = "new_reason"
    body["unknown_top"] = {"kept": True}
    return body


def test_unknown_keys_reach_the_caller_through_the_client(client_kind: str) -> None:
    api = MockAPI(lambda request: json_response(_chat_body()))
    client: Any = api.client(client_kind)
    messages = [{"role": "user", "content": "hi"}]
    if client_kind == "sync":
        result: Any = client.chat.completions.create(model=MODEL_A, messages=messages)
    else:

        async def main() -> Any:
            return await client.chat.completions.create(model=MODEL_A, messages=messages)

        result = anyio.run(main)
    assert len(api.calls) == 1
    assert isinstance(result, model_class("ChatCompletion"))
    assert result.model_extra["unknown_top"] == {"kept": True}
    assert result.choices[0].finish_reason == "new_reason"
    assert result.choices[0].message.model_extra["unknown_nested"] == 7
    assert result.to_dict()["unknown_top"] == {"kept": True}
