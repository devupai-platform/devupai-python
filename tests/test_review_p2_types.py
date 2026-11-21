"""Review P2, finding R2-4: a required field missing from a body is read as ``None`` after the lenient parse.

``parse_model`` falls back to a lenient construct when a body does not validate. A declared field that is absent and
has no default must then read as ``None`` (never an ``AttributeError``), stay out of ``to_dict()`` (it was not
received), and leave the present fields and the unknown keys as received.
"""

from __future__ import annotations

from typing import Any

import pytest

from devupai.types import (
    BalanceResponse,
    ChatCompletion,
    DevupModel,
    ImageGenerateResponse,
    ModelObject,
    WebhookSuccessPayload,
    parse_model,
)

MODEL_ID = "fixture/model-a"

#: (model, body, the required fields missing from the body, the fields present with their expected values).
CASES: list[tuple[type[DevupModel], dict[str, Any], tuple[str, ...], dict[str, Any]]] = [
    (
        ChatCompletion,
        # ``choices`` missing; ``created`` wrong-typed as well.
        {"id": "chatcmpl-1", "object": "chat.completion", "created": "soon", "model": MODEL_ID, "extra_key": 1},
        ("choices",),
        {"id": "chatcmpl-1", "object": "chat.completion", "created": "soon", "model": MODEL_ID},
    ),
    (
        ModelObject,
        {"id": MODEL_ID, "object": "model", "name": "Model A", "extra_key": {"a": [1]}},
        ("created", "owned_by", "type", "context_window", "is_replaced", "pricing", "supports"),
        {"id": MODEL_ID, "object": "model", "name": "Model A"},
    ),
    (
        ImageGenerateResponse,
        # The required ``_devup`` is missing.
        {"data": [{"url": "https://example.test/image.png"}], "created": 1, "extra_key": "kept"},
        ("devup",),
        {"created": 1},
    ),
    (
        WebhookSuccessPayload,
        {"id": "whd_1", "status": "succeeded", "cost_dzd": "wrong-type", "results": [], "extra_key": True},
        ("usage", "settlement"),
        {"id": "whd_1", "status": "succeeded", "cost_dzd": "wrong-type", "results": []},
    ),
    (
        BalanceResponse,
        {"extra_key": 2.5},
        ("balance_dzd",),
        {},
    ),
]
IDS = [cls.__name__ for cls, _, _, _ in CASES]


def _wire_name(cls: type[DevupModel], field: str) -> str:
    info = cls.model_fields[field]
    return info.alias or field


@pytest.mark.parametrize(("cls", "body", "missing", "present"), CASES, ids=IDS)
def test_missing_required_fields_read_as_none(
    cls: type[DevupModel], body: dict[str, Any], missing: tuple[str, ...], present: dict[str, Any]
) -> None:
    for field in missing:
        assert cls.model_fields[field].is_required(), f"{cls.__name__}.{field} must be a required field"
    result = parse_model(cls, dict(body))
    assert isinstance(result, cls)
    for field in missing:
        assert getattr(result, field) is None, f"{cls.__name__}.{field}"


@pytest.mark.parametrize(("cls", "body", "missing", "present"), CASES, ids=IDS)
def test_present_fields_keep_their_values(
    cls: type[DevupModel], body: dict[str, Any], missing: tuple[str, ...], present: dict[str, Any]
) -> None:
    result = parse_model(cls, dict(body))
    for field, value in present.items():
        assert getattr(result, field) == value, f"{cls.__name__}.{field}"


@pytest.mark.parametrize(("cls", "body", "missing", "present"), CASES, ids=IDS)
def test_to_dict_leaves_the_missing_fields_out(
    cls: type[DevupModel], body: dict[str, Any], missing: tuple[str, ...], present: dict[str, Any]
) -> None:
    data = parse_model(cls, dict(body)).to_dict()
    for field in missing:
        assert _wire_name(cls, field) not in data, f"{cls.__name__}.{field}"
        assert field not in data, f"{cls.__name__}.{field}"
    for field, value in present.items():
        assert data[_wire_name(cls, field)] == value
    assert data["extra_key"] == body["extra_key"]


@pytest.mark.parametrize(("cls", "body", "missing", "present"), CASES, ids=IDS)
def test_unknown_keys_stay_in_model_extra(
    cls: type[DevupModel], body: dict[str, Any], missing: tuple[str, ...], present: dict[str, Any]
) -> None:
    result = parse_model(cls, dict(body))
    assert result.model_extra is not None
    assert result.model_extra == {"extra_key": body["extra_key"]}
    assert not set(missing) & result.model_fields_set


@pytest.mark.parametrize("cls", [cls for cls, _, _, _ in CASES], ids=IDS)
def test_a_non_object_body_reads_required_fields_as_none(cls: type[DevupModel]) -> None:
    result = parse_model(cls, ["not", "an", "object"])
    for field, info in cls.model_fields.items():
        if info.is_required():
            assert getattr(result, field) is None, f"{cls.__name__}.{field}"
    assert result.to_dict() == {}
    assert not result.model_extra
