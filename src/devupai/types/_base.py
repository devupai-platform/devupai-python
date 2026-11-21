"""The base class of every response model, and the parser that never rejects a body."""

import types
import typing
from collections.abc import Mapping, Sequence
from typing import TypeVar, Union, cast, get_args, get_origin

import pydantic
from pydantic import ConfigDict, TypeAdapter, ValidationError

__all__ = ["DevupModel", "parse_model"]


class DevupModel(pydantic.BaseModel):
    """A response object of the API.

    Keys the SDK does not know yet are kept: read them with ``model_extra`` or ``to_dict()``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())

    def to_dict(self) -> dict[str, object]:
        """The object as JSON-ready data, with the wire names (for example ``_devup``) and only the keys received."""
        data: dict[str, object] = self.model_dump(mode="json", by_alias=True, exclude_unset=True, warnings=False)
        return data


M = TypeVar("M", bound=DevupModel)

_UNION_ORIGINS: tuple[object, ...] = (Union, types.UnionType)
_LIST_ORIGINS: tuple[object, ...] = (list, Sequence, typing.Sequence)
_DICT_ORIGINS: tuple[object, ...] = (dict, Mapping, typing.Mapping)


def parse_model(cls: type[M], data: object) -> M:
    """Builds ``cls`` from decoded JSON and never raises.

    A body that validates gives a validated model. Otherwise each declared field present in ``data`` is kept: a value
    that matches its annotation is validated, a nested object is built the same way, and any other value stays as
    received. Unknown keys are kept as extras. A body that is not a JSON object gives an empty model.
    """
    if not isinstance(data, dict):
        return _fill_missing(cls.model_construct())
    body = cast(dict[str, object], data)
    try:
        return cls.model_validate(body)
    except ValidationError:
        return _lenient(cls, body)


def _lenient(cls: type[M], data: dict[str, object]) -> M:
    names: dict[str, str] = {}
    for name, info in cls.model_fields.items():
        names.setdefault(name, name)
        if info.alias is not None:
            names[info.alias] = name
    values: dict[str, object] = {}
    extras: dict[str, object] = {}
    for key, value in data.items():
        field = names.get(key)
        if field is None:
            extras[key] = value
        else:
            values[field] = _lenient_value(cls.model_fields[field].annotation, value)
    instance = cls.model_construct(_fields_set=set(values), **values)
    if extras:
        # Set after construction so that no received key can clash with a keyword of ``model_construct``.
        if instance.__pydantic_extra__ is None:
            instance.__pydantic_extra__ = {}
        instance.__pydantic_extra__.update(extras)
    return _fill_missing(instance)


def _fill_missing(instance: M) -> M:
    """Sets every declared field that was not received and has no default to ``None``.

    The value is written to the instance ``__dict__`` directly, so the field is not marked as set: ``to_dict()``
    leaves it out, and reading it gives ``None`` instead of an ``AttributeError``.
    """
    values = vars(instance)  # the instance ``__dict__`` itself
    for name in type(instance).model_fields:
        if name not in values:
            values[name] = None
    return instance


def _members(annotation: object) -> list[object]:
    """The flattened members of a union annotation (the annotation itself when it is not a union)."""
    if get_origin(annotation) in _UNION_ORIGINS:
        out: list[object] = []
        for arg in get_args(annotation):
            out.extend(_members(arg))
        return out
    return [annotation]


def _is_model(member: object) -> bool:
    return isinstance(member, type) and issubclass(member, DevupModel)


#: One validator per field annotation, built on first use (field annotations are a small, fixed set).
_ADAPTERS: dict[object, TypeAdapter[object]] = {}


def _adapter(annotation: object) -> TypeAdapter[object]:
    # A field annotation is a type form; ``object`` here only keeps this helper free of ``Any``.
    form = cast(type[object], annotation)
    try:
        found = _ADAPTERS.get(annotation)
    except TypeError:  # an unhashable annotation: build its validator each time
        return TypeAdapter(form)
    if found is None:
        found = TypeAdapter(form)
        _ADAPTERS[annotation] = found
    return found


def _validates(annotation: object, value: object) -> tuple[bool, object]:
    adapter = _adapter(annotation)
    try:
        return True, adapter.validate_python(value)
    except ValidationError:
        return False, value


def _lenient_value(annotation: object, value: object) -> object:
    """``value`` validated against ``annotation`` when it matches, else built leniently (nested objects and the items
    of arrays and maps), else unchanged."""
    if annotation is None:
        return value
    ok, validated = _validates(annotation, value)
    if ok:
        return validated
    members = _members(annotation)
    if isinstance(value, dict):
        body = cast(dict[str, object], value)
        models = [cast(type[DevupModel], m) for m in members if _is_model(m)]
        if models:
            return _pick(models, body, keep_dict=any(get_origin(m) in _DICT_ORIGINS or m is dict for m in members))
        for member in members:
            args = get_args(member)
            if get_origin(member) in _DICT_ORIGINS and len(args) == 2:
                return {key: _lenient_value(args[1], item) for key, item in body.items()}
        return body
    if isinstance(value, list):
        items = cast(list[object], value)
        for member in members:
            args = get_args(member)
            if get_origin(member) in _LIST_ORIGINS and len(args) == 1:
                return [_lenient_value(args[0], item) for item in items]
        return items
    return value


def _pick(models: list[type[DevupModel]], body: dict[str, object], *, keep_dict: bool) -> object:
    """One model of the union: the first one the body validates against, else the plain object when the union has an
    open member, else the first model built leniently."""
    if len(models) == 1:
        return parse_model(models[0], body)
    for model in models:
        try:
            return model.model_validate(body)
        except ValidationError:
            continue
    if keep_dict:
        return body
    return parse_model(models[0], body)
