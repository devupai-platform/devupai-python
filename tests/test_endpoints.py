"""Every path the SDK can call exists in the vendored spec with the same HTTP method (decisions B item 4).

The ``Endpoint`` of each decorated method (found by introspection) is checked against the ``paths`` of
``openapi/devupai-v1.yaml``: a template is relative to the versioned base (``/v1``), or to the host root when
``root=True``. For each method, the path actually sent must match its template. Every operation of the spec, the
Messages-compatible one aside, is covered by at least one SDK method.

``images.download`` takes an absolute URL (contract §3.3), so its template is not looked up; the spec's
``proxyImage`` operation counts as covered by it, and the path it sends for the sample URL must be that operation's.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml
from helpers.mock_transport import MockAPI, json_response, sequence
from helpers.samples import IMPLEMENTED, SAMPLES, call_sample, discover, sample_response, sample_url

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "openapi" / "devupai-v1.yaml"
VERSION_PREFIX = "/v1"
HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
#: The Messages-compatible operation is out of scope for this SDK.
EXCLUDED_OPERATION_IDS = frozenset({"createMessage"})
#: Methods whose URL is given whole by the caller; their spec operation id.
ABSOLUTE_URL_METHODS = {"images.download": "proxyImage"}


@cache
def spec_operations() -> dict[tuple[str, str], str]:
    """``(METHOD, spec path) → operationId`` for every in-scope operation of the vendored spec."""
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    document = yaml.load(SPEC.read_text(encoding="utf-8"), Loader=loader)  # a safe loader only
    operations: dict[tuple[str, str], str] = {}
    for path, item in document["paths"].items():
        for method in HTTP_METHODS:
            operation = item.get(method)
            if operation is None:
                continue
            operation_id = operation.get("operationId", "")
            if operation_id in EXCLUDED_OPERATION_IDS:
                continue
            operations[(method.upper(), path)] = operation_id
    return operations


def spec_path(endpoint: Any) -> str:
    """The spec path of an SDK ``Endpoint`` template."""
    return endpoint.path if endpoint.root else VERSION_PREFIX + endpoint.path


def template_regex(template: str) -> re.Pattern[str]:
    """``/v1/models/{model}`` → a pattern where each ``{name}`` is one non-empty path segment."""
    parts = re.split(r"(\{[^}/]+\})", template)
    pattern = "".join("[^/]+" if part.startswith("{") else re.escape(part) for part in parts)
    return re.compile(pattern)


def make_client(kind: str, *entries: Any) -> tuple[MockAPI, Any]:
    api = MockAPI(sequence(*entries) if entries else sequence(json_response({})))
    return api, api.client(kind)


def test_the_spec_has_17_in_scope_operations() -> None:
    operations = spec_operations()
    assert len(operations) == 17
    assert all(path.startswith(VERSION_PREFIX + "/") or path == "/health" for _, path in operations)
    assert set(ABSOLUTE_URL_METHODS.values()) <= set(operations.values())


def test_the_template_regex() -> None:
    pattern = template_regex("/v1/models/{model}")
    assert pattern.fullmatch("/v1/models/fixture%2Fmodel-a")
    assert not pattern.fullmatch("/v1/models/fixture/model-a")
    assert not pattern.fullmatch("/v1/models/")
    assert template_regex("/health").fullmatch("/health")


def test_every_decorated_method_exists_in_the_spec_with_the_same_method(client_kind: str) -> None:
    _, client = make_client(client_kind)
    operations = spec_operations()
    found = discover(client)
    assert found, "no decorated method was found on the client"
    for name, item in found.items():
        endpoint = item.endpoint
        assert endpoint.method in ("GET", "POST"), name
        if name in ABSOLUTE_URL_METHODS:
            expected_id = ABSOLUTE_URL_METHODS[name]
            assert any(op == expected_id and method == endpoint.method for (method, _), op in operations.items()), name
            continue
        assert endpoint.path.startswith("/"), f"{name}: {endpoint.path!r} is not a path template"
        key = (endpoint.method, spec_path(endpoint))
        assert key in operations, f"{name}: {key} is not an operation of the vendored spec"


def test_only_health_is_at_the_host_root(client_kind: str) -> None:
    _, client = make_client(client_kind)
    for name, item in discover(client).items():
        assert item.endpoint.root == (name == "health.check"), name
        assert not item.endpoint.path.startswith(VERSION_PREFIX + "/"), f"{name}: the template repeats /v1"


def test_every_spec_operation_is_covered_by_an_sdk_method(client_kind: str) -> None:
    _, client = make_client(client_kind)
    operations = spec_operations()
    found = discover(client)
    covered: set[str] = set()
    for name, item in found.items():
        if name in ABSOLUTE_URL_METHODS:
            covered.add(ABSOLUTE_URL_METHODS[name])
            continue
        key = (item.endpoint.method, spec_path(item.endpoint))
        if key in operations:
            covered.add(operations[key])
    missing = sorted(set(operations.values()) - covered)
    assert not missing, f"spec operations without an SDK method: {missing}"
    assert len(covered) == 17


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_each_sample_path_matches_its_template(client_kind: str, name: str) -> None:
    _, client = make_client(client_kind)
    item = discover(client).get(name)
    assert item is not None, f"{name} is not a decorated method of the client"
    sample = SAMPLES[name]
    assert sample.http_method == item.endpoint.method
    if name in ABSOLUTE_URL_METHODS:
        return
    assert template_regex(spec_path(item.endpoint)).fullmatch(sample.path), (sample.path, item.endpoint.path)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_the_path_sent_matches_the_endpoint_template(client_kind: str, name: str) -> None:
    sample = SAMPLES[name]
    api, client = make_client(client_kind, sample_response(name))
    endpoint = discover(client)[name].endpoint
    call_sample(client, name)
    assert len(api.calls) == 1
    sent = api.calls[0]
    assert sent.method == endpoint.method
    sent_path = urlsplit(sent.url).path
    assert sent_path == sample.path
    if name in ABSOLUTE_URL_METHODS:
        # The URL is sent as given: its path is the path of the operation it stands for.
        assert sent.url == sample_url(name)
        assert spec_operations().get((sent.method, sent_path)) == ABSOLUTE_URL_METHODS[name]
        return
    assert template_regex(spec_path(endpoint)).fullmatch(sent_path), (sent_path, endpoint.path)
    assert (endpoint.method, spec_path(endpoint)) in spec_operations()
