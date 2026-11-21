"""Which retry policy each API method uses (decisions B item 4, contract §3.3 and §6).

The methods are found by introspection of the client, never from a hand-kept list, so a new method that is not
bound to a policy (or has no sample) fails here. Exactly the 11 billed POST methods are ``billed=True`` and send with
``BILLED_POST_RETRY``; every GET is ``billed=False`` and sends with ``DEFAULT_RETRY``. The policy is checked twice:
from the ``retry`` argument the client's request entry receives, and from what the mock sees (a 502 is sent again
only under ``DEFAULT_RETRY``).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from helpers.mock_transport import MockAPI, error_response, json_response, sequence
from helpers.samples import (
    BILLED_POST_METHODS,
    GET_METHODS,
    IMPLEMENTED,
    SAMPLES,
    call_sample,
    discover,
    sample_kind,
    sample_response,
    split_name,
)

from devupai import AsyncDevupAI, DevupAI, DevupAPIError


def _client_module() -> Any:
    return importlib.import_module("devupai._client")


def make_client(kind: str, *entries: Any, **opts: Any) -> tuple[MockAPI, DevupAI | AsyncDevupAI]:
    api = MockAPI(sequence(*entries) if entries else sequence(json_response({})))
    return api, api.client(kind, **opts)


def spy_request(monkeypatch: pytest.MonkeyPatch, client: DevupAI | AsyncDevupAI) -> list[dict[str, Any]]:
    """Wraps the client's request entry on the instance; each call's keyword arguments are kept."""
    captured: list[dict[str, Any]] = []
    original = client._request

    if isinstance(client, AsyncDevupAI):

        async def async_spy(*args: Any, **kwargs: Any) -> Any:
            captured.append(kwargs)
            return await original(*args, **kwargs)

        monkeypatch.setattr(client, "_request", async_spy)
    else:

        def sync_spy(*args: Any, **kwargs: Any) -> Any:
            captured.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(client, "_request", sync_spy)
    return captured


def test_the_sample_lists_are_consistent() -> None:
    assert len(BILLED_POST_METHODS) == 11
    assert len(GET_METHODS) == 7
    assert not BILLED_POST_METHODS & GET_METHODS
    assert set(SAMPLES) == BILLED_POST_METHODS | GET_METHODS
    for name, sample in SAMPLES.items():
        assert sample.http_method == ("POST" if name in BILLED_POST_METHODS else "GET"), name


def test_discovery_finds_every_method_and_nothing_else(client_kind: str) -> None:
    _, client = make_client(client_kind)
    found = discover(client)
    missing = sorted(set(found) - set(SAMPLES))
    unknown = sorted(set(SAMPLES) - set(found))
    assert not missing, f"decorated methods without a SAMPLES entry: {missing}"
    assert not unknown, f"SAMPLES entries that name no decorated method: {unknown}"


def test_the_post_methods_are_exactly_the_billed_list(client_kind: str) -> None:
    _, client = make_client(client_kind)
    found = discover(client)
    posts = {name for name, item in found.items() if item.endpoint.method == "POST"}
    gets = {name for name, item in found.items() if item.endpoint.method == "GET"}
    assert posts == BILLED_POST_METHODS
    assert gets == GET_METHODS
    assert posts | gets == set(found)


def test_every_post_is_billed_and_every_get_is_not(client_kind: str) -> None:
    _, client = make_client(client_kind)
    for name, item in discover(client).items():
        endpoint = item.endpoint
        if endpoint.method == "POST":
            assert endpoint.billed is True, name
        else:
            assert endpoint.method == "GET", name
            assert endpoint.billed is False, name


def test_the_endpoint_metadata_is_an_endpoint(client_kind: str) -> None:
    base = importlib.import_module("devupai.resources._base")
    _, client = make_client(client_kind)
    for name, item in discover(client).items():
        assert isinstance(item.endpoint, base.Endpoint), name
        assert item.endpoint.root == (name == "health.check"), name


def test_with_response_twins_carry_the_same_metadata(client_kind: str) -> None:
    _, client = make_client(client_kind)
    for name, item in discover(client).items():
        _, method = split_name(name)
        twin = getattr(item.resource.with_response, method)
        assert getattr(twin, "__devup_endpoint__", None) == item.endpoint, name


def test_the_sync_and_async_clients_agree() -> None:
    _, sync_client = make_client("sync")
    _, async_client = make_client("async")
    sync_found = discover(sync_client)
    async_found = discover(async_client)
    assert set(sync_found) == set(async_found)
    for name, item in sync_found.items():
        assert item.endpoint == async_found[name].endpoint, name


def _expected_policy(name: str) -> Any:
    module = _client_module()
    return module.BILLED_POST_RETRY if name in BILLED_POST_METHODS else module.DEFAULT_RETRY


@pytest.mark.parametrize("with_response", [False, True], ids=["plain", "with_response"])
@pytest.mark.parametrize("name", IMPLEMENTED)
def test_the_request_entry_receives_the_bound_policy(
    monkeypatch: pytest.MonkeyPatch, client_kind: str, name: str, with_response: bool
) -> None:
    _, client = make_client(client_kind, sample_response(name))
    captured = spy_request(monkeypatch, client)
    call_sample(client, name, with_response=with_response)
    assert len(captured) == 1, f"{name} did not send through the client's request entry exactly once"
    assert captured[0].get("retry") == _expected_policy(name)
    # The key is left out for health and for images.download (Node's rule); every other method sends it.
    assert captured[0].get("auth", True) is SAMPLES[name].auth
    assert captured[0].get("auth", True) is (name not in ("health.check", "images.download"))
    assert captured[0].get("kind", "json") == sample_kind(name)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_a_502_is_sent_again_only_for_get(client_kind: str, name: str) -> None:
    api, client = make_client(client_kind, error_response(502), sample_response(name))
    if name in BILLED_POST_METHODS:
        with pytest.raises(DevupAPIError) as caught:
            call_sample(client, name)
        assert caught.value.status == 502
        assert len(api.calls) == 1
    else:
        call_sample(client, name)
        assert len(api.calls) == 2


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_a_503_is_sent_again_on_both_policies(client_kind: str, name: str) -> None:
    api, client = make_client(client_kind, error_response(503), sample_response(name))
    call_sample(client, name)
    assert len(api.calls) == 2


def test_every_billed_method_has_a_policy_check() -> None:
    # No method is bound by metadata only any more (P3 removed the placeholders): every billed POST and every GET
    # is called by the policy tests above.
    assert sorted(BILLED_POST_METHODS - set(IMPLEMENTED)) == []
    assert sorted(GET_METHODS - set(IMPLEMENTED)) == []
    assert set(IMPLEMENTED) == BILLED_POST_METHODS | GET_METHODS
    assert len(IMPLEMENTED) == 18
    for name in ("audio.transcriptions.create", "video.edits.create", "images.download", "me.avatar"):
        assert name in IMPLEMENTED, name
