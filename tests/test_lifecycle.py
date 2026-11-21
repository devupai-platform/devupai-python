"""Client lifecycle: context managers, close(), caller-owned HTTP clients and the default HTTP client."""

from __future__ import annotations

from typing import Any

import anyio
import httpx
import pytest
from helpers.mock_transport import BASE, TEST_KEY, LiveNetworkCall, MockAPI, json_response, run_request

from devupai import APIConnectionError, AsyncDevupAI, DevupAI

CLOSED_MESSAGE = "The client is closed."


def ok_handler(request: httpx.Request) -> httpx.Response:
    return json_response({"ok": True})


@pytest.fixture
def created_http_clients(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Client | httpx.AsyncClient]:
    """Records every httpx.Client / httpx.AsyncClient built during the test (through the public constructor)."""
    created: list[httpx.Client | httpx.AsyncClient] = []
    for cls in (httpx.Client, httpx.AsyncClient):
        original = cls.__init__

        def spy(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> None:
            created.append(self)
            _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", spy)
    return created


def close_client(client: DevupAI | AsyncDevupAI) -> None:
    if isinstance(client, AsyncDevupAI):
        anyio.run(client.close)
    else:
        client.close()


# --- sync client -------------------------------------------------------------------------------------------------


def test_with_statement_returns_the_client_and_closes_the_owned_http_client(
    created_http_clients: list[httpx.Client | httpx.AsyncClient],
) -> None:
    client = DevupAI(api_key=TEST_KEY, base_url=BASE)
    with client as entered:
        assert entered is client
    owned = [c for c in created_http_clients if isinstance(c, httpx.Client)]
    assert len(owned) == 1
    assert owned[0].is_closed


def test_with_statement_closes_the_client_when_the_block_raises(
    created_http_clients: list[httpx.Client | httpx.AsyncClient],
) -> None:
    client = DevupAI(api_key=TEST_KEY, base_url=BASE)
    with pytest.raises(KeyError):
        with client:
            raise KeyError("fixture")
    with pytest.raises(RuntimeError) as info:
        client._request("GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE
    assert all(c.is_closed for c in created_http_clients)


def test_close_is_idempotent() -> None:
    client = DevupAI(api_key=TEST_KEY, base_url=BASE)
    client.close()
    client.close()
    with pytest.raises(RuntimeError):
        client._request("GET", "/models")


def test_close_never_closes_a_caller_http_client() -> None:
    http = httpx.Client(transport=httpx.MockTransport(ok_handler))
    with DevupAI(api_key=TEST_KEY, base_url=BASE, http_client=http) as client:
        assert run_request(client, "GET", "/models").data == {"ok": True}
    client.close()
    assert not http.is_closed
    assert http.get(f"{BASE}/models").json() == {"ok": True}
    http.close()


def test_use_after_close_raises_runtime_error_and_sends_nothing(client_kind: str) -> None:
    api = MockAPI(ok_handler)
    client = api.client(client_kind)
    run_request(client, "GET", "/models")
    close_client(client)
    with pytest.raises(RuntimeError) as info:
        run_request(client, "GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE
    assert not isinstance(info.value, APIConnectionError)
    assert len(api.calls) == 1


def test_use_after_the_with_statement_raises_runtime_error() -> None:
    api = MockAPI(ok_handler)
    with api.sync_client() as client:
        run_request(client, "GET", "/models")
    with pytest.raises(RuntimeError) as info:
        run_request(client, "GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE
    assert len(api.calls) == 1


def test_default_http_client_does_not_follow_redirects_and_trusts_the_environment(
    created_http_clients: list[httpx.Client | httpx.AsyncClient], fake_clock: Any
) -> None:
    client = DevupAI(api_key=TEST_KEY, base_url=BASE)
    # The default transport is the real one, which the test guard turns into LiveNetworkCall. It is a
    # BaseException, so the SDK neither wraps nor retries it.
    with pytest.raises(LiveNetworkCall):
        client._request("GET", "/models")
    assert fake_clock.sleeps == []
    owned = [c for c in created_http_clients if isinstance(c, httpx.Client)]
    assert len(owned) == 1
    assert owned[0].follow_redirects is False
    assert owned[0].trust_env is True
    assert owned[0].timeout == httpx.Timeout(None)
    client.close()
    assert owned[0].is_closed


# --- async client ------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_with_returns_the_client_and_closes_the_owned_http_client(
    created_http_clients: list[httpx.Client | httpx.AsyncClient],
) -> None:
    client = AsyncDevupAI(api_key=TEST_KEY, base_url=BASE)
    async with client as entered:
        assert entered is client
    owned = [c for c in created_http_clients if isinstance(c, httpx.AsyncClient)]
    assert len(owned) == 1
    assert owned[0].is_closed
    with pytest.raises(RuntimeError) as info:
        await client._request("GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE


@pytest.mark.anyio
async def test_async_with_closes_the_client_when_the_block_raises() -> None:
    client = AsyncDevupAI(api_key=TEST_KEY, base_url=BASE)
    with pytest.raises(KeyError):
        async with client:
            raise KeyError("fixture")
    with pytest.raises(RuntimeError) as info:
        await client._request("GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE


@pytest.mark.anyio
async def test_async_close_is_idempotent() -> None:
    client = AsyncDevupAI(api_key=TEST_KEY, base_url=BASE)
    await client.close()
    await client.close()
    with pytest.raises(RuntimeError):
        await client._request("GET", "/models")


@pytest.mark.anyio
async def test_async_close_never_closes_a_caller_http_client() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(ok_handler))
    async with AsyncDevupAI(api_key=TEST_KEY, base_url=BASE, http_client=http) as client:
        result = await client._request("GET", "/models")
        assert result.data == {"ok": True}
    await client.close()
    assert not http.is_closed
    assert (await http.get(f"{BASE}/models")).json() == {"ok": True}
    await http.aclose()


@pytest.mark.anyio
async def test_async_use_after_close_raises_runtime_error_and_sends_nothing() -> None:
    api = MockAPI(ok_handler)
    client = api.async_client()
    await client._request("GET", "/models")
    await client.close()
    with pytest.raises(RuntimeError) as info:
        await client._request("GET", "/models")
    assert str(info.value) == CLOSED_MESSAGE
    assert len(api.calls) == 1


@pytest.mark.anyio
async def test_async_default_http_client_does_not_follow_redirects_and_trusts_the_environment(
    created_http_clients: list[httpx.Client | httpx.AsyncClient], fake_clock: Any
) -> None:
    client = AsyncDevupAI(api_key=TEST_KEY, base_url=BASE)
    with pytest.raises(LiveNetworkCall):
        await client._request("GET", "/models")
    assert fake_clock.sleeps == []
    owned = [c for c in created_http_clients if isinstance(c, httpx.AsyncClient)]
    assert len(owned) == 1
    assert owned[0].follow_redirects is False
    assert owned[0].trust_env is True
    assert owned[0].timeout == httpx.Timeout(None)
    await client.close()
    assert owned[0].is_closed
