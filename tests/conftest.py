"""Shared fixtures: no live network, a fake clock, and the sync/async client switch.

``devupai`` is never imported at module level here, so tests that do not need the package still run.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from helpers.mock_transport import LiveNetworkCall


class FakeClock:
    """A clock that only moves when a test (or a patched sleep) moves it."""

    def __init__(self) -> None:
        self.monotonic: float = 1000.0
        self.wall: float = 1_800_000_000.0
        self.sleeps: list[float] = []
        self.random_value: float = 0.0

    def advance(self, seconds: float) -> None:
        """Moves both the monotonic and the wall clock forward."""
        self.monotonic += seconds
        self.wall += seconds

    def sleep(self, seconds: float) -> None:
        """Records the sleep and advances the clock, without waiting."""
        self.sleeps.append(seconds)
        self.advance(seconds)

    async def async_sleep(self, seconds: float) -> None:
        """The async twin of :meth:`sleep`."""
        self.sleep(seconds)

    def random(self) -> float:
        """Returns ``random_value``."""
        return self.random_value


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every real transport raises ``LiveNetworkCall``."""

    def blocked(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise LiveNetworkCall(f"A test tried a live network call: {request.method} {request.url}")

    async def blocked_async(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        raise LiveNetworkCall(f"A test tried a live network call: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeClock]:
    """Patches the SDK's sleep, random and clocks. Patches nothing when ``devupai`` cannot be imported."""
    clock = FakeClock()
    try:
        import devupai._client as client_module
        import devupai._errors as errors_module
    except ImportError:
        yield clock
        return
    monkeypatch.setattr(client_module, "_sleep", clock.sleep)
    monkeypatch.setattr(client_module, "_async_sleep", clock.async_sleep)
    monkeypatch.setattr(client_module, "_random", clock.random)
    monkeypatch.setattr(client_module, "_monotonic", lambda: clock.monotonic)
    monkeypatch.setattr(client_module, "_time", lambda: clock.wall)
    monkeypatch.setattr(errors_module, "_time", lambda: clock.wall)
    yield clock


@pytest.fixture(params=["sync", "async"], ids=["sync", "async"])
def client_kind(request: pytest.FixtureRequest) -> str:
    """Runs a test once with ``DevupAI`` and once with ``AsyncDevupAI``."""
    kind: str = request.param
    return kind


@pytest.fixture
def anyio_backend() -> str:
    """Async-only tests run on asyncio."""
    return "asyncio"
