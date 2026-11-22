"""``x-should-retry: false``: an answer that carries it is never sent again (1.0.1).

The check runs in ``_status_wait``, the one place both clients decide whether to send a request again after an error
status, before the status rules. So it covers every method, plain and ``with_response`` calls and streams (the
decision is made on the headers, before any chunk). Only ``false`` is honoured: ``x-should-retry: true`` never widens
the retries, so a billed ``POST`` is still sent again only after 429, 503 or a connection that was never opened.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from helpers.mock_transport import MODEL_A, MockAPI, error_response, json_response, run_request, sequence

from devupai import DevupAPIError, InternalServerError, RateLimitError
from devupai._client import _retry_refused

if TYPE_CHECKING:
    from conftest import FakeClock

    from devupai import AsyncDevupAI, DevupAI

FALSE = {"x-should-retry": "false"}
MESSAGES = [{"role": "user", "content": "Hi"}]


def _call(make: Callable[[], Any]) -> Any:
    """Runs ``make()``; the coroutine of an async client is awaited with ``anyio.run``."""
    result = make()
    if inspect.isawaitable(result):
        awaitable = result

        async def main() -> Any:
            return await awaitable

        return anyio.run(main)
    return result


def _chat(client: DevupAI | AsyncDevupAI) -> Any:
    return _call(lambda: client.chat.completions.create(model=MODEL_A, messages=MESSAGES))


def _attempts(
    api: MockAPI, client_kind: str, call: Callable[[DevupAI | AsyncDevupAI], Any], **opts: Any
) -> tuple[int, BaseException]:
    """The number of requests sent and the error raised by ``call``."""
    client = api.client(client_kind, **opts)
    with pytest.raises(DevupAPIError) as info:
        call(client)
    return len(api.calls), info.value


# --- the header rule -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "refused"),
    [
        ({"x-should-retry": "false"}, True),
        ({"X-Should-Retry": "FALSE "}, True),
        ({"x-should-retry": " False"}, True),
        ({"x-should-retry": "true"}, False),
        ({"x-should-retry": "no"}, False),
        ({"x-should-retry": ""}, False),
        ({}, False),
    ],
)
def test_only_false_is_read_case_and_spaces_aside(headers: dict[str, str], refused: bool) -> None:
    assert _retry_refused(headers) is refused


# --- billed POSTs, both clients ----------------------------------------------------------------------------------


def test_503_with_false_is_sent_once_and_raises_the_usual_error(client_kind: str, fake_clock: FakeClock) -> None:
    api = MockAPI(
        sequence(error_response(503, "busy", code="service_unavailable", headers={**FALSE, "Retry-After": "5"}))
    )
    count, error = _attempts(api, client_kind, _chat, max_retries=2)
    assert count == 1
    assert isinstance(error, InternalServerError)
    assert error.status == 503
    assert error.code == "service_unavailable"
    assert error.headers["x-should-retry"] == "false"  # the hint is visible on the error's headers
    assert fake_clock.sleeps == []  # no wait: nothing is sent again


def test_429_with_false_is_sent_once(client_kind: str, fake_clock: FakeClock) -> None:
    api = MockAPI(sequence(error_response(429, "busy", headers={**FALSE, "retry-after-ms": "10"})))
    count, error = _attempts(api, client_kind, _chat, max_retries=2)
    assert count == 1
    assert isinstance(error, RateLimitError)
    assert fake_clock.sleeps == []


def test_503_without_the_header_retries_as_before(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503)))
    count, _error = _attempts(api, client_kind, _chat, max_retries=2)
    assert count == 3


def test_503_with_true_retries_exactly_as_without_the_header(client_kind: str) -> None:
    without = MockAPI(sequence(error_response(503)))
    with_true = MockAPI(sequence(error_response(503, headers={"x-should-retry": "true"})))
    assert (
        _attempts(with_true, client_kind, _chat, max_retries=2)[0]
        == _attempts(without, client_kind, _chat, max_retries=2)[0]
        == 3
    )


def test_true_never_widens_a_billed_post_to_502(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(502, headers={"x-should-retry": "true"})))
    count, _error = _attempts(api, client_kind, _chat, max_retries=2)
    assert count == 1  # 502 is not retried on a billed POST, whatever the header says


def test_case_and_spaces_in_the_header_are_sent_once(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503, headers={"X-Should-Retry": "FALSE "})))
    count, _error = _attempts(api, client_kind, _chat, max_retries=2)
    assert count == 1


def test_a_larger_retry_budget_does_not_change_it(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503, headers=FALSE)))
    count, _error = _attempts(api, client_kind, _chat, max_retries=10)
    assert count == 1


def test_the_error_is_the_same_as_without_the_header(client_kind: str) -> None:
    with_false = _attempts(
        MockAPI(sequence(error_response(503, "busy", code="c", headers=FALSE))), client_kind, _chat, max_retries=0
    )[1]
    without = _attempts(MockAPI(sequence(error_response(503, "busy", code="c"))), client_kind, _chat, max_retries=0)[1]
    assert isinstance(with_false, DevupAPIError)
    assert isinstance(without, DevupAPIError)
    assert type(with_false) is type(without)
    assert (with_false.status, with_false.code, with_false.message) == (without.status, without.code, without.message)


# --- every path ----------------------------------------------------------------------------------------------------


def test_a_not_billed_get_with_502_and_false_is_sent_once(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(502, headers=FALSE)))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError):
        _call(lambda: client.models.list())
    assert len(api.calls) == 1


def test_a_not_billed_get_with_502_and_no_header_still_retries(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(502), json_response({"object": "list", "data": []})))
    client = api.client(client_kind, max_retries=2)
    _call(lambda: client.models.list())
    assert len(api.calls) == 2


def test_a_stream_request_with_503_and_false_is_sent_once(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503, headers=FALSE)))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError):
        _call(lambda: client.chat.completions.create(model=MODEL_A, messages=MESSAGES, stream=True))
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c: c.embeddings.create(model=MODEL_A, input="x"), id="embeddings.create"),
        pytest.param(lambda c: c.rerank.create(model=MODEL_A, query="q", documents=["a"]), id="rerank.create"),
        pytest.param(lambda c: c.inference.run(MODEL_A, {"input": "x"}), id="inference.run"),
        pytest.param(
            lambda c: c.inference.run(MODEL_A, {"input": "x"}, response_type="binary"), id="inference.run-binary"
        ),
        pytest.param(lambda c: c.responses.create(model=MODEL_A, input="Hi"), id="responses.create"),
        pytest.param(
            lambda c: c.chat.completions.with_response.create(model=MODEL_A, messages=MESSAGES), id="with_response"
        ),
    ],
)
def test_every_billed_resource_is_sent_once(call: Callable[[Any], Any], client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503, headers=FALSE)))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError):
        _call(lambda: call(client))
    assert len(api.calls) == 1


def test_the_low_level_request_path_honours_it(client_kind: str) -> None:
    api = MockAPI(sequence(error_response(503, headers=FALSE)))
    client = api.client(client_kind, max_retries=2)
    with pytest.raises(InternalServerError):
        run_request(client, "GET", "/models")
    assert len(api.calls) == 1
