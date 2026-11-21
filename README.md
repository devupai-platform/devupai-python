# DEVUP AI Python SDK

The official Python SDK for the DEVUP AI API, with a synchronous client,
`DevupAI`, and an asynchronous client, `AsyncDevupAI`.

## Requirements

Python 3.11 or later.

## Installation

```bash
pip install devupai
```

## Create a client

Synchronous client:

```python
from devupai import DevupAI

with DevupAI() as client:
    print(client.base_url)
```

Asynchronous client:

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        print(client.base_url)


asyncio.run(main())
```

You can also pass the key yourself:

```python
from devupai import DevupAI

client = DevupAI(api_key="your-api-key")
try:
    print(client.base_url)
finally:
    client.close()
```

Without a `with` block, call `client.close()` (or `await client.close()` for
`AsyncDevupAI`) when you are done. Closing a client more than once is allowed.

## API key

The client takes the API key from the first of these that is set:

1. The `api_key` argument.
2. The `DEVUP_API_KEY` environment variable.
3. The `DEVUPAI_API_KEY` environment variable. It is read only when
   `DEVUP_API_KEY` is not set at all.

When none of them is set, or the environment variable is empty, creating the
client raises `ValueError`. A key that contains a line break or a NUL
character also raises `ValueError`.

`api_key=""` is allowed: the client then sends an empty key, which is useful
only for requests that do not need one.

## Options

Every option is a keyword argument, and both clients take the same options.

```python
from devupai import DevupAI

client = DevupAI(
    api_key="your-api-key",
    base_url="https://api.devupai.com/v1",
    timeout=60.0,
    max_retries=2,
    default_headers={"X-Application-Name": "my-application"},
)
```

| Option            | Type                                    | Default                      | Description |
| ----------------- | --------------------------------------- | ---------------------------- | ----------- |
| `api_key`         | `str \| None`                           | See [API key](#api-key)      | Your DEVUP AI API key. |
| `base_url`        | `str \| None`                           | `https://api.devupai.com/v1` | DEVUP AI API base URL. A trailing `/` is removed. It must be an absolute `http` or `https` URL without a user name or password (`ValueError` otherwise). |
| `timeout`         | `float \| None`                         | `600.0` (10 minutes)         | Seconds. One limit for the whole call, retries included. The Node.js SDK takes this value in milliseconds; this SDK takes seconds. |
| `max_retries`     | `int \| None`                           | `2`                          | How many times a request is sent again. See [Retries](#retries). |
| `default_headers` | `Mapping[str, str] \| None`             | —                            | Headers included in every request. |
| `http_client`     | `httpx.Client` / `httpx.AsyncClient`    | A client the SDK creates     | Your own `httpx.Client` (for `DevupAI`) or `httpx.AsyncClient` (for `AsyncDevupAI`). Only its transport, connection pool and event hooks apply: its own `auth`, base URL, headers, query parameters, cookies and timeouts are not used (each request takes the `timeout` of this client or of the call). The SDK never closes a client you pass. |
| `on_debug`        | `Callable[[DebugEvent], None] \| None`  | —                            | Receives debug events, such as `retry` before each retry. An exception raised in it is ignored. |

`timeout` must be a positive number of seconds, and `max_retries` a
non-negative integer. A wrong type raises `TypeError` and a wrong value raises
`ValueError`, before any request is sent.

The SDK sends `User-Agent: devupai-python/<version>`. It manages the
`Authorization` and `Content-Type` headers itself; custom headers cannot
override them. Redirects are not followed.

## Retries

The client sends a request again, up to `max_retries` times (default 2), after
a 408, 429, 502, 503 or 504 answer, or a connection error.

| Requests | Sent again after |
| -------- | ---------------- |
| Billed requests: chat completions, Responses, embeddings, rerank, image generation and edits, audio, video, and native inference | 429 and 503, and a connection that could not be opened |
| Other requests | 408, 429, 502, 503, and 504, and connection errors |

A billed request is sent again only after a 429 or 503 answer, or when the
connection could not be opened. After any other answer, or a connection lost
after the request may have been sent, the error is returned to you without a
retry, because the request may already have been processed and charged.

A 500 answer is not retried.

When the answer has a `retry-after-ms` or `Retry-After` header, the client
waits that long before the next attempt. Otherwise it waits a random time that
grows with each attempt, at most 60 seconds between attempts. A wait that does
not fit in the time left of `timeout` ends the call with the error.

`max_retries=0` turns retries off.

## Errors

Every error from the API is a `DevupAPIError`. The SDK raises the subclass
that matches the HTTP status, so you can catch a specific case or the base
class. All the classes below are subclasses of `DevupAPIError`.

| Class                       | Status     | Notes |
| --------------------------- | ---------- | ----- |
| `BadRequestError`           | 400        | The request is not valid as sent; `code` names the reason. |
| `AuthenticationError`       | 401        | Missing or invalid API key. |
| `PaymentRequiredError`      | 402        | The balance does not cover the request (`insufficient_balance`), or a scoped credential reached its budget (`budget_exhausted`). |
| `PermissionDeniedError`     | 403        | The key cannot use this resource. |
| `NotFoundError`             | 404        | For example, an unknown model. |
| `ConflictError`             | 409        | The outcome of this request is not confirmed yet (`request_outcome_pending`). Do not send it again: contact support with `request_id`. |
| `PayloadTooLargeError`      | 413        | The request body is larger than the limit (`payload_too_large`). |
| `UnsupportedMediaTypeError` | 415        | The model does not accept the content type of the request. Send the `Content-Type` and body format that the API reference lists for the endpoint. |
| `UnprocessableEntityError`  | 422        | The request was not accepted in its current form. |
| `RateLimitError`            | 429        | Read `retry_after_ms` for the suggested wait. |
| `InternalServerError`       | 500 and up | After a 500 or 502 the request may already have been processed and charged: check its outcome before sending it again. |
| `APIConnectionError`        | `0`        | The API could not be reached. |
| `APITimeoutError`           | `0`        | The call exceeded `timeout`. A subclass of `APIConnectionError`. |
| `DevupAPIError`             | other      | Any other status. |

| Field               | Type             | Description |
| ------------------- | ---------------- | ----------- |
| `status`            | `int`            | HTTP status code, or `0` for connection errors and timeouts. |
| `message`           | `str`            | Human-readable error message (also `str(err)`). |
| `code`              | `str \| None`    | Machine-readable error code, when available. |
| `type`              | `str \| None`    | Error category, when available. |
| `param`             | `str \| None`    | The request parameter the error refers to, when available. |
| `request_id`        | `str \| None`    | The `x-request-id` header. Include it when you contact support. |
| `client_request_id` | `str \| None`    | The `x-client-request-id` header. |
| `retry_after`       | `str \| None`    | Raw `Retry-After` header value. |
| `retry_after_ms`    | `float \| None`  | Suggested wait in milliseconds, when the answer gives one. |
| `warnings`          | `list[str]`      | Notice codes from the `X-DEVUP-Warnings` header (empty when none). |
| `headers`           | `dict[str, str]` | Response headers, with lower-case names. |

The underlying exception, when there is one, is chained as `__cause__`.

A 409 means the outcome of the request is not confirmed yet. Do not send it
again: contact support with its `request_id`.

```python
from devupai import DevupAPIError, RateLimitError

try:
    ...  # a call to the DEVUP AI API
except RateLimitError as err:
    print(err.retry_after_ms)
except DevupAPIError as err:
    print(err.status, err.code, err.request_id)
```

The error codes you can act on are listed at
[docs.devupai.com/docs/errors](https://docs.devupai.com/docs/errors).

## License

The DEVUP AI Python SDK is released under the
[MIT License](https://github.com/devupai-platform/devupai-python/blob/main/LICENSE).
