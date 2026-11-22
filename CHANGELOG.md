# Changelog

All notable changes to the `devupai` Python package are documented in this file.

## [1.0.1]

No breaking changes. A request whose answer carries `x-should-retry: false`
is not sent again. Upgrade with `pip install --upgrade devupai`.

### Changed

- `DevupAI` and `AsyncDevupAI` do not send a request again when its answer
  carries `x-should-retry: false`, whatever the status, the method or
  `max_retries`. This applies to every method, to `with_response` calls and
  to streams. The error you receive is the same as before, and its `headers`
  show the value. Any other value of the header, or no header, keeps the
  retries described in the README; `x-should-retry: true` adds no retries.

## [1.0.0]

The first stable release. Install it with `pip install devupai`; `--pre` is
no longer needed. No breaking changes from 1.0.0a1, except two checks that
now run before a request is sent: `client.inference.run()` raises
`ValueError` when `response_type` is anything other than `"json"` or
`"binary"` (1.0.0a1 sent such a call as a JSON request), and
`client.images.edit()` with a file input raises `TypeError` for an input
image, mask or markup it cannot send (1.0.0a1 sent the request without it).

### Changed

- `client.inference.run()`, on `DevupAI` and on `AsyncDevupAI`, raises
  `ValueError` for a `response_type` other than `"json"` or `"binary"`, and
  sends no request. A call with `"json"`, `"binary"` or no `response_type`
  works as before.
- `client.images.edit()` sent as `multipart/form-data` (at least one input is
  a file) raises `TypeError`, and sends no request, when an item of `image`,
  `images`, `mask` or `markup` is not a file, a URL or data URI string, or
  `{"image_url": "..."}`. Valid inputs are sent as before.
- The message of a webhook error with reason `invalid_json` now reads "The
  verified webhook payload is not valid JSON". The error class and its reason
  are unchanged.
- The package metadata gives the development status as Production/Stable.

### Docs

- The README has a short example for every resource, and covers streaming
  and binary results in a `with` block (`async with` on `AsyncDevupAI`), file
  uploads, `with_response`, errors and `request_id`, retries of billed
  requests, webhooks, which settings of an `http_client` you pass are kept,
  and when the SDK raises `TypeError` or `ValueError`.
- The repository has an `examples/` folder with runnable scripts. A script
  that sends a model id reads it from the `DEVUP_MODEL` environment variable;
  `examples/models_list.py` lists the ids you can use.

## [1.0.0a1]

The first pre-release of the official Python SDK for the DEVUP AI API, with a
synchronous client (`DevupAI`) and an asynchronous client (`AsyncDevupAI`).
It needs Python 3.11 or later. Install it with `pip install --pre devupai`.

### Added

- `client.chat.completions.create()` and `client.responses.create()`, with
  JSON responses and `stream=True`. A stream is an iterator (`async for` on
  `AsyncDevupAI`) that also works in a `with` block, and its `request_id` and
  `warnings` are readable before the first chunk.
- `client.models.list()` and `client.models.retrieve()`.
- `client.images.generate()`, `client.images.edit()` and
  `client.images.download()`.
- `client.embeddings.create()` and `client.rerank.create()`.
- `client.audio.speech.create()` and `client.audio.transcriptions.create()`.
- `client.video.generations.create()` and `client.video.edits.create()`.
- `client.inference.run()`, with JSON or binary results.
- `client.balance.retrieve()`, `client.me.retrieve()`, `client.me.avatar()`
  and `client.health.check()`.
- File uploads from `bytes`, a binary file object, a `pathlib.Path` or a
  `(filename, content, content_type)` tuple. The SDK closes the files it opens
  from a path and never closes a file object you pass.
- `BinaryResponse` and `AsyncBinaryResponse` for binary results, with
  `read()`, `iter_bytes()` and `write_to_file()`.
- `with_response` on each resource that calls the API, to read the status
  code, headers and `request_id` together with the parsed result.
- Webhook signature verification: `client.webhooks.verify_signature()`,
  `client.webhooks.construct_event()`, `verify_webhook_signature()` and
  `construct_webhook_event()`.
- Error classes for each HTTP status the API documents, all subclasses of
  `DevupAPIError`, plus `APIConnectionError` and `APITimeoutError`.
  `error.code` names the reason, and `error.request_id` identifies the request.
- Retries with backoff for requests that are safe to send again. A billed
  request is sent again only after a 429 or 503 answer, or when the
  connection could not be opened.
- Type hints for every request and response (`py.typed`).
