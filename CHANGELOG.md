# Changelog

All notable changes to the `devupai` Python package are documented in this file.

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
