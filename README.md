<p align="center">
  <a href="https://devupai.com">
    <img
      src="https://devupai.com/image/devu.png"
      alt="DEVUP AI"
      width="200"
    />
  </a>
</p>

<h1 align="center">DEVUP AI — Official Python SDK</h1>

<p align="center">
  <strong>Build with leading AI models. Pay in Algerian Dinar.</strong>
</p>

<p align="center">
  A typed Python SDK for chat, Responses, images, embeddings, reranking, audio, video, and native inference,
  with a synchronous and an asynchronous client.
</p>

<p align="center">
  <a href="https://pypi.org/project/devupai/">
    <img src="https://img.shields.io/pypi/v/devupai?style=flat-square&label=pypi" alt="PyPI version" />
  </a>
  <a href="https://github.com/devupai-platform/devupai-python/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/devupai?style=flat-square" alt="MIT License" />
  </a>
  <img src="https://img.shields.io/pypi/pyversions/devupai?style=flat-square" alt="Python versions" />
  <img src="https://img.shields.io/badge/Typed-py.typed-3776AB?style=flat-square" alt="Typed" />
  <a href="https://status.devupai.com/">
    <img src="https://img.shields.io/badge/API-Status-10B981?style=flat-square" alt="DEVUP AI Status" />
  </a>
</p>

<p align="center">
  <a href="https://docs.devupai.com/"><strong>Documentation</strong></a>
  ·
  <a href="https://devupai.com/dashboard"><strong>Dashboard</strong></a>
  ·
  <a href="https://devupai.com/dashboard/api-keys"><strong>API Keys</strong></a>
  ·
  <a href="https://devupai.com/models"><strong>Models</strong></a>
  ·
  <a href="https://status.devupai.com/"><strong>Status</strong></a>
  ·
  <a href="https://pypi.org/project/devupai/"><strong>PyPI</strong></a>
</p>

---

## Overview

`devupai` is the official Python SDK for [DEVUP AI](https://devupai.com), an AI
platform built for developers and businesses in Algeria.

It gives you one consistent interface for:

- Chat completions, with streaming and reasoning output
- The Responses API
- Image generation and editing
- Text embeddings and reranking
- Text-to-speech and transcription
- Video generation and editing
- Native inference for specialized models
- Model discovery, account profile and balance, and health checks
- Webhook signature verification

> **Two clients, one interface:** `DevupAI` for synchronous code and `AsyncDevupAI` for `asyncio`, with the
> same methods and options.  
> **Runtime dependencies:** `httpx` and `pydantic`. The package ships type information (`py.typed`).

## Contents

- [Why DEVUP AI](#why-devup-ai)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Capability Matrix](#capability-matrix)
- [Core SDK](#core-sdk)
  - [Chat Completions](#chat-completions)
  - [Streaming](#streaming)
  - [Reasoning Output](#reasoning-output)
  - [Responses API](#responses-api)
  - [Images](#images)
  - [Model Options](#model-options)
  - [Downloading Generated Files](#downloading-generated-files)
  - [Embeddings](#embeddings)
  - [Reranking](#reranking)
  - [Audio](#audio)
  - [Video](#video)
  - [Native Inference](#native-inference)
  - [Models, Balance, and Health](#models-balance-and-health)
  - [Webhooks](#webhooks)
- [Configuration](#configuration)
- [Request Options](#request-options)
- [Request IDs, Headers, and Notices](#request-ids-headers-and-notices)
- [Retries](#retries)
- [Timeouts and Cancellation](#timeouts-and-cancellation)
- [Error Handling](#error-handling)
  - [Request Size Limits](#request-size-limits)
- [Type Hints](#type-hints)
- [Runtime and Security](#runtime-and-security)
- [Models](#models)
- [Get an API Key](#get-an-api-key)
- [Resources](#resources)
- [License](#license)

---

## Why DEVUP AI

| Capability                  | What it gives you                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| **Billing in DZD**          | Track AI usage and costs directly in Algerian Dinar.                                       |
| **Local payment access**    | Top up with supported local payment methods.                                               |
| **Familiar API shapes**     | Chat completions, Responses, embeddings, and more, with the request shapes you know.       |
| **Multimodal inference**    | Access text, images, embeddings, audio, video, and specialized models from one SDK.        |
| **Typed contracts**         | `TypedDict` request parameters and `pydantic` response models, checked against the DEVUP AI OpenAPI specification. |
| **Sync and async**          | `DevupAI` and `AsyncDevupAI` share every method and option.                                |
| **Modern Python**           | Python 3.11 or later, built on `httpx`.                                                    |
| **Operational visibility**  | Request IDs, response headers, models, account balance, health, and billing metadata.      |

---

## Requirements

- Python 3.11 or later.

---

## Installation

```bash
pip install devupai
```

---

## Quick Start

```python
import os

from devupai import DevupAI

client = DevupAI(api_key=os.environ["DEVUP_API_KEY"])

completion = client.chat.completions.create(
    model="<chat-model-id>",
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "Explain artificial intelligence in one paragraph."},
    ],
)

message = completion.choices[0].message
print(message.content if message else "")
if completion.devup is not None:
    print("Cost:", completion.devup.cost_dzd, "DZD")
```

`DevupAI()` without `api_key` reads the key from the environment; see
[API key](#api-key). The asynchronous client takes the same options, and every
API method is awaited:

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        completion = await client.chat.completions.create(
            model="<chat-model-id>",
            messages=[{"role": "user", "content": "Hello from Algeria."}],
        )
        message = completion.choices[0].message
        print(message.content if message else "")


asyncio.run(main())
```

The model ids in this README are placeholders such as `"<chat-model-id>"`.
List the models available to your key with `client.models.list()`:

```python
for model in client.models.list().data:
    print(model.id, model.type)
```

Use a `with` block (`async with` for `AsyncDevupAI`), or call `client.close()`
(`await client.close()`), to release the client's connections when you are
done. Closing a client more than once is allowed; a closed client raises
`RuntimeError` on its next call.

---

## Capability Matrix

| Capability           | SDK method                               | API key sent             |
| -------------------- | ---------------------------------------- | ------------------------ |
| Chat completions     | `client.chat.completions.create()`       | Yes                      |
| Responses            | `client.responses.create()`              | Yes                      |
| Image generation     | `client.images.generate()`               | Yes                      |
| Image editing        | `client.images.edit()`                   | Yes                      |
| Generated files      | `client.images.download(url)`            | No                       |
| Embeddings           | `client.embeddings.create()`             | Yes                      |
| Reranking            | `client.rerank.create()`                 | Yes                      |
| Text-to-speech       | `client.audio.speech.create()`           | Yes                      |
| Transcription        | `client.audio.transcriptions.create()`   | Yes                      |
| Video generation     | `client.video.generations.create()`      | Yes                      |
| Video editing        | `client.video.edits.create()`            | Yes                      |
| Native inference     | `client.inference.run()`                 | Yes                      |
| List models          | `client.models.list()`                   | Yes                      |
| Retrieve a model     | `client.models.retrieve(id)`             | Yes                      |
| Account balance      | `client.balance.retrieve()`              | Yes                      |
| Account profile      | `client.me.retrieve()`                   | Yes                      |
| Account picture      | `client.me.avatar()`                     | Yes                      |
| Service health       | `client.health.check()`                  | No                       |
| Webhook verification | `client.webhooks.verify_signature()`     | None (local computation) |
| Webhook construction | `client.webhooks.construct_event()`      | None (local computation) |

Every API method returns the data of the answer. Each resource also has a
`with_response` twin with the same methods, which returns the data together
with the HTTP response, the request ID, and notices; see
[Request IDs, Headers, and Notices](#request-ids-headers-and-notices). The
webhook helpers are synchronous on both clients.

---

# Core SDK

The examples below use a `client` created as in [Quick Start](#quick-start).
Body fields are keyword arguments, named as in the API reference.

## Chat Completions

```python
completion = client.chat.completions.create(
    model="<chat-model-id>",
    messages=[{"role": "user", "content": "What is retrieval-augmented generation?"}],
    temperature=0.4,
    max_tokens=800,
)

message = completion.choices[0].message
print(message.content if message else "")
if completion.usage is not None:
    print("Tokens:", completion.usage.total_tokens)
if completion.devup is not None:
    print("Cost:", completion.devup.cost_dzd, "DZD")
    print("Balance:", completion.devup.balance_dzd, "DZD")
```

`completion.devup` is the `_devup` object of the answer: the cost of the
request and the balance after it.

## Streaming

Pass `stream=True` to receive the reply as it is generated. The call returns a
`Stream[ChatCompletionChunk]` on `DevupAI` and an `AsyncStream[ChatCompletionChunk]`
on `AsyncDevupAI`. Open it in a `with` block (`async with` on `AsyncDevupAI`),
so the response is closed when you leave the block, also when you stop early.

- Every item is a `ChatCompletionChunk`. The last chunk has an empty `choices`
  list and carries `usage`.
- `stream.request_id`, `stream.client_request_id`, `stream.warnings`, and
  `stream.response` can be read before the first chunk.
- A stream can be iterated once; a second iteration raises `RuntimeError`.
- An event that is not valid JSON is skipped and reported to `on_debug` as
  `malformed_frame`.

```python
with client.chat.completions.create(
    model="<chat-model-id>",
    messages=[{"role": "user", "content": "Write a short poem about Algeria."}],
    stream=True,
) as stream:
    print("Request ID:", stream.request_id)
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        if chunk.usage is not None:
            print("\nTotal tokens:", chunk.usage.total_tokens)
```

With `AsyncDevupAI`, use `async with await` and `async for`:

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        async with await client.chat.completions.create(
            model="<chat-model-id>",
            messages=[{"role": "user", "content": "Write a short poem about Algeria."}],
            stream=True,
        ) as stream:
            async for chunk in stream:
                if chunk.choices:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)


asyncio.run(main())
```

On `DevupAI`, leaving the loop with `break` or an exception also closes the
response. An `AsyncStream` that is dropped without being closed emits a
`ResourceWarning`: `async with` (or `await stream.close()`) closes it.

If the API reports an error after the stream has started, the loop raises a
[`DevupAPIError`](#error-handling) carrying the error's `code` and the stream's
`request_id`. A connection lost while reading raises `APIConnectionError`.
Nothing is sent again once the stream has started.

Pass `stream` as an argument. `extra_body` cannot turn streaming on or off:
when `extra_body` sets `stream` to `True` without the `stream=True` argument,
or `stream=True` comes with an `extra_body` `stream` that is not `True`, the
call raises `ValueError` before anything is sent. Any other `stream` value in
`extra_body` is sent as given. `stream_idle_timeout` ends a stream that
receives no data for that many seconds; see
[Timeouts and Cancellation](#timeouts-and-cancellation).

---

## Reasoning Output

Reasoning models can return their reasoning text in `reasoning_content` on the
message (and on `delta` when streaming). When the model returns it,
`usage.completion_tokens_details.reasoning_tokens` counts the reasoning tokens.

Set `max_tokens` high enough for both the reasoning and the answer, and check
`finish_reason`. The request also takes `reasoning_effort` and
`reasoning={"enabled": ...}`; `client.models.retrieve(id).reasoning` gives a
model's reasoning `mode`, the request `parameter` that controls it, and its
`values`.

```python
completion = client.chat.completions.create(
    model="<reasoning-model-id>",
    messages=[{"role": "user", "content": "Is 3571 a prime number? Explain briefly."}],
    max_tokens=2048,
)

choice = completion.choices[0]
if choice.finish_reason == "length":
    print("The reply reached max_tokens before it was complete.")

if choice.message is not None:
    if choice.message.reasoning_content:
        print("Reasoning:\n", choice.message.reasoning_content)
    print("Answer:\n", choice.message.content)

usage = completion.usage
details = usage.completion_tokens_details if usage is not None else None
if details is not None and details.reasoning_tokens is not None:
    print("Reasoning tokens:", details.reasoning_tokens)
```

---

## Responses API

`client.responses.create()` calls `POST /v1/responses`, the Responses API.

```python
response = client.responses.create(
    model="<chat-model-id>",
    instructions="Answer in two sentences.",
    input="Describe the Sahara.",
    max_output_tokens=300,
)

print("Status:", response.status)
print("Output:", response.output)
if response.usage is not None:
    print("Tokens:", response.usage.total_tokens)
```

With `stream=True`, the call returns a `Stream[ResponseStreamEvent]`
(`AsyncStream[ResponseStreamEvent]` on `AsyncDevupAI`). Each event has a
`type`, such as `response.output_text.delta` or `response.completed`. Skip any
event type your code does not handle. An `error` event makes the loop raise a
[`DevupAPIError`](#error-handling). The rules of [Streaming](#streaming) apply.

```python
with client.responses.create(
    model="<chat-model-id>",
    input="Write a haiku about the sea.",
    stream=True,
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta or "", end="", flush=True)
        elif event.type == "response.completed" and event.response and event.response.usage:
            print("\nTokens:", event.response.usage.total_tokens)
```

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        async with await client.responses.create(
            model="<chat-model-id>",
            input="Write a haiku about the sea.",
            stream=True,
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta or "", end="", flush=True)


asyncio.run(main())
```

---

## Images

<details open>
<summary><strong>Generate an image</strong></summary>

```python
result = client.images.generate(
    model="<image-model-id>",
    prompt="A cinematic view of Algiers at sunset",
    size="1024x1024",
)

print(result.data[0].url)
print("Cost:", result.devup.cost_dzd, "DZD")
```

`size` takes the sizes the model lists. Every image model lists its sizes and
other options in `image_params`; see [Model Options](#model-options).

</details>

<details>
<summary><strong>Edit an image</strong></summary>

`client.images.edit()` posts to `POST /v1/images/edits`. Each input image is
an `https://` URL or a `data:` URI (as a `str` or as `{"image_url": ...}`), or
a [file](#file-inputs). Pass one input as `image`, or several as a list in
`image` or `images`; `mask` and `markup` take one input each. Every
`images.generate()` field (`prompt`, `size`, `quality`, `output_format`, ...)
is accepted.

- When every input is a URL or a `data:` URI, the request is sent as JSON,
  exactly as given (`ImageEditJsonParams`). With no file input, any other
  JSON value is sent as given too.
- When at least one input is a file, the request is sent as
  `multipart/form-data` (`ImageEditParams`): one `image` part per input image,
  in order, one `mask` and one `markup` part when given, and every other field
  as a text part (an object or a list as its JSON text).

With image URLs:

```python
result = client.images.edit(
    model="<image-edit-model-id>",
    image="https://example.com/street.png",
    prompt="Add a red sports car in the foreground",
)

print(result.data[0].url)
```

With files:

```python
from pathlib import Path

result = client.images.edit(
    model="<image-edit-model-id>",
    image=Path("input.png"),
    mask=Path("mask.png"),
    prompt="Replace the masked area with a garden",
)

print(result.data[0].url)
```

Use the mask format the model lists in `image_params.inputs.mask_format`:
`alpha` or `black_and_white`. See [Model Options](#model-options).

</details>

### File inputs

Every method that uploads a file (`images.edit()`, `audio.transcriptions.create()`,
`video.generations.create()`, `video.edits.create()`, and `inference.run()`)
takes the same file inputs:

| Input                                                            | Filename sent                     | Notes                                                        |
| ---------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `bytes`, `bytearray`, or `memoryview`                            | `upload`                          | Sent as `application/octet-stream`.                          |
| A path (`pathlib.Path` or any `os.PathLike`)                     | The file's base name              | The SDK opens, reads, and closes the file.                   |
| A file opened in binary mode (`open(path, "rb")`)                | The base name of its `name`, else `upload` | Read with `read()` and left open. A text-mode file raises `TypeError`. |
| `(filename, content)` or `(filename, content, content_type)`     | `filename`                        | `content` is bytes or a binary file object.                  |

A plain `str` is never read as a file. In the fields that also take a URL
(image edits, video generation, and `inference.run()`), it is sent as text (a
URL or a `data:` URI); the transcription `file` and the video edit `video` and
`mask` take files only, so a `str` there raises `TypeError`. The file is read
once, before the request is sent, so every retry sends the same bytes.

## Model Options

Image models take different options. `client.models.list()` and
`client.models.retrieve(id)` return them in `image_params` (type
`ImageParams`) on each model that lists image options; the field is `None` on
other models. Field names and values are the ones a request sends.

- `endpoints`: `generations`, `edits`, or both.
- `options` (`list[ImageParamsOption]`): each option's request `field` and
  what it takes: `values`, `range`, `default`, `required`, `max_length`, and,
  for `size`, `free` (the free sizes, for example `min_edge` and `max_edge`).
  An option with a `kind` (`boolean`, `choice`, `int`, `decimal`, `text`, or
  `structured`) is an option of the model's own, sent under its own name.
- `inputs` (`ImageParamsInputs`): the input images an edit takes (`image.min`,
  `image.max`, and, when listed, `roles`, `max_pixels`, and `max_edge`),
  whether the model takes `mask` or `markup`, `mask_required`,
  `mask_max_images`, and `mask_format` (`alpha` or `black_and_white`).
- `prompt`: `none` (the model takes no prompt), `optional`, or `None` (a
  prompt is required). `prompt_max_length` is the longest prompt.

An option of the model's own is not a field of `ImageGenerateParams`: send it
in `extra_body` under its `field` name.

```python
model = client.models.retrieve("<image-model-id>")
params = model.image_params
if params is not None:
    print("Endpoints:", ", ".join(params.endpoints))
    for option in params.options:
        print(option.field, option.kind or "image option", option.values, option.range, option.default)
    print("Prompt:", params.prompt or "required", params.prompt_max_length)
    if params.inputs is not None and params.inputs.image is not None:
        print("Input images:", params.inputs.image.min, "to", params.inputs.image.max)

    own_options = {
        option.field: option.default
        for option in params.options
        if option.kind is not None and option.default is not None
    }
    result = client.images.generate(
        model="<image-model-id>",
        prompt="A watercolor map of the Algerian coast",
        negative_prompt="text, watermark",
        extra_body=own_options,
    )
    print(result.data[0].url)
```

A value the model does not accept returns HTTP 400 `invalid_value`, and the
message names what the model accepts. An option the model marks as
`required`, left out, returns HTTP 400 `missing_required_parameter`. Both are
raised as `BadRequestError`.

---

## Downloading Generated Files

`client.images.download(url)` fetches a generated image or video file from the
URL returned by the API (for example `result.data[0].url`). Pass the URL
exactly as you received it.

- It accepts only URLs on the origin of the client's `base_url` (scheme, host,
  and port). Any other URL, and any URL that contains a user name or password,
  raises a `DevupAPIError` with code `invalid_url` and the message
  "images.download only fetches URLs returned by the API." before a request is
  sent.
- It does not send your API key.
- Send a `Range` header in `extra_headers` to fetch part of the file.
- A link returned by the API expires after a short period. Downloading an
  expired link raises a `PermissionDeniedError` (HTTP 403): generate the image
  or video again to get a new link.

```python
generated = client.images.generate(
    model="<image-model-id>",
    prompt="A lighthouse on the Mediterranean coast",
)

url = generated.data[0].url
if url:
    with client.images.download(url) as file:
        print("Content-Type:", file.headers.get("content-type"))
        file.write_to_file("lighthouse.png")
```

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        generated = await client.images.generate(
            model="<image-model-id>",
            prompt="A lighthouse on the Mediterranean coast",
        )
        url = generated.data[0].url
        if url:
            async with await client.images.download(url) as file:
                data = await file.read()
                print("Image bytes:", len(data))


asyncio.run(main())
```

### Binary results

`images.download()`, `me.avatar()`, and `inference.run(..., response_type="binary")`
return a file answer with its body not read yet: a `BinaryResponse` on
`DevupAI`, an `AsyncBinaryResponse` on `AsyncDevupAI`.

| Member                    | `BinaryResponse`            | `AsyncBinaryResponse`              |
| ------------------------- | --------------------------- | ---------------------------------- |
| Whole body                | `read()`                    | `await read()`                     |
| Body in pieces            | `iter_bytes(chunk_size)`    | `aiter_bytes(chunk_size)`          |
| Write to a file           | `write_to_file(path)`       | `await write_to_file(path)`        |
| Close                     | `close()` or `with`         | `await close()` or `async with`    |
| Status, headers, request  | `status_code`, `headers`, `request_id`, `response` | the same               |

Open a binary result in a `with` block (`async with await ...` on
`AsyncDevupAI`), so the connection is given back also when you do not read the
body to the end. A `BinaryResponse` dropped unread is closed when it is
collected; an `AsyncBinaryResponse` dropped while open emits a
`ResourceWarning`. A read that does not complete raises `APIConnectionError`
(`APITimeoutError` when it times out).

---

## Embeddings

`embedding` is a list of numbers, or a base64 string when the request sets
`encoding_format="base64"`.

```python
result = client.embeddings.create(
    model="<embedding-model-id>",
    input=["Hello world", "مرحبا بالعالم", "Bonjour le monde"],
)

vector = result.data[0].embedding
if isinstance(vector, list):
    print("Dimensions:", len(vector))
```

See [Request Size Limits](#request-size-limits) for the largest `input`.

---

## Reranking

`top_n` and `return_documents` are optional.

```python
result = client.rerank.create(
    model="<rerank-model-id>",
    query="What is DEVUP AI?",
    documents=[
        "DEVUP AI is an AI platform with billing in Algerian Dinar.",
        "Algiers is the capital of Algeria.",
    ],
    top_n=1,
)

for item in result.results:
    print(item.index, item.relevance_score)
```

---

## Audio

<details open>
<summary><strong>Text-to-speech</strong></summary>

```python
speech = client.audio.speech.create(
    model="<speech-model-id>",
    input="Welcome to DEVUP AI.",
)

print("Audio URL:", speech.url)
print("Cost:", speech.devup.cost_dzd, "DZD")
```

The request also takes `voice`, `response_format` (`mp3`, `opus`, `aac`,
`flac`, `wav`, or `pcm`), and `referenceAudioUrl`.

</details>

<details>
<summary><strong>Speech-to-text</strong></summary>

`file` is a [file input](#file-inputs). The file and the fields are sent as
`multipart/form-data`; a field left as `None` is not sent. The result follows
`response_format`: `json` (the default) returns a `TranscriptionResponse`,
`verbose_json` a `TranscriptionVerboseResponse`, and `text`, `srt`, and `vtt` a
`str`. The request also takes `language`, `prompt`, `temperature`, and, with
`response_format="verbose_json"`, `timestamp_granularities` (a list of
`"word"` and `"segment"`).

```python
from pathlib import Path

transcript = client.audio.transcriptions.create(
    file=Path("speech.wav"),
    model="<transcription-model-id>",
    language="ar",
    response_format="json",
)

print(transcript.text)
```

With `verbose_json`, request word or segment timestamps:

```python
from pathlib import Path

transcript = client.audio.transcriptions.create(
    file=Path("speech.wav"),
    model="<transcription-model-id>",
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
    prompt="DEVUP AI, Algiers",
    temperature=0,
)

print(transcript.text, transcript.duration)
for word in transcript.words or []:
    print(word.word, word.start, word.end)
```

</details>

---

## Video

<details open>
<summary><strong>Generate a video</strong></summary>

```python
video = client.video.generations.create(
    model="<video-model-id>",
    prompt="A camel walking through the Sahara at golden hour",
)

print(video.data[0].url)
print("Cost:", video.devup.cost_dzd, "DZD")
```

</details>

<details>
<summary><strong>Generate a video from frames or reference media</strong></summary>

When no field is a file, the request is sent as JSON (`VideoGenerationCreateParams`):
`image`, `last_frame_image`, and the `reference_*` fields take `https://` URLs
or `data:` URIs, as strings or as objects that carry one. When any field is a
[file](#file-inputs), or a list with a file in it, the request is sent as
`multipart/form-data` (`VideoGenerationCreateFormParams`): every field is
sent, a list as one part per item, an object as its JSON text, and any other
value as text (`duration=8` is `"8"`, `generate_audio=True` is `"true"`).
Replace `<image-to-video-model-id>` with a model that takes a first frame; see
the [model catalog](https://devupai.com/models).

```python
from pathlib import Path

video = client.video.generations.create(
    model="<image-to-video-model-id>",
    prompt="The camera slowly pulls back from the harbour",
    image=Path("first-frame.png"),
    reference_images=[Path("person.png"), "https://example.com/background.png"],
    duration=8,
)

print(video.data[0].url)
```

</details>

<details>
<summary><strong>Edit a video</strong></summary>

`client.video.edits.create()` uploads the video (and an optional mask) as
`multipart/form-data`. It takes exactly these fields: `video`, `model`,
`prompt`, `mask`, `mask_url`, `key_points`, `frame_index`, `auto_trim`,
`preserve_audio`, `background_color`, `desired_increase`,
`output_container_and_codec`, and `response_format`; any other keyword raises
`TypeError`. `key_points` is a JSON string, or a list of points sent as its
JSON text. Replace `<video-edit-model-id>` with a model that supports video
edits; see the [model catalog](https://devupai.com/models).

```python
from pathlib import Path

from devupai.types import VideoKeyPoint

key_points: list[VideoKeyPoint] = [{"x": 320, "y": 240, "type": "positive"}]

result = client.video.edits.create(
    model="<video-edit-model-id>",
    video=Path("input.mp4"),
    prompt="Enhance the subject while keeping the background",
    key_points=key_points,
)

print(result.data[0].url)
```

</details>

---

## Native Inference

Native inference sends a payload in the model's own request format to
`POST /v1/inference/{model}`. The payload is a mapping of field names to
values; anything else raises `TypeError`. It is sent as JSON, as it is. When a
field is a [file](#file-inputs), or a list contains one, the payload is sent as
`multipart/form-data` instead: a file is a file part, a `str` a text part, a
list one part per item, a mapping its JSON text, and a number or a boolean
text.

The payload and the response shape depend on the model; the result is the
decoded JSON. For a model that answers with a file, pass
`response_type="binary"`: the call then returns a
[binary result](#binary-results) with its body not read yet. The default is
`"json"`. Any other `response_type` value raises `ValueError` before anything
is sent.

`model_path` is the model id: leading slashes are removed, and an id that is
already percent-encoded is decoded first, so it is sent as one path segment
encoded exactly once.

```python
from pathlib import Path

result = client.inference.run("<model-id>", {"input": "Hello from Algeria!"})
print(result)

# A file in the payload: sent as multipart/form-data, JSON answer.
labels = client.inference.run("<model-id>", {"image": Path("photo.jpg")})
print(labels)

# A model that answers with a file.
with client.inference.run(
    "<model-id>", {"input": "A lighthouse at night"}, response_type="binary"
) as file:
    print("Content-Type:", file.headers.get("content-type"))
    file.write_to_file("lighthouse.png")
```

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        async with await client.inference.run(
            "<model-id>", {"input": "A lighthouse at night"}, response_type="binary"
        ) as file:
            await file.write_to_file("lighthouse.png")


asyncio.run(main())
```

Add a `webhook` URL to the payload to receive the result asynchronously. It
must be a public `https://` URL without a user name or password; any other
value returns HTTP 400 `invalid_webhook_url` before anything is charged. See
[Webhooks](#webhooks).

---

## Models, Balance, and Health

`models.list()` and `models.retrieve()` send your API key. `health.check()`
does not.

```python
health = client.health.check()
print("API status:", health.status)

models = client.models.list()
print("Available models:", len(models.data))

model = client.models.retrieve("<model-id>")
print(model.id, model.type, model.supports.structured_output)

balance = client.balance.retrieve()
print("Balance:", balance.balance_dzd, "DZD")
```

`models.retrieve(id)` accepts ids that contain `/`; the SDK encodes the id as
one path segment.

Both model calls take `if_none_match="<etag>"`, sent as `If-None-Match`. When
the API answers 304 Not Modified, the call returns `None`, and
`with_response` gives the 304 response with its `ETag` header.

```python
first = client.models.with_response.list()
print(len(first.data.data), "models")

etag = first.response.headers.get("etag")
if etag:
    refreshed = client.models.list(if_none_match=etag)
    print("Not modified" if refreshed is None else f"{len(refreshed.data)} models")
```

Image models also carry `image_params`, their image options; see
[Model Options](#model-options). A model object can also carry `family`, a
short key for the model's family, or `None` when the model belongs to none.
New values can be added, so do not treat the values you know as a complete
list.

### Account profile

`client.me.retrieve()` calls `GET /v1/me` and returns an `AccountProfile`:
`display_name`, `initial`, `email_hint`, and `avatar` (an `AccountAvatar` with
`url` and `etag`, or `None` when the account has no picture).
`client.me.avatar()` fetches the picture itself as a
[binary result](#binary-results); read `content-type` (`image/png`,
`image/jpeg`, or `image/webp`) and `etag` from its headers. Pass
`if_none_match="<etag>"` to send `If-None-Match`: an unchanged picture gives a
result with `status_code == 304` and an empty body. An account without a
picture raises `NotFoundError` (code `avatar_not_found`).

```python
profile = client.me.retrieve()
print(profile.display_name or profile.initial, profile.email_hint)

if profile.avatar is not None:
    with client.me.avatar() as picture:
        print("Content-Type:", picture.headers.get("content-type"))
        print("Bytes:", len(picture.read()))

    with client.me.avatar(if_none_match=profile.avatar.etag) as again:
        print("Same picture" if again.status_code == 304 else "New picture")
```

```python
import asyncio

from devupai import AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI() as client:
        profile = await client.me.retrieve()
        if profile.avatar is not None:
            async with await client.me.avatar() as picture:
                await picture.write_to_file("avatar")


asyncio.run(main())
```

---

## Webhooks

Verify incoming webhook deliveries and construct typed event payloads. Use the
functions `construct_webhook_event` and `verify_webhook_signature`, or the same
methods on `client.webhooks`. They run locally, send nothing, and are
synchronous on both clients.

A delivery carries the header `X-DevUp-Signature: t=<unix_timestamp>,v1=<hex_signature>`.
Pass the raw request body exactly as it was received (`bytes`, `bytearray`,
`memoryview`, or `str`), before any JSON parsing: a re-serialised body changes its bytes, and the
signature no longer matches.

### Verifying and parsing events

```python
import os
from collections.abc import Mapping

from devupai import DevupWebhookVerificationError, construct_webhook_event
from devupai.types import WebhookErrorPayload

WEBHOOK_SECRET = os.environ["DEVUP_WEBHOOK_SECRET"]


def handle_webhook(raw_body: bytes, headers: Mapping[str, str]) -> int:
    """Returns the HTTP status to answer the delivery with.

    ``headers`` are the request headers with lower-case names.
    """
    signature = headers.get("x-devup-signature")
    if signature is None:
        return 400
    try:
        event = construct_webhook_event(
            raw_body=raw_body,
            signature_header=signature,
            secret=WEBHOOK_SECRET,
        )
    except DevupWebhookVerificationError as err:
        print(f"Webhook verification error ({err.reason}): {err}")
        return 400

    if isinstance(event, WebhookErrorPayload):
        print("Inference error:", event.id, event.error.type, event.error.message)
    else:
        print("Inference result:", event.id, event.results)
        print("Cost:", event.cost_dzd, "DZD")
    return 200
```

The payload must be a JSON object. The result is a `WebhookErrorPayload` when
the event reports an error, and a `WebhookSuccessPayload` otherwise; fields the
SDK does not know yet are kept as extras.

The same check is available on a client, with the same arguments:

```python
event = client.webhooks.construct_event(
    raw_body=raw_body,
    signature_header=signature,
    secret=WEBHOOK_SECRET,
)
```

### Boolean signature verification

If you prefer a boolean result to an exception, use `verify_signature`. It
returns `True` when one `v1` signature matches and the timestamp is within the
tolerance, `False` otherwise, and never raises.

```python
import json
import os

from devupai import DevupAI

client = DevupAI()


def read_webhook(raw_body: bytes, signature_header: str) -> object:
    if not client.webhooks.verify_signature(
        raw_body=raw_body,
        signature_header=signature_header,
        secret=os.environ["DEVUP_WEBHOOK_SECRET"],
        tolerance_seconds=300,
    ):
        return None
    return json.loads(raw_body)
```

### Verification error reasons

When `construct_webhook_event` cannot verify a delivery, it raises a
`DevupWebhookVerificationError` with a `reason`:

| Reason | When it applies | What to do |
| ------ | --------------- | ---------- |
| `malformed_header` | The `X-DevUp-Signature` header is missing, empty, lacks a numeric `t=` timestamp, or contains no `v1=` signature; or `raw_body` is not `str` or bytes-like. | Make sure your web server passes the `X-DevUp-Signature` header through unchanged. |
| `timestamp_outside_tolerance` | The difference between `t` and the local clock exceeds `tolerance_seconds` (default 300), or `tolerance_seconds` is not a finite, non-negative number. | Synchronize your server clock with NTP, or increase `tolerance_seconds`. |
| `no_matching_signature` | No `v1` signature matches the HMAC of the raw body and timestamp, or `secret` is empty. | Confirm that `secret` matches your dashboard webhook secret, and that `raw_body` is the untouched raw body. |
| `invalid_json` | The signature matches, but the body is not valid JSON, is not a JSON object, or contains `NaN` or `Infinity`. | Make sure nothing between the sender and your handler changes or truncates the body. |

### Deduplication

The same delivery can arrive more than once: when your endpoint answers HTTP
408, 429, or 5xx, times out, or cannot be reached, it is sent again. Key
idempotency checks on the `X-DevUp-Delivery-Id` header, which stays the same
for every delivery of an event. The `t=` timestamp and the signature change on
each delivery, so do not key idempotency on the signature.

After a webhook secret rotation, the signature header can carry several `v1`
signatures for a while (one per valid secret). One matching signature is
enough, so you can rotate secrets without downtime.

---

# Configuration

```python
import os

from devupai import DevupAI

client = DevupAI(
    api_key=os.environ["DEVUP_API_KEY"],
    base_url="https://api.devupai.com/v1",
    default_headers={"X-Application-Name": "my-application"},
    max_retries=2,
    timeout=600.0,
    stream_idle_timeout=30.0,
)
```

Every option is a keyword argument, and `DevupAI` and `AsyncDevupAI` take the
same options.

| Option                | Type                                   | Default                      | Description |
| --------------------- | -------------------------------------- | ---------------------------- | ----------- |
| `api_key`             | `str \| None`                          | See [API key](#api-key)      | Your DEVUP AI API key. |
| `base_url`            | `str \| None`                          | `https://api.devupai.com/v1` | DEVUP AI API base URL. A trailing `/` is removed. It must be an absolute `http` or `https` URL without a user name or password. A user name or password, or a value that cannot be parsed as a URL (for example `http://[::1`), raises `ValueError` when the client is created. Any other value that is not an absolute `http` or `https` URL (another scheme, no host, a relative value) raises `ValueError` at the first call, before anything is sent; `images.download()` raises its `invalid_url` error instead. |
| `default_headers`     | `Mapping[str, str] \| None`            | —                            | Headers included in every request. |
| `max_retries`         | `int \| None`                          | `2`                          | How many times a request is sent again. See [Retries](#retries). |
| `timeout`             | `float \| None`                        | `600.0` (10 minutes)         | **Seconds.** One limit for the whole call, retries included. The Node.js SDK takes this value in milliseconds; this SDK takes seconds. See [Timeouts](#timeouts-and-cancellation). |
| `stream_idle_timeout` | `float \| None`                        | Off                          | Seconds a stream may go without data before it ends with `APITimeoutError`. `0` keeps it off. |
| `http_client`         | `httpx.Client` / `httpx.AsyncClient`   | A client the SDK creates     | Your own `httpx.Client` (for `DevupAI`) or `httpx.AsyncClient` (for `AsyncDevupAI`). See [Your own HTTP client](#your-own-http-client). |
| `on_debug`            | `Callable[[DebugEvent], None] \| None` | —                            | Receives debug events: `retry` (before each retry, with `attempt`, `status`, and `wait_ms`) and `malformed_frame` (a stream event that could not be parsed and was skipped). An exception raised in it does not change the outcome of the request. |

`timeout` and `stream_idle_timeout` must be numbers of seconds (`timeout`
positive, `stream_idle_timeout` non-negative), and `max_retries` a
non-negative integer. A wrong type raises `TypeError` and a wrong value raises
`ValueError`, before any request is sent.

The SDK sends `User-Agent: devupai-python/<version>`. It manages the
`Authorization` and `Content-Type` headers itself; custom headers cannot
override them. Redirects are not followed.

### API key

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

### Your own HTTP client

Pass `http_client` to use your own connection settings. Only its transport,
connection pool, and event hooks apply: its own `auth`, base URL, headers,
query parameters, cookies, and timeouts are not used (each request takes the
`timeout` of this client or of the call), and redirects are not followed. An
exception raised by one of its event hooks reaches you unchanged and is not
retried. The SDK never closes a client you pass.

```python
import httpx

from devupai import DevupAI


def log_request(request: httpx.Request) -> None:
    print(request.method, request.url)


http_client = httpx.Client(
    limits=httpx.Limits(max_connections=20),
    event_hooks={"request": [log_request]},
)

with DevupAI(http_client=http_client) as client:
    print(client.balance.retrieve().balance_dzd)

http_client.close()
```

---

# Request Options

Every API method takes request options as keyword arguments, next to the
body fields. They apply to that call only.

| Option                | Type                      | Methods                                                       | Description |
| --------------------- | ------------------------- | ------------------------------------------------------------- | ----------- |
| `timeout`             | `float`                   | All                                                           | Seconds for this call, instead of the client `timeout`. |
| `max_retries`         | `int`                     | All                                                           | Retries for this call, instead of the client `max_retries`. |
| `extra_headers`       | `Mapping[str, str]`       | All                                                           | Extra headers for this request. |
| `extra_body`          | `Mapping[str, object]`    | Every `POST` method except `audio.transcriptions.create()` and `video.edits.create()` | Fields added to the body, merged last. |
| `stream_idle_timeout` | `float`                   | `chat.completions.create()`, `responses.create()`             | Idle limit for this stream, instead of the client setting. `0` turns it off. |
| `if_none_match`       | `str`                     | `models.list()`, `models.retrieve()`, `me.avatar()`           | Sends `If-None-Match` with this value. |
| `response_type`       | `"json"` or `"binary"`    | `inference.run()`                                             | `"binary"` returns a [binary result](#binary-results). |

```python
completion = client.chat.completions.create(
    model="<chat-model-id>",
    messages=[{"role": "user", "content": "Summarize Algeria in three sentences."}],
    timeout=30.0,
    max_retries=0,
    extra_headers={"X-Request-Source": "documentation-example"},
)
```

---

# Request IDs, Headers, and Notices

Every resource has a `with_response` twin with the same methods. Its methods
return an `APIResponse`:

| Field               | Type             | Description                                                         |
| ------------------- | ---------------- | ------------------------------------------------------------------- |
| `data`              | the result       | The same value that the plain method returns.                       |
| `response`          | `httpx.Response` | The HTTP response, with its status and headers.                     |
| `request_id`        | `str \| None`    | The `x-request-id` header. Include it when you contact support.     |
| `client_request_id` | `str \| None`    | The `x-client-request-id` header.                                   |
| `warnings`          | `list[str]`      | Notice codes from the `X-DEVUP-Warnings` header (empty when none).  |

```python
result = client.chat.completions.with_response.create(
    model="<chat-model-id>",
    messages=[{"role": "user", "content": "Hello."}],
)

message = result.data.choices[0].message
print(message.content if message else "")
print("Request ID:", result.request_id)
print("Status:", result.response.status_code)
for notice in result.warnings:
    print("Notice:", notice)
```

On `AsyncDevupAI`, await it the same way:
`await client.chat.completions.with_response.create(...)`.

Streams expose the same information directly: `stream.response`,
`stream.request_id`, `stream.client_request_id`, and `stream.warnings`. Binary
results carry `request_id` and `headers`. Errors carry `request_id`,
`client_request_id`, and `warnings` too.

`warnings` lists the notice codes of the `X-DEVUP-Warnings` response header,
split at the commas (for example `parameter_not_supported`). Some answers also
list the notices in the body, in `devup.warnings`. The codes are stable: use
them in your code, not the message text. Notices are informational: the
request was answered.

---

# Retries

The client sends a request again, up to `max_retries` times (default 2). When
the answer has a `retry-after-ms` or `Retry-After` header, the client waits
that long before the next attempt. Otherwise it waits a random time that grows
with each attempt, at most 60 seconds between attempts. A wait that does not
fit in the time left of `timeout` ends the call with the error. A 500 answer
is not retried.

| Requests | Sent again after |
| -------- | ---------------- |
| Billed requests: chat completions, Responses, embeddings, rerank, `images.generate()`, `images.edit()`, speech, transcriptions, video generations and edits, and `inference.run()` | 429 and 503, and a connection that could not be opened |
| Other requests: models, balance, account profile and picture, health, and `images.download()` | 408, 429, 502, 503, and 504, and connection errors other than a read or write timeout |

A billed request is sent again only after a 429 or 503 answer, or when the
connection could not be opened (it could not be established, or no connection
from the pool became free in time). A TLS handshake that does not complete is
not sent again. After any other answer, or a connection lost after the request
may have been sent, the error is returned to you without a retry, because the
request may already have been processed and charged.

On every request, a read or write timeout ends the call with
`APITimeoutError` and is not sent again.

`max_retries` applies to both groups, and `max_retries=0` turns retries off.
To check a billed request's outcome before anything is sent again, set
`max_retries=0` on that call.

```python
from devupai import DebugEvent, DevupAI


def on_debug(event: DebugEvent) -> None:
    if event.type == "retry":
        print("Retry:", event.detail)


# No retries for any call made with this client.
client = DevupAI(max_retries=0, on_debug=on_debug)

# Up to 5 retries for this call only.
models = client.models.list(max_retries=5)
print(len(models.data))
```

---

# Timeouts and Cancellation

- `timeout` is in **seconds** (default `600.0`, 10 minutes). It is one
  deadline for the whole call, retries and the waits between them included.
  Opening a connection takes at most 10 seconds of it.
- For streams and binary results, `timeout` covers the time until the response
  headers arrive. Set `stream_idle_timeout` to end a stream when no data
  arrives for that many seconds.
- A timeout raises `APITimeoutError`.
- For video generation and long `inference.run()` calls, set a longer
  `timeout` (for example `timeout=900.0`) and `max_retries=0`.
- There is no cancel argument. On `AsyncDevupAI`, cancel the task that awaits
  the call, for example with `asyncio.timeout()`; the client stays usable.

```python
import asyncio

from devupai import APITimeoutError, AsyncDevupAI


async def main() -> None:
    async with AsyncDevupAI(timeout=60.0, stream_idle_timeout=30.0) as client:
        try:
            async with asyncio.timeout(10):
                completion = await client.chat.completions.create(
                    model="<chat-model-id>",
                    messages=[{"role": "user", "content": "Hello."}],
                )
            message = completion.choices[0].message
            print(message.content if message else "")
        except APITimeoutError:
            print("The request exceeded the client timeout.")
        except TimeoutError:
            print("The call was cancelled after 10 seconds.")


asyncio.run(main())
```

---

# Error Handling

Every error from the API is a `DevupAPIError`. The SDK raises the subclass
that matches the HTTP status, so you can catch a specific case or the base
class. Every class below except `DevupWebhookVerificationError` is a subclass
of `DevupAPIError`.

| Class                           | Status     | Notes |
| ------------------------------- | ---------- | ----- |
| `BadRequestError`               | 400        | The request is not valid as sent; `code` names the reason. |
| `AuthenticationError`           | 401        | Missing or invalid API key. |
| `PaymentRequiredError`          | 402        | The balance does not cover the request (`insufficient_balance`), or a scoped credential reached its budget (`budget_exhausted`). |
| `PermissionDeniedError`         | 403        | The key cannot use this resource. From `images.download()`: the link has expired; generate the file again. |
| `NotFoundError`                 | 404        | For example, an unknown model. |
| `ConflictError`                 | 409        | The outcome of this request is not confirmed yet (`request_outcome_pending`). Do not send it again: contact support with `request_id`. |
| `PayloadTooLargeError`          | 413        | The request body is larger than the limit (`payload_too_large`). See [Request Size Limits](#request-size-limits). |
| `UnsupportedMediaTypeError`     | 415        | The model does not accept the content type of the request. Send the `Content-Type` and body format that the API reference lists for the endpoint. |
| `UnprocessableEntityError`      | 422        | The request was not accepted in its current form. |
| `RateLimitError`                | 429        | Read `retry_after_ms` for the suggested wait. |
| `InternalServerError`           | 500 and up | After a 500 or 502 (`server_error`, `DEVUP_COMPUTE_ERROR`) the request may already have been processed and charged: check its outcome before sending it again. |
| `APIConnectionError`            | `0`        | The API could not be reached. |
| `APITimeoutError`               | `0`        | The call exceeded `timeout` or `stream_idle_timeout`. A subclass of `APIConnectionError`. |
| `DevupAPIError`                 | other      | Any other status. |
| `DevupWebhookVerificationError` | —          | A webhook delivery could not be verified; see [Webhooks](#verification-error-reasons). Not an API error: it carries a `reason` and a message, and no status. |

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

The underlying exception, when there is one, is chained as `__cause__`. The
error codes you can act on are listed at
[docs.devupai.com/docs/errors](https://docs.devupai.com/docs/errors).

```python
from devupai import (
    AuthenticationError,
    BadRequestError,
    DevupAI,
    DevupAPIError,
    RateLimitError,
)

client = DevupAI()

try:
    client.chat.completions.create(
        model="<chat-model-id>",
        messages=[{"role": "user", "content": "Hello."}],
    )
except AuthenticationError:
    print("Check your API key.")
except BadRequestError as err:
    print("Not valid as sent:", err.code, err.param, err.message)
except RateLimitError as err:
    print("Suggested wait (ms):", err.retry_after_ms)
except DevupAPIError as err:
    print("Status:", err.status)
    print("Code:", err.code)
    print("Request ID:", err.request_id)
```

A 409 means the outcome of the request is not confirmed yet. Do not send it
again: contact support with its `request_id`.

An error reported inside a stream after it has started (chat completions or
Responses) is raised from the loop as a `DevupAPIError` with the error's
`code` and the stream's `request_id`. A successful answer whose body is not
valid JSON raises a `DevupAPIError` with the HTTP status and the message "The
response body is not valid JSON."

### Arguments checked before sending

The SDK checks options and arguments before anything is sent:

- A value of the wrong type raises `TypeError`: for example `timeout="5"`,
  `max_retries=1.5`, an `api_key` that is not a `str`, an `http_client` of the
  wrong class, an `inference.run()` payload that is not a mapping, a file input
  of the wrong type (a `str` in a file-only field included) or a file opened in
  text mode, or an unknown keyword on `video.edits.create()`.
- A value of the right type that is not allowed raises `ValueError`: for
  example `timeout=0`, `max_retries=-1`, a negative `stream_idle_timeout`, a
  `base_url` that is not an absolute `http` or `https` URL or that contains a
  user name or password (see [Configuration](#configuration) for when it is
  raised), a missing API key, an `extra_body` `stream` that disagrees with the
  `stream` argument (see [Streaming](#streaming)), or an `inference.run()`
  `response_type` other than `"json"` or `"binary"` (of any type).
- In `images.edit()` sent as `multipart/form-data` (at least one input is a
  file), an item of `image`, `images`, `mask`, or `markup` that is not a `str`,
  a [file input](#file-inputs), or a mapping with a `str` `image_url` raises
  `TypeError`. A `None` item is left out.
- A header name or value that cannot be sent in an HTTP header (a line break, a
  NUL or another control character, a name that is not a `str` or not a
  valid token) raises `ValueError`; the message names the header but never
  repeats its value. A header value that is not a `str` is not sent.
- A call on a closed client, and a second iteration of a stream, raise
  `RuntimeError`.

## Request Size Limits

The request body of chat completions, Responses, embeddings, rerank, image
generation and edits, speech, transcriptions, video, and native inference can
be up to 10,485,760 bytes (10 MiB), uploaded files and Base64-encoded file data
included. A larger body returns HTTP 413
`payload_too_large`, raised as `PayloadTooLargeError`; the body is not
processed and the request is not charged. For large input images or video
media, send an `https://` URL instead of the file.

Some arrays have a limit too: chat `messages` takes up to 512 entries and
embeddings `input` up to 2,048 texts. A longer array returns HTTP 400
`payload_too_large`, raised as `BadRequestError`.

---

# Type Hints

The package ships `py.typed`. Request bodies are `TypedDict` classes and
responses are `pydantic` models, all in `devupai.types`: for example
`ChatCompletion`, `ChatCompletionChunk`, `ChatCompletionCreateParams`,
`ResponseObject`, `ResponseStreamEvent`, `EmbeddingCreateResponse`,
`ModelObject`, `ImageParams`, and `DevUpBilling`. With `stream=True` a type
checker sees a `Stream` (or `AsyncStream`) result, and without it the data
model.

```python
from devupai.types import ChatCompletion, ChatCompletionCreateParamsNonStreaming

request: ChatCompletionCreateParamsNonStreaming = {
    "model": "<chat-model-id>",
    "messages": [{"role": "user", "content": "Hello from Python."}],
}

completion: ChatCompletion = client.chat.completions.create(**request)
print(completion.to_dict())
```

### Response objects

- Keys the SDK does not know yet are kept: read them with `model_extra` or
  `to_dict()`. `to_dict()` gives JSON-ready data with the wire names (for
  example `_devup`) and only the keys received.
- A response that does not match its type never raises while it is read: each
  field keeps the value received, and a field that was not received reads as
  `None`.
- Types that list string values, such as `finish_reason` or
  `ImageParamsOption.kind`, list the values the API sends today. New values may
  be added, so give a `match` over them a default case.

### Nesting limits

- `to_dict()` handles up to 99 levels of nested objects and lists, counting the
  response object itself. A response nested 100 levels or more is still read,
  and its attributes work, but `to_dict()` raises `ValueError` ("Maximum
  serialization depth exceeded"). `repr()` works up to at least 2,900 levels on
  Python 3.12 and 3.14; closer to the decoding limit below, it can raise
  `RecursionError` on Python 3.12.
- A response body nested deeper than Python's `json` module can decode (about
  3,000 levels on Python 3.12; the limit depends on the Python version) raises
  `DevupAPIError` with the message "The response body is not valid JSON." In a
  stream, such an event is skipped and reported to `on_debug` as
  `malformed_frame`; a webhook body that deep raises
  `DevupWebhookVerificationError` with the reason `invalid_json`.

---

# Runtime and Security

- Python 3.11 or later, with `DevupAI` for synchronous code and `AsyncDevupAI`
  for `asyncio`.
- Built on `httpx`, with `pydantic` for the response models. Pass
  `http_client` to use your own connection settings.
- Webhook signature checks run locally and compare signatures in constant time
  (`hmac.compare_digest`).
- The SDK manages the `Authorization` and `Content-Type` headers, and refuses a
  header value with a line break or another control character.
- `health.check()` and `images.download()` never send your API key.
- `images.download()` only fetches URLs on the API's own origin, and refuses a
  URL that contains a user name or password.
- The SDK never puts your API key in an error message, and `repr(client)`
  shows only the `base_url`.

Store API keys in environment variables or a server-side secret manager.

Never commit API keys to a repository, and never ship them in client-side code.

---

# Models

DEVUP AI offers models across multiple modalities:

- Chat and reasoning
- Image generation and editing
- Embeddings
- Speech recognition
- Text-to-speech
- Video generation and editing
- Reranking and specialized inference

Model availability may change over time. `client.models.list()` and the live
catalog are the source of truth:

<p align="center">
  <a href="https://devupai.com/models"><strong>Browse available models →</strong></a>
</p>

---

# Get an API Key

1. Create an account at [devupai.com](https://devupai.com).
2. Open the [DEVUP AI Dashboard](https://devupai.com/dashboard).
3. Top up your balance using a supported payment method.
4. Generate an API key from the [API Keys page](https://devupai.com/dashboard/api-keys).
5. Store it securely as `DEVUP_API_KEY`.

```bash
export DEVUP_API_KEY="<your-api-key>"
```

---

# Resources

| Resource      | Link                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------- |
| Website       | [devupai.com](https://devupai.com)                                                           |
| Documentation | [docs.devupai.com](https://docs.devupai.com/)                                                |
| Error codes   | [docs.devupai.com/docs/errors](https://docs.devupai.com/docs/errors)                         |
| Dashboard     | [devupai.com/dashboard](https://devupai.com/dashboard)                                       |
| API Keys      | [devupai.com/dashboard/api-keys](https://devupai.com/dashboard/api-keys)                     |
| Models        | [devupai.com/models](https://devupai.com/models)                                             |
| Status        | [status.devupai.com](https://status.devupai.com/)                                            |
| PyPI          | [pypi.org/project/devupai](https://pypi.org/project/devupai/)                                |
| Examples      | [examples/](https://github.com/devupai-platform/devupai-python/tree/main/examples)           |
| Changelog     | [CHANGELOG.md](https://github.com/devupai-platform/devupai-python/blob/main/CHANGELOG.md)    |
| Source        | [github.com/devupai-platform/devupai-python](https://github.com/devupai-platform/devupai-python) |

The scripts in `examples/` read the model id from the `DEVUP_MODEL`
environment variable.

---

# License

The DEVUP AI Python SDK is released under the
[MIT License](https://github.com/devupai-platform/devupai-python/blob/main/LICENSE).

<p align="center">
  <strong>Built for Algeria. Compatible with the global AI ecosystem.</strong>
</p>

<p align="center">
  <a href="https://devupai.com">Website</a>
  ·
  <a href="https://docs.devupai.com/">Documentation</a>
  ·
  <a href="https://pypi.org/project/devupai/">PyPI</a>
</p>
