# Examples

Small scripts that call the DEVUP AI API with the `devupai` package. Each one
does one thing and prints its result.

| Script             | What it does |
| ------------------ | ------------ |
| `models_list.py`   | Lists the model ids your key can use. Pass a type (for example `text-generation`) to list only that type. |
| `chat.py`          | Sends one chat message and prints the reply. |
| `chat_stream.py`   | Streams a chat reply and prints it as it arrives. |
| `async_chat.py`    | Streams a chat reply with the asynchronous client. |
| `responses.py`     | Creates a response with the Responses API and prints its text. |
| `images.py`        | Generates an image and prints its link, or saves it when the API returns the bytes. |
| `embeddings.py`    | Creates embeddings for two texts and prints their size. |
| `speech.py`        | Turns a sentence into speech and prints the link to the audio file. |
| `transcription.py` | Transcribes the audio file given as its argument. |
| `balance.py`       | Prints the balance of your account. |
| `webhooks.py`      | Verifies a signed sample webhook delivery and reads its event. |

## Run them

Install the package and set your API key:

```bash
pip install devupai
export DEVUP_API_KEY="your-api-key"
```

On Windows PowerShell, use `$env:DEVUP_API_KEY = "your-api-key"`.

The scripts never contain a model id. Pick one from the list, then set
`DEVUP_MODEL` to it:

```bash
python examples/models_list.py
export DEVUP_MODEL="<model-id>"
python examples/chat.py
```

Use a model of the right type for each script: a chat model for the chat
scripts and `responses.py`, an image model for `images.py`, and so on. A script
that needs a model stops with a message when `DEVUP_MODEL` is not set.
`models_list.py` and `balance.py` need only the API key.

`transcription.py` takes the path of an audio file:

```bash
python examples/transcription.py recording.wav
```

`webhooks.py` needs no API key. It reads your webhook signing secret from
`DEVUP_WEBHOOK_SECRET`:

```bash
export DEVUP_WEBHOOK_SECRET="your-signing-secret"
python examples/webhooks.py
```
