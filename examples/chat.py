"""Sends one chat message and prints the reply.

Run it with your API key in DEVUP_API_KEY and a chat model id in DEVUP_MODEL:

    python examples/chat.py

python examples/models_list.py lists the model ids your key can use.
"""

import os

from devupai import DevupAI


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of a chat model.\n"
            "Run python examples/models_list.py to list the model ids your key can use."
        )
    return model


def main() -> None:
    model = require_model()
    with DevupAI() as client:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
                {"role": "user", "content": "What is an API key used for?"},
            ],
        )
    for choice in completion.choices:
        if choice.message is not None and isinstance(choice.message.content, str):
            print(choice.message.content)


if __name__ == "__main__":
    main()
