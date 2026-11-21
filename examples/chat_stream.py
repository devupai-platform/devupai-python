"""Streams a chat reply and prints each piece of text as it arrives.

Run it with your API key in DEVUP_API_KEY and a chat model id in DEVUP_MODEL:

    python examples/chat_stream.py

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
        # The stream is used in a ``with`` block, so its connection is released even if the loop stops early.
        with client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write a short poem about the sea."}],
            stream=True,
        ) as stream:
            for chunk in stream:
                for choice in chunk.choices:
                    if choice.delta.content:
                        print(choice.delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
