"""Turns a sentence into speech and prints the link to the audio file.

Run it with your API key in DEVUP_API_KEY and a text-to-speech model id in DEVUP_MODEL:

    python examples/speech.py

python examples/models_list.py lists the model ids your key can use.
"""

import os

from devupai import DevupAI


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of a text-to-speech model.\n"
            "Run python examples/models_list.py to list the model ids your key can use."
        )
    return model


def main() -> None:
    model = require_model()
    with DevupAI() as client:
        speech = client.audio.speech.create(model=model, input="Welcome to DEVUP AI.")
    print(f"Audio URL: {speech.url}")


if __name__ == "__main__":
    main()
