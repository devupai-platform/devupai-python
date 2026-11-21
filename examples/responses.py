"""Creates a response with the Responses API and prints its text.

Run it with your API key in DEVUP_API_KEY and a model id in DEVUP_MODEL:

    python examples/responses.py

python examples/models_list.py lists the model ids your key can use.
"""

import os
from typing import cast

from devupai import DevupAI
from devupai.types import ResponseObject


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of a model.\n"
            "Run python examples/models_list.py to list the model ids your key can use."
        )
    return model


def output_text(response: ResponseObject) -> str:
    """The text parts of the message items in ``response.output``, joined."""
    texts: list[str] = []
    for item in response.output or []:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in cast("list[object]", content):
            if not isinstance(part, dict):
                continue
            fields = cast("dict[str, object]", part)
            text = fields.get("text")
            if fields.get("type") == "output_text" and isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def main() -> None:
    model = require_model()
    with DevupAI() as client:
        response = client.responses.create(
            model=model,
            instructions="Answer in one sentence.",
            input="Why do streamed replies feel faster?",
        )
    print(output_text(response))


if __name__ == "__main__":
    main()
