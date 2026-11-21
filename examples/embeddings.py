"""Creates embeddings for two texts and prints the size of each vector and its first values.

Run it with your API key in DEVUP_API_KEY and an embedding model id in DEVUP_MODEL:

    python examples/embeddings.py

python examples/models_list.py lists the model ids your key can use.
"""

import os

from devupai import DevupAI


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of an embedding model.\n"
            "Run python examples/models_list.py to list the model ids your key can use."
        )
    return model


def main() -> None:
    model = require_model()
    texts = ["The cat sleeps on the sofa.", "A kitten naps on the couch."]
    with DevupAI() as client:
        result = client.embeddings.create(model=model, input=texts)

    print(f"{len(result.data)} embeddings")
    for item in result.data:
        vector = item.embedding
        if isinstance(vector, list):
            first = ", ".join(f"{value:g}" for value in vector[:3])
            print(f"#{item.index}: {len(vector)} dimensions, starts with [{first}, ...]")


if __name__ == "__main__":
    main()
