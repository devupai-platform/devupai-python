"""Generates an image from a prompt and prints its link, or saves it when the API returns the image bytes.

Run it with your API key in DEVUP_API_KEY and an image model id in DEVUP_MODEL:

    python examples/images.py

An image returned as base64 (``b64_json``) is saved in the current folder as image-1.png (or .jpg, .webp).
python examples/models_list.py text-to-image lists the image model ids your key can use.
"""

import base64
import os
from pathlib import Path

from devupai import DevupAI


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of an image model.\n"
            "Run python examples/models_list.py text-to-image to list the image model ids your key can use."
        )
    return model


def extension(data: bytes) -> str:
    """A file extension for the image bytes, read from their first bytes."""
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def main() -> None:
    model = require_model()
    with DevupAI() as client:
        result = client.images.generate(model=model, prompt="A lighthouse on a cliff at sunset, watercolor")

    for number, image in enumerate(result.data, start=1):
        if image.url:
            print(f"Image {number}: {image.url}")
        elif image.b64_json:
            data = base64.b64decode(image.b64_json)
            path = Path(f"image-{number}{extension(data)}")
            path.write_bytes(data)
            print(f"Image {number} saved to {path}")


if __name__ == "__main__":
    main()
