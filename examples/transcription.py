"""Transcribes an audio file and prints the text.

Run it with your API key in DEVUP_API_KEY, a speech-to-text model id in DEVUP_MODEL, and the path of an audio
file (for example a .wav or .mp3 file):

    python examples/transcription.py recording.wav

python examples/models_list.py lists the model ids your key can use.
"""

import os
import sys
from pathlib import Path

from devupai import DevupAI


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of a speech-to-text model.\n"
            "Run python examples/models_list.py to list the model ids your key can use."
        )
    return model


def audio_path() -> Path:
    """The audio file named on the command line. Exits with a message when it is missing."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python examples/transcription.py <path to an audio file>")
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"No file at {path}")
    return path


def main() -> None:
    path = audio_path()
    model = require_model()
    with DevupAI() as client:
        # A path is opened and closed by the SDK, and sent as a file upload.
        transcript = client.audio.transcriptions.create(file=path, model=model)
    print(transcript.text)


if __name__ == "__main__":
    main()
