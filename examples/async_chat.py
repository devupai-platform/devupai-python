"""Streams a chat reply with the asynchronous client, AsyncDevupAI.

Run it with your API key in DEVUP_API_KEY and a chat model id in DEVUP_MODEL:

    python examples/async_chat.py

python examples/models_list.py lists the model ids your key can use.
"""

import asyncio
import os

from devupai import AsyncDevupAI


def require_model() -> str:
    """The model id from DEVUP_MODEL. Exits with a message when it is not set."""
    model = os.environ.get("DEVUP_MODEL", "").strip()
    if not model:
        raise SystemExit(
            "Set the DEVUP_MODEL environment variable to the id of a chat model.\n"
            "Run python examples/models_list.py to list the model ids your key can use."
        )
    return model


async def main() -> None:
    model = require_model()
    async with AsyncDevupAI() as client:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Name three uses of asynchronous code."}],
            stream=True,
        )
        # ``async with`` releases the stream's connection even if the loop stops early.
        async with stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    if choice.delta.content:
                        print(choice.delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
