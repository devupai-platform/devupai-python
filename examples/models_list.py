"""Lists the models your API key can use, so you can pick a value for DEVUP_MODEL.

Run it with your API key in DEVUP_API_KEY:

    python examples/models_list.py

Pass a model type to list only the models of that type, for example:

    python examples/models_list.py text-generation

Then set DEVUP_MODEL to one of the ids before you run the other examples.
"""

import sys

from devupai import DevupAI
from devupai.types import ModelObject


def pick(models: list[ModelObject], model_type: str | None) -> list[ModelObject]:
    """The models whose ``type`` is ``model_type`` (all of them when it is ``None``).

    Models that a newer version replaces come last.
    """
    chosen = [model for model in models if model_type is None or model.type == model_type]
    return sorted(chosen, key=lambda model: bool(model.is_replaced))


def main() -> None:
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    with DevupAI() as client:
        catalog = client.models.list()

    models = pick(catalog.data, model_type)
    if not models:
        types = sorted({model.type or "" for model in catalog.data})
        print(f"No model of type {model_type!r}. Types in the list: {', '.join(types)}")
        return

    width = max(len(model.id) for model in models)
    for model in models:
        note = " (a newer version replaces it)" if model.is_replaced else ""
        print(f"{model.id:<{width}}  {model.type or '':<20}  {model.name}{note}")
    print()
    print(f"Set DEVUP_MODEL to one of these ids, for example DEVUP_MODEL={models[0].id}")


if __name__ == "__main__":
    main()
