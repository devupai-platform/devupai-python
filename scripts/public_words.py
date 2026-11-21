"""The hashed word list of public text, shared by ``tests/test_public_surface.py`` and ``check_wheel_contents.py``.

Twin of the Node SDK's ``tests/public-surface.test.ts`` word list, adapted to Python: the word split also cuts
snake_case names, and the list adds the third-party model companies and products, hashed so that this public file
does not name them.

File names in ``ALLOWED_TEXT`` are repository paths (``src/devupai/...``). A checker that reads an archive maps each
member to that form first.

Standard library only.
"""

from __future__ import annotations

import hashlib
import re

#: Node's list (``tests/public-surface.test.ts``), copied verbatim: 13 wording-rule entries + 26 review-list entries.
NODE_TERM_HASHES = frozenset(
    {
        # Wording rules of the SDK documentation and the error codes that are not public.
        "4ea5ee68fea05586",
        "a74e503354fdd464",
        "1581e27de87bffae",
        "7b76de4c4c1b0103",
        "ed2db96c7d99f13e",
        "bcd475104a873a42",
        "5c7ee2074b65853f",
        "dbf880608a5fe7ae",
        "fc6672ade6fd208a",
        "88fc4d7bf12e0d73",
        "7b9fde2dd505251b",
        "6b86ac84384ccbf2",
        "09c27ea4804eaf5d",
        # The review word list, with inflections.
        # Keeping stored copies of responses: 4 forms.
        "5e1ecee06a7fc06f",
        "efbe9d5743db5752",
        "3673014e72b67383",
        "6fe78deb222783bc",
        # Setting funds aside before a request: 4 forms.
        "e8b22d83b417e85b",
        "3b97260d76d3d7df",
        "c20dea4d876b5b8f",
        "e77b141e2388d56c",
        # Finalizing a charge: 5 forms (incl. the noun).
        "6868e83de35c465d",
        "36809acb6fbb9c3e",
        "d31e014808ecdd2a",
        "0a0dcac5952689a5",
        "a75c433ab2ae1daf",
        # Choosing where a request goes: 4 more forms.
        "8a84e406c08ac959",
        "e22bc966d9352957",
        "a45814b02024a5fa",
        "74c95604043427f0",
        # Conditional code paths: 3 forms.
        "f38c764c8aa00b65",
        "dce1905351040e79",
        "0b007b691d007943",
        # The service behind a model: 2 forms.
        "5c4c1964340aca5b",
        "1e19bab29bc6432a",
        # Forwarding: 4 forms.
        "1241936d4dd3aad6",
        "a04c3b825fab3627",
        "b09f17ea1b5ca77b",
        "ea2fae23d900274b",
    }
)

#: Architecture words: 14 entries. Every one is already in Node's list, so they add nothing to the set size.
ARCHITECTURE_TERM_HASHES = frozenset(
    {
        "1581e27de87bffae",
        "5c4c1964340aca5b",
        "1e19bab29bc6432a",
        "1241936d4dd3aad6",
        "a04c3b825fab3627",
        "4ea5ee68fea05586",
        "a74e503354fdd464",
        "a45814b02024a5fa",
        "8a84e406c08ac959",
        "e22bc966d9352957",
        "fc6672ade6fd208a",
        "74c95604043427f0",
        "5e1ecee06a7fc06f",
        "3673014e72b67383",
    }
)

#: Third-party model companies and products: 50 entries (owner decision: hashed, no plain names in the repository).
COMPANY_TERM_HASHES = frozenset(
    {
        "7d3194f79e645c42",
        "c70eca6b0f88f44d",
        "c857d09db23e6822",
        "bbdefa2950f49882",
        "5d72436256ada538",
        "90642c3dd333bc0e",
        "920510199770f4d6",
        "c97504235ab15009",
        "72e8088a05b38ae3",
        "0caee180da9d0f3f",
        "22df1c188b2f4e2e",
        "7f871cbf905f6e0c",
        "0b79cf2de0ee6b8e",
        "6fcba5d96a4fadfd",
        "add92b9cde2bdbf3",
        "6f7ac1823da81d2e",
        "047364b9be67f665",
        "67f2d22514622d1b",
        "6749675d80a1faa7",
        "e4f9c522e1c89280",
        "b5d641b432cca81e",
        "e1517be14d06cdb7",
        "a89151ba4b5ac0f2",
        "5388bc10736a0eff",
        "3669ffb921cca6ee",
        "45d780f82c9247fc",
        "a8182c1c17bb2847",
        "79f43c3b9c23fbdf",
        "2d7f45d7b98b427f",
        "cbc62794911ff31b",
        "9fbf261b62c1d7c0",
        "8d4321d936320802",
        "77b30b8a88051252",
        "dfc0fdff7af68ed2",
        "7ea0a2c0ff0cb888",
        "2864ab1d9269d363",
        "86c0ecb1d45c6af6",
        "e853b4e183ed0870",
        "8abe43a7f9a7bd94",
        "8a105898989c6e17",
        "b858f760399ed936",
        "b23b068c8142639b",
        "2834e64f3da4f1eb",
        "0cbcb08ff19b748e",
        "fc5a1047f5919892",
        "053ea4804ef1bb33",
        "60965168ce762e94",
        "d931372ba7f836a0",
        "66c6f57f73358bfa",
        "665e05102ac943a0",
    }
)

#: sha256 (first 16 hex chars) of lower-cased terms that must not appear in public text.
FORBIDDEN_TERM_HASHES = NODE_TERM_HASHES | ARCHITECTURE_TERM_HASHES | COMPANY_TERM_HASHES

#: A legitimate use of a listed word: ``(text, files or None for everywhere, why)``. The text is removed from a line
#: before its words are checked. Only exact field names (and enum values) copied from the vendored spec are allowed
#: here, each limited to the files that type that field; ``why`` cites the spec ``file:line`` the self-test reads.
ALLOWED_TEXT: list[tuple[str, tuple[str, ...] | None, str]] = [
    (
        "cached:",
        ("src/devupai/types/models.py",),
        "ModelPricing field name, spec openapi/devupai-v1.yaml:6544",
    ),
    (
        "cached_tokens:",
        ("src/devupai/types/shared.py",),
        "TokenUsage.prompt_tokens_details field name, spec openapi/devupai-v1.yaml:4164",
    ),
    (
        "cached_tokens:",
        ("src/devupai/types/responses.py",),
        "ResponsesResponse usage input_tokens_details field name, spec openapi/devupai-v1.yaml:4633",
    ),
    (
        "settlement:",
        ("src/devupai/types/webhooks.py",),
        "webhook payload field name, spec openapi/devupai-v1.yaml:3293 (also :3336)",
    ),
    (
        '"settled"',
        ("src/devupai/types/webhooks.py",),
        "enum value of that webhook field, spec openapi/devupai-v1.yaml:3295",
    ),
]

_WORD = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def term_hash(term: str) -> str:
    """The first 16 hex chars of sha256 of the lower-cased term."""
    return hashlib.sha256(term.lower().encode("utf-8")).hexdigest()[:16]


def words(text: str) -> list[str]:
    """Words of a text, lower-cased: identifiers and hyphenated words, plus their hyphen, snake_case and camelCase
    parts."""
    out: list[str] = []
    for token in _WORD.findall(text):
        out.append(token)
        for part in token.split("-"):
            out.append(part)
            out.extend(_CAMEL.split(part))
            for sub in part.split("_"):
                out.append(sub)
                out.extend(_CAMEL.split(sub))
    return [word.lower() for word in out if word]


def _applies(files: tuple[str, ...] | None, name: str) -> bool:
    return files is None or name in files


def line_hits(name: str, line: str, hashes: frozenset[str] = FORBIDDEN_TERM_HASHES) -> list[str]:
    """The listed words left in ``line`` of file ``name`` once the allowed texts of that file are removed."""
    rest = line
    for text, files, _why in ALLOWED_TEXT:
        if _applies(files, name):
            rest = rest.replace(text, " ")
    return [word for word in words(rest) if term_hash(word) in hashes]
