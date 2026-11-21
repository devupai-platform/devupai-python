"""Public text checks: the hashed word list, the single version literal, and synthetic model ids only.

Twin of the Node SDK's ``tests/public-surface.test.ts`` (word test, version literal, model ids), adapted to Python:
the word split also cuts snake_case names, and the list adds the third-party model companies and products, hashed so
that this public file does not name them.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
VERSION_FILE = "src/devupai/_version.py"
ROOT_FILES = ["README.md", "pyproject.toml"]

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
_SKIPPED_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dll"}
_SKIPPED_DIRS = {"__pycache__"}


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


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _walk(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if any(part in _SKIPPED_DIRS or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and path.suffix not in _SKIPPED_SUFFIXES:
            found.append(path)
    return found


def src_files() -> list[Path]:
    """Every text file under ``src/``."""
    return list(_walk(SRC))


def public_files() -> list[Path]:
    """The files the word test reads: ``src/``, README.md and pyproject.toml."""
    files = src_files()
    files.extend(ROOT / name for name in ROOT_FILES if (ROOT / name).is_file())
    return files


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - reported as a failure
        pytest.fail(f"{_rel(path)} is not UTF-8 text: {exc}")


# --- self-checks -----------------------------------------------------------------------------------------------


def test_the_self_check_hashes_the_way_the_list_was_built() -> None:
    assert term_hash("Example") == term_hash("example")
    assert term_hash("example") == hashlib.sha256(b"example").hexdigest()[:16]
    assert all(re.fullmatch(r"[0-9a-f]{16}", value) for value in FORBIDDEN_TERM_HASHES)


def test_the_word_split_covers_hyphen_snake_and_camel_parts() -> None:
    found = words("fooBar baz-qux snake_case_word HTTPClient __version__ camel_caseWord")
    assert {"foobar", "foo", "bar", "baz-qux", "baz", "qux"} <= set(found)
    assert {"snake_case_word", "snake", "case", "word"} <= set(found)
    assert {"camel", "caseword", "case", "word"} <= set(found)
    assert "version" in found
    assert "" not in found
    assert all(word == word.lower() for word in found)


def test_whole_word_matching_skips_longer_words_but_finds_snake_parts() -> None:
    probe = frozenset({term_hash("fixtureterm")})
    assert set(line_hits("README.md", "a fixtureterm b", probe)) == {"fixtureterm"}
    assert line_hits("README.md", "fixtureterms prefixfixtureterm", probe) == []
    assert "fixtureterm" in line_hits("README.md", "x_fixtureterm_tokens", probe)
    assert "fixtureterm" in line_hits("README.md", "someFixtureterm", probe)
    assert "fixtureterm" in line_hits("README.md", "pre-fixtureterm-post", probe)


def test_the_list_holds_node_s_entries_plus_the_company_entries() -> None:
    assert len(NODE_TERM_HASHES) == 13 + 26
    assert len(ARCHITECTURE_TERM_HASHES) == 14
    assert ARCHITECTURE_TERM_HASHES <= NODE_TERM_HASHES
    assert len(COMPANY_TERM_HASHES) == 50
    assert not COMPANY_TERM_HASHES & NODE_TERM_HASHES
    assert NODE_TERM_HASHES <= FORBIDDEN_TERM_HASHES
    # 39 Node entries + 0 new architecture entries + 50 company entries.
    assert len(FORBIDDEN_TERM_HASHES) == 39 + 0 + 50


_SPEC_CITE = re.compile(r"openapi/devupai-v1\.yaml:(\d+)")


@pytest.mark.parametrize("entry", ALLOWED_TEXT, ids=[f"{e[0]}@{(e[1] or ('*',))[0]}" for e in ALLOWED_TEXT])
def test_each_exception_is_an_exact_spec_name_limited_to_its_files(
    entry: tuple[str, tuple[str, ...] | None, str],
) -> None:
    # P2 decision B-5: the exception list holds only exact field names copied from the spec. Each entry must name a
    # listed word, be limited to named files, appear in each of them, apply nowhere else, and match the spec line it
    # cites.
    text, files, why = entry
    assert files, "an exception must be limited to the files that type the field"
    assert any(term_hash(word) in FORBIDDEN_TERM_HASHES for word in words(text)), f"{text!r} names no listed word"
    for name in files:
        assert text in _read_text(ROOT / name), f"{text!r} does not appear in {name}"
    assert line_hits("README.md", text), f"{text!r} must not be allowed outside its files"
    cite = _SPEC_CITE.search(why)
    assert cite, f"{why!r} must cite the spec line"
    spec_lines = _read_text(ROOT / "openapi" / "devupai-v1.yaml").splitlines()
    spec_line = spec_lines[int(cite.group(1)) - 1]
    name_in_spec = text.strip('"').rstrip(":")
    assert re.search(rf"(^|[\s\[,]){re.escape(name_in_spec)}(:|[\],]|$)", spec_line.strip()), (
        f"{name_in_spec!r} is not on the cited spec line {cite.group(1)}: {spec_line.strip()!r}"
    )


def test_an_exception_removes_only_its_own_text() -> None:
    probe = frozenset({term_hash("cached"), term_hash("settlement")})
    assert line_hits("src/devupai/types/models.py", "    cached: float | None = None", probe) == []
    assert set(line_hits("src/devupai/types/models.py", "    # cached values", probe)) == {"cached"}
    assert set(line_hits("src/devupai/types/account.py", "    cached: float | None = None", probe)) == {"cached"}


def test_scope_is_not_empty() -> None:
    checked = {_rel(path) for path in public_files()}
    assert VERSION_FILE in checked
    assert "src/devupai/__init__.py" in checked
    assert "README.md" in checked
    assert "pyproject.toml" in checked


# --- the word test ---------------------------------------------------------------------------------------------


def test_no_listed_word_in_src_readme_or_pyproject() -> None:
    hits: list[str] = []
    for path in public_files():
        name = _rel(path)
        for number, line in enumerate(re.split(r"\r?\n", _read_text(path)), start=1):
            found = line_hits(name, line)
            if found:
                hits.append(f"{name}:{number} [{', '.join(dict.fromkeys(found))}]")
    assert "\n".join(hits) == ""


# --- version ---------------------------------------------------------------------------------------------------


def _version() -> str:
    text = (ROOT / VERSION_FILE).read_text(encoding="utf-8")
    match = re.search(r"""^__version__\s*(?::\s*str\s*)?=\s*["']([^"']+)["']""", text, re.MULTILINE)
    assert match, f"{VERSION_FILE} has no __version__ string"
    return match.group(1)


def test_the_version_is_a_pep_440_version() -> None:
    from packaging.version import Version

    version = _version()
    assert str(Version(version)) == version


def test_src_contains_no_other_literal_copy_of_the_version() -> None:
    version = _version()
    hits = [_rel(path) for path in src_files() if _rel(path) != VERSION_FILE and version in _read_text(path)]
    assert hits == []


# --- model ids -------------------------------------------------------------------------------------------------

_SYNTHETIC_ID = re.compile(r"fixture/[a-z0-9-]+|<[^>]+>")


def _is_model_name(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "model") or (
        isinstance(node, ast.Attribute) and node.attr == "model"
    )


def model_values(source: str) -> list[str]:
    """String literals given as a model value: ``model=...``, ``"model": ...``, ``model = ...`` and defaults of a
    ``model`` parameter."""
    found: list[ast.expr] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "model":
            found.append(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "model":
                    found.append(value)
        elif isinstance(node, ast.Assign) and any(_is_model_name(target) for target in node.targets):
            found.append(node.value)
        elif isinstance(node, ast.AnnAssign) and _is_model_name(node.target) and node.value is not None:
            found.append(node.value)
        elif isinstance(node, ast.arguments):
            positional = node.posonlyargs + node.args
            for arg, default in zip(positional[len(positional) - len(node.defaults) :], node.defaults, strict=True):
                if arg.arg == "model":
                    found.append(default)
            for arg, kw_default in zip(node.kwonlyargs, node.kw_defaults, strict=True):
                if arg.arg == "model" and kw_default is not None:
                    found.append(kw_default)
    return [value.value for value in found if isinstance(value, ast.Constant) and isinstance(value.value, str)]


def _bad_model_values(files: Iterable[Path]) -> list[str]:
    bad: list[str] = []
    for path in files:
        if path.suffix != ".py":
            continue
        for value in model_values(_read_text(path)):
            if not _SYNTHETIC_ID.fullmatch(value):
                bad.append(f"{_rel(path)}: {value}")
    return bad


def test_the_model_value_scan_finds_every_form() -> None:
    source = (
        'f(model="a/one")\n'
        'x = {"model": "b/two"}\n'
        'model = "c/three"\n'
        'self.model = "d/four"\n'
        'model: str = "e/five"\n'
        'def g(model="f/six"): ...\n'
        'def h(*, model="g/seven"): ...\n'
        'def k(model: "SomeType") -> None: ...\n'
        'f(model=MODEL_A, other="h/eight")\n'
    )
    assert sorted(model_values(source)) == ["a/one", "b/two", "c/three", "d/four", "e/five", "f/six", "g/seven"]
    assert _SYNTHETIC_ID.fullmatch("fixture/model-a")
    assert _SYNTHETIC_ID.fullmatch("<model-id>")
    assert not _SYNTHETIC_ID.fullmatch("vendor/model-b")


def test_src_uses_only_synthetic_ids_or_placeholders_as_model_values() -> None:
    files = src_files()
    assert files, "src/ has no files"
    assert _bad_model_values(files) == []


def test_tests_use_only_synthetic_ids_or_placeholders_as_model_values() -> None:
    assert _bad_model_values(_walk(ROOT / "tests")) == []
