"""Public text checks: the hashed word list, the single version literal, and synthetic model ids only.

Twin of the Node SDK's ``tests/public-surface.test.ts`` (word test, version literal, model ids), adapted to Python:
the word split also cuts snake_case names, and the list adds the third-party model companies and products, hashed so
that this public file does not name them.

The list and the word split live in ``scripts/public_words.py``, shared with ``scripts/check_wheel_contents.py``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pytest
from public_words import (
    ALLOWED_TEXT,
    ARCHITECTURE_TERM_HASHES,
    COMPANY_TERM_HASHES,
    FORBIDDEN_TERM_HASHES,
    NODE_TERM_HASHES,
    line_hits,
    term_hash,
    words,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
VERSION_FILE = "src/devupai/_version.py"
ROOT_FILES = ["README.md", "pyproject.toml"]

_SKIPPED_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dll"}
_SKIPPED_DIRS = {"__pycache__"}


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
