"""``CHANGELOG.md``: the newest entry is the package version, in the Node SDK's entry format.

Twin of the Node SDK's changelog layout: one ``## [x.y.z]`` heading per release (no date), an untitled lead paragraph,
then ``### `` sections. The GitHub release body of a version is exactly the body of its entry, so the release notes
file kept next to this repository must match it.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import public_words
import pytest
from packaging.version import Version

import devupai

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"

#: The GitHub release body of each version, kept with the release reports next to this repository. A standalone
#: checkout of the SDK does not have them.
_REPORTS = ROOT.parent / ".agent-prompts" / "reports"
RELEASE_NOTES = _REPORTS / "SDK-RETRY-HINT-1" / "release-notes-1.0.1.md"
PUBLISHED_NOTES = {"1.0.0": _REPORTS / "PYSDK-1" / "p4" / "release-notes-1.0.0.md"}

RELEASE = "1.0.1"
PREVIOUS = "1.0.0"

#: sha256 of each published entry body. A released entry never changes.
PUBLISHED_BODY_SHA256 = {
    "1.0.0": "aef59d26ac3f7f95854e4d991df89c960e219df3a7fd67a3db8ebb624e13a439",
    "1.0.0a1": "4333eaa4588e6cc2694e58976e9bdf815bc858f0a6ea1837aac75cd36a0bfe87",
}

#: The ``### `` section titles the Node SDK changelog uses.
SECTIONS = frozenset({"Breaking changes", "Notes", "Added", "Changed", "Fixed", "Removed", "Docs"})

STABLE = "Development Status :: 5 - Production/Stable"

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]$", re.MULTILINE)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_NEGATIVE = re.compile(r"\b(ignored|failed|not working|unfortunately|sorry|bugs?)\b", re.IGNORECASE)


def _text() -> str:
    return CHANGELOG.read_bytes().decode("utf-8")


def _entries() -> list[tuple[str, str]]:
    """``(version, body)`` per entry, newest first; each body without the heading, ending in one newline."""
    text = _text()
    matches = list(_HEADING.finditer(text))
    entries: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries.append((match.group("version"), text[match.end() : end].strip("\n") + "\n"))
    return entries


def _entry(version: str) -> str:
    for found, body in _entries():
        if found == version:
            return body
    raise AssertionError(f"CHANGELOG.md has no ## [{version}] entry")


def _classifiers() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    classifiers = data["project"]["classifiers"]
    assert isinstance(classifiers, list)
    return [str(item) for item in classifiers]


def _is_final(version: str) -> bool:
    parsed = Version(version)
    return not (parsed.is_prerelease or parsed.is_devrelease or parsed.is_postrelease or parsed.local)


# --- the version -----------------------------------------------------------------------------------------------


def test_the_first_entry_is_the_package_version() -> None:
    entries = _entries()
    assert entries, "CHANGELOG.md has no ## [x.y.z] entry"
    assert entries[0][0] == devupai.__version__


def test_the_package_version_is_the_release() -> None:
    assert devupai.__version__ == RELEASE


def test_the_release_sits_directly_above_the_previous_entry() -> None:
    versions = [version for version, _body in _entries()]
    assert versions[:2] == [RELEASE, PREVIOUS], versions


def test_the_version_is_a_final_release() -> None:
    version = devupai.__version__
    assert _is_final(version), version
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version


def test_the_development_status_follows_the_version() -> None:
    statuses = [item for item in _classifiers() if item.startswith("Development Status ::")]
    assert len(statuses) == 1, statuses
    if _is_final(devupai.__version__):
        assert statuses == [STABLE]


# --- the format ------------------------------------------------------------------------------------------------


def test_every_release_heading_is_a_bare_version() -> None:
    for line in _text().splitlines():
        if line.startswith("## "):
            assert _HEADING.fullmatch(line), line
            Version(line[4:-1])


def test_no_heading_carries_a_date() -> None:
    for line in _text().splitlines():
        if line.startswith("#"):
            assert not _DATE.search(line), line


def test_no_entry_repeats_a_version() -> None:
    versions = [version for version, _body in _entries()]
    assert len(versions) == len(set(versions)), versions


def test_the_release_entry_has_a_lead_paragraph_then_node_sections() -> None:
    body = _entry(RELEASE)
    lead, _sep, rest = body.partition("\n### ")
    assert lead.strip(), "the entry starts with a section instead of a lead paragraph"
    assert not lead.lstrip().startswith(("#", "-")), lead
    titles = re.findall(r"^### (.+)$", body, re.MULTILINE)
    assert titles, "the entry has no ### section"
    assert len(titles) == len(set(titles)), titles
    assert set(titles) <= SECTIONS, sorted(set(titles) - SECTIONS)
    assert not re.search(r"^#{1,2} |^#{4,} ", body, re.MULTILINE), "only ### headings belong inside an entry"
    for section in ("### " + rest).split("\n### ")[1:]:
        title, _nl, content = section.partition("\n")
        assert re.search(r"^- \S", content, re.MULTILINE), f"### {title} has no list item"


def test_the_release_entry_names_the_one_behaviour_change_in_its_lead() -> None:
    lead = _entry(RELEASE).partition("\n### ")[0]
    assert "`x-should-retry: false`" in lead
    assert "No breaking changes" in lead


def test_the_release_entry_installs_without_pre() -> None:
    body = _entry(RELEASE)
    assert re.search(r"`pip install (--upgrade )?devupai`", body), body
    assert not re.search(r"pip install [^`\n]*--pre", body), body


@pytest.mark.parametrize("version", sorted(PUBLISHED_BODY_SHA256))
def test_a_published_entry_is_unchanged(version: str) -> None:
    body = _entry(version)
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == PUBLISHED_BODY_SHA256[version]


def test_the_previous_entry_is_a_published_one() -> None:
    assert PREVIOUS in PUBLISHED_BODY_SHA256


# --- the wording -----------------------------------------------------------------------------------------------


def test_the_changelog_has_no_date() -> None:
    hits = [line for line in _text().splitlines() if _DATE.search(line)]
    assert hits == []


def test_the_changelog_names_no_listed_word() -> None:
    hits: list[str] = []
    for number, line in enumerate(_text().splitlines(), start=1):
        found = public_words.line_hits("CHANGELOG.md", line)
        if found:
            hits.append(f"CHANGELOG.md:{number} [{', '.join(dict.fromkeys(found))}]")
    assert hits == []


def test_the_changelog_uses_neutral_wording() -> None:
    hits = [line for line in _text().splitlines() if _NEGATIVE.search(line)]
    assert hits == []


# --- the GitHub release body -----------------------------------------------------------------------------------


def test_the_release_notes_are_the_release_entry_body() -> None:
    if not RELEASE_NOTES.is_file():
        pytest.skip(f"{RELEASE_NOTES.name} is kept next to the repository, not in a standalone checkout")
    notes = RELEASE_NOTES.read_bytes().decode("utf-8")
    assert "\r" not in notes
    assert notes == _entry(RELEASE)


@pytest.mark.parametrize("version", sorted(PUBLISHED_NOTES))
def test_published_release_notes_still_match_their_entry(version: str) -> None:
    path = PUBLISHED_NOTES[version]
    if not path.is_file():
        pytest.skip(f"{path.name} is kept next to the repository, not in a standalone checkout")
    assert path.read_bytes().decode("utf-8") == _entry(version)
