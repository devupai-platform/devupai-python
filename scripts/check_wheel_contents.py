"""Check the built wheel and sdist before they are uploaded.

Usage::

    python scripts/check_wheel_contents.py <dist dir>

The folder must hold exactly one ``.whl`` and one ``.tar.gz``. The checks, one output line each:

1. the file lists, with no duplicate member name in either archive:
   - the wheel holds only ``devupai/**/*.py``, ``devupai/py.typed`` and the ``devupai-<v>.dist-info`` files
     ``METADATA``, ``WHEEL``, ``RECORD`` and ``licenses/LICENSE``, and ``py.typed`` is present;
   - the sdist holds only ``devupai-<v>/{PKG-INFO, pyproject.toml, README.md, CHANGELOG.md, LICENSE, .gitignore}``
     and ``devupai-<v>/src/devupai/**/*.py`` plus ``devupai-<v>/src/devupai/py.typed``;
2. neither archive has a member under ``openapi/``, ``tests/`` or ``scripts/``;
3. the hashed word test of ``public_words`` passes over every text member of both archives, ``METADATA`` included
   (every member as stored, so an earlier copy of a duplicate name is read too);
4. the size budget: the wheel unpacked, the wheel file, the wheel file count and the sdist file.

Members are read in memory; nothing is extracted. Exit codes: 0 every check passes, 1 a check fails (each problem is
printed as ``member:line [words]`` or as the limit exceeded), 2 a usage error.

Standard library only (plus ``public_words`` next to this file).
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import public_words

PREFIX = "check_wheel_contents"
USAGE = "usage: python scripts/check_wheel_contents.py <dist dir>"

PACKAGE = "devupai"

#: The Q16 size budget, in bytes and files. The checks read these names at call time.
WHEEL_UNPACKED_LIMIT = 600_000
WHEEL_FILE_LIMIT = 200_000
WHEEL_FILE_COUNT_LIMIT = 60
SDIST_FILE_LIMIT = 700_000

#: Folders that never belong in either archive.
FORBIDDEN_DIRS = frozenset({"openapi", "tests", "scripts"})

#: The files of the wheel's ``.dist-info`` folder.
DIST_INFO_FILES = frozenset({"METADATA", "WHEEL", "RECORD", "licenses/LICENSE"})

#: The files allowed at the root of the sdist, next to ``src/devupai/``.
SDIST_ROOT_FILES = frozenset({"PKG-INFO", "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", ".gitignore"})

_WHEEL_NAME = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>[^-]+)-.+\.whl$")
_SDIST_SUFFIX = ".tar.gz"


class UsageError(Exception):
    """A command line or a dist folder that this script does not accept."""


@dataclass
class Archive:
    """The regular-file members of one archive, read in memory, in stored order (duplicate names kept)."""

    path: Path
    entries: list[tuple[str, bytes]] = field(default_factory=list[tuple[str, bytes]])
    #: Members that are neither a regular file nor a folder (links, devices).
    irregular: list[str] = field(default_factory=list[str])

    @property
    def names(self) -> list[str]:
        """Every member name, sorted, duplicates included."""
        return sorted(name for name, _data in self.entries)

    def has(self, name: str) -> bool:
        return any(entry_name == name for entry_name, _data in self.entries)

    def duplicates(self) -> list[str]:
        counts = Counter(name for name, _data in self.entries)
        counts.update(self.irregular)
        return sorted(name for name, count in counts.items() if count > 1)


@dataclass
class Result:
    """The problems found by one check."""

    title: str
    problems: list[str] = field(default_factory=list[str])


def read_wheel(path: Path) -> Archive:
    archive = Archive(path)
    with zipfile.ZipFile(path) as wheel:
        for info in wheel.infolist():
            if info.is_dir():
                continue
            archive.entries.append((info.filename, wheel.read(info)))
    return archive


def read_sdist(path: Path) -> Archive:
    archive = Archive(path)
    with tarfile.open(path, "r:gz") as sdist:
        for member in sdist.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                archive.irregular.append(member.name)
                continue
            handle = sdist.extractfile(member)
            if handle is None:  # pragma: no cover - a regular file always has a handle
                archive.irregular.append(member.name)
                continue
            with handle:
                archive.entries.append((member.name, handle.read()))
    return archive


def _only(folder: Path, suffix: str) -> Path:
    found = sorted(path for path in folder.iterdir() if path.is_file() and path.name.endswith(suffix))
    if len(found) != 1:
        names = ", ".join(path.name for path in found) or "none"
        raise UsageError(f"{folder} must hold exactly one {suffix} file, found {len(found)}: {names}")
    return found[0]


def find_archives(folder: Path) -> tuple[Path, Path]:
    """The single wheel and the single sdist of ``folder``."""
    if not folder.is_dir():
        raise UsageError(f"{folder} is not a folder")
    return _only(folder, ".whl"), _only(folder, _SDIST_SUFFIX)


def dist_info_dir(wheel: Path) -> str:
    """``<name>-<version>.dist-info``, from the wheel file name."""
    match = _WHEEL_NAME.match(wheel.name)
    if not match:
        raise UsageError(f"{wheel.name} is not a wheel file name")
    return f"{match.group('name')}-{match.group('version')}.dist-info"


def sdist_root_dir(sdist: Path) -> str:
    """``<name>-<version>``, from the sdist file name."""
    return sdist.name[: -len(_SDIST_SUFFIX)]


def sdist_inner(name: str) -> str:
    """An sdist member without its ``<name>-<version>/`` root folder."""
    return name.split("/", 1)[1] if "/" in name else ""


def repository_path(archive: str, name: str) -> str:
    """The repository path of an archive member, the form ``public_words.ALLOWED_TEXT`` names.

    ``devupai/x.py`` (wheel) and ``devupai-<v>/src/devupai/x.py`` (sdist) both map to ``src/devupai/x.py``. Any other
    wheel member keeps its name; any other sdist member loses only its root folder.
    """
    if archive == "wheel":
        return f"src/{name}" if name.startswith(f"{PACKAGE}/") else name
    return sdist_inner(name)


def _package_file(path: str) -> bool:
    """``devupai/**/*.py`` (outside ``__pycache__``) or ``devupai/py.typed``."""
    if not path.startswith(f"{PACKAGE}/"):
        return False
    return path == f"{PACKAGE}/py.typed" or (path.endswith(".py") and "__pycache__" not in path.split("/"))


# --- the checks -------------------------------------------------------------------------------------------------


def check_file_lists(wheel: Archive, dist_info: str, sdist: Archive, sdist_root: str) -> Result:
    result = Result("wheel and sdist file lists")
    for name in wheel.duplicates():
        result.problems.append(f"{name}: a duplicate member name in the wheel")
    for name in sdist.duplicates():
        result.problems.append(f"{name}: a duplicate member name in the sdist")

    for name in sorted(set(wheel.names)):
        if name.startswith(f"{dist_info}/"):
            allowed = name[len(dist_info) + 1 :] in DIST_INFO_FILES
        else:
            allowed = _package_file(name)
        if not allowed:
            result.problems.append(f"{name}: not an allowed wheel member")
    if not wheel.has(f"{PACKAGE}/py.typed"):
        result.problems.append(f"{PACKAGE}/py.typed: missing from the wheel")
    for name in (f"{dist_info}/METADATA", f"{dist_info}/RECORD"):
        if not wheel.has(name):
            result.problems.append(f"{name}: missing from the wheel")

    for name in sorted(set(sdist.names)):
        inner = name[len(sdist_root) + 1 :] if name.startswith(f"{sdist_root}/") else None
        allowed = inner is not None and (
            inner in SDIST_ROOT_FILES or (inner.startswith("src/") and _package_file(inner[len("src/") :]))
        )
        if not allowed:
            result.problems.append(f"{name}: not an allowed sdist member")
    return result


def _forbidden_dir(path: str) -> bool:
    return any(part in FORBIDDEN_DIRS for part in path.split("/")[:-1])


def check_forbidden_dirs(wheel: Archive, sdist: Archive) -> Result:
    result = Result("no openapi/, tests/ or scripts/ folder")
    for name in sorted(set(wheel.names)):
        if _forbidden_dir(name):
            result.problems.append(f"{name}: a forbidden folder in the wheel")
    for name in sorted(set(sdist.names) | set(sdist.irregular)):
        if _forbidden_dir(sdist_inner(name)):
            result.problems.append(f"{name}: a forbidden folder in the sdist")
    for name in sorted(sdist.irregular):
        result.problems.append(f"{name}: not a regular file in the sdist")
    return result


def _word_hits(archive: str, entries: list[tuple[str, bytes]]) -> Iterable[str]:
    # Every stored entry, in archive order: an earlier copy of a duplicate name is read too.
    for name, data in entries:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            yield f"{name}: not UTF-8 text"
            continue
        mapped = repository_path(archive, name)
        for number, line in enumerate(re.split(r"\r?\n", text), start=1):
            found = public_words.line_hits(mapped, line)
            if found:
                yield f"{name}:{number} [{', '.join(dict.fromkeys(found))}]"


def check_words(wheel: Archive, sdist: Archive) -> Result:
    result = Result("hashed word test over the wheel, the sdist and METADATA")
    result.problems.extend(_word_hits("wheel", wheel.entries))
    result.problems.extend(_word_hits("sdist", sdist.entries))
    return result


def _over(label: str, value: int, limit: int, unit: str) -> list[str]:
    if value <= limit:
        return []
    return [f"{label}: {value:,} {unit} exceeds the limit of {limit:,} {unit}"]


def check_budget(wheel: Archive, sdist: Archive) -> Result:
    # The limits are module globals read here, at call time, so a test can lower one.
    result = Result("size budget")
    unpacked = sum(len(data) for _name, data in wheel.entries)
    count = len(wheel.entries)
    wheel_size = wheel.path.stat().st_size
    sdist_size = sdist.path.stat().st_size
    result.problems.extend(_over("wheel unpacked size", unpacked, WHEEL_UNPACKED_LIMIT, "B"))
    result.problems.extend(_over("wheel file size", wheel_size, WHEEL_FILE_LIMIT, "B"))
    result.problems.extend(_over("wheel file count", count, WHEEL_FILE_COUNT_LIMIT, "files"))
    result.problems.extend(_over("sdist file size", sdist_size, SDIST_FILE_LIMIT, "B"))
    result.title = (
        f"size budget (wheel {unpacked:,} B unpacked, {wheel_size:,} B file, {count} files; sdist {sdist_size:,} B)"
    )
    return result


def run(folder: Path) -> list[Result]:
    wheel_path, sdist_path = find_archives(folder)
    dist_info = dist_info_dir(wheel_path)
    wheel = read_wheel(wheel_path)
    sdist = read_sdist(sdist_path)
    return [
        check_file_lists(wheel, dist_info, sdist, sdist_root_dir(sdist_path)),
        check_forbidden_dirs(wheel, sdist),
        check_words(wheel, sdist),
        check_budget(wheel, sdist),
    ]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0]:
        print(f"{PREFIX}: {USAGE}", file=sys.stderr)
        return 2
    try:
        results = run(Path(args[0]))
    except UsageError as exc:
        print(f"{PREFIX}: FAIL: {exc}")
        return 1
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(f"{PREFIX}: FAIL: cannot read the archives: {exc}")
        return 1
    failed = False
    for result in results:
        print(f"{PREFIX}: {'FAIL' if result.problems else 'ok'}: {result.title}")
        for problem in result.problems:
            print(f"  {problem}")
        failed = failed or bool(result.problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
