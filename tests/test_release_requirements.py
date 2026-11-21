"""``requirements-release.txt``: the hashed, fully pinned set the release build job installs (STOP-3 answer 1).

Every requirement is ``name==version`` with at least one ``--hash=sha256:<64 hex>``, no name appears twice, and the
three build pins equal ``requirements-dev.txt`` and the ``[build-system] requires`` of ``pyproject.toml``.

The closure check walks the declared requirements (``importlib.metadata``) of the distributions installed in the test
venv, starting at the three build pins, with the markers evaluated for the release runner: Linux, CPython 3.12,
x86_64, no extras (an edge that names extras also walks those extras of its target). Every name it reaches must be
pinned in the file, on a line that applies on that runner, and every walked edge's specifier must accept the pinned
version. The edges of a node come from its installed metadata when the installed version is its pinned version (the
metadata is then the pinned release's). The Linux-only packages are not installed on a Windows machine, so their
edges come from ``LINUX_ONLY_REQUIRES``: a fixed copy of the ``requires_dist`` that PyPI publishes for those pinned
versions (review C-1). The table is checked against the file's pins, and against the installed metadata wherever that
release is installed. A node that is neither installed at its pin nor in the table is reached, and checked as
pinned, but its own edges are not walked. When every reached node was walked, the reached set must equal the lines
that apply on the runner, so the file holds no package beyond the closure.

``pip install --require-hashes`` on the runner is the final proof: it refuses a file that misses a dependency or a
hash for that platform.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
RELEASE_REQUIREMENTS = ROOT / "requirements-release.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"
PYPROJECT = ROOT / "pyproject.toml"

#: The packages the release build job runs; the closure starts here.
ROOTS = ("build", "hatchling", "twine")

#: The marker environment of the release runner (GitHub ``ubuntu-latest``, ``python-version: "3.12"``).
RUNNER: dict[str, str] = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.0",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
}

#: The ``requires_dist`` lines without an ``extra`` marker, as ``https://pypi.org/pypi/<name>/<version>/json``
#: publishes them for the pinned versions of the packages the runner needs on Linux only, and of keyring, whose
#: Linux-only edges start that part of the closure. Used where the release is not installed (any non-Linux machine).
LINUX_ONLY_REQUIRES: dict[str, tuple[str, tuple[str, ...]]] = {
    "keyring": (
        "25.7.0",
        (
            'pywin32-ctypes>=0.2.0; sys_platform == "win32"',
            'SecretStorage>=3.2; sys_platform == "linux"',
            'jeepney>=0.4.2; sys_platform == "linux"',
            'importlib_metadata>=4.11.4; python_version < "3.12"',
            "jaraco.classes",
            "jaraco.functools",
            "jaraco.context",
        ),
    ),
    "secretstorage": ("3.5.0", ("cryptography>=2.0", "jeepney>=0.6")),
    "jeepney": ("0.9.0", ()),
    "cryptography": (
        "50.0.2",
        (
            'cffi>=2.0.0; platform_python_implementation != "PyPy"',
            'typing-extensions>=4.13.2; python_full_version < "3.11"',
        ),
    ),
    "cffi": ("2.1.1", ('pycparser; implementation_name != "PyPy"',)),
    "pycparser": ("3.12", ()),
}

_HASH_OPTION = re.compile(r"--hash=sha256:[0-9a-f]{64}")
_DEV_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+)$")


@dataclass(frozen=True)
class Pin:
    """One requirement of the file."""

    line: str
    name: NormalizedName
    version: str
    marker: Marker | None
    hashes: tuple[str, ...]

    def applies_on_runner(self) -> bool:
        return self.marker is None or self.marker.evaluate(dict(RUNNER, extra=""))


def _logical_lines(text: str) -> list[str]:
    """The requirement lines, with ``\\`` continuations joined, comments and blank lines dropped."""
    lines: list[str] = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        lines.append((current + stripped).strip())
        current = ""
    assert not current, "the file ends with a line continuation"
    return lines


def _parse(line: str) -> Pin:
    start = line.find("--")
    assert start > 0, f"no --hash option: {line[:80]}"
    requirement_text, options = line[:start].strip(), line[start:].split()
    assert options, f"no --hash option: {line[:80]}"
    for option in options:
        assert _HASH_OPTION.fullmatch(option), f"not a --hash=sha256:<64 hex> option: {option}"
    requirement = Requirement(requirement_text)
    assert not requirement.extras, f"extras are not pinned: {requirement_text}"
    assert requirement.url is None, f"a URL is not a pin: {requirement_text}"
    specifiers = list(requirement.specifier)
    assert len(specifiers) == 1, f"exactly one specifier: {requirement_text}"
    (specifier,) = specifiers
    assert specifier.operator == "==" and "*" not in specifier.version, f"not an exact == pin: {requirement_text}"
    Version(specifier.version)  # a valid, exact version
    hashes = tuple(option.split(":", 1)[1] for option in options)
    assert len(set(hashes)) == len(hashes), f"a hash is repeated: {requirement_text}"
    return Pin(
        line=requirement_text,
        name=canonicalize_name(requirement.name),
        version=specifier.version,
        marker=requirement.marker,
        hashes=hashes,
    )


def _pins() -> list[Pin]:
    assert RELEASE_REQUIREMENTS.is_file(), "requirements-release.txt does not exist"
    return [_parse(line) for line in _logical_lines(RELEASE_REQUIREMENTS.read_text(encoding="utf-8"))]


def _by_name() -> dict[NormalizedName, Pin]:
    return {pin.name: pin for pin in _pins()}


def _dev_pins() -> dict[NormalizedName, str]:
    pins: dict[NormalizedName, str] = {}
    for raw in DEV_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        match = _DEV_PIN.match(line)
        if match:
            pins[canonicalize_name(match.group("name"))] = match.group("version")
    return pins


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _requires_of_pin(name: NormalizedName, version: str) -> tuple[list[str], str] | None:
    """The declared requirements of ``name==version`` and where they come from, or ``None`` when unknown here."""
    installed = _installed_version(name)
    if installed is not None and Version(installed) == Version(version):
        return list(metadata.requires(name) or []), "installed"
    table = LINUX_ONLY_REQUIRES.get(name)
    if table is not None and Version(table[0]) == Version(version):
        return list(table[1]), "table"
    return None


def _without_extras(lines: list[str]) -> list[str]:
    """The requirement lines whose marker names no ``extra``, normalised by ``packaging``."""
    kept: list[str] = []
    for text in lines:
        requirement = Requirement(text)
        if requirement.marker is None or "extra" not in str(requirement.marker):
            kept.append(str(requirement))
    return sorted(kept)


# --- the file's shape -------------------------------------------------------------------------------------------


def test_the_file_starts_with_a_comment_saying_what_it_is_for() -> None:
    assert RELEASE_REQUIREMENTS.is_file(), "requirements-release.txt does not exist"
    first = RELEASE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# ") and "release" in first.lower()


def test_every_requirement_is_an_exact_pin_with_sha256_hashes() -> None:
    pins = _pins()
    assert pins, "requirements-release.txt holds no requirement"
    for pin in pins:
        assert pin.hashes, pin.line


def test_no_name_appears_twice() -> None:
    names = [pin.name for pin in _pins()]
    assert len(names) == len(set(names)), sorted(name for name in names if names.count(name) > 1)


def test_the_file_has_no_option_other_than_hashes() -> None:
    # An index, a nested file or an editable install would let pip fetch something the hashes do not cover.
    assert RELEASE_REQUIREMENTS.is_file(), "requirements-release.txt does not exist"
    for raw in RELEASE_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("-"):
            assert stripped.startswith("--hash=sha256:"), stripped


def test_the_build_pins_equal_requirements_dev() -> None:
    release, dev = _by_name(), _dev_pins()
    for name in ROOTS:
        root = canonicalize_name(name)
        assert root in release, f"{name} is not pinned in requirements-release.txt"
        assert root in dev, f"{name} is not pinned in requirements-dev.txt"
        assert release[root].version == dev[root], name
        assert release[root].marker is None, f"{name} must install on every platform"


def test_the_build_backend_pin_equals_pyproject() -> None:
    release = _by_name()
    requires = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["build-system"]["requires"]
    assert requires, "pyproject.toml has no [build-system] requires"
    for text in requires:
        requirement = Requirement(text)
        name = canonicalize_name(requirement.name)
        assert name in release, f"{requirement.name} is not pinned in requirements-release.txt"
        assert str(requirement.specifier) == f"=={release[name].version}", text


def test_every_marker_names_only_the_platform() -> None:
    # A marker only says on which platform a line applies (the closure test checks the runner gets what it needs).
    allowed = {"sys_platform", "os_name", "==", "!=", "and", "or"}
    for pin in _pins():
        if pin.marker is not None:
            words = {word for word in str(pin.marker).split() if not word.startswith('"')}
            assert words and words <= allowed, pin.line


# --- the closure ------------------------------------------------------------------------------------------------


def test_every_dependency_on_the_runner_is_pinned_and_accepts_its_pin() -> None:
    pins = _by_name()
    queue: list[tuple[NormalizedName, frozenset[str]]] = [(canonicalize_name(name), frozenset()) for name in ROOTS]
    reached: set[NormalizedName] = set()
    walked: dict[NormalizedName, set[str]] = {}
    not_walked: dict[NormalizedName, str] = {}
    while queue:
        name, extras = queue.pop(0)
        reached.add(name)
        assert name in pins, f"{name} is needed on the runner but not pinned"
        pin = pins[name]
        assert pin.applies_on_runner(), f"{name} is needed on the runner but its line does not apply there"
        found = _requires_of_pin(name, pin.version)
        if found is None:
            installed = _installed_version(name)
            not_walked[name] = "not installed" if installed is None else f"installed {installed}"
            continue
        lines, source = found
        assert not (source == "table" and extras), f"{name}[{','.join(sorted(extras))}]: the table holds no extras"
        done = walked.setdefault(name, set())
        wanted = {""} | set(extras)
        if wanted <= done:
            continue
        done |= wanted
        for text in lines:
            requirement = Requirement(text)
            if requirement.marker is not None and not any(
                requirement.marker.evaluate(dict(RUNNER, extra=extra)) for extra in sorted(wanted)
            ):
                continue
            target = canonicalize_name(requirement.name)
            assert target in pins, f"{name}=={pin.version} needs {text!r}, which is not pinned"
            assert requirement.specifier.contains(pins[target].version, prereleases=True), (
                f"{name}=={pin.version} needs {text!r}, but the pin is {pins[target].version}"
            )
            queue.append((target, frozenset(requirement.extras)))

    for name in ROOTS:
        assert canonicalize_name(name) in walked, f"{name} is not installed at its pin here, the walk proves nothing"
    # Review C-1: the part only the Linux runner installs is walked on every machine (installed metadata or table).
    linux_only = {canonicalize_name(name) for name in LINUX_ONLY_REQUIRES}
    assert linux_only <= reached, sorted(linux_only - reached)
    assert not linux_only & set(not_walked), {name: not_walked[name] for name in linux_only & set(not_walked)}
    on_runner = {pin.name for pin in pins.values() if pin.applies_on_runner()}
    assert reached <= on_runner
    if not not_walked:
        assert reached == on_runner, f"pinned but not needed on the runner: {sorted(on_runner - reached)}"


def test_the_linux_only_table_matches_the_pins() -> None:
    pins = _by_name()
    for name, (version, _lines) in LINUX_ONLY_REQUIRES.items():
        pin = pins.get(canonicalize_name(name))
        assert pin is not None, f"{name} is in LINUX_ONLY_REQUIRES but not pinned"
        assert pin.version == version, f"LINUX_ONLY_REQUIRES has {name} {version}, the pin is {pin.version}"
        assert pin.applies_on_runner(), name


def test_the_linux_only_table_matches_the_installed_metadata() -> None:
    # Wherever a release of the table is installed at its pinned version (keyring in the dev venvs, all of them on a
    # Linux machine with the same releases), the table must hold exactly its requirement lines without an extra. The
    # dev venvs do not pin keyring itself, so a machine with another keyring release compares nothing here.
    for name, (version, lines) in LINUX_ONLY_REQUIRES.items():
        installed = _installed_version(name)
        if installed is None or Version(installed) != Version(version):
            continue
        assert _without_extras(list(lines)) == _without_extras(list(metadata.requires(name) or [])), name


@pytest.mark.parametrize("name", ROOTS)
def test_each_build_pin_has_a_wheel_hash_and_an_sdist_hash(name: str) -> None:
    # PyPI publishes a wheel and an sdist for each of them; both hashes let pip take either file.
    assert len(_by_name()[canonicalize_name(name)].hashes) >= 2, name
