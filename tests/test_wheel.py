"""``scripts/check_wheel_contents.py``: the file list, ``py.typed``, the hashed word test and the size budget.

Twin of the Node SDK's ``npm pack --dry-run`` content check. Each rule is tested on small synthetic archives built in
a temporary folder, then once on the real wheel and sdist built from this checkout.

The word probe never spells a listed word: it takes one from an ``ALLOWED_TEXT`` entry of ``public_words`` and checks
its hash against the list, so this file names no listed word in plain text.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import check_wheel_contents as cwc
import public_words
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_wheel_contents.py"

VERSION_FILE = ROOT / "src" / "devupai" / "_version.py"


def _real_version() -> str:
    """The ``__version__`` string of this checkout, read from the file (the package is not imported)."""
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"""^__version__\s*(?::\s*str\s*)?=\s*["']([^"']+)["']""", text, re.MULTILINE)
    assert match, f"{VERSION_FILE} has no __version__ string"
    return match.group(1)


#: The package version: the synthetic archives carry it, and the real archives must.
VERSION = _real_version()
WHEEL_NAME = f"devupai-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"devupai-{VERSION}.tar.gz"
DIST_INFO = f"devupai-{VERSION}.dist-info"
SDIST_ROOT = f"devupai-{VERSION}"

#: The Q16 budget (block0 section-E, contract §7 rule 4), by the checker's module constant.
BUDGET = {
    "WHEEL_UNPACKED_LIMIT": 600_000,
    "WHEEL_FILE_LIMIT": 200_000,
    "WHEEL_FILE_COUNT_LIMIT": 60,
    "SDIST_FILE_LIMIT": 700_000,
}

#: The ``ALLOWED_TEXT`` entry used as the word probe, and the archive paths it maps from.
PROBE_SRC_PATH = "src/devupai/types/models.py"
PROBE_WHEEL_PATH = "devupai/types/models.py"
PROBE_SDIST_PATH = f"{SDIST_ROOT}/src/devupai/types/models.py"

BLOCKED = public_words.FORBIDDEN_TERM_HASHES


# --- synthetic archives ----------------------------------------------------------------------------------------


def clean_wheel_members() -> dict[str, bytes]:
    """The smallest wheel the checker accepts."""
    members = {
        "devupai/__init__.py": b'"""Fixture package."""\n',
        "devupai/py.typed": b"",
        "devupai/types/__init__.py": b"",
        "devupai/types/models.py": b"x = 1\n",
        f"{DIST_INFO}/METADATA": (
            f"Metadata-Version: 2.4\nName: devupai\nVersion: {VERSION}\nSummary: Fixture package.\n\nFixture text.\n"
        ).encode(),
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{DIST_INFO}/licenses/LICENSE": b"MIT License\n",
    }
    record = "".join(f"{name},,\n" for name in members) + f"{DIST_INFO}/RECORD,,\n"
    members[f"{DIST_INFO}/RECORD"] = record.encode()
    return members


def clean_sdist_members() -> dict[str, bytes]:
    """The smallest sdist the checker accepts."""
    return {
        f"{SDIST_ROOT}/PKG-INFO": f"Metadata-Version: 2.4\nName: devupai\nVersion: {VERSION}\n".encode(),
        f"{SDIST_ROOT}/pyproject.toml": b'[project]\nname = "devupai"\n',
        f"{SDIST_ROOT}/README.md": b"# devupai\n\nFixture text.\n",
        f"{SDIST_ROOT}/CHANGELOG.md": b"# Changelog\n",
        f"{SDIST_ROOT}/LICENSE": b"MIT License\n",
        f"{SDIST_ROOT}/src/devupai/__init__.py": b'"""Fixture package."""\n',
        f"{SDIST_ROOT}/src/devupai/py.typed": b"",
        f"{SDIST_ROOT}/src/devupai/types/__init__.py": b"",
        f"{SDIST_ROOT}/src/devupai/types/models.py": b"x = 1\n",
    }


def write_wheel(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def write_sdist(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return path


@dataclass
class Dist:
    folder: Path
    wheel_members: dict[str, bytes]
    sdist_members: dict[str, bytes]

    @property
    def wheel(self) -> Path:
        return self.folder / WHEEL_NAME

    @property
    def sdist(self) -> Path:
        return self.folder / SDIST_NAME

    def write(self) -> Path:
        """Writes both archives from the current member maps and returns the folder."""
        self.folder.mkdir(parents=True, exist_ok=True)
        write_wheel(self.wheel, self.wheel_members)
        write_sdist(self.sdist, self.sdist_members)
        return self.folder


@pytest.fixture
def dist(tmp_path: Path) -> Dist:
    return Dist(tmp_path / "dist", clean_wheel_members(), clean_sdist_members())


def run_check(folder: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = cwc.main([str(folder)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def probe() -> tuple[str, str]:
    """``(allowed text, listed word)`` from the ``ALLOWED_TEXT`` entry limited to ``PROBE_SRC_PATH``."""
    for text, files, _why in public_words.ALLOWED_TEXT:
        if files and PROBE_SRC_PATH in files:
            listed = [word for word in public_words.words(text) if public_words.term_hash(word) in BLOCKED]
            assert listed, f"the ALLOWED_TEXT entry {text!r} names no listed word"
            return text, listed[0]
    raise AssertionError(f"no ALLOWED_TEXT entry is limited to {PROBE_SRC_PATH}")


# --- self-checks -----------------------------------------------------------------------------------------------


def test_the_budget_constants_are_the_q16_budget() -> None:
    for name, value in BUDGET.items():
        assert getattr(cwc, name) == value, name


def test_the_fixture_text_names_no_listed_word() -> None:
    members = {**clean_wheel_members(), **clean_sdist_members()}
    for name, data in members.items():
        for line in data.decode("utf-8").splitlines():
            assert public_words.line_hits(name, line) == [], name


def test_the_probe_is_a_listed_word_allowed_only_in_its_file() -> None:
    text, word = probe()
    assert public_words.term_hash(word) in BLOCKED
    assert public_words.line_hits(PROBE_SRC_PATH, f"    {text} float") == []
    assert word in public_words.line_hits("src/devupai/types/account.py", f"    {text} float")


# --- a clean pair passes ---------------------------------------------------------------------------------------


def test_a_clean_wheel_and_sdist_pass(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_check(dist.write(), capsys)
    assert code == 0, out
    # One line per check: file list, forbidden folders, words, budget.
    assert len([line for line in out.splitlines() if line.strip()]) >= 4, out


def test_the_script_runs_from_the_command_line(dist: Dist) -> None:
    folder = dist.write()
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), str(folder)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    dist.wheel_members["devupai/data.json"] = b"{}\n"
    dist.write()
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), str(folder)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "devupai/data.json" in bad.stdout + bad.stderr


# --- rule 1: the wheel file list and py.typed ------------------------------------------------------------------


@pytest.mark.parametrize(
    "member",
    [
        "devupai/data.json",
        "devupai/types/models.pyi",
        "devupai/__pycache__/_client.cpython-312.pyc",
        "tests/x.py",
        "scripts/x.py",
        "other/x.py",
        "x.py",
        f"{DIST_INFO}/entry_points.txt",
        f"{DIST_INFO}/top_level.txt",
    ],
)
def test_a_stray_wheel_member_fails(member: str, dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    dist.wheel_members[member] = b"x = 1\n"
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert member in out


def test_a_wheel_without_py_typed_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    del dist.wheel_members["devupai/py.typed"]
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert "py.typed" in out


# --- rule 2: no openapi/, tests/ or scripts/ in either archive ------------------------------------------------


@pytest.mark.parametrize(
    "member", ["openapi/devupai-v1.yaml", "devupai/openapi/devupai-v1.yaml", "tests/test_x.py", "scripts/x.py"]
)
def test_a_forbidden_folder_in_the_wheel_fails(member: str, dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    dist.wheel_members[member] = b"x: 1\n"
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert member in out


@pytest.mark.parametrize(
    "member",
    [
        f"{SDIST_ROOT}/openapi/devupai-v1.yaml",
        f"{SDIST_ROOT}/openapi/SPEC_SHA256",
        f"{SDIST_ROOT}/tests/test_x.py",
        f"{SDIST_ROOT}/scripts/sync_spec.py",
    ],
)
def test_a_forbidden_folder_in_the_sdist_fails(member: str, dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    dist.sdist_members[member] = b"x: 1\n"
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert member in out


# --- rule 3: the hashed word test ------------------------------------------------------------------------------


def _with_word(word: str) -> bytes:
    return f'"""Fixture package."""\nvalue = "{word}"\n'.encode()


def test_a_listed_word_in_a_wheel_module_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    _text, word = probe()
    dist.wheel_members["devupai/__init__.py"] = _with_word(word)
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert "devupai/__init__.py:2" in out
    assert word in out


def test_a_listed_word_inside_a_snake_case_name_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    _text, word = probe()
    dist.wheel_members["devupai/types/__init__.py"] = f"x_{word}_y = 1\n".encode()
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert "devupai/types/__init__.py:1" in out


def test_a_listed_word_in_an_sdist_module_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    _text, word = probe()
    member = f"{SDIST_ROOT}/src/devupai/__init__.py"
    dist.sdist_members[member] = _with_word(word)
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert f"{member}:2" in out
    assert word in out


def test_a_listed_word_in_metadata_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    _text, word = probe()
    member = f"{DIST_INFO}/METADATA"
    dist.wheel_members[member] = (
        f"Metadata-Version: 2.4\nName: devupai\nVersion: {VERSION}\nSummary: Fixture package.\n\nThe {word} text.\n"
    ).encode()
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert f"{member}:6" in out
    assert word in out


@pytest.mark.parametrize("member", [f"{SDIST_ROOT}/README.md", f"{SDIST_ROOT}/PKG-INFO"])
def test_a_listed_word_in_sdist_text_fails(member: str, dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    _text, word = probe()
    dist.sdist_members[member] = dist.sdist_members[member] + f"The {word} text.\n".encode()
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert member in out
    assert word in out


def test_an_allowed_text_applies_at_its_mapped_archive_paths(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    # ALLOWED_TEXT names src/devupai/...; the wheel member devupai/... and the sdist member devupai-<v>/src/devupai/...
    # map to that path, so the exact field name passes in both archives.
    text, _word = probe()
    line = f"class Fixture:\n    {text} float | None = None\n".encode()
    dist.wheel_members[PROBE_WHEEL_PATH] = line
    dist.sdist_members[PROBE_SDIST_PATH] = line
    code, out = run_check(dist.write(), capsys)
    assert code == 0, out


@pytest.mark.parametrize(
    ("archive", "member"),
    [("wheel", "devupai/types/account.py"), ("sdist", f"{SDIST_ROOT}/src/devupai/types/account.py")],
)
def test_an_allowed_text_does_not_apply_to_other_files(
    archive: str, member: str, dist: Dist, capsys: pytest.CaptureFixture[str]
) -> None:
    text, word = probe()
    line = f"class Fixture:\n    {text} float | None = None\n".encode()
    (dist.wheel_members if archive == "wheel" else dist.sdist_members)[member] = line
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert f"{member}:2" in out
    assert word in out


def test_an_allowed_text_removes_only_its_own_text_in_an_archive(
    dist: Dist, capsys: pytest.CaptureFixture[str]
) -> None:
    text, word = probe()
    dist.wheel_members[PROBE_WHEEL_PATH] = f"x = 1\n    {text} float  # {word} values\n".encode()
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert f"{PROBE_WHEEL_PATH}:2" in out


# --- rule 4: the size budget -----------------------------------------------------------------------------------


def _unpacked(d: Dist) -> int:
    return sum(len(data) for data in d.wheel_members.values())


def _wheel_size(d: Dist) -> int:
    return os.path.getsize(d.wheel)


def _wheel_count(d: Dist) -> int:
    return len(d.wheel_members)


def _sdist_size(d: Dist) -> int:
    return os.path.getsize(d.sdist)


MEASURES: dict[str, Callable[[Dist], int]] = {
    "WHEEL_UNPACKED_LIMIT": _unpacked,
    "WHEEL_FILE_LIMIT": _wheel_size,
    "WHEEL_FILE_COUNT_LIMIT": _wheel_count,
    "SDIST_FILE_LIMIT": _sdist_size,
}


@pytest.mark.parametrize("limit", sorted(MEASURES))
def test_a_budget_limit_is_inclusive(
    limit: str, dist: Dist, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = dist.write()
    monkeypatch.setattr(cwc, limit, MEASURES[limit](dist))
    code, out = run_check(folder, capsys)
    assert code == 0, out


@pytest.mark.parametrize("limit", sorted(MEASURES))
def test_a_budget_limit_exceeded_fails(
    limit: str, dist: Dist, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = dist.write()
    value = MEASURES[limit](dist) - 1
    monkeypatch.setattr(cwc, limit, value)
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert str(value) in out or f"{value:,}" in out, out


# --- exactly one wheel and one sdist ---------------------------------------------------------------------------


def test_no_wheel_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    folder = dist.write()
    dist.wheel.unlink()
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert ".whl" in out


def test_two_wheels_fail(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    folder = dist.write()
    write_wheel(folder / "devupai-1.0.0a2-py3-none-any.whl", dist.wheel_members)
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert ".whl" in out


def test_no_sdist_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    folder = dist.write()
    dist.sdist.unlink()
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert ".tar.gz" in out


def test_two_sdists_fail(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    folder = dist.write()
    write_sdist(folder / "devupai-1.0.0a2.tar.gz", dist.sdist_members)
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert ".tar.gz" in out


def test_a_missing_folder_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_check(tmp_path / "missing", capsys)
    assert code != 0, out


# --- the real archives -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    out = tmp_path_factory.mktemp("build-dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    yield out


def _real_members(folder: Path) -> tuple[list[str], list[str]]:
    wheels = sorted(folder.glob("*.whl"))
    sdists = sorted(folder.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, sorted(p.name for p in folder.iterdir())
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = archive.getnames()
    return wheel_names, sdist_names


def test_the_real_archives_carry_the_package_version(built: Path) -> None:
    assert sorted(path.name for path in built.iterdir()) == sorted([WHEEL_NAME, SDIST_NAME])
    wheel_names, sdist_names = _real_members(built)
    assert f"{DIST_INFO}/METADATA" in wheel_names
    assert f"{SDIST_ROOT}/PKG-INFO" in sdist_names
    with zipfile.ZipFile(built / WHEEL_NAME) as archive:
        metadata = archive.read(f"{DIST_INFO}/METADATA").decode("utf-8")
    assert re.search(rf"^Version: {re.escape(VERSION)}$", metadata, re.MULTILINE), metadata[:400]
    statuses = re.findall(r"^Classifier: (Development Status :: .+)$", metadata, re.MULTILINE)
    if re.fullmatch(r"\d+\.\d+\.\d+", VERSION):
        assert statuses == ["Development Status :: 5 - Production/Stable"], statuses


def test_the_checker_passes_on_the_real_archives(built: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_check(built, capsys)
    assert code == 0, out


def test_the_real_archives_have_no_spec_tests_or_scripts(built: Path) -> None:
    wheel_names, sdist_names = _real_members(built)
    for name in wheel_names:
        assert not re.match(r"(openapi|tests|scripts)/", name), name
    for name in sdist_names:
        inner = name.split("/", 1)[1] if "/" in name else ""
        assert not re.match(r"(openapi|tests|scripts)(/|$)", inner), name


def test_the_real_wheel_has_every_module_and_py_typed(built: Path) -> None:
    wheel_names, _sdist_names = _real_members(built)
    src = ROOT / "src"
    modules = {
        path.relative_to(src).as_posix()
        for path in (src / "devupai").rglob("*.py")
        if "__pycache__" not in path.relative_to(src).parts
    }
    assert {name for name in wheel_names if name.endswith(".py")} == modules
    assert "devupai/py.typed" in wheel_names
