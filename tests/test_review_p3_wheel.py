"""P3 review fixes in ``scripts/check_wheel_contents.py``.

- L-7: the sdist has an allowlist too: ``devupai-<v>/{PKG-INFO, pyproject.toml, README.md, CHANGELOG.md, LICENSE,
  .gitignore}`` and ``devupai-<v>/src/devupai/**/*.py`` plus ``py.typed``. Any other member fails.
- I-4: a duplicate member name fails, and the word test reads every member, so an earlier copy of a name cannot hide.
"""

from __future__ import annotations

import io
import tarfile
import warnings
import zipfile
from pathlib import Path

import check_wheel_contents as cwc
import pytest
from test_wheel import SDIST_ROOT, Dist, clean_sdist_members, clean_wheel_members, probe


@pytest.fixture
def dist(tmp_path: Path) -> Dist:
    return Dist(tmp_path / "dist", clean_wheel_members(), clean_sdist_members())


def run_check(folder: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = cwc.main([str(folder)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def write_zip_entries(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns on a duplicate name; the duplicate is the point here.
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries:
                archive.writestr(name, data)


def write_tar_entries(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))


# --- L-7: the sdist allowlist ----------------------------------------------------------------------------------


def test_an_sdist_with_a_clean_gitignore_passes(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    dist.sdist_members[f"{SDIST_ROOT}/.gitignore"] = b"# Build output\ndist/\n"
    code, out = run_check(dist.write(), capsys)
    assert code == 0, out


@pytest.mark.parametrize(
    "member",
    [
        f"{SDIST_ROOT}/notes.txt",
        f"{SDIST_ROOT}/setup.py",
        f"{SDIST_ROOT}/.env.local",
        f"{SDIST_ROOT}/requirements-dev.txt",
        f"{SDIST_ROOT}/src/devupai/data.json",
        f"{SDIST_ROOT}/src/devupai/types/models.pyi",
        f"{SDIST_ROOT}/src/devupai/__pycache__/x.py",
        f"{SDIST_ROOT}/src/other.py",
        f"{SDIST_ROOT}/src/devupai_extra/x.py",
        f"{SDIST_ROOT}/docs/README.md",
        "other-1.0.0/README.md",
        "README.md",
    ],
)
def test_a_stray_sdist_member_fails(member: str, dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    dist.sdist_members[member] = b"x = 1\n"
    code, out = run_check(dist.write(), capsys)
    assert code == 1, out
    assert member in out


@pytest.mark.parametrize("name", ["PKG-INFO", "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"])
def test_each_allowed_root_file_is_accepted_alone(name: str, dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    for other in ["pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"]:
        if other != name:
            dist.sdist_members.pop(f"{SDIST_ROOT}/{other}", None)
    code, out = run_check(dist.write(), capsys)
    assert code == 0, out


# --- I-4: duplicate member names -------------------------------------------------------------------------------


def test_a_duplicate_wheel_member_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    folder = dist.write()
    entries = list(clean_wheel_members().items()) + [("devupai/types/models.py", b"x = 2\n")]
    write_zip_entries(dist.wheel, entries)
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert "devupai/types/models.py" in out
    assert "duplicate" in out.lower()


def test_the_word_test_reads_an_earlier_copy_of_a_duplicate_wheel_member(
    dist: Dist, capsys: pytest.CaptureFixture[str]
) -> None:
    _text, word = probe()
    folder = dist.write()
    entries = [("devupai/__init__.py", f'value = "{word}"\n'.encode()), *clean_wheel_members().items()]
    write_zip_entries(dist.wheel, entries)
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert "devupai/__init__.py:1" in out
    assert word in out


def test_a_duplicate_sdist_member_fails(dist: Dist, capsys: pytest.CaptureFixture[str]) -> None:
    _text, word = probe()
    folder = dist.write()
    member = f"{SDIST_ROOT}/README.md"
    entries = [(member, f"The {word} text.\n".encode()), *clean_sdist_members().items()]
    write_tar_entries(dist.sdist, entries)
    code, out = run_check(folder, capsys)
    assert code == 1, out
    assert "duplicate" in out.lower()
    assert f"{member}:1" in out
    assert word in out
