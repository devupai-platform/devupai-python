"""``scripts/check_tag_version.py``: a release tag must be ``v`` + the package version.

The script reads ``__version__`` from ``src/devupai/_version.py`` with ``ast`` and never imports it. Each test runs a
copy of it inside a temporary layout:

    <tmp>/repo/scripts/check_tag_version.py     the copy under test
    <tmp>/repo/src/devupai/_version.py          a fake version file
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_SOURCE = ROOT / "scripts" / "check_tag_version.py"
VERSION_FILE = ROOT / "src" / "devupai" / "_version.py"

#: The fake version file. The trailing ``raise`` proves the script parses the file instead of importing it.
FAKE_VERSION = '"""Version of the devupai package."""\n\n__version__ = "1.0.0a1"\n\nraise RuntimeError("imported")\n'


@dataclass
class Layout:
    root: Path
    script: Path

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return _run(self.script, self.root, *args)


def _run(script: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )


def _names(output: str, value: str) -> bool:
    """Whether ``output`` names ``value`` as a whole version, not as part of a longer one."""
    return re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", output) is not None


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "devupai").mkdir(parents=True)
    (root / "src" / "devupai" / "_version.py").write_text(FAKE_VERSION, encoding="utf-8")
    script = root / "scripts" / "check_tag_version.py"
    shutil.copyfile(SCRIPT_SOURCE, script)
    return Layout(root=root, script=script)


def test_the_matching_tag_exits_0(layout: Layout) -> None:
    result = layout.run("v1.0.0a1")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_different_version_exits_1_and_names_both_values(layout: Layout) -> None:
    result = layout.run("v1.0.0")
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert _names(output, "v1.0.0"), output
    assert _names(output, "1.0.0a1"), output


def test_a_tag_without_the_v_prefix_exits_1(layout: Layout) -> None:
    result = layout.run("1.0.0a1")
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert _names(output, "1.0.0a1"), output


def test_a_missing_argument_exits_2(layout: Layout) -> None:
    result = layout.run()
    assert result.returncode == 2, result.stdout + result.stderr


def test_the_real_repository_tag_matches_the_real_version() -> None:
    match = re.search(r"""^__version__\s*(?::\s*str\s*)?=\s*["']([^"']+)["']""", VERSION_FILE.read_text("utf-8"), re.M)
    assert match, f"{VERSION_FILE} has no __version__ string"
    result = _run(ROOT / "scripts" / "check_tag_version.py", ROOT, f"v{match.group(1)}")
    assert result.returncode == 0, result.stdout + result.stderr
