"""``scripts/sync_spec.py``: dry run by default, ``--write`` copies byte for byte, ``--check`` verifies the hash file.

Twin of the Node SDK's ``tests/sync-spec.test.ts``. The script resolves its paths from its own location, so each test
runs a copy of it inside a temporary layout and never touches the repository's ``openapi/``:

    <tmp>/platform/openapi/devupai-v1.yaml     the default source, as in the platform checkout
    <tmp>/platform/sdk/scripts/sync_spec.py    the copy under test
    <tmp>/platform/sdk/openapi/                the target folder
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT_SOURCE = Path(__file__).resolve().parent.parent / "scripts" / "sync_spec.py"

#: Small known bytes: a CRLF and a non-ASCII character must survive the copy unchanged.
SOURCE_BYTES = b"openapi: 3.1.0\r\ninfo:\n  title: fixture spec \xc3\xa9\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Layout:
    platform: Path
    sdk: Path
    source: Path
    script: Path
    target: Path
    hash_file: Path

    def run(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.script), *args],
            cwd=cwd or self.sdk,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    platform = tmp_path / "platform"
    sdk = platform / "sdk"
    (platform / "openapi").mkdir(parents=True)
    (sdk / "scripts").mkdir(parents=True)
    (sdk / "openapi").mkdir(parents=True)
    source = platform / "openapi" / "devupai-v1.yaml"
    source.write_bytes(SOURCE_BYTES)
    script = sdk / "scripts" / "sync_spec.py"
    shutil.copyfile(SCRIPT_SOURCE, script)
    return Layout(
        platform=platform,
        sdk=sdk,
        source=source,
        script=script,
        target=sdk / "openapi" / "devupai-v1.yaml",
        hash_file=sdk / "openapi" / "SPEC_SHA256",
    )


def test_the_default_is_a_dry_run_that_names_the_hash(layout: Layout) -> None:
    result = layout.run()
    assert result.returncode == 0, result.stderr
    assert sha256(SOURCE_BYTES) in result.stdout
    assert not layout.target.exists()
    assert not layout.hash_file.exists()

    # An existing, different target is not changed either.
    old = b"old\n"
    old_hash = f"{sha256(old)}\n".encode("ascii")
    layout.target.write_bytes(old)
    layout.hash_file.write_bytes(old_hash)
    result = layout.run()
    assert result.returncode == 0, result.stderr
    assert sha256(SOURCE_BYTES) in result.stdout
    assert layout.target.read_bytes() == old
    assert layout.hash_file.read_bytes() == old_hash


def test_write_copies_byte_for_byte_and_writes_the_hash_file(layout: Layout) -> None:
    result = layout.run("--write")
    assert result.returncode == 0, result.stderr
    assert layout.target.read_bytes() == SOURCE_BYTES
    assert layout.hash_file.read_bytes() == f"{sha256(SOURCE_BYTES)}\n".encode("ascii")
    assert sha256(SOURCE_BYTES) in result.stdout


def test_from_path_copies_the_given_file(layout: Layout) -> None:
    release = layout.platform / "release" / "devupai-v1.yaml"
    release.parent.mkdir()
    content = b"openapi: 3.1.0\r\ninfo:\n  title: release copy \xc3\xa9\n"
    release.write_bytes(content)
    result = layout.run("--from", str(release), "--write")
    assert result.returncode == 0, result.stderr
    assert layout.target.read_bytes() == content
    assert layout.hash_file.read_bytes() == f"{sha256(content)}\n".encode("ascii")
    assert sha256(content) in result.stdout


def test_from_equals_path_resolves_a_relative_path_from_the_current_directory(layout: Layout) -> None:
    content = b"openapi: 3.1.0\n# relative\n"
    (layout.platform / "relative.yaml").write_bytes(content)
    result = layout.run("--from=relative.yaml", "--write", cwd=layout.platform)
    assert result.returncode == 0, result.stderr
    assert layout.target.read_bytes() == content
    assert layout.hash_file.read_bytes() == f"{sha256(content)}\n".encode("ascii")


@pytest.mark.parametrize("mode", [(), ("--write",)], ids=["dry-run", "write"])
def test_a_missing_source_exits_1_and_writes_nothing(layout: Layout, mode: tuple[str, ...]) -> None:
    layout.source.unlink()
    result = layout.run(*mode)
    assert result.returncode == 1
    assert result.stderr.strip() != ""
    assert not layout.target.exists()
    assert not layout.hash_file.exists()

    result = layout.run("--from", str(layout.platform / "missing.yaml"), *mode)
    assert result.returncode == 1
    assert result.stderr.strip() != ""
    assert not layout.target.exists()
    assert not layout.hash_file.exists()


def test_a_source_that_is_a_directory_exits_1(layout: Layout) -> None:
    result = layout.run("--from", str(layout.platform), "--write")
    assert result.returncode == 1
    assert result.stderr.strip() != ""
    assert not layout.target.exists()
    assert not layout.hash_file.exists()


def test_check_passes_after_write_without_a_source(layout: Layout) -> None:
    assert layout.run("--write").returncode == 0
    layout.source.unlink()
    result = layout.run("--check")
    assert result.returncode == 0, result.stderr


def test_check_after_tampering_the_target_exits_1_and_names_write(layout: Layout) -> None:
    assert layout.run("--write").returncode == 0
    layout.target.write_bytes(SOURCE_BYTES.replace(b"\r\n", b"\n"))
    result = layout.run("--check")
    assert result.returncode == 1
    assert "--write" in result.stderr


@pytest.mark.parametrize("missing", ["hash_file", "target"])
def test_check_with_a_missing_file_exits_1(layout: Layout, missing: str) -> None:
    assert layout.run("--write").returncode == 0
    getattr(layout, missing).unlink()
    result = layout.run("--check")
    assert result.returncode == 1
    assert result.stderr.strip() != ""


@pytest.mark.parametrize("args", [("--to", "x"), ("--frobnicate",), ("extra",)], ids=lambda a: " ".join(a))
def test_an_unknown_argument_exits_2_with_usage(layout: Layout, args: tuple[str, ...]) -> None:
    result = layout.run(*args)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()
    assert not layout.target.exists()
    assert not layout.hash_file.exists()


def test_write_together_with_check_exits_2(layout: Layout) -> None:
    result = layout.run("--write", "--check")
    assert result.returncode == 2
    assert result.stderr.strip() != ""
    assert not layout.target.exists()
    assert not layout.hash_file.exists()
