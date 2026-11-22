"""The vendored spec on checkout: its sha256 equals ``openapi/SPEC_SHA256``, and Git never converts its line ends.

Owner request: a test that the vendored spec sha256 equals ``SPEC_SHA256`` on checkout. Bytes are compared, never
decoded text, so a line-end conversion on checkout is a failure.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "openapi" / "devupai-v1.yaml"
HASH_FILE = ROOT / "openapi" / "SPEC_SHA256"
SCRIPT = ROOT / "scripts" / "sync_spec.py"
GITATTRIBUTES = ROOT / ".gitattributes"

EXPECTED_SHA256 = "c1aa27c94578e72bcdaab37d6a0b65cb4dd01ab9e49c64a680b641a982198d97"
EXPECTED_SIZE = 300390


def _attribute_lines() -> list[str]:
    text = GITATTRIBUTES.read_bytes().decode("utf-8")
    return [line.removesuffix("\r") for line in text.split("\n")]


def test_spec_sha256_file_has_the_exact_format() -> None:
    recorded = HASH_FILE.read_bytes()
    assert re.fullmatch(rb"[0-9a-f]{64}\n", recorded), recorded
    assert recorded == f"{EXPECTED_SHA256}\n".encode("ascii")


def test_the_vendored_spec_bytes_match_spec_sha256() -> None:
    data = SPEC.read_bytes()
    recorded = HASH_FILE.read_bytes().decode("ascii").removesuffix("\n")
    assert hashlib.sha256(data).hexdigest() == recorded == EXPECTED_SHA256


def test_the_vendored_spec_has_the_released_size() -> None:
    assert SPEC.stat().st_size == EXPECTED_SIZE
    assert len(SPEC.read_bytes()) == EXPECTED_SIZE


def test_the_vendored_spec_has_no_line_end_conversion() -> None:
    # The released document uses LF only; a CR anywhere means Git (or an editor) converted it.
    data = SPEC.read_bytes()
    assert b"\r" not in data
    assert not data.startswith(b"\xef\xbb\xbf")


def test_gitattributes_starts_with_lf_normalization() -> None:
    lines = [line for line in _attribute_lines() if line.strip() != ""]
    assert lines, ".gitattributes is empty"
    assert lines[0] == "* text=auto eol=lf"


def test_gitattributes_never_converts_the_vendored_files() -> None:
    lines = _attribute_lines()
    assert "openapi/devupai-v1.yaml -text" in lines
    assert "openapi/SPEC_SHA256 -text" in lines


def test_sync_spec_check_passes_on_the_repository() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr


def test_sync_spec_check_passes_from_another_directory(tmp_path: Path) -> None:
    # The script resolves its paths from its own location, not from the current directory.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
