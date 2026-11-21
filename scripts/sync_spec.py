"""Copy the platform's OpenAPI document into this repository, or check the copy.

Source: ``../openapi/devupai-v1.yaml`` (relative to the SDK root; only present in the platform repository checkout),
or the file given with ``--from PATH`` (also ``--from=PATH``), for example a release copy built elsewhere.

Target: ``openapi/devupai-v1.yaml`` (a byte-for-byte copy) and ``openapi/SPEC_SHA256`` (64 lowercase hex characters
followed by ``"\\n"``).

Usage::

    python scripts/sync_spec.py [--from PATH | --from=PATH] [--write | --check]

Modes:

- default (dry run): print the source, its sha256 and whether the target would change; write nothing.
- ``--write``: copy the source byte for byte and write ``SPEC_SHA256``.
- ``--check``: no source needed; exit 0 when the sha256 of the target equals ``SPEC_SHA256``, else exit 1.

Exit codes: 0 success, 1 a missing or mismatching file, 2 a usage error.

Standard library only.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

PREFIX = "sync_spec"
USAGE = "usage: python scripts/sync_spec.py [--from PATH | --from=PATH] [--write | --check]"

SDK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = SDK_ROOT.parent / "openapi" / "devupai-v1.yaml"
TARGET_DIR = SDK_ROOT / "openapi"
TARGET = TARGET_DIR / "devupai-v1.yaml"
HASH_FILE = TARGET_DIR / "SPEC_SHA256"


class UsageError(Exception):
    """A command line that this script does not accept (exit code 2)."""


@dataclass(frozen=True)
class Arguments:
    """The parsed command line."""

    source: str | None
    write: bool
    check: bool


def sha256(data: bytes) -> str:
    """The sha256 of ``data`` as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def parse_args(argv: list[str]) -> Arguments:
    """Parse ``argv`` (without the program name). Raises ``UsageError`` for anything this script does not accept."""
    source: str | None = None
    write = False
    check = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        value: str | None
        if arg == "--write":
            write = True
            index += 1
            continue
        if arg == "--check":
            check = True
            index += 1
            continue
        if arg == "--from":
            value = argv[index + 1] if index + 1 < len(argv) else None
            index += 2
        elif arg.startswith("--from="):
            value = arg[len("--from=") :]
            index += 1
        else:
            raise UsageError(f'unknown argument "{arg}".')
        if value is None or value == "" or value.startswith("--"):
            raise UsageError("--from needs a file path.")
        if source is not None:
            raise UsageError("--from was given more than once.")
        source = value
    if write and check:
        raise UsageError("--write and --check cannot be used together.")
    return Arguments(source=source, write=write, check=check)


def _out(message: str) -> None:
    print(f"{PREFIX}: {message}")


def _err(message: str) -> None:
    print(f"{PREFIX}: {message}", file=sys.stderr)


def _relative(path: Path) -> str:
    try:
        return path.relative_to(SDK_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_check() -> int:
    """Compare the sha256 of the vendored spec with ``SPEC_SHA256``. Returns the exit code."""
    fix = "Run `python scripts/sync_spec.py --write` to copy the spec again."
    if not TARGET.is_file():
        _err(f"{_relative(TARGET)} not found. {fix}")
        return 1
    if not HASH_FILE.is_file():
        _err(f"{_relative(HASH_FILE)} not found. {fix}")
        return 1
    recorded_bytes = HASH_FILE.read_bytes()
    try:
        recorded = recorded_bytes.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError:
        _err(f"{_relative(HASH_FILE)} does not hold a sha256 value. {fix}")
        return 1
    actual = sha256(TARGET.read_bytes())
    if recorded_bytes != f"{actual}\n".encode("ascii"):
        _err(
            f"the sha256 of {_relative(TARGET)} is {actual}, but {_relative(HASH_FILE)} records "
            f"{recorded or '(empty)'}. {fix}"
        )
        return 1
    _out(f"{_relative(TARGET)} matches {_relative(HASH_FILE)} ({actual}).")
    return 0


def run_sync(source_arg: str | None, write: bool) -> int:
    """Dry run (``write=False``) or copy (``write=True``) the source spec. Returns the exit code."""
    source = DEFAULT_SOURCE if source_arg is None else (Path.cwd() / source_arg).resolve()
    if not source.is_file():
        if source_arg is None:
            _err(f"{source} not found. Run this script inside the platform repository checkout, or pass --from PATH.")
        else:
            _err(f"{source} (from --from) is not a file. Pass the path of the spec file.")
        return 1

    data = source.read_bytes()
    source_hash = sha256(data)
    _out(f"source {source}")
    _out(f"source sha256 {source_hash} ({len(data)} bytes)")

    current = TARGET.read_bytes() if TARGET.is_file() else None
    current_hash = HASH_FILE.read_bytes() if HASH_FILE.is_file() else None
    expected_hash = f"{source_hash}\n".encode("ascii")
    changes = current != data or current_hash != expected_hash

    if not write:
        state = "would change" if changes else "is up to date"
        _out(f"{_relative(TARGET)} {state}. Nothing written (dry run); pass --write to copy.")
        return 0

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    copied_hash = sha256(TARGET.read_bytes())
    if copied_hash != source_hash:
        _err(f"the copy at {_relative(TARGET)} does not match the source (sha256 {copied_hash} != {source_hash}).")
        return 1
    HASH_FILE.write_bytes(expected_hash)
    _out(f"{_relative(TARGET)} ({len(data)} bytes)")
    _out(f"{_relative(HASH_FILE)} = {source_hash}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the script with ``argv`` (default: ``sys.argv[1:]``) and return the exit code."""
    args_list = sys.argv[1:] if argv is None else argv
    try:
        args = parse_args(args_list)
    except UsageError as exc:
        _err(str(exc))
        print(USAGE, file=sys.stderr)
        return 2
    if args.check:
        return run_check()
    return run_sync(args.source, args.write)


if __name__ == "__main__":
    raise SystemExit(main())
