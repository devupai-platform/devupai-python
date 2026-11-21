"""Check that a release tag is ``v`` followed by the package version.

The version is read from ``src/devupai/_version.py`` (relative to the SDK root, the parent of this script's folder)
with ``ast``. The file is parsed, never imported.

Usage::

    python scripts/check_tag_version.py <tag>

Exit codes: 0 the tag matches, 1 it does not (or the version cannot be read), 2 a usage error.

Standard library only.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PREFIX = "check_tag_version"
USAGE = "usage: python scripts/check_tag_version.py <tag>"

SDK_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = SDK_ROOT / "src" / "devupai" / "_version.py"


class VersionError(Exception):
    """The version file is missing, or holds no single ``__version__`` string."""


def _is_version_target(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "__version__"


def read_version(path: Path = VERSION_FILE) -> str:
    """The ``__version__`` string assigned at the top level of ``path``, read with ``ast``."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VersionError(f"cannot read {path.name}: {exc.strerror or exc}") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise VersionError(f"cannot parse {path.name}: {exc.msg} (line {exc.lineno})") from exc

    found: list[str] = []
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(_is_version_target(target) for target in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and _is_version_target(node.target):
            value = node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            raise VersionError(f"__version__ in {path.name} is not a string literal")
        found.append(value.value)
    if len(found) != 1:
        raise VersionError(f"{path.name} must assign __version__ exactly once, found {len(found)}")
    return found[0]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0]:
        print(f"{PREFIX}: {USAGE}", file=sys.stderr)
        return 2
    tag = args[0]
    try:
        version = read_version()
    except VersionError as exc:
        print(f"{PREFIX}: {exc}", file=sys.stderr)
        return 1
    expected = f"v{version}"
    if tag != expected:
        print(
            f"{PREFIX}: the tag {tag!r} does not match the package version {version!r}; the tag must be {expected!r}",
            file=sys.stderr,
        )
        return 1
    print(f"{PREFIX}: the tag {tag!r} matches the package version {version!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
