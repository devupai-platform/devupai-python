"""The release workflow follows the Trusted Publishing rules, and every action in both workflows is pinned.

``release.yml`` runs only on a pushed ``v*`` tag. The ``build`` job builds and checks the two archives without any
publishing right; the ``publish`` job alone has ``id-token: write``, runs in the ``pypi`` environment, and uploads
with no stored secret. Every ``uses:`` names a full commit SHA with its tag in a comment, the pins of contract §8.

The Python matrix of ``ci.yml`` is covered by ``test_python_floor.py``; this file checks only that it is unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
CI = WORKFLOWS / "ci.yml"

#: Contract §8: ``owner/repo`` -> (full commit SHA, tag).
PINS = {
    "actions/checkout": ("3d3c42e5aac5ba805825da76410c181273ba90b1", "v7.0.1"),
    "actions/setup-python": ("6d4fd22b20465ccc0f7354476588e8df7c9ca0e0", "v7.1.0"),
    "actions/upload-artifact": ("cf430e030ddbb5b0abf93d22962f4752f3646cd9", "v7.0.2"),
    "actions/download-artifact": ("9000827ccba6bdab643e8b6fd33ac0654aef8333", "v8.0.2"),
    "pypa/gh-action-pypi-publish": ("7245906fb61fa33a13554f9d5fb856fe61758086", "v1.14.3"),
}

#: The Q3 matrix of the ``test`` job: (os, python-version).
CI_MATRIX = [
    ("ubuntu-22.04", "3.11.0"),
    ("ubuntu-latest", "3.13"),
    ("ubuntu-latest", "3.15"),
    ("windows-latest", "3.15"),
    ("macos-latest", "3.15"),
]

_USES_LINE = re.compile(r"^\s*(?:-\s+)?uses\s*:")
_PINNED = re.compile(
    r"^\s*(?:-\s+)?uses:\s+(?P<ref>(?P<action>[\w.-]+/[\w.-]+)@(?P<sha>[0-9a-f]{40}))"
    r"\s+#\s+(?P<tag>v\d+\.\d+\.\d+)\s*$"
)
_REF = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")
_SECRET_KEYS = {"password", "user", "api-token"}


def _load(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"{path.relative_to(ROOT).as_posix()} does not exist"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} is not a mapping"
    return data


def _triggers(data: dict[str, Any]) -> Any:
    # YAML 1.1 reads the bare key ``on`` as the boolean True.
    assert ("on" in data) != (True in data), "the workflow needs exactly one 'on' key"
    return data["on"] if "on" in data else data[True]


def _jobs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = data.get("jobs")
    assert isinstance(jobs, dict) and jobs, "the workflow has no jobs"
    assert all(isinstance(job, dict) for job in jobs.values())
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list) and steps, "the job has no steps"
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _release_job(name: str) -> dict[str, Any]:
    job = _jobs(_load(RELEASE)).get(name)
    assert isinstance(job, dict), f"release.yml has no {name!r} job"
    return job


def _action(step: dict[str, Any]) -> str:
    return str(step.get("uses", "")).split("@", 1)[0]


def _index(steps: list[dict[str, Any]], pattern: str) -> int:
    """The index of the single step whose ``run`` matches ``pattern``."""
    found = [i for i, step in enumerate(steps) if re.search(pattern, str(step.get("run", "")))]
    assert len(found) == 1, f"expected one step running {pattern!r}, found {len(found)}"
    return found[0]


def _uses_index(steps: list[dict[str, Any]], action: str) -> int:
    found = [i for i, step in enumerate(steps) if _action(step) == action]
    assert len(found) == 1, f"expected one step using {action}, found {len(found)}"
    return found[0]


# --- release.yml: trigger and permissions ---------------------------------------------------------------------


def test_release_runs_only_on_pushed_v_tags() -> None:
    assert _triggers(_load(RELEASE)) == {"push": {"tags": ["v*"]}}


def test_release_reads_the_repository_at_workflow_level() -> None:
    assert _load(RELEASE).get("permissions") == {"contents": "read"}


def test_release_has_exactly_the_build_and_publish_jobs() -> None:
    assert set(_jobs(_load(RELEASE))) == {"build", "publish"}


def test_only_the_publish_job_has_id_token() -> None:
    data = _load(RELEASE)
    assert "id-token" not in json.dumps(data.get("permissions")), "id-token must not be set at workflow level"
    assert "id-token" not in json.dumps(_release_job("build")), "the build job must not have id-token"


# --- release.yml: the publish job -----------------------------------------------------------------------------


def test_publish_needs_build() -> None:
    needs = _release_job("publish").get("needs")
    assert needs in ("build", ["build"]), needs


def test_publish_runs_in_the_pypi_environment() -> None:
    job = _release_job("publish")
    assert job.get("environment") == {"name": "pypi", "url": "https://pypi.org/project/devupai/"}
    assert job.get("runs-on") == "ubuntu-latest"


def test_publish_permissions_are_exactly_id_token_write() -> None:
    assert _release_job("publish").get("permissions") == {"id-token": "write"}


def test_publish_downloads_dist_then_uploads_with_the_publish_action() -> None:
    steps = _steps(_release_job("publish"))
    assert [_action(step) for step in steps] == ["actions/download-artifact", "pypa/gh-action-pypi-publish"]
    assert not any("run" in step for step in steps), "the publish job runs no shell step"
    download = steps[0].get("with", {})
    assert download.get("name") == "dist"
    assert str(download.get("path", "")).rstrip("/") == "dist"


# --- release.yml: the build job -------------------------------------------------------------------------------


def test_build_runs_on_ubuntu_with_python_3_12() -> None:
    job = _release_job("build")
    assert job.get("runs-on") == "ubuntu-latest"
    setup = _steps(job)[_uses_index(_steps(job), "actions/setup-python")]
    assert str(setup.get("with", {}).get("python-version")) == "3.12"


def test_build_checks_out_without_persisting_credentials() -> None:
    steps = _steps(_release_job("build"))
    checkout = steps[_uses_index(steps, "actions/checkout")]
    assert checkout.get("with", {}).get("persist-credentials") is False


def test_build_checks_the_tag_builds_checks_and_uploads_in_order() -> None:
    steps = _steps(_release_job("build"))
    install = _index(steps, r"pip install -r requirements-dev\.txt")
    tag = _index(steps, r"python scripts/check_tag_version\.py \"\$\{GITHUB_REF_NAME\}\"")
    build = _index(steps, r"-m build --no-isolation")
    twine = _index(steps, r"twine check --strict dist/\*")
    wheel = _index(steps, r"python scripts/check_wheel_contents\.py dist(\s|$)")
    upload = _uses_index(steps, "actions/upload-artifact")
    assert install < tag < build < twine < upload
    assert build < wheel < upload
    artifact = steps[upload].get("with", {})
    assert artifact.get("name") == "dist"
    assert str(artifact.get("path", "")).rstrip("/") == "dist"


# --- both files: no stored secret, every action pinned --------------------------------------------------------


@pytest.mark.parametrize("path", [RELEASE, CI], ids=["release", "ci"])
def test_no_step_takes_a_password_token_or_secret(path: Path) -> None:
    for name, job in _jobs(_load(path)).items():
        for step in _steps(job):
            given = step.get("with") or {}
            assert isinstance(given, dict)
            assert not _SECRET_KEYS & set(given), f"{name}: {step}"
            assert "secrets." not in json.dumps(step), f"{name}: {step}"
    assert "secrets." not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [RELEASE, CI], ids=["release", "ci"])
def test_every_uses_line_is_a_contract_pin(path: Path) -> None:
    assert path.is_file(), f"{path.name} does not exist"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if _USES_LINE.match(line)]
    assert lines, f"{path.name} has no uses: line"
    for line in lines:
        match = _PINNED.match(line)
        assert match, f"not pinned to a full SHA with a '# vX.Y.Z' comment: {line.strip()}"
        assert _REF.match(match.group("ref")), line
        assert match.group("action") in PINS, f"no contract pin for {match.group('action')}"
        assert (match.group("sha"), match.group("tag")) == PINS[match.group("action")], line.strip()


@pytest.mark.parametrize("path", [RELEASE, CI], ids=["release", "ci"])
def test_every_parsed_uses_is_on_a_checked_line(path: Path) -> None:
    # A ``uses`` hidden in a flow mapping would escape the line check above.
    parsed = [str(step["uses"]) for job in _jobs(_load(path)).values() for step in _steps(job) if "uses" in step]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if _USES_LINE.match(line)]
    assert len(parsed) == len(lines)
    assert all(_REF.match(ref) for ref in parsed), parsed


def test_release_uses_every_pinned_action() -> None:
    # Only ``uses`` steps name an action (a ``run`` step has none); the release must use exactly the pinned set.
    used = {_action(step) for job in _jobs(_load(RELEASE)).values() for step in _steps(job) if "uses" in step}
    assert used == set(PINS)


# --- ci.yml ---------------------------------------------------------------------------------------------------


def _ci_test_steps() -> list[dict[str, Any]]:
    job = _jobs(_load(CI)).get("test")
    assert isinstance(job, dict), "ci.yml has no 'test' job"
    return _steps(job)


def test_ci_runs_the_wheel_check_after_the_build() -> None:
    steps = _ci_test_steps()
    build = _index(steps, r"-m build --no-isolation")
    twine = _index(steps, r"twine check --strict dist/\*")
    wheel = _index(steps, r"python scripts/check_wheel_contents\.py dist(\s|$)")
    assert build < twine < wheel


def test_ci_no_longer_lists_the_archives_in_bash() -> None:
    text = "\n".join(str(step.get("run", "")) for step in _ci_test_steps())
    assert "zipfile -l" not in text
    assert "tar -tzf" not in text


def test_ci_keeps_the_q3_matrix() -> None:
    job = _jobs(_load(CI)).get("test")
    assert isinstance(job, dict)
    include = job.get("strategy", {}).get("matrix", {}).get("include")
    assert isinstance(include, list), "the test job has no strategy.matrix.include list"
    assert [(entry.get("os"), entry.get("python-version")) for entry in include] == CI_MATRIX
