"""The package admits every Python 3.11 release, and CI runs that floor, so the floor is tested and not only declared.

Twin of the Node SDK's ``tests/node-floor.test.ts``. It also checks the job that installs the lowest allowed runtime
dependency versions, so the lower bounds in ``pyproject.toml`` are tested too.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"

FLOOR = "3.11.0"
#: The newest CPython in the CI matrix (decision Q3: measured at build time).
LATEST = "3.15"
CLASSIFIER_MINORS = range(11, 16)


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(CI.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml is not a mapping"
    return data


def _job(name: str) -> dict[str, Any]:
    jobs = _workflow().get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs"
    job = jobs.get(name)
    assert isinstance(job, dict), f"ci.yml has no {name!r} job"
    return job


def _test_matrix() -> list[dict[str, Any]]:
    include = _job("test").get("strategy", {}).get("matrix", {}).get("include")
    assert isinstance(include, list) and include, "the test job has no strategy.matrix.include list"
    assert all(isinstance(entry, dict) for entry in include)
    return include


def _entries(version: str) -> list[dict[str, Any]]:
    return [entry for entry in _test_matrix() if str(entry.get("python-version")) == version]


def _runtime_requirements() -> list[Requirement]:
    deps = _pyproject()["project"]["dependencies"]
    assert isinstance(deps, list) and deps
    return [Requirement(dep) for dep in deps]


def _lower_bound(req: Requirement) -> str:
    bounds = [spec.version for spec in req.specifier if spec.operator == ">="]
    assert len(bounds) == 1, f"{req.name} needs exactly one >= lower bound, found {bounds}"
    release = list(Version(bounds[0]).release)
    while len(release) < 3:
        release.append(0)
    return ".".join(str(part) for part in release)


def _run_text(job: dict[str, Any]) -> str:
    steps = job.get("steps")
    assert isinstance(steps, list) and steps, "the job has no steps"
    return "\n".join(str(step["run"]) for step in steps if isinstance(step, dict) and "run" in step)


def _setup_python_versions(job: dict[str, Any]) -> list[str]:
    versions: list[str] = []
    for step in job.get("steps", []):
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/setup-python@"):
            versions.append(str(step.get("with", {}).get("python-version")))
    return versions


def test_requires_python_is_the_floor() -> None:
    assert _pyproject()["project"]["requires-python"] == ">=3.11"


def test_the_matrix_python_versions_are_quoted_strings() -> None:
    # An unquoted 3.10 would be read as the number 3.1.
    for entry in _test_matrix():
        assert isinstance(entry.get("python-version"), str), entry


def test_the_ci_matrix_runs_the_floor_release_next_to_the_current_lines() -> None:
    versions = {str(entry.get("python-version")) for entry in _test_matrix()}
    assert {FLOOR, "3.13", LATEST} <= versions


def test_the_floor_release_runs_on_an_ubuntu_image() -> None:
    entries = _entries(FLOOR)
    assert entries, f"no matrix entry for {FLOOR}"
    assert any(str(entry.get("os", "")).startswith("ubuntu-") for entry in entries), entries


@pytest.mark.parametrize("image", ["ubuntu-", "windows-latest", "macos-latest"])
def test_the_latest_release_runs_on_every_platform(image: str) -> None:
    images = [str(entry.get("os", "")) for entry in _entries(LATEST)]
    if image.endswith("-"):
        assert any(name.startswith(image) for name in images), images
    else:
        assert image in images


def test_the_test_job_runs_on_the_matrix_image() -> None:
    assert str(_job("test").get("runs-on")).replace(" ", "") == "${{matrix.os}}"


def test_the_classifiers_name_every_supported_minor_and_nothing_below_the_floor() -> None:
    classifiers = _pyproject()["project"]["classifiers"]
    for minor in CLASSIFIER_MINORS:
        assert f"Programming Language :: Python :: 3.{minor}" in classifiers
    for classifier in classifiers:
        match = re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", classifier)
        if match:
            assert int(match.group(1)) >= 11, classifier
        assert not re.fullmatch(r"Programming Language :: Python :: 2(\..*)?", classifier), classifier


def test_the_runtime_dependencies_have_the_decided_lower_bounds() -> None:
    bounds = {req.name.lower(): _lower_bound(req) for req in _runtime_requirements()}
    assert bounds.get("httpx") == "0.28.1"
    assert bounds.get("pydantic") == "2.10.0"


def test_the_lowest_dependencies_job_installs_exactly_the_lower_bounds() -> None:
    job = _job("lowest-dependencies")
    text = _run_text(job)
    pins = {req.name.lower(): f"{req.name.lower()}=={_lower_bound(req)}" for req in _runtime_requirements()}
    assert pins.get("httpx") == "httpx==0.28.1"
    assert pins.get("pydantic") == "pydantic==2.10.0"
    lowered = text.lower()
    for name, pin in pins.items():
        assert pin in lowered, f"the lowest-dependencies job does not install {pin}"
        found = re.findall(rf"(?<![a-z0-9_.-]){re.escape(name)}==([0-9][0-9a-z.]*)", lowered)
        assert found and all(version == pin.split("==", 1)[1] for version in found), (name, found)


def test_the_lowest_dependencies_job_runs_on_the_floor_release() -> None:
    assert FLOOR in _setup_python_versions(_job("lowest-dependencies"))


def test_the_workflow_only_reads_the_repository() -> None:
    assert _workflow().get("permissions") == {"contents": "read"}
