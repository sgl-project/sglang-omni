# SPDX-License-Identifier: Apache-2.0
"""Verify == pins in pyproject.toml match versions installed in a venv."""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement


def _exact_pins(
    pyproject_path: Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return exact pins whose PEP 508 markers match this interpreter.

    A project can declare different exact versions for different platforms.
    Parsing the complete requirement (rather than regexing the raw string)
    avoids both treating the marker's semicolon as part of the version and
    accidentally letting an inactive platform overwrite the active pin.
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    marker_environment = dict(default_environment())
    if environment is not None:
        marker_environment.update(environment)

    pins: dict[str, str] = {}
    project_requirements = data.get("project", {}).get("dependencies", [])
    override_requirements = (
        data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    )
    # Keep uv's existing precedence: an active override intentionally wins over
    # the corresponding project dependency.
    for raw_spec in [*project_requirements, *override_requirements]:
        requirement = Requirement(raw_spec.strip())
        if requirement.marker is not None and not requirement.marker.evaluate(
            marker_environment
        ):
            continue
        exact_versions = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator == "==" and not specifier.version.endswith(".*")
        ]
        if len(exact_versions) != 1:
            continue
        pins[requirement.name.lower()] = exact_versions[0]
    return pins


def _installed_version(distribution: str) -> str | None:
    candidates = [
        distribution,
        distribution.lower(),
        distribution.lower().replace("_", "-"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return version(candidate)
        except PackageNotFoundError:
            continue
    return None


def _matches_pin(installed: str, expected: str) -> bool:
    if "+" in expected:
        return installed == expected
    return installed.partition("+")[0] == expected


def main() -> int:
    python = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    repo_root = Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve()
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.is_file():
        print(f"missing {pyproject_path}", file=sys.stderr)
        return 1

    pins = _exact_pins(pyproject_path)
    mismatches: list[str] = []
    missing: list[str] = []

    for distribution, expected in sorted(pins.items()):
        installed = _installed_version(distribution)
        if installed is None:
            missing.append(f"{distribution}=={expected}")
            continue
        if not _matches_pin(installed, expected):
            mismatches.append(
                f"{distribution}: installed={installed} expected={expected}"
            )

    if missing:
        print("Missing exact-pinned distributions:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
    if mismatches:
        print("Installed pin mismatches:", file=sys.stderr)
        for item in mismatches:
            print(f"  {item}", file=sys.stderr)

    if missing or mismatches:
        return 1

    print(f"Verified {len(pins)} exact dependency pins via {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
