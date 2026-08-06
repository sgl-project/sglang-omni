from __future__ import annotations

import argparse
import importlib.metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def project_config(pyproject: Path) -> dict:
    return tomllib.loads(pyproject.read_text())


def unsatisfied_requirements(dependencies: list[str]) -> list[str]:
    missing = []
    for spec in dependencies:
        requirement = Requirement(spec)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)
            continue
        if installed not in requirement.specifier:
            missing.append(spec)
    return missing


def missing_requirements(pyproject: Path) -> list[str]:
    return unsatisfied_requirements(
        project_config(pyproject)["project"]["dependencies"]
    )


def override_requirements(pyproject: Path) -> list[str]:
    config = project_config(pyproject)
    return config.get("tool", {}).get("uv", {}).get("override-dependencies", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyproject", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--overrides", action="store_true")
    args = parser.parse_args()
    if args.overrides:
        print("\n".join(override_requirements(args.pyproject)))
        return 0
    missing = missing_requirements(args.pyproject)
    if args.check:
        missing += unsatisfied_requirements(override_requirements(args.pyproject))
    if args.check and missing:
        print("Unsatisfied project dependencies:")
        print("\n".join(f"  {item}" for item in missing))
        return 1
    print("\n".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
