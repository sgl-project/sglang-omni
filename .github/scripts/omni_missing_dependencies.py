from __future__ import annotations

import argparse
import importlib.metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def missing_requirements(pyproject: Path) -> list[str]:
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyproject", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    missing = missing_requirements(args.pyproject)
    if args.check and missing:
        print("Unsatisfied project dependencies:")
        print("\n".join(f"  {item}" for item in missing))
        return 1
    print("\n".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
