# SPDX-License-Identifier: Apache-2.0
"""Verify == pins in pyproject.toml match versions installed in a venv."""

from __future__ import annotations

import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_EXACT_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^,\s]+)")


def _exact_pins(pyproject_path: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    pins: dict[str, str] = {}
    for spec in data.get("project", {}).get("dependencies", []):
        match = _EXACT_PIN.match(spec.strip())
        if match is not None:
            pins[match.group(1).lower()] = match.group(2)
    for spec in data.get("tool", {}).get("uv", {}).get("override-dependencies", []):
        match = _EXACT_PIN.match(spec.strip())
        if match is not None:
            pins[match.group(1).lower()] = match.group(2)
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
        if installed != expected:
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
