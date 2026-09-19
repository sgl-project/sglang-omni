# SPDX-License-Identifier: Apache-2.0
"""Read the NPU manifest without importing Omni or accelerator dependencies."""

import argparse
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 source installations
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def read_config(root: Path = ROOT) -> tuple[dict, list[dict]]:
    manifest = tomllib.loads((root / "pyproject_npu.toml").read_text())
    config = manifest["tool"]["sglang-omni"]["npu"]
    version = config["sglang-version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("sglang-version must be a numeric X.Y.Z release")
    matrix = []
    for device in ("a3", "910b"):
        image = config[f"base-image-{device}"]
        match = re.fullmatch(
            rf"[^\s@]+:v{re.escape(version)}-cann(\d+\.\d+\.\d+)-{device}"
            r"@sha256:[0-9a-f]{64}",
            image,
        )
        if not match:
            raise ValueError(
                f"base-image-{device} must pin SGLang {version}, {device} and a digest"
            )
        matrix.append({"device": device, "base_image": image, "cann": match[1]})
    if matrix[0]["cann"] != matrix[1]["cann"]:
        raise ValueError("A3 and 910B must use the same CANN release")
    return config, matrix


def check_requirements(root: Path = ROOT) -> None:
    """Check direct runtime requirements against the shared NPU pins."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    manifest = tomllib.loads((root / "pyproject_npu.toml").read_text())
    pins = {}
    for line in (root / "scripts/npu/requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        specs = list(requirement.specifier)
        if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
            raise ValueError(f"NPU requirement must be exactly pinned: {line}")
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise ValueError(f"Duplicate NPU requirement: {name}")
        pins[name] = specs[0].version
    for text in manifest["project"]["dependencies"]:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(
            {
                "sys_platform": "linux",
                "platform_machine": "aarch64",
                "python_version": "3.11",
                "python_full_version": "3.11.0",
            }
        ):
            continue
        pin = pins.get(canonicalize_name(requirement.name))
        if pin is None or not requirement.specifier.contains(pin, prereleases=True):
            raise ValueError(f"NPU pins do not satisfy {text}: {pin}")


def check_installed(config: dict) -> None:
    """Verify the actual base image version, not just its tag spelling."""
    from importlib.metadata import version

    from packaging.version import Version

    expected = Version(config["sglang-version"]).release
    actual = version("sglang")
    if Version(actual).release[: len(expected)] != expected:
        raise ValueError(f"Installed SGLang {actual} does not match {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--get", choices=["sglang-version", "base-image-a3", "base-image-910b"]
    )
    modes.add_argument("--matrix", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--check-installed", action="store_true")
    args = parser.parse_args()
    config, matrix = read_config()
    if args.get:
        print(config[args.get])
    elif args.matrix:
        print(json.dumps({"include": matrix}))
    elif args.check:
        check_requirements()
        print("NPU image configuration and direct dependency pins are consistent")
    else:
        check_installed(config)


if __name__ == "__main__":
    main()
