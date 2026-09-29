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
        matrix.append({"device": device, "cann": match[1]})
    if matrix[0]["cann"] != matrix[1]["cann"]:
        raise ValueError("A3 and 910B must use the same CANN release")
    return config, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--get", choices=["sglang-version", "base-image-a3", "base-image-910b"]
    )
    modes.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    config, matrix = read_config()
    if args.get:
        print(config[args.get])
    elif args.matrix:
        print(json.dumps({"include": matrix}))


if __name__ == "__main__":
    main()
