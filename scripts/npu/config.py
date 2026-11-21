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


def read_config(root: Path = ROOT) -> tuple[dict[str, str], list[dict[str, str]]]:
    manifest = tomllib.loads((root / "pyproject_npu.toml").read_text())
    config = manifest["tool"]["sglang-omni"]["npu"]
    version = config["sglang-version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("sglang-version must be a numeric X.Y.Z release")
    matrix = []
    for device in ("a3", "950"):
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
        else:
            matrix.append({"device": device, "cann": match[1], "base_image": image})
    cann_version = matrix[0]["cann"]
    if cann_version != matrix[1]["cann"]:
        raise ValueError("A3 and 950 must use the same CANN release")
    else:
        matrix.append(
            {
                "device": "910b",
                "cann": cann_version,
                "base_image": "sglang-a2-base",
                "sglang_tag": f"v{version}",
            }
        )
        return config, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--get",
        choices=[
            "sglang-version",
            "sglang-dockerfile-sha256",
            "cann-image-910b",
            "modelscope-version",
            "evalscope-version",
            "base-image-a3",
            "base-image-950",
        ],
    )
    modes.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    config, matrix = read_config()
    if args.get:
        config_value = config[args.get]
        cann_version = matrix[0]["cann"]
        if args.get == "sglang-dockerfile-sha256" and not re.fullmatch(
            r"[0-9a-f]{64}", config_value
        ):
            raise ValueError("sglang-dockerfile-sha256 must be a SHA-256 digest")
        elif args.get == "cann-image-910b" and not re.fullmatch(
            rf"quay\.io/ascend/cann:{re.escape(cann_version)}-910b-ubuntu22\.04-py3\.12"
            r"@sha256:[0-9a-f]{64}",
            config_value,
        ):
            raise ValueError(
                f"cann-image-910b must pin the CANN {cann_version} 910B image"
            )
        elif args.get in (
            "modelscope-version",
            "evalscope-version",
        ) and not re.fullmatch(r"\d+\.\d+\.\d+", config_value):
            raise ValueError(f"{args.get} must be a numeric X.Y.Z release")
        else:
            print(config_value)
    else:
        print(json.dumps({"include": matrix}))


if __name__ == "__main__":
    main()
