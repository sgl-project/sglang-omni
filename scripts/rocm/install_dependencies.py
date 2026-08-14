# SPDX-License-Identifier: Apache-2.0
"""Install the accelerator-neutral dependencies from the ROCm manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            "Python 3.10 requires tomli to read pyproject_rocm.toml; "
            "install tomli==2.4.0 first"
        ) from exc


PROTECTED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "torchaudio",
    "sglang",
    "amd-aiter",
    "pytorch-triton-rocm",
    "sglang-kernel",
    "torchao",
    "compressed-tensors",
)

FORBIDDEN_DISTRIBUTIONS = {
    "flash-attn",
    "flash-attn-4",
    "flashinfer-python",
    "mooncake-transfer-engine-cuda12",
    "mooncake-transfer-engine-cuda13",
    "nixl-cu12",
    "nixl-cu13",
    "sgl-deep-gemm",
}


def _canonicalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _installed_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _forbidden_installed() -> set[str]:
    installed = {
        _canonicalize(distribution.metadata["Name"])
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        name
        for name in installed
        if name in FORBIDDEN_DISTRIBUTIONS or name.startswith("nvidia-")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pyproject_rocm.toml",
    )
    args = parser.parse_args()

    with args.manifest.open("rb") as manifest_file:
        dependencies = tomllib.load(manifest_file)["project"]["dependencies"]

    if not dependencies:
        raise SystemExit(f"no project dependencies found in {args.manifest}")

    protected_before = _installed_versions(PROTECTED_DISTRIBUTIONS)
    if missing := {"torch", "sglang"} - protected_before.keys():
        raise SystemExit(
            f"ROCm base image is missing protected package(s): {sorted(missing)}"
        )
    if forbidden := _forbidden_installed():
        raise SystemExit(
            f"ROCm base image contains CUDA-only package(s): {sorted(forbidden)}"
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as constraints:
        for name, version in protected_before.items():
            constraints.write(f"{name}==={version}\n")
        constraints.flush()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--constraint",
                constraints.name,
                *dependencies,
            ],
            check=True,
        )

    protected_after = _installed_versions(PROTECTED_DISTRIBUTIONS)
    if protected_after != protected_before:
        raise SystemExit(
            "dependency installation changed the protected ROCm stack: "
            f"before={protected_before}, after={protected_after}"
        )
    if forbidden := _forbidden_installed():
        raise SystemExit(
            f"dependency installation added CUDA-only package(s): {sorted(forbidden)}"
        )


if __name__ == "__main__":
    main()
