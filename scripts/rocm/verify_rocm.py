#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-free ROCm environment and primitive correctness probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

EXPECTED_SGLANG = "0.5.12.post1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-install", action="store_true")
    args = parser.parse_args()

    import torch

    errors: list[str] = []
    hip_version = getattr(torch.version, "hip", None)
    if not hip_version:
        errors.append("PyTorch is not a ROCm build (torch.version.hip is empty)")
    if not torch.cuda.is_available():
        errors.append("No AMD GPU is visible through torch.cuda")

    try:
        sglang_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        sglang_version = None
        errors.append("sglang is not installed")
    if sglang_version and sglang_version != EXPECTED_SGLANG:
        errors.append(
            f"sglang {sglang_version} is installed; expected {EXPECTED_SGLANG}"
        )

    print(f"torch={torch.__version__} hip={hip_version or 'unavailable'}")
    print(f"sglang={sglang_version or 'unavailable'}")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"accelerators={count}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        lhs = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        rhs = torch.eye(4, dtype=torch.float32, device=device)
        actual = lhs @ rhs
        torch.cuda.synchronize(device)
        if not torch.equal(actual.cpu(), lhs.cpu()):
            errors.append("GPU matrix-multiply correctness probe failed")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not args.pre_install:
        import sglang_omni  # noqa: F401

    print("ROCm environment probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
