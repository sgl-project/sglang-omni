# SPDX-License-Identifier: Apache-2.0
"""Run the small Apple backend suites on real Metal hardware."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

SUITES = {
    "mps": [
        "tests/unit_test/platforms/test_apple.py",
        "tests/unit_test/model_runner/test_audio_torch_mps.py",
    ],
    "mlx": [
        "tests/unit_test/qwen3_asr/test_mlx_model.py",
        "tests/unit_test/qwen3_asr/test_mlx_scheduler_runner.py",
    ],
}


def check_device(backend: str) -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Apple UT requires an arm64 macOS runner")
    if backend == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("Torch MPS is unavailable")
        result = (torch.ones(4, device="mps") + 1).cpu()
        torch.mps.synchronize()
        assert result.tolist() == [2.0] * 4
    else:
        import mlx.core as mx

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable")
        mx.set_default_device(mx.gpu)
        result = mx.ones(4) + 1
        mx.eval(result)
        assert result.tolist() == [2.0] * 4
    return {
        "backend": backend,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "mlx", "mlx-lm", "sglang", "sglang-omni")
        },
    }


def verify_report(report: Path) -> None:
    cases = ET.parse(report).findall(".//testcase")
    if not cases:
        raise RuntimeError("Apple UT collected no test cases")
    skipped = [
        case.attrib.get("name") for case in cases if case.find("skipped") is not None
    ]
    if skipped:
        raise RuntimeError(
            f"Apple UT must execute every selected case; skipped: {skipped}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=SUITES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report_dir = Path(
        os.environ.get("APPLE_UT_REPORT_DIR", root / "results/apple-ut")
    ).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_USE_MLX"] = "1" if args.backend == "mlx" else "0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    # Tiny randomly initialized models and mocked loaders need no HF downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    details = check_device(args.backend)
    details["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    (report_dir / f"{args.backend}-environment.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )
    print(json.dumps(details, indent=2), flush=True)
    report = report_dir / f"{args.backend}.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *SUITES[args.backend],
            "-v",
            "-ra",
            f"--junitxml={report}",
        ],
        cwd=root,
    )
    if result.returncode:
        return result.returncode
    verify_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
