# SPDX-License-Identifier: Apache-2.0
"""Exercise the native draft policy without model weights or GPU timings."""

import subprocess
from pathlib import Path


def test_draft_backoff_windows(tmp_path: Path) -> None:
    native_directory = Path(__file__).resolve().parents[1]
    binary = tmp_path / "draft_backoff_test"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(native_directory / "src"),
            str(native_directory / "tests" / "draft_backoff_test.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
