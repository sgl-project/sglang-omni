# SPDX-License-Identifier: Apache-2.0
"""Documentation preview launcher regression tests."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("clean_exit_code", [0, 23])
def test_docs_preview_from_another_directory(
    tmp_path: Path, clean_exit_code: int
) -> None:
    """Preview finds its docs directory and stops when cleaning fails."""
    docs_directory = tmp_path / "checkout with spaces" / "docs"
    docs_directory.mkdir(parents=True)
    launcher = docs_directory / "serve.sh"
    shutil.copyfile(Path(__file__).resolve().parents[3] / "docs/serve.sh", launcher)
    trace_file = tmp_path / "targets.log"
    (docs_directory / "Makefile").write_text(
        "clean:\n"
        '\t@printf "clean:%s\\n" "$(CURDIR)" > "$(TRACE_FILE)"\n'
        "\t@exit $(CLEAN_EXIT_CODE)\n"
        "serve:\n"
        '\t@printf "serve:%s:%s\\n" "$(CURDIR)" "$(PORT)" >> "$(TRACE_FILE)"\n'
    )
    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path if clean_exit_code == 0 else docs_directory,
        env={
            **os.environ,
            "TRACE_FILE": str(trace_file),
            "CLEAN_EXIT_CODE": str(clean_exit_code),
            "PORT": "8081",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    expected_targets = [f"clean:{docs_directory}"]
    if clean_exit_code == 0:
        expected_targets.append(f"serve:{docs_directory}:8081")
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
    assert trace_file.read_text().splitlines() == expected_targets
