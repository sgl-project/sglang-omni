# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint qualification for Cosmos3-Super T2I."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODE = os.getenv("SGLANG_COSMOS3_SUPER_E2E_MODE")


@pytest.mark.accelerator
@pytest.mark.skipif(
    _MODE not in {"direct", "sdk", "http"},
    reason=(
        "set SGLANG_COSMOS3_SUPER_E2E_MODE to direct, sdk, or http; "
        "the HTTP mode expects an already-running server"
    ),
)
def test_full_super_t2i_back_to_back(tmp_path: Path) -> None:
    output_dir = tmp_path / str(_MODE)
    command = [
        sys.executable,
        str(_REPO_ROOT / "examples/cosmos3_super_t2i.py"),
        str(_MODE),
        "--repeats",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    if _MODE == "http":
        command.extend(
            [
                "--server-url",
                os.getenv("SGLANG_COSMOS3_SUPER_SERVER_URL", "http://127.0.0.1:30010"),
            ]
        )
    subprocess.run(command, cwd=_REPO_ROOT, check=True, timeout=14_400)

    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.rglob("*.png.json"))
    ]
    assert len(manifests) == 2
    assert {manifest["mode"] for manifest in manifests} == {_MODE}
    assert {manifest["checkpoint_revision"] for manifest in manifests} == {
        "fe77b66696d645f663b8f27e942b3b43e4629e23"
    }
    assert len({manifest["path"] for manifest in manifests}) == 2
    assert len({manifest["pixel_sha256"] for manifest in manifests}) == 1
