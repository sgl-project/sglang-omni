# SPDX-License-Identifier: Apache-2.0
"""Prepared runtimes activate in new processes without changing the base install."""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sglang_omni.vendor.sglang.runtime_patch import prepare_runtime


def test_patch_installation_and_reuse_preserve_the_base(tmp_path: Path) -> None:
    source_root = tmp_path / "checkout" / "sglang"
    source_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (source_root / "__init__.py").write_text("from .state import VALUE\n")
    before = "VALUE = 1\n"
    after = "VALUE = 2\n"
    (source_root / "state.py").write_text(before)
    resources = tmp_path / "patches"
    resources.mkdir()
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/state.py",
            tofile="b/state.py",
        )
    )
    (resources / "state.patch").write_text(patch)
    (resources / "runtime.json").write_text(
        json.dumps(
            {
                "base_revision": "fixture",
                "patches": ["state.patch"],
                "files": {
                    "state.py": {
                        "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
                        "patched_sha256": hashlib.sha256(after.encode()).hexdigest(),
                    }
                },
            }
        )
    )
    site_packages = tmp_path / "environment" / "site-packages"
    prepared = prepare_runtime(source_root, site_packages, resources)
    assert (source_root / "state.py").read_text() == before
    assert (prepared / "state.py").read_text() == after
    assert prepare_runtime(source_root, site_packages, resources) == prepared
    script = (
        "import site\n"
        f"site.addsitedir({str(site_packages)!r})\n"
        "from pathlib import Path\n"
        "from sglang_omni.vendor.sglang.runtime_patch import verify_active_runtime\n"
        f"verify_active_runtime(Path({str(resources)!r}))\n"
        "import sglang\n"
        f"assert Path(sglang.__file__).parent == Path({str(prepared)!r})\n"
        "assert sglang.VALUE == 2\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    activation = site_packages / "sglang_omni_runtime.pth"
    previous_activation = activation.read_bytes()
    (source_root / "state.py").write_text("VALUE = 3\n")
    with pytest.raises(RuntimeError, match="state.py"):
        prepare_runtime(source_root, site_packages, resources)
    assert activation.read_bytes() == previous_activation
    assert (prepared / "state.py").read_text() == after
