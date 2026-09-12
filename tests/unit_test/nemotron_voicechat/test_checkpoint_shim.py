# SPDX-License-Identifier: Apache-2.0
"""Filesystem regression tests that do not require SGLang or a GPU."""

import ast
import atexit
import shutil
import tempfile
import unittest
from pathlib import Path


def _load_shim_dir():
    # Execute the actual helper without importing GPU-dependent engine classes.
    source = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni/models/nemotron_voicechat/engine_builder.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_shim_dir"
    )
    namespace = {
        "Path": Path,
        "atexit": atexit,
        "shutil": shutil,
        "tempfile": tempfile,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["_shim_dir"]


class TestCheckpointShim(unittest.TestCase):
    def test_switch_checkpoints_keeps_weights_and_configs_isolated(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shim_dir = _load_shim_dir()
        for name in ("A", "B"):
            source = root / name
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "model.safetensors").write_text(name)
        shims = []
        for name in ("A", "B", "A"):
            shim = shim_dir("voicechat", root / name)
            self.addCleanup(shutil.rmtree, shim, ignore_errors=True)
            self.assertFalse((shim / "config.json").exists())
            (shim / "config.json").write_text("adapted " + name)
            shims.append((shim, name))
        # Check after all loads, so switching to B cannot silently redirect A.
        for shim, name in shims:
            weight = shim / "model.safetensors"
            self.assertTrue(weight.is_symlink())
            self.assertEqual(weight.resolve(), (root / name / weight.name).resolve())
            self.assertEqual(weight.read_text(), name)
            self.assertEqual((shim / "config.json").read_text(), "adapted " + name)


if __name__ == "__main__":
    unittest.main()
