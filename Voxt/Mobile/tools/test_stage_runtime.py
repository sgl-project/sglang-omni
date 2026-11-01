"""Bundle staging tests use synthetic executables; no signing or process launch."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/stage-runtime.py"


class RuntimeStagingTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "build/connector-tmp/staging-fixtures"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / "Connector.app"
        (self.app / "Contents/MacOS").mkdir(parents=True)
        (self.app / "Contents/Info.plist").write_text("fixture")
        self.runtime = self.root / "runtime"
        self.runtime.write_bytes(b"fixture runtime bytes")
        self.runtime.chmod(0o755)

    def stage(self, runtime=None):
        return subprocess.run(
            ["python3", str(SCRIPT), str(self.app), str(runtime or self.runtime)],
            capture_output=True,
        )

    def test_embeds_executable_without_launching_or_signing(self):
        result = self.stage()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        output = self.app / "Contents/Helpers/voxt-tailcat-host"
        self.assertEqual(output.read_bytes(), self.runtime.read_bytes())
        self.assertTrue(os.access(output, os.X_OK))

    def test_missing_runtime_does_not_modify_bundle(self):
        self.assertNotEqual(self.stage(self.root / "missing").returncode, 0)
        self.assertFalse((self.app / "Contents/Helpers").exists())

    def test_nonexecutable_runtime_is_rejected(self):
        self.runtime.chmod(0o644)
        self.assertNotEqual(self.stage().returncode, 0)
        self.assertFalse((self.app / "Contents/Helpers").exists())

    def test_helper_directory_symlink_cannot_write_outside_bundle(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.app / "Contents/Helpers").symlink_to(outside)
        self.assertNotEqual(self.stage().returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
