#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import audit_model_stack as audit


class ModelStackAuditTests(unittest.TestCase):
    def test_current_source_has_no_retired_runtime(self):
        self.assertEqual(audit.source_errors(audit.ROOT), [])

    def test_dependency_name_detection_does_not_confuse_llama_with_onnx(self):
        for name in ["FluidAudio", "NemoTextProcessing", "sherpa-onnx", "onnxruntime"]:
            self.assertIsNotNone(audit.FORBIDDEN.search(name))
        for name in ["llama.swift", "MLXAudioVAD", "OmniVAD"]:
            self.assertIsNone(audit.FORBIDDEN.search(name))

    def test_resolved_graph_requires_expected_mlx_and_rejects_retired_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Package.resolved"
            path.write_text(json.dumps({"pins": [
                {"identity": "fluidaudio", "location": "https://example.invalid/repo", "state": {"version": "0.15.6"}}
            ]}))
            errors = audit.resolved_errors(path)
            self.assertTrue(any("Retired resolved dependency" in error for error in errors))
            self.assertTrue(any("mlx-swift version" in error for error in errors))

    def test_upgraded_compatibility_set_is_accepted_and_old_runtime_rejected(self):
        expected = {
            "mlx-audio-swift": {"revision": "2a6e75d28ae6a399ba7c7aec842384ef7a3142b5"},
            "mlx-swift": {"version": "0.31.6"},
            "mlx-swift-lm": {"revision": "c6446cf7bfb7cea76408013b614d4b2c530eaa03"},
            "swift-transformers": {"version": "1.3.4"},
            "swift-huggingface": {"version": "0.10.2"},
            "sparkle": {"version": "2.10.0"},
            "grdb.swift": {"version": "7.11.1"},
            "swift-log": {"version": "1.15.1"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Package.resolved"
            pins = [{"identity": name, "state": state} for name, state in expected.items()]
            path.write_text(json.dumps({"pins": pins}))
            self.assertEqual(audit.resolved_errors(path), [])
            expected["mlx-swift"]["version"] = "0.31.4"
            path.write_text(json.dumps({"pins": pins}))
            self.assertTrue(any("mlx-swift version" in error for error in audit.resolved_errors(path)))

    def test_missing_required_pins_are_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Package.resolved"
            path.write_text('{"pins": []}')
            self.assertTrue(audit.resolved_errors(path))


if __name__ == "__main__":
    unittest.main()
