#!/usr/bin/env python3
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import audit_model_stack as audit


class ModelStackAuditTests(unittest.TestCase):
    def test_current_source_has_no_retired_runtime(self):
        self.assertEqual(audit.source_errors(audit.ROOT), [])

    def test_local_artifact_package_does_not_allow_model_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path in (
                "Voxt.xcodeproj/project.pbxproj",
                "Voxt/Transcription/MLXModelSupport.swift",
                "Voxt/Core/Models/CustomLLMModelSupport.swift",
            ):
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(audit.ROOT / relative_path, target)
            project_path = root / "Voxt.xcodeproj/project.pbxproj"
            project = project_path.read_text()
            self.assertEqual(audit.source_errors(root), [])
            for package_path in ("../mlx-audio-swift", "Packages/OmniXPCContracts",
                                 "Packages/OmniModelArtifacts/../mlx-audio-swift", "Packages/UnreviewedModelRuntime"):
                project_path.write_text(project + "\nisa = XCLocalSwiftPackageReference;\n"
                                        + f"relativePath = {package_path};\n")
                self.assertIn("Local package override in shared release project", audit.source_errors(root))
            project_path.write_text(project + "\nisa = XCLocalSwiftPackageReference;\n")
            self.assertIn("Local package override in shared release project", audit.source_errors(root))

    def test_dependency_name_detection_does_not_confuse_llama_with_onnx(self):
        for name in ["FluidAudio", "NemoTextProcessing", "sherpa-onnx", "onnxruntime",
                     "MLXAudioCore", "MLXAudioSTT", "MLXAudioVAD", "mlx-audio-swift",
                     "OmniXPCContracts", "OmniInference.xpc", "libOmniCore.dylib"]:
            self.assertIsNotNone(audit.FORBIDDEN.search(name))
        for name in ["llama.swift", "MLX", "MLXLLM", "MLXVLM", "OmniVAD", "model-storage/mlx-audio/checkpoint"]:
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

    def test_audio_dependency_reintroduction_is_rejected_without_rejecting_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Package.resolved"
            path.write_text(json.dumps({"pins": [{"identity": "mlx-audio-swift", "state": {
                "revision": "2a6e75d28ae6a399ba7c7aec842384ef7a3142b5"}}]}))
            self.assertTrue(any("Retired resolved dependency: mlx-audio-swift" in error
                                for error in audit.resolved_errors(path)))
        self.assertIsNone(audit.FORBIDDEN.search("MLXLMCommon"))

    def test_missing_required_pins_are_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Package.resolved"
            path.write_text('{"pins": []}')
            self.assertTrue(audit.resolved_errors(path))


if __name__ == "__main__":
    unittest.main()
