"""Check regression selection and packaging with fake tools, not macOS execution."""
import ast
import copy
import datetime
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET

from Voxt.tools.signing_profile_policy import prepare_entitlements

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools/run_local_regression_matrix.sh"


class LocalRegressionMatrixTests(unittest.TestCase):
    def run_matrix(self, group, fail_selector=""):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            log = temp / "commands.jsonl"
            executable = temp / "xcodebuild"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['VOXT_COMMAND_LOG'], 'a') as log:\n"
                "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "sys.exit(1 if os.environ.get('VOXT_FAIL_SELECTOR') in sys.argv[1:] else 0)\n"
            )
            executable.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": str(temp) + os.pathsep + os.environ.get("PATH", ""),
                "CI": "false",
                "GITHUB_ACTIONS": "false",
                "VOXT_RUN_MODEL_TESTS": "0",
                "VOXT_SPM_CACHE_PATH": str(temp / "cache"),
                "VOXT_SPM_CLONE_PATH": str(temp / "packages"),
                "VOXT_COMMAND_LOG": str(log),
                "VOXT_FAIL_SELECTOR": fail_selector,
            }
            result = subprocess.run(
                ["bash", str(SCRIPT), group], cwd=temp, env=environment,
                text=True, capture_output=True, check=False,
            )
            commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, commands

    def test_static_selectors_reference_existing_suites(self):
        suites = {}
        for path in (ROOT / "VoxtTests").rglob("*.swift"):
            text = path.read_text()
            classes = list(re.finditer(r"^(?:final )?class (\w+)", text, re.M))
            for index, match in enumerate(classes):
                end = classes[index + 1].start() if index + 1 < len(classes) else len(text)
                suites[match[1]] = set(re.findall(r"func (test\w+)\(", text[match.end():end]))
        selectors = re.findall(r"-only-testing:VoxtTests/(\w+)(?:/(\w+))?", SCRIPT.read_text())
        self.assertTrue(selectors)
        for suite, method in selectors:
            self.assertIn(suite, suites)
            if method:
                self.assertIn(method, suites[suite], f"{suite}/{method}")
            else:
                pass

    def test_refactor_selects_split_suites_with_portable_unsigned_strict_commands(self):
        result, commands = self.run_matrix("refactor")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 2)
        selectors = {arg for command in commands for arg in command if arg.startswith("-only-testing:")}
        for family in ("RemoteLLMRuntimeClient", "RemoteModelConfiguration", "HotkeyManager", "MLXModelManager", "MeetingDetailViewModel"):
            for path in (ROOT / "VoxtTests").glob(f"{family}*Tests.swift"):
                self.assertIn(f"-only-testing:VoxtTests/{path.stem}", selectors)
        for suite in ("DoubaoPacketCodecTests", "RemoteASRResponseStateTests", "RemoteASRCompletionTests", "MeetingRemoteSessionLifecycleTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("TrackedTaskStoreTests", "LLMRequestLifecycleTests", "MeetingLiveSessionRegistryTests", "MLXCorrectionPassCoordinatorTests", "MeetingCaptureTimelineTests", "MLXNativeLiveRuntimeTests", "HotkeyEventTapRunLoopTests", "HotkeyManagerLifetimeTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("SharedModelLoadCoordinatorTests", "MeetingImportedFileAnalyzerTests", "MeetingFileTaskQueueTests", "MeetingFinalizationContextTests", "MeetingFinalizationCheckpointStoreTests", "RecordingSessionLifecycleTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("AutomaticDictionaryLearningMonitorTests", "DictionaryStoreAsyncTests", "TranscriptionHistoryStoreAsyncTests", "TranscriptionHistoryEntryAudioTests", "MeetingTranscriptVirtualListTests", "MeetingDetailTranscriptListCacheTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("CustomLLMRequestRuntimeTests", "CustomLLMModelDownloadSupportTests", "CustomLLMModelSupportTests", "ModelDownloadSourceSupportTests", "MLXModelSupportTests", "ModelInstallationCacheTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("TextInjectionTransactionTests", "PasteboardTextWriterTests", "RemoteProviderConnectivityTesterTests", "RemoteProviderConfigurationPolicyTests", "DictionarySuggestionStoreTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("DictionarySuggestionReloadTests", "RemoteProviderSheetOperationsTests", "ConnectivityWebSocketSessionTests", "RemoteASRPreviewControllerTests", "ModelDownloadProgressTests", "HotkeyPreferenceCorruptStorageTests", "MLXInferenceConfigurationTests", "MeetingLiveTranscriptPresentationTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for suite in ("ModelConfigurationIssueResolverTests", "VoxtLogRedactorTests"):
            self.assertIn(f"-only-testing:VoxtTests/{suite}", selectors)
        for command in commands:
            self.assertEqual(command[0], "test")
            self.assertIn(str(ROOT / "Voxt.xcodeproj"), command)
            self.assertIn("CODE_SIGNING_ALLOWED=NO", command)
            self.assertIn("-onlyUsePackageVersionsFromResolvedFile", command)

    def test_all_does_not_run_core_vad_suites_twice(self):
        result, commands = self.run_matrix("all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 3)
        selectors = [arg for command in commands for arg in command if arg.startswith("-only-testing:")]
        self.assertEqual(len(selectors), len(set(selectors)))

    def test_removed_groups_fail_instead_of_selecting_missing_tests(self):
        for group in ("whisper", "diagnostic", "unknown"):
            with self.subTest(group=group):
                result, commands = self.run_matrix(group)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(commands, [])

    def test_later_success_does_not_hide_core_failure(self):
        result, commands = self.run_matrix(
            "refactor", "-only-testing:VoxtTests/TranscriptionCapturePipelineTests"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(commands), 2)
        self.assertIn("Group failed: run_core", result.stdout)


class NativeHTTPPackagingTests(unittest.TestCase):
    def test_audio_and_xpc_retirement_keeps_required_products(self) -> None:
        project = (ROOT / "Voxt.xcodeproj/project.pbxproj").read_text()
        self.assertNotRegex(project, r"MLXAudio(?:Core|STT|VAD)|mlx-audio-swift|OmniXPC|OmniInference|OmniCore")
        for product in ("Tokenizers", "MLX", "HuggingFace", "MLXLLM", "MLXVLM", "OmniModelArtifacts"):
            self.assertIn(f"productName = {product};", project)
        self.assertIn('repositoryURL = "https://github.com/huggingface/swift-transformers.git";', project)
        for version in ("1.3.4", "0.31.6", "0.10.2"):
            self.assertIn(f"version = {version};", project)
        for directory in (ROOT / "Voxt", ROOT / "VoxtTests"):
            for source in directory.rglob("*.swift"):
                self.assertNotRegex(source.read_text(), r"import (?:MLXAudio(?:STT|VAD|Core)|OmniXPCContracts)", str(source))
        default_scheme = ET.parse(ROOT / "Voxt.xcodeproj/xcshareddata/xcschemes/Voxt.xcscheme")
        self.assertEqual(default_scheme.find("TestAction").get("buildConfiguration"), "TestDebug")
        self.assertNotIn("OmniInference", ET.tostring(default_scheme.getroot()).decode())
        self.assertFalse((ROOT / "Voxt.xcodeproj/xcshareddata/xcschemes/VoxtOmniXPCSmoke.xcscheme").exists())
        metadata = plistlib.loads((ROOT / "Voxt/Voxt/Info.plist").read_bytes())
        for name in ("OmniAppGroupIdentifier", "OmniInferenceServiceIdentifier", "OmniHostBundleIdentifier", "OmniTeamIdentifier"):
            self.assertNotIn(name, metadata)
        entitlements = plistlib.loads((ROOT / "Voxt/Voxt.entitlements").read_bytes())
        self.assertNotIn("com.apple.security.application-groups", entitlements)

    def test_artifact_validation_is_retained_without_download_inference(self) -> None:
        project = (ROOT / "Voxt.xcodeproj/project.pbxproj").read_text()
        self.assertIn("relativePath = Packages/OmniModelArtifacts;", project)
        for source in ("Voxt/Meeting/Capture/SileroVADModelProvisioner.swift",
                       "Voxt/Meeting/SpeakerAnalysis/MeetingDiarizationModelManager.swift"):
            text = (ROOT / source).read_text()
            self.assertIn("import OmniModelArtifacts", text)
            self.assertIn("OmniModelArtifactValidation.validateSafetensorsDirectory(at:", text)
            self.assertNotRegex(text, r"import MLX(?:AudioVAD)?\b|\.fromModelDirectory\(|SileroVADModelSupport\.loadModel\(")
        loader = (ROOT / "Voxt/Meeting/Capture/SileroVADModelLoader.swift").read_text()
        self.assertIn("OmniModelDirectoryLease.acquire(for:", loader)
        self.assertIn("OmniModelArtifactValidation.validateSafetensorsDirectory(at:", loader)
        self.assertNotRegex(loader, r"import MLX(?:AudioVAD)?\b|\.fromModelDirectory\(|SileroVADModelLease")
        workflow = (ROOT.parent / ".github/workflows/voxt-mac-ci.yaml").read_text()
        self.assertIn("swift test --package-path Voxt/Packages/OmniModelArtifacts", workflow)
        self.assertNotRegex(workflow, r"OmniXPC|OmniInference|OmniCore")
        self.assertIn("vars.VOXT_ENABLE_MAC_CI == 'true'", workflow)
        self.assertIn("vars.VOXT_ENABLE_CI_PACKAGE == 'true'", workflow)

    def test_packaging_shell_and_embedded_python_parse(self) -> None:
        script = ROOT / "tools/package_local_app.sh"
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for source in re.findall(r"<<'PY'\n(.*?)^PY$", script.read_text(), re.M | re.S):
            ast.parse(source, filename=script.name)

    def test_signed_packaging_cannot_ignore_verification_failure(self) -> None:
        script = (ROOT / "tools/package_local_app.sh").read_text()
        self.assertIn('codesign --verify --deep --strict --verbose=2 "$APP_PATH" || die', script)
        self.assertNotIn("artifact was still packaged", script)
        self.assertIn("Release packaging must not contain", script)
        self.assertIn("--preserve-metadata=entitlements", script)
        leaf_signing = script.index("while IFS= read -r framework_code")
        bundle_signing = script.index('find "$APP_PATH/Contents/Frameworks" -depth')
        host_signing = script.index('--entitlements "$SIGNING_ENTITLEMENTS"')
        self.assertLess(leaf_signing, bundle_signing)
        self.assertLess(bundle_signing, host_signing)
        self.assertIn('"$NATIVE_RUNTIME_DIR/bin"', script)
        self.assertNotIn("build_omni_xpc_smoke.sh", script)


class SigningProfilePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle_identifier = "com.example.Voxt"
        self.app_identity = f"PREFIX1234.{self.bundle_identifier}"
        self.source = plistlib.loads((ROOT / "Voxt/Voxt.entitlements").read_bytes())
        self.profile: dict[str, object] = {
            "CreationDate": datetime.datetime(2020, 1, 1),
            "ExpirationDate": datetime.datetime(2099, 1, 1),
            "Platform": ["OSX"],
            "ProvisionsAllDevices": True,
            "TeamIdentifier": ["TEAM123456"],
            "ApplicationIdentifierPrefix": ["PREFIX1234"],
            "DeveloperCertificates": [b"synthetic DER placeholder, not a certificate"],
            "Entitlements": {
                "com.apple.application-identifier": self.app_identity,
                "com.apple.developer.team-identifier": "TEAM123456",
                "keychain-access-groups": [self.app_identity],
                "get-task-allow": False,
            },
        }

    def test_expands_preserves_app_identity_and_default_group(self) -> None:
        expanded = prepare_entitlements(self.profile, self.source, self.bundle_identifier)
        self.assertEqual(expanded["com.apple.application-identifier"], self.app_identity)
        self.assertEqual(expanded["com.apple.developer.team-identifier"], "TEAM123456")
        self.assertEqual(expanded["keychain-access-groups"], [self.app_identity])
        self.assertEqual(expanded["com.apple.security.temporary-exception.mach-lookup.global-name"], ["com.example.Voxt-spks", "com.example.Voxt-spki"])
        self.assertTrue(expanded["com.apple.security.app-sandbox"])
        self.assertNotIn(b"$(", plistlib.dumps(expanded))
        self.assertIn(b"$(", plistlib.dumps(self.source))

    def test_profile_prefix_allowlist_authorizes_only_private_emitted_group(self) -> None:
        profile = copy.deepcopy(self.profile)
        profile["Entitlements"]["keychain-access-groups"] = ["PREFIX1234.*"]
        expanded = prepare_entitlements(profile, self.source, self.bundle_identifier)
        self.assertEqual(expanded["keychain-access-groups"], [self.app_identity])
        self.assertNotIn(b"*", plistlib.dumps(expanded))

    def test_invalid_profile_shapes_and_distribution_are_rejected(self) -> None:
        changes: tuple[tuple[str, object], ...] = (
            ("ExpirationDate", datetime.datetime(2000, 1, 1)),
            ("CreationDate", datetime.datetime(2099, 1, 1)),
            ("ExpirationDate", "2099-01-01"),
            ("CreationDate", None),
            ("Platform", ["iOS"]),
            ("ProvisionsAllDevices", False),
            ("ProvisionedDevices", ["synthetic-device"]),
            ("TeamIdentifier", []),
            ("TeamIdentifier", ["TEAM123456", "OTHER12345"]),
            ("ApplicationIdentifierPrefix", "PREFIX1234"),
            ("ApplicationIdentifierPrefix", ["*"]),
            ("DeveloperCertificates", []),
            ("DeveloperCertificates", ["not DER data"]),
            ("Entitlements", []),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                profile = copy.deepcopy(self.profile)
                profile[name] = value
                with self.assertRaises(ValueError):
                    prepare_entitlements(profile, self.source, self.bundle_identifier)

    def test_unauthorized_identity_and_groups_are_rejected(self) -> None:
        changes: tuple[tuple[str, object], ...] = (
            ("com.apple.application-identifier", "PREFIX1234.com.example.Other"),
            ("com.apple.application-identifier", "PREFIX1234.*"),
            ("com.apple.application-identifier", None),
            ("com.apple.developer.team-identifier", "OTHER12345"),
            ("get-task-allow", True),
            ("keychain-access-groups", ["OTHER12345.*"]),
            ("keychain-access-groups", ["*"]),
            ("keychain-access-groups", ["PREFIX1234.com.example.*"]),
            ("keychain-access-groups", ["PREFIX1234.com.example.Other"]),
            ("keychain-access-groups", []),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                profile = copy.deepcopy(self.profile)
                profile["Entitlements"][name] = value
                with self.assertRaises(ValueError):
                    prepare_entitlements(profile, self.source, self.bundle_identifier)

    def test_source_identity_cannot_be_changed_or_left_unresolved(self) -> None:
        changes: tuple[tuple[str, object], ...] = (
            ("com.apple.application-identifier", "OTHER12345.com.example.Voxt"),
            ("com.apple.developer.team-identifier", "OTHER12345"),
            ("keychain-access-groups", ["PREFIX1234.com.example.Other"]),
            ("keychain-access-groups", [self.app_identity, "PREFIX1234.shared"]),
            ("keychain-access-groups", []),
            ("com.apple.security.temporary-exception.mach-lookup.global-name", ["$(UNKNOWN)"]),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                source = copy.deepcopy(self.source)
                source[name] = value
                with self.assertRaises(ValueError):
                    prepare_entitlements(self.profile, source, self.bundle_identifier)


class NotarizedWorkflowArgumentsTests(unittest.TestCase):
    def test_real_workflow_block_optional_arguments_under_nounset(self) -> None:
        workflow = (ROOT.parent / ".github/workflows/voxt-mac-notarized-release.yaml").read_text()
        self.assertIn("vars.VOXT_ENABLE_NOTARIZED_RELEASE == 'true'", workflow)
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("VOXT_DEVELOPER_ID_PROFILE_BASE64", workflow)
        block = re.search(r"      - name: Build and sign app packages\n.*?        run: \|\n(.*?)\n      - name:", workflow, re.S)
        self.assertIsNotNone(block)
        shell_source = textwrap.dedent(block[1])
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            executable = temporary / "Voxt/tools/package_local_app.sh"
            executable.parent.mkdir(parents=True)
            executable.write_text(
                "#!/usr/bin/env python3\nimport json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['VOXT_ARGUMENT_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
                "sys.exit(int(os.environ.get('VOXT_FAKE_FAILURE', '0')))\n"
            )
            executable.chmod(0o755)
            for version, build_number in (("", ""), ("1.2.3", ""), ("", "123"), ("1.2.3", "123")):
                with self.subTest(version=version, build_number=build_number):
                    environment = {
                        **os.environ, "VERSION_INPUT": version, "BUILD_NUMBER_INPUT": build_number,
                        "RUNNER_TEMP": str(temporary / "runner space"),
                        "DEVELOPER_ID_APP_IDENTITY": "synthetic app identity",
                        "DEVELOPER_ID_INSTALLER_IDENTITY": "synthetic installer identity",
                        "VOXT_ARGUMENT_LOG": str(temporary / "arguments.json"),
                    }
                    result = subprocess.run(
                        [os.environ.get("VOXT_TEST_BASH", "/bin/bash"), "-c", shell_source],
                        cwd=temporary, env=environment, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    optional = (["--version", version] if version else []) + (["--build-number", build_number] if build_number else [])
                    expected = optional + [
                        "--optimized", "--native-runtime", str(temporary / "runner space/voxt-native-runtime"),
                        "--output-dir", str(temporary / "runner space/voxt-package"),
                        "--app-sign-identity", "synthetic app identity",
                        "--pkg-sign-identity", "synthetic installer identity",
                        "--provisioning-profile", str(temporary / "runner space/voxt-developer-id.provisionprofile"),
                    ]
                    self.assertEqual(json.loads((temporary / "arguments.json").read_text()), expected)
                    failed = subprocess.run(
                        [os.environ.get("VOXT_TEST_BASH", "/bin/bash"), "-c", shell_source],
                        cwd=temporary, env={**environment, "VOXT_FAKE_FAILURE": "37"},
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(failed.returncode, 37, failed.stderr)


if __name__ == "__main__":
    unittest.main()
