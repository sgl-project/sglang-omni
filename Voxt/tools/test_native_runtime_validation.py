"""Test filesystem preflight with placeholders, not Mach-O compatibility or signing."""

from __future__ import annotations

import datetime
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

VALIDATION_SCRIPT: Path = (
    Path(__file__).with_name("validate_native_runtime.sh").resolve()
)
SERVER_FILENAMES: tuple[str, ...] = (
    "qwen3_asr_server",
    "whisper_server",
    "cohere_transcribe_server",
    "moss_transcribe_diarize_server",
    "nemotron_asr_server",
    "parakeet_server",
    "sensevoice_server",
)
RESOURCE_FILENAMES: tuple[str, ...] = (
    "libmlx.dylib",
    "libjaccl.dylib",
    "mlx.metallib",
)


class NativeRuntimeValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory()
        )
        self.addCleanup(self.temporary_directory.cleanup)

    def create_runtime_fixture(self) -> Path:
        runtime_directory = Path(
            tempfile.mkdtemp(
                prefix="native runtime ", dir=self.temporary_directory.name
            )
        )
        bin_directory = runtime_directory / "bin"
        bin_directory.mkdir()
        for filename in SERVER_FILENAMES:
            server_path = bin_directory / filename
            server_path.write_text(
                "test-only executable placeholder, not native code\n"
            )
            server_path.chmod(0o755)
        for filename in RESOURCE_FILENAMES:
            resource_path = bin_directory / filename
            resource_path.write_text("test-only resource placeholder\n")
            resource_path.chmod(0o644)
        return runtime_directory

    def run_validator(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(VALIDATION_SCRIPT), *arguments],
            cwd=self.temporary_directory.name,
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_invalid_runtime(
        self, runtime_directory: Path, invalid_path: Path
    ) -> None:
        result = self.run_validator(str(runtime_directory))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(str(invalid_path), result.stderr)

    def test_complete_runtime_without_core_abi_library_passes(self) -> None:
        runtime_directory = self.create_runtime_fixture()
        result = self.run_validator(str(runtime_directory))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_relative_runtime_directory_passes(self) -> None:
        runtime_directory = self.create_runtime_fixture()
        relative_directory = runtime_directory.relative_to(
            self.temporary_directory.name
        )
        result = self.run_validator(str(relative_directory))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_missing_server_is_rejected(self) -> None:
        for filename in SERVER_FILENAMES:
            with self.subTest(filename=filename):
                runtime_directory = self.create_runtime_fixture()
                server_path = runtime_directory / "bin" / filename
                server_path.unlink()
                self.assert_invalid_runtime(runtime_directory, server_path)

    def test_every_non_executable_server_is_rejected(self) -> None:
        for filename in SERVER_FILENAMES:
            with self.subTest(filename=filename):
                runtime_directory = self.create_runtime_fixture()
                server_path = runtime_directory / "bin" / filename
                server_path.chmod(0o644)
                self.assert_invalid_runtime(runtime_directory, server_path)

    def test_server_directories_are_rejected(self) -> None:
        for filename in SERVER_FILENAMES:
            with self.subTest(filename=filename):
                runtime_directory = self.create_runtime_fixture()
                server_path = runtime_directory / "bin" / filename
                server_path.unlink()
                server_path.mkdir()
                self.assert_invalid_runtime(runtime_directory, server_path)

    def test_every_missing_resource_is_rejected(self) -> None:
        for filename in RESOURCE_FILENAMES:
            with self.subTest(filename=filename):
                runtime_directory = self.create_runtime_fixture()
                resource_path = runtime_directory / "bin" / filename
                resource_path.unlink()
                self.assert_invalid_runtime(runtime_directory, resource_path)

    def test_resource_directories_are_rejected(self) -> None:
        for filename in RESOURCE_FILENAMES:
            with self.subTest(filename=filename):
                runtime_directory = self.create_runtime_fixture()
                resource_path = runtime_directory / "bin" / filename
                resource_path.unlink()
                resource_path.mkdir()
                self.assert_invalid_runtime(runtime_directory, resource_path)

    def test_every_unreadable_resource_is_rejected(self) -> None:
        for filename in RESOURCE_FILENAMES:
            with self.subTest(filename=filename):
                runtime_directory = self.create_runtime_fixture()
                resource_path = runtime_directory / "bin" / filename
                resource_path.chmod(0o000)
                if os.access(resource_path, os.R_OK):
                    self.skipTest("Current user can read mode-000 files")
                else:
                    self.assert_invalid_runtime(runtime_directory, resource_path)

    def test_unsearchable_bin_directory_is_rejected(self) -> None:
        runtime_directory = self.create_runtime_fixture()
        bin_directory = runtime_directory / "bin"
        bin_directory.chmod(0o600)
        try:
            if os.access(bin_directory, os.X_OK):
                self.skipTest(
                    "Current user can search directories without execute access"
                )
            else:
                self.assert_invalid_runtime(runtime_directory, bin_directory)
        finally:
            bin_directory.chmod(0o755)

    def test_missing_bin_directory_is_rejected(self) -> None:
        runtime_directory = Path(self.temporary_directory.name)
        self.assert_invalid_runtime(runtime_directory, runtime_directory / "bin")

    def test_bin_regular_file_is_rejected(self) -> None:
        runtime_directory = Path(self.temporary_directory.name)
        bin_path = runtime_directory / "bin"
        bin_path.write_text("not a directory\n")
        self.assert_invalid_runtime(runtime_directory, bin_path)

    def test_missing_runtime_directory_is_rejected(self) -> None:
        runtime_directory = Path(self.temporary_directory.name) / "missing runtime"
        self.assert_invalid_runtime(runtime_directory, runtime_directory / "bin")

    def test_invalid_arguments_are_rejected(self) -> None:
        runtime_directory = self.create_runtime_fixture()
        for arguments in ((), ("",), (str(runtime_directory), "extra")):
            with self.subTest(arguments=arguments):
                result = self.run_validator(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Usage:", result.stderr)


FAKE_PACKAGING_TOOL = r'''#!/usr/bin/env python3
import json, os, plistlib, shutil, sys
from pathlib import Path
command = Path(sys.argv[0]).name
arguments = sys.argv[1:]
with open(os.environ['VOXT_COMMAND_LOG'], 'a') as log:
    log.write(json.dumps([command, *arguments]) + '\n')
app = Path(os.environ['VOXT_FAKE_APP'])
mode = os.environ.get('VOXT_FAKE_FAILURE', '')
if command == 'xcodebuild':
    (app / 'Contents/Frameworks/Synthetic.framework').mkdir(parents=True)
    (app / 'Contents/Frameworks/Synthetic.framework/code').write_text('fake Mach-O')
    (app / 'Contents/Info.plist').write_bytes(plistlib.dumps({
        'CFBundleIdentifier': 'com.example.Voxt',
        'CFBundleShortVersionString': '1.2.3', 'CFBundleVersion': '123',
    }))
elif command == 'PlistBuddy':
    print(plistlib.loads(Path(arguments[-1]).read_bytes())[arguments[1].split(':')[1]])
elif command == 'security':
    if arguments[:3] != ['cms', '-D', '-i']:
        sys.exit('fake tools forbid keychain access or identity discovery')
    if mode == 'cms':
        sys.exit(29)
    sys.stdout.buffer.write(Path(arguments[3]).read_bytes())
elif command == 'codesign':
    if '--verify' in arguments:
        sys.exit(23 if mode == 'verify' else 0)
    elif '--display' in arguments:
        if '--extract-certificates' in arguments:
            if mode == 'extract':
                sys.exit(41)
            prefix = arguments[arguments.index('--extract-certificates') + 1]
            Path(prefix + '0').write_bytes(b'wrong certificate' if mode == 'certificate' else b'synthetic certificate')
        elif '--entitlements' in arguments:
            entitlements = plistlib.loads((app / 'signed-entitlements.plist').read_bytes())
            if mode == 'entitlements':
                entitlements.pop('keychain-access-groups')
            sys.stdout.buffer.write(plistlib.dumps(entitlements))
        elif '--verbose=4' in arguments:
            if mode == 'inspect':
                sys.exit(37)
            print('TeamIdentifier=' + ('OTHER12345' if mode == 'team' else 'TEAM123456'), file=sys.stderr)
            print('Authority=' + ('Apple Development: synthetic' if mode == 'authority' else 'Developer ID Application: synthetic'), file=sys.stderr)
        else:
            sys.exit(1)
    elif '--entitlements' in arguments:
        shutil.copyfile(arguments[arguments.index('--entitlements') + 1], app / 'signed-entitlements.plist')
        if mode == 'sign':
            sys.exit(31)
    else:
        pass
elif command == 'file':
    print('Mach-O synthetic test placeholder')
elif command == 'ditto':
    if '-c' in arguments:
        Path(arguments[-1]).touch()
    else:
        shutil.copytree(arguments[-2], arguments[-1], dirs_exist_ok=True)
elif command in ('hdiutil', 'productbuild'):
    Path(arguments[-1]).touch()
else:
    sys.exit('unexpected fake command: ' + command)
'''


class SignedPackagingShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="voxt fake packaging ")
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary = Path(self.temporary_directory.name)
        self.root = self.temporary / "Voxt"
        tools = self.root / "tools"
        tools.mkdir(parents=True)
        original_tools = VALIDATION_SCRIPT.parent
        source = (original_tools / "package_local_app.sh").read_text()
        self.fake_bin = self.temporary / "fake-bin"
        self.fake_bin.mkdir()
        executable = self.fake_bin / "fake-tool"
        executable.write_text(FAKE_PACKAGING_TOOL)
        executable.chmod(0o755)
        for name in ("xcodebuild", "security", "codesign", "PlistBuddy", "file", "ditto", "hdiutil", "productbuild"):
            (self.fake_bin / name).symlink_to(executable)
        self.script = tools / "package_local_app.sh"
        self.script.write_text(source.replace("/usr/libexec/PlistBuddy", shlex.quote(str(self.fake_bin / "PlistBuddy"))))
        shutil.copyfile(original_tools / "signing_profile_policy.py", tools / "signing_profile_policy.py")
        (self.root / "Voxt.xcodeproj").mkdir()
        (self.root / "Voxt").mkdir()
        shutil.copyfile(original_tools.parent / "Voxt/Voxt.entitlements", self.root / "Voxt/Voxt.entitlements")
        self.profile = self.temporary / "synthetic profile.provisionprofile"
        self.profile.write_bytes(plistlib.dumps({
            "CreationDate": datetime.datetime(2020, 1, 1),
            "ExpirationDate": datetime.datetime(2099, 1, 1),
            "Platform": ["OSX"], "ProvisionsAllDevices": True,
            "TeamIdentifier": ["TEAM123456"], "ApplicationIdentifierPrefix": ["PREFIX1234"],
            "DeveloperCertificates": [b"synthetic certificate"],
            "Entitlements": {
                "com.apple.application-identifier": "PREFIX1234.com.example.Voxt",
                "com.apple.developer.team-identifier": "TEAM123456",
                "keychain-access-groups": ["PREFIX1234.*"],
                "get-task-allow": False,
            },
        }))
        self.app = self.temporary / "derived/Build/Products/Release/Voxt.app"
        self.command_log = self.temporary / "commands.jsonl"
        self.environment = {
            **os.environ, "PATH": str(self.fake_bin) + os.pathsep + os.environ.get("PATH", ""),
            "VOXT_COMMAND_LOG": str(self.command_log), "VOXT_FAKE_APP": str(self.app),
            "VOXT_NATIVE_RUNTIME_DIR": "", "VOXT_PROVISIONING_PROFILE": "",
            "DEVELOPER_ID_APP_IDENTITY": "synthetic app identity",
            "DEVELOPER_ID_INSTALLER_IDENTITY": "synthetic installer identity",
            "VOXT_FAKE_FAILURE": "",
        }

    def run_packager(self, *arguments: str, failure: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [os.environ.get("VOXT_TEST_BASH", "/bin/bash"), str(self.script),
             "--version", "1.2.3", "--build-number", "123", "--no-clean",
             "--derived-data", str(self.temporary / "derived"),
             "--output-dir", str(self.temporary / "output"), *arguments],
            env={**self.environment, "VOXT_FAKE_FAILURE": failure},
            capture_output=True, text=True, check=False,
        )

    def commands(self) -> list[list[str]]:
        if self.command_log.exists():
            return [json.loads(line) for line in self.command_log.read_text().splitlines()]
        else:
            return []

    def test_signed_package_requires_explicit_profile_before_any_tools(self) -> None:
        result = self.run_packager()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("requires an explicit matching Developer ID provisioning profile", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_missing_profile_fails_before_any_tools(self) -> None:
        result = self.run_packager("--provisioning-profile", str(self.temporary / "missing"))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("provisioning profile not found", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_full_fake_signed_packaging_embeds_supplied_profile_and_preserves_identity(self) -> None:
        result = self.run_packager("--provisioning-profile", str(self.profile))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.app / "Contents/embedded.provisionprofile").read_bytes(), self.profile.read_bytes())
        entitlements = plistlib.loads((self.app / "signed-entitlements.plist").read_bytes())
        self.assertEqual(entitlements["keychain-access-groups"], ["PREFIX1234.com.example.Voxt"])
        self.assertEqual(entitlements["com.apple.application-identifier"], "PREFIX1234.com.example.Voxt")
        self.assertEqual(entitlements["com.apple.developer.team-identifier"], "TEAM123456")
        commands = self.commands()
        self.assertTrue(any(command[:2] == ["codesign", "--verify"] for command in commands))
        extraction = next(index for index, command in enumerate(commands) if "--extract-certificates" in command)
        archive = next(index for index, command in enumerate(commands) if command[:2] == ["ditto", "-c"])
        self.assertLess(extraction, archive)
        self.assertTrue(any(command[0] == "productbuild" for command in commands))
        self.assertIn("not notarization or Keychain runtime proof", result.stdout)

    def test_environment_profile_is_explicit_and_supported(self) -> None:
        self.environment["VOXT_PROVISIONING_PROFILE"] = str(self.profile)
        result = self.run_packager()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.app / "Contents/embedded.provisionprofile").read_bytes(), self.profile.read_bytes())

    def test_unsigned_inspection_never_decodes_or_embeds_profile(self) -> None:
        result = self.run_packager("--no-sign")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.app / "Contents/embedded.provisionprofile").exists())
        self.assertFalse(any(command[0] in ("security", "codesign") for command in self.commands()))
        self.assertIn("unsigned inspection artifacts", result.stderr)

    def test_unauthorized_profile_fails_before_signing_or_artifacts(self) -> None:
        profile = plistlib.loads(self.profile.read_bytes())
        profile["Entitlements"]["com.apple.application-identifier"] = "PREFIX1234.*"
        self.profile.write_bytes(plistlib.dumps(profile))
        result = self.run_packager("--provisioning-profile", str(self.profile))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wildcard profiles are not permitted", result.stderr)
        self.assertFalse(any(command[0] in ("codesign", "ditto", "hdiutil", "productbuild") for command in self.commands()))
        self.assertFalse((self.app / "Contents/embedded.provisionprofile").exists())

    def test_signing_policy_failures_never_produce_archives(self) -> None:
        failures = {
            "cms": "security failed with status 29", "verify": "codesign verification failed",
            "sign": "", "team": "profile's team", "authority": "Developer ID Application",
            "inspect": "codesign failed with status 37", "extract": "codesign failed with status 41",
            "certificate": "certificate is not authorized", "entitlements": "signed entitlements differ",
        }
        for failure, expected_error in failures.items():
            with self.subTest(failure=failure):
                shutil.rmtree(self.app, ignore_errors=True)
                self.command_log.unlink(missing_ok=True)
                result = self.run_packager("--provisioning-profile", str(self.profile), failure=failure)
                self.assertNotEqual(result.returncode, 0)
                if failure == "sign":
                    self.assertEqual(result.returncode, 31)
                elif failure in ("cms", "inspect", "extract"):
                    self.assertEqual(result.returncode, {"cms": 29, "inspect": 37, "extract": 41}[failure])
                    self.assertIn(expected_error, result.stderr)
                else:
                    self.assertIn(expected_error, result.stderr)
                self.assertFalse(any(command[0] in ("ditto", "hdiutil", "productbuild") for command in self.commands()))
                self.assertFalse(list((self.temporary / "output").rglob("*.zip")))


if __name__ == "__main__":
    unittest.main()
else:
    pass
