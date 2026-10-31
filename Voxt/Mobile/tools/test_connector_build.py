"""Build orchestration contracts; fixture runner never launches an app."""

import importlib.util
import plistlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConnectorBuildTests(unittest.TestCase):
    def test_sources_and_scheme_ship_in_repository(self):
        app = ROOT / "connector/VoxtConnector"
        self.assertTrue((app / "Sources/VoxtConnectorApp.swift").is_file())
        self.assertTrue((app / "project.yml").is_file())
        self.assertTrue(
            (
                app
                / "VoxtConnector.xcodeproj/xcshareddata/xcschemes/VoxtConnector.xcscheme"
            ).is_file()
        )
        self.assertNotIn("app-draft", (app / "project.yml").read_text())

    def test_bundle_ids_are_the_neutral_defaults(self):
        # One setting names the Connector: the neutral default in Signing.shared.xcconfig, which a
        # signer overrides in the uncommitted Signing.local.xcconfig.
        app = ROOT / "connector/VoxtConnector"
        shared = (app / "Config/Signing.shared.xcconfig").read_text()
        self.assertIn("VOXT_CONNECTOR_BUNDLE_ID = com.voxt.VoxtConnector\n", shared)
        self.assertIn('#include? "Signing.local.xcconfig"', shared)
        self.assertIn(
            "connector/VoxtConnector/Config/Signing.local.xcconfig",
            (ROOT / ".gitignore").read_text().splitlines(),
        )
        self.assertIn(
            "VOXT_CONNECTOR_BUNDLE_ID =",
            (app / "Config/Signing.local.xcconfig.example").read_text(),
        )
        expected = {
            "$(VOXT_CONNECTOR_BUNDLE_ID)",
            "$(VOXT_CONNECTOR_BUNDLE_ID).uitests",
        }
        spec = re.findall(
            r"PRODUCT_BUNDLE_IDENTIFIER: (\S+)", (app / "project.yml").read_text()
        )
        project = re.findall(
            r'PRODUCT_BUNDLE_IDENTIFIER = "?([^;"]+)"?;',
            (app / "VoxtConnector.xcodeproj/project.pbxproj").read_text(),
        )
        self.assertEqual({s.strip('"') for s in spec}, expected)
        self.assertEqual(set(project), expected)
        # The app code names the neutral default once, as the fallback; a signer's own ID from
        # Signing.local.xcconfig appears in no Swift file.
        swift = [
            p
            for folder in (
                ROOT / "connector/VoxtConnector",
                ROOT / "connector/VoxtConnectionKit",
            )
            for p in folder.rglob("*.swift")
            if ".build" not in p.parts
        ]
        named = [
            str(p.relative_to(ROOT))
            for p in swift
            if "Tests" not in p.parts and "com.voxt.VoxtConnector" in p.read_text()
        ]
        self.assertEqual(
            named,
            [
                "connector/VoxtConnectionKit/Sources/VoxtConnectionKit/ConnectorIdentity.swift",
            ],
        )
        local = app / "Config/Signing.local.xcconfig"
        own = set(
            re.findall(
                r"^VOXT_CONNECTOR_BUNDLE_ID = (\S+)",
                local.read_text() if local.exists() else "",
                re.M,
            )
        )
        self.assertEqual(
            [
                str(p.relative_to(ROOT))
                for p in swift
                if any(i in p.read_text() for i in own)
            ],
            [],
        )

    def _config(self, folder, local=None):
        config = Path(folder) / "Config"
        config.mkdir()
        (config / "Signing.shared.xcconfig").write_text(
            '// The Connector\nVOXT_CONNECTOR_BUNDLE_ID = com.voxt.VoxtConnector\n#include? "Signing.local.xcconfig"\n'
        )
        if local is not None:
            (config / "Signing.local.xcconfig").write_text(local)
        return config

    def test_bundle_id_is_neutral_unless_a_private_setting_names_one(self):
        builder = self._builder()
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                builder.bundle_id(self._config(folder), env={}),
                "com.voxt.VoxtConnector",
            )
        with tempfile.TemporaryDirectory() as folder:
            config = self._config(
                folder,
                "// ours\nVOXT_CONNECTOR_BUNDLE_ID = com.example.official.connector\n",
            )
            self.assertEqual(
                builder.bundle_id(config, env={}), "com.example.official.connector"
            )
            # A release builds a clean public checkout, so it names the private file instead.
            release = Path(folder) / "release.xcconfig"
            release.write_text(
                "VOXT_CONNECTOR_BUNDLE_ID = com.example.release.connector\n"
            )
            self.assertEqual(
                builder.bundle_id(
                    config, env={"VOXT_CONNECTOR_SIGNING_LOCAL": str(release)}
                ),
                "com.example.release.connector",
            )

    def test_bundle_id_refuses_a_broken_private_setting(self):
        builder = self._builder()
        with tempfile.TemporaryDirectory() as folder:
            config = self._config(folder)
            empty = Path(folder) / "empty.xcconfig"
            empty.write_text("DEVELOPMENT_TEAM = ABCDE12345\n")
            quoted = Path(folder) / "quoted.xcconfig"
            quoted.write_text('VOXT_CONNECTOR_BUNDLE_ID = a" or identifier "b\n')
            for path in (Path(folder) / "missing.xcconfig", empty, quoted):
                with self.assertRaises(SystemExit):
                    builder.bundle_id(
                        config, env={"VOXT_CONNECTOR_SIGNING_LOCAL": str(path)}
                    )

    def test_build_passes_the_bundle_id_to_xcodebuild(self):
        command = self._builder().xcodebuild_command(
            Path("/derived"), Path("/evidence/x.xcresult"), "com.example.c"
        )
        self.assertIn("VOXT_CONNECTOR_BUNDLE_ID=com.example.c", command)
        self.assertIn("CODE_SIGNING_ALLOWED=NO", command)

    def test_builder_signs_runtime_then_app_with_timeout(self):
        spec = importlib.util.spec_from_file_location(
            "connector_build", ROOT / "tools/build-connector.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))

            class Result:
                stderr = "flags=0x10000(runtime)"

            return Result()

        app = ROOT / "build/fixture/Voxt Connector.app"
        module.sign(app, "fixture-identity", run=run)
        self.assertEqual(
            [call[0][-1] for call in calls[:2]],
            [str(app / "Contents/Helpers/voxt-tailcat-host"), str(app)],
        )
        self.assertTrue(all(call[1]["timeout"] == 90 for call in calls))
        self.assertEqual(calls[2][0][1:4], ["--verify", "--deep", "--strict"])

    def test_helper_and_app_are_signed_with_the_hardened_runtime(self):
        # Voxt trusts the Connector by signature only with the hardened runtime: without it an
        # injected library passed the signature check.
        spec = importlib.util.spec_from_file_location(
            "connector_build", ROOT / "tools/build-connector.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []

        def run(args, **kwargs):
            calls.append(args)

            class Result:
                stderr = (
                    "CodeDirectory v=20500 size=1 flags=0x10000(runtime) hashes=1+1"
                )

            return Result()

        app = ROOT / "build/fixture/Voxt Connector.app"
        module.sign(app, "fixture-identity", run=run)
        signing = [call for call in calls if "--sign" in call]
        self.assertEqual(len(signing), 2)
        self.assertTrue(
            all(
                call[call.index("--options") + 1] == "runtime"
                for call in signing
                if "--options" in call
            )
        )
        self.assertTrue(all("--options" in call for call in signing))

    def test_build_fails_when_a_signature_lacks_the_runtime_flag(self):
        spec = importlib.util.spec_from_file_location(
            "connector_build", ROOT / "tools/build-connector.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def run(args, **kwargs):
            class Result:
                stderr = "CodeDirectory v=20400 size=1 flags=0x0(none) hashes=1+1"

            return Result()

        with self.assertRaises(SystemExit):
            module.sign(
                Path("/nonexistent/Voxt Connector.app"), "fixture-identity", run=run
            )

    def test_a_missing_identity_names_the_ad_hoc_build(self):
        # Measured on macOS 27: codesign with an unknown identity prints "<name>: no identity
        # found" and exits 1; --sign - with --options runtime gives flags=0x10002(adhoc,runtime).
        module = self._builder()

        def run(args, **kwargs):
            raise subprocess.CalledProcessError(1, args)

        with self.assertRaises(SystemExit) as raised:
            module.sign(
                Path("/nonexistent/Voxt Connector.app"), "Apple Development", run=run
            )
        message = str(raised.exception)
        self.assertIn("Apple Development", message)
        self.assertIn("VOXT_CONNECTOR_SIGNING_IDENTITY=-", message)

    def test_a_missing_tool_is_named_with_its_install_command(self):
        module = self._builder()
        with self.assertRaises(SystemExit) as raised:
            module.require("xcodegen", "brew install xcodegen", which=lambda name: None)
        self.assertIn("xcodegen", str(raised.exception))
        self.assertIn("brew install xcodegen", str(raised.exception))
        module.require(
            "xcodegen", "brew install xcodegen", which=lambda name: "/bin/" + name
        )

    def _builder(self):
        spec = importlib.util.spec_from_file_location(
            "connector_build", ROOT / "tools/build-connector.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _bare_app(self, folder):
        app = Path(folder) / "Voxt Connector.app"
        (app / "Contents/MacOS").mkdir(parents=True)
        plistlib_dump(
            app / "Contents/Info.plist",
            {"CFBundleIdentifier": "com.voxt.VoxtConnector"},
        )
        return app

    def test_icon_check_fails_on_a_bundle_without_an_icon(self):
        # Without Contents/Resources the Dock shows a blank icon.
        builder = self._builder()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit):
                builder.check_icon(self._bare_app(folder))

    def test_add_icon_copies_the_icns_and_names_it_in_info_plist(self):
        builder = self._builder()
        with tempfile.TemporaryDirectory() as folder:
            app = self._bare_app(folder)
            builder.add_icon(app)
            builder.check_icon(app)
            info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
            self.assertEqual(info["CFBundleIconFile"], "AppIcon")
            self.assertEqual(info["CFBundleIdentifier"], "com.voxt.VoxtConnector")
            self.assertEqual(
                (app / "Contents/Resources/AppIcon.icns").read_bytes()[:4], b"icns"
            )

    def test_icon_check_fails_when_the_plist_names_a_missing_file(self):
        builder = self._builder()
        with tempfile.TemporaryDirectory() as folder:
            app = self._bare_app(folder)
            plistlib_dump(app / "Contents/Info.plist", {"CFBundleIconFile": "AppIcon"})
            with self.assertRaises(SystemExit):
                builder.check_icon(app)


def plistlib_dump(path, value):
    path.write_bytes(plistlib.dumps(value))


if __name__ == "__main__":
    unittest.main()
